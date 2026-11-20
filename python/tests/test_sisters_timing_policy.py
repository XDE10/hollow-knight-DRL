from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import torch


def test_measured_returns_use_game_seconds_gaps_and_terminal(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    t = np.array([1, 1.05, 1.14, 1.5, 1.55, 1.61])
    rewards = np.array([0, 4, 100, 2, 6, 10])
    discounts = np.array([0.5, 0.8, 0.7, 0.25, 0, 1])
    valid = np.array([True, True, False, True, True, False])
    value, continuation, last = mod.measured_returns(t, rewards, discounts, valid, 0.20)
    np.testing.assert_allclose(value, [2, 4, 0, 3.5, 6, 0])
    np.testing.assert_allclose(continuation, [0.4, 0.8, 1, 0, 0, 1])
    np.testing.assert_array_equal(last, [1, 1, 2, 4, 4, 5])
    short, _, _ = mod.measured_returns(t, rewards, discounts, valid, 0.04)
    np.testing.assert_array_equal(short, [0, 4, 0, 2, 6, 0])


def test_masked_inference_and_native_export(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    cat = np.array([[-1, 0, 8], [0, 1, 0], [1, 1, 128], [1, -1, 16]])
    torch.manual_seed(15)
    actor = mod.TimingPolicy(torch.from_numpy(cat))
    state = torch.randn(11, mod.WIDTH)
    for k in (24, 60, 96, 132, 146):
        state[:, k] = (torch.arange(11) % 3 != 0).float()
    state[0, 25:60] = 0
    with torch.no_grad():
        expected = actor(state)[0]
        state[0, 25:60] = 100
        torch.testing.assert_close(expected, actor(state)[0])
    path = tmp_path / "actor.bin"
    mod.save_native(actor, cat, path)
    with path.open("rb") as f:
        assert struct.unpack("<5i", f.read(20)) == (0x324C5149, 192, 256, 192, len(cat))
        np.testing.assert_array_equal(np.fromfile(f, "<i4", len(cat) * 3).reshape(-1, 3), cat)

        def weights(n, j):
            return np.fromfile(f, "<f4", n * j).reshape(j, n).T, np.fromfile(f, "<f4", j)

        boss = weights(36, 48)
        disc = weights(14, 24)
        blocks = [weights(264, 256), weights(256, 192), weights(192, len(cat))]
        assert not f.read(1)
    x = state.numpy().copy()
    pools = []
    for start, stop, n, width, (w, b) in ((24, 132, 3, 36, boss), (132, 160, 2, 14, disc)):
        slots = x[:, start:stop].reshape(len(x), n, width)
        slots *= slots[:, :, :1].copy()
        pools.append((np.maximum(slots @ w + b, 0) * slots[:, :, :1]).max(axis=1))
    x = np.concatenate((x, *pools), axis=1)
    for i, (w, b) in enumerate(blocks):
        x = x @ w + b
        if i < 2:
            x = np.maximum(x, 0)
    np.testing.assert_allclose(expected.numpy(), x, atol=1e-5)


def test_previous_action_for_corrective_supervision(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    state = torch.zeros(2, mod.WIDTH)
    c = mod.CONTROL
    state[0, c : c + 2] = torch.tensor([-1, 1])
    state[0, [c + 4 + 2, c + 4 + 8]] = 1
    state[1, c + 13 :] = 1
    assert mod.previous_commands(state).tolist() == [[-1, 1, 260], [0, 0, 0]]


def test_teacher_history_keeps_the_first_corrective_label(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    t = np.array([0, 0.04, 0.08, 0.12, 1.8])
    applied = np.zeros((5, 3))
    label = applied.copy()
    label[1, 2] = 8
    age = mod.teacher_history_age(t, applied, label)
    np.testing.assert_allclose(age, [np.inf, np.inf, 0.04, 0.08, 1.76])
    np.testing.assert_array_equal(age >= 1.6, [True, True, False, False, True])


def test_future_box_labels_respect_family_position_and_episode_end(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("sisters_future_occupancy")
    t = np.arange(21, dtype="f4") * 0.04
    states = np.zeros((21, 192), "f4")
    states[:, 24] = 1
    states[3, 24 + 9] = 1
    states[3, 24 + 1] = 0.25
    field = mod.future_occupancy(t, states)
    assert field.shape == (21, 3, 2, 3, 3)
    assert field[0, 0, 0, 0, 2] == 1
    assert field[0, 0, 0, 0, 0] < field[0, 0, 0, 0, 1]
    assert not field[0, :, 1].any()
    assert not field[0, 1:].any()
    assert np.isnan(field[-1]).all()


def test_phase_bundle_preserves_the_opener_and_maps_action_rows(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    old = torch.tensor([[-1, 0, 8], [0, 1, 0], [1, 0, 2]])
    new = torch.tensor([[0, 1, 0], [1, 1, 128], [-1, 0, 8]])
    teacher = mod.TimingPolicy(old)
    student = mod.TimingPolicy(new)
    learner = tmp_path / "previous.pt"
    torch.save({"actor": teacher.state_dict(), "catalogue": old}, learner)
    assert mod.initialize_actor(student, new.numpy(), learner) == 2
    torch.testing.assert_close(student.out.weight[0], teacher.out.weight[1])
    torch.testing.assert_close(student.out.bias[2], teacher.out.bias[0])
    torch.testing.assert_close(student.basis, mod.TimingPolicy(new).basis)
    paths = [tmp_path / "opener.bin", tmp_path / "trio.bin"]
    for net, cat, path in zip((teacher, student), (old, new), paths, strict=True):
        mod.save_native(net, cat.numpy(), path)
    a, b = (p.read_bytes() for p in paths)
    pair = mod.phase_checkpoint(a, b)
    assert struct.unpack_from("<3i", pair) == (0x334C5149, len(a), len(b))
    assert pair[12 : 12 + len(a)] == a and pair[12 + len(a) :] == b

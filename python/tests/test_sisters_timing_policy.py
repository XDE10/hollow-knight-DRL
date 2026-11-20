from __future__ import annotations

import hashlib
import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
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


def test_opening_target_uses_only_resources_at_first_trio(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    s = np.zeros((5, mod.WIDTH), dtype="f4")
    s[:, 4] = np.array([9, 8, 9, 9, 1]) / 9
    s[:, 5] = [0, 0, 1/3, 1/3, 0]
    s[3:, 23] = 1
    r, done = mod.opening_rewards(s, np.zeros(5, "f4"), np.zeros(5, bool))
    np.testing.assert_allclose(r, [-1.25, 1.75, 10, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(done, [False, False, True, False, False])
    s[:, 23] = 0
    _, death = mod.opening_rewards(s, np.zeros(5, "f4"), np.array([0, 0, 0, 0, 1], bool))
    assert death.nonzero()[0].tolist() == [4]


def test_no_hit_reward_counts_contacts_even_after_healing(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    events = np.zeros((4, 8), dtype="f4")
    state = np.ones((4, mod.WIDTH), dtype="f4")
    base = np.zeros(4, dtype="f4")
    events[-1, 6] = 1
    np.testing.assert_array_equal(mod.no_hit_rewards(base, events, state, 2), [0, 0, 0, 4])
    state[1, 4] = 8 / 9
    np.testing.assert_array_equal(mod.no_hit_rewards(base, events, state, 2), base)
    state[:, 4] = 1
    events[1, 2], events[2, 5], events[3, 3] = 1, 1, 2
    np.testing.assert_array_equal(mod.no_hit_rewards(base, events, state, 2), [0, -4, 0, -4])
    np.testing.assert_array_equal(mod.no_hit_rewards(base, events, state, 0), base)


def test_winning_teacher_filter_keeps_all_real_critic_examples(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    records = []
    for i, (won, dt) in enumerate(((True, .04), (False, .04), (True, .08), (True, .04))):
        t = np.arange(5, dtype="f4") * dt
        state = np.zeros((len(t), mod.WIDTH), dtype="f4")
        state[:, 23] = 1
        keys = np.zeros((len(t), 3), dtype="i2")
        events = np.zeros((len(t), 8), dtype="f4")
        events[1, 1] = int(not won)
        path = tmp_path / f"test-{i}.npz"
        np.savez(path, timing_state=state, control_time=t, controls=keys, teacher_controls=keys,
                 time=t, elapsed_ticks=np.full(len(t), 2), events=events)
        records.append({"file": path.name, "source": f"native-development/test-{i}",
                        "won": won, "split": "holdout" if i == 3 else "train",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (tmp_path / "manifest.json").write_text(json.dumps(records))
    data, _, stats = mod.build_data(
        tmp_path, "cpu", mirror=False, trio_only=True, teacher_win_min_hz=23
    )
    assert stats["train"] == {"fights": 3, "decisions": 15}
    assert (data["train"][7] >= 0).tolist() == [True] * 5 + [False] * 10
    assert data["train"][4][5:10].min() < 0
    assert data["train"][6][5:10].any()


def test_teacher_free_transitions_have_real_actions_and_no_imitation_target(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    rows = []
    for i in range(3):
        t = np.arange(5, dtype="f4") * .04
        state = np.zeros((len(t), mod.WIDTH), dtype="f4")
        keys = np.zeros((len(t), 3), dtype="i2")
        keys[:, 0] = 1
        events = np.zeros((len(t), 8), dtype="f4")
        events[1, 1] = 1
        path = tmp_path / f"{i}.npz"
        labeled = i == 0
        np.savez(path, timing_state=state, control_time=t, controls=keys,
                 time=t, elapsed_ticks=np.full(len(t), 2), events=events,
                 **({"teacher_controls": keys} if labeled else {}))
        rows.append(dict(file=path.name, source=f"native-development/test-{i}",
                         split="holdout" if i==2 else "train", won=False,
                         teacher_labels=labeled, behavior_policy_sha256="a"*64,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    data, actions, _ = mod.build_data(tmp_path, "cpu", mirror=True)
    np.testing.assert_array_equal(actions, [[-1, 0, 0], [1, 0, 0]])
    for split, count in (("train", 10), ("holdout", 5)):
        _, executed, target, _, reward, _, good, age = data[split][:8]
        missing = target == -2
        assert missing.sum() == count and (age[missing] < 0).all()
        assert (executed[missing] >= 0).all() and good[missing].any()
        assert reward[missing].min() < 0


def test_motion_history_masks_gaps_and_only_reads_past(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    clock = np.array([0, .08, .16, .24, .32, .40, .48, 1., 1.08, 1.16], dtype="f4")
    state = np.zeros((len(clock), mod.WIDTH), dtype="f4")
    state[:, 0] = np.arange(len(clock)) + 1
    x = mod.motion_history(clock, state)
    short = x[:, mod.WIDTH:mod.WIDTH + mod.HISTORY_BLOCK]
    long = x[:, mod.WIDTH + mod.HISTORY_BLOCK:]
    np.testing.assert_array_equal(short[:, 0], [0, 0, 1, 2, 3, 4, 5, 0, 0, 8])
    np.testing.assert_array_equal(long[:, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose([short[3, -1], long[6, -1]], [.16, .48], atol=1e-6)
    policy = mod.TimingPolicy(torch.tensor([[-1, 0, 2], [1, 0, 8]]), history=True)
    raw = torch.from_numpy(x)
    expected = policy(raw)[0]
    raw[:, mod.WIDTH + 25:mod.WIDTH + 60] = 100
    torch.testing.assert_close(expected, policy(raw)[0])


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


@pytest.mark.parametrize("history", [False, True])
def test_phase_bundle_preserves_the_opener_and_maps_action_rows(
    tmp_path: Path, monkeypatch, history: bool
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    old = torch.tensor([[-1, 0, 8], [0, 1, 0], [1, 0, 2]])
    new = torch.tensor([[0, 1, 0], [1, 1, 128], [-1, 0, 8]])
    teacher = mod.TimingPolicy(old)
    student = mod.TimingPolicy(new, history)
    learner = tmp_path / "previous.pt"
    torch.save({"actor": teacher.state_dict(), "catalogue": old}, learner)
    assert mod.initialize_actor(student, new.numpy(), learner) == 2
    torch.testing.assert_close(student.out.weight[0], teacher.out.weight[1])
    torch.testing.assert_close(student.out.bias[2], teacher.out.bias[0])
    torch.testing.assert_close(student.basis, mod.TimingPolicy(new).basis)
    raw = torch.randn(6, mod.WIDTH)
    extra = torch.randn(6, len(mod.HISTORY_LAGS) * mod.HISTORY_BLOCK)
    historical = torch.cat((raw, extra), 1) if history else raw
    torch.testing.assert_close(student(historical)[0][:, [0, 2]], teacher(raw)[0][:, [1, 0]])
    paths = [tmp_path / "opener.bin", tmp_path / "trio.bin"]
    for net, cat, path in zip((teacher, student), (old, new), paths, strict=True):
        mod.save_native(net, cat.numpy(), path)
    a, b = (p.read_bytes() for p in paths)
    assert struct.unpack_from("<i", b)[0] == (0x344C5149 if history else 0x324C5149)
    pair = mod.phase_checkpoint(a, b)
    assert struct.unpack_from("<3i", pair) == (0x334C5149, len(a), len(b))
    assert pair[12 : 12 + len(a)] == a and pair[12 + len(a) :] == b
    assert mod.late_component(pair) == b and mod.late_component(b) == b
    assert mod.phase_components(pair) == (a, b)
    assert mod.phase_components(a) == (a, a)
    with pytest.raises(ValueError):
        mod.phase_components(pair[:-1])

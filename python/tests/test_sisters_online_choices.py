from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


def test_randomized_record_matches_frozen_catalogue_and_executed_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("sisters_online_choices")
    actions = np.array([[-1, 0, 8], [1, 1, 2]], dtype="<i4")
    policy = struct.pack("<5i", 0x324C5149, 192, 256, 192, 2) + actions.tobytes()
    combined = struct.pack("<3i", 0x334C5149, len(policy), len(policy)) + 2 * policy
    np.testing.assert_array_equal(mod.late_catalogue(combined), actions)
    t = np.arange(40, dtype="f4") * 0.04
    s = np.zeros((len(t), 192), dtype="f4")
    s[:, 23] = 1
    applied = np.tile(actions[0], (len(t), 1))
    applied[24] = actions[1]
    raw = np.array([(t[4], [0, 1], 0, 0.5, 0.3), (t[24], [0, 1], 1, 0.5, 2.0)], dtype=mod.ROW)
    path = tmp_path / "1.choices"
    path.write_bytes(struct.pack("<IQ", 0x31434B48, 1) + raw.tobytes())
    trace = mod.read_choices(path, 1, t, s, applied, actions)
    np.testing.assert_array_equal(trace["choice_rows"], [4, 24])
    applied[24] = actions[0]
    with pytest.raises(ValueError, match="actually issued"):
        mod.read_choices(path, 1, t, s, applied, actions)


def test_measured_trial_rewards_exclude_pre_draw_and_unobserved_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("sisters_online_choices")
    t = np.arange(61, dtype="f4") * 0.04
    starts = np.arange(24, dtype="f4") * 0.10
    events = np.zeros((len(starts), 8), dtype="f4")
    events[0, 1] = 90  # Straddles the first draw.
    events[2, 1] = 1
    events[13, 7] = 1
    weights = np.array([0, -2, 0, 0, 0, 0, 0, -6], dtype="f4")
    indices, returns, _, continuation = mod.choice_outcomes(
        t, starts, np.full(len(starts), 5), events, np.array([1, 28, 56]), weights
    )
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(
        returns, [-2 * 0.995 ** (0.26 / 0.04), -6 * 0.995 ** (0.28 / 0.04)], atol=1e-5
    )
    assert 0 < continuation[0] < 1 and continuation[1] == 0


def test_random_choice_policy_gradient_uses_executed_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_online_choices")
    old = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    new = torch.tensor([[0.25, 0.75], [0.25, 0.75]])
    arm = torch.tensor([0, 1])
    payoff = torch.tensor([2.0, 6.0])
    p = torch.full((2,), 0.5)
    assert mod.realized_gain(new, old, arm, payoff, p).mean().item() == 3.0
    assert mod.realized_gain(new, old, arm, payoff - 5, p).mean().item() == 3.0


def test_online_fit_preserves_parent_when_neither_arm_earns_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_online_choices")
    catalogue = torch.tensor([[0, 0, 8], [1, 0, 8], [-1, 0, 8]])
    actor = mod.TimingPolicy(catalogue)
    value = mod.mlp(192, 1)
    with torch.no_grad():
        actor.out.weight.zero_()
        actor.out.bias.copy_(torch.tensor([0.5, 0, -1.0]))
        actor.factors.weight.zero_()
        actor.factors.bias.zero_()
        for parameter in value.parameters():
            parameter.zero_()
    parent = tmp_path / "parent.bin"
    learner = tmp_path / "learner.pt"
    mod.save_native(actor, catalogue.numpy(), parent)
    torch.save(
        {"actor": actor.state_dict(), "catalogue": catalogue, "value": value.state_dict()}, learner
    )
    t = np.arange(121, dtype="f4") * 0.04
    state = np.zeros((len(t), 192), "f4")
    state[:, 23] = 1
    rows = np.arange(3, 104, 20)
    draws = np.arange(len(rows)) % 2
    applied = np.repeat(catalogue.numpy()[:1], len(t), axis=0)
    applied[rows] = catalogue.numpy()[draws]
    manifest = []
    for i in range(4):
        path = tmp_path / f"test-{i}.npz"
        np.savez(
            path,
            timing_state=state,
            control_time=t,
            controls=applied,
            time=np.arange(48, dtype="f4") * 0.1,
            elapsed_ticks=np.full(48, 5),
            events=np.zeros((48, 8), "f4"),
            choice_rows=rows,
            choice_draw=draws,
            choice_pairs=np.tile([0, 1], (len(rows), 1)),
            choice_probability=np.full(len(rows), 0.5, "f4"),
            choice_gap=np.full(len(rows), 0.5, "f4"),
        )
        manifest.append(
            {
                "source": f"native-development/synthetic-test-{i}",
                "file": path.name,
                "sha256": mod.sha(path),
                "behavior_policy_sha256": mod.sha(parent),
                "split": "train" if i < 2 else "holdout",
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "result"
    result = mod.fit(tmp_path, parent, learner, out)
    assert min(result["trials"].values()) > 0 and result["selected"]["scale"] == 0
    assert (out / "world_iql.bin").read_bytes() == parent.read_bytes()

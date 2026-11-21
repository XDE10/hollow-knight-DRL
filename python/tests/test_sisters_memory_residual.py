from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def model(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_memory_residual")


def test_memory_reset_matches_game_clock_and_phase(model):
    time = np.arange(90, dtype="f4") * 0.04
    state = np.zeros((len(time), 192), dtype="f4")
    state[1:, 23] = 1
    time[85:] += 0.20
    blocks = model.segments(time, state)
    assert [b[0] for b in blocks] == [1, 81, 85]
    assert [i for block in blocks for i in block] == list(range(1, 90))


def test_discounted_targets_censor_trajectories_at_gaps_and_keep_terminal(model):
    t = np.arange(151, dtype="f8") * 0.04
    s = np.zeros((len(t), 192), "f4")
    s[:, 4] = 1
    s[:, 23] = 1
    ev = np.zeros((len(t), 8), "f4")
    ev[6, 0] = 21
    ev[12, 1] = 1
    s[13:, 4] = 8 / 9
    v = np.ones(len(t))
    _, valid, labels, known = model.measured_targets(t, s, ev, v, v)
    assert valid[0] and known[0].all()
    assert tuple(labels[0]) == (1, 0)
    assert not valid[-2]
    ev[-1, 6] = 1
    rd, valid, _, known = model.measured_targets(t, s, ev, v, v)
    assert valid[-1] and known[-1].all() and rd[-1] == pytest.approx(7.0)
    t[50:] += 0.20
    _, valid, _, known = model.measured_targets(t, s, ev, v, v)
    assert not valid[0] and not known[46].any()


def test_recurrent_payload_matches_pytorch_layout(model, tmp_path):
    from train_sisters_timing_iql import (
        TimingPolicy,
        memory_components,
        phase_checkpoint,
        phase_components,
        save_native,
    )

    cat = torch.tensor([[0, 0, 0], [1, 0, 8]])
    p = tmp_path / "leaf.bin"
    save_native(TimingPolicy(cat, True), cat.numpy(), p)
    leaf = p.read_bytes()
    net = model.MemoryActor(len(cat), "raw")
    params = list(net.memory.parameters()) + list(net.delta.parameters())
    weights = b"".join(p.detach().numpy().astype("<f4").tobytes() for p in params)
    branch = struct.pack("<2i", model.MAGIC, len(leaf)) + leaf + weights
    assert phase_components(phase_checkpoint(leaf, branch)) == (leaf, branch)
    assert memory_components(branch)[0] == leaf
    with pytest.raises(ValueError):
        memory_components(branch[:-4])


def test_external_assignments_distinguish_enforced_holds_from_releases(model):
    draw = np.array([False, False, True, True, False])
    assert list(model.external_assignments({"candidate_draw": draw})) == [2, 4]
    assert list(model.external_assignments({"teacher_acted": draw[:, None]})) == [2, 4]
    tape = dict(choice_rows=np.array([15, 40]), choice_draw=np.array([0, 1]))
    assert list(model.external_assignments(tape)) == [40]
    tape["choice_hold_stop"] = np.array([19, 44])
    assert list(model.external_assignments(tape)) == [15, 40]
    tape["choice_scores"] = np.ones((2, 2))
    assert list(model.external_assignments(tape)) == [40]
    assert not len(model.external_assignments({"controls": draw}))


def test_intervention_cuts_real_reward_and_retains_successor_value(model):
    n = 140
    t = np.arange(n) * 0.04
    s = np.zeros((n, 192))
    s[:, 23] = 1
    ev = np.zeros((n, 8))
    ev[4, 0] = 21
    ev[5, 2] = 1
    v = np.zeros(n)
    v[5] = 4
    audit = {}
    residual, valid, labels, known = model.measured_targets(
        t, s, ev, v, np.zeros(n), np.array([5]), audit
    )
    expected = 0.75 * (0.995**4 * 0.45 + 0.995**5 * 4)
    assert valid[0] and residual[0] == pytest.approx(expected)
    assert not known[0].any() and known[5].all() and labels[5, 0] == 1
    assert audit["cutoff_states_then_hurt"] == 1
    ev[4, 6] = 1
    residual, valid, _, known = model.measured_targets(t, s, ev, v, v, np.array([5]))
    assert residual[0] == pytest.approx(0.75 * 0.995**4 * 8.45)
    assert valid[0] and known[0].all()


def test_full_late_fight_rewards_keep_numerical_precision(model):
    t = np.arange(8200) * 0.04
    s = np.zeros((len(t), 192))
    s[:, 23] = 1
    ev = np.zeros((len(t), 8))
    ev[0, 0] = 99
    ev[7960, 0] = 21
    v = np.zeros(len(t))
    residual, valid, _, _ = model.measured_targets(t, s, ev, v, v, np.array([], dtype=int))
    assert valid[7900] and residual[7900] == pytest.approx(0.75 * 0.995**60 * 0.45)

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

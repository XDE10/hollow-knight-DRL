from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


def test_packed_option_policy_and_invalid_scale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    commands = torch.tensor([[-1, 0, 0], [1, 0, 8]])
    path = tmp_path / "leaf.bin"
    mod.save_native(mod.TimingPolicy(commands, True), commands.numpy(), path)
    leaf = path.read_bytes()
    params = np.ones(mod.WIDTH * mod.OPTION_FEATURES + 3 * mod.OPTION_FEATURES + 1, dtype="<f4")
    header = struct.pack("<3i", mod.OPTION_MAGIC, len(leaf), len(leaf)) + leaf + leaf
    routed = header + params.tobytes()
    pair = mod.phase_checkpoint(leaf, routed)
    assert mod.phase_components(pair) == (leaf, routed)
    h, r, weights = mod.option_components(routed)
    assert (h, r) == (leaf, leaf)
    np.testing.assert_array_equal(params, weights)
    with pytest.raises(ValueError):
        mod.phase_components(pair[:-4])
    params[(mod.WIDTH + 1) * mod.OPTION_FEATURES] = 0
    with pytest.raises(ValueError):
        mod.phase_checkpoint(leaf, header + params.tobytes())


def test_option_screen_compares_to_both_static_arms(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_option_router")
    arm = np.array([0, 1, 0, 1])
    games = np.array(["a", "a", "b", "b"])
    reward = np.array([0.0, 2.0, 0.0, 2.0])
    estimates = np.tile([0.0, 2.0], (4, 1))
    rows = mod.paired_screen(games, np.ones(4, bool), arm, reward, estimates)
    assert all(r["vs_fair"] == 1 and r["vs_H"] == 2 and r["vs_R"] == 0 for r in rows)
    assert not mod.eligible(rows)

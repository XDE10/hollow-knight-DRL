from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def test_champion_and_its_zero_update_roundtrip(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    replay = importlib.import_module("replay_sisters_champion_router")
    train = importlib.import_module("train_sisters_champion_router")
    parent = ROOT / "models/sisters-native-iql/timing-learned-options/world_iql.bin"
    path = ROOT / "models/sisters-native-iql/timing-ninewin-continuation"
    x = torch.randn(19, 40)
    for name in ("parent-equivalent.bin", "world_iql.bin"):
        saved = path / name
        _, bound, net = replay.load_router(parent, saved)
        assert train.packet(net, bound, parent.read_bytes()) == saved.read_bytes()
        if name == "parent-equivalent.bin":
            torch.testing.assert_close(net(x), torch.zeros(len(x)), atol=0, rtol=0)
    timing = importlib.import_module("train_sisters_timing_iql")
    _, stage = timing.phase_components((path / "world_iql.bin").read_bytes())
    with pytest.raises(ValueError, match="layout"):
        timing.router_residual_components(stage[:-8])
    with pytest.raises(ValueError, match="parameters"):
        timing.router_residual_components(stage[:-4] + struct.pack("<f", float("nan")))


def test_sampled_route_cannot_use_other_clock_or_probability(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = importlib.import_module("sisters_router_rollout")
    clock = np.arange(60, dtype="f4") * 0.04
    states = np.zeros((len(clock), 32), dtype="f4")
    states[5:, 23] = 1
    selected = mod.neural_block_starts(clock)[1:]
    data = np.zeros(len(selected), mod.ROW)
    data["time"] = clock[selected]
    data["p"] = 0.5
    data["uniform"] = [0.9, 0.12]
    data["draw"] = data["uniform"] < 0.5
    header = mod.HEADER.pack(0x31525453, 17, bytes(32), 34, 0.125)
    path = tmp_path / "17.routes"
    path.write_bytes(header + data.tobytes())

    def read():
        return mod.read_routes(path, 17, "00" * 32, 34, 0.125, clock, states)

    np.testing.assert_array_equal(read()["route_rows"], selected)
    for name in ("p", "time", "draw"):
        corrupt = data.copy()
        corrupt[name][0] += 1
        path.write_bytes(header + corrupt.tobytes())
        with pytest.raises(ValueError):
            read()


def test_elapsed_advantage_and_policy_gradient(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    mod = importlib.import_module("train_sisters_champion_ppo")
    val = np.array([2.0, 3.0])
    adv = mod.advantages(np.array([1.0, 5.0]), np.array([0.9, 0]), np.array([0.7, 0]), val)
    np.testing.assert_allclose(adv, [1 + 0.9 * 3 - 2 + 0.9 * 0.7 * (5 - 3), 2.0])
    margin = torch.tensor([0.0, 0.0])
    delta = torch.tensor([0.1, -0.1], requires_grad=True)
    p = mod.probabilities(margin, delta, 0.07, 0.02)
    (-p[0].log() - torch.log1p(-p[1])).backward()
    assert p[0] > 0.5 > p[1] and delta.grad[0] < 0 < delta.grad[1]

from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def test_update_changes_only_native_output_head(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    api = importlib.import_module("train_sisters_primitive_heads")
    parent = ROOT / "models/sisters-native-iql/timing-learned-options/world_iql.bin"
    _, route = api.phase_components(parent.read_bytes())
    arm, _, _ = api.option_components(route)
    net, cat = api.read_leaf(arm)
    ck = torch.load(
        ROOT / "models/sisters-native-iql/timing-preferences/learner.pt",
        map_location="cpu",
        weights_only=True,
    )
    original = api.TimingPolicy(ck["catalogue"], True).eval()
    original.load_state_dict(ck["actor"])
    rng = torch.Generator().manual_seed(3)
    h = torch.randn(5, 192, generator=rng)
    torch.testing.assert_close(original.heads(h)[0], net.heads(h)[0], rtol=1e-4, atol=1e-4)
    delta = torch.nn.Linear(192, len(cat))
    assert api.replace_head(arm, delta, 0) == arm
    changed, _ = api.read_leaf(api.replace_head(arm, delta, 0.1))
    torch.testing.assert_close(
        changed.heads(h)[0], net.heads(h)[0] + 0.1 * delta(h), rtol=1e-4, atol=1e-4
    )


def test_native_primitive_receipt_rejects_a_clock_shift(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    api = importlib.import_module("sisters_primitive_rollout")
    clock = np.arange(60, dtype="f4") * np.float32(0.04)
    states = np.zeros((len(clock), 192), "f4")
    states[:, 23] = 1
    states[4:25, 146] = 1
    states[29:, 96] = 1
    states[29:31, 9] = 1
    rows = api.onsets(clock, states)
    assert (
        rows[0] == 4
        and (rows >= 31).any()
        and not np.isin(rows, [29, 30]).any()
        and (np.diff(clock[rows]) >= 0.3999).all()
    )
    values = np.zeros(len(rows), api.ROW)
    values["time"] = clock[rows]
    values["p"] = 0.81
    p = tmp_path / "1.keys"
    header = struct.pack("<IQ32sid", 0x31534B50, 1, bytes(32), 674, 1.0)
    p.write_bytes(header + values.tobytes())
    np.testing.assert_array_equal(
        api.read_keys(p, 1, "00" * 32, 674, 1.0, clock, states)["key_rows"], rows
    )
    values["time"][0] += 0.04
    p.write_bytes(header + values.tobytes())
    with pytest.raises(ValueError, match="decision"):
        api.read_keys(p, 1, "00" * 32, 674, 1.0, clock, states)


def test_true_ppo_distribution_keeps_feasible_exploration_and_native_greedy(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    api = importlib.import_module("train_sisters_primitive_ppo")
    logits = torch.tensor([[3.0, 2.0, -1.0, 1.0]], requires_grad=True)
    legal = torch.tensor([[False, True, False, True]])
    p = api.distribution(logits, legal, 1.0)
    assert p[0, 0] > 0.8 and p[0, 1] > 0.02 / 3 and p[0, 2] == 0 and p[0, 3] > 0.02 / 3
    torch.testing.assert_close(p.sum(1), torch.ones(1))
    decisive = api.distribution(torch.tensor([[100.0, 2.0, -1.0, 1.0]]), legal, 1.0)
    assert decisive[0, 0] <= 0.900001
    solo = api.distribution(logits, torch.zeros_like(legal), 1.0)
    torch.testing.assert_close(solo, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    (-p[0, 1].log()).backward()
    assert logits.grad[0, 1] < 0 and logits.grad[0, 2] == 0

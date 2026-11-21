from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return tuple(
        importlib.import_module(name)
        for name in (
            "sisters_online_choices",
            "sisters_compound_returns",
            "train_sisters_timing_iql",
            "train_sisters_compound_options",
        )
    )


def test_compound_receipt_keeps_scores_and_hold(modules, tmp_path):
    choices, _, _, _ = modules
    times = np.arange(13, dtype="f4") * 0.04
    state = np.zeros((len(times), 192), "f4")
    state[:, 23] = 1
    catalogue = np.array([[1, 0, 0], [-1, 0, 3]])
    keys = np.broadcast_to(catalogue[1], (len(times), 3)).copy()
    sample = struct.pack("<f3i5f", times[1], 0, 1, 1, 0.5, 4.0, 9.0, 5.0, 0.2)
    path = tmp_path / "draw.choices"
    path.write_bytes(struct.pack("<IQ", 0x33434B48, 12) + sample)
    record = choices.read_choices(path, 12, times, state, keys, catalogue)
    np.testing.assert_allclose(record["choice_scores"], [[9.0, 5.0]])
    assert record["choice_hold_stop"].tolist() == [9]
    catalogue[1, 2] = 0
    keys[:, 2] = 0
    with pytest.raises(ValueError, match="same maneuver"):
        choices.read_choices(path, 12, times, state, keys, catalogue)


def test_full_horizon_keeps_terminal_and_both_native_mask_margins(modules):
    _, label, _, _ = modules
    t = np.arange(11, dtype="f4") * 0.04
    state = np.zeros((len(t), 192), "f4")
    state[:, 4] = np.array([3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]) / 9.0
    ob = np.array([0, 0.1, 0.2, 0.3, 0.4], "f4")
    ticks = np.full(5, 5)
    events = np.zeros((5, 8), "f4")
    events[2, 1] = 1
    events[4, 7] = 1
    idx, v = label.observed_outcomes(t, state, np.array([2]), ob, ticks, events, 4)
    assert idx.tolist() == [0]
    assert v[0, 1] == 3 and v[0, 3] == 1
    events[4, 7] = 0
    idx, _ = label.observed_outcomes(t, state, np.array([2]), ob, ticks, events, 4)
    assert len(idx) == 0


def test_return_selector_and_serialized_layout(modules, tmp_path):
    _, _, model, trainer = modules
    x = np.tile([[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], (30, 1))
    arm = np.tile([0, 1, 0, 1], 30)
    y = np.where(arm == 1, 3 * x[:, 0], 0.0)
    w = trainer.fit_gate(x, arm, y, np.repeat(np.arange(6), 20), np.eye(2))
    assert w[0] + w[1] > 0 and w[0] - w[1] < 0
    cat = torch.tensor([[0, 0, 0], [1, 0, 8]])
    raw_path = tmp_path / "leaf.bin"
    model.save_native(model.TimingPolicy(cat, True), cat.numpy(), raw_path)
    leaf = raw_path.read_bytes()
    heads = np.ones((2, 193), dtype="<f4")
    late = struct.pack("<2i", model.COMPOUND_MAGIC, len(leaf)) + leaf + heads.tobytes()
    pair = model.phase_checkpoint(leaf, late)
    assert model.phase_components(pair) == (leaf, late)
    other, risk, selector = model.compound_components(late)
    assert other == leaf
    np.testing.assert_array_equal(risk, heads[0])
    np.testing.assert_array_equal(selector, heads[1])
    with pytest.raises(ValueError):
        model.compound_components(late[:-4])

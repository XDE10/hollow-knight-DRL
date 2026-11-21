from __future__ import annotations

import hashlib
import importlib
import struct
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def options(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("sisters_directional_options")


def test_masked_proposals_preserve_base_keys_and_sampled_input(options, tmp_path):
    catalogue = np.array([[-1, 0, 8], [1, 0, 1], [0, -1, 16], [1, 0, 8], [0, 0, 8]])
    state = np.zeros((35, 192), "f4")
    state[:, 23] = 1
    state[:, 7] = 1
    scores = np.array([[1, 12, 10, 2, 3.0]])
    slots, primary = options.menu_slots(scores, catalogue, state[:1])
    np.testing.assert_array_equal(slots, [[-1, 0, -1, 4, 1, 3]])
    np.testing.assert_array_equal(primary, [4])
    logits = options.menu_logits(np.zeros((1, 192)), np.zeros((6, 193)), slots, primary)
    p = options.behavior_probabilities(logits)
    assert p[0, 4] == pytest.approx(0.95 * 0.75 + 0.05 / 4)
    assert p[0, 0] == 0 and abs(p.sum() - 1) < 1e-10
    head = struct.pack("<if", options.HEAD_MAGIC, 0.9) + bytes(7 * 193 * 4)
    times = np.arange(35, dtype="f4") * 0.04
    issued = np.tile(catalogue[0], (35, 1))
    row = np.zeros(1, dtype=options.ROW)
    row["time"], row["primary"], row["draw"] = times[2], 4, 1
    row["slots"], row["probability"] = slots, p
    packet = struct.pack("<IQf", options.TAPE_MAGIC, 33, 0.9) + hashlib.sha256(head).digest()
    tape = tmp_path / "rollout"
    tape.write_bytes(packet + row.tobytes())
    trace = options.read_menu(tape, 33, times, state, issued, catalogue, head)
    assert trace["choice_rows"].tolist() == [2]
    assert trace["choice_hold_stop"].tolist() == [
        np.searchsorted(times, times[2] + np.float32(0.12))
    ]
    row["draw"] = 0
    tape.write_bytes(packet + row.tobytes())
    with pytest.raises(ValueError, match="Invalid directional"):
        options.read_menu(tape, 33, times, state, issued, catalogue, head)


def test_categorical_update_uses_sampled_advantages_and_finite_mask(options):
    train = importlib.import_module("train_sisters_directional_ppo")
    rng = np.random.default_rng(34)
    x = rng.normal(size=(100, 3))
    baseline = np.tile([0, 0, -np.inf, -np.inf, 0, 0.0], (len(x), 1))
    probability = options.behavior_probabilities(baseline)
    drawn = rng.choice([0, 1, 4, 5], size=len(x))
    advant = np.where(drawn == 1, 2, -1)
    slopes, bias, stats = train.fit_selector(x, baseline, probability, drawn, advant)
    updated = options.behavior_probabilities(baseline + x @ slopes.T + bias)
    assert stats["epochs"] and 0 < stats["kl"] <= 0.035
    assert updated[:, 1].mean() > 0.25
    assert np.all(updated[:, 2:4] == 0)

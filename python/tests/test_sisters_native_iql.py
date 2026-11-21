from __future__ import annotations

import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def native(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_native_iql")


def test_interval_damage_uses_inputs_before_observed_losses(native):
    t = np.arange(9, dtype="f4") * 0.04
    starts = np.array([0, 0.16], dtype="f4")
    ticks = np.array([8, 8])
    events = np.zeros((2, 8), dtype="f4")
    events[0, 1] = 2
    events[1, [0, 2, 7]] = [21, 1, 1]
    hp = np.array([9, 8, 8, 8, 7, 7, 6, 6, 6])
    actual = native.attribute_events(t, starts, ticks, events, health_masks=hp)
    expected = np.zeros((len(t), 8), dtype="f4")
    expected[[0, 3], 1] = 1  # The loss at t=0.16 belongs to the first interval.
    expected[5, 2] = 1
    expected[7, [0, 7]] = [21, 1]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("t", "hp", "values"),
    [
        ([0, 0.04, 0.08, 0.12], [9, 8, 7, 7], {1: 1, 2: 1}),
        ([0, 0.04, 0.08, 0.12], [8, 9, 8, 8], {2: 1, 5: 1}),
        ([0, 0.04, 0.08, 0.12], [9, 8, 8, 8], {1: 2}),
        ([0, 0.10, 0.12], [9, 8, 8], {3: 1}),
        ([0, 0.04, 0.08, 0.12], [9, 8.5, 8, 8], {2: 1}),
    ],
)
def test_ambiguous_damage_preserves_original_event_totals(native, t, hp, values):
    events = np.zeros((1, 8), dtype="f4")
    for column, value in values.items():
        events[0, column] = value
    args = np.asarray(t), np.array([0.0]), np.array([8]), events
    actual = native.attribute_events(*args, health_masks=np.asarray(hp))
    np.testing.assert_array_equal(actual, native.attribute_events(*args))
    np.testing.assert_array_equal(actual.sum(0), events.sum(0))


def test_history_and_clocks_are_causal_and_restart_at_fight_boundary(native):
    raw = np.tile(np.arange(7, dtype="f4")[:, None], (1, 56))
    times = np.arange(7, dtype="f4") * 0.04
    actions = np.array([0, 10, 2, 0, 0, 0, 0])
    state = native.decision_states(raw, times, actions)
    np.testing.assert_allclose(state[:, 56], [0, 0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(state[:, 112], [0, 0, 0, 0, 0, 1, 2])
    np.testing.assert_allclose(state[:4, 170], [1, 1, 0.04, 0.08], atol=1e-5)
    np.testing.assert_allclose(state[:4, 174], [0, 0, 1, 1], atol=1e-5)
    actions[3:] = 511
    np.testing.assert_array_equal(state[:4], native.decision_states(raw, times, actions)[:4])
    np.testing.assert_array_equal(
        native.decision_states(raw[:1], times[:1], actions[:1])[0, 168:174], np.ones(6)
    )


def test_credit_events_once_to_last_action_before_observation_end(native):
    t = np.array([0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24])
    events = np.zeros((3, 8))
    events[0, 0] = 21
    events[1, 2] = 1
    events[2, 6] = 1
    out = native.attribute_events(t, np.array([0, 0.08, 0.2]), np.array([4, 6, 3]), events)
    assert out[1, 0] == 21 and out[4, 2] == 1 and out[6, 6] == 1
    np.testing.assert_array_equal(out.sum(0), events.sum(0))


def test_exported_dense_logits_match_training_policy(native, tmp_path):
    torch.manual_seed(19)
    cat = np.array([[-1, 0, 0], [0, -1, 8], [1, 1, 130], [1, 0, 16]])
    actor = native.Policy(torch.from_numpy(cat))
    out = tmp_path / "native.bin"
    native.export(actor, cat, out)
    with out.open("rb") as f:
        magic, n, h, j, k = struct.unpack("<5i", f.read(20))
        assert (magic, n, h, j, k) == (0x314C5149, 180, 160, 128, len(cat))
        np.testing.assert_array_equal(np.frombuffer(f.read(k * 12), "<i4").reshape(-1, 3), cat)
        w = np.frombuffer(f.read(), "<f4")
    s = torch.randn(5, n)
    expected = actor(s).detach().numpy()
    x = s.numpy()
    pos = 0
    for inputs, outputs, activate in [(n, h, True), (h, j, True), (j, k, False)]:
        end = pos + inputs * outputs
        x = x @ w[pos:end].reshape(outputs, inputs).T + w[end : end + outputs]
        if activate:
            x = np.maximum(x, 0)
        pos = end + outputs
    assert pos == len(w)
    np.testing.assert_allclose(x, expected, atol=2e-6)


def test_corrections_keep_executed_credit_and_isolate_holdout(native, tmp_path):
    manifest = []
    for split, executed, label in [("train", 8, 4), ("holdout", 256, 256)]:
        actions = np.zeros((4, 3), dtype="i2")
        actions[0, 2] = executed
        labels = actions.copy()
        labels[0, 2] = label
        events = np.zeros((2, 8), "f4")
        events[0, 0] = 21
        np.savez(
            tmp_path / f"{split}.npz",
            controls=actions,
            teacher_controls=labels,
            student_features=np.zeros((4, 56), "f4"),
            control_time=np.arange(4, dtype="f4") * 0.04,
            time=np.array([0, 0.04], "f4"),
            elapsed_ticks=np.array([2, 6]),
            events=events,
        )
        manifest.append(dict(source=split, file=f"{split}.npz", split=split, won=False))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    batches, catalogue, _ = native.load_data(tmp_path, "cpu")
    states, executed, _, reward, _, _, _, expert, corrected = batches["train"]
    assert catalogue[executed[0], 2] == 8 and catalogue[expert[0], 2] == 4
    assert reward[0].item() == pytest.approx(0.45)
    assert states[1, 170].item() == pytest.approx(0.04)
    assert states[1, 169].item() == 1 and corrected.all()
    assert batches["holdout"][1][0].item() == -1 and 256 not in catalogue[:, 2]

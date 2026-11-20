from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def outcomes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("sisters_episode_returns")


def test_only_an_observed_final_outcome_counts(outcomes) -> None:
    state = np.zeros((5, 192), dtype="f4")
    state[1:, 23] = 1
    clock = np.array([0, 45, 90, 135, 140], dtype="f4")
    events = np.zeros((5, 8), dtype="f4")
    features, missing = outcomes.outcome_samples(clock, state, events, False)
    np.testing.assert_allclose(features[:, -1], [0, 0, 0.45, 0.9, 0.95])
    assert np.isnan(missing).all()
    events[3, 6] = 1
    _, wins = outcomes.outcome_samples(clock, state, events, True)
    np.testing.assert_allclose(wins[:4], np.exp2(np.array([-1.5, -1, -0.5, 0])))
    assert np.isnan(wins[-1])
    assert np.isnan(outcomes.outcome_samples(clock, state, events, False)[1]).all()
    events[3, 7] = 1
    assert np.isnan(outcomes.outcome_samples(clock, state, events, True)[1]).all()


def test_baseline_omits_own_fight_and_weights_each_fight_equally(outcomes) -> None:
    def fight(n: int, value: float):
        return np.ones((n, 6), dtype="f4"), np.full(n, value, dtype="f4")

    data = {
        "train": [fight(3, 1), fight(9, -1), fight(4, np.nan)],
        "holdout": [fight(6, 1), fight(2, np.nan)],
    }
    weights, stats = outcomes.outcome_advantages(data)
    np.testing.assert_allclose(weights["train"][0], 2)
    np.testing.assert_allclose(weights["train"][1], -2)
    np.testing.assert_allclose(weights["train"][2], 0)
    np.testing.assert_allclose(weights["holdout"][0], 1)
    np.testing.assert_allclose(weights["holdout"][1], 0)
    assert stats["holdout"]["censored_fights"] == 1
    data["holdout"][0] = fight(2, -1)
    changed, _ = outcomes.outcome_advantages(data)
    for a, b in zip(changed["train"], weights["train"], strict=True):
        np.testing.assert_array_equal(a, b)

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def test_native_window_counts_an_early_mask_once(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("sisters_maneuver_credit")
    t = np.arange(51, dtype="f4") * 0.04
    s = np.zeros((len(t), 192), "f4")
    s[:, 4] = 1
    s[2:, 4] = 8 / 9
    et = np.arange(16, dtype="f4") * 0.12
    e = np.zeros((len(et), 8), "f4")
    e[0, 1] = 1  # This interval straddles the moment the maneuver was drawn.
    w = np.array([0.02, -1.5, -1.5, -2, -1.5, 0, 8, -6], dtype="f4")
    rows = np.array([1, 20])
    args = (et, np.full(len(et), 6), e, rows, w)
    old = module.choice_outcomes(t, *args)
    new = module.native_health_outcomes(t, s, *args)
    assert new[0].tolist() == [0, 1] and np.all(old[1] == 0)
    assert np.isclose(new[1][0], -1.5 * 0.995**15) and new[1][1] == 0
    # A hit visible on both clocks is already penalized by the event reward.
    s[2:5, 4] = 1
    e[0, 1], e[1, 1] = 0, 1
    old = module.choice_outcomes(t, *args)
    new = module.native_health_outcomes(t, s, *args)
    np.testing.assert_allclose(new[1], old[1])


def test_doubly_robust_coin_identity_and_disjoint_games(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("sisters_maneuver_credit")
    arm = np.array([0, 1] * 9)
    q = np.tile([3.0, -4.0], (len(arm), 1))
    out = mod.dr_contrast(arm, 2.0 + 5 * arm, q)
    assert out.mean() == 5
    x = np.arange(len(arm) * 2, dtype=float).reshape(-1, 2)
    games = np.repeat(np.arange(3), 6)
    y = arm.astype(float) * 2 + 0.5
    base, _ = mod.crossfit_options(x, arm, y, games, x[:2])
    modified = y.copy()
    modified[:6] += 3
    other, _ = mod.crossfit_options(x, arm, modified, games, x[:2])
    np.testing.assert_equal(base[:6], other[:6])
    assert not np.allclose(base[6:], other[6:])

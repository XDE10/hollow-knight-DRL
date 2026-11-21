"""Estimate short maneuver returns using the native health clock and randomized assignments."""

from __future__ import annotations

import numpy as np
from sisters_online_choices import choice_outcomes


def native_health_outcomes(
    times: np.ndarray,
    state: np.ndarray,
    observation_times: np.ndarray,
    ticks: np.ndarray,
    events: np.ndarray,
    rows: np.ndarray,
    reward_weights: np.ndarray,
    horizon: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep observed event rewards and charge additional losses at either margin.

    Coarse observations that cross the draw cannot attribute a specific source.
    A measured health loss in that gap still incurs the common mask cost.
    """
    kept, payout, future, discount = choice_outcomes(
        times, observation_times, ticks, events, rows, reward_weights, horizon
    )
    ends = np.r_[observation_times[1:], observation_times[-1] + ticks[-1] * 0.02]
    hp = np.rint(state[:, 4] * 9)
    if not np.isfinite(hp).all() or not np.allclose(hp, state[:, 4] * 9, atol=0.001):
        raise ValueError("Expected a native nine-mask health trace")
    valid = np.ones(len(kept), bool)
    for j, trial in enumerate(kept):
        row = int(rows[trial])
        start = float(times[row])
        a = np.searchsorted(observation_times, start - 1e-5)
        b = np.searchsorted(ends, start + horizon + 1e-5, side="right")
        terminal = np.flatnonzero(events[a:b, 6:].any(1))
        if len(terminal):
            b = a + int(terminal[0]) + 1
            last = int(future[j])
        else:
            last = np.searchsorted(times, start + horizon + 1e-5, side="right") - 1
            if start + horizon - times[last] > 0.081:
                valid[j] = False
                continue
            # A terminal inside the trailing partial observation is censored.
            if (
                b < len(events)
                and events[b, 6:].any()
                and observation_times[b] < start + horizon
            ):
                valid[j] = False
                continue
        if (np.diff(times[row : last + 1]) > 0.081).any():
            valid[j] = False
            continue
        debits = np.maximum(-np.diff(hp[row : last + 1]), 0).astype(int)
        captured = int(events[a:b, 1:5].sum())
        extra = int(debits.sum()) - captured
        if extra > 0:
            # The aggregate events already account for 'captured' masks. Count
            # only the excess at the endpoint discount; no guessed damage type.
            payout[j] += (
                extra
                * float(np.max(reward_weights[1:5]))
                * 0.995 ** ((times[last] - start) / 0.04)
            )
        future[j] = last
        if not len(terminal):
            discount[j] = 0.995 ** ((times[last] - start) / 0.04)
    return kept[valid], payout[valid], future[valid], discount[valid]


def dr_contrast(
    arms: np.ndarray, returns: np.ndarray, predictions: np.ndarray
) -> np.ndarray:
    """Fair-coin return contrast for the two arms on observed outcomes."""
    if predictions.shape != (len(arms), 2) or not np.isin(arms, [0, 1]).all():
        raise ValueError(
            "Expected two potential outcomes and a binary random assignment"
        )
    arms = np.asarray(arms, dtype=np.int64)
    chosen = predictions[np.arange(len(arms)), arms]
    return (
        predictions[:, 1] - predictions[:, 0] + 2 * (2 * arms - 1) * (returns - chosen)
    )


def option_features(hidden: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Small fixed projection of the frozen encoder, plus both available directions."""
    rng = np.random.default_rng(137)
    projection = rng.standard_normal((hidden.shape[1], 12)) / np.sqrt(hidden.shape[1])
    return np.column_stack(
        [hidden @ projection, np.eye(3)[moves[:, 0] + 1], np.eye(3)[moves[:, 1] + 1]]
    )


def predict_arms(
    x: np.ndarray, arm: np.ndarray, y: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Fit a regularized return model in each arm on the supplied training fights."""
    center, scale = x.mean(0), x.std(0).clip(0.1)
    X = np.column_stack([np.ones(len(x)), ((x - center) / scale).clip(-6, 6)])
    Xtest = np.column_stack([np.ones(len(test)), ((test - center) / scale).clip(-6, 6)])
    ridge = np.diag(np.r_[1e-3, np.full(x.shape[1], 25.0)])
    result = []
    for choice in (0, 1):
        selected = arm == choice
        if not selected.any():
            result.append(np.full(len(test), y.mean()))
            continue
        design = X[selected]
        w = np.linalg.solve(design.T @ design + ridge, design.T @ y[selected])
        result.append((Xtest @ w).clip(-6, 6))
    return np.column_stack(result)


def crossfit_options(
    x: np.ndarray, arm: np.ndarray, y: np.ndarray, games: np.ndarray, held_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude the entire evaluated game when predicting support for its policy gradient."""
    unique = np.unique(games)
    if len(unique) < 2:
        raise ValueError(
            "Need at least two independent training games for cross-fitting"
        )
    train_pred = np.empty((len(x), 2))
    for game in unique:
        valid = games == game
        train_pred[valid] = predict_arms(x[~valid], arm[~valid], y[~valid], x[valid])
    return train_pred, predict_arms(x, arm, y, held_x)

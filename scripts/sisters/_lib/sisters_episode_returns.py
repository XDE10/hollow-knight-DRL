"""Fight-balanced Monte Carlo advantages for the actual inputs in Sisters replay."""

from __future__ import annotations

import numpy as np

HALF_LIFE_SECONDS = 90.0


def outcome_samples(
    clock: np.ndarray, state: np.ndarray, events: np.ndarray, won: bool
) -> tuple[np.ndarray, np.ndarray]:
    """A causal state baseline and the signed, discounted final outcome on this tape."""
    phase = state[:, 23] > 0.5
    start = np.maximum.accumulate(
        np.where(np.r_[True, phase[1:] != phase[:-1]], clock, -np.inf)
    )
    features = np.column_stack(
        (
            state[:, 4],
            state[:, 5],
            *(state[:, i] * state[:, i + 13] for i in (24, 60, 96)),
            (clock - start) / 100,
        )
    )
    target = np.full(len(clock), np.nan, dtype="f4")
    ends = np.flatnonzero((events[:, 6:] > 0).any(1))
    if len(ends):
        end = int(ends[0])
        victory, death = events[end, 6:8] > 0
        if victory != death and bool(victory) == won:
            dt = clock[end] - clock[: end + 1]
            target[: end + 1] = (1 if won else -1) * np.exp2(-dt / HALF_LIFE_SECONDS)
    return features, target


def outcome_advantages(
    fights: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> tuple[dict[str, list[np.ndarray]], dict[str, dict]]:
    """Fit one ridge state baseline per omitted training fight, and score unseen fights."""
    if not fights.get("train"):
        raise ValueError("Whole-fight advantages need completed training fights")
    moments = []
    for features, target in fights["train"]:
        good = np.isfinite(target)
        width = features.shape[1] + 1
        if not good.any():
            moments.append((np.zeros((width, width)), np.zeros(width)))
            continue
        x = np.column_stack((np.ones(int(good.sum())), features[good])).astype("f8")
        moments.append((x.T @ x / len(x), x.T @ target[good] / len(x)))
    total_gram = sum((m[0] for m in moments), start=np.zeros_like(moments[0][0]))
    total_rhs = sum((m[1] for m in moments), start=np.zeros_like(moments[0][1]))
    if total_gram[0, 0] < 2:
        raise ValueError(
            "Whole-fight advantages need at least two completed training fights"
        )

    def fit(gram: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float, float]:
        gram, rhs = gram / gram[0, 0], rhs / gram[0, 0]
        mean = gram[0, 1:]
        cov = gram[1:, 1:] - np.outer(mean, mean)
        ridge = 0.25 * np.diag(np.maximum(np.diag(cov), 0.05**2))
        coef = np.linalg.solve(cov + ridge, rhs[1:] - mean * rhs[0])
        return coef, float(rhs[0] - mean @ coef), float(rhs[0])

    held_model = fit(total_gram, total_rhs)
    advantages: dict[str, list[np.ndarray]] = {}
    stats: dict[str, dict] = {}
    for split, samples in fights.items():
        advantages[split] = []
        errors = []
        for i, (x, target) in enumerate(samples):
            good = np.isfinite(target)
            adv = np.zeros(len(target), dtype="f4")
            if good.any():
                coef, offset, constant = (
                    fit(total_gram - moments[i][0], total_rhs - moments[i][1])
                    if split == "train"
                    else held_model
                )
                prediction = np.clip(x[good] @ coef + offset, -1, 1)
                adv[good] = target[good] - prediction
                errors.append(
                    (
                        float(np.mean(adv[good] ** 2)),
                        float(np.mean((target[good] - constant) ** 2)),
                    )
                )
            advantages[split].append(adv)
        stats[split] = {
            "completed_fights": len(errors),
            "censored_fights": len(samples) - len(errors),
            "baseline_mse": float(np.mean([e[0] for e in errors])) if errors else None,
            "constant_mse": float(np.mean([e[1] for e in errors])) if errors else None,
        }
    return advantages, stats

"""Estimate the value of continuing a real focus attempt from complete development fights."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
from train_sisters_timing_iql import (
    FOCUS_FEATURES,
    FOCUS_MAGIC,
    phase_checkpoint,
    phase_components,
)

HORIZON = 0.52
# These outcomes use actual changes while the game is running. Capped/coarse heal requests are excluded.
REWARD = np.array([2.0, -3.0, 0.45 / 21.0, -0.12 / 33.0])


def focus_features(state: np.ndarray) -> np.ndarray:
    """Fixed nonlinear time basis and current visible threat geometry, shared with the game."""
    age = state[:, 178]
    boss_x = np.where(state[:, 24] > 0, np.abs(state[:, 25]), 2.0)
    disc_x = np.where(state[:, 132] > 0, np.abs(state[:, 133]), 2.0)
    return np.column_stack(
        (
            np.ones(len(age)),
            age,
            np.maximum(0, age - 0.6),
            np.maximum(0, age - 1.0),
            np.maximum(0, age - 1.3),
            np.maximum(0, age - 1.6),
            state[:, 4],
            state[:, 5],
            state[:, 7],
            state[:, 9],
            state[:, 12],
            state[:, 21],
            boss_x,
            disc_x,
            state[:, 23],
            age * state[:, 5],
        )
    )


def focus_examples(d) -> tuple[np.ndarray, ...]:
    t = d["control_time"].astype("f8")
    s, action = d["timing_state"], d["controls"]
    hp = np.rint(9 * s[:, 4]).astype(int)
    soul = np.rint(99 * s[:, 5]).astype(int)
    held = (action[:, 2].astype(int) & 32) != 0
    if not len(t):
        raise ValueError("Empty native tape")
    starts = np.r_[
        0.0,
        np.maximum.accumulate(
            np.where(held & ~np.r_[False, held[:-1]], np.arange(len(t)), 0)
        )[:-1],
    ].astype(int)
    # The most recent input must belong to the same genuine, still uncompleted focus attempt.
    segment = np.cumsum(np.r_[False, np.diff(t) > 0.081])
    hp_changes = np.r_[0, np.cumsum(np.diff(hp) != 0)]
    last = np.searchsorted(t, t + HORIZON + 0.00001, "right") - 1
    eligible = (
        np.r_[False, held[:-1]]
        & (s[:, 169] > 0.5)
        & (hp > 0)
        & (hp < 9)
        & (hp_changes[starts] == hp_changes)
        & (segment[starts] == segment[last])
        & (s[starts, 23] == s[last, 23])
        & (s[:, 178] <= 1.95)
        & (t[last] >= t + HORIZON - 0.081)
    )
    # Count full event intervals only. Measured HP gains and debits come from the native clock.
    coarse = d["time"].astype("f8")
    ends = np.r_[coarse[1:], coarse[-1] + d["elapsed_ticks"][-1] * 0.02]
    rows = np.flatnonzero(eligible)
    a = np.searchsorted(coarse, t[rows] - 0.00001)
    b = np.searchsorted(ends, t[rows] + HORIZON + 0.00001, "right")
    valid = a < b
    rows, a, b = rows[valid], a[valid], b[valid]
    u = last[rows]
    gained = np.r_[0, np.cumsum(np.maximum(np.diff(hp), 0))]
    lost = np.r_[0, np.cumsum(np.maximum(-np.diff(hp), 0))]
    cost = np.r_[0, np.cumsum(np.maximum(-np.diff(soul), 0))]
    damage = np.r_[0, np.cumsum(d["events"][:, 0])]
    outcomes = np.column_stack(
        (
            gained[u] - gained[rows],
            lost[u] - lost[rows],
            damage[b] - damage[a],
            cost[u] - cost[rows],
        )
    )
    return focus_features(s[rows]), held[rows].astype(int), outcomes, starts[rows], rows


def fit(
    x: np.ndarray, arm: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Give each recorded focus bout equal weight within its observed action."""
    weights = []
    for k in (0, 1):
        select = arm == k
        X, Y, bouts = x[select], y[select], groups[select]
        if len(np.unique(bouts)) < 4:
            raise ValueError("Insufficient distinct observed bouts for a focus action")
        _, inverse, n = np.unique(bouts, return_inverse=True, return_counts=True)
        w = 1.0 / n[inverse]
        penalty = np.diag(np.r_[0.02, np.full(FOCUS_FEATURES - 1, 0.8)])
        weights.append(
            np.linalg.solve(X.T @ (w[:, None] * X) + penalty, X.T @ (w[:, None] * Y))
        )
    return np.stack(weights)


def study(archives: list[Path], parent: Path, out: Path) -> None:
    import torch
    from sisters_online_choices import late_catalogue

    seen = set()
    splits = {"train": [], "holdout": []}
    meta = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for game in json.loads(z.read("manifest.json")):
                source = game["source"]
                if source in seen or game["split"] not in splits:
                    raise ValueError("Fight repeated or missing its original split")
                seen.add(source)
                blob = z.read(game["file"])
                if hashlib.sha256(blob).hexdigest() != game["sha256"]:
                    raise ValueError("Changed development tape")
                with np.load(io.BytesIO(blob)) as d:
                    x, arm, y, bouts, rows = focus_examples(d)
                    is_neural = "teacher_controls" not in d
                meta[source] = {
                    "split": game["split"],
                    "teacher_free": is_neural,
                    "rows": len(rows),
                }
                if len(rows):
                    groups = np.array([f"{source}/{i}" for i in bouts])
                    splits[game["split"]].append((x, arm, y, groups, source, is_neural))
    train = [np.concatenate([g[k] for g in splits["train"]]) for k in range(4)]
    model = fit(*train)
    diff = ((model[1] - model[0]) @ REWARD).astype("<f4")
    per_game = []
    for x, arm, y, bouts, source, neural in splits["holdout"]:
        scores = x @ diff
        factual = np.where(arm[:, None] == 1, x @ model[1], x @ model[0])
        release = (arm == 0) & (x[:, 14] > 0.5)
        chosen = release & (scores > 0)
        per_game.append(
            {
                "source": source,
                "teacher_free": bool(neural),
                "rows": len(x),
                "unique_bouts": len(set(bouts)),
                "trio_releases": int(release.sum()),
                "would_extend_release": int(chosen.sum()),
                "realized_reward_mae_logged_action": float(
                    np.abs((factual - y) @ REWARD).mean()
                ),
                "observed_hp_gains_in_overlapping_logged_windows": int(y[:, 0].sum()),
            }
        )
    raw = parent.read_bytes()
    opening, late = phase_components(raw)
    catalogue = late_catalogue(raw)
    if ((catalogue == np.array([0, 0, 32])).all(1)).sum() != 1:
        raise ValueError("Parent has no unambiguous native focus input")
    payload = struct.pack("<2i", FOCUS_MAGIC, len(late)) + late + diff.tobytes()
    deployed = phase_checkpoint(opening, payload)
    out.mkdir(parents=True, exist_ok=False)
    (out / "world_iql.bin").write_bytes(deployed)
    torch.save(
        {
            "outcome_models": torch.from_numpy(model),
            "utility": torch.from_numpy(REWARD),
            "contrast": torch.from_numpy(diff),
        },
        out / "learner.pt",
    )
    doc = {
        "method": "Fit two action-conditioned 0.52-game-second outcome models on executed focus continuations and releases; maximize their measured utility during a still-active native focus attempt. The parent neural actor and its world-assisted IQL training are frozen.",
        "scope": "Observational returns under the recorded follow-on policies. Selection and subsequent actions can confound the contrast. Native game evaluation is required.",
        "horizon": HORIZON,
        "parent_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_sha256": hashlib.sha256(deployed).hexdigest(),
        "archives": {
            p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in archives
        },
        "episodes": meta,
        "training_rows_by_arm": np.bincount(train[1], minlength=2).tolist(),
        "training_bouts_by_arm": [len(set(train[3][train[1] == i])) for i in (0, 1)],
        "heldout": per_game,
    }
    (out / "training.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf8"
    )
    print(
        json.dumps(
            {
                k: doc[k]
                for k in (
                    "training_rows_by_arm",
                    "training_bouts_by_arm",
                    "policy_sha256",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive", dest="archives", type=Path, action="append", required=True
    )
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    study(**vars(p.parse_args()))

"""Compare held-out per-source risk forecasting with and without the issued key."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from sisters_contact_windows import contact_windows
from sklearn.ensemble import HistGradientBoostingClassifier
from train_sisters_native_iql import attribute_events


def load(archive: Path) -> dict[str, tuple[np.ndarray, ...]]:
    parts = {"train": [], "holdout": []}
    with zipfile.ZipFile(archive) as z:
        games = json.loads(z.read("manifest.json"))
        if len({g["source"] for g in games}) != len(games):
            raise ValueError("Duplicate development fight")
        for game in games:
            seed = game["source"]
            if not seed.startswith("native-development/") or game["split"] not in parts:
                raise ValueError("Expected presplit development fights")
            raw = z.read(game["file"])
            if hashlib.sha256(raw).hexdigest() != game["sha256"]:
                raise ValueError("Changed development fight")
            with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                t, s, a = f["control_time"], f["timing_state"], f["controls"]
                events = attribute_events(
                    t,
                    f["time"],
                    f["elapsed_ticks"],
                    f["events"],
                    health_masks=s[:, 4] * 9,
                )
                if "block_starts" in f:
                    bounds = f["block_starts"]
                elif "choice_rows" in f:
                    bounds = f["choice_rows"]
                elif "teacher_controls" in f and "teacher_acted" not in f:
                    raise ValueError(
                        "Restore native controller assignments before fitting contact risk"
                    )
                elif "teacher_acted" in f:
                    bounds = np.flatnonzero(np.diff(f["teacher_acted"])) + 1
                else:
                    bounds = np.empty(0, dtype=int)
                valid, y = contact_windows(t, events, bounds)
                valid &= (s[:, 23] > 0.5) & (s[:, 9] < 0.5) & (s[:, 4] > 0)
                # Use every third decision on a fixed phase independent of action or contact.
                rows = np.flatnonzero(
                    valid & ((np.arange(len(t)) % 3) == (int(seed.split("/")[-1]) % 3))
                )
                if not len(rows):
                    continue
                action = np.column_stack(
                    (a[rows, :2], (a[rows, 2, None] >> np.arange(9)) & 1)
                )
                parts[game["split"]].append(
                    (
                        s[rows],
                        action.astype("f4"),
                        y[rows, :2],
                        np.full(len(rows), int(seed.split("/")[-1])),
                    )
                )
    return {
        split: tuple(np.concatenate(x) for x in zip(*items, strict=True))
        for split, items in parts.items()
    }


def losses(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return -np.where(y, np.log(p), np.log1p(-p))


def run(archive: Path, output: Path) -> None:
    sets = load(archive)
    train, train_action, y, seeds = sets["train"]
    held, held_action, target, held_seeds = sets["holdout"]
    train_with_keys = np.column_stack((train, train_action))
    held_with_keys = np.column_stack((held, held_action))
    report = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "horizon_seconds": 0.32,
        "stride_decisions": 3,
        "train_rows": len(y),
        "held_rows": len(target),
        "train_fights": len(np.unique(seeds)),
        "held_fights": len(np.unique(held_seeds)),
        "comparison": "Paired held-fight log losses. Observational prediction, conditioned on subsequent behavior.",
        "sources": {},
    }
    for i, source in enumerate(("body", "projectile")):
        scores = []
        for inputs, evaluate in ((train, held), (train_with_keys, held_with_keys)):
            model = HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=45,
                learning_rate=0.09,
                l2_regularization=2,
                early_stopping=False,
                random_state=921,
            )
            model.fit(inputs, y[:, i])
            scores.append(losses(target[:, i], model.predict_proba(evaluate)[:, 1]))
        episode = {
            str(seed): [float(v[held_seeds == seed].mean()) for v in scores]
            for seed in np.unique(held_seeds)
        }
        difference = np.array([v[0] - v[1] for v in episode.values()])
        rng = np.random.default_rng(9725)
        bootstrap = rng.choice(
            difference, size=(4000, len(difference)), replace=True
        ).mean(1)
        report["sources"][source] = {
            "train_positive_windows": int(y[:, i].sum()),
            "held_positive_windows": int(target[:, i].sum()),
            "state_log_loss": float(scores[0].mean()),
            "state_and_key_log_loss": float(scores[1].mean()),
            "macro_game_loss_reduction": float(difference.mean()),
            "game_bootstrap_90": np.quantile(bootstrap, [0.05, 0.95]).tolist(),
            "per_game_losses": episode,
        }
        print(source, report["sources"][source]["game_bootstrap_90"], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".partial")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output)

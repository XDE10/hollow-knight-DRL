"""Measure how well past single-sister rollouts predict subsequent net HP drops.

Requires scikit-learn for fitting and scoring. Inputs are two independent rollout
directories from the same task. No games or policy updates are performed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from hkrl import protocol, spaces

HORIZONS = (1, 15)
TARGETS = ("player_hp_drop", "boss_hp_drop")
E = spaces.ENTITY_FEATURE_INDEX
P = spaces.PLAYER_FEATURE_INDEX
G = spaces.GLOBAL_FEATURE_INDEX


def batch_examples(data: Mapping[str, np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Use only contiguous, nonterminal decisions for the same boss and worker."""
    player = data["obs_player"]
    entities = data["obs_entities"]
    mask = data["entity_mask"].astype(bool)
    kinds = entities[..., E["entity_type"]]
    bodies = mask & (kinds == protocol.EntityType.BOSS)
    boss = np.take_along_axis(entities, bodies.argmax(axis=2)[..., None, None], axis=2)[:, :, 0]
    single = (bodies.sum(axis=2) == 1) & (boss[..., E["max_hp"]] > 0)
    ids = data["episode_ids"]
    times = data["obs_global"][..., G["time_in_episode"]]
    valid = (
        single[:-1]
        & single[1:]
        & (boss[:-1, :, E["entity_id"]] == boss[1:, :, E["entity_id"]])
        & (ids[:-1] == ids[1:])
        & (times[1:] > times[:-1])
        & ~data["dones"][:-1].astype(bool)
        & ~data["truncateds"][:-1].astype(bool)
    )

    center = [E["hurtbox_center_x"], E["hurtbox_center_y"]]
    hero_xy = player[..., [P["pos_x"], P["pos_y"]]]
    delta = entities[..., center] - hero_xy[:, :, None, :]
    blades = (
        mask
        & (kinds == protocol.EntityType.PROJECTILE)
        & (entities[..., E["team"]] == protocol.Team.ENEMY)
    )
    distance_sq = np.where(blades, np.square(delta).sum(axis=-1), np.inf)
    nearest = distance_sq.argmin(axis=2)[..., None, None]
    blade = np.take_along_axis(entities, nearest, axis=2)[:, :, 0]
    blade_features = np.concatenate(
        (blade[..., center] - hero_xy, blade[..., [E["vel_x"], E["vel_y"]]]), axis=-1
    )
    blade_features[~blades.any(axis=2)] = 0.0
    player_fields = (
        "pos_x",
        "pos_y",
        "vel_x",
        "vel_y",
        "hp",
        "facing",
        "on_ground",
        "shadow_dashing",
        "invulnerable",
        "dash_cooldown",
        "double_jump_available",
    )
    features = np.concatenate(
        (
            player[..., [P[name] for name in player_fields]],
            boss[..., center] - hero_xy,
            boss[..., [E["vel_x"], E["vel_y"], E["hp"]]],
            blades.sum(axis=2)[..., None],
            blade_features,
            data["actions"],
        ),
        axis=-1,
    ).astype(np.float32)
    hp = np.stack((player[..., P["hp"]], boss[..., E["hp"]]), axis=-1)
    if len(hp) <= horizon:
        return features[:0, 0], np.empty((0, len(TARGETS)), dtype=bool)
    keep = sliding_window_view(valid, horizon, axis=0).all(axis=-1)
    drops = hp[1:] < hp[:-1]
    labels = sliding_window_view(drops, horizon, axis=0).any(axis=-1)
    return features[:-horizon][keep], labels[keep]


def load_examples(directory: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise ValueError(f"No rollout batches in {directory}")
    examples: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {h: [] for h in HORIZONS}
    fields = (
        "obs_player",
        "obs_global",
        "obs_entities",
        "entity_mask",
        "actions",
        "episode_ids",
        "dones",
        "truncateds",
    )
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            data = {field: archive[field] for field in fields}
        for horizon in HORIZONS:
            examples[horizon].append(batch_examples(data, horizon))
    return {
        h: (np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs]))
        for h, pairs in examples.items()
    }


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    args = parser.parse_args()
    if args.train.resolve() == args.holdout.resolve():
        parser.error("supply independent training and holdout runs")
    train = load_examples(args.train)
    holdout = load_examples(args.holdout)
    rows = []
    for horizon in HORIZONS:
        x, y = train[horizon]
        x_test, y_test = holdout[horizon]
        for i, target in enumerate(TARGETS):
            if np.unique(y[:, i]).size != 2 or np.unique(y_test[:, i]).size != 2:
                raise ValueError(f"Both runs need positive and negative {target} examples")
            model = HistGradientBoostingClassifier(
                max_iter=45, max_leaf_nodes=15, early_stopping=False, random_state=0
            ).fit(x, y[:, i])
            prob = model.predict_proba(x_test)[:, 1]
            truth = y_test[:, i]
            rows.append(
                dict(
                    horizon=horizon,
                    target=target,
                    train=len(x),
                    holdout=len(x_test),
                    holdout_positives=int(truth.sum()),
                    constant_ap=float(truth.mean()),
                    average_precision=float(average_precision_score(truth, prob)),
                    brier=float(brier_score_loss(truth, prob)),
                    constant_brier=float(np.square(truth.astype(float) - y[:, i].mean()).mean()),
                )
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

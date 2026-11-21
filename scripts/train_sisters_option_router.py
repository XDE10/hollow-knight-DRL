"""Fit a short-horizon selector between two frozen world-assisted IQL policies."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sisters_control_blocks import control_blocks, neural_block_starts
from sisters_maneuver_credit import crossfit_options, dr_contrast, predict_arms
from train_sisters_native_iql import REWARD_WEIGHTS
from train_sisters_timing_iql import (
    OPTION_FEATURES,
    OPTION_MAGIC,
    WIDTH,
    TimingPolicy,
    mlp,
    motion_history,
    phase_checkpoint,
    phase_components,
    save_native,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Blocks:
    games: np.ndarray
    arms: np.ndarray
    advantages: np.ndarray
    rewards: np.ndarray
    features: np.ndarray


def block_data(
    data: Path,
    learner: dict,
    projection: np.ndarray,
    weights: np.ndarray,
    spec: dict,
    primary_trio: bytes,
) -> dict[str, Blocks]:
    model = TimingPolicy(learner["catalogue"], history=True).eval()
    model.load_state_dict(learner["actor"])
    # The gate reads the primary encoder in the game. Require the same learned
    # embedding and action catalogue before estimating any outcome contrasts.
    with tempfile.TemporaryDirectory() as folder:
        exported = Path(folder) / "trio.bin"
        save_native(model, learner["catalogue"].numpy(), exported)
        payload = exported.read_bytes()
    head_bytes = len(learner["catalogue"]) * (WIDTH + 1) * 4
    if (
        len(payload) != len(primary_trio)
        or payload[:-head_bytes] != primary_trio[:-head_bytes]
        or not np.allclose(
            np.frombuffer(payload[-head_bytes:], "<f4"),
            np.frombuffer(primary_trio[-head_bytes:], "<f4"),
            rtol=1e-6,
            atol=1e-6,
        )
    ):
        raise ValueError("Source learner does not match the deployed primary option")
    value = mlp(WIDTH + 2 * (WIDTH + 1), 1).eval()
    value.load_state_dict(learner["value"])
    by_split = {"train": [], "holdout": []}
    with zipfile.ZipFile(data) as archive, torch.inference_mode():
        seen = set()
        for game in json.loads(archive.read("manifest.json")):
            if (
                game["source"] in seen
                or game["teacher_labels"] is not False
                or game["student_policy_sha256"] != spec["actor_sha256"]
                or game["alternate_policy_sha256"] != spec["alternate_policy_sha256"]
                or game["candidate_probability"] != 0.5
            ):
                raise ValueError(
                    "Unexpected controller assignment in development archive"
                )
            seen.add(game["source"])
            raw = archive.read(game["file"])
            if digest(raw) != game["sha256"]:
                raise ValueError("Development tape digest changed")
            with np.load(io.BytesIO(raw)) as tape:
                clock, state = tape["control_time"], tape["timing_state"]
                if not np.array_equal(tape["block_starts"], neural_block_starts(clock)):
                    raise ValueError(
                        "Runtime block starts differ from the training clock"
                    )
                history = torch.from_numpy(motion_history(clock, state))
                encoded = torch.cat(
                    [model.encode(batch) for batch in history.split(512)]
                ).numpy()
                values = (
                    torch.cat([value(batch) for batch in history.split(512)])
                    .flatten()
                    .numpy()
                )
                blocks = control_blocks(
                    clock,
                    tape["candidate_draw"],
                    state[:, 23] > 0.5,
                    tape["time"],
                    tape["elapsed_ticks"],
                    tape["events"],
                    weights,
                )
                for block in blocks:
                    start, stop = block.start, block.stop
                    if np.diff(clock[start : stop + 1]).max(initial=0) > 0.081:
                        continue
                    adv = (
                        block.reward
                        + block.discount * values[min(stop, len(clock) - 1)]
                        - values[start]
                    )
                    by_split[game["split"]].append(
                        (
                            game["source"],
                            int(block.teacher),
                            adv,
                            block.reward,
                            encoded[start],
                        )
                    )
    result = {}
    for split, rows in by_split.items():
        games, arms, advantages, rewards, features = zip(*rows, strict=True)
        result[split] = Blocks(
            np.array(games),
            np.array(arms),
            np.clip(advantages, -6, 6),
            np.array(rewards),
            np.array(features) @ projection,
        )
    if set(result["train"].games) & set(result["holdout"].games):
        raise ValueError("Development fight appears in both splits")
    return result


def design(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(x)), ((x - center) / scale).clip(-6, 6)))


def paired_screen(
    games: np.ndarray,
    route: np.ndarray,
    arms: np.ndarray,
    returns: np.ndarray,
    estimates: np.ndarray,
) -> list[dict]:
    contrast = dr_contrast(arms, returns, estimates)
    result = []
    for source in sorted(set(games)):
        chosen = games == source
        choose_r = route[chosen].astype(float)
        dy = contrast[chosen]
        result.append(
            {
                "source": source,
                "blocks": int(chosen.sum()),
                "fraction_R": choose_r.mean(),
                "assigned_primary": int((arms[chosen] == 0).sum()),
                "assigned_alternate": int(arms[chosen].sum()),
                "vs_fair": float(np.mean((choose_r - 0.5) * dy)),
                "vs_H": float(np.mean(choose_r * dy)),
                "vs_R": float(np.mean((choose_r - 1) * dy)),
                "raw_ips_vs_fair": float(
                    np.mean((2 * (arms[chosen] == route[chosen]) - 1) * returns[chosen])
                ),
            }
        )
    return result


def eligible(per_game: list[dict]) -> bool:
    gains = ("vs_fair", "vs_H", "vs_R", "raw_ips_vs_fair")
    return bool(
        all(np.mean([row[key] for row in per_game]) > 0 for key in gains)
        and sum(row["vs_fair"] > 0 for row in per_game) > len(per_game) / 2
    )


def train(
    data: Path,
    experiment: Path,
    primary: Path,
    alternate: Path,
    learner: Path,
    output: Path,
    extra_mask_cost: float = 0,
) -> None:
    raw_h, raw_r = primary.read_bytes(), alternate.read_bytes()
    spec = json.loads(experiment.read_bytes())
    if (
        digest(raw_h),
        digest(raw_r),
        spec["candidate_probability"],
        spec["block_game_seconds"],
    ) != (spec["actor_sha256"], spec["alternate_policy_sha256"], 0.5, 0.8):
        raise ValueError(
            "Options or randomization do not match the development experiment"
        )
    opener, h = phase_components(raw_h)
    other_opener, r = phase_components(raw_r)
    if opener != other_opener:
        raise ValueError("Options must share their opening policy")
    projection = np.random.default_rng(137).standard_normal(
        (WIDTH, OPTION_FEATURES)
    ) / np.sqrt(WIDTH)
    weights = REWARD_WEIGHTS.copy()
    weights[1:5] -= extra_mask_cost
    saved = torch.load(learner, weights_only=True, map_location="cpu")
    datasets = block_data(data, saved, projection, weights, spec, h)
    tr, test = datasets["train"], datasets["holdout"]
    for dataset, key in ((tr, "train_seeds"), (test, "heldout_seeds")):
        if {int(x.split("/")[-1]) for x in dataset.games} != set(spec[key]):
            raise ValueError("Unexpected randomized fight split")
    nuisance, held = crossfit_options(
        tr.features, tr.arms, tr.advantages, tr.games, test.features
    )
    labels = dr_contrast(tr.arms, tr.advantages, nuisance).clip(-6, 6)
    center, scale = tr.features.mean(0), tr.features.std(0).clip(0.1)
    X = design(tr.features, center, scale)
    ridge = np.diag(np.r_[0.001, np.full(OPTION_FEATURES, 100.0)])
    coef = np.linalg.solve(X.T @ X + ridge, X.T @ labels)
    route = design(test.features, center, scale) @ coef > 0
    reward_est = predict_arms(tr.features, tr.arms, tr.rewards, test.features)
    scores = {
        "critic_continuation": paired_screen(
            test.games, route, test.arms, test.advantages, held
        ),
        "measured_block_reward": paired_screen(
            test.games, route, test.arms, test.rewards, reward_est
        ),
    }
    params = {
        "projection": projection,
        "mean": center,
        "scale": scale,
        "coefficient": coef,
    }
    blob = struct.pack("<3i", OPTION_MAGIC, len(h), len(r)) + h + r
    blob += b"".join(v.astype("<f4").tobytes() for v in params.values())
    native = phase_checkpoint(opener, blob)
    report = {
        "method": "Whole-training-fight cross-fitted return contrast for a 0.8 second neural option.",
        "scope": "Single-block estimates under the randomized development distribution. Each row is one held fight.",
        "extra_mask_cost": extra_mask_cost,
        "data_sha256": digest(data.read_bytes()),
        "primary_sha256": digest(raw_h),
        "alternate_sha256": digest(raw_r),
        "source_learner_sha256": digest(learner.read_bytes()),
        "policy_sha256": digest(native),
        "train_games": len(set(tr.games)),
        "held_games": len(set(test.games)),
        "train_blocks": len(tr.arms),
        "held_blocks": len(test.arms),
        "results": scores,
        "trial_eligible": all(eligible(rows) for rows in scores.values()),
    }
    output.mkdir(exist_ok=True, parents=True)
    (output / "world_iql.bin").write_bytes(native)
    torch.save(
        {k: torch.from_numpy(v) for k, v in params.items()}, output / "learner.pt"
    )
    (output / "training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: report[k] for k in ("policy_sha256", "held_blocks", "trial_eligible")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("data", "experiment", "primary", "alternate", "learner", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--extra-mask-cost", type=float, default=0)
    args = parser.parse_args()
    if not np.isfinite(args.extra_mask_cost) or args.extra_mask_cost < 0:
        parser.error("extra mask cost must be nonnegative and finite")
    torch.set_num_threads(1)
    train(**vars(args))

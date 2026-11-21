"""Learn the choice between measured compound controls using randomized real fights."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch
from sisters_compound_returns import observed_outcomes
from sisters_maneuver_credit import crossfit_options, dr_contrast, predict_arms
from train_sisters_option_router import paired_screen
from train_sisters_timing_iql import (
    COMPOUND_MAGIC,
    WIDTH,
    TimingPolicy,
    motion_history,
    phase_checkpoint,
    phase_components,
)


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def fit_gate(
    x: np.ndarray,
    arm: np.ndarray,
    outcome: np.ndarray,
    games: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Fit a compact linear head on out-of-game return contrasts and fold its projection."""
    features = x @ projection
    fitted, _ = crossfit_options(features, arm, outcome, games, features[:1])
    difference = dr_contrast(arm, outcome, fitted)
    center = features.mean(0)
    scale = features.std(0).clip(0.1)
    matrix = np.column_stack([np.ones(len(x)), (features - center) / scale])
    reg = np.diag(np.r_[0.001, np.full(features.shape[1], 100.0)])
    beta = np.linalg.solve(matrix.T @ matrix + reg, matrix.T @ difference.clip(-12, 12))
    slope = beta[1:] / scale
    return np.r_[beta[0] - center @ slope, projection @ slope].astype("<f4")


def study(
    data: Path,
    parent: Path,
    learner: Path,
    risk: Path,
    output: Path,
    extra_mask_cost: float = 0.0,
) -> None:
    risk_bytes = risk.read_bytes()
    if (
        len(risk_bytes) != 4 + 4 * (WIDTH + 1)
        or struct.unpack_from("<i", risk_bytes)[0] != 0x314B5352
    ):
        raise ValueError("Invalid risk readout")
    risk_weights = np.frombuffer(risk_bytes, "<f4", offset=4)
    ck = torch.load(learner, weights_only=True, map_location="cpu")
    net = TimingPolicy(ck["catalogue"], history=True).eval()
    net.load_state_dict(ck["actor"])
    native = parent.read_bytes()
    opening, late = phase_components(native)
    samples = {4.0: [], 8.0: []}
    coverage = {}
    with zipfile.ZipFile(data) as z, torch.inference_mode():
        spec = json.loads(z.read("experiment.json"))
        if (
            spec["actor_sha256"] != sha(native)
            or spec["risk_sha256"] != sha(risk_bytes)
            or spec["probability"] != 0.5
            or spec["minimum_spacing_seconds"] < 8.4
        ):
            raise ValueError("Frozen controller or experimental assignment changed")
        manifest = json.loads(z.read("manifest.json"))
        all_ids = [int(g["source"].split("/")[-1]) for g in manifest]
        if sorted(all_ids) != sorted(
            spec["train_seeds"] + spec["heldout_seeds"]
        ) or len(set(all_ids)) != len(all_ids):
            raise ValueError("Unexpected development fight population")
        for g in manifest:
            seed = int(g["source"].split("/")[-1])
            if (
                (g["split"] == "holdout") != (seed in spec["heldout_seeds"])
                or g["teacher_labels"] is not False
                or g["behavior_policy_sha256"] != sha(native)
            ):
                raise ValueError("Unexpected game assignment")
            raw = z.read(g["file"])
            if sha(raw) != g["sha256"]:
                raise ValueError("Unexpected game bytes")
            with np.load(io.BytesIO(raw)) as d:
                clock, state = d["control_time"], d["timing_state"]
                ix, arm, pairs = d["choice_rows"], d["choice_draw"], d["choice_pairs"]
                n = len(ix)
                coverage[g["source"]] = {
                    "assigned": np.bincount(arm, minlength=2).tolist()
                }
                if not n:
                    continue
                h = torch.from_numpy(motion_history(clock, state)[ix])
                hidden = net.encode(h)
                scores = net.heads(hidden)[0].numpy()
                x = hidden.numpy().astype("f8")
                measured_risk = x @ risk_weights[1:] + risk_weights[0]
                if (
                    not np.all(d["choice_probability"] == 0.5)
                    or not np.allclose(
                        measured_risk, d["choice_risk"], atol=0.001, rtol=0.001
                    )
                    or not np.allclose(
                        scores[np.arange(n)[:, None], pairs],
                        d["choice_scores"],
                        atol=0.003,
                        rtol=0.0003,
                    )
                    or (np.diff(clock[ix]) < 8.399).any()
                    or not np.array_equal(
                        d["controls"][ix],
                        ck["catalogue"].numpy()[pairs[np.arange(n), arm]],
                    )
                ):
                    raise ValueError(
                        "Recorded native choices differ from the frozen neural model"
                    )
                for horizon, dest in samples.items():
                    kept, y = observed_outcomes(
                        clock,
                        state,
                        ix,
                        d["time"],
                        d["elapsed_ticks"],
                        d["events"],
                        horizon,
                    )
                    coverage[g["source"]][str(horizon)] = np.bincount(
                        arm[kept], minlength=2
                    ).tolist()
                    for j, yrow in zip(kept, y, strict=True):
                        dest.append((g["source"], g["split"], x[j], int(arm[j]), yrow))

    def arrays(duration: float, split: str) -> tuple[np.ndarray, ...]:
        rows = [(g, x, a, y) for g, kind, x, a, y in samples[duration] if kind == split]
        games, features, arms, returns = zip(*rows, strict=True)
        return np.array(games), np.stack(features), np.array(arms), np.stack(returns)

    groups, x, arms, y = arrays(4.0, "train")
    projection = np.random.default_rng(2781).standard_normal((WIDTH, 12)) / np.sqrt(
        WIDTH
    )
    selector = fit_gate(
        x, arms, y[:, 0] - extra_mask_cost * y[:, 1], groups, projection
    )
    metrics = {}
    for horizon in samples:
        _, xx, aa, yy = arrays(horizon, "train")
        games, hh, ah, yh = arrays(horizon, "holdout")
        decisions = hh @ selector[1:] + selector[0] > 0
        here = {}
        for col, name in enumerate(
            ("observed_reward", "native_masks_lost", "boss_damage", "terminal")
        ):
            scale = max(1, float(yy[:, col].std()))
            nuisance = (
                predict_arms(xx @ projection, aa, yy[:, col] / scale, hh @ projection)
                * scale
            )
            here[name] = paired_screen(games, decisions, ah, yh[:, col], nuisance)
        metrics[str(horizon)] = here
    short = metrics["4.0"]["observed_reward"]
    health = metrics["4.0"]["native_masks_lost"]
    long = metrics["8.0"]["observed_reward"]
    complete_held = {row["source"] for row in short} == {
        f"native-development/{i}" for i in spec["heldout_seeds"]
    }
    eligible = (
        complete_held
        and sum(row["vs_H"] > 0 for row in short) >= 4
        and np.mean([row["vs_H"] for row in short]) > 0
        and np.mean([row["vs_H"] for row in health]) < 0
        and np.mean([row["vs_H"] for row in long]) >= 0
    )
    compound = (
        struct.pack("<2i", COMPOUND_MAGIC, len(late))
        + late
        + risk_bytes[4:]
        + selector.tobytes()
    )
    deployed = phase_checkpoint(opening, compound)
    output.mkdir(parents=True, exist_ok=False)
    (output / "world_iql.bin").write_bytes(deployed)
    torch.save(
        {
            "risk": torch.from_numpy(risk_weights.copy()),
            "selector": torch.from_numpy(selector.copy()),
        },
        output / "learner.pt",
    )
    report = {
        "method": "Whole-fight cross-fitted, doubly robust offline policy improvement with measured 4-second rewards. Both options use the frozen T native action scores and resume its unmodified controls after 0.32s.",
        "parent_sha256": sha(native),
        "learner_input_sha256": sha(learner.read_bytes()),
        "data_sha256": sha(data.read_bytes()),
        "risk_sha256": sha(risk_bytes),
        "policy_sha256": sha(deployed),
        "extra_mask_cost": extra_mask_cost,
        "coverage_by_arm": coverage,
        "heldout": metrics,
        "trial_eligible": bool(eligible),
    }
    (output / "training.json").write_bytes(
        (json.dumps(report, indent=2) + "\n").encode()
    )
    print(
        json.dumps({"trial_eligible": bool(eligible), "policy_sha256": sha(deployed)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "parent", "learner", "risk", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--extra-mask-cost", type=float, default=0.0)
    torch.set_num_threads(1)
    study(**vars(parser.parse_args()))

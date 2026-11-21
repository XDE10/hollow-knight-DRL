"""Package completed native timing demonstrations, splitting by complete development fight."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

CONTROLLER_FIELDS = (
    "student_policy_sha256",
    "teacher_probability",
    "alternate_policy_sha256",
    "candidate_probability",
    "sampler_sha256",
    "sampler_draw_seed",
)


def pack(
    manifests: list[Path],
    output: Path,
    original_speed_only: bool = False,
    block_experiment: Path | None = None,
    compound_experiment: Path | None = None,
) -> None:
    games: dict[int, tuple[Path, dict]] = {}
    seen = set()
    for manifest in manifests:
        for i, game in enumerate(json.loads(manifest.read_bytes())):
            meta = json.loads((manifest.parent / f"episode-{i:03d}.json").read_bytes())
            seed = meta["seed"]
            if (
                seed in seen
                or game["source"] != f"native-development/{seed}"
                or meta["win"] != game["won"]
                or game.get("behavior_policy_sha256")
                != meta.get("behavior_policy_sha256")
                or game.get("teacher_labels", True) != meta.get("teacher_labels", True)
                or any(game.get(k) != meta.get(k) for k in CONTROLLER_FIELDS)
            ):
                raise ValueError("Repeated seed or unmatched development record")
            seen.add(seed)
            if original_speed_only:
                speeds = meta["speed_counts"]
                if float(max(speeds, key=speeds.get)) != 1 or any(
                    float(v) > 1.001 for v in speeds
                ):
                    continue
            games[seed] = (manifest.parent / game["file"], game)
    split_by_seed = {}
    spec = None
    neural_trial = False
    if compound_experiment is not None:
        if block_experiment is not None:
            raise ValueError("Choose one experiment per archive")
        spec = json.loads(compound_experiment.read_bytes())
        train, held = spec["train_seeds"], spec["heldout_seeds"]
        if (not train or not held or len(set(train + held)) != len(train) + len(held)
            or set(train + held) != set(games) or any(
                game.get("behavior_policy_sha256") != spec["actor_sha256"]
                for _, game in games.values())):
            raise ValueError("Unexpected compound trial population")
        split_by_seed = {seed: "train" for seed in train} | {seed: "holdout" for seed in held}
    if block_experiment is not None:
        spec = json.loads(block_experiment.read_bytes())
        train, held = spec["train_seeds"], spec["heldout_seeds"]
        neural_trial = "alternate_policy_sha256" in spec
        expected_fields = {"student_policy_sha256": spec["actor_sha256"]}
        if neural_trial:
            expected_fields.update(
                alternate_policy_sha256=spec["alternate_policy_sha256"],
                candidate_probability=0.5,
            )
        else:
            expected_fields.update(teacher_probability=0.5)
        if (
            not train
            or not held
            or len(set(train + held)) != len(train) + len(held)
            or set(train + held) != set(games)
            or spec.get(
                "candidate_probability" if neural_trial else "teacher_probability"
            )
            != 0.5
            or not original_speed_only
        ):
            raise ValueError(
                "The complete registered original-speed block experiment is required"
            )
        split_by_seed = {seed: "train" for seed in train} | {
            seed: "holdout" for seed in held
        }
        if any(
            any(g.get(k) != v for k, v in expected_fields.items())
            for _, g in games.values()
        ):
            raise ValueError(
                "Recorded actor or controller probability differs from the experiment"
            )
    entries = []
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial")
    with zipfile.ZipFile(partial, "x") as archive:
        for seed, (path, game) in sorted(games.items()):
            with np.load(path, allow_pickle=False) as raw:
                features = raw["timing_state"]
                t = raw["control_time"]
                if (
                    features.shape != (len(t), 192)
                    or not np.isfinite(features).all()
                    or not np.all(np.diff(t) > 0)
                ):
                    raise ValueError(f"Invalid decision trace in {path}")
                if not any(features[:, 24]):
                    raise ValueError(f"No observed Sisters in {path}")
                plan_keys: tuple[str, ...] = ()
                if "plan_costs" in raw or "plan_choice" in raw:
                    plan_keys = ("plan_costs", "plan_choice")
                    if (
                        not all(key in raw for key in plan_keys)
                        or raw["plan_costs"].shape != (len(t), 12)
                        or raw["plan_choice"].shape != (len(t),)
                    ):
                        raise ValueError(f"Incomplete movement alternatives in {path}")
                buf = io.BytesIO()
                online_keys: tuple[str, ...] = ()
                if game.get("behavior_policy_sha256"):
                    from sisters_online_choices import CHOICE_KEYS

                    online_keys = CHOICE_KEYS
                    if "choice_hold_stop" in raw:
                        stops = raw["choice_hold_stop"]
                        if (
                            stops.shape != raw["choice_rows"].shape
                            or stops.dtype.kind not in "iu"
                            or (stops <= raw["choice_rows"]).any()
                            or (stops > len(t)).any()
                        ):
                            raise ValueError(f"Invalid duration of randomized movement in {path}")
                        online_keys += ("choice_hold_stop",)
                    if "choice_scores" in raw or "choice_risk" in raw:
                        if ("choice_hold_stop" not in raw or "choice_scores" not in raw
                            or "choice_risk" not in raw or raw["choice_risk"].shape != raw["choice_rows"].shape
                            or raw["choice_scores"].shape != (len(raw["choice_rows"]), 2)):
                            raise ValueError(f"Incomplete compound option in {path}")
                        online_keys += ("choice_scores","choice_risk")
                    if compound_experiment is not None and "choice_risk" not in raw:
                        raise ValueError("Missing compound risk record")
                    if not all(k in raw for k in online_keys):
                        raise ValueError(f"Incomplete online behavior record in {path}")
                labeled = game.get("teacher_labels", True)
                trial_keys = ()
                if game.get("alternate_policy_sha256"):
                    from sisters_control_blocks import neural_block_starts

                    trial_keys = ("block_starts", "candidate_draw")
                    if (
                        labeled
                        or online_keys
                        or game.get("candidate_probability") != 0.5
                        or game["alternate_policy_sha256"]
                        == game["student_policy_sha256"]
                        or not all(k in raw for k in trial_keys)
                    ):
                        raise ValueError(f"Invalid neural block trial in {path}")
                    assigned = raw["candidate_draw"]
                    starts = neural_block_starts(t)
                    if (
                        not np.array_equal(starts, raw["block_starts"])
                        or assigned.shape != t.shape
                        or assigned.dtype != np.bool_
                        or not np.array_equal(
                            assigned,
                            np.repeat(assigned[starts], np.diff(np.r_[starts, len(t)])),
                        )
                    ):
                        raise ValueError(f"Inconsistent neural draws in {path}")
                elif "block_starts" in raw or "candidate_draw" in raw:
                    raise ValueError(f"Undeclared neural block trial in {path}")
                if ("teacher_controls" in raw) != labeled or (
                    not labeled and (not (online_keys or trial_keys) or plan_keys)
                ):
                    raise ValueError(f"Inconsistent teacher availability in {path}")
                assignment_keys = ()
                if "student_controls" in raw or "teacher_acted" in raw:
                    assignment_keys = ("student_controls", "teacher_acted")
                    if not labeled or not all(k in raw for k in assignment_keys):
                        raise ValueError(f"Incomplete controller assignment in {path}")
                    assigned = raw["teacher_acted"]
                    student = raw["student_controls"]
                    teacher = raw["teacher_controls"]
                    if (
                        assigned.dtype != np.bool_
                        or assigned.shape != (len(t),)
                        or student.shape != (len(t), 3)
                        or teacher.shape != student.shape
                        or not np.array_equal(
                            raw["controls"],
                            np.where(assigned[:, None], teacher, student),
                        )
                    ):
                        raise ValueError(
                            f"Controller assignment differs from applied keys in {path}"
                        )
                if block_experiment is not None and not (
                    trial_keys if neural_trial else assignment_keys
                ):
                    raise ValueError(f"No recorded controller assignments in {path}")
                np.savez_compressed(
                    buf,
                    **{
                        k: raw[k]
                        for k in (
                            "control_time",
                            "timing_state",
                            "controls",
                            *(("teacher_controls",) if labeled else ()),
                            *assignment_keys,
                            "time",
                            "events",
                            "elapsed_ticks",
                            *plan_keys,
                            *online_keys,
                            *trial_keys,
                        )
                    },
                )
                data = buf.getvalue()
            name = f"fight-{seed}.npz"
            archive.writestr(name, data)
            entries.append(
                {
                    "source": game["source"],
                    "file": name,
                    "split": split_by_seed.get(
                        seed, "holdout" if (seed % 1000) % 6 == 1 else "train"
                    ),
                    "won": game["won"],
                    **({"teacher_labels": False} if not labeled else {}),
                    **{k: game[k] for k in CONTROLLER_FIELDS if k in game},
                    **(
                        {"behavior_policy_sha256": game["behavior_policy_sha256"]}
                        if online_keys
                        else {}
                    ),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        if {g["split"] for g in entries} != {"train", "holdout"}:
            raise ValueError("Missing fight split")
        archive.writestr("manifest.json", json.dumps(entries, indent=2) + "\n")
        if spec is not None:
            archive.writestr("experiment.json", json.dumps(spec, indent=2) + "\n")
    partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original-speed-only", action="store_true")
    parser.add_argument(
        "--block-experiment",
        type=Path,
        help="Preassigned policy hashes and complete fight splits.",
    )
    parser.add_argument("--compound-experiment", type=Path)
    args = parser.parse_args()
    pack(args.manifests, args.output, args.original_speed_only, args.block_experiment, args.compound_experiment)

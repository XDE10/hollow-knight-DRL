"""Package completed native timing demonstrations, splitting by complete development fight."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def pack(
    manifests: list[Path], output: Path, original_speed_only: bool = False
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
                or game.get("behavior_policy_sha256") != meta.get("behavior_policy_sha256")
                or game.get("teacher_labels", True) != meta.get("teacher_labels", True)
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
                    if not all(k in raw for k in online_keys):
                        raise ValueError(f"Incomplete online behavior record in {path}")
                labeled = game.get("teacher_labels", True)
                if ("teacher_controls" in raw) != labeled or (
                    not labeled and (not online_keys or plan_keys)
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
                        or not np.array_equal(raw["controls"], np.where(assigned[:, None], teacher, student))
                    ):
                        raise ValueError(f"Controller assignment differs from applied keys in {path}")
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
                    "split": "holdout" if (seed % 1000) % 6 == 1 else "train",
                    "won": game["won"],
                    **({"teacher_labels": False} if not labeled else {}),
                    **({"behavior_policy_sha256": game["behavior_policy_sha256"]} if online_keys else {}),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        if {g["split"] for g in entries} != {"train", "holdout"}:
            raise ValueError("Missing fight split")
        archive.writestr("manifest.json", json.dumps(entries, indent=2) + "\n")
    partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original-speed-only", action="store_true")
    args = parser.parse_args()
    pack(args.manifests, args.output, args.original_speed_only)

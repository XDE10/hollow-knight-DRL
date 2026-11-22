"""Append registered original-speed pure-teacher fights to an audited replay archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from audit_sisters_mainline_replay import audit, read_episode
from train_hkrl_wm_recurrent_rl import sha256


def verified_native_commits(timing: np.ndarray, controls: np.ndarray) -> int:
    """Verify each submitted key from the following timing observation."""
    buttons = timing[1:, 164:173] @ (1 << np.arange(9))
    previous = np.column_stack((timing[1:, 160:162], buttons))
    if not np.array_equal(previous, controls[:-1]):
        raise ValueError("The next native observation disagrees with the submitted keys")
    return len(previous)


def teacher_fight(
    root: Path, row: dict[str, Any], plan: dict[str, Any], plan_sha: str
) -> tuple[dict, bytes]:
    seed = row["seed"]
    port = row["port"]
    slot = row["slot"]
    log = root / "receipts" / str(seed)
    worker = json.loads((log / "worker.json").read_bytes())
    result = json.loads((log / "result.json").read_bytes())
    receipt_bytes = (log / "game/episode-000.json").read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    receipt = json.loads(receipt_bytes)
    tape = (log / "game/dev-000.npz").read_bytes()
    tape_sha = hashlib.sha256(tape).hexdigest()
    save = root / f"teacher-{port}-{seed}-original-saves"
    assets = plan["assets"]
    actual_teacher = {
        name: value for name, value in assets.items() if name.startswith("teacher.txt")
    }
    speeds = {float(k): v for k, v in receipt["speed_counts"].items()}
    if (
        (worker["seed"], worker["port"], worker["slot"]) != (seed, port, slot)
        or (result["seed"], result["port"], result["slot"], result["split"])
        != (seed, port, slot, row["split"])
        or (save / "manifest.json").read_bytes() != (save / "restored.json").read_bytes()
        or receipt["seed"] != seed
        or receipt["unity_reset_seed"] != seed
        or receipt["game_speed"] != 1
        or max(speeds, key=speeds.__getitem__) != 1
        or max(speeds) > 1.001
        or receipt["teacher_fraction"] != 1
        or receipt["teacher_probability"] != 1
        or receipt["student_policy_sha256"] != assets["recorder.bin"]
        or receipt["mod_sha256"] != assets["HKRLEnvMod.dll"]
        or receipt.get("teacher_asset_sha256", actual_teacher) != actual_teacher
        or result["win"] != receipt["win"]
        or result["truncated"] != receipt["truncated"]
        or tape_sha != result["tape_sha256"]
        or tape_sha != receipt["tape_sha256"]
        or receipt_sha != result["receipt_sha256"]
    ):
        raise ValueError(f"Original-speed pure teacher evidence mismatch for {seed}")
    member = {
        "source": f"native-development/{seed}",
        "file": f"teacher-{seed}.npz",
        "split": row["split"],
        "won": receipt["win"],
        "truncated": receipt["truncated"],
        "death_inferred_from_hp": receipt.get("death_inferred_from_hp"),
        "sha256": tape_sha,
        "unity_reset_seed": seed,
        "teacher_asset_sha256": actual_teacher,
        "teacher_hash_source": "game_process_receipt"
        if "teacher_asset_sha256" in receipt
        else "frozen_launcher_assets",
        "policy_build_sha256": receipt["mod_sha256"],
        "receipt_sha256": receipt_sha,
        "collection_plan_sha256": plan_sha,
        "collection_port": port,
        "save_slot": slot,
    }
    with np.load(io.BytesIO(tape), allow_pickle=False) as recorded:
        episode = read_episode(recorded, member)
        if not (episode.controller == "teacher").all():
            raise ValueError(f"A non-teacher action was taken in {seed}")
        member["verified_native_commits"] = verified_native_commits(
            recorded["timing_state"],
            recorded["controls"],
        )
        sums = episode.events.sum(axis=0)
        if not np.isclose(sums[0], receipt["damage"]) or not np.isclose(
            sums[1:5].sum(), sum(receipt["damage_taken"].values())
        ):
            raise ValueError(f"Reward event totals differ in {seed}")
    return member, tape


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "base-audit", "plan", "output", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--acceptance-plan", type=Path, nargs="+", required=True)
    args = p.parse_args()
    base_report = json.loads(args.base_audit.read_bytes())
    if sha256(args.base) != base_report["archive_sha256"]:
        raise ValueError("Historical archive has changed")
    plan = json.loads(args.plan.read_bytes())
    if (
        not plan["development_only"]
        or not plan["teacher_acts_all_frames"]
        or (
            plan["game_speed"],
            plan["nail_multiplier"],
            plan["initial_masks"],
            plan["initial_soul"],
        )
        != (1, 1, 9, 0)
    ):
        raise ValueError("Expected a locked pure teacher development plan")
    assignments = {row["seed"]: row for row in plan["assignments"]}
    seeds = args.seeds
    if (
        len(assignments) != len(plan["assignments"])
        or len(set(seeds)) != len(seeds)
        or not set(seeds) <= assignments.keys()
    ):
        raise ValueError("Repeated or unassigned teacher seed")
    plan_sha = sha256(args.plan)
    for name, expected in plan["assets"].items():
        if Path(name).name != name or sha256(args.plan.parent / "frozen" / name) != expected:
            raise ValueError(f"Frozen teacher input changed: {name}")
    additions = [
        teacher_fight(args.plan.parent, assignments[seed], plan, plan_sha) for seed in seeds
    ]
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Keep existing locked corpus artifacts intact")
    partial = args.output.with_suffix(".partial")
    with ZipFile(args.base) as original, ZipFile(partial, "x") as target:
        entries = json.loads(original.read("manifest.json"))
        if {m["source"] for m in entries} & {m["source"] for m, _ in additions}:
            raise ValueError("Teacher cohort reuses a historical replay seed")
        for member in entries:
            target.writestr(member["file"], original.read(member["file"]))
        for member, raw in additions:
            target.writestr(member["file"], raw)
        target.writestr("manifest.json", json.dumps(entries + [m for m, _ in additions], indent=2))
        target.writestr("teacher-plan.json", args.plan.read_bytes())
    report = audit(partial, sha256(partial), args.acceptance_plan)
    new_sources = {m["source"] for m, _ in additions}
    if any(
        "excluded" in entry
        for entry in report["episodes"]
        if entry["provenance"]["source"] in new_sources
    ):
        raise ValueError("New teacher fight was excluded by the independent replay audit")
    partial.replace(args.output)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "new_fights": len(additions),
                "new_wins": sum(m["won"] for m, _ in additions),
                "splits": report["splits"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

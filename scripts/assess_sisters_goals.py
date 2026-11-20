"""Freeze and score the two successive teacher-free Sisters acceptance cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPEC = (
    Path(__file__).resolve().parents[1] / "configs/experiments/sisters_acceptance.json"
)


def goal(number: int) -> dict[str, Any]:
    return next(g for g in json.loads(SPEC.read_bytes())["goals"] if g["id"] == number)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_range(plan: dict) -> range:
    start = int(plan["first_seed"])
    return range(start, start + goal(plan["goal"])["episodes"])


def score(plan: dict, games: list[dict]) -> dict:
    rule = goal(plan["goal"])
    assigned = seed_range(plan)
    seen: set[int] = set()
    wins = flawless = 0
    losses: Counter[str] = Counter()
    for g in games:
        seed = g["seed"]
        if seed not in assigned or seed in seen:
            raise ValueError(f"Repeated or unassigned game {seed}")
        seen.add(seed)
        if (
            g["policy_sha256"] != plan["policy_sha256"]
            or g["mod_sha256"] != plan["mod_sha256"]
        ):
            raise ValueError(f"Different weights or game build in game {seed}")
        speeds = {float(k): n for k, n in g["speed_counts"].items()}
        if (
            max(speeds, key=speeds.get) != 1
            or max(speeds) > 1.001
            or g["decisions"] < 1
        ):
            raise ValueError(f"Invalid speed or empty actor receipt in game {seed}")
        if "teacher_fraction" in g or (g["win"] and (g["truncated"] or g["hp"] <= 0)):
            raise ValueError(f"Unexpected controller or terminal state in game {seed}")
        losses.update(g["damage_taken"])
        clean_win = bool(
            g["win"] and g["minimum_hp"] == 9 and sum(g["damage_taken"].values()) == 0
        )
        if clean_win != g["no_hit_win"]:
            raise ValueError(f"Inconsistent damage evidence in game {seed}")
        wins += int(g["win"])
        flawless += int(clean_win)
    complete = len(seen) == len(assigned)
    passed = (
        complete
        and wins >= rule["minimum_wins"]
        and flawless >= rule["minimum_no_hit_wins"]
    )
    return {
        "finished": len(seen),
        "wins": wins,
        "no_hit_wins": flawless,
        "damage_taken": dict(losses),
        "complete": complete,
        "passed": passed,
        "next_goal": rule["next_goal"] if passed else None,
    }


def freeze(
    number: int,
    checkpoint: Path,
    game_mod: Path,
    first_seed: int,
    previous: Path | None,
) -> dict:
    plan = {
        "goal": number,
        "first_seed": first_seed,
        "policy_sha256": digest(checkpoint),
        "mod_sha256": digest(game_mod),
    }
    if number == 2:
        if previous is None:
            raise ValueError("Goal 2 opens after a passing goal 1 report")
        first = json.loads(previous.read_bytes())
        prior_plan = first["plan"]
        if (
            prior_plan["goal"] != 1
            or not score(prior_plan, first["episodes"])["passed"]
        ):
            raise ValueError("The first goal has not passed")
        if set(seed_range(prior_plan)).intersection(seed_range(plan)):
            raise ValueError("Use a new cohort for the no-hit goal")
        plan["goal_one_report_sha256"] = digest(previous)
        plan["goal_one_policy_sha256"] = prior_plan["policy_sha256"]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    create = sub.add_parser("freeze")
    create.add_argument("--goal", required=True, type=int, choices=(1, 2))
    create.add_argument("--checkpoint", required=True, type=Path)
    create.add_argument("--game-mod", required=True, type=Path)
    create.add_argument("--first-seed", required=True, type=int)
    create.add_argument("--previous-report", type=Path)
    grade = sub.add_parser("score")
    grade.add_argument("--plan", required=True, type=Path)
    grade.add_argument("--directories", nargs="+", required=True, type=Path)
    for cmd in (create, grade):
        cmd.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "freeze":
        out = freeze(
            args.goal,
            args.checkpoint,
            args.game_mod,
            args.first_seed,
            args.previous_report,
        )
    else:
        plan = json.loads(args.plan.read_bytes())
        episodes = [
            json.loads(p.read_bytes())
            for d in args.directories
            for p in d.glob("episode-*.json")
        ]
        out = dict(
            plan=plan,
            **score(plan, episodes),
            episodes=sorted(episodes, key=lambda x: x["seed"]),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

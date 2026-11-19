#!/usr/bin/env python3
"""Score a complete ten-fight Sisters evaluation and its interleaved replay."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def score(report: dict[str, Any], rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    meta = report["metadata"]
    task = "sisters_of_battle"
    count = 10
    seeds = meta["seeds"]
    if meta["episodes"] != count or meta["task_ids"] != [task] or not seeds:
        raise ValueError(
            "expected a ten-episode Sisters evaluation with recorded seeds"
        )
    episodes: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        i = row["episode"]
        if (
            row["task_id"] != task
            or i not in range(count)
            or row["seed"] != seeds[i % len(seeds)]
        ):
            raise ValueError("replay task, episode index or seed disagrees with report")
        group = episodes.setdefault(i, [])
        if group and (
            group[-1]["terminated"]
            or group[-1]["truncated"]
            or group[0]["episode_id"] != row["episode_id"]
            or group[0].get("eval_worker") != row.get("eval_worker")
        ):
            raise ValueError(
                f"episode {i} changed identity or continued after termination"
            )
        if row["step"] != len(group) + 1 or row["elapsed_ticks"] <= 0:
            raise ValueError(f"missing, duplicate or invalid step in episode {i}")
        if row["won"] and (not row["terminated"] or row["truncated"]):
            raise ValueError(f"win without encounter completion in episode {i}")
        contact_hp = sum(event["amount"] for event in row["damage_contacts"])
        if not math.isclose(contact_hp, row["damage_taken"]):
            raise ValueError(f"damage contacts disagree with HP loss in episode {i}")
        group.append(row)
    if len(episodes) != count:
        raise ValueError(f"expected {count} episodes, found {len(episodes)}")

    totals: Counter[str] = Counter()
    movements: Counter[int] = Counter()
    fights = []
    for i in range(count):
        group = episodes[i]
        end = group[-1]
        if not (end["terminated"] or end["truncated"]):
            raise ValueError(f"episode {i} is incomplete")
        contacts: Counter[str] = Counter()
        directions: set[int] = set()
        for row in group:
            for event in row["damage_contacts"]:
                contacts[event["source"]] += event["amount"]
            action = row["action"]
            if action.get("macro", 0) == 0:
                direction = action["movement_x"] - 1
                movements[direction] += 1
                directions.add(direction)
        totals.update(contacts)
        fights.append(
            {
                "episode": i,
                "seed": seeds[i % len(seeds)],
                "contacts": dict(contacts),
                "damage": sum(row["damage_dealt"] for row in group),
                "ticks": sum(row["elapsed_ticks"] for row in group),
                "win": end["won"],
                "truncated": end["truncated"],
                "both_directions": {-1, 1} <= directions,
                "max_bosses_seen": max(row["observed_boss_count"] for row in group),
            }
        )
    ticks = sum(f["ticks"] for f in fights)
    damage = sum(f["damage"] for f in fights) / count
    wins = sum(f["win"] for f in fights)
    metrics = report["metrics"][task]
    replay_means = {
        "damage_dealt": damage,
        "damage_taken": sum(totals.values()) / count,
        "survival_ticks": ticks / count,
        "win_rate": wins / count,
    }
    if any(
        not math.isclose(value, metrics[key], abs_tol=1e-6)
        for key, value in replay_means.items()
    ):
        raise ValueError("replay and evaluator summaries disagree")
    damaging = sum(f["damage"] > 0 for f in fights)
    combat = 1000 * (totals["boss_body"] + totals["enemy_projectile"]) / ticks
    spikes = 1000 * totals["spikes"] / ticks
    opening_skills_passed = damaging >= 8 and damage >= 60 and combat <= 7 and spikes <= 0.8
    multi_boss_fights = sum(f["max_bosses_seen"] >= 2 for f in fights)
    return {
        "checkpoint": meta.get("checkpoint"),
        "model": meta.get("model"),
        "seeds": seeds,
        "wins": wins,
        "damaging": damaging,
        "damage": damage,
        "ticks": ticks / count,
        "survival_game_seconds": metrics.get("survival_game_seconds"),
        "contacts": dict(totals),
        "body_blade_per_1000_ticks": combat,
        "spikes_per_1000_ticks": spikes,
        "spike_fights": sum(f["contacts"].get("spikes", 0) > 0 for f in fights),
        "primitive_movement": {
            label: movements[k]
            for k, label in enumerate(("left", "neutral", "right"), -1)
        },
        "both_directions_fights": sum(f["both_directions"] for f in fights),
        "multi_boss_fights": multi_boss_fights,
        "opening_skills_passed": opening_skills_passed,
        "continue_training": bool(wins or (multi_boss_fights and opening_skills_passed)),
        "episodes": fights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", type=Path, help="JSON metrics produced by scripts/run_eval.py"
    )
    parser.add_argument("replay", type=Path, help="corresponding --replay-jsonl file")
    parser.add_argument("--output", type=Path, help="write the score JSON to this path")
    args = parser.parse_args()
    with args.replay.open(encoding="utf-8") as replay:
        result = score(
            json.loads(args.report.read_text(encoding="utf-8")),
            (json.loads(line) for line in replay if line.strip()),
        )
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

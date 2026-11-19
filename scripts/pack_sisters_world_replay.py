"""Pack full entity observations and the actual native inputs into per-episode arrays."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

EVENTS = (
    "dealt",
    "body",
    "projectile",
    "spikes",
    "other_hurt",
    "heal",
    "arena_win",
    "death",
)


def pack_game(report: Path, target: Path) -> dict[str, object]:
    meta = json.loads(report.read_text(encoding="utf-8"))
    tape = report.with_suffix(".trace")
    with tape.open("rb") as stream:
        magic, episode, width = struct.unpack("<IQi", stream.read(16))
        if (magic, episode, width) != (0x31534B48, meta["episode_id"], 56):
            raise ValueError(f"Action identity differs from the game in {report}")
        actions = np.frombuffer(stream.read(), dtype="<f4").reshape(-1, 60)
    times, players, entities, events, ticks = [], [], [], [], []
    pointers = [0]
    with gzip.open(report.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as stream:
        for line in stream:
            start, player, objects, _, reward_events, elapsed = json.loads(line)
            row = np.zeros(len(EVENTS), np.float32)
            terminal_win = any(
                k == "BOSS_KILLED" and e == 0 for k, e, _, _ in reward_events
            )
            for kind, eid, amount, source in reward_events:
                if kind == "DAMAGE_DEALT":
                    row[0] += amount
                elif kind == "DAMAGE_TAKEN":
                    row[source if source in (1, 2, 3) else 4] += amount
                elif kind == "HEAL" and not terminal_win:
                    row[5] += amount
                elif kind == "BOSS_KILLED" and eid == 0:
                    row[6] = 1
                elif kind == "PLAYER_DEATH":
                    row[7] = 1
            times.append(start)
            players.append(player)
            entities.extend(objects)
            pointers.append(len(entities))
            events.append(row)
            ticks.append(elapsed)
    t = np.asarray(times, dtype=np.float32)
    if not meta["truncated"] and not meta["won"] and meta["hp"] == 0:
        events[-1][7] = 1
    if (
        len(t) < 4
        or len(actions) < 4
        or np.any(np.diff(t) <= 0)
        or np.any(np.diff(actions[:, 0]) <= 0)
    ):
        raise ValueError(f"Discontinuous episode in {report}")
    ent = np.asarray(entities, dtype=np.float32).reshape(-1, 24)
    lens = np.diff(pointers)
    np.savez_compressed(
        target,
        time=t,
        player=np.asarray(players, dtype=np.float32),
        entities=ent,
        entity_ptr=np.asarray(pointers, dtype=np.int32),
        events=np.asarray(events),
        elapsed_ticks=np.asarray(ticks, dtype=np.int16),
        control_time=actions[:, 0],
        student_features=actions[:, 1:57],
        controls=actions[:, 57:].astype(np.int16),
    )
    return {
        "rows": len(t),
        "native_decisions": len(actions),
        "won": meta["won"],
        "phase_two": meta["max_boss"] >= 3,
        "truncated": bool(meta["truncated"]),
        "max_entities": int(lens.max()),
        "episode_id": meta["episode_id"],
        "scale": meta["scale_distribution"][0][0],
        "record_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }


def split_games(manifest: list[dict], registry: Path) -> None:
    assignments = json.loads(registry.read_text()) if registry.exists() else {}
    if not assignments:
        groups: dict[tuple[bool, float], list[dict]] = defaultdict(list)
        for game in manifest:
            groups[game["won"], game["scale"]].append(game)
        for group in groups.values():
            ranked = sorted(group, key=lambda game: game["file"])
            for idx, game in enumerate(ranked):
                assignments[game["source"]] = "holdout" if idx % 5 == 0 else "train"
    for game in manifest:
        key = game["source"]
        if key not in assignments:
            assignments[key] = (
                "holdout" if int(game["file"][:8], 16) % 5 == 0 else "train"
            )
        game["split"] = assignments[key]
    registry.write_text(json.dumps(assignments, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--populations", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    seen = set()
    for name in args.populations:
        population = args.root / name
        freeze = population / "frozen.json"
        if (
            freeze.exists()
            and json.loads(freeze.read_text()).get("kind", "official") != "development"
        ):
            raise ValueError(f"Official evaluation games are reserved: {population}")
        markers = sorted(population.glob("actor-*/c*/verified-*.json"))
        markers.extend(sorted(population.glob("actor-*/validated-*.json")))
        for verified in markers:
            report = verified.with_name("fight-" + verified.name.split("-", 1)[1])
            if not report.with_suffix(".trace").is_file():
                continue
            record = json.loads(report.read_text())
            verified_record = json.loads(verified.read_text())
            if any(
                record[k] != verified_record[k]
                for k in ("seed", "episode_id", "won", "ticks")
            ):
                raise ValueError(f"Verified episode differs from replay: {report}")
            path = report.relative_to(args.root).as_posix()
            if path in seen:
                continue
            seen.add(path)
            key = hashlib.sha256(path.encode()).hexdigest()
            output = args.output / f"{key[:16]}.npz"
            entry = pack_game(report, output)
            manifest.append({"file": output.name, "source": path, **entry})
    if not manifest:
        parser.error(
            "No completed games with matching native action traces were found."
        )
    split_games(manifest, args.root / "world-splits-v1.json")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "episodes": len(manifest),
                "snapshots": sum(int(x["rows"]) for x in manifest),
                "wins": sum(bool(x["won"]) for x in manifest),
                "holdout": sum(x["split"] == "holdout" for x in manifest),
            }
        )
    )


if __name__ == "__main__":
    main()

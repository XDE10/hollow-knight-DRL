"""Audit and read locked, whole-fight Sisters trajectories for recurrent training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from assess_sisters_goals import seed_range
from train_sisters_native_iql import REWARD_WEIGHTS
from train_sisters_option_ppo import verified_events

MIN_INTERVAL = np.float32(0.025)
MAX_INTERVAL = np.float32(0.081)
INTERVENTION_FIELDS = {
    "candidate_draw",
    "choice_rows",
    "choice_hold_stop",
    "pair_rows",
    "plan_costs",
    "plan_choice",
}


class UnusableFight(ValueError):
    """A legacy fight lacks evidence needed to train on its transitions."""


@dataclass(frozen=True)
class Episode:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    events: np.ndarray
    clock: np.ndarray
    next_row: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    controller: np.ndarray

    @property
    def valid_transition(self) -> np.ndarray:
        return (self.next_row >= 0) | self.terminated


def read_episode(tape: Mapping[str, np.ndarray], item: Mapping[str, Any]) -> Episode:
    clock, observation, controls = (
        tape[k] for k in ("control_time", "timing_state", "controls")
    )
    starts, ticks, observed = (tape[k] for k in ("time", "elapsed_ticks", "events"))
    n = len(clock)
    if (
        clock.shape != (n,)
        or n < 2
        or observation.shape != (n, 192)
        or controls.shape != (n, 3)
        or not np.issubdtype(controls.dtype, np.integer)
        or starts.ndim != 1
        or not len(starts)
        or ticks.shape != starts.shape
        or observed.shape != (len(starts), 8)
        or not all(
            np.isfinite(a).all() for a in (clock, observation, starts, ticks, observed)
        )
        or (np.diff(clock) <= 0).any()
        or (np.diff(starts) <= 0).any()
        or (ticks <= 0).any()
        or (observed < 0).any()
    ):
        raise UnusableFight("malformed native or reward observations")
    if not np.allclose(observation[0, 4:6], (1.0, 0.0), rtol=0, atol=1e-5):
        raise UnusableFight("missing nine-mask zero-SOUL start")
    if INTERVENTION_FIELDS.intersection(tape):
        raise UnusableFight("recorded randomized or planner intervention")

    controller = np.full(n, "unknown", dtype="U7")
    if "teacher_acted" in tape or "student_controls" in tape:
        if not {"teacher_acted", "teacher_controls", "student_controls"}.issubset(tape):
            raise ValueError("Incomplete controller assignment")
        acted, teacher, student = (
            tape[k] for k in ("teacher_acted", "teacher_controls", "student_controls")
        )
        if (
            acted.dtype != np.bool_
            or acted.shape != (n,)
            or teacher.shape != controls.shape
            or student.shape != controls.shape
            or not np.array_equal(controls, np.where(acted[:, None], teacher, student))
        ):
            raise ValueError("Controller assignment disagrees with logged native keys")
        controller = np.where(acted, "teacher", "student")
    elif (
        item.get("deterministic_native_trace")
        and item.get("behavior_policy_sha256")
        and "teacher_controls" not in tape
    ):
        controller[:] = "student"

    try:
        events = verified_events(clock, observation[:, 4] * 9, starts, ticks, observed)
    except ValueError as error:
        raise UnusableFight(str(error)) from error
    terminal_rows = np.flatnonzero(events[:, 6:].any(axis=1))
    truncated = item.get("truncated", False)
    if type(item["won"]) is not bool or type(truncated) is not bool:
        raise ValueError("Expected boolean outcome flags")
    if len(terminal_rows) == 0 and not truncated:
        raise UnusableFight(
            "legacy recording has no observed terminal or explicit truncation"
        )
    if len(terminal_rows) != int(not truncated):
        raise ValueError("Conflicting terminal and truncation evidence")
    if truncated:
        if item["won"]:
            raise ValueError("A truncated recording cannot claim a win")
        end = n
    else:
        end = int(terminal_rows[0]) + 1
        expected = [int(item["won"]), int(not item["won"])]
        if events[end - 1, 6:].tolist() != expected or events[end:].any():
            raise ValueError("Recorded outcome disagrees with terminal events")

    # An interval with missing decisions cannot provide a one-step dynamics or TD target.
    delta = np.diff(clock[:end])
    valid = (delta >= MIN_INTERVAL - 1e-5) & (delta <= MAX_INTERVAL + 1e-5)
    next_row = np.full(end, -1, dtype=np.int64)
    next_row[:-1] = np.where(valid, np.arange(1, end), -1)
    terminated = np.zeros(end, dtype=np.bool_)
    truncated_rows = np.zeros(end, dtype=np.bool_)
    terminated[-1] = not truncated
    truncated_rows[-1] = truncated
    return Episode(
        observation[:end],
        controls[:end],
        events[:end] @ REWARD_WEIGHTS,
        events[:end],
        clock[:end],
        next_row,
        terminated,
        truncated_rows,
        controller[:end],
    )


def iter_episodes(
    archive: ZipFile,
    reserved: set[int],
) -> Iterator[tuple[dict[str, Any], Episode | str]]:
    items = json.loads(archive.read("manifest.json"))
    seen_seeds: set[int] = set()
    seen_members: set[str] = set()
    for item in items:
        cohort, separator, seed_text = item["source"].partition("/")
        if (
            cohort != "native-development"
            or not separator
            or not seed_text.isascii()
            or not seed_text.isdecimal()
        ):
            raise ValueError(
                f"Expected an identified development fight, got {item['source']}"
            )
        seed = int(seed_text)
        if (
            seed in seen_seeds
            or item["file"] in seen_members
            or item["split"] not in {"train", "holdout"}
        ):
            raise ValueError(
                f"Repeated seed, repeated archive member, or invalid split for {item['source']}"
            )
        seen_seeds.add(seed)
        seen_members.add(item["file"])
        raw = archive.read(item["file"])
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Archive member changed for {item['source']}")
        if seed in reserved:
            yield item, "reserved acceptance seed"
            continue
        with np.load(io.BytesIO(raw), allow_pickle=False) as tape:
            try:
                yield item, read_episode(tape, item)
            except UnusableFight as error:
                yield item, str(error)


def audit(
    archive_path: Path, expected_sha256: str, acceptance_plans: Sequence[Path]
) -> dict[str, Any]:
    reserved: set[int] = set()
    plan_hashes = {}
    for path in acceptance_plans:
        raw = path.read_bytes()
        plan = json.loads(raw)
        plan_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        reserved.update(seed_range(plan.get("provenance", plan)))

    groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
    excluded: Counter[str] = Counter()
    entries = []
    with archive_path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        sha256 = checksum.hexdigest()
        if sha256 != expected_sha256.lower():
            raise ValueError("Replay archive SHA-256 disagrees with the locked corpus")
        stream.seek(0)
        with ZipFile(stream) as archive:
            for item, episode in iter_episodes(archive, reserved):
                entry = {"provenance": item}
                if isinstance(episode, str):
                    excluded[episode] += 1
                    entry["excluded"] = episode
                else:
                    valid = episode.valid_transition
                    counts = Counter(
                        {
                            "episodes": 1,
                            "wins": int(item["won"]),
                            "decisions": len(episode.clock),
                            "valid_transitions": int(valid.sum()),
                            "missing_successor": int((~valid).sum()),
                            "truncations": int(episode.truncated.sum()),
                        }
                    )
                    counts.update(
                        f"{kind}_transitions" for kind in episode.controller[valid]
                    )
                    groups[item["split"]].update(counts)
                    entry["rows"] = dict(counts)
                entries.append(entry)
    if set(groups) != {"train", "holdout"}:
        raise ValueError("Expected usable whole-fight training and holdout splits")
    return {
        "archive_sha256": sha256,
        "acceptance_plan_sha256": plan_hashes,
        "reserved_seed_count": len(reserved),
        "splits": dict(groups),
        "excluded": dict(excluded),
        "episodes": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--acceptance-plan", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.archive, args.archive_sha256, args.acceptance_plan)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("splits", "excluded")}))


if __name__ == "__main__":
    main()

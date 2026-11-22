"""Audit which native Sisters dash requests have an observable in-game onset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

HORIZON_SECONDS = 0.12
MAX_SAMPLE_GAP_SECONDS = 0.06
TOLERANCE = 1e-5
DASH = 1 << 2
# Indices in SistersTimingState, including the prior native input's dash bit.
GROUNDED = 7
DASH_AVAILABLE = 13
DASHING = 14
SHADOW_DASHING = 15
TRIO_PHASE = 23
PREVIOUS_DASH = 166
TIMING_WIDTH = 192

Status = Literal[
    "onset_observed", "no_onset_observed", "observation_gap", "truncated", "superseded"
]


class DashAttempt(TypedDict):
    row: int
    time: float
    phase: Literal["opening", "trio"]
    grounded: bool
    requested_direction: int
    status: Status
    onset_seconds_after_request: float | None
    observed_direction: int | None
    shadow_dash: bool | None


def audit_dash_attempts(
    timestamps: np.ndarray, state: np.ndarray, controls: np.ndarray
) -> list[DashAttempt]:
    """Attribute a sampled dash onset to at most one legal rising input."""
    if (
        timestamps.ndim != 1
        or state.shape != (len(timestamps), TIMING_WIDTH)
        or controls.shape != (len(timestamps), 3)
        or not np.issubdtype(controls.dtype, np.integer)
    ):
        raise ValueError(
            "Expected native timestamps, timing states, and primitive controls"
        )
    clock = timestamps.astype("f8")
    if not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
        raise ValueError("Native timestamps must be finite and strictly increasing")
    available, dashing, shadow, trio, previous = (
        state[:, field] > 0.5
        for field in (
            DASH_AVAILABLE,
            DASHING,
            SHADOW_DASHING,
            TRIO_PHASE,
            PREVIOUS_DASH,
        )
    )
    dash_pressed = (controls[:, 2] & DASH) != 0
    requests = np.flatnonzero(dash_pressed & ~previous & available & ~dashing)
    onsets = np.flatnonzero(dashing[1:] & ~dashing[:-1]) + 1
    attempts: list[DashAttempt] = []
    for position, index in enumerate(requests):
        row = int(index)
        time = float(clock[row])
        deadline = time + HORIZON_SECONDS
        stop = int(np.searchsorted(clock, deadline + TOLERANCE, side="right"))
        next_request = (
            int(requests[position + 1]) if position + 1 < len(requests) else len(clock)
        )
        next_onset = int(np.searchsorted(onsets, row, side="right"))
        onset = int(onsets[next_onset]) if next_onset < len(onsets) else len(clock)
        status: Status
        if onset < min(stop, next_request):
            status, final = "onset_observed", onset
        elif next_request < stop and clock[next_request] < deadline - TOLERANCE:
            status, final = "superseded", next_request
        elif clock[-1] < deadline - TOLERANCE:
            status, final = "truncated", len(clock) - 1
        else:
            status = "no_onset_observed"
            final = int(np.searchsorted(clock, deadline - TOLERANCE))
        if (np.diff(clock[row : final + 1]) > MAX_SAMPLE_GAP_SECONDS + TOLERANCE).any():
            status = "observation_gap"
        observed = status == "onset_observed"
        attempts.append(
            {
                "row": row,
                "time": time,
                "phase": "trio" if trio[row] else "opening",
                "grounded": bool(state[row, GROUNDED] > 0.5),
                "requested_direction": int(controls[row, 0]),
                "status": status,
                "onset_seconds_after_request": float(clock[onset] - time)
                if observed
                else None,
                "observed_direction": int(np.sign(state[onset, 2]))
                if observed
                else None,
                "shadow_dash": bool(shadow[onset]) if observed else None,
            }
        )
    return attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tapes", nargs="+", type=Path, help="Native timing recordings (.npz)."
    )
    args = parser.parse_args()
    counts = {
        phase: {stance: Counter[str]() for stance in ("grounded", "airborne")}
        for phase in ("opening", "trio")
    }
    fights = []
    for path in dict.fromkeys(tape.resolve() for tape in args.tapes):
        with np.load(path, allow_pickle=False) as data:
            attempts = audit_dash_attempts(
                data["control_time"], data["timing_state"], data["controls"]
            )
        for attempt in attempts:
            stance = "grounded" if attempt["grounded"] else "airborne"
            counts[attempt["phase"]][stance][attempt["status"]] += 1
        fights.append({"tape": str(path), "attempts": attempts})
    print(
        json.dumps(
            {
                "horizon_seconds": HORIZON_SECONDS,
                "maximum_sample_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
                "counts_by_phase": {
                    phase: stance["grounded"] + stance["airborne"]
                    for phase, stance in counts.items()
                },
                "counts_by_phase_and_stance": counts,
                "fights": fights,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Audit label-only Sisters of Battle tapes with zero teacher takeover."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

try:
    from hkrl.spaces import BUTTON_BITS, N_BUTTONS
except ModuleNotFoundError as exc:
    if exc.name not in {"hkrl", "hkrl.spaces"}:
        raise
    _python_dir = Path(__file__).resolve().parents[1] / "python"
    if (_python_dir / "hkrl" / "spaces.py").is_file():
        sys.path.insert(0, str(_python_dir))
    from hkrl.spaces import BUTTON_BITS, N_BUTTONS

REQUIRED_FIELDS = (
    "control_time",
    "timing_state",
    "controls",
    "student_controls",
    "teacher_controls",
    "teacher_acted",
)
INTERVENTION_FIELDS: frozenset[str] = frozenset(
    {
        "block_starts",
        "candidate_draw",
        "choice_rows",
        "choice_hold_stop",
        "key_rows",
        "pair_rows",
        "plan_costs",
        "plan_choice",
        "ppo_rows",
        "route_rows",
    }
)

MAX_INTERVAL = np.float32(0.081)
STATE_WIDTH = 192
CONTROL_WIDTH = 3
SOUL_INDEX = 5
GROUNDED_INDEX = 7
CAN_FOCUS_INDEX = 12
TRIO_INDEX = 23
SOUL_SCALE = 99.0
MIN_FOCUS_SOUL = 33.0
FLOAT_TOL = 1e-5

DASH_MASK = 1 << BUTTON_BITS["dash"]
CAST_MASK = 1 << BUTTON_BITS["cast"]
FOCUS_HOLD_MASK = 1 << BUTTON_BITS["focus_hold"]
MAX_BUTTON_MASK = (1 << N_BUTTONS) - 1


def _validate_controls(name: str, arr: object, n: int) -> np.ndarray:
    if (
        not isinstance(arr, np.ndarray)
        or arr.shape != (n, CONTROL_WIDTH)
        or not np.issubdtype(arr.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be an integer ndarray with shape ({n}, {CONTROL_WIDTH})")
    xy = arr[:, :2]
    bits = arr[:, 2]
    if int(xy.min()) < -1 or int(xy.max()) > 1:
        raise ValueError(f"{name} x/y values must be in {{-1, 0, 1}}")
    if int(bits.min()) < 0 or int(bits.max()) > MAX_BUTTON_MASK:
        raise ValueError(f"{name} button bitfield must be in [0, {MAX_BUTTON_MASK}]")
    return arr


def _trio_run_lengths(
    trio_mask: np.ndarray,
    control_time: np.ndarray,
    active: np.ndarray,
) -> list[int]:
    """Return lengths of maximal contiguous active runs within Trio segments.

    A run breaks on any non-Trio row, inactive row, or time gap > 0.081s.
    """
    runs: list[int] = []
    current = 0
    prev_time = 0.0
    max_dt = float(MAX_INTERVAL) + FLOAT_TOL
    for in_trio, timestamp, flag in zip(
        trio_mask.tolist(), control_time.tolist(), active.tolist(), strict=True
    ):
        if not in_trio or not flag:
            if current:
                runs.append(current)
                current = 0
            continue
        if current and (timestamp - prev_time) > max_dt:
            runs.append(current)
            current = 1
        else:
            current += 1
        prev_time = timestamp
    if current:
        runs.append(current)
    return runs


def audit_label_only(tape: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Validate a label-only tape and summarize Trio advisory disagreements."""
    if not isinstance(tape, Mapping):
        raise ValueError("Expected a mapping of tape arrays")

    forbidden = sorted(INTERVENTION_FIELDS.intersection(tape))
    if forbidden:
        raise ValueError(
            f"Forbidden intervention fields in label-only tape: {', '.join(forbidden)}"
        )

    missing = [key for key in REQUIRED_FIELDS if key not in tape]
    if missing:
        raise ValueError(f"Missing required tape fields: {', '.join(missing)}")

    control_time = tape["control_time"]
    timing_state = tape["timing_state"]
    controls = tape["controls"]
    student_controls = tape["student_controls"]
    teacher_controls = tape["teacher_controls"]
    teacher_acted = tape["teacher_acted"]

    if (
        not isinstance(control_time, np.ndarray)
        or control_time.ndim != 1
        or len(control_time) == 0
        or not np.issubdtype(control_time.dtype, np.floating)
    ):
        raise ValueError("control_time must be a non-empty 1D floating ndarray")
    if not np.isfinite(control_time).all():
        raise ValueError("control_time contains non-finite values")
    if (control_time[1:] <= control_time[:-1]).any():
        raise ValueError("control_time must be strictly increasing")

    n = int(control_time.shape[0])

    if (
        not isinstance(timing_state, np.ndarray)
        or timing_state.shape != (n, STATE_WIDTH)
        or not np.issubdtype(timing_state.dtype, np.floating)
    ):
        raise ValueError(f"timing_state must be a floating ndarray with shape ({n}, {STATE_WIDTH})")
    if not np.isfinite(timing_state).all():
        raise ValueError("timing_state contains non-finite values")

    if (
        not isinstance(teacher_acted, np.ndarray)
        or teacher_acted.shape != (n,)
        or teacher_acted.dtype != np.bool_
    ):
        raise ValueError(f"teacher_acted must be a bool ndarray with shape ({n},)")
    if teacher_acted.any():
        raise ValueError("Teacher takeover is forbidden in label-only tapes")

    controls = _validate_controls("controls", controls, n)
    student_controls = _validate_controls("student_controls", student_controls, n)
    teacher_controls = _validate_controls("teacher_controls", teacher_controls, n)

    if not np.array_equal(controls, student_controls):
        raise ValueError("Executed controls must match student_controls")

    trio_mask = timing_state[:, TRIO_INDEX] > 0.5
    trio_rows = int(trio_mask.sum())
    if trio_rows == 0:
        raise ValueError("Tape contains no Trio rows (timing_state[:, 23] > 0.5)")

    student_bits = student_controls[:, 2].astype(np.int64, copy=False)
    teacher_bits = teacher_controls[:, 2].astype(np.int64, copy=False)

    teacher_dash = (teacher_bits & DASH_MASK) != 0
    teacher_cast = (teacher_bits & CAST_MASK) != 0
    teacher_focus = (teacher_bits & FOCUS_HOLD_MASK) != 0

    unexec_dash = teacher_dash & ((student_bits & DASH_MASK) == 0)
    unexec_cast = teacher_cast & ((student_bits & CAST_MASK) == 0)
    unexec_focus = teacher_focus & ((student_bits & FOCUS_HOLD_MASK) == 0)

    trio_state = timing_state[trio_mask]
    focus_legal = (
        (trio_state[:, GROUNDED_INDEX] > 0.5)
        & (trio_state[:, SOUL_INDEX].astype(np.float64) * SOUL_SCALE >= MIN_FOCUS_SOUL - FLOAT_TOL)
        & (trio_state[:, CAN_FOCUS_INDEX] > 0.5)
    )

    disagreements = np.any(teacher_controls[trio_mask] != student_controls[trio_mask], axis=1)
    focus_runs = _trio_run_lengths(trio_mask, control_time, unexec_focus)
    cast_runs = _trio_run_lengths(trio_mask, control_time, unexec_cast)

    return {
        "trio_rows": trio_rows,
        "teacher_and_student_disagreements": int(disagreements.sum()),
        "unexecuted_teacher_dash": int(unexec_dash[trio_mask].sum()),
        "unexecuted_teacher_cast": int(unexec_cast[trio_mask].sum()),
        "unexecuted_teacher_focus": int(unexec_focus[trio_mask].sum()),
        "illegal_teacher_focus_requests": int((teacher_focus[trio_mask] & ~focus_legal).sum()),
        "unexecuted_teacher_focus_longest_run": max(focus_runs, default=0),
        "unexecuted_teacher_cast_run_count": len(cast_runs),
    }


def audit_npz(path: str | Path) -> dict[str, int]:
    """Load an .npz tape with allow_pickle=False and audit it."""
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected an .npz archive, got {type(loaded).__name__}")
    with loaded as tape:
        return audit_label_only(tape)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape", type=Path, help="Path to a label-only .npz tape.")
    args = parser.parse_args(argv)
    print(json.dumps(audit_npz(args.tape)))


if __name__ == "__main__":
    main()

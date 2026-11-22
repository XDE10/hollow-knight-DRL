from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import hkrl.spaces as hkrl_spaces
import numpy as np
import pytest

# Independent verified bit masks from python/hkrl/spaces.py BUTTON_BITS.
DASH_MASK = 1 << 2  # dash == 2
ATTACK_MASK = 1 << 3  # attack == 3 (distinct from cast)
CAST_MASK = 1 << 4  # cast == 4 (distinct from focus_hold)
FOCUS_HOLD_MASK = 1 << 5  # focus_hold == 5
DREAM_NAIL_MASK = 1 << 6  # dream_nail == 6

INTERVENTION_FIELDS = (
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
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_sisters_label_only.py"


def _load_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPT_PATH.parent))
    return importlib.import_module("audit_sisters_label_only")


def _make_tape(n: int = 6, *, dt: float = 0.04) -> dict[str, np.ndarray]:
    timing_state = np.zeros((n, 192), dtype=np.float32)
    timing_state[:, 23] = 1.0
    timing_state[:, 7] = 1.0
    timing_state[:, 5] = np.float32(33.0 / 99.0)
    timing_state[:, 12] = 1.0
    student_controls = np.zeros((n, 3), dtype=np.int16)
    return {
        "control_time": np.arange(n, dtype=np.float32) * np.float32(dt),
        "timing_state": timing_state,
        "controls": student_controls.copy(),
        "student_controls": student_controls,
        "teacher_controls": student_controls.copy(),
        "teacher_acted": np.zeros(n, dtype=np.bool_),
    }


def test_canonical_button_bits_and_non_target_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert hkrl_spaces.BUTTON_BITS["dash"] == 2
    assert hkrl_spaces.BUTTON_BITS["cast"] == 4
    assert hkrl_spaces.BUTTON_BITS["focus_hold"] == 5

    mod = _load_module(monkeypatch)
    assert mod.DASH_MASK == DASH_MASK
    assert mod.CAST_MASK == CAST_MASK
    assert mod.FOCUS_HOLD_MASK == FOCUS_HOLD_MASK

    tape = _make_tape(4)
    # Unsigned and signed 8-bit controls are valid when values are in range.
    tape["controls"] = np.zeros((4, 3), dtype=np.uint8)
    tape["student_controls"] = np.zeros((4, 3), dtype=np.uint8)
    tape["teacher_controls"] = np.zeros((4, 3), dtype=np.int8)
    # Bit 3 is attack (not cast) and bit 6 is dream_nail (not focus/dash).
    tape["teacher_controls"][0, 2] = ATTACK_MASK
    tape["teacher_controls"][1, 2] = DREAM_NAIL_MASK
    tape["teacher_controls"][2, 2] = 1 << 3  # old wrong cast bit

    assert mod.audit_label_only(tape) == {
        "trio_rows": 4,
        "teacher_and_student_disagreements": 3,
        "unexecuted_teacher_dash": 0,
        "unexecuted_teacher_cast": 0,
        "unexecuted_teacher_focus": 0,
        "illegal_teacher_focus_requests": 0,
        "unexecuted_teacher_focus_longest_run": 0,
        "unexecuted_teacher_cast_run_count": 0,
    }


def test_opening_phase_excluded_and_bitwise_counts_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module(monkeypatch)

    tape = _make_tape(10)
    tape["control_time"] = np.array(
        [0.00, 0.04, 0.08, 0.161, 0.201, 0.283, 0.323, 0.363, 0.403, 0.443],
        dtype=np.float32,
    )
    # Rows 0-1 are opening phase (non-Trio) and must not be counted.
    tape["timing_state"][:2, 23] = 0.0
    tape["timing_state"][:2, 7] = 0.0
    tape["timing_state"][:2, 5] = 0.0
    tape["timing_state"][:2, 12] = 0.0
    tape["teacher_controls"][:2, 0] = 1
    tape["teacher_controls"][:2, 2] = DASH_MASK | CAST_MASK | FOCUS_HOLD_MASK

    # Trio segment 1 (rows 2-4): dt(2->3) == 0.081s stays in the same run.
    tape["teacher_controls"][2, 2] = DASH_MASK | FOCUS_HOLD_MASK
    tape["student_controls"][3, 2] = DASH_MASK
    tape["controls"][3, 2] = DASH_MASK
    tape["teacher_controls"][3, 2] = DASH_MASK | CAST_MASK | FOCUS_HOLD_MASK
    tape["teacher_controls"][4, 2] = CAST_MASK | FOCUS_HOLD_MASK

    # Gap 0.082s > 0.081s between row 4 and row 5 splits Trio segments.
    # Trio segment 2 (rows 5-9): focus run len 1, cast runs 2 (rows 5-6) and 3 (row 9).
    tape["teacher_controls"][5, 2] = CAST_MASK | FOCUS_HOLD_MASK
    tape["student_controls"][6, 2] = FOCUS_HOLD_MASK
    tape["controls"][6, 2] = FOCUS_HOLD_MASK
    tape["teacher_controls"][6, 2] = CAST_MASK | FOCUS_HOLD_MASK
    tape["student_controls"][7, 2] = CAST_MASK
    tape["controls"][7, 2] = CAST_MASK
    tape["teacher_controls"][7, 2] = CAST_MASK
    tape["student_controls"][8, 0] = -1
    tape["controls"][8, 0] = -1
    tape["teacher_controls"][8] = [1, 0, ATTACK_MASK]
    tape["teacher_controls"][9, 2] = DASH_MASK | CAST_MASK

    report = mod.audit_label_only(tape)
    assert all(type(value) is int for value in report.values())
    assert report == {
        "trio_rows": 8,
        "teacher_and_student_disagreements": 7,
        "unexecuted_teacher_dash": 2,
        "unexecuted_teacher_cast": 5,
        "unexecuted_teacher_focus": 4,
        "illegal_teacher_focus_requests": 0,
        "unexecuted_teacher_focus_longest_run": 3,
        "unexecuted_teacher_cast_run_count": 3,
    }

    # An interior non-Trio row also breaks active runs even with dt == 0.04s.
    split_tape = _make_tape(5)
    split_tape["timing_state"][2, 23] = 0.0
    split_tape["teacher_controls"][:, 2] = CAST_MASK | FOCUS_HOLD_MASK
    split_report = mod.audit_label_only(split_tape)
    assert split_report["trio_rows"] == 4
    assert split_report["unexecuted_teacher_focus_longest_run"] == 2
    assert split_report["unexecuted_teacher_cast_run_count"] == 2


def test_illegal_teacher_focus_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(monkeypatch)

    tape = _make_tape(6)
    # Row 0: legal focus request (grounded, SOUL=33, CanFocus=1).
    tape["teacher_controls"][0, 2] = FOCUS_HOLD_MASK
    # Row 1: illegal (not grounded).
    tape["timing_state"][1, 7] = 0.0
    tape["teacher_controls"][1, 2] = FOCUS_HOLD_MASK
    # Row 2: illegal (SOUL < 33).
    tape["timing_state"][2, 5] = np.float32(32.0 / 99.0)
    tape["teacher_controls"][2, 2] = FOCUS_HOLD_MASK
    # Row 3: illegal (CanFocus=0).
    tape["timing_state"][3, 12] = 0.0
    tape["teacher_controls"][3, 2] = FOCUS_HOLD_MASK
    # Row 4: illegal even when student also presses focus (executed, not unexecuted).
    tape["timing_state"][4, [7, 5, 12]] = 0.0
    tape["student_controls"][4, 2] = FOCUS_HOLD_MASK
    tape["controls"][4, 2] = FOCUS_HOLD_MASK
    tape["teacher_controls"][4, 2] = FOCUS_HOLD_MASK
    # Row 5: illegal state, teacher requests cast (bit 4) instead of focus (bit 5).
    tape["timing_state"][5, [7, 5, 12]] = 0.0
    tape["teacher_controls"][5, 2] = CAST_MASK

    report = mod.audit_label_only(tape)
    assert report == {
        "trio_rows": 6,
        "teacher_and_student_disagreements": 5,
        "unexecuted_teacher_dash": 0,
        "unexecuted_teacher_cast": 1,
        "unexecuted_teacher_focus": 4,
        "illegal_teacher_focus_requests": 4,
        "unexecuted_teacher_focus_longest_run": 4,
        "unexecuted_teacher_cast_run_count": 1,
    }


def test_rejects_takeover_and_student_control_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module(monkeypatch)

    trio_takeover = _make_tape(4)
    trio_takeover["teacher_acted"][2] = True
    with pytest.raises(ValueError, match="takeover"):
        mod.audit_label_only(trio_takeover)

    opening_takeover = _make_tape(4)
    opening_takeover["timing_state"][0, 23] = 0.0
    opening_takeover["teacher_acted"][0] = True
    with pytest.raises(ValueError, match="takeover"):
        mod.audit_label_only(opening_takeover)

    axis_mismatch = _make_tape(4)
    axis_mismatch["controls"][1, 0] = 1
    with pytest.raises(ValueError, match="student_controls"):
        mod.audit_label_only(axis_mismatch)

    bits_mismatch = _make_tape(4)
    bits_mismatch["controls"][2, 2] = DASH_MASK
    with pytest.raises(ValueError, match="student_controls"):
        mod.audit_label_only(bits_mismatch)


def test_rejects_intervention_fields_missing_fields_and_non_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module(monkeypatch)

    for field in INTERVENTION_FIELDS:
        tape = _make_tape(4)
        tape[field] = np.zeros(1, dtype=np.int16)
        with pytest.raises(ValueError, match=field):
            mod.audit_label_only(tape)

    missing_tape = _make_tape(4)
    del missing_tape["teacher_controls"]
    with pytest.raises(ValueError, match="Missing required"):
        mod.audit_label_only(missing_tape)

    with pytest.raises(ValueError, match="mapping"):
        mod.audit_label_only(["not", "a", "mapping"])  # type: ignore[arg-type]


def _set_executed_cell(tape: dict[str, np.ndarray], col: int, value: int) -> None:
    for key in ("controls", "student_controls"):
        tape[key][0, col] = value


def _set_at(arr: np.ndarray, idx: int | tuple[int | slice, ...], value: object) -> None:
    arr[idx] = value


@pytest.mark.parametrize(
    "mutator",
    [
        lambda t: _set_executed_cell(t, 0, 2),
        lambda t: _set_at(t["teacher_controls"], (0, 0), -2),
        lambda t: _set_at(t["teacher_controls"], (0, 1), 2),
        lambda t: _set_at(t["teacher_controls"], (0, 2), 512),
        lambda t: _set_executed_cell(t, 2, -1),
        lambda t: t.update(controls=t["controls"][:, :2]),
        lambda t: t.update(
            controls=t["controls"].astype(np.float32),
            student_controls=t["student_controls"].astype(np.float32),
        ),
        lambda t: t.update(teacher_controls=t["teacher_controls"].astype(np.bool_)),
    ],
)
def test_rejects_invalid_controls(
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[dict[str, np.ndarray]], object],
) -> None:
    mod = _load_module(monkeypatch)
    tape = _make_tape(4)
    mutator(tape)
    with pytest.raises(ValueError):
        mod.audit_label_only(tape)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda t: t.update(control_time=t["control_time"][:, None]),
        lambda t: t.update(control_time=np.empty(0, dtype=np.float32)),
        lambda t: _set_at(t["control_time"], 2, np.nan),
        lambda t: _set_at(t["control_time"], 2, np.inf),
        lambda t: _set_at(t["control_time"], 2, t["control_time"][1]),
        lambda t: _set_at(t["control_time"], 2, t["control_time"][1] - np.float32(0.01)),
        lambda t: t.update(timing_state=t["timing_state"][:, :191]),
        lambda t: t.update(timing_state=t["timing_state"].astype(np.int32)),
        lambda t: _set_at(t["timing_state"], (1, 0), np.nan),
        lambda t: _set_at(t["timing_state"], (1, 0), np.inf),
        lambda t: _set_at(t["timing_state"], (slice(None), 23), 0.0),
        lambda t: t.update(teacher_acted=np.zeros(len(t["teacher_acted"]), dtype=np.int8)),
    ],
)
def test_rejects_invalid_time_state_acted_and_missing_trio(
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[dict[str, np.ndarray]], object],
) -> None:
    mod = _load_module(monkeypatch)
    tape = _make_tape(4)
    mutator(tape)
    with pytest.raises(ValueError):
        mod.audit_label_only(tape)


def test_npz_mapping_and_cli_roundtrip_without_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert issubclass(np.lib.npyio.NpzFile, Mapping)

    mod = _load_module(monkeypatch)
    tape = _make_tape(4)
    tape["teacher_controls"][1, 2] = DASH_MASK
    tape["teacher_controls"][2, 2] = CAST_MASK
    tape["teacher_controls"][3, 2] = FOCUS_HOLD_MASK

    # Benign extra arrays written by collect_sisters_native_labels.py are permitted.
    saved = {
        **tape,
        "student_features": np.zeros((4, 56), dtype=np.float32),
        "time": np.arange(4, dtype=np.float32),
        "events": np.zeros((4, 8), dtype=np.float32),
        "elapsed_ticks": np.full(4, 2, dtype=np.int16),
    }

    npz_path = tmp_path / "label_only.npz"
    np.savez(npz_path, **saved)
    with np.load(npz_path, allow_pickle=False) as loaded:
        assert isinstance(loaded, Mapping)

    expected = mod.audit_label_only(tape)
    assert expected["unexecuted_teacher_dash"] == 1
    assert expected["unexecuted_teacher_cast"] == 1
    assert expected["unexecuted_teacher_focus"] == 1

    mod.main([str(npz_path)])
    assert json.loads(capsys.readouterr().out) == expected

    compressed_path = tmp_path / "label_only_compressed.npz"
    np.savez_compressed(compressed_path, **saved)
    env = os.environ.copy()
    hkrl_root = str(Path(hkrl_spaces.__file__).resolve().parents[1])
    existing_pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{hkrl_root}{os.pathsep}{existing_pp}" if existing_pp else hkrl_root

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(compressed_path)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert json.loads(proc.stdout) == expected

    # Include all required keys so audit_npz hits NumPy's allow_pickle=False check.
    pickled_tape = _make_tape(1)
    pickled_tape["control_time"] = np.array([object()], dtype=object)
    pickled_path = tmp_path / "pickled.npz"
    np.savez(pickled_path, **pickled_tape)
    with pytest.raises(ValueError, match=r"(?i)pickle"):
        mod.audit_npz(pickled_path)

    bad_proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(pickled_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert bad_proc.returncode != 0
    assert "pickle" in bad_proc.stderr.lower()

    npy_path = tmp_path / "not_an_archive.npy"
    np.save(npy_path, np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match=r"\.npz"):
        mod.audit_npz(npy_path)

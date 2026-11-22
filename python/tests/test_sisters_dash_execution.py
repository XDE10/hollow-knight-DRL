from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def dash_audit(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("audit_sisters_dash_execution")


def frames(clock: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.array(clock, dtype="f4")
    state = np.zeros((len(timestamps), 192), dtype="f4")
    state[:, 13] = 1.0
    controls = np.zeros((len(timestamps), 3), dtype="i2")
    return timestamps, state, controls


def test_observed_dash_and_held_input_are_attributed_once(dash_audit: ModuleType) -> None:
    clock, state, controls = frames([0.0, 0.04, 0.08, 0.12, 0.16])
    controls[:3, 0] = -1
    controls[:3, 2] = 4
    state[1:4, 166] = 1
    state[1:3, 14:16] = 1
    state[1:3, 2] = 2
    state[:, 23] = 1

    (attempt,) = dash_audit.audit_dash_attempts(clock, state, controls)
    assert attempt == {
        "row": 0,
        "time": 0.0,
        "phase": "trio",
        "requested_direction": -1,
        "status": "onset_observed",
        "onset_seconds_after_request": pytest.approx(0.04),
        "observed_direction": 1,
        "shadow_dash": True,
    }


def test_truncated_and_sparse_windows_are_distinguished(dash_audit: ModuleType) -> None:
    clock, state, controls = frames([0.0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.28, 0.32, 0.36])
    controls[[0, 3, 8], 2] = 4
    state[[1, 4], 166] = 1
    first, sparse, last = dash_audit.audit_dash_attempts(clock, state, controls)
    assert first["status"] == "no_onset_observed"
    assert sparse["status"] == "observation_gap"
    assert last["status"] == "truncated"
    assert all(attempt["observed_direction"] is None for attempt in (first, sparse, last))


def test_a_second_request_supersedes_the_first(dash_audit: ModuleType) -> None:
    clock, state, controls = frames([0.0, 0.04, 0.08, 0.12])
    controls[[0, 2], 2] = 4
    state[1, 166] = 1
    state[3, 14] = 1
    first, second = dash_audit.audit_dash_attempts(clock, state, controls)
    assert first["status"] == "superseded"
    assert second["status"] == "onset_observed"


def test_rejects_unordered_timestamps(dash_audit: ModuleType) -> None:
    clock, state, controls = frames([0.04, 0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        dash_audit.audit_dash_attempts(clock, state, controls)

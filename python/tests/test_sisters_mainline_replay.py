from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import numpy as np
import pytest


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("audit_sisters_mainline_replay")


@pytest.fixture
def native_tape() -> dict[str, np.ndarray]:
    clock = np.array([0, 0.04, 0.08, 0.24, 0.28, 0.32], dtype="f4")
    state = np.zeros((len(clock), 192), dtype="f4")
    state[:, 4] = 1
    student = np.zeros((len(clock), 3), dtype="i2")
    teacher = np.ones_like(student)
    teacher_acted = np.array([True, True, False, False, False, False])
    events = np.zeros((2, 8), dtype="f4")
    events[0, 0] = 21
    events[1, 6] = 1
    return {
        "control_time": clock,
        "timing_state": state,
        "student_controls": student,
        "teacher_controls": teacher,
        "teacher_acted": teacher_acted,
        "controls": np.where(teacher_acted[:, None], teacher, student),
        "time": np.array([0, 0.24], dtype="f4"),
        "elapsed_ticks": np.array([12, 4]),
        "events": events,
    }


def test_terminal_and_gap_use_only_observed_successors(
    replay: ModuleType, native_tape: dict[str, np.ndarray]
) -> None:
    episode = replay.read_episode(native_tape, {"won": True})
    assert episode.next_row.tolist() == [1, 2, -1, 4, -1]
    assert episode.valid_transition.tolist() == [True, True, False, True, True]
    assert episode.terminated.tolist() == [False, False, False, False, True]
    assert not episode.truncated.any()
    assert episode.controller.tolist() == ["teacher", "teacher", "student", "student", "student"]
    assert episode.events.sum(axis=0).tolist() == native_tape["events"].sum(axis=0).tolist()
    assert episode.reward.sum() == pytest.approx(8.45)


def test_legacy_unknown_outcome_and_explicit_truncation(
    replay: ModuleType, native_tape: dict[str, np.ndarray]
) -> None:
    native_tape["events"][1, 6] = 0
    with pytest.raises(replay.UnusableFight, match="no observed terminal"):
        replay.read_episode(native_tape, {"won": False})
    episode = replay.read_episode(native_tape, {"won": False, "truncated": True})
    assert episode.truncated.tolist() == [False, False, False, False, False, True]
    assert not episode.terminated.any()
    assert not episode.valid_transition[-1]
    assert episode.next_row[-2] == len(episode.clock) - 1


def test_audit_locks_archive_and_holds_out_acceptance_seeds(
    replay: ModuleType, native_tape: dict[str, np.ndarray], tmp_path: Path
) -> None:
    data = io.BytesIO()
    np.savez_compressed(data, **native_tape)
    raw = data.getvalue()
    archive_path = tmp_path / "replay.zip"
    items = []
    with ZipFile(archive_path, "w") as archive:
        for seed, split in [(1, "train"), (150, "train"), (151, "holdout")]:
            name = f"fight-{seed}.npz"
            items.append(
                {
                    "source": f"native-development/{seed}",
                    "split": split,
                    "file": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "won": True,
                }
            )
            archive.writestr(name, raw)
        archive.writestr("manifest.json", json.dumps(items))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"goal": 3, "first_seed": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        replay.audit(archive_path, "0" * 64, [plan])
    report = replay.audit(
        archive_path, hashlib.sha256(archive_path.read_bytes()).hexdigest(), [plan]
    )
    assert report["excluded"] == {"reserved acceptance seed": 1}
    assert report["splits"]["train"]["episodes"] == 1
    assert report["splits"]["holdout"]["episodes"] == 1

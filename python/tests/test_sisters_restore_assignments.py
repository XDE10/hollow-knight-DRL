from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest


def test_original_assignment_disambiguates_identical_keys_and_rejects_mismatched_tape(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    restore = importlib.import_module("restore_sisters_assignments")
    rows = np.zeros((3, 259), "<f4")
    rows[:, 0] = [0, 0.04, 0.08]
    rows[:, 57] = [1, 1, -1]
    rows[:, 60] = [1, 1, 1]
    rows[:, 63] = [1, 1, -1]
    rows[:, 66] = [1, 0, 0]
    src = struct.pack("<IQ", 0x32474144, 12) + rows.tobytes()
    game = {
        "control_time": rows[:, 0].copy(),
        "timing_state": rows[:, 67:].copy(),
        "controls": rows[:, 57:60].astype("i2"),
        "teacher_controls": rows[:, 60:63].astype("i2"),
    }
    result = restore.recover_decisions(src, 12, game)
    assert result["teacher_acted"].tolist() == [True, False, False]
    game["timing_state"][0, 1] = 0.5
    with pytest.raises(ValueError, match="archived game"):
        restore.recover_decisions(src, 12, game)

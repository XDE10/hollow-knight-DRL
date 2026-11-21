from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest


def test_held_descent_draw_requires_real_alternatives_and_contiguous_issued_direction(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("sisters_online_choices")
    path = tmp_path / "1.choices"
    rows = np.array([(0.08, [0, 1], 1, 0.5, 0.8)], dtype=mod.ROW)
    path.write_bytes(struct.pack("<IQ", 0x32434B48, 1) + rows.tobytes())
    t = np.arange(12, dtype="f4") * 0.04
    state = np.zeros((len(t), 192), "f4")
    state[:, [23, 60, 81]] = 1
    cat = np.array([[-1, 0, 8], [1, 0, 8]], "i4")
    applied = np.tile(cat[0], (len(t), 1))
    applied[2:6] = cat[1]
    logged = mod.read_choices(path, 1, t, state, applied, cat)
    assert logged["choice_rows"].tolist() == [2]
    assert logged["choice_hold_stop"].tolist() == [6]
    applied[4, 0] = -1
    with pytest.raises(ValueError, match="during the hold"):
        mod.read_choices(path, 1, t, state, applied, cat)
    state[:, 60] = 0
    with pytest.raises(ValueError, match="supported horizontal"):
        mod.read_choices(path, 1, t, state, applied, cat)

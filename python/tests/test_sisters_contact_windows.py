from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def test_contact_window_respects_exact_handoff_and_both_sources(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    windows = importlib.import_module("sisters_contact_windows").contact_windows
    t = np.arange(10, dtype="f4") * 0.04
    events = np.zeros((len(t), 8))
    events[1, 1] = 1
    events[3, 2] = 1
    valid, label = windows(t, events, np.array([3]), 0.12)
    assert valid[0] and not valid[1] and not valid[2] and valid[3]
    assert label[0].tolist() == [True, False, False, False]
    assert label[3].tolist() == [False, True, False, False]
    assert not valid[-2:].any()


def test_terminal_is_absorbing_and_unobserved_time_cannot_win_a_label(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    windows = importlib.import_module("sisters_contact_windows").contact_windows
    t = np.array([0, 0.04, 0.19, 0.23, 0.27, 0.31])
    events = np.zeros((len(t), 8))
    events[3, [2, 7]] = 1
    valid, label = windows(t, events, np.array([], dtype=int))
    assert valid.tolist() == [False, False, True, True, False, False]
    assert label[valid, 1].all()
    events[3] = 0
    assert not windows(t, events, np.array([], dtype=int))[0].any()

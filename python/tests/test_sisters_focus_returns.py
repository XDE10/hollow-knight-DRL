from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def focus(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_focus_returns")


def test_focus_windows_use_issued_keys_and_actual_health(focus):
    t = np.arange(60, dtype="f4") * 0.04
    s = np.zeros((len(t), 192), dtype="f4")
    s[:, 23] = 1
    s[:, 4] = 7 / 9
    s[26:, 4] = 8 / 9
    s[:, 5] = (99 - np.arange(len(t)).clip(0, 32)) / 99
    s[6:36, 169] = 1
    s[6:36, 178] = t[6:36] - t[5]
    controls = np.zeros((len(t), 3), dtype="i4")
    controls[5:35, 2] = 32
    events = np.zeros((len(t), 8), dtype="f4")
    events[12, 5] = 9  # A request to heal does not increase the Knight's observed HP.
    data = dict(
        control_time=t,
        timing_state=s,
        controls=controls,
        events=events,
        time=t,
        elapsed_ticks=np.full(len(t), 2),
    )
    _, arm, outcomes, starts, rows = focus.focus_examples(data)
    assert rows[0] == 6 and rows[-1] == 25
    assert np.all(starts == 5) and arm.min() == 1
    assert outcomes[rows == 7, 0].item() == 0
    assert outcomes[rows == 20, 0].item() == 1
    s[26:, 4] = 7 / 9
    _, arm, outcomes, _, rows = focus.focus_examples(data)
    assert arm[rows == 35].item() == 0
    assert outcomes[:, 0].sum() == 0
    t[27:] += 0.6
    _, _, _, _, rows = focus.focus_examples(data)
    assert 20 not in rows and 35 not in rows


def test_focus_checkpoint_preserves_phase_and_rejects_invalid_values(focus, tmp_path):
    import torch
    from train_sisters_timing_iql import (
        TimingPolicy,
        focus_components,
        phase_checkpoint,
        phase_components,
        save_native,
    )

    cat = torch.tensor([[0, 0, 0], [0, 0, 32]])
    path = tmp_path / "native.bin"
    save_native(TimingPolicy(cat, True), cat.numpy(), path)
    leaf = path.read_bytes()
    w = np.linspace(-1, 1, focus.FOCUS_FEATURES, dtype="<f4")
    prefix = struct.pack("<2i", focus.FOCUS_MAGIC, len(leaf)) + leaf
    wrap = prefix + w.tobytes()
    assert phase_components(phase_checkpoint(leaf, wrap)) == (leaf, wrap)
    actor, values = focus_components(wrap)
    assert actor == leaf
    np.testing.assert_array_equal(w, values)
    w[0] = np.nan
    with pytest.raises(ValueError):
        phase_checkpoint(leaf, prefix + w.tobytes())

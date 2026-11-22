from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


def test_plan_trace_alignment_overrides_and_supported_motions(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    plans = importlib.import_module("sisters_plan_labels")
    fit = importlib.import_module("train_sisters_timing_iql")
    times = np.array([0.1, 0.2, 0.3], dtype="f4")
    tape = np.zeros(2, dtype=plans.PLAN_ROW)
    tape["time"] = times[[0, 2]]
    tape["cost"] = np.inf
    tape["cost"][:, [0, 4, 8]] = [1, 3, 7]
    path = tmp_path / "1.plans"
    path.write_bytes(struct.pack("<IQi", 0x31504B48, 1, 12) + tape.tobytes())
    table = plans.read_plans(path, 1, times)
    np.testing.assert_array_equal(table["plan_choice"], [0, -1, 0])
    labels = np.array([[-1, 0, 8], [-1, 0, 0], [1, 0, 8]])
    weights = plans.plan_preferences(table["plan_costs"], table["plan_choice"], labels)
    np.testing.assert_array_equal(weights[0, [0, 4, 8]], [0, 0.5, 1])
    assert np.count_nonzero(weights) == 2
    with pytest.raises(ValueError, match="timestamps"):
        plans.read_plans(path, 1, times + 0.01)
    # Different attack or aim keys can contribute to the same motion probability.
    commands = torch.tensor([[-1, 0, 0], [-1, 1, 8], [1, 0, 0], [0, 0, 7]])
    groups = fit.movement_groups(commands)
    assert groups.tolist() == [0, 0, 8, 7]
    member = torch.nn.functional.one_hot(groups, 12).T.bool()
    logits = torch.zeros(1, 4, requires_grad=True)
    margin = fit.movement_margins(logits, member, torch.tensor([0]))
    assert margin[0, 8].item() == pytest.approx(np.log(2))
    torch.nn.functional.softplus(-margin[0, 8]).backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 2] > 0

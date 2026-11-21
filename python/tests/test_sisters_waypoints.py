from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def waypoints(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_waypoints")


def test_motion_target_excludes_knockback_handoffs_and_gaps(waypoints):
    t = np.arange(50, dtype="f4") * 0.04
    s = np.zeros((len(t), 192), "f4")
    s[:, 23], s[:, 4], s[:, 0] = 1, 1, t * 0.4
    debit = np.zeros(len(t))
    debit[11] = 1
    s[27:, 4] = 8 / 9
    t[39:] += 0.4
    g, valid = waypoints.waypoint_labels(t, s, debit, np.array([22]))
    assert g[0] == 7 and valid[0]
    assert not any(valid[[8, 17, 24, 33, 46]])
    assert valid[12] and valid[28]


def test_shuffled_targets_stay_within_kinematic_groups(waypoints):
    r = np.random.default_rng(314)
    s = np.zeros((40, 192), "f4")
    s[20:, 2] = 1
    goal = np.arange(40) % 9
    valid = np.arange(40) % 3 != 2
    shuffled = waypoints.conditional_shuffle(s, goal, valid, r)
    np.testing.assert_array_equal(shuffled[~valid], goal[~valid])
    for mask in (s[:, 2] < 0.5, s[:, 2] >= 0.5):
        np.testing.assert_array_equal(np.sort(shuffled[mask & valid]), np.sort(goal[mask & valid]))
    assert (shuffled != goal).any()


def test_deployable_policy_uses_only_current_state(waypoints):
    net = waypoints.GoalActor(5)
    h = torch.zeros(3, 192)
    base = torch.randn(3, 5)
    with torch.no_grad():
        assert torch.allclose(net.policy(h, base), base.softmax(1))
        net.motor[-1].weight.uniform_(-0.1, 0.1)
        assert torch.allclose(net.policy(h, base).sum(1), torch.ones(3))

"""Optional teacher movement preferences measured during development fights."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

PLAN_ACTIONS = 12
PLAN_ROW = np.dtype([("time", "<f4"), ("choice", "<i4"), ("cost", "<f4", PLAN_ACTIONS)])


def plan_preferences(
    costs: np.ndarray, choices: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Preference weights relative to the teacher's final movement, jump and dash.

    The path planner may be overruled later in the same teacher update. In that
    case its rankings cannot supervise the issued motion. Unscored alternatives
    and equal-cost alternatives contribute zero.
    """
    if costs.shape != (len(labels), PLAN_ACTIONS) or choices.shape != (len(labels),):
        raise ValueError("Development preferences and teacher keys must align")
    flags = labels[:, 2]
    issued = (labels[:, 0] + 1) * 4 + 2 * ((flags & 3) != 0) + ((flags & 4) != 0)
    chosen = costs[np.arange(len(issued)), issued]
    allowed = np.isfinite(costs) & (np.isfinite(chosen) & (choices == issued))[:, None]
    difference = np.zeros_like(costs, dtype="f4")
    np.subtract(costs, chosen[:, None], out=difference, where=allowed)
    # A cost difference of four receives full weight. Tiny ordering changes in
    # nearly tied trajectories receive proportionally little influence.
    return np.clip(difference / 4, 0, 1)


def read_plans(path: Path, episode: int, times: np.ndarray) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        if struct.unpack("<IQi", f.read(16)) != (0x31504B48, episode, PLAN_ACTIONS):
            raise ValueError("Invalid development plan header")
        rows = np.frombuffer(f.read(), dtype=PLAN_ROW)
    clock = rows["time"]
    indices = np.searchsorted(times, clock)
    if (
        not len(rows)
        or not np.isfinite(clock).all()
        or (np.diff(clock) <= 0).any()
        or (indices == len(times)).any()
        or not np.array_equal(times[indices], clock)
    ):
        raise ValueError("Plan timestamps do not match native decisions")
    costs = rows["cost"]
    choice = rows["choice"]
    if (
        np.isnan(costs).any()
        or np.isneginf(costs).any()
        or ((choice < 0) | (choice >= PLAN_ACTIONS)).any()
    ):
        raise ValueError("Invalid development plan labels")
    best = costs.min(axis=1)
    known = np.isfinite(best)
    if (costs[np.arange(len(rows)), choice][known] != best[known]).any():
        raise ValueError("Selected plan cost differs from the best recorded option")
    aligned = np.full((len(times), PLAN_ACTIONS), np.inf, dtype="f4")
    aligned[indices] = costs
    selected = np.full(len(times), -1, dtype="i1")
    selected[indices[known]] = choice[known]
    return {"plan_costs": aligned, "plan_choice": selected}

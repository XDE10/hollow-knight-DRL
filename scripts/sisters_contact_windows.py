"""Build complete, controller-consistent contact windows on development trajectories."""

from __future__ import annotations

import numpy as np


def contact_windows(
    times: np.ndarray,
    events: np.ndarray,
    boundaries: np.ndarray,
    seconds: float = 0.32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eligibility and per-source contact occurrence from the issued input onward.

    The last included transition ends at the first observation at or beyond the
    requested horizon. A terminal transition is absorbing. Missing observations or
    a controller handoff invalidate an unfinished window regardless of its events.
    """
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    if times.ndim != 1 or events.shape != (n, 8) or n < 2:
        raise ValueError("Expected an aligned, nonempty trajectory")
    if (
        not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
        or not np.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("Expected increasing clocks and a positive horizon")
    cuts = np.asarray(boundaries)
    if (
        cuts.ndim != 1
        or cuts.dtype.kind not in "iu"
        or (np.diff(cuts.astype(np.int64)) <= 0).any()
    ):
        raise ValueError("Expected ordered controller boundaries")
    if len(cuts) and (cuts[0] < 0 or cuts[-1] >= n):
        raise ValueError("Controller boundary outside the game")

    row = np.arange(n)
    terminal = np.flatnonzero(events[:, 6:].any(1))
    terminal_end = int(terminal[0] + 1) if len(terminal) else n + 1
    # End indices are exclusive, so a handoff exactly at the end is allowed.
    required = np.minimum(np.searchsorted(times, times + seconds - 1e-5), terminal_end)
    next_cut = np.r_[cuts, n][np.searchsorted(cuts, row, side="right")]
    missing = np.r_[np.diff(times) > 0.081, terminal_end != n]
    next_gap = np.minimum.accumulate(np.where(missing, row, n)[::-1])[::-1]
    valid = (row < terminal_end) & (required <= next_cut) & (required <= next_gap)
    totals = np.vstack((np.zeros((1, 4)), np.cumsum(events[:, 1:5], axis=0)))
    return valid, totals[np.minimum(required, n)] > totals[row]

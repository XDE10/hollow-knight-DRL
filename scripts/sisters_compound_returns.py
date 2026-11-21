"""Observed returns after isolated randomized compound actions in a Sisters fight."""

from __future__ import annotations

import numpy as np
from train_sisters_native_iql import REWARD_WEIGHTS


def observed_outcomes(
    times: np.ndarray,
    state: np.ndarray,
    rows: np.ndarray,
    starts: np.ndarray,
    elapsed: np.ndarray,
    events: np.ndarray,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rows with a measured reward and mask count through the horizon or terminal.

    Native HP changes include both observation margins. Event intervals wholly inside
    the window supply boss damage, terminal outcomes and source-specific mask costs.
    A death ends the native HP trace at zero even if the input clock stopped first.
    """
    ends = np.r_[starts[1:], starts[-1] + elapsed[-1] * 0.02].astype("f8")
    hp = np.rint(state[:, 4] * 9).astype(int)
    term = np.flatnonzero(events[:, 6:].any(1))
    if len(term) > 1 or (len(term) and term[0] != len(events) - 1):
        raise ValueError("Expected one final arena outcome")
    if (
        len(times) != len(hp)
        or np.any(np.diff(times) <= 0)
        or not np.allclose(hp, state[:, 4] * 9, atol=0.001)
    ):
        raise ValueError("Invalid native clock or starting health")
    kept, returns = [], []
    for index, row in enumerate(rows):
        draw = float(times[row])
        end = draw + horizon
        a = np.searchsorted(starts, draw - 0.00001)
        b = np.searchsorted(ends, end + 0.00001, side="right")
        if a >= b or (ends[-1] < end - 0.00001 and not len(term)):
            continue
        terminal = bool(len(term) and a <= term[0] < b)
        observed_until = ends[-1] if terminal else end
        last = np.searchsorted(times, observed_until + 0.00001, side="right")
        if last <= row:
            continue
        native_hp = hp[row:last]
        debits = np.maximum(native_hp[:-1] - native_hp[1:], 0)
        lost = int(debits.sum())
        # All masks left at the final input are lost at a confirmed player death.
        if terminal and events[-1, 7] > 0:
            lost += int(native_hp[-1])
        captured = int(events[a:b, 1:5].sum())
        discount = 0.995 ** ((ends[a:b] - draw) / 0.04)
        reward = float(np.dot(discount, events[a:b] @ REWARD_WEIGHTS))
        extra = max(lost - captured, 0)
        reward += (
            extra
            * float(np.max(REWARD_WEIGHTS[1:5]))
            * 0.995 ** ((observed_until - draw) / 0.04)
        )
        returns.append(
            (
                reward,
                max(lost, captured),
                events[a:b, 0].sum(),
                int(terminal),
                int(np.diff(times[row:last]).max(initial=0) > 0.081),
            )
        )
        kept.append(index)
    return np.asarray(kept, dtype="i8"), np.asarray(returns, dtype="f8").reshape(-1, 5)

"""Read randomized native controller blocks and their observed development returns."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def neural_block_starts(clock: np.ndarray) -> np.ndarray:
    if (
        clock.ndim != 1
        or not len(clock)
        or not np.isfinite(clock).all()
        or (np.diff(clock) <= 0).any()
    ):
        raise ValueError("Invalid neural block clock")
    starts = [0]
    while True:
        boundary = np.float32(clock[starts[-1]]) + np.float32(0.8)
        after = int(np.searchsorted(clock, boundary))
        if after >= len(clock):
            break
        starts.append(after)
    return np.asarray(starts, dtype="i4")


def neural_assignments(
    path: Path, episode: int, clock: np.ndarray, parent_hash: str, candidate_hash: str
) -> tuple[np.ndarray, np.ndarray]:
    """Verify the two deployed files and the actual random draws against the native clock."""
    header = struct.Struct("<IQ32s32s")
    with path.open("rb") as stream:
        if header.unpack(stream.read(header.size)) != (
            0x314D5241,
            episode,
            bytes.fromhex(parent_hash),
            bytes.fromhex(candidate_hash),
        ):
            raise ValueError("Unexpected neural block trial receipt")
        draws = np.frombuffer(
            stream.read(), dtype=[("time", "<f4"), ("candidate", "u1")]
        )
    starts = neural_block_starts(clock)
    if (
        not np.array_equal(draws["time"], clock[starts])
        or not np.isin(draws["candidate"], [0, 1]).all()
    ):
        raise ValueError("Actual controller draws do not match the registered clock")
    lengths = np.diff(np.r_[starts, len(clock)])
    return starts, np.repeat(draws["candidate"].astype(bool), lengths)


@dataclass(frozen=True)
class ControlBlock:
    start: int
    stop: int
    teacher: bool
    reward: float
    discount: float


def control_blocks(
    times: np.ndarray,
    assigned: np.ndarray,
    trio: np.ndarray,
    interval_starts: np.ndarray,
    elapsed_ticks: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
) -> list[ControlBlock]:
    """Match the recorder's float32 0.8 s clock, including repeated draws of the same arm."""
    if (
        not len(times)
        or assigned.shape != times.shape
        or trio.shape != times.shape
        or assigned.dtype != np.bool_
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not len(interval_starts)
        or elapsed_ticks.shape != interval_starts.shape
        or np.any(np.diff(interval_starts) <= 0)
        or np.any(elapsed_ticks <= 0)
        or events.shape != (len(interval_starts), 8)
        or weights.shape != (8,)
    ):
        raise ValueError("Inconsistent controller or outcome trace")
    ends = np.r_[interval_starts[1:], interval_starts[-1] + elapsed_ticks[-1] * 0.02]
    payoff = events @ weights
    terminal = (events[:, 6:] > 0).any(1)
    blocks = []
    i = 0
    while i < len(times):
        boundary = np.float32(times[i]) + np.float32(0.8)
        k = int(np.searchsorted(times, boundary))
        if k <= i or (assigned[i:k] != assigned[i]).any():
            raise ValueError("Controller changed within its randomized block")
        # Before the next draw the previous controller owns the actually applied keys.
        end = (
            float(times[k]) if k < len(times) else min(float(boundary), float(ends[-1]))
        )
        first = np.searchsorted(interval_starts, float(times[i]) - 1e-5)
        last = np.searchsorted(ends, end + 1e-5, side="right")
        if trio[i] and first < last:
            term = np.flatnonzero(terminal[first:last])
            if len(term):
                last = first + int(term[0]) + 1
            if len(term) or k < len(times):
                delays = np.maximum(ends[first:last] - float(times[i]), 0)
                reward = float(np.dot(0.995 ** (delays / 0.04), payoff[first:last]))
                continuation = (
                    0.0 if len(term) else float(0.995 ** ((end - times[i]) / 0.04))
                )
                stop = (
                    min(k, int(np.searchsorted(times, ends[last - 1], side="left")))
                    if len(term)
                    else k
                )
                if stop > i:
                    blocks.append(
                        ControlBlock(i, stop, bool(assigned[i]), reward, continuation)
                    )
        i = k
    return blocks

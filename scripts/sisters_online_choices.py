"""Read randomized native inputs and measure outcomes on the observed trajectory."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

CHOICE_KEYS = (
    "choice_rows",
    "choice_pairs",
    "choice_draw",
    "choice_probability",
    "choice_gap",
)
ROW = np.dtype(
    [("time", "<f4"), ("pair", "<i4", 2), ("draw", "<i4"), ("p", "<f4"), ("gap", "<f4")]
)


def late_catalogue(checkpoint: bytes) -> np.ndarray:
    if struct.unpack_from("<i", checkpoint)[0] == 0x334C5149:
        _, opening, late = struct.unpack_from("<3i", checkpoint)
        if min(opening, late) < 20 or 12 + opening + late != len(checkpoint):
            raise ValueError("Incomplete phase actor")
        checkpoint = checkpoint[12 + opening :]
    magic, width, first, second, count = struct.unpack_from("<5i", checkpoint)
    if magic not in (0x324C5149, 0x344C5149) or (width, first, second) != (
        192,
        256,
        192,
    ):
        raise ValueError("Expected a timing actor")
    if not 1 <= count <= 4096:
        raise ValueError("Invalid native action catalogue")
    return np.frombuffer(checkpoint, "<i4", count=count * 3, offset=20).reshape(
        count, 3
    )


def read_choices(
    path: Path,
    episode: int,
    times: np.ndarray,
    state: np.ndarray,
    applied: np.ndarray,
    catalogue: np.ndarray,
    sampling_policy: bytes | None = None,
) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        magic, recorded_episode = struct.unpack("<IQ", f.read(12))
        learned = magic == 0x34434B48
        if (
            magic not in (0x31434B48, 0x32434B48, 0x33434B48, 0x34434B48)
            or recorded_episode != episode
        ):
            raise ValueError("Wrong online episode")
        if learned != (sampling_policy is not None):
            raise ValueError("Missing or unexpected sampled policy")
        interval = 8.4
        if sampling_policy is not None:
            (interval,) = struct.unpack("<f", f.read(4))
            head_magic, head_interval = struct.unpack_from("<if", sampling_policy)
            if (
                len(sampling_policy) != 8 + 386 * 4
                or head_magic != 0x31504753
                or interval != head_interval
                or not 0.31999 <= interval <= 8.40001
                or f.read(32) != hashlib.sha256(sampling_policy).digest()
            ):
                raise ValueError("Option sampler differed from the recorded behavior")
        compound = magic == 0x33434B48 or learned
        dtype = (
            np.dtype(ROW.descr + [("scores", "<f4", (2,)), ("risk", "<f4")])
            if compound
            else ROW
        )
        raw = np.frombuffer(f.read(), dtype=dtype)
    trial_time, pairs, draw, prob, gap = (
        raw[k] for k in ("time", "pair", "draw", "p", "gap")
    )
    descent = magic == 0x32434B48
    rows = np.searchsorted(times, trial_time)
    if (
        not np.isfinite(trial_time).all()
        or (
            np.diff(trial_time)
            < (interval - 0.001 if compound else 1.499 if descent else 0.719)
        ).any()
        or (rows >= len(times)).any()
        or not np.array_equal(times[rows], trial_time)
        or (state[rows, 23] <= 0.5).any()
        or ((pairs < 0) | (pairs >= len(catalogue))).any()
        or (pairs[:, 0] == pairs[:, 1]).any()
        or ((draw != 0) & (draw != 1)).any()
        or not np.isfinite(prob).all()
        or ((prob < 0.05) | (prob > 0.95)).any()
        or (not learned and (prob != 0.5).any())
        or not np.isfinite(gap).all()
        or (gap < 0).any()
        or (not compound and (gap > 3).any())
    ):
        raise ValueError("Invalid randomized action record")
    if not np.array_equal(catalogue[pairs[np.arange(len(draw)), draw]], applied[rows]):
        raise ValueError("Randomized control differs from the input actually issued")
    result = dict(zip(CHOICE_KEYS, (rows.astype("i4"), pairs, draw, prob, gap), strict=True))
    if compound:
        scores, risk = raw["scores"], raw["risk"]
        if (not np.isfinite(scores).all() or not np.isfinite(risk).all() or (risk < 0).any()
            or not np.allclose(scores[:,0]-scores[:,1], gap, rtol=0, atol=1e-5)
            or (state[rows,9] > .5).any()):
            raise ValueError("Invalid prospective risk or option scores")
        primitive = catalogue[pairs]
        bits = primitive[:,:,2]
        group = ((bits & 3) != 0).astype(int) + 2*((bits & 4) != 0) + 4*((bits & 16 != 0) & (primitive[:,:,1] < 0))
        if (group[:,0] == group[:,1]).any():
            raise ValueError("The two neural options have the same maneuver")
        result.update(choice_hold_stop=np.searchsorted(times, trial_time + np.float32(.32)).astype("i4"),
                      choice_scores=scores, choice_risk=risk)
    if descent:
        bosses = state[rows, 24:132].reshape(-1, 3, 36)
        near = (bosses[:, :, 0] > .5) & (np.abs(bosses[:, :, 1]) <= .4) & (bosses[:, :, 20:23] > .5).any(2)
        if not near.any(1).all() or (state[rows, 9] > .5).any() or (catalogue[pairs[:, 0], 0] == catalogue[pairs[:, 1], 0]).any():
            raise ValueError("Descent trial did not start on a supported horizontal disagreement")
        stops = np.searchsorted(times, trial_time + np.float32(.16))
        for row, end, key in zip(rows, stops, catalogue[pairs[np.arange(len(draw)), draw]], strict=True):
            if (applied[row:end, 0] != key[0]).any():
                raise ValueError("Recorded development movement differed during the hold")
        result["choice_hold_stop"] = stops.astype("i4")
    return result


def choice_outcomes(
    times: np.ndarray,
    starts: np.ndarray,
    ticks: np.ndarray,
    events: np.ndarray,
    rows: np.ndarray,
    reward_weights: np.ndarray,
    horizon: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use complete observation intervals before another randomized action is possible."""
    if not 0 < horizon < 0.72:
        raise ValueError("Outcome horizon must end before the next randomized action")
    ends = np.r_[starts[1:], starts[-1] + 0.02 * ticks[-1]]
    terminal = events[:, 6:].any(1)
    entered = np.searchsorted(starts, times[rows] - 1e-5, side="left")
    stopped = np.searchsorted(ends, times[rows] + horizon + 1e-5, side="right")
    recorded = events @ reward_weights
    kept, returns, successors, discounts = [], [], [], []
    for trial, (row, a, b) in enumerate(zip(rows, entered, stopped, strict=True)):
        if a >= b:
            continue
        stop = np.flatnonzero(terminal[a:b])
        if len(stop):
            b = a + int(stop[0]) + 1
        elif times[-1] < times[row] + horizon - 1e-5:
            continue
        end = ends[b - 1]
        last = int(np.searchsorted(times, end + 1e-5, side="right") - 1)
        if (
            last < row
            or end - times[last] > 0.24 + 1e-5
            or (np.diff(times[row : last + 1]) > 0.24 + 1e-5).any()
        ):
            continue
        # Do not award an event whose observation interval began before the draw.
        delays = np.maximum(0, ends[a:b] - times[row])
        returns.append(np.dot(0.995 ** (delays / 0.04), recorded[a:b]))
        discounts.append(0 if len(stop) else 0.995 ** ((end - times[row]) / 0.04))
        successors.append(last)
        kept.append(trial)
    return tuple(
        np.asarray(x, dtype=d)
        for x, d in zip(
            (kept, returns, successors, discounts),
            ("i8", "f4", "i8", "f4"),
            strict=True,
        )
    )

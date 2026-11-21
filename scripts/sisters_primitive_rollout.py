"""Native, local exploration opportunities for the champion's primitive key network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sisters_router_rollout import HEADER

ROW = np.dtype(
    [
        ("time", "<f4"),
        ("greedy", "<i4"),
        ("draw", "<i4"),
        ("uniform", "<f8"),
        ("p", "<f8"),
    ]
)


def onsets(clock, state):
    near = np.zeros(len(clock), bool)
    for k in (24, 60, 96, 132, 146):
        x, y = (0.48, 0.8) if k < 132 else (0.6, 0.42)
        near |= (
            (state[:, k] > 0.5)
            & (np.abs(state[:, k + 1]) < np.float32(x))
            & (np.abs(state[:, k + 2]) < np.float32(y))
        )
    ready = near & (state[:, 23] > 0.5) & (state[:, 9] < 0.5)
    rows = []
    deadline = np.float32(0)
    for j in np.flatnonzero(ready):
        if clock[j] >= deadline:
            rows.append(j)
            deadline = clock[j] + np.float32(0.4)
    return np.asarray(rows, dtype="i4")


def read_keys(path: Path, episode, behavior_hash, seed, temperature, clock, state, *, smooth=False):
    packet = path.read_bytes()
    if len(packet) < HEADER.size or HEADER.unpack_from(packet) != (
        0x32534B50 if smooth else 0x31534B50,
        episode,
        bytes.fromhex(behavior_hash),
        seed,
        temperature,
    ):
        raise ValueError("Wrong native primitive behavior, temperature or seed")
    d = np.frombuffer(packet, ROW, offset=HEADER.size)
    rows = onsets(clock, state)
    if (
        not np.array_equal(d["time"], clock[rows])
        or not np.isfinite(d["p"]).all()
        or not ((d["p"] > 0) & (d["p"] <= 1)).all()
        or not ((d["uniform"] >= 0) & (d["uniform"] < 1)).all()
        or (d["draw"] < 0).any()
        or (d["greedy"] < 0).any()
    ):
        raise ValueError("Invalid real primitive decision")
    return {
        "key_rows": rows,
        "key_greedy": d["greedy"].copy(),
        "key_draw": d["draw"].copy(),
        "key_uniform": d["uniform"].copy(),
        "key_probability": d["p"].copy(),
    }

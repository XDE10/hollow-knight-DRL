"""Decode real on-policy draws of the two frozen neural controllers."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from sisters_control_blocks import neural_block_starts

HEADER = struct.Struct("<IQ32sid")
ROW = np.dtype(
    [
        ("time", "<f4"),
        ("margin", "<f8"),
        ("p", "<f8"),
        ("uniform", "<f8"),
        ("draw", "u1"),
    ]
)


def read_routes(
    path: Path,
    episode: int,
    expected_hash: str,
    seed: int,
    temperature: float,
    clock: np.ndarray,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    packet = path.read_bytes()
    if len(packet) < HEADER.size or HEADER.unpack_from(packet) != (
        0x31525453,
        episode,
        bytes.fromhex(expected_hash),
        seed,
        temperature,
    ):
        raise ValueError("Router behavior, episode or seed differs from its receipt")
    rows = neural_block_starts(clock)
    rows = rows[states[rows, 23] > 0.5]
    tape = np.frombuffer(packet, ROW, offset=HEADER.size)
    p = 0.04 + 0.92 / (1 + np.exp(-np.clip(tape["margin"] / temperature, -50, 50)))
    if (
        not 0 < temperature < 1000
        or not np.array_equal(clock[rows], tape["time"])
        or not np.isfinite(tape["margin"]).all()
        or not np.isfinite(tape["uniform"]).all()
        or (tape["uniform"] < 0).any()
        or (tape["uniform"] >= 1).any()
        or not np.allclose(p, tape["p"], rtol=0, atol=1e-10)
        or not np.array_equal(tape["draw"], tape["uniform"] < p)
    ):
        raise ValueError(
            "Router rollout probabilities or assignments differ from the actual block clock"
        )
    return {
        "route_rows": rows,
        "route_probability": tape["p"].copy(),
        "route_draw": tape["draw"].copy(),
        "route_margin": tape["margin"].copy(),
        "route_uniform": tape["uniform"].copy(),
    }

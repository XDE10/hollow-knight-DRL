"""Native six-way maneuver proposals and their logged behavior probabilities."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

HEAD_MAGIC = 0x33504753
TAPE_MAGIC = 0x36434B48
HOLD_SECONDS = np.float32(0.12)
INTERVAL_SECONDS = 0.4
GROUPS = 6
MENU_KEYS = (
    "choice_rows",
    "menu_primary",
    "menu_draw",
    "menu_risk",
    "menu_slots",
    "menu_probability",
)
ROW = np.dtype(
    [
        ("time", "<f4"),
        ("primary", "<i4"),
        ("draw", "<i4"),
        ("risk", "<f4"),
        ("slots", "<i4", GROUPS),
        ("probability", "<f4", GROUPS),
    ]
)


def available_keys(catalogue: np.ndarray, state: np.ndarray) -> np.ndarray:
    bits = catalogue[:, 2]
    feasible = ((bits & 1) == 0) | (
        (state[:, 164:165] <= 0.5) & ((state[:, 7:8] >= 0.5) | (state[:, 8:9] >= 0.5))
    )
    feasible &= ((bits & 3) != 2) | (state[:, 165:166] >= 0.5)
    feasible &= ((bits & 4) == 0) | (
        (state[:, 13:14] >= 0.5) & (state[:, 166:167] <= 0.5)
    )
    feasible &= ((bits & 16) == 0) | (
        (state[:, 11:12] >= 0.5)
        & (state[:, 5:6] >= np.float32(33 / 99))
        & (state[:, 168:169] <= 0.5)
    )
    return feasible


def menu_slots(
    scores: np.ndarray, catalogue: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rank feasible keys within movement direction and same/changed maneuver family."""
    best = scores.argmax(1)
    bits = catalogue[:, 2]
    family = (
        ((bits & 3) != 0).astype(int)
        + 2 * ((bits & 4) != 0)
        + 4 * ((bits & 16 != 0) & (catalogue[:, 1] < 0))
    )
    direction = (catalogue[:, 0] + 1) * 2
    primary = direction[best]
    groups = direction + (family != family[best, None])
    feasible = available_keys(catalogue, state)
    slots = np.full((len(best), GROUPS), -1, dtype="i4")
    for g in range(GROUPS):
        eligible = feasible & (groups == g)
        slots[:, g] = np.where(
            eligible.any(1), np.where(eligible, scores, -np.inf).argmax(1), -1
        )
    slots[np.arange(len(best)), primary] = best
    return slots, primary


def menu_logits(
    hidden: np.ndarray, heads: np.ndarray, slots: np.ndarray, primary: np.ndarray
) -> np.ndarray:
    valid = slots >= 0
    logits = np.where(valid, hidden @ heads[:, 1:].T + heads[:, 0], -np.inf)
    logits[np.arange(len(primary)), primary] += np.log(
        3 * np.maximum(1, valid.sum(1) - 1)
    )
    return logits


def behavior_probabilities(logits: np.ndarray) -> np.ndarray:
    valid = np.isfinite(logits)
    exp = np.exp(logits - logits.max(1, keepdims=True))
    return 0.95 * exp / exp.sum(1, keepdims=True) + 0.05 * valid / valid.sum(
        1, keepdims=True
    )


def read_menu(
    path: Path,
    episode: int,
    times: np.ndarray,
    state: np.ndarray,
    applied: np.ndarray,
    catalogue: np.ndarray,
    sampling_policy: bytes,
) -> dict[str, np.ndarray]:
    packet = path.read_bytes()
    if len(packet) < 48 or len(sampling_policy) != 8 + 7 * 193 * 4:
        raise ValueError("Truncated directional rollout")
    magic, eid, interval = struct.unpack_from("<IQf", packet)
    marker, expected_interval = struct.unpack_from("<if", sampling_policy)
    if (
        magic != TAPE_MAGIC
        or episode != eid
        or marker != HEAD_MAGIC
        or interval != expected_interval
        or not 0.31999 <= interval <= 8.40001
        or packet[16:48] != hashlib.sha256(sampling_policy).digest()
    ):
        raise ValueError("Directional behavior differs from the registered weights")
    raw = np.frombuffer(packet, dtype=ROW, offset=48)
    t, primary, draw, risk, slots, prob = (raw[k] for k in ROW.names)
    rows = np.searchsorted(times, t)
    idx = np.arange(len(t))
    if (
        not np.isfinite(t).all()
        or (np.diff(t) < interval - 0.001).any()
        or (rows >= len(times)).any()
        or not np.array_equal(times[rows], t)
        or (state[rows, 23] < 0.5).any()
        or (state[rows, 9] > 0.5).any()
        or not np.isfinite(risk).all()
        or (risk < 0).any()
        or ((slots < -1) | (slots >= len(catalogue))).any()
        or not np.isin(primary, [0, 2, 4]).all()
        or ((draw < 0) | (draw >= GROUPS)).any()
        or (slots[idx, primary] < 0).any()
        or (slots[idx, draw] < 0).any()
        or ((slots >= 0).sum(1) < 2).any()
        or not np.isfinite(prob).all()
        or (prob < 0).any()
        or not np.array_equal(prob > 0, slots >= 0)
        or not np.allclose(prob.sum(1), 1, atol=1e-6, rtol=0)
        or not np.array_equal(catalogue[slots[idx, draw]], applied[rows])
    ):
        raise ValueError("Invalid directional rollout or issued key")
    data = dict(
        zip(
            MENU_KEYS,
            (rows.astype("i4"), primary, draw, risk, slots, prob),
            strict=True,
        )
    )
    data["choice_hold_stop"] = np.searchsorted(times, t + HOLD_SECONDS).astype("i4")
    return data

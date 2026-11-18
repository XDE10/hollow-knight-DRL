"""Offline outcome labels for the slot-3 dodge-and-counter demonstrations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from hkrl import protocol, spaces

DODGE_BUTTONS = ("jump_tap", "dash")


def dodge_outcomes(data: Mapping[str, np.ndarray], record: dict[str, Any]) -> np.ndarray:
    """Grade a requested dodge from observed motion, actual injuries and verified nail hits."""
    player, actions = data["player"], data["actions"]
    n = len(actions)
    ticks = np.asarray(data["elapsed_ticks"], dtype=np.int64)
    if (
        not n
        or record["steps"] != n
        or ticks.shape != (n,)
        or len(player) != n
        or np.any(ticks <= 0)
    ):
        raise ValueError("invalid episode length or tick alignment")
    end = ticks.cumsum()
    start = np.r_[0, end[:-1]]
    if any(not 1 <= event["step"] <= n for event in (*record["contacts"], *record["hits"])):
        raise ValueError("event step outside demonstration episode")
    hurt = end[[event["step"] - 1 for event in record["contacts"] if event["amount"] > 0]]
    hits = end[[hit["step"] - 1 for hit in record["hits"]]]
    p = spaces.PLAYER_FEATURE_INDEX
    motion = (
        (player[:, p["jumping"]] > 0)
        | ((player[:, p["on_ground"]] == 0) & (player[:, p["vel_y"]] > 0)),
        (player[:, p["dashing"]] > 0) | (player[:, p["shadow_dashing"]] > 0),
    )
    grades = np.zeros((n, len(DODGE_BUTTONS)), dtype=np.int8)
    for column, name in enumerate(DODGE_BUTTONS):
        requests = np.flatnonzero(actions[:, 2 + spaces.BUTTON_BITS[name]])
        for index, i in enumerate(requests):
            grades[i, column] = -1
            t = start[i]
            if t + 18 > end[-1] or player[i, p["invulnerable"]]:
                continue
            if np.any((hurt > t - 10) & (hurt <= t + 18)):
                continue
            # The next request's pre-action observation still belongs to this request.
            stop = min(n, int(np.searchsorted(start, t + 12, side="right")))
            if index + 1 < len(requests):
                stop = min(stop, requests[index + 1] + 1)
            if motion[column][i] or not np.any(motion[column][i + 1 : stop]):
                continue
            later = hits[(hits > t) & (hits <= t + 50)]
            counter = bool(len(later) and not np.any((hurt > t) & (hurt <= later[0])))
            grades[i, column] = 2 if counter else 1
    return grades


def safe_nail_windows(data: Mapping[str, np.ndarray], record: dict[str, Any]) -> np.ndarray:
    """Mark the approach and slash for verified hits with a fully observed injury-free interval."""
    end = np.cumsum(data["elapsed_ticks"], dtype=np.int64)
    start = np.r_[0, end[:-1]]
    mask = np.zeros(len(end), dtype=bool)
    injury = end[[event["step"] - 1 for event in record["contacts"] if event["amount"] > 0]]
    for hit in record["hits"]:
        slash, contact = hit["slash_step"] - 1, hit["step"] - 1
        if not 0 <= slash <= contact < len(end):
            raise ValueError("invalid slash or contact step")
        lo, hi = start[slash] - 12, end[contact] + 12
        if lo < 0 or hi > end[-1] or np.any((injury >= lo) & (injury <= hi)):
            continue
        begin = int(np.searchsorted(start, lo))
        if not data["player"][begin : slash + 1, spaces.PLAYER_FEATURE_INDEX["invulnerable"]].any():
            mask[begin : slash + 1] = True
    return mask


def landed_counter_labels(
    data: Mapping[str, np.ndarray], record: dict[str, Any], landed_state: float
) -> tuple[np.ndarray, np.ndarray]:
    """Select grounded side slashes on the landed sister from the collector's verified hits."""
    actions = data["actions"]
    windows = np.zeros(len(actions), dtype=bool)
    # Sign is the target's side at the slash decision. Zero means unlabelled.
    sides = np.zeros(len(actions), dtype=np.int8)
    e = spaces.ENTITY_FEATURE_INDEX
    for hit in record["hits"]:
        step = hit["slash_step"] - 1
        if not 0 <= step < len(actions):
            raise ValueError("invalid slash step")
        action = actions[step]
        if not (
            data["player"][step, spaces.PLAYER_FEATURE_INDEX["on_ground"]]
            and action[1] == 1
            and action[2 + spaces.BUTTON_BITS["attack"]]
        ):
            continue
        rows = data["entities"][step, data["entity_mask"][step].astype(bool)]
        matched = rows[
            (rows[:, e["entity_id"]] == hit["target_id"])
            & (rows[:, e["entity_type"]] == protocol.EntityType.BOSS)
            & (rows[:, e["fsm_state_hash"]] == landed_state)
        ]
        if len(matched) != 1:
            continue
        side = np.sign(matched[0, e["rel_x"]])
        if not side or side != np.sign(hit["dx_at_slash"]):
            continue
        window = safe_nail_windows(data, {"hits": [hit], "contacts": record["contacts"]})
        if window[step]:
            windows |= window
            sides[step] = side
    return windows, sides

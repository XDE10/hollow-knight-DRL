from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from hkrl import protocol, spaces

batch_examples = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "probe_sisters_replay.py")
)["batch_examples"]


def batch() -> dict[str, np.ndarray]:
    ticks, workers = 18, 2
    shape = (ticks, workers)
    entities = np.zeros((*shape, 2, spaces.ENTITY_FEATURE_DIMS["privileged"]), np.float32)
    e = spaces.ENTITY_FEATURE_INDEX
    ents = entities[:, :, 1]
    ents[..., e["entity_type"]] = protocol.EntityType.BOSS
    ents[..., e["entity_id"]] = [101, 102]
    ents[..., e["hp"]] = ents[..., e["max_hp"]] = 1
    player = np.zeros((*shape, spaces.PLAYER_FEATURE_DIMS["privileged"]), np.float32)
    player[..., spaces.PLAYER_FEATURE_INDEX["hp"]] = 1
    g = np.zeros((*shape, spaces.GLOBAL_FEATURE_DIM), np.float32)
    g[..., spaces.GLOBAL_FEATURE_INDEX["time_in_episode"]] = np.arange(ticks)[:, None]
    return {
        "obs_entities": entities,
        "obs_global": g,
        "obs_player": player,
        "entity_mask": entities[..., e["entity_type"]] != 0,
        "actions": np.zeros((*shape, 13)),
        "episode_ids": np.ones(shape),
        "dones": np.zeros(shape, bool),
        "truncateds": np.zeros(shape, bool),
    }


def test_future_drops_are_labels_only_and_stay_in_worker() -> None:
    data = batch()
    x, _ = batch_examples(data, 15)
    data["obs_player"][15:, 0, spaces.PLAYER_FEATURE_INDEX["hp"]] = 0.5
    data["obs_entities"][16:, 1, 1, spaces.ENTITY_FEATURE_INDEX["hp"]] = 0.75
    new_x, y = batch_examples(data, 15)
    np.testing.assert_array_equal(new_x, x)
    assert y.tolist() == [
        [True, False],
        [False, False],
        [True, False],
        [False, True],
        [True, False],
        [False, True],
    ]


@pytest.mark.parametrize(
    "boundary", ["dones", "truncateds", "reset", "replaced", "masked", "clock"]
)
def test_excludes_windows_spanning_discontinuities(boundary: str) -> None:
    data = batch()
    if boundary in ("dones", "truncateds"):
        data[boundary][8, 1] = True
    elif boundary == "reset":
        data["episode_ids"][9:, 1] += 1
    elif boundary == "replaced":
        data["obs_entities"][9:, 1, 1, spaces.ENTITY_FEATURE_INDEX["entity_id"]] += 1
    elif boundary == "masked":
        data["entity_mask"][8, 1, 1] = False
    else:
        data["obs_global"][9, 1, spaces.GLOBAL_FEATURE_INDEX["time_in_episode"]] = 0
    x, y = batch_examples(data, 15)
    assert len(x) == len(y) == 3

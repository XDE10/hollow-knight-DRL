"""Pack PPO batches into a small single-sister dataset for offline dynamics experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from probe_sisters_replay import decision_rows

from hkrl import spaces


def packed_state(geometry: np.ndarray, boss: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    e = spaces.ENTITY_FEATURE_INDEX
    # The first four columns are normalized absolute hero and boss hurtbox XY.
    xy = np.concatenate(
        (geometry[..., :2], boss[..., [e["hurtbox_center_x"], e["hurtbox_center_y"]]]), axis=-1
    )
    return np.concatenate((xy, geometry[..., 2:], elapsed[..., None]), axis=-1)


def pack(directory: Path, output: Path) -> None:
    fields = (
        "obs_player",
        "obs_entities",
        "obs_global",
        "entity_mask",
        "episode_ids",
        "dones",
        "truncateds",
        "actions",
        "action_masks",
    )
    parts: dict[str, list[np.ndarray]] = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in fields}
        geometry, hp, links, boss = decision_rows(data)
        times = data["obs_global"][..., spaces.GLOBAL_FEATURE_INDEX["time_in_episode"]]
        elapsed = np.zeros_like(times)
        elapsed[1:] = np.where(links, times[1:] - times[:-1], 0)
        e = spaces.ENTITY_FEATURE_INDEX
        state = packed_state(geometry, boss, elapsed)
        valid = np.zeros_like(times, dtype=bool)
        valid[:-1] = links
        block = dict(
            state=state.astype(np.float32),
            hp=hp,
            valid=valid,
            fsm=boss[..., e["fsm_state_hash"]].astype(np.int64),
            actions=data["actions"].astype(np.uint8),
            mask=data["action_masks"],
            terminated=data["dones"].astype(bool) & ~data["truncateds"].astype(bool),
        )
        for key, value in block.items():
            worker_major = value.swapaxes(0, 1)
            parts.setdefault(key, []).append(worker_major.reshape(-1, *value.shape[2:]))
    if not parts:
        raise ValueError(f"No rollout batches in {directory}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        np.savez_compressed(f, **{key: np.concatenate(v) for key, v in parts.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pack(args.directory, args.output)

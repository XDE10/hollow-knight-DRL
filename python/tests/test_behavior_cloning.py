from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def test_cloning_clips_supervise_each_frame_once_within_episode() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "clone_sisters_policy.py"
    spec = importlib.util.spec_from_file_location("clone_sisters_policy", path)
    assert spec and spec.loader
    clone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clone)

    episodes = []
    for start in (0, 100):
        values = torch.arange(start, start + 7)
        episodes.append({key: values.clone() for key in (*clone.OBS, "actions", "masks", "weight")})
    clips = clone.padded_clips(episodes, length=3, burn=2)
    assert torch.cat([c["actions"][c["loss_mask"]] for c in clips]).tolist() == [
        *range(7),
        *range(100, 107),
    ]
    assert clips[1]["actions"][:2].tolist() == [1, 2]
    assert not clips[1]["loss_mask"][:2].any()
    for clip in clips:
        assert clip["actions"].max() - clip["actions"].min() < 7

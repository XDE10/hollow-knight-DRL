from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from hkrl import protocol


def test_only_a_live_arena_completion_counts(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_sisters_tactics.py"
    spec = importlib.util.spec_from_file_location("eval_sisters_tactics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = tmp_path / "model.txt"
    policy.write_text("mode=0\n")
    marker = Path(str(policy) + ".active")
    marker.write_text("12\nmode=0\n")
    entities = np.zeros((1, 24))
    entities[0, [1, 12, 13]] = [protocol.EntityType.BOSS, 500, 500]
    player = np.zeros(32)
    player[4:6] = 9
    obs = {"player": player, "entities": entities, "entity_mask": np.ones(1, dtype=bool)}
    env = Mock()
    env.task.action.enable_macro_actions = True
    env.reset.return_value = obs, {"episode_id": 12}
    event = SimpleNamespace(kind=protocol.RewardEventKind.BOSS_KILLED, entity_id=5)
    info = {"episode_id": 12, "reward_events": [event], "elapsed_ticks": 5}
    env.step.return_value = obs, 0.0, True, False, info
    assert not module.play_episode(env, policy, tmp_path, 1)["win"]
    event.entity_id = 0
    assert module.play_episode(env, policy, tmp_path, 2)["win"]
    env.step.return_value = obs, 0.0, False, True, info
    assert not module.play_episode(env, policy, tmp_path, 3)["win"]
    marker.write_text("11\nmode=0\n")
    with pytest.raises(RuntimeError, match="different active episode"):
        module.play_episode(env, policy, tmp_path, 4)

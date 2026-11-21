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
    global_state = np.zeros(6)
    global_state[5] = 1.0
    obs = {
        "player": player,
        "entities": entities,
        "entity_mask": np.ones(1, dtype=bool),
        "global": global_state,
    }
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
    marker.write_text("12\nmode=0\n")
    (tmp_path / "world.weights").touch()
    usage = Path(str(policy) + ".usage")
    usage.write_text("12 90 4")
    result = module.play_episode(env, policy, tmp_path, 5)
    assert (result["planner_decisions"], result["world_changed_plan"]) == (90, 4)
    assert result["speed_counts"] == {1.0: 1} and result["minimum_hp"] == 9
    usage.write_text("11 90 4")
    with pytest.raises(RuntimeError, match="usage"):
        module.play_episode(env, policy, tmp_path, 6)


def test_second_phase_damage_and_health_use_observed_intervals() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_sisters_tactics.py"
    spec = importlib.util.spec_from_file_location("eval_sisters_telemetry", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def snapshot(hp: int, t: float, trio: bool) -> dict[str, np.ndarray]:
        b = np.zeros((3 if trio else 1, 24))
        b[:, 0] = np.arange(1, len(b) + 1)
        b[:, 1] = protocol.EntityType.BOSS
        b[:, 12:14] = 750 if trio else 500
        p = np.zeros(32)
        p[4:7] = [hp, 9, 0]
        p[13] = 1 if t == 1 else 0
        p[26] = 32 if t >= 1 else 0
        return {
            "player": p,
            "entities": b,
            "entity_mask": np.ones(len(b), bool),
            "global": np.array([0, 0, 0, 0, t, 1]),
        }

    def info(*events) -> dict:
        return {"episode_id": 7, "reward_events": events, "elapsed_ticks": 5}

    k = protocol.RewardEventKind
    incoming = protocol.RewardEvent(
        kind=k.DAMAGE_TAKEN, entity_id=11, aux_int=2,
        aux_int2=protocol.DamageSource.ENEMY_PROJECTILE, amount=1,
    )
    outgoing = SimpleNamespace(kind=k.DAMAGE_DEALT, amount=21)
    heal = SimpleNamespace(kind=k.HEAL, amount=1)
    env = Mock()
    env.task.action.enable_macro_actions = False
    before = snapshot(9, 0, False)
    # Two live enemy discs, a player spell, and a masked slot. Disc 11
    # despawns between the two observations of its damaging contact.
    projectiles = np.zeros((4, 24))
    projectiles[:, 0] = [10, 11, 12, 13]
    projectiles[:, 1] = protocol.EntityType.PROJECTILE
    projectiles[:, 2] = [protocol.Team.ENEMY, protocol.Team.ENEMY,
                         protocol.Team.PLAYER_CREATED, protocol.Team.ENEMY]
    projectiles[1, 6:8] = [23, 10]
    before["entities"] = np.vstack([before["entities"], projectiles])
    before["entity_mask"] = np.array([1, 1, 1, 1, 0], dtype=bool)
    env.reset.return_value = before, info()
    env.step.side_effect = [
        (snapshot(8, 1, True), 0, False, False, info(incoming, outgoing)),
        (snapshot(9, 2, True), 0, False, True, info(heal, outgoing)),
    ]
    game = module.play_game(env, 77, record_skill_use=True)
    assert game["sampled_skill_use"]["opening"]["observed_quake_starts"] == 1
    assert game["sampled_skill_use"]["opening"]["observed_health_gained"] == 0
    assert not game["sampled_skill_use"]["trio"]
    assert game["first_cleared"] and not game["win"]
    assert game["trio_entry"] == {"game_time": 1, "physics_ticks": 5, "hp": 8, "soul": 0}
    assert game["damage"] == 42 and game["damage_after_trio_observed"] == 21
    assert game["heal_event_amount"] == 1 and game["observed_dash_starts"] == {"dashing": 1}
    assert game["last_observed_trio"]["observed_hp_by_entity"] == {"1": 750, "2": 750, "3": 750}
    assert game["hit_intervals"][0]["source"] == "ENEMY_PROJECTILE"
    assert not game["hit_intervals"][0]["trio_already_observed"]
    hit = game["hit_intervals"][0]
    assert (hit["source_entity_id"], hit["hazard_type"]) == (11, 2)
    assert [e["entity_id"] for e in hit["entities_before_interval"]] == [1, 10, 11]
    assert hit["entities_before_interval"][-1]["position"] == [23, 10]
    assert [e["entity_id"] for e in hit["entities_after_interval"]] == [1, 2, 3]

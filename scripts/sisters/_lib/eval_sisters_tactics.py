"""Play the frozen Sisters movement checkpoint using its opt-in local game actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from hkrl import protocol, spaces
from hkrl.env import HKRLEnv
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config


def verify_checkpoint(directory: Path, manifest: dict[str, Any]) -> None:
    for name, expected in (
        ("model.txt", manifest["policy_sha256"]),
        ("head.weights", manifest.get("head_sha256")),
        ("world.weights", manifest.get("world_sha256")),
    ):
        if expected is not None:
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Checkpoint digest differs for {name}")


def verify_actor(policy_file: Path, episode: int, checkpoint: Path) -> None:
    active = Path(str(policy_file) + ".active")
    identity, _, parameters = active.read_text(encoding="utf-8-sig").partition("\n")
    if int(identity) != episode or parameters != (checkpoint / "model.txt").read_text():
        raise RuntimeError("The game has a different active episode or policy.")


def contact_entities(visible: np.ndarray) -> list[dict[str, Any]]:
    """Capture the observed threats at one endpoint of a damage interval."""
    e = spaces.ENTITY_FEATURE_INDEX
    result = []
    for row in visible:
        kind = int(row[e["entity_type"]])
        if kind not in (protocol.EntityType.BOSS, protocol.EntityType.HAZARD) and not (
            kind == protocol.EntityType.PROJECTILE and row[e["team"]] == protocol.Team.ENEMY
        ):
            continue
        item = {
            "entity_id": int(row[e["entity_id"]]),
            "type": protocol.EntityType(kind).name,
            "position": row[e["pos_x"] : e["pos_y"] + 1].tolist(),
            "velocity": row[e["vel_x"] : e["vel_y"] + 1].tolist(),
            "observed_fsm_state_hash": int(row[e["fsm_state_hash"]]),
        }
        if len(row) > e["hurtbox_size_y"]:
            item["hurtbox_center"] = row[e["hurtbox_center_x"] : e["hurtbox_center_y"] + 1].tolist()
            item["hurtbox_size"] = row[e["hurtbox_size_x"] : e["hurtbox_size_y"] + 1].tolist()
        result.append(item)
    return result


def play_game(
    env: HKRLEnv,
    seed: int,
    *,
    scale: float = 1.0,
    on_transition: Callable[[float, float, dict], None] | None = None,
    record_skill_use: bool = False,
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed, options={"reset_timeout_s": 90})
    episode = int(info["episode_id"])
    if list(obs["player"][4:7]) != [9, 9, 0]:
        raise RuntimeError(
            f"Expected an original nine-mask start with empty SOUL, got {list(obs['player'][4:7])}."
        )
    env.set_timescale(scale)
    damage = 0.0
    losses: Counter[str] = Counter()
    ticks = 0
    peak = 0
    arena_complete = False
    minimum_hp = 9
    trio_entry: dict[str, Any] | None = None
    hits: list[dict[str, Any]] = []
    trio_damage = 0.0
    heals = 0.0
    observed_dashes: Counter[str] = Counter()
    skill_use = {"opening": Counter(), "trio": Counter()}
    last_trio: dict[str, Any] | None = None
    speeds: Counter[float] = Counter()
    noop: dict[str, Any] = {
        "movement_x": 1,
        "aim_y": 1,
        "buttons": [0] * spaces.N_BUTTONS,
        "duration": 0,
    }
    if env.task.action.enable_macro_actions:
        noop["macro"] = 0
    while True:
        visible = obs["entities"][obs["entity_mask"].astype(bool)]
        bosses = visible[visible[:, 1] == protocol.EntityType.BOSS]
        peak = max(peak, len(bosses))
        if ticks == 0 and sorted(int(b[13]) for b in bosses) != [500]:
            raise RuntimeError("Unexpected Sisters opening health.")
        if len(bosses) == 3 and sorted(int(b[13]) for b in bosses) != [750] * 3:
            raise RuntimeError("Unexpected Sisters second-phase health.")
        previous_health = int(obs["player"][4])
        previous_time = float(obs["global"][4])
        phase_bosses = bosses[bosses[:, 13] == 750]
        if len(phase_bosses):
            last_trio = {
                "game_time": previous_time,
                "observed_hp_by_entity": {
                    str(int(b[0])): int(b[12]) for b in phase_bosses
                },
            }
        if trio_entry is None and len(bosses) >= 3:
            trio_entry = {
                "game_time": previous_time,
                "physics_ticks": ticks,
                "hp": previous_health,
                "soul": int(obs["player"][6]),
            }
        position = [float(obs["player"][j]) for j in (0, 1)]
        previous_dash = [
            bool(obs["player"][spaces.PLAYER_FEATURE_INDEX[k]])
            for k in ("dashing", "shadow_dashing")
        ]
        before_player = obs["player"]
        obs, _, terminated, truncated, info = env.step(noop)
        if record_skill_use and not (terminated or truncated):
            # The game can refill health as the arena closes. Only compare live samples.
            phase = skill_use["trio" if trio_entry is not None else "opening"]
            after_player = obs["player"]
            flags = spaces.PLAYER_FEATURE_INDEX["action_flags"]
            quake = 1 << spaces.PLAYER_ACTION_FLAG_BITS["spell_quake"]
            phase["observed_quake_starts"] += int(
                not (int(before_player[flags]) & quake) and bool(int(after_player[flags]) & quake)
            )
            phase["observed_health_gained"] += max(0, int(after_player[4]) - previous_health)
            phase["observed_soul_spent_net"] += max(0, int(before_player[6]) - int(after_player[6]))
        for before, name in zip(
            previous_dash, ("dashing", "shadow_dashing"), strict=True
        ):
            if not before and obs["player"][spaces.PLAYER_FEATURE_INDEX[name]]:
                observed_dashes[name] += 1
        if on_transition is not None:
            on_transition(previous_time, float(obs["global"][4]), info)
        minimum_hp = min(minimum_hp, previous_health, int(obs["player"][4]))
        speeds[round(float(obs["global"][5]), 3)] += 1
        if int(info["episode_id"]) != episode:
            raise RuntimeError("The game changed episode during the fight.")
        ticks += int(info["elapsed_ticks"])
        for event in info["reward_events"]:
            if event.kind == protocol.RewardEventKind.DAMAGE_DEALT:
                damage += event.amount
                if trio_entry is not None:
                    trio_damage += event.amount
            elif event.kind == protocol.RewardEventKind.DAMAGE_TAKEN:
                source = protocol.DamageSource(event.aux_int2).name
                losses[source] += int(event.amount)
                hits.append(
                    {
                        "interval_start": previous_time,
                        "interval_end": float(obs["global"][4]),
                        "trio_already_observed": trio_entry is not None,
                        "source": source,
                        "source_entity_id": event.entity_id,
                        "hazard_type": event.aux_int,
                        "masks": int(event.amount),
                        "hp_before_interval": previous_health,
                        "hp_after_interval": int(obs["player"][4]),
                        "position_before_interval": position,
                        "position_after_interval": obs["player"][:2].tolist(),
                        "entities_before_interval": contact_entities(visible),
                        "entities_after_interval": contact_entities(
                            obs["entities"][obs["entity_mask"].astype(bool)]
                        ),
                    }
                )
            elif event.kind == protocol.RewardEventKind.HEAL:
                heals += event.amount
            elif (
                event.kind == protocol.RewardEventKind.BOSS_KILLED
                and event.entity_id == 0
            ):
                arena_complete = True
        if terminated or truncated:
            break
    won = bool(
        arena_complete
        and terminated
        and not truncated
        and previous_health > 0
        and obs["player"][4] > 0
    )
    return {
        "episode_id": episode,
        "win": won,
        "first_cleared": peak >= 3,
        "damage": damage,
        "damage_taken": dict(losses),
        "physics_ticks": ticks,
        "hp": previous_health if won else int(obs["player"][4]),
        "minimum_hp": minimum_hp,
        "trio_entry": trio_entry,
        "damage_after_trio_observed": trio_damage,
        "hit_intervals": hits,
        "heal_event_amount": heals,
        **({"sampled_skill_use": {k: dict(v) for k, v in skill_use.items()}} if record_skill_use else {}),
        "observed_dash_starts": dict(observed_dashes),
        "last_observed_trio": last_trio,
        "truncated": bool(truncated),
        "speed_counts": dict(speeds),
    }


def play_episode(
    env: HKRLEnv, policy_file: Path, checkpoint: Path, seed: int
) -> dict[str, Any]:
    result = play_game(env, seed)
    episode = result["episode_id"]
    verify_actor(policy_file, episode, checkpoint)
    usage = {}
    if (checkpoint / "world.weights").is_file():
        identity, decisions, changed = map(
            int,
            Path(str(policy_file) + ".usage").read_text(encoding="utf-8-sig").split(),
        )
        if identity != episode or decisions <= 0 or not 0 <= changed <= decisions:
            raise RuntimeError(
                "The world student usage record differs from this episode."
            )
        usage = {"planner_decisions": decisions, "world_changed_plan": changed}
    return {
        **result,
        **usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--policy-file",
        required=True,
        type=Path,
        help="The running game's HKRL_TACTICS_POLICY file.",
    )
    parser.add_argument("--port", default=5555, type=int)
    parser.add_argument("--episodes", default=1, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least one")
    manifest = json.loads((args.checkpoint / "frozen.json").read_text())
    verify_checkpoint(args.checkpoint, manifest)
    if args.policy_file.read_bytes() != (args.checkpoint / "model.txt").read_bytes():
        raise ValueError("The active policy file differs from the frozen checkpoint.")
    for suffix, key in ((".weights", "head_sha256"), (".wm.weights", "world_sha256")):
        if manifest.get(key):
            active_weights = Path(str(args.policy_file) + suffix)
            if hashlib.sha256(active_weights.read_bytes()).hexdigest() != manifest[key]:
                raise ValueError(
                    f"The active weights differ from the checkpoint for {suffix}."
                )
    task = load_task_config("configs/sisters_task.yaml")
    task.action.action_repeat = 5
    task.time_limit_seconds = manifest["time_limit"]
    train = load_train_config("configs/sisters_train.yaml")
    env = HKRLEnv(make_transport(train, port=args.port), task, step_timeout_s=18)
    results = []
    try:
        for index in range(args.episodes):
            result = play_episode(
                env, args.policy_file, args.checkpoint, 9000000 + index
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            if result["win"] and index + 1 < args.episodes:
                time.sleep(8)
    finally:
        try:
            env.set_timescale(1.0)
        finally:
            env.close()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

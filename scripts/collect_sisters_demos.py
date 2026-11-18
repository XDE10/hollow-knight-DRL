"""Collect episode-separated, ordinary-nail demonstrations on the real Sisters task."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from hkrl import protocol, spaces
from hkrl.env import HKRLEnv
from hkrl.eval.scripted_policies import ScriptedAggroPolicy
from hkrl.eval.sisters_survival import (
    SistersBodyCounterPolicy,
    SistersCounterPolicy,
    SistersDodgeCounterPolicy,
    SistersSurvivalPolicy,
)
from hkrl.training.local_session import exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.live_tuning import atomic_write_json
from hkrl.wrappers import NormalizeObservation


def boss_rows(obs: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    active = obs["entities"][obs["entity_mask"].astype(bool)]
    return {int(row[0]): row for row in active if row[1] == 1}


def collect_episode(
    env: HKRLEnv, teacher: ScriptedAggroPolicy
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    normalized = NormalizeObservation(env)
    obs, info = env.reset(options={"reset_timeout_s": 90})
    episode_id = int(info["episode_id"])
    saved: dict[str, list[Any]] = {
        key: []
        for key in (
            "global",
            "player",
            "entities",
            "entity_mask",
            "actions",
            "masks",
            "rewards",
            "elapsed_ticks",
        )
    }
    history: list[dict[str, Any]] = []
    hits = []
    contacts = []
    max_concurrent = {"bosses": 0, "projectiles": 0}
    maximum_spikes = 0
    first_hit_tick = None
    attack = spaces.BUTTON_BITS["attack"]
    p = spaces.PLAYER_FEATURE_INDEX
    for step in range(4096):
        state = normalized.observation(obs)
        action = teacher.act(state, info["action_mask"])
        after, reward, terminated, truncated, next_info = env.step(action)
        if next_info["episode_id"] != episode_id:
            raise RuntimeError("episode ID changed before reset")
        for key in ("global", "player", "entities", "entity_mask"):
            saved[key].append(state[key])
        packed = [
            action["movement_x"],
            action["aim_y"],
            *action["buttons"],
            action["duration"],
            action["macro"],
        ]
        saved["actions"].append(packed)
        saved["masks"].append(info["action_mask"])
        saved["rewards"].append(reward)
        saved["elapsed_ticks"].append(next_info["elapsed_ticks"])
        visible = obs["entities"][obs["entity_mask"].astype(bool)]
        types = visible[:, spaces.ENTITY_FEATURE_INDEX["entity_type"]]
        max_concurrent["bosses"] = max(
            max_concurrent["bosses"], int((types == 1).sum())
        )
        max_concurrent["projectiles"] = max(
            max_concurrent["projectiles"],
            int(((types == 3) & (visible[:, 2] == 1)).sum()),
        )
        maximum_spikes = max(maximum_spikes, int((types == 4).sum()))
        for event in next_info["reward_events"]:
            if event.kind == protocol.RewardEventKind.DAMAGE_TAKEN:
                contacts.append(
                    dict(
                        step=step + 1,
                        source=protocol.DamageSource(event.aux_int2).name,
                        source_id=event.entity_id,
                        amount=event.amount,
                    )
                )
                if first_hit_tick is None:
                    first_hit_tick = sum(saved["elapsed_ticks"])
        before_boss, after_boss = boss_rows(obs), boss_rows(after)
        target = min(
            before_boss,
            key=lambda i: np.linalg.norm(before_boss[i][8:10]),
            default=None,
        )
        dx = float(before_boss[target][8]) if target is not None else 0.0
        history.append(
            dict(
                step=step + 1,
                tick=int(next_info["server_tick"]),
                target=target,
                dx=dx,
                move=int(action["movement_x"]) - 1,
                slash=bool(action["buttons"][attack]),
                x=float(obs["player"][p["pos_x"]]),
                vx=float(obs["player"][p["vel_x"]]),
                facing=int(obs["player"][p["facing"]]),
                facing_after=int(after["player"][p["facing"]]),
                swinging=bool(int(after["player"][p["action_flags"]]) & 1),
                applied=bool(
                    int(after["player"][p["applied_input_buttons"]]) & (1 << attack)
                ),
            )
        )
        for event in next_info["reward_events"]:
            if event.kind != protocol.RewardEventKind.DAMAGE_DEALT:
                continue
            target_id = int(event.entity_id)
            if target_id not in before_boss or target_id not in after_boss:
                continue
            before_hp = int(before_boss[target_id][12])
            after_hp = int(after_boss[target_id][12])
            if before_hp - after_hp != event.amount or event.amount <= 0:
                continue
            recent = [h for h in history[-32:] if history[-1]["tick"] - h["tick"] <= 18]
            slash = next(
                (
                    h
                    for h in reversed(recent)
                    if h["slash"] and h["target"] == target_id
                ),
                None,
            )
            if slash is None or not any(
                h["applied"] or h["swinging"]
                for h in recent
                if h["step"] >= slash["step"]
            ):
                continue
            side = int(np.sign(slash["dx"]))
            window = history[max(0, slash["step"] - 32) : slash["step"]]
            approach = side != 0 and any(
                h["target"] == target_id
                and h["dx"] * side > 0
                and h["move"] == side
                and h["vx"] * side > 0.2
                for h in window[-12:]
            )
            # Match an observed facing flip on the approach to this same target,
            # with an earlier, opposite movement command in the same episode.
            turned = approach and any(
                h["target"] == target_id
                and h["dx"] * side > 0
                and h["move"] == side
                and h["facing_after"] != h["facing"]
                and any(prev["move"] == -side for prev in window[:j])
                for j, h in enumerate(window)
            )
            hits.append(
                dict(
                    step=step + 1,
                    target_id=target_id,
                    hp_before=before_hp,
                    hp_after=after_hp,
                    amount=float(event.amount),
                    slash_step=slash["step"],
                    dx_at_slash=slash["dx"],
                    side="from_left"
                    if side > 0
                    else "from_right"
                    if side < 0
                    else "overlap",
                    approached=approach,
                    turned_and_approached=turned,
                )
            )
        obs, info = after, next_info
        if terminated or truncated:
            break
    else:
        raise RuntimeError("demonstration episode exceeded decision cap")
    return {k: np.asarray(v) for k, v in saved.items()}, dict(
        episode_id=episode_id,
        steps=len(history),
        hits=hits,
        contacts=contacts,
        max_concurrent=max_concurrent,
        max_hazards=maximum_spikes,
        elapsed_ticks=sum(saved["elapsed_ticks"]),
        first_hit_or_end_ticks=first_hit_tick
        if first_hit_tick is not None
        else sum(saved["elapsed_ticks"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument(
        "--teacher",
        choices=("aggro", "survival", "counter", "dodge_counter", "body_counter"),
        default="aggro",
    )
    parser.add_argument(
        "--ports", nargs="+", type=int, help="one loopback TCP port per game"
    )
    args = parser.parse_args()
    if os.environ.get("HKRL_SAVE_SLOT") != "3":
        raise ValueError("Sisters demonstrations require save slot 3")
    task = load_task_config(args.task)
    if task.task_id != "sisters_of_battle":
        raise ValueError("expected the Sisters of Battle task")
    root = args.output
    root.mkdir(exist_ok=True, parents=True)
    cfg = load_train_config(args.config)
    ports = args.ports or [cfg.transport.port]
    if (
        args.episodes < 1
        or len(set(ports)) != len(ports)
        or any(not 1 <= n <= 65535 for n in ports)
    ):
        parser.error("require positive episode count and distinct valid ports")
    if args.ports and (
        cfg.transport.name != "tcp"
        or cfg.transport.host not in ("localhost", "127.0.0.1")
    ):
        parser.error("--ports requires loopback TCP games")

    def collect_port(index: int, port: int) -> None:
        env = HKRLEnv(make_transport(cfg, port=port), task, step_timeout_s=15)
        policy_class = {
            "aggro": ScriptedAggroPolicy,
            "survival": SistersSurvivalPolicy,
            "counter": SistersCounterPolicy,
            "body_counter": SistersBodyCounterPolicy,
            "dodge_counter": SistersDodgeCounterPolicy,
        }[args.teacher]
        teacher = policy_class(
            env.action_space,
            attack_range=0.22,
            approach_deadzone=0.04,
            vertical_deadzone=0.06,
        )
        try:
            for i in range(index, args.episodes, len(ports)):
                dest = root / f"episode-{i + 1:03d}.npz"
                audit = dest.with_suffix(".json")
                if dest.exists() and audit.exists():
                    continue
                started = time.time()
                arrays, record = collect_episode(env, teacher)
                record.update(
                    port=port,
                    teacher=args.teacher,
                    started=started,
                    finished=time.time(),
                )
                with dest.with_suffix(".tmp").open("wb") as f:
                    np.savez_compressed(f, **arrays)
                dest.with_suffix(".tmp").replace(dest)
                atomic_write_json(audit, record)
                sides = Counter(h["side"] for h in record["hits"] if h["approached"])
                turns = Counter(
                    h["side"] for h in record["hits"] if h["turned_and_approached"]
                )
                print(
                    json.dumps(
                        dict(
                            episode=i + 1,
                            port=port,
                            seconds=round(record["finished"] - started, 2),
                            steps=record["steps"],
                            approaches=sides,
                            turns=turns,
                        )
                    ),
                    flush=True,
                )
        finally:
            env.close()

    with (
        exclusive_game_lock(Path("runs/training/game.lock")),
        ThreadPoolExecutor(max_workers=len(ports)) as pool,
    ):
        futures = [
            pool.submit(collect_port, index, port) for index, port in enumerate(ports)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()

"""Evaluate the direct native policy on complete battles with per-episode model receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import psutil
from eval_sisters_tactics import play_game
from hkrl.env import HKRLEnv
from hkrl.training.local_session import exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("At least one episode required")
    connections = [
        c
        for c in psutil.net_connections("tcp")
        if c.status == "LISTEN" and c.laddr.port == args.port
    ]
    if len(connections) != 1 or connections[0].laddr.ip != "127.0.0.1":
        raise RuntimeError("Expected one loopback game")
    process = psutil.Process(connections[0].pid)
    environment = process.environ()
    if (
        process.name().lower() != "hollow_knight.exe"
        or environment.get("HKRL_TACTICS_POLICY")
        or environment.get("HKRL_NATIVE_DEMO_DIR")
    ):
        raise RuntimeError(
            "Expected direct native control without a tactical controller"
        )
    if (
        Path(environment["HKRL_NATIVE_ACTOR"]).resolve() != args.checkpoint.resolve()
        or float(environment.get("HKRL_SISTERS_NAIL_MULTIPLIER", "1")) != 1.0
    ):
        raise RuntimeError("Wrong model or damage setting")
    ledger = Path(environment["HKRL_NATIVE_ACTOR_LEDGER"])
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    task = load_task_config("configs/experiments/sisters_counter.yaml")
    task.action.action_repeat = 5
    task.time_limit_seconds = 480
    cfg = load_train_config("configs/train/sisters_platform.yaml")
    with exclusive_game_lock(Path(f"runs/training/sisters-actor-{args.port}.lock")):
        env = HKRLEnv(make_transport(cfg, port=args.port), task, step_timeout_s=25)
        try:
            for i in range(args.episodes):
                result = play_game(env, args.seed + i)
                episode, sha, n, attack, cast, dash, art = ledger.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if int(episode) != result["episode_id"] or sha != digest or int(n) < 1:
                    raise RuntimeError(
                        "Native policy receipt does not match this battle"
                    )
                if max(result["speed_counts"], key=result["speed_counts"].get) != 1.0:
                    raise RuntimeError("Expected normal speed")
                result.update(
                    seed=args.seed + i,
                    policy_sha256=sha,
                    decisions=int(n),
                    attack=int(attack),
                    cast=int(cast),
                    dash=int(dash),
                    nail_art=int(art),
                    no_hit_win=bool(
                        result["win"]
                        and result["minimum_hp"] == 9
                        and not sum(result["damage_taken"].values())
                    ),
                )
                (args.output / f"episode-{i:03d}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(json.dumps(result), flush=True)
                if result["win"] and i + 1 < args.episodes:
                    time.sleep(8)
        finally:
            env.close()


if __name__ == "__main__":
    main()

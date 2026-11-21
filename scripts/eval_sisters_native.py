"""Evaluate the direct native policy on complete battles with per-episode model receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
    p.add_argument("--verify-unity-seed", action="store_true",
                   help="Require the reset-time Unity RNG receipt from the dedicated game process.")
    p.add_argument("--record-skills", action="store_true",
                   help="Track observed quake starts, health gain and net SOUL decreases in live samples.")
    p.add_argument("--record-memory-changes", action="store_true",
                   help="Require the executed trio actor's per-game count of changed native decisions.")
    p.add_argument(
        "--expected-mod-sha256", help="Verify the staged game mod before this cohort."
    )
    p.add_argument(
        "--restart-after-win",
        action="store_true",
        help="Return after saving a win so a supervisor can reopen the clean save.",
    )
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
        or environment.get("HKRL_NATIVE_TRIAL_CANDIDATE")
        or environment.get("HKRL_NATIVE_ONLINE_CHOICES") == "1"
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
    policy = args.checkpoint.read_bytes()
    digest = hashlib.sha256(policy).hexdigest()
    focus_policy = False
    if policy[:4] == struct.pack("<i", 0x334C5149):
        late_offset = 12 + struct.unpack_from("<i", policy, 4)[0]
        focus_policy = policy[late_offset:late_offset+4] == struct.pack("<i", 0x374C5149)
    mod_hash = None
    if args.expected_mod_sha256:
        game_root = Path(process.exe()).parent
        mod = (
            game_root
            / "hollow_knight_Data"
            / "Managed"
            / "Mods"
            / "HKRLEnvMod"
            / "HKRLEnvMod.dll"
        )
        mod_hash = hashlib.sha256(mod.read_bytes()).hexdigest()
        if mod_hash != args.expected_mod_sha256.lower():
            raise RuntimeError("The game mod differs from the staged evaluation build")
    args.output.mkdir(parents=True, exist_ok=False)
    task = load_task_config("configs/experiments/sisters_counter.yaml")
    task.action.action_repeat = 5
    task.time_limit_seconds = 480
    cfg = load_train_config("configs/train/sisters_platform.yaml")
    with exclusive_game_lock(Path(f"runs/training/sisters-actor-{args.port}.lock")):
        env = HKRLEnv(make_transport(cfg, port=args.port), task, step_timeout_s=25)
        try:
            for i in range(args.episodes):
                result = play_game(env, args.seed + i, record_skill_use=args.record_skills)
                episode, sha, n, attack, cast, dash, art = ledger.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if int(episode) != result["episode_id"] or sha != digest or int(n) < 1:
                    raise RuntimeError(
                        "Native policy receipt does not match this battle"
                    )
                focus_receipt = Path(str(ledger) + ".focus")
                if focus_policy:
                    receipt_episode, starts, inputs, gains, debits = map(
                        int, focus_receipt.read_text(encoding="utf-8-sig").splitlines()
                    )
                    if receipt_episode != int(episode) or min(starts, inputs, gains, debits) < 0 or inputs < starts:
                        raise ValueError("Invalid native focus receipt")
                    result["focus_continuations"] = {
                        "starts": starts, "changed_inputs": inputs,
                        "observed_masks_gained_within_052s": gains,
                        "observed_masks_lost_within_052s": debits,
                    }
                if args.verify_unity_seed:
                    seed_file = Path(environment["HKRL_ARENA_SEED_RECEIPT"])
                    ids = list(map(int, seed_file.read_text(encoding="utf-8-sig").splitlines()))
                    if ids != [result["episode_id"], args.seed + i]:
                        raise RuntimeError("Unity RNG receipt differs from the assigned fight")
                    result["unity_reset_seed"] = ids[1]
                if args.record_memory_changes:
                    source, total, changed = map(int, Path(str(ledger) + ".memory")
                                                 .read_text(encoding="utf-8-sig").splitlines())
                    if source != int(episode) or not 0 <= changed <= total <= int(n):
                        raise ValueError("Invalid native recurrent decision receipt")
                    result["recurrent_trio_decisions"] = total
                    result["changed_native_decisions"] = changed
                speeds = result["speed_counts"]
                if max(speeds, key=speeds.get) != 1.0 or max(speeds) > 1.001:
                    raise RuntimeError("Expected normal speed")
                result.update(
                    seed=args.seed + i,
                    policy_sha256=sha,
                    mod_sha256=mod_hash,
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
                if result["win"] and args.restart_after_win:
                    break
                if result["win"] and i + 1 < args.episodes:
                    time.sleep(8)
        finally:
            env.close()


if __name__ == "__main__":
    main()

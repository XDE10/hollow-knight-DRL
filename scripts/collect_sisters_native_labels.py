"""Collect corrective labels in development battles; only executed actions receive rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import psutil
from eval_sisters_tactics import play_game
from hkrl.env import HKRLEnv
from hkrl.protocol import RewardEventKind as K
from hkrl.training.local_session import exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config


def transition_recorder():
    starts = []
    elapsed = []
    events = []

    def record(start, end, info):
        row = np.zeros(8, "f4")
        for e in info["reward_events"]:
            if e.kind == K.DAMAGE_DEALT:
                row[0] += e.amount
            elif e.kind == K.DAMAGE_TAKEN:
                row[e.aux_int2 if e.aux_int2 in (1, 2, 3) else 4] += e.amount
            elif e.kind == K.HEAL:
                row[5] += e.amount
            elif e.kind == K.BOSS_KILLED and e.entity_id == 0:
                row[6] = 1
            elif e.kind == K.PLAYER_DEATH:
                row[7] = 1
        starts.append(start)
        elapsed.append(int(info["elapsed_ticks"]))
        events.append(row)

    return starts, elapsed, events, record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument(
        "--yield-file", type=Path,
        help="Release this game copy after saving a completed fight when the reservation file exists.",
    )
    ap.add_argument(
        "--restart-after-win",
        action="store_true",
        help="Finish this worker after a victory so the supervisor can reopen the original save.",
    )
    args = ap.parse_args()
    listeners = [
        c
        for c in psutil.net_connections("tcp")
        if c.status == "LISTEN" and c.laddr.port == args.port
    ]
    if len(listeners) != 1 or listeners[0].laddr.ip != "127.0.0.1":
        raise RuntimeError("Expected private game copy")
    proc = psutil.Process(listeners[0].pid)
    envs = proc.environ()
    if (
        proc.name().lower() != "hollow_knight.exe"
        or not envs.get("HKRL_TACTICS_POLICY")
        or float(envs.get("HKRL_SISTERS_NAIL_MULTIPLIER", "1")) != 1
    ):
        raise RuntimeError("Wrong development game mode")
    if args.checkpoint.resolve() != Path(envs["HKRL_NATIVE_ACTOR"]).resolve():
        raise RuntimeError("Wrong actor")
    tape_dir = Path(envs["HKRL_NATIVE_DEMO_DIR"])
    scale = float(envs.get("HKRL_DEVELOPMENT_SPEED", "2"))
    if scale not in (1.0, 2.0):
        raise ValueError("Development collection supports 1x or 2x speed")
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output.mkdir(exist_ok=False, parents=True)
    manifest = []
    cfg = load_train_config("configs/train/sisters_platform.yaml")
    task = load_task_config("configs/experiments/sisters_counter.yaml")
    task.action.action_repeat = 5
    task.time_limit_seconds = 480
    with exclusive_game_lock(Path(f"runs/training/sisters-actor-{args.port}.lock")):
        game = HKRLEnv(make_transport(cfg, port=args.port), task, step_timeout_s=25)
        try:
            for i in range(args.episodes):
                starts, elapsed, events, record = transition_recorder()
                result = play_game(game, args.seed + i, scale=scale, on_transition=record)
                eid = result["episode_id"]
                mode = max(result["speed_counts"], key=result["speed_counts"].get)
                if mode != scale:
                    raise RuntimeError("Wrong development speed")
                if not result["win"] and not result["truncated"] and result["hp"] == 0:
                    events[-1][7] = 1
                with (tape_dir / f"{eid}.native").open("rb") as f:
                    timing = envs.get("HKRL_NATIVE_TIMING") == "1"
                    if struct.unpack("<IQ", f.read(12)) != (
                        0x32474144 if timing else 0x31474144,
                        eid,
                    ):
                        raise ValueError("Wrong training tape")
                    rows = np.frombuffer(f.read(), "<f4").reshape(
                        -1, 259 if timing else 67
                    )
                receipt = (
                    Path(envs["HKRL_NATIVE_ACTOR_LEDGER"])
                    .read_text(encoding="utf-8-sig")
                    .splitlines()
                )
                if (
                    receipt[1] != sha
                    or int(receipt[0]) != eid
                    or int(receipt[2]) != len(rows)
                ):
                    raise ValueError("Actor receipt differs")
                applied, labels, student = (
                    rows[:, j : j + 3].astype("i2") for j in (57, 60, 63)
                )
                teacher = rows[:, 66].astype(bool)
                if not np.array_equal(
                    applied, np.where(teacher[:, None], labels, student)
                ):
                    raise ValueError("Incorrect executed action trace")
                file = f"dev-{i:03d}.npz"
                plans = {}
                if envs.get("HKRL_NATIVE_PLAN_TRACE") == "1":
                    from sisters_plan_labels import read_plans

                    plans = read_plans(tape_dir / f"{eid}.plans", eid, rows[:, 0])
                np.savez_compressed(
                    args.output / file,
                    control_time=rows[:, 0],
                    student_features=rows[:, 1:57],
                    controls=applied,
                    teacher_controls=labels,
                    time=np.array(starts, "f4"),
                    events=np.array(events, "f4"),
                    elapsed_ticks=np.array(elapsed, "i2"),
                    **({"timing_state": rows[:, 67:]} if timing else {}),
                    **plans,
                )
                result.update(
                    seed=args.seed + i,
                    teacher_fraction=float(teacher.mean()),
                    disagreement=float(np.any(student != labels, 1).mean()),
                )
                (args.output / f"episode-{i:03d}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                manifest.append(
                    {
                        "source": f"native-development/{args.seed + i}",
                        "file": file,
                        "split": "train",
                        "won": result["win"],
                    }
                )
                (args.output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                print(json.dumps(result), flush=True)
                if args.yield_file is not None and args.yield_file.exists():
                    break
                if result["win"] and args.restart_after_win:
                    break
                if result["win"]:
                    time.sleep(6)
        finally:
            game.close()


if __name__ == "__main__":
    main()

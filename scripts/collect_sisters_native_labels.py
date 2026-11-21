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


def read_decisions(
    path: Path, episode: int, timing: bool, has_teacher: bool
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Read a native episode using the recording mode of the actual game process."""
    magic, width = (0x32474144, 259) if timing else (0x31474144, 67)
    if not has_teacher:
        if not timing:
            raise ValueError("Student-only trials need timing observations")
        magic, width = 0x33474144, 196
    with path.open("rb") as f:
        if struct.unpack("<IQ", f.read(12)) != (magic, episode):
            raise ValueError("Wrong training tape")
        rows = np.frombuffer(f.read(), "<f4").reshape(-1, width)
    if not len(rows) or not np.isfinite(rows).all() or (np.diff(rows[:, 0]) <= 0).any():
        raise ValueError("Invalid decision sequence")
    data = {"control_time": rows[:, 0]}
    if not has_teacher:
        data.update(controls=rows[:, 1:4].astype("i2"), timing_state=rows[:, 4:])
        return data, {"teacher_fraction": 0.0}
    applied, labels, student = (rows[:, j : j + 3].astype("i2") for j in (57, 60, 63))
    if not np.isin(rows[:, 66], [0.0, 1.0]).all():
        raise ValueError("Invalid native controller assignment")
    teacher = rows[:, 66].astype(bool)
    if not np.array_equal(applied, np.where(teacher[:, None], labels, student)):
        raise ValueError("Incorrect executed action trace")
    data.update(
        controls=applied,
        teacher_controls=labels,
        student_controls=student,
        teacher_acted=teacher,
        student_features=rows[:, 1:57],
    )
    if timing:
        data["timing_state"] = rows[:, 67:]
    return data, {
        "teacher_fraction": float(teacher.mean()),
        "disagreement": float(np.any(student != labels, 1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-unity-seed", action="store_true")
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
    has_teacher = bool(envs.get("HKRL_TACTICS_POLICY"))
    routing = envs.get("HKRL_NATIVE_ROUTER_ROLLOUT") == "1"
    sampling = envs.get("HKRL_NATIVE_ONLINE_CHOICES") == "1"
    candidate = envs.get("HKRL_NATIVE_TRIAL_CANDIDATE")
    directional = envs.get("HKRL_NATIVE_DIRECTIONAL_SAMPLER")
    if directional and envs.get("HKRL_NATIVE_OPTION_SAMPLER"):
        raise ValueError("Conflicting option samplers")
    sampler_path = directional or envs.get("HKRL_NATIVE_OPTION_SAMPLER")
    sampler = Path(sampler_path).read_bytes() if sampler_path else None
    if sampler is not None and (has_teacher or not sampling or candidate):
        raise ValueError(
            "A sampled neural option requires a teacher-free development worker"
        )
    if (
        proc.name().lower() != "hollow_knight.exe"
        or not (has_teacher or sampling or candidate or routing)
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
    if routing and (has_teacher or sampling or candidate or envs.get("HKRL_NATIVE_TIMING") != "1"
                    or float(envs.get("HKRL_NATIVE_DEMO_RATIO", "-1")) != 0):
        raise ValueError("A winning-router rollout requires exclusive student control")
    alternate_sha = hashlib.sha256(Path(candidate).read_bytes()).hexdigest() if candidate else None
    if candidate and (
        has_teacher or sampling or scale != 1 or envs.get("HKRL_NATIVE_TIMING") != "1"
        or float(envs.get("HKRL_NATIVE_DEMO_RATIO", "-1")) != 0 or alternate_sha == sha
    ):
        raise ValueError("Expected two distinct frozen neural arms at original speed")
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
                wall_start = time.monotonic()
                result = play_game(game, args.seed + i, scale=scale, on_transition=record)
                play_wall_seconds = time.monotonic() - wall_start
                eid = result["episode_id"]
                mode = max(result["speed_counts"], key=result["speed_counts"].get)
                if mode != scale:
                    raise RuntimeError("Wrong development speed")
                if not result["win"] and not result["truncated"] and result["hp"] == 0:
                    events[-1][7] = 1
                timing = envs.get("HKRL_NATIVE_TIMING") == "1"
                decisions, control_stats = read_decisions(
                    tape_dir / f"{eid}.native", eid, timing, has_teacher
                )
                clock = decisions["control_time"]
                receipt = (
                    Path(envs["HKRL_NATIVE_ACTOR_LEDGER"])
                    .read_text(encoding="utf-8-sig")
                    .splitlines()
                )
                if (
                    receipt[1] != sha
                    or int(receipt[0]) != eid
                    or int(receipt[2]) != len(clock)
                ):
                    raise ValueError("Actor receipt differs")
                controller: dict[str, str | float] = {"student_policy_sha256": sha}
                if alternate_sha:
                    from sisters_control_blocks import neural_assignments

                    block_starts, assigned = neural_assignments(
                        tape_dir / f"{eid}.arms", eid, clock, sha, alternate_sha
                    )
                    decisions.update(block_starts=block_starts, candidate_draw=assigned)
                    controller.update(alternate_policy_sha256=alternate_sha, candidate_probability=0.5)
                if has_teacher and envs.get("HKRL_NATIVE_DEMO_RATIO"):
                    controller["teacher_probability"] = float(envs["HKRL_NATIVE_DEMO_RATIO"])
                if args.verify_unity_seed:
                    seed_file = Path(envs["HKRL_ARENA_SEED_RECEIPT"])
                    ids = list(map(int, seed_file.read_text(encoding="utf-8-sig").splitlines()))
                    if ids != [eid, args.seed + i]:
                        raise RuntimeError("Development Unity seed differs from this assigned fight")
                    result["unity_reset_seed"] = ids[1]
                file = f"dev-{i:03d}.npz"
                plans = {}
                if envs.get("HKRL_NATIVE_PLAN_TRACE") == "1":
                    from sisters_plan_labels import read_plans

                    plans = read_plans(tape_dir / f"{eid}.plans", eid, clock)
                online = {}
                if routing:
                    from sisters_router_rollout import read_routes

                    seed = int(envs["HKRL_NATIVE_DEMO_SEED"])
                    temperature = float(envs["HKRL_NATIVE_ROUTER_TEMPERATURE"])
                    online = read_routes(tape_dir / f"{eid}.routes", eid, sha, seed, temperature,
                                         clock, decisions["timing_state"])
                    controller.update(router_temperature=temperature, sampler_draw_seed=seed)
                    result.update(game_speed=scale, play_wall_seconds=play_wall_seconds,
                                  native_inputs=len(clock), control_game_seconds=float(clock[-1]-clock[0]))
                    result.update(behavior_policy_sha256=sha, randomized_choices=len(online["route_rows"]))
                if sampling:
                    from sisters_online_choices import late_catalogue, read_choices

                    if not timing or scale != 1 or control_stats["teacher_fraction"] != 0:
                        raise ValueError(
                            "Randomized training requires original-speed student control"
                        )
                    read_online = read_choices
                    if directional:
                        from sisters_directional_options import read_menu

                        read_online = read_menu
                        controller["directional_groups"] = 6
                    online = read_online(
                        tape_dir / f"{eid}.choices",
                        eid,
                        clock,
                        decisions["timing_state"],
                        decisions["controls"],
                        late_catalogue(args.checkpoint.read_bytes()),
                        sampler,
                    )
                    result.update(
                        behavior_policy_sha256=sha,
                        randomized_choices=len(online["choice_rows"]),
                    )
                    if sampler is not None:
                        controller["sampler_sha256"] = hashlib.sha256(
                            sampler
                        ).hexdigest()
                        controller["sampler_draw_seed"] = int(
                            envs["HKRL_NATIVE_DEMO_SEED"]
                        )
                np.savez_compressed(
                    args.output / file,
                    **decisions,
                    time=np.array(starts, "f4"),
                    events=np.array(events, "f4"),
                    elapsed_ticks=np.array(elapsed, "i2"),
                    **plans,
                    **online,
                )
                result.update(
                    seed=args.seed + i,
                    teacher_labels=has_teacher,
                    **control_stats,
                    **controller,
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
                        "teacher_labels": has_teacher,
                        **controller,
                        **({"behavior_policy_sha256": sha} if online else {}),
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

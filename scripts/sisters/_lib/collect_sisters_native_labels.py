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
from hkrl.protocol import LifecycleState
from hkrl.protocol import RewardEventKind as K
from hkrl.training.local_session import exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config


def transition_recorder():
    starts = []
    ends = []
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
        ends.append(end)
        elapsed.append(int(info["elapsed_ticks"]))
        events.append(row)

    return starts, ends, elapsed, events, record


def ppo_timeout_prefix(
    decisions: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    cutoff: float,
    final_step_start: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Keep the last observed state for bootstrap and exclude later native rows."""
    clock = decisions["control_time"]
    count = int(np.searchsorted(clock, np.float32(cutoff), side="right"))
    if count < 2 or not clock[count - 1] > np.float32(final_step_start):
        raise ValueError("Timeout has no native boundary observation from the final STEP")
    if draws:
        keep = draws["ppo_rows"] < count
        draws = {key: rows[keep] for key, rows in draws.items()}
    return {key: rows[:count] for key, rows in decisions.items()}, draws


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


def corrective_label_mask(
    decisions: dict[str, np.ndarray], catalogue: np.ndarray
) -> np.ndarray:
    """Mark supervisable Trio rows without substituting a different teacher key."""
    if decisions["teacher_acted"].any() or not np.array_equal(
        decisions["controls"], decisions["student_controls"]
    ):
        raise ValueError("Corrective tape contains teacher-controlled actions")
    state = decisions["timing_state"]
    labels = decisions["teacher_controls"]
    exact = (labels[:, None, :] == catalogue[None, :, :]).all(axis=2).any(axis=1)
    trio_boss = np.zeros(len(labels), dtype=bool)
    for slot in (24, 60, 96):
        trio_boss |= (state[:, slot] > .5) & (state[:, slot + 13] > 0) & (state[:, slot + 14] > .5)
    return exact & trio_boss & (state[:, 23] > .5) & (state[:, 4] > 0)


def teacher_assets(path: Path) -> dict[str, str]:
    result = {}
    for suffix in ("", ".weights", ".wm.weights"):
        asset = Path(str(path) + suffix)
        if suffix == "" or asset.is_file():
            result["teacher.txt" + suffix] = hashlib.sha256(asset.read_bytes()).hexdigest()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-unity-seed", action="store_true")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--expected-mod-sha256")
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
    timing = envs.get("HKRL_NATIVE_TIMING") == "1"
    trace_only = envs.get("HKRL_NATIVE_TRACE_ONLY") == "1"
    routing = envs.get("HKRL_NATIVE_ROUTER_ROLLOUT") == "1"
    primitive = envs.get("HKRL_NATIVE_PRIMITIVE_ROLLOUT") == "1"
    ppo = envs.get("HKRL_NATIVE_PPO_ROLLOUT") == "1"
    corrective = envs.get("HKRL_NATIVE_CORRECTIVE_LABELS") == "1"
    if primitive and routing:
        raise ValueError("Select one behavior policy")
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
    if trace_only and (
        has_teacher or sampling or candidate or routing or primitive or not timing
        or envs.get("HKRL_NATIVE_PLAN_TRACE") == "1"
        or float(envs.get("HKRL_NATIVE_DEMO_RATIO", "-1")) != 0
    ):
        raise ValueError("Trace-only recording requires exclusive deterministic neural control")
    if (
        not (proc.name().lower().startswith("hollow_knight") or Path(proc.exe()).name.lower() in ("hollow_knight.exe", "hollow_knight.x86_64"))
        or not (has_teacher or sampling or candidate or routing or primitive or trace_only or ppo)
        or float(envs.get("HKRL_SISTERS_NAIL_MULTIPLIER", "1")) != 1
    ):
        raise RuntimeError("Wrong development game mode")
    if args.checkpoint.resolve() != Path(envs["HKRL_NATIVE_ACTOR"]).resolve():
        raise RuntimeError("Wrong actor")
    tape_dir = Path(envs["HKRL_NATIVE_DEMO_DIR"])
    scale = float(envs.get("HKRL_DEVELOPMENT_SPEED", "2"))
    if scale not in (1.0, 2.0):
        raise ValueError("Development collection supports 1x or 2x speed")
    if corrective and (
        not has_teacher or sampling or candidate or routing or primitive or trace_only or not timing
        or scale != 1 or not args.verify_unity_seed or not args.expected_mod_sha256 or args.episodes != 1
        or envs.get("HKRL_NATIVE_PLAN_TRACE") == "1" or float(envs.get("HKRL_NATIVE_DEMO_RATIO", "-1")) != 0
    ):
        raise ValueError("Corrective recording requires a single seeded label-only original-speed fight")
    if ppo:
        if ((has_teacher and not corrective) or sampling or candidate or routing or primitive or trace_only or not timing
                or scale != 1 or not args.verify_unity_seed or not args.expected_mod_sha256
                or args.episodes != 1 or float(envs.get("HKRL_NATIVE_DEMO_RATIO", "-1")) != 0):
            raise ValueError("PPO requires a single original-speed, teacher-free seeded fight")
    if args.expected_mod_sha256:
        mod = (Path(proc.exe()).parent / "hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll")
        if hashlib.sha256(mod.read_bytes()).hexdigest() != args.expected_mod_sha256:
            raise ValueError("Wrong installed development game mod")
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    teacher_path = Path(envs["HKRL_TACTICS_POLICY"]) if has_teacher else None
    teacher_hashes = teacher_assets(teacher_path) if teacher_path else {}
    corrective_catalogue = None
    if corrective:
        raw_ckpt = args.checkpoint.read_bytes()
        if len(raw_ckpt) >= 12 and struct.unpack_from("<i", raw_ckpt, 0)[0] == 0x334C5149:
            _, first_sz, late_sz = struct.unpack_from("<3i", raw_ckpt, 0)
            trio = raw_ckpt[12 + first_sz : 12 + first_sz + late_sz]
        else:
            trio = raw_ckpt
        if len(trio) >= 44 and struct.unpack_from("<i", trio, 0)[0] == 0x3E4C5149:
            n_act = struct.unpack_from("<11i", trio, 0)[10]
            corrective_catalogue = np.frombuffer(trio, "<i4", count=n_act * 3, offset=44).reshape(n_act, 3).copy()
        elif len(trio) >= 52 and struct.unpack_from("<i", trio, 0)[0] == 0x3F4C5149:
            n_act = struct.unpack_from("<13i", trio, 0)[12]
            corrective_catalogue = np.frombuffer(trio, "<i4", count=n_act * 3, offset=52).reshape(n_act, 3).copy()
        else:
            from sisters_online_choices import late_catalogue
            from train_sisters_timing_iql import memory_components, phase_components

            _, trio = phase_components(raw_ckpt)
            leaf, _ = memory_components(trio)
            corrective_catalogue = late_catalogue(leaf)
    if (routing or primitive) and (has_teacher or sampling or candidate or envs.get("HKRL_NATIVE_TIMING") != "1"
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
    cfg = load_train_config("configs/sisters_train.yaml")
    task = load_task_config("configs/sisters_task.yaml")
    task.action.action_repeat = 5
    task.time_limit_seconds = 480
    with exclusive_game_lock(Path(f"runs/training/sisters-actor-{args.port}.lock")):
        game = HKRLEnv(make_transport(cfg, port=args.port), task, step_timeout_s=25)
        paused_for_timeout = False
        try:
            for i in range(args.episodes):
                starts, ends, elapsed, events, record = transition_recorder()
                wall_start = time.monotonic()
                result = play_game(game, args.seed + i, scale=scale, on_transition=record)
                play_wall_seconds = time.monotonic() - wall_start
                eid = result["episode_id"]
                clip_timeout = (ppo or corrective or trace_only) and result["truncated"]
                if clip_timeout:
                    paused_for_timeout = True
                    ack = game.pause()
                    if ack["episode_id"] != eid or ack["lifecycle_state"] != LifecycleState.RUNNING:
                        raise ValueError("Timeout pause did not acknowledge the active recorded episode")
                    result["ppo_timeout_game_seconds"] = ends[-1]
                mode = max(result["speed_counts"], key=result["speed_counts"].get)
                if mode != scale:
                    raise RuntimeError("Wrong development speed")
                if teacher_path and teacher_hashes != teacher_assets(teacher_path):
                    raise ValueError("Teacher files changed while collecting the fight")
                inferred_death = (
                    not result["win"] and not result["truncated"] and result["hp"] == 0
                    and not any(row[7] for row in events)
                )
                if inferred_death:
                    events[-1][7] = 1
                result["death_inferred_from_hp"] = inferred_death
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
                controller: dict[str, str | float | bool] = {"student_policy_sha256": sha}
                if trace_only:
                    controller["deterministic_native_trace"] = True
                if trace_only or corrective:
                    result.update(behavior_policy_sha256=sha, native_inputs=len(clock))
                if corrective_catalogue is not None:
                    valid = corrective_label_mask(decisions, corrective_catalogue)
                    decisions["corrective_label_valid"] = valid
                    controller["corrective_labels"] = True
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
                if ppo:
                    from train_sisters_recurrent_ppo import read_native_draws

                    seed = int(envs["HKRL_NATIVE_DEMO_SEED"])
                    temperature = float(envs["HKRL_NATIVE_PPO_TEMPERATURE"])
                    online = read_native_draws(tape_dir / f"{eid}.ppo", eid, sha, seed, temperature,
                                               clock, decisions["timing_state"])
                    controller.update(ppo_temperature=temperature, sampler_draw_seed=seed)
                    result.update(behavior_policy_sha256=sha, mod_sha256=args.expected_mod_sha256)
                if clip_timeout:
                    decisions, online = ppo_timeout_prefix(decisions, online, ends[-1], starts[-1])
                    result["timeout_discarded_native_rows"] = len(clock) - len(decisions["control_time"])
                    clock = decisions["control_time"]
                if ppo:
                    result["ppo_decisions"] = len(online["ppo_rows"])
                if corrective:
                    result["valid_corrective_labels"] = int(decisions["corrective_label_valid"].sum())
                if routing or primitive:
                    from sisters_primitive_rollout import read_keys
                    from sisters_router_rollout import read_routes

                    seed = int(envs["HKRL_NATIVE_DEMO_SEED"])
                    temperature = float(envs["HKRL_NATIVE_ROUTER_TEMPERATURE"])
                    if primitive:
                        smooth = envs.get("HKRL_NATIVE_SMOOTH_KEYS") == "1"
                        online = read_keys(tape_dir / f"{eid}.keys", eid, sha, seed, temperature,
                                           clock, decisions["timing_state"], smooth=smooth)
                        if smooth:
                            controller["smooth_key_sampling"] = True
                        index = "key_rows"
                    else:
                        online = read_routes(tape_dir / f"{eid}.routes", eid, sha, seed, temperature,
                                             clock, decisions["timing_state"])
                        index = "route_rows"
                    controller.update(router_temperature=temperature, sampler_draw_seed=seed)
                    result.update(game_speed=scale, play_wall_seconds=play_wall_seconds,
                                  native_inputs=len(clock), control_game_seconds=float(clock[-1]-clock[0]))
                    result.update(behavior_policy_sha256=sha, randomized_choices=len(online[index]))
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
                    game_speed=scale,
                    play_wall_seconds=play_wall_seconds,
                    native_inputs=len(clock),
                    control_game_seconds=float(clock[-1] - clock[0]),
                    **control_stats,
                    **controller,
                )
                if args.expected_mod_sha256:
                    result["mod_sha256"] = args.expected_mod_sha256
                if teacher_hashes:
                    result["teacher_asset_sha256"] = teacher_hashes
                result["tape_sha256"] = hashlib.sha256((args.output / file).read_bytes()).hexdigest()
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
                        "truncated": result["truncated"],
                        "tape_sha256": result["tape_sha256"],
                        **controller,
                        **({"behavior_policy_sha256": sha} if online or trace_only or corrective else {}),
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
            try:
                if paused_for_timeout:
                    game.resume()
            finally:
                game.close()


if __name__ == "__main__":
    main()

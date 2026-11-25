"""Colab G4 High-Concurrency Native Linux Hollow Knight Runner (Eval + Closed-Loop Online No-Hit PPO).

Designed to saturate Colab G4's multi-core CPU, 170+ GB RAM, and 96 GB RTX PRO 6000 VRAM:
  - Pulls hk-linux-1.5.78-v2.tar.gz from local NVMe (/content/hk-linux-base) once.
  - Spawns N parallel headless Hollow Knight Linux instances (e.g. 10-16 lanes) via Xvfb
    with per-lane isolated XDG_CONFIG_HOME/HOME saves and unique TCP ports (5600..5600+N-1).
  - Supports:
      1) --mode smoke      : 1-episode sanity verification (checks Xvfb/nographics, Mod, Unity seed, 1x speed).
      2) --mode eval       : High-concurrency multi-episode evaluation (dev8 / confirm30 / formal100) + scoring.
      3) --mode online-rl  : Closed-loop On-G4 Online Safe Recurrent No-Hit PPO:
                             16-lane parallel sampled rollouts -> 20s GPU PPO update on same G4
                             -> 16-lane validation gate -> auto-promote & push champion to Drive!
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_no_hit_win(g: dict[str, Any]) -> bool:
    if g.get("no_hit_win") is True:
        return True
    dmg = sum(int(v) for v in (g.get("damage_taken") or {}).values())
    min_hp = g.get("minimum_hp")
    if min_hp is not None:
        return bool(g.get("win") and int(min_hp) == 9 and dmg == 0)
    return bool(g.get("win") and int(g.get("hp", 0)) == 9 and dmg == 0)


def wait_for_port(port: int, proc: subprocess.Popen, timeout_s: float = 150.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def kill_proc_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5.0)
    except Exception:
        pass


def stage_lane_game(game_root: Path, lane_root: Path) -> Path:
    """Create an isolated per-lane game dir (14KB exe copy + symlinks) to bypass Unity's SingleInstance unity.lock."""
    game_dir = lane_root / "game"
    game_dir.mkdir(parents=True, exist_ok=True)
    lock_file = game_dir / "unity.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except OSError:
            pass
    for item in game_root.iterdir():
        if item.name == "unity.lock":
            continue
        dst = game_dir / item.name
        if item.name == "hollow_knight.x86_64":
            if dst.is_symlink():
                dst.unlink()
            if not dst.is_file() or dst.stat().st_size != item.stat().st_size:
                shutil.copy2(item, dst)
                os.chmod(dst, 0o755)
        else:
            if not dst.exists() and not dst.is_symlink():
                os.symlink(item, dst)
    appid_dst = game_dir / "steam_appid.txt"
    if not appid_dst.exists():
        appid_dst.write_text("367520\n", encoding="utf-8")
    return game_dir


def stage_lane_saves(game_root: Path, lane_root: Path) -> tuple[Path, Path]:
    home_dir = lane_root / "home"
    xdg_dir = lane_root / "xdg"
    save_dst = xdg_dir / "unity3d/Team Cherry/Hollow Knight"
    home_save_dst = home_dir / ".config/unity3d/Team Cherry/Hollow Knight"
    for d in (save_dst, home_save_dst):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        for fname in ("user3.dat", "user3.modded.json"):
            src = game_root / "hkrl_saves" / fname
            shutil.copy2(src, d / fname)
    return home_dir, xdg_dir


def run_single_episode(
    *,
    lane_id: int,
    seed: int,
    game_root: Path,
    repo_root: Path,
    observe_script: Path,
    actor_bin: Path,
    mod_sha: str,
    run_dir: Path,
    mode: str,
    temperature: float,
    draw_seed: int,
    use_nographics: bool,
    no_batchmode: bool = False,
    soft_gl: bool = False,
    force_glcore: bool = True,
    max_attempts: int = 3,
    radiant_max_trio_hits: int = 0,
) -> dict[str, Any]:
    port = 5600 + lane_id
    display_num = 100 + lane_id
    base = run_dir / f"games/{seed}-policy"
    res_file = base / "result.json"
    actor_sha = sha256(actor_bin)
    eff_radiant = int(radiant_max_trio_hits) if mode == "ppo" else 0
    if res_file.is_file():
        r = json.loads(res_file.read_bytes())
        cache_ok = (
            r.get("policy_sha256") == actor_sha
            and r.get("mod_sha256") == mod_sha
            and r.get("mode") in (None, mode)
            and int(r.get("radiant_max_trio_hits", 0)) == eff_radiant
        )
        if cache_ok and mode == "ppo":
            try:
                epc = run_dir / r["folder"] / "episode-000.json"
                gg = json.loads(epc.read_bytes())
                cache_ok = (
                    abs(float(gg.get("ppo_temperature", 0.0)) - temperature) <= 1e-6
                    and int(gg.get("sampler_draw_seed", -1)) == draw_seed
                )
            except Exception:
                cache_ok = False
        if cache_ok:
            if r.get("win") and not r.get("no_hit_win") and sum(int(v) for v in (r.get("damage_taken") or {}).values()) == 0:
                ep_path = run_dir / r["folder"] / "episode-000.json"
                if ep_path.is_file():
                    g = json.loads(ep_path.read_bytes())
                    fixed = compute_no_hit_win(g)
                    if fixed != bool(r.get("no_hit_win", False)):
                        r["no_hit_win"] = fixed
                        part = base / "result.partial"
                        part.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
                        part.replace(res_file)
            return r
        res_file.unlink(missing_ok=True)
    steam_appid_file = game_root / "steam_appid.txt"
    if not steam_appid_file.is_file():
        steam_appid_file.write_text("367520\n", encoding="utf-8")
    lane_tmp = Path(f"/tmp/hkrl-lane-{lane_id}")
    lane_game_dir = stage_lane_game(game_root, lane_tmp)
    exe = lane_game_dir / "hollow_knight.x86_64"
    lane_tmpdir = lane_tmp / "tmp"
    lane_tmpdir.mkdir(parents=True, exist_ok=True)

    last_err = ""
    for attempt in range(1, max_attempts + 1):
        ep_dir = base / f"attempt-{attempt}"
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
        ep_dir.mkdir(parents=True, exist_ok=True)
        native_dir = ep_dir / "native"
        native_dir.mkdir(exist_ok=True)

        home_dir, xdg_dir = stage_lane_saves(game_root, lane_tmp)
        env = dict(os.environ)
        for k in list(env):
            if k.startswith("HKRL_NATIVE_") or k == "HKRL_TACTICS_POLICY":
                env.pop(k, None)
        env.update(
            {
                "HOME": str(home_dir),
                "XDG_CONFIG_HOME": str(xdg_dir),
                "XDG_DATA_HOME": str(lane_tmp / "data"),
                "TMPDIR": str(lane_tmpdir),
                "SteamAppId": "367520",
                "SteamGameId": "367520",
                "HKRL_NATIVE_DEMO_DIR": str(native_dir),
                "HKRL_NATIVE_TIMING": "1",
                "HKRL_NATIVE_DEMO_RATIO": "0",
                "HKRL_NATIVE_ACTOR": str(actor_bin),
                "HKRL_NATIVE_ACTOR_LEDGER": str(ep_dir / "actor-ledger.txt"),
                "HKRL_ARENA_BASE_SEED": str(seed),
                "HKRL_ARENA_SEED_RECEIPT": str(ep_dir / "unity-seed.txt"),
                "HKRL_SISTERS_NAIL_MULTIPLIER": "1",
                "HKRL_DEVELOPMENT_SPEED": "1",
                "HKRL_COMPACT_RENDER": "1",
                "HKRL_SAVE_SLOT": "3",
                "HKRL_HOST": "127.0.0.1",
                "HKRL_PORT": str(port),
                "SDL_AUDIODRIVER": "dummy",
            }
        )
        if soft_gl:
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            env["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        if mode == "ppo":
            env["HKRL_NATIVE_PPO_ROLLOUT"] = "1"
            env["HKRL_NATIVE_DEMO_SEED"] = str(draw_seed)
            env["HKRL_NATIVE_PPO_TEMPERATURE"] = f"{temperature:.8g}"
        else:
            env["HKRL_NATIVE_TRACE_ONLY"] = "1"

        game_args = [
            str(exe),
            "-screen-fullscreen", "0",
            "-screen-width", "384",
            "-screen-height", "216",
            "-logFile", str(ep_dir / "unity.log"),
        ]
        if force_glcore and not use_nographics:
            game_args.extend(["-force-glcore", "-force-opengl"])
        if not no_batchmode:
            game_args.insert(1, "-batchmode")
        if use_nographics:
            game_args.insert(2 if not no_batchmode else 1, "-nographics")
            cmd = game_args
        else:
            for stale in (Path(f"/tmp/.X{display_num}-lock"), Path(f"/tmp/.X11-unix/X{display_num}")):
                try:
                    stale.unlink()
                except OSError:
                    pass
            cmd = ["xvfb-run", "-n", str(display_num), "-s", "-screen 0 384x216x24 -ac +extension GLX +render -noreset -nolisten tcp", *game_args]

        t0 = time.monotonic()
        if attempt == 1 and lane_id > 0:
            time.sleep(min(lane_id * 0.25, 8.0))
        game_proc_log = ep_dir / "game_proc.log"
        with game_proc_log.open("wb") as gplog:
            proc = subprocess.Popen(
                cmd,
                cwd=str(lane_game_dir),
                env=env,
                stdout=gplog,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            if not wait_for_port(port, proc, timeout_s=120.0):
                gp_tail = game_proc_log.read_text(encoding="utf-8", errors="ignore")[-600:] if game_proc_log.is_file() else ""
                ulog = ep_dir / "unity.log"
                u_tail = ulog.read_text(encoding="utf-8", errors="ignore")[-800:] if ulog.is_file() else "(no unity.log)"
                mlog = xdg_dir / "unity3d/Team Cherry/Hollow Knight/ModLog.txt"
                m_tail = mlog.read_text(encoding="utf-8", errors="ignore")[-600:] if mlog.is_file() else "(no ModLog.txt)"
                last_err = f"Game did not listen on port {port} (exit={proc.poll()})\n---game_proc---\n{gp_tail}\n---unity.log---\n{u_tail}\n---ModLog---\n{m_tail}"
                print(f"[lane={lane_id:02d} seed={seed} att={attempt}] {last_err}", flush=True)
                continue
            print(f"  [lane={lane_id:02d}] seed={seed} port={port} ready in {time.monotonic()-t0:.1f}s (fighting...)", flush=True)

            py_env = dict(os.environ)
            py_env["PYTHONPATH"] = f"{repo_root / 'python'}:{repo_root / 'scripts'}"
            if eff_radiant > 0:
                py_env["HKRL_RADIANT_MAX_TRIO_HITS"] = str(eff_radiant)
            else:
                py_env.pop("HKRL_RADIANT_MAX_TRIO_HITS", None)
            with (ep_dir / "game.out").open("wb") as fout, (ep_dir / "game.err").open("wb") as ferr:
                cp = subprocess.run(
                    [
                        sys.executable, "-u", str(observe_script),
                        "--checkpoint", str(actor_bin),
                        "--port", str(port),
                        "--episodes", "1",
                        "--seed", str(seed),
                        "--output", str(ep_dir / "game"),
                        "--verify-unity-seed",
                        "--expected-mod-sha256", mod_sha,
                    ],
                    cwd=str(repo_root),
                    env=py_env,
                    stdout=fout,
                    stderr=ferr,
                    timeout=480,
                )
            if cp.returncode != 0:
                err_tail = (ep_dir / "game.err").read_text(encoding="utf-8", errors="ignore")[-1200:]
                out_tail = (ep_dir / "game.out").read_text(encoding="utf-8", errors="ignore")[-400:]
                last_err = f"observe-collect exit={cp.returncode}:\n---game.err---\n{err_tail}\n---game.out---\n{out_tail}"
                print(f"[lane={lane_id:02d} seed={seed} att={attempt}] {last_err}", flush=True)
                continue

            ep_json = ep_dir / "game/episode-000.json"
            g = json.loads(ep_json.read_bytes())
            if (
                g["seed"] != seed
                or g["unity_reset_seed"] != seed
                or g["behavior_policy_sha256"] != actor_sha
                or g["mod_sha256"] != mod_sha
                or g["teacher_fraction"] != 0
                or g["game_speed"] != 1
            ):
                raise RuntimeError(f"Provenance mismatch on seed {seed}")

            _tot = sum(int(v) for v in (g.get("damage_taken") or {}).values())
            _ehp = int((g.get("trio_entry") or {}).get("hp", 9))
            _trio = max(0, _tot - max(0, 9 - _ehp))
            _rad_stop = (ep_dir / "radiant-stop.json").is_file()
            receipt = {
                "mode": mode,
                "radiant_max_trio_hits": eff_radiant,
                "radiant_early_stopped": _rad_stop,
                "trio_masks_lost": _trio,
                "trio_no_hit_win": bool(g["win"]) and _trio == 0,
                "seed": seed,
                "lane": lane_id,
                "port": port,
                "attempt": attempt,
                "win": bool(g["win"]),
                "no_hit_win": compute_no_hit_win(g),
                "trio": bool(g.get("first_cleared", False)),
                "truncated": bool(g.get("truncated", False)),
                "hp": int(g.get("hp", 0)),
                "entry_hp": int((g.get("trio_entry") or {}).get("hp", 9)),
                "damage_taken": g.get("damage_taken", {}),
                "wall_sec": round(time.monotonic() - t0, 1),
                "receipt": f"attempt-{attempt}/game/episode-000.json",
                "folder": f"games/{seed}-policy/attempt-{attempt}/game",
                "policy_sha256": actor_sha,
                "mod_sha256": mod_sha,
            }
            part = base / "result.partial"
            part.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            part.replace(res_file)
            return receipt
        finally:
            kill_proc_tree(proc)

    raise RuntimeError(f"Seed {seed} failed after {max_attempts} attempts: {last_err}")


def run_parallel_cohort(
    *,
    seeds: list[int],
    workers: int,
    game_root: Path,
    repo_root: Path,
    observe_script: Path,
    actor_bin: Path,
    run_dir: Path,
    mode: str,
    temperatures: dict[int, float],
    draw_seeds: dict[int, int],
    use_nographics: bool,
    no_batchmode: bool = False,
    soft_gl: bool = False,
    max_losses_early_stop: int | None = None,
    max_attempts: int = 3,
    radiant_max_trio_hits: int = 0,
) -> list[dict[str, Any]]:
    mod_sha = sha256(game_root / "hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll")
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    losses = 0
    stop_early = False

    # Assign seeds round-robin across workers so each worker owns a fixed lane_id/port/display!
    buckets: list[list[int]] = [[] for _ in range(workers)]
    for i, s in enumerate(seeds):
        buckets[i % workers].append(s)

    t0 = time.monotonic()

    def lane_worker(lane_id: int, lane_seeds: list[int]) -> list[dict[str, Any]]:
        nonlocal losses, stop_early
        out: list[dict[str, Any]] = []
        for s in lane_seeds:
            if stop_early:
                break
            r = run_single_episode(
                lane_id=lane_id,
                seed=s,
                game_root=game_root,
                repo_root=repo_root,
                observe_script=observe_script,
                actor_bin=actor_bin,
                mod_sha=mod_sha,
                run_dir=run_dir,
                mode=mode,
                temperature=temperatures.get(s, 0.0),
                draw_seed=draw_seeds.get(s, 0),
                use_nographics=use_nographics,
                no_batchmode=no_batchmode,
                soft_gl=soft_gl,
                max_attempts=max_attempts,
                radiant_max_trio_hits=radiant_max_trio_hits,
            )
            out.append(r)
            dt = r["damage_taken"]
            dsum = sum(int(v) for v in dt.values())
            print(
                f"  [lane={lane_id:02d}] seed={s} win={r['win']} nohit={r['no_hit_win']} "
                f"ehp={r['entry_hp']} dmg={dsum}({dt}) t={r['wall_sec']}s",
                flush=True,
            )
            if not r["win"]:
                losses += 1
                if max_losses_early_stop is not None and losses >= max_losses_early_stop:
                    stop_early = True
                    (run_dir / "STOP_UNREACHABLE").write_text(f"losses={losses}\n", encoding="utf-8")
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(lane_worker, i, b) for i, b in enumerate(buckets) if b]
        for f in as_completed(futs):
            results.extend(f.result())

    results.sort(key=lambda r: r["seed"])
    wins = sum(int(r["win"]) for r in results)
    no_hits = sum(int(r["no_hit_win"]) for r in results)
    tot_dmg: dict[str, int] = {}
    for r in results:
        for k, v in r["damage_taken"].items():
            tot_dmg[k] = tot_dmg.get(k, 0) + int(v)
    d_all = sum(tot_dmg.values())
    p1_damage_total = sum(max(0, 9 - int(r.get("entry_hp", 9))) for r in results)
    trio_damage_total = max(0, d_all - p1_damage_total)
    trio_no_hits = sum(
        int(
            bool(
                r.get(
                    "trio_no_hit_win",
                    bool(r["win"])
                    and max(
                        0,
                        sum(int(v) for v in (r.get("damage_taken") or {}).values())
                        - max(0, 9 - int(r.get("entry_hp", 9))),
                    )
                    == 0,
                )
            )
        )
        for r in results
    )
    summary = {
        "trio_no_hit_wins": trio_no_hits,
        "mode": mode,
        "workers": workers,
        "episodes_done": len(results),
        "episodes_target": len(seeds),
        "wins": wins,
        "no_hit_wins": no_hits,
        "win_rate": round(wins / max(1, len(results)), 4),
        "no_hit_rate": round(no_hits / max(1, len(results)), 4),
        "avg_damage": round(d_all / max(1, len(results)), 3),
        "p1_damage_total": p1_damage_total,
        "trio_damage_total": trio_damage_total,
        "damage_totals": tot_dmg,
        "wall_seconds": round(time.monotonic() - t0, 1),
        "early_stopped": stop_early,
        "rows": results,
    }
    (run_dir / "cloud-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "COLAB_MULTI_SUMMARY",
        json.dumps({k: summary[k] for k in ("mode", "workers", "episodes_done", "wins", "no_hit_wins", "trio_no_hit_wins", "avg_damage", "p1_damage_total", "trio_damage_total", "damage_totals", "wall_seconds")}),
        flush=True,
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("smoke", "eval"))
    ap.add_argument("--game-root", type=Path, default=Path("/content/hk-linux-base"))
    ap.add_argument("--repo-root", type=Path, default=Path("/content/hkrl-repo"))
    ap.add_argument("--observe-script", type=Path, default=Path("/content/hkrl-repo/observe-collect.py"))
    ap.add_argument("--actor-bin", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=10845201)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0, help="0 = auto-size to CPU cores")
    ap.add_argument("--nographics", action="store_true")
    ap.add_argument("--no-batchmode", action="store_true")
    ap.add_argument("--soft-gl", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-losses", type=int, default=0)
    args = ap.parse_args()

    cpus = os.cpu_count() or 8
    workers = args.workers if args.workers > 0 else min(32, max(4, cpus - 4))
    if args.mode == "smoke":
        workers = 1
        seeds = [args.first_seed]
    else:
        workers = min(workers, args.episodes)
        seeds = list(range(args.first_seed, args.first_seed + args.episodes))

    print(
        "COLAB_LINUX_RUNNER_INIT",
        json.dumps({"cpus": cpus, "workers": workers, "mode": args.mode, "episodes": len(seeds), "nographics": args.nographics, "no_batchmode": args.no_batchmode, "soft_gl": args.soft_gl}),
        flush=True,
    )
    run_parallel_cohort(
        seeds=seeds,
        workers=workers,
        game_root=args.game_root,
        repo_root=args.repo_root,
        observe_script=args.observe_script,
        actor_bin=args.actor_bin,
        run_dir=args.run_dir,
        mode="eval",
        temperatures={},
        draw_seeds={},
        use_nographics=args.nographics,
        no_batchmode=args.no_batchmode,
        soft_gl=args.soft_gl,
        max_losses_early_stop=args.max_losses if args.max_losses > 0 else None,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()

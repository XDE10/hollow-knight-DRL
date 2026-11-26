#!/usr/bin/env python3
"""One-command playback & evaluation runner for the frozen Sisters of Battle champion.

Examples:
    python scripts/sisters/play.py
    python scripts/sisters/play.py --episodes 5 --seed 19900001
    python scripts/sisters/play.py --game-root "/path/to/Hollow Knight" --auto-install-mod
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "python", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hkrl.env import HKRLEnv
from hkrl.protocol import RewardEventKind as K
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config
from install_mod import discover_hollow_knight, discover_save_slot, install_mod, resolve_port

discover_game_root = discover_hollow_knight

DEFAULT_MODEL = REPO_ROOT / "models/sisters_champion/sisters_champion.bin"
DEFAULT_MODEL_URL = os.environ.get(
    "HKRL_MODEL_URL",
    "https://github.com/XDE10/hollow-knight-DRL/releases/download/"
    "v1.0.0-sisters-champion/sisters_champion.bin",
)
DEFAULT_MODEL_SHA256 = "f97d6aba7911a4983ce8a242edc8e8ef708743483ac7dcd1e57aef8132a43b08"
DEFAULT_TRAIN_CFG = REPO_ROOT / "configs/sisters_train.yaml"
DEFAULT_TASK_CFG = REPO_ROOT / "configs/sisters_task.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_default_model(model_path: Path) -> Path:
    """Auto-download the frozen champion binary from GitHub Releases on first run if absent."""
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    if model_path.resolve() != DEFAULT_MODEL.resolve():
        raise FileNotFoundError(f"Champion model not found: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".bin.download")
    print(f"[HKRL] Downloading frozen champion binary from GitHub Releases -> {model_path} ...", flush=True)
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, tmp)
    digest = _sha256(tmp)
    if digest != DEFAULT_MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded champion SHA-256 mismatch: {digest} != {DEFAULT_MODEL_SHA256}")
    tmp.replace(model_path)
    print(f"[HKRL] Verified champion binary (sha256={digest[:12]})", flush=True)
    return model_path


def find_game_exe(game_root: Path) -> Path:
    names = ("hollow_knight.exe", "hollow_knight.x86_64") if os.name == "nt" else ("hollow_knight.x86_64", "hollow_knight.exe")
    for name in names:
        exe = game_root / name
        if exe.is_file():
            return exe
    raise FileNotFoundError(f"Hollow Knight executable not found in {game_root}")


def read_runtime_conf(game_root: Path) -> dict[str, str]:
    conf = game_root / "hollow_knight_Data/Managed/Mods/HKRLEnvMod/hkrl-runtime.conf"
    values: dict[str, str] = {}
    if not conf.is_file():
        return values
    for raw in conf.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


def wait_for_port(port: int, proc: subprocess.Popen[bytes], timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def run_episodes(
    *,
    game_root: Path,
    model_bin: Path,
    episodes: int,
    first_seed: int,
    port: int | None,
    save_slot: int | None,
    headless: bool,
    width: int,
    height: int,
    auto_install_mod: bool = False,
) -> list[dict[str, Any]]:
    # Validate configs & files BEFORE spawning Hollow Knight so errors never leave orphan windows
    cfg = load_train_config(str(DEFAULT_TRAIN_CFG))
    task = load_task_config(str(DEFAULT_TASK_CFG))
    task.action.action_repeat = 16
    task.time_limit_seconds = 360
    noop: dict[str, Any] = {
        "movement_x": 1,
        "aim_y": 1,
        "buttons": np.zeros(9, dtype=np.int8),
        "duration": 3,
    }
    if task.action.enable_macro_actions:
        noop["macro"] = 0

    exe = find_game_exe(game_root)
    mod_dll = game_root / "hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll"
    if not mod_dll.is_file():
        if auto_install_mod:
            install_mod(game_root)
        else:
            raise FileNotFoundError(
                f"HKRLEnvMod.dll not found at {mod_dll}.\n"
                "Run `python scripts/install_mod.py` (or pass --auto-install-mod) to install it automatically."
            )

    runtime_conf = read_runtime_conf(game_root)
    eff_port = resolve_port(port, runtime_conf, default=cfg.transport.port)
    eff_slot, found_slots = discover_save_slot(save_slot, runtime_conf)
    if save_slot is None:
        slot_note = f"found slots {found_slots}" if found_slots else "no local user*.dat found, fallback"
        print(
            f"[HKRL] Save slot: {eff_slot} ({slot_note}; pass --save-slot N or set HKRL_SAVE_SLOT to override)",
            flush=True,
        )

    env_vars = dict(os.environ)
    for k in list(env_vars):
        if k.startswith("HKRL_NATIVE_") or k in ("HKRL_TACTICS_POLICY", "HKRL_TCP_ACTOR_PORT"):
            env_vars.pop(k, None)

    ledger_dir = tempfile.TemporaryDirectory(prefix="hkrl_play_")
    ledger_path = Path(ledger_dir.name) / "actor-ledger.txt"

    env_vars.update(
        {
            "HKRL_HOST": "127.0.0.1",
            "HKRL_PORT": str(eff_port),
            "HKRL_SAVE_SLOT": str(eff_slot),
            "HKRL_NATIVE_ACTOR": str(model_bin.resolve()),
            "HKRL_NATIVE_ACTOR_LEDGER": str(ledger_path),
            "HKRL_SISTERS_NAIL_MULTIPLIER": "1",
            "SteamAppId": "367520",
            "SteamGameId": "367520",
        }
    )
    if "HKRL_AUTH_TOKEN" in runtime_conf and "HKRL_AUTH_TOKEN" not in env_vars:
        env_vars["HKRL_AUTH_TOKEN"] = runtime_conf["HKRL_AUTH_TOKEN"]
    if headless:
        env_vars["HKRL_COMPACT_RENDER"] = "1"
        env_vars["SDL_AUDIODRIVER"] = "dummy"

    args = [
        str(exe),
        "-screen-fullscreen", "0",
        "-screen-width", str(width),
        "-screen-height", str(height),
    ]
    if headless:
        args.extend(["-batchmode", "-nographics"])

    print(
        f"[HKRL] Launching Sisters Champion ({model_bin.name}, sha256={_sha256(model_bin)[:12]}) "
        f"on port {eff_port}, save slot {eff_slot} ({'headless' if headless else f'{width}x{height} windowed'})...",
        flush=True,
    )
    proc = subprocess.Popen(args, cwd=str(game_root), env=env_vars)

    results: list[dict[str, Any]] = []
    game: HKRLEnv | None = None
    try:
        if not wait_for_port(eff_port, proc, timeout_s=90.0):
            raise RuntimeError(
                f"Hollow Knight did not listen on 127.0.0.1:{eff_port} (exit={proc.poll()}). "
                "Verify HK Modding API + HKRLEnvMod.dll are installed (see docs/setup_guide.md)."
            )

        if "HKRL_AUTH_TOKEN" in env_vars:
            os.environ["HKRL_AUTH_TOKEN"] = env_vars["HKRL_AUTH_TOKEN"]
        game = HKRLEnv(make_transport(cfg, port=eff_port), task, step_timeout_s=20.0)
        dmg_names = {1: "BOSS_BODY", 2: "ENEMY_PROJECTILE", 3: "SPIKES", 4: "OTHER"}

        for ep_i in range(episodes):
            seed = first_seed + ep_i
            t0 = time.monotonic()
            obs, _ = game.reset(seed=seed, options={"reset_timeout_s": 90.0})
            hp0, max_hp0, soul0 = (int(round(float(x))) for x in obs["player"][4:7])
            dmg_counts: dict[str, int] = {}
            win = False
            game_sec = 0.0

            while True:
                obs, _, term, trunc, info = game.step(noop)
                game_sec = float(obs["global"][4])
                arena_won = False
                for ev in info.get("reward_events", ()):
                    if ev.kind == K.DAMAGE_TAKEN:
                        amt = int(round(float(ev.amount)))
                        if amt > 0:
                            name = dmg_names.get(int(ev.aux_int2), "OTHER")
                            dmg_counts[name] = dmg_counts.get(name, 0) + amt
                    elif ev.kind == K.BOSS_KILLED and ev.entity_id == 0:
                        arena_won = True
                if term and arena_won:
                    win = True
                if term or trunc:
                    break

            total_dmg = sum(dmg_counts.values())
            no_hit = bool(win and total_dmg == 0 and hp0 == 9 and soul0 == 0)
            rec = {
                "episode": ep_i + 1,
                "seed": seed,
                "initial_masks": hp0,
                "initial_max_masks": max_hp0,
                "initial_soul": soul0,
                "win": win,
                "no_hit_win": no_hit,
                "total_damage": total_dmg,
                "damage_taken": dmg_counts,
                "game_sec": round(game_sec, 2),
                "wall_sec": round(time.monotonic() - t0, 1),
            }
            results.append(rec)
            print(
                f"  [Episode {ep_i + 1}/{episodes}] seed={seed} win={win} no_hit={no_hit} "
                f"dmg={total_dmg} ({dmg_counts}) game_t={rec['game_sec']}s wall_t={rec['wall_sec']}s",
                flush=True,
            )
    finally:
        if game is not None:
            try:
                game.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        ledger_dir.cleanup()

    wins = sum(int(r["win"]) for r in results)
    no_hits = sum(int(r["no_hit_win"]) for r in results)
    avg_dmg = sum(int(r["total_damage"]) for r in results) / max(1, len(results))
    print(
        f"[SUMMARY] episodes={len(results)} wins={wins}/{len(results)} "
        f"no_hit={no_hits}/{len(results)} avg_dmg={avg_dmg:.3f}",
        flush=True,
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Play or evaluate the frozen Sisters of Battle champion.")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to frozen native actor .bin")
    ap.add_argument("--game-root", type=Path, default=None, help="Path to Hollow Knight directory (auto-detected if omitted)")
    ap.add_argument("--episodes", type=int, default=1, help="Number of fights to run")
    ap.add_argument("--seed", type=int, default=19900001, help="First deterministic evaluation seed")
    ap.add_argument("--port", type=int, default=None, help="Local TCP port for HKRLEnvMod (default: HKRL_PORT or 5555)")
    ap.add_argument("--save-slot", type=int, default=None, choices=(1, 2, 3, 4), help="Godhome save slot 1-4 (auto-detected if omitted)")
    ap.add_argument("--headless", action="store_true", help="Run headless (-batchmode -nographics)")
    ap.add_argument("--width", type=int, default=1280, help="Window width when not headless")
    ap.add_argument("--height", type=int, default=720, help="Window height when not headless")
    ap.add_argument("--auto-install-mod", action="store_true", help="Auto-download HKRLEnvMod.dll if missing")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON path to save episode results")
    args = ap.parse_args()

    model_bin = ensure_default_model(args.model)
    game_root = discover_hollow_knight(args.game_root)
    results = run_episodes(
        game_root=game_root,
        model_bin=model_bin,
        episodes=args.episodes,
        first_seed=args.seed,
        port=args.port,
        save_slot=args.save_slot,
        headless=args.headless,
        width=args.width,
        height=args.height,
        auto_install_mod=args.auto_install_mod,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"[SAVED] {args.output}", flush=True)


if __name__ == "__main__":
    main()

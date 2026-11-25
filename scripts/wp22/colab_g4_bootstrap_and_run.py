"""One-shot Colab G4 Bootstrapper for High-Concurrency Native Linux Hollow Knight (Eval + Online No-Hit PPO).

Controlled via environment variables:
  HKRL_CLOUD_TASK   : "smoke_and_confirm30" | "online_ppo" | "formal100"
  HKRL_CLOUD_MODEL  : "m_disc_384_2lgru_pure" | "g_attn_320_2lgru" | "j_disc_320_1lgru_hi"
  HKRL_CLOUD_WORKERS: "0" (auto) or explicit int e.g. "16", "24"
  HKRL_FIRST_SEED   : e.g. "10900031" or "10910001"
  HKRL_RL_ROUNDS    : e.g. "6"
  HKRL_RL_ROLLOUTS  : e.g. "64"
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch


def run_cmd(cmd: list[str] | str, *, cwd: Path | None = None, check: bool = True, env: dict | None = None) -> int:
    """Stream child stdout/stderr directly into Python sys.stdout so Jupyter/colab-cli shows live output."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=isinstance(cmd, str),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def ensure_linux_deps() -> None:
    print("=== [1/4] Installing headless Linux Unity deps & python packages ===", flush=True)
    if not shutil.which("rclone"):
        run_cmd("curl -fsSL https://rclone.org/install.sh | bash")
    if not (shutil.which("xvfb-run") and shutil.which("pigz")):
        run_cmd(
            "apt-get -qq update && "
            "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends "
            "xvfb xauth libgl1 libglu1-mesa libgl1-mesa-dri libxcursor1 libxrandr2 libxi6 libxinerama1 libpulse0 pigz >/dev/null 2>&1 "
            "&& (DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends libasound2t64 >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends libasound2 >/dev/null 2>&1 || true)"
        )
    missing = []
    for mod, pkg in (("flatbuffers", "flatbuffers"), ("gymnasium", "gymnasium"), ("psutil", "psutil"), ("yaml", "pyyaml"), ("pydantic", "pydantic")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", *missing])


def pull_from_drive(conf: Path, model_name: str) -> None:
    print(f"=== [2/4] Pulling code, opener, game, and {model_name} from Drive ===", flush=True)
    t0 = time.monotonic()
    stage = Path("/content/hkrl-stage")
    stage.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        ("mydrive:HKRL-Linux/hkrl-cloud-code.tar.gz", str(stage)),
        ("mydrive:HKRL-Linux/models/opener-wp22-attn139.pt", str(stage / "models")),
        (f"mydrive:HKRL-Linux/models/{model_name}", str(stage / "models" / model_name)),
    ):
        run_cmd(["rclone", "--config", str(conf), "copy", "--transfers", "8", "--checkers", "8", src, dst])

    repo_root = Path("/content/hkrl-repo")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    run_cmd(["tar", "-C", "/content", "-xzf", str(stage / "hkrl-cloud-code.tar.gz")])

    waited = 0
    while True:
        chk = subprocess.run(
            ["rclone", "--config", str(conf), "lsf", "mydrive:HKRL-Linux/hk-linux-manifest.sha256"],
            capture_output=True, text=True,
        )
        if chk.returncode == 0 and "hk-linux-manifest.sha256" in chk.stdout:
            break
        if waited % 30 == 0:
            print(f"G4 PREWARMED & READY! Waiting for hk-linux-manifest.sha256 ({waited}s)...", flush=True)
        time.sleep(10)
        waited += 10

    run_cmd(["rclone", "--config", str(conf), "copy", "--transfers", "8", "--checkers", "8", "mydrive:HKRL-Linux/hk-linux-1.5.78-v2.tar.gz", str(stage)])

    game_root = Path("/content/hk-linux-base")
    if not (game_root / "hollow_knight.x86_64").is_file():
        if game_root.exists():
            shutil.rmtree(game_root)
        game_root.mkdir(parents=True)
        run_cmd(f'pigz -dc "{stage / "hk-linux-1.5.78-v2.tar.gz"}" | tar -C "{game_root}" -xf -')
        os.chmod(game_root / "hollow_knight.x86_64", 0o755)
    (game_root / "steam_appid.txt").write_text("367520\n", encoding="utf-8")
    boot_cfg = game_root / "hollow_knight_Data/boot.config"
    if boot_cfg.is_file():
        btxt = boot_cfg.read_text(encoding="utf-8", errors="ignore")
        blines = [l for l in btxt.splitlines() if "single-instance" not in l.lower()]
        boot_cfg.write_text("\n".join(blines) + "\n", encoding="utf-8")
    print(f"PULLED_AND_EXTRACTED in {time.monotonic() - t0:.1f}s (cpus={os.cpu_count()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'})", flush=True)


def main() -> None:
    task = os.environ.get("HKRL_CLOUD_TASK", "smoke_and_confirm30")
    model_name = os.environ.get("HKRL_CLOUD_MODEL", "m_disc_384_2lgru_pure")
    workers = int(os.environ.get("HKRL_CLOUD_WORKERS", "0"))
    first_seed = int(os.environ.get("HKRL_FIRST_SEED", "10900031"))
    rl_rounds = int(os.environ.get("HKRL_RL_ROUNDS", "8"))
    rl_rollouts = int(os.environ.get("HKRL_RL_ROLLOUTS", "64"))
    rl_resume = os.environ.get("HKRL_RL_RESUME", "0") == "1"
    conf = Path("/content/.hk-rclone.conf")

    subprocess.run("pkill -9 -f hollow_knight.x86_64 2>/dev/null || true; pkill -9 -f Xvfb 2>/dev/null || true", shell=True)
    ensure_linux_deps()
    pull_from_drive(conf, model_name)

    repo_root = Path("/content/hkrl-repo")
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "scripts"))
    sys.path.insert(0, str(repo_root / "python"))
    from colab_linux_online_ppo import export_native_from_pt

    m_dir = Path("/content/hkrl-stage/models") / model_name
    opener_pt = Path("/content/hkrl-stage/models/opener-wp22-attn139.pt")
    learner_pt = m_dir / "learner-wp22-skill.pt"
    orig_bin = m_dir / "world_wp22_skill_scaledopener.bin"
    upgraded_bin = m_dir / f"{model_name}_v2opener139.bin"
    exp_meta = export_native_from_pt(learner_pt, upgraded_bin, orig_bin, opener_pt=opener_pt)
    print("UPGRADED_V2_OPENER_BIN", json.dumps({"bin": str(upgraded_bin), "sha256": exp_meta["checkpoint_sha256"], "opening_sha256": exp_meta["opening_sha256"]}), flush=True)

    py_env = dict(os.environ)
    py_env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'scripts'}:{repo_root / 'python'}"

    if os.environ.get("HKRL_SKIP_SMOKE", "0") == "1":
        winning_flags = ["--nographics"]
        print("=== [3/4] Skipping redundant smoke probe (HKRL_SKIP_SMOKE=1, using --nographics) ===", flush=True)
    else:
        print("=== [3/4] Running 1-episode Colab Linux smoke probe (auto-selecting launch mode) ===", flush=True)
        probe_modes = [
            ("batch_nographics", ["--nographics"]),
            ("xvfb_glcore_softgl_batch", ["--soft-gl"]),
            ("xvfb_glcore_softgl_nobatch", ["--no-batchmode", "--soft-gl"]),
            ("xvfb_glcore_hw_batch", []),
        ]
        winning_flags = None
        for mode_name, extra_flags in probe_modes:
            smoke_dir = Path(f"/content/hkrl-runs/smoke-{model_name}-{mode_name}")
            if smoke_dir.exists():
                shutil.rmtree(smoke_dir)
            print(f"--- Smoke probe mode={mode_name} flags={extra_flags} ---", flush=True)
            rc = run_cmd(
                [
                    sys.executable, "-u", str(repo_root / "colab_linux_multi_runner.py"),
                    "--mode", "smoke",
                    "--actor-bin", str(upgraded_bin),
                    "--run-dir", str(smoke_dir),
                    "--first-seed", "10845201",
                    "--max-attempts", "1",
                    *extra_flags,
                ],
                cwd=repo_root,
                env=py_env,
                check=False,
            )
            if rc == 0:
                winning_flags = extra_flags
                print(f"COLAB_LINUX_SMOKE_OK! winning_mode={mode_name} flags={winning_flags}", flush=True)
                break
        if winning_flags is None:
            raise RuntimeError("All 4 Colab Linux Unity launch modes failed smoke probe!")

    print(f"=== [4/4] Executing cloud task: {task} (model={model_name}, workers={workers}, flags={winning_flags}) ===", flush=True)
    out_tag = f"{task}-{model_name}"
    run_out = Path("/content/hkrl-runs") / out_tag
    if task == "smoke_and_confirm30":
        run_cmd(
            [
                sys.executable, "-u", str(repo_root / "colab_linux_multi_runner.py"),
                "--mode", "eval",
                "--actor-bin", str(upgraded_bin),
                "--run-dir", str(run_out),
                "--first-seed", str(first_seed),
                "--episodes", "30",
                "--workers", str(workers),
                *winning_flags,
            ],
            cwd=repo_root,
            env=py_env,
        )
    elif task == "formal100":
        formal_workers = workers if workers > 0 else 34
        run_cmd(
            [
                sys.executable, "-u", str(repo_root / "colab_linux_multi_runner.py"),
                "--mode", "eval",
                "--actor-bin", str(upgraded_bin),
                "--run-dir", str(run_out),
                "--first-seed", str(first_seed),
                "--episodes", "100",
                "--workers", str(formal_workers),
                "--max-losses", "6",
                *winning_flags,
            ],
            cwd=repo_root,
            env=py_env,
        )
        if conf.is_file():
            dst = f"mydrive:HKRL-Linux/results/{out_tag}"
            for fname in ("cloud-summary.json", "STOP_UNREACHABLE"):
                p = run_out / fname
                if p.is_file():
                    run_cmd(["rclone", "--config", str(conf), "copy", str(p), dst], check=False)
    elif task == "online_ppo":
        cmd = [
            sys.executable, "-u", str(repo_root / "colab_linux_online_ppo.py"),
            "--init-learner", str(learner_pt),
            "--init-native", str(orig_bin),
            "--world", str(m_dir / "world/world.pt"),
            "--world-val", str(m_dir / "world/validation.json"),
            "--opener-pt", str(opener_pt),
            "--work-dir", str(run_out),
            "--rounds", str(rl_rounds),
            "--rollouts-per-round", str(rl_rollouts),
            "--val-episodes", "30",
            "--workers", str(workers),
            "--rclone-conf", str(conf),
            "--drive-dest", f"mydrive:HKRL-Linux/results/{out_tag}",
            *winning_flags,
        ]
        if rl_resume:
            cmd.append("--resume")
        run_cmd(cmd, cwd=repo_root, env=py_env)
    else:
        raise ValueError(f"Unknown task: {task}")

    if conf.is_file():
        dst = f"mydrive:HKRL-Linux/results/{out_tag}"
        for fname in ("cloud-summary.json", "rl-history.json", "champion.pt", "champion.bin"):
            p = run_out / fname
            if p.is_file():
                run_cmd(["rclone", "--config", str(conf), "copy", str(p), dst])
        conf.unlink()
    print(f"COLAB_CLOUD_TASK_COMPLETE {out_tag}", flush=True)


if __name__ == "__main__":
    main()

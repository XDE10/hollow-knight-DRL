#!/usr/bin/env python3
"""Remote launcher: migrate best g_attn checkpoint onto colab-d slot2 and launch V4 True Radiant Curriculum PPO."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(cmd: list[str] | str, *, shell: bool = False) -> None:
    cp = subprocess.run(cmd, shell=shell, text=True, capture_output=True)
    if cp.stdout:
        print(cp.stdout, end="", flush=True)
    if cp.stderr:
        print(cp.stderr, file=sys.stderr, end="", flush=True)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed (rc={cp.returncode}): {cmd}")


def main() -> None:
    conf = Path("/content/.hk-rclone.conf")
    stage = Path("/content/hkrl-stage")
    repo = Path("/content/hkrl-repo")
    work = Path("/content/hkrl-runs/online_ppo-g_attn_radiant")
    drive_src = "mydrive:HKRL-Linux/results/online_ppo-g_attn_320_2lgru"
    drive_dest = "mydrive:HKRL-Linux/results/online_ppo-g_attn_radiant"
    model = "g_attn_320_2lgru"

    if not conf.is_file() or conf.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {conf}")

    for pat in ("[r]un_ppo_", "[r]un_v2_", "[r]un_v4_", "[c]olab_linux_online_ppo.py"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(1)
    for pat in ("[h]ollow_knight.x86_64", "[X]vfb"):
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(1)

    stage.mkdir(parents=True, exist_ok=True)
    run(["rclone", "--config", str(conf), "copy", "--transfers", "4", "mydrive:HKRL-Linux/hkrl-cloud-code.tar.gz", str(stage)])
    if repo.exists():
        shutil.rmtree(repo)
    run(["tar", "-C", "/content", "-xzf", str(stage / "hkrl-cloud-code.tar.gz")])

    m_dir = stage / "models" / model
    if not (m_dir / "learner-wp22-skill.pt").is_file() or not (m_dir / "world/world.pt").is_file():
        run(["rclone", "--config", str(conf), "copy", "--transfers", "8", "--checkers", "8", f"mydrive:HKRL-Linux/models/{model}", str(m_dir)])

    work.mkdir(parents=True, exist_ok=True)
    if not (work / "champion.pt").is_file():
        for fn in ("champion.pt", "champion.bin", "latest.pt", "latest.bin", "rl-history.json", "elite-pool.tar.gz"):
            subprocess.run(["rclone", "--config", str(conf), "copy", f"{drive_src}/{fn}", str(work)], check=False)
        if not (work / "latest.pt").is_file() and (work / "champion.pt").is_file():
            shutil.copy2(work / "champion.pt", work / "latest.pt")
            shutil.copy2(work / "champion.bin", work / "latest.bin")
        ep_tar = work / "elite-pool.tar.gz"
        if ep_tar.is_file() and not (work / "elite-pool/elite-index.json").is_file():
            run(["tar", "-C", str(work), "-xzf", str(ep_tar)])
        # Reset rl-history round index cleanly if needed or resume directly from R8 -> R9!
        print("MIGRATED_G_ATTN_TO_COLAB_D_RADIANT_OK", flush=True)

    for stale in (work / "DONE", work / "FAILED"):
        if stale.exists():
            stale.unlink()

    wrapper = Path("/tmp/run_v4_g_attn_radiant.sh")
    wrapper.write_text(
        f"""#!/usr/bin/env bash
set -u
export PYTHONPATH=/content/hkrl-repo:/content/hkrl-repo/scripts:/content/hkrl-repo/python
cd /content/hkrl-runs
echo "=== [V4 TRUE RADIANT CURRICULUM PPO (2-HIT -> 1-HIT) AT $(date -Iseconds)] ===" >> {work}/live.log
python3 -u /content/hkrl-repo/colab_linux_online_ppo.py \\
  --init-learner /content/hkrl-stage/models/{model}/learner-wp22-skill.pt \\
  --init-native /content/hkrl-stage/models/{model}/world_wp22_skill_scaledopener.bin \\
  --world /content/hkrl-stage/models/{model}/world/world.pt \\
  --world-val /content/hkrl-stage/models/{model}/world/validation.json \\
  --opener-pt /content/hkrl-stage/models/opener-wp22-attn139.pt \\
  --work-dir {work} \\
  --rounds 60 --rollouts-per-round 64 --val-episodes 30 \\
  --workers 32 --updates 20 --actor-lr 3.2e-5 --kl-limit 0.045 --clip-eps 0.18 \\
  --temps 0.14,0.24,0.34 \\
  --proj-pen 2.40 --body-pen 2.00 --first-hit-pen 5.50 --iframe-zero-sec 1.30 \\
  --hitless-dps-bonus 0.010 --post-hit-reward-scale 0.20 --post-hit-ppo-weight 0.60 \\
  --no-hit-bonus 8.00 --one-hit-bonus 0.50 \\
  --pre-proj-mult 2.8 --pre-body-mult 2.1 \\
  --offline-kl-coef 0.0 --offline-kl-decay-rounds 0 \\
  --anchor-kl-coef 0.005 --behavior-kl-coef 0.008 --dash-drift-mult 1.60 \\
  --critic-warmup 14 --value-clip 1.0 \\
  --elite-bc-coef 0.025 --elite-batch 14 \\
  --continuous --val-every 2 --collapse-min-wins 27 --collapse-dmg-margin 1.50 \\
  --auto-curriculum --radiant-curriculum \\
  --radiant-max-hits-s1 2 --radiant-max-hits-s2 2 --radiant-max-hits-s3 1 --radiant-fail-pen 4.5 \\
  --auto-formal100 --formal100-first-seed 12400001 \\
  --first-train-seed 14000001 \\
  --holdout-every 4 --holdout-episodes 32 --first-holdout-seed 11600001 \\
  --resume --nographics \\
  --rclone-conf /content/.hk-rclone.conf \\
  --drive-dest {drive_dest} \\
  >> {work}/live.log 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "EXIT_RC=0 AT $(date -Iseconds)" > {work}/DONE
else
  echo "EXIT_RC=$rc AT $(date -Iseconds)" > {work}/FAILED
fi
""",
        encoding="utf-8",
    )
    os.chmod(wrapper, 0o755)
    proc = subprocess.Popen(
        ["nohup", "bash", str(wrapper)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    (work / "ppo.pid").write_text(str(proc.pid), encoding="utf-8")
    time.sleep(2)
    if proc.poll() is not None:
        tail = (work / "live.log").read_text(errors="ignore")[-1500:] if (work / "live.log").is_file() else ""
        raise RuntimeError(f"g_attn_radiant died immediately rc={proc.returncode}:\n{tail}")
    print(f"LAUNCHED_V4_RADIANT pid={proc.pid}", flush=True)


if __name__ == "__main__":
    main()

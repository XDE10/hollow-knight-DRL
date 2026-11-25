#!/usr/bin/env python3
"""Unified A/B concurrent G4 swarm hunter (colab-d at 3/3 cap, 4s cycle, strictly ZERO release)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time

WP22 = Path("/mnt/d/CodexArtifacts/HKRL-Handoff/WP22-ScaledDRL")
SCRATCH = Path("/mnt/c/Users/Mr.Z/.gemini/antigravity-cli/brain/70eb9b29-5500-478b-ad3a-eef89836b441/scratch")
BIN = Path.home() / ".local/bin"
RCLONE_CONF = Path.home() / ".config/rclone/rclone.conf"
HUNT_LOG = WP22 / "hunt_abd_swarm.log"
NOTIFY_PS1 = r"D:\CodexArtifacts\HKRL-Handoff\WP22-ScaledDRL\notify_windows.ps1"

JOBS = [
    {
        "id": "g_attn_agg",
        "sess": "hkrl-g4-g_attn_agg",
        "eligible": ("colab-b",),
        "script": SCRATCH / "remote_start_g_attn_agg.py",
        "env": {},
        "expect": "LAUNCHED_G_ATTN_AGG",
        "label": "g_attn_agg 7/30分支加速无伤",
    },
    {
        "id": "formal100_a",
        "sess": "hkrl-g4-formal100",
        "eligible": ("colab-a",),
        "script": SCRATCH / "remote_start_formal100_a.py",
        "env": {},
        "expect": "LAUNCHED_FORMAL100",
        "label": "colab-a 34路100场验收",
    },
    {
        "id": "slot2",
        "sess": "hkrl-g4-slot2",
        "eligible": ("colab-b",),
        "script": SCRATCH / "remote_start_slot2_d.py",
        "env": {},
        "expect": "LAUNCHED_SLOT2_D",
        "label": "formal100 + j_disc_agg Stage A-Fast",
    },
]

ACCOUNT_MAX_NEW = {"colab-b": 2, "colab-a": 1}

LOG_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
ACCT_LOCKS = {a: threading.Lock() for a in ACCOUNT_MAX_NEW}

pending: list[dict] = list(JOBS)
claimed: set[str] = set()
in_flight: set[str] = set()
acquired_by_acct = {a: 0 for a in ACCOUNT_MAX_NEW}
launch_threads: list[threading.Thread] = []


def ts() -> str:
    return time.strftime("%H:%M:%S")


def hlog(msg: str) -> None:
    line = f"[{ts()}] {msg}\n"
    with LOG_LOCK:
        print(line, end="", flush=True)
        with HUNT_LOG.open("a", encoding="utf-8") as f:
            f.write(line)


def notify_windows(title: str, message: str) -> None:
    try:
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-WindowStyle",
                "Hidden",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                NOTIFY_PS1,
                "-Title",
                title,
                "-Message",
                message,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def has_session(acct: str, sess: str) -> bool:
    try:
        cp = subprocess.run(
            ["timeout", "15s", str(BIN / acct), "sessions"],
            capture_output=True,
            text=True,
            timeout=18,
        )
        return any(sess in line for line in cp.stdout.splitlines())
    except Exception:
        return False


def formal100_already_done_on_slot2() -> bool:
    for p in (WP22 / "live-slot2-formal100.log",):
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if '"episodes_done": 100' in txt:
                return True
    return False


def bootstrap_and_launch(acct: str, job: dict) -> None:
    sess = job["sess"]
    label = job["label"]
    up_ok = False
    with ACCT_LOCKS[acct]:
        for u_try in range(1, 4):
            try:
                up = subprocess.run(
                    ["timeout", "45s", str(BIN / acct), "upload", str(RCLONE_CONF), "/content/.hk-rclone.conf", "-s", sess],
                    capture_output=True,
                    text=True,
                    timeout=50,
                )
                if up.returncode == 0:
                    up_ok = True
                    break
                hlog(f"[{acct}/{sess}] rclone upload try {u_try} rc={up.returncode}: {up.stderr.strip()[:160]}")
            except Exception as exc:
                hlog(f"[{acct}/{sess}] rclone upload try {u_try} err={exc}")
            time.sleep(3)

        if not up_ok:
            hlog(f"[{acct}/{sess}] ERROR: rclone upload failed (G4 kept alive!).")
            notify_windows(f"HKRL {acct} 上传配置失败", f"{sess} 已保活但未启动，请检查")
            return

        hlog(f"[{acct}/{sess}] Bootstrapping & launching {label}...")
        cmd = [
            "timeout",
            "480s",
            str(BIN / acct),
            "exec",
            "-s",
            sess,
            "--timeout",
            "420",
        ]
        for k, v in job["env"].items():
            cmd.extend(["--env", f"{k}={v}"])
        cmd.extend(["-f", str(job["script"])])

        try:
            ex = subprocess.run(cmd, capture_output=True, text=True, timeout=490)
            out_tail = (ex.stdout or "").strip()[-800:]
            err_tail = (ex.stderr or "").strip()[-400:]
            hlog(f"[{acct}/{sess}] launch exec rc={ex.returncode} tail:\n{out_tail}")
            if err_tail:
                hlog(f"[{acct}/{sess}] launch stderr:\n{err_tail}")
            if job["expect"] in (ex.stdout or ""):
                notify_windows(f"HKRL {acct} 新G4已开跑", f"{label} ({sess}) 已启动，全程保活中")
            else:
                notify_windows(f"HKRL {acct} 启动待核查", f"{sess} 已抢到并保活，请查看 hunt_abd_swarm.log")
        except Exception as exc:
            hlog(f"[{acct}/{sess}] launch exception (G4 kept alive!): {exc}")


def account_hunter(acct: str) -> None:
    req_env = os.environ.copy()
    req_env["COLAB_CLI_ASSIGN_RETRIES"] = "0"
    attempt = 0
    precond_streak = 0

    while True:
        job: dict | None = None
        with STATE_LOCK:
            if acquired_by_acct[acct] >= ACCOUNT_MAX_NEW[acct]:
                hlog(f"[{acct}] Reached max new G4 cap ({ACCOUNT_MAX_NEW[acct]}); hunter finished cleanly.")
                return
            formal_via_slot2 = ("slot2" in claimed) or formal100_already_done_on_slot2()
            if acct == "colab-a" and formal_via_slot2:
                hlog("[colab-a] formal100 already covered via slot2; colab-a hunter finished cleanly.")
                return
            formal_covered = ("formal100_a" in claimed) or formal100_already_done_on_slot2()
            if acct == "colab-b" and ("g_attn_agg" in claimed) and formal_covered:
                hlog("[colab-b] g_attn_agg claimed and formal100 covered by colab-a; keeping 2-G4 CU runway.")
                return
            eligible_left = [j for j in JOBS if acct in j["eligible"] and j["id"] not in claimed]
            if not eligible_left:
                hlog(f"[{acct}] All eligible jobs claimed; hunter finished cleanly.")
                return
            for i, cand in enumerate(pending):
                if cand["id"] == "slot2" and (("g_attn_agg" not in claimed) or formal_covered):
                    continue
                if acct in cand["eligible"] and cand["id"] not in claimed and cand["id"] not in in_flight:
                    job = pending.pop(i)
                    in_flight.add(job["id"])
                    break

        if job is None:
            time.sleep(1.0)
            continue

        attempt += 1
        sess = job["sess"]
        jid = job["id"]
        hlog(f"[{acct}/{sess}] Attempt {attempt} -> requesting G4...")
        t0 = time.monotonic()
        acquired = False
        err_line = ""
        rc = -1
        with ACCT_LOCKS[acct]:
            if has_session(acct, sess):
                acquired = True
                hlog(f"[{acct}/{sess}] Existing session found; reusing directly.")
            else:
                try:
                    cp = subprocess.run(
                        ["timeout", "48s", str(BIN / acct), "new", "-s", sess, "--gpu", "G4"],
                        env=req_env,
                        capture_output=True,
                        text=True,
                        timeout=52,
                    )
                    rc = cp.returncode
                    lines = (cp.stderr or cp.stdout or "").strip().splitlines()
                    err_line = lines[-1] if lines else ""
                    if cp.returncode == 0 or has_session(acct, sess):
                        acquired = True
                except Exception as exc:
                    err_line = str(exc)

        elapsed = time.monotonic() - t0
        if acquired:
            precond_streak = 0
            with STATE_LOCK:
                in_flight.discard(jid)
                claimed.add(jid)
                acquired_by_acct[acct] += 1
                lt = threading.Thread(target=bootstrap_and_launch, args=(acct, job), daemon=False)
                launch_threads.append(lt)
            hlog(f"[{acct}/{sess}] +++ ACQUIRED G4 on attempt {attempt} in {elapsed:.1f}s! +++")
            lt.start()
            time.sleep(2.0)
        else:
            low_err = err_line.lower()
            if "precondition failed" in low_err or "too many active sessions" in low_err:
                precond_streak += 1
            else:
                precond_streak = 0
            with STATE_LOCK:
                in_flight.discard(jid)
                if jid not in claimed:
                    pending.append(job)
            hlog(f"[{acct}/{sess}] miss ({elapsed:.1f}s) rc={rc}: {err_line}")
            if precond_streak >= 2:
                hlog(f"[{acct}] Stopping hunter after {precond_streak} consecutive precondition/session-cap refusals.")
                return
            time.sleep(4.0)


def main() -> None:
    hlog("Starting unified A/B concurrent G4 swarm (colab-d at 3/3 cap, 4s interval, strict zero-release)...")
    hunters = [threading.Thread(target=account_hunter, args=(acct,), daemon=False) for acct in ACCOUNT_MAX_NEW]
    for th in hunters:
        th.start()
    for th in hunters:
        th.join()
    for lt in list(launch_threads):
        lt.join()
    hlog("All A/B/D swarm hunter & launcher threads finished.")


if __name__ == "__main__":
    main()

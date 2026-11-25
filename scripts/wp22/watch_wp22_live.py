#!/usr/bin/env python3
"""Hold-alive & alert-only WSL supervisor for Colab G4 WP22 jobs (NEVER releases any card)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

WP22 = Path("/mnt/d/CodexArtifacts/HKRL-Handoff/WP22-ScaledDRL")
BIN = Path.home() / ".local/bin"
WATCH_LOG = WP22 / "watch_wp22_live.log"
NOTIFY_PS1 = r"D:\CodexArtifacts\HKRL-Handoff\WP22-ScaledDRL\notify_windows.ps1"

PROBE_CODE = r'''
import json, os, subprocess
from pathlib import Path

rdir = Path(os.environ["HKRL_WATCH_DIR"])
pid_name = os.environ.get("HKRL_WATCH_PID", "ppo.pid")
offset = int(os.environ.get("HKRL_LOG_OFFSET", "0"))
out = {"exists": rdir.is_dir(), "alive": False, "done": False, "failed": False, "gpu": "", "log_offset": offset, "log_chunk": ""}
if rdir.is_dir():
    pf = rdir / pid_name
    if pf.is_file():
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, 0)
            out["alive"] = True
            out["pid"] = pid
        except Exception:
            out["alive"] = False
    done_f = rdir / "DONE"
    fail_f = rdir / "FAILED"
    if fail_f.is_file():
        out["failed"] = True
        out["failed_text"] = fail_f.read_text(errors="ignore")[:300]
    if done_f.is_file():
        dtxt = done_f.read_text(errors="ignore")[:300]
        out["done_text"] = dtxt
        if "EXIT_RC=0" in dtxt:
            out["done"] = True
    lf = rdir / "live.log"
    if lf.is_file():
        sz = lf.stat().st_size
        if sz < offset:
            offset = 0
        with lf.open("rb") as f:
            f.seek(offset)
            chunk = f.read(32768)
        out["log_offset"] = offset + len(chunk)
        out["log_chunk"] = chunk.decode("utf-8", errors="replace")
        if "CLOUD_ONLINE_PPO_ALL_DONE" in out["log_chunk"]:
            out["done"] = True
    hf = rdir / "rl-history.json"
    if hf.is_file():
        try:
            out["history"] = json.loads(hf.read_text(encoding="utf-8"))
        except Exception:
            pass
    sf = rdir / "cloud-summary.json"
    if sf.is_file():
        try:
            s = json.loads(sf.read_text(encoding="utf-8"))
            out["summary"] = {k: s.get(k) for k in ("episodes_done", "episodes_target", "wins", "no_hit_wins", "avg_damage", "damage_totals", "early_stopped")}
        except Exception:
            pass
out["gpu"] = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true").strip()
print("WATCH_PROBE_JSON:" + json.dumps(out, ensure_ascii=False))
'''

PROBE_FILE = Path("/tmp/hkrl_watch_probe.py")


def ts() -> str:
    return time.strftime("%H:%M:%S")


def wlog(msg: str) -> None:
    line = f"[{ts()}] {msg}\n"
    print(line, end="", flush=True)
    with WATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def notify_windows(title: str, message: str) -> None:
    wlog(f"[ALERT] {title}: {message}")
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
    except Exception as exc:
        wlog(f"[ALERT-ERR] {exc}")


def check_sessions(acct: str) -> list[str]:
    try:
        cp = subprocess.run(
            ["timeout", "12s", str(BIN / acct), "sessions"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return [l.strip() for l in cp.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def check_cu_balance(acct: str) -> float | None:
    try:
        cp = subprocess.run(
            ["timeout", "12s", str(BIN / acct), "usage"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in cp.stdout.splitlines():
            if "Current balance:" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return None


def main() -> None:
    PROBE_FILE.write_text(PROBE_CODE, encoding="utf-8")
    targets: list[dict[str, Any]] = [
        {
            "id": "j_disc",
            "acct": "colab-d",
            "sess": "hkrl-g4-j_disc",
            "remote_dir": "/content/hkrl-runs/online_ppo-j_disc_320_1lgru_hi",
            "pid_file": "ppo.pid",
            "local_log": WP22 / "live-j_disc.log",
            "kind": "ppo",
            "drive_dest": "mydrive:HKRL-Linux/results/online_ppo-j_disc_320_1lgru_hi",
        },
        {
            "id": "g_attn",
            "acct": "colab-b",
            "sess": "hkrl-g4-g_attn",
            "remote_dir": "/content/hkrl-runs/online_ppo-g_attn_320_2lgru",
            "pid_file": "ppo.pid",
            "local_log": WP22 / "live-g_attn.log",
            "kind": "ppo",
            "drive_dest": "mydrive:HKRL-Linux/results/online_ppo-g_attn_320_2lgru",
        },
        {
            "id": "formal100",
            "acct": "colab-a",
            "sess": "hkrl-g4-formal100",
            "remote_dir": "/content/hkrl-runs/formal100-j_disc_320_1lgru_hi",
            "pid_file": "formal.pid",
            "local_log": WP22 / "live-formal100.log",
            "kind": "formal",
            "drive_dest": "mydrive:HKRL-Linux/results/formal100-j_disc_320_1lgru_hi",
        },
        {
            "id": "slot2_formal",
            "acct": "colab-d",
            "sess": "hkrl-g4-slot2",
            "remote_dir": "/content/hkrl-runs/formal100-j_disc_320_1lgru_hi",
            "pid_file": "slot2.pid",
            "local_log": WP22 / "live-slot2-formal100.log",
            "kind": "formal",
            "drive_dest": "mydrive:HKRL-Linux/results/formal100-j_disc_320_1lgru_hi",
        },
        {
            "id": "j_disc_agg",
            "acct": "colab-d",
            "sess": "hkrl-g4-slot2",
            "remote_dir": "/content/hkrl-runs/online_ppo-j_disc_agg",
            "pid_file": "slot2.pid",
            "local_log": WP22 / "live-j_disc_agg.log",
            "kind": "ppo",
            "drive_dest": "mydrive:HKRL-Linux/results/online_ppo-j_disc_agg",
        },
        {
            "id": "b_slot2_formal",
            "acct": "colab-b",
            "sess": "hkrl-g4-slot2",
            "remote_dir": "/content/hkrl-runs/formal100-j_disc_320_1lgru_hi",
            "pid_file": "slot2.pid",
            "local_log": WP22 / "live-slot2-formal100.log",
            "kind": "formal",
            "drive_dest": "mydrive:HKRL-Linux/results/formal100-j_disc_320_1lgru_hi",
        },
        {
            "id": "b_j_disc_agg",
            "acct": "colab-b",
            "sess": "hkrl-g4-slot2",
            "remote_dir": "/content/hkrl-runs/online_ppo-j_disc_agg",
            "pid_file": "slot2.pid",
            "local_log": WP22 / "live-j_disc_agg.log",
            "kind": "ppo",
            "drive_dest": "mydrive:HKRL-Linux/results/online_ppo-j_disc_agg",
        },
        {
            "id": "d_g_attn_agg",
            "acct": "colab-d",
            "sess": "hkrl-g4-g_attn_agg",
            "remote_dir": "/content/hkrl-runs/online_ppo-g_attn_agg",
            "pid_file": "ppo.pid",
            "local_log": WP22 / "live-g_attn_agg.log",
            "kind": "ppo",
            "drive_dest": "mydrive:HKRL-Linux/results/online_ppo-g_attn_agg",
        },
        {
            "id": "b_g_attn_agg",
            "acct": "colab-b",
            "sess": "hkrl-g4-g_attn_agg",
            "remote_dir": "/content/hkrl-runs/online_ppo-g_attn_agg",
            "pid_file": "ppo.pid",
            "local_log": WP22 / "live-g_attn_agg.log",
            "kind": "ppo",
            "drive_dest": "mydrive:HKRL-Linux/results/online_ppo-g_attn_agg",
        },
        {
            "id": "g_attn_radiant",
            "acct": "colab-d",
            "sess": "hkrl-g4-m_disc",
            "remote_dir": "/content/hkrl-runs/online_ppo-g_attn_radiant",
            "pid_file": "ppo.pid",
            "local_log": WP22 / "live-g_attn_radiant.log",
            "kind": "ppo",
            "drive_dest": "mydrive:HKRL-Linux/results/online_ppo-g_attn_radiant",
        },
    ]

    state: dict[str, dict[str, Any]] = {
        t["id"]: {
            "offset": 0,
            "rounds_seen": 0,
            "best_nohit": -1,
            "dead_count": 0,
            "session_seen": False,
            "done_notified": False,
            "failed_notified": False,
            "low_cu_notified": False,
        }
        for t in targets
    }
    wlog("Hold-alive supervisor started (zero-release mode active).")

    cycle = 0
    while True:
        cycle += 1
        acct_sessions: dict[str, list[str]] = {}
        for t in targets:
            tid = t["id"]
            st = state[tid]
            acct, sess = t["acct"], t["sess"]

            if not st["session_seen"]:
                if acct not in acct_sessions:
                    acct_sessions[acct] = check_sessions(acct)
                lines = acct_sessions[acct]
                if not any(sess in l for l in lines):
                    continue
                st["session_seen"] = True
                wlog(f"[{tid}] Detected live session {acct}/{sess}; keeping alive every 30s.")

            try:
                cp = subprocess.run(
                    [
                        "timeout",
                        "22s",
                        str(BIN / acct),
                        "exec",
                        "-s",
                        sess,
                        "--timeout",
                        "12",
                        "--env",
                        f"HKRL_WATCH_DIR={t['remote_dir']}",
                        "--env",
                        f"HKRL_WATCH_PID={t['pid_file']}",
                        "--env",
                        f"HKRL_LOG_OFFSET={st['offset']}",
                        "-f",
                        str(PROBE_FILE),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=26,
                )
                probe_line = ""
                for line in cp.stdout.splitlines():
                    if line.startswith("WATCH_PROBE_JSON:"):
                        probe_line = line[len("WATCH_PROBE_JSON:") :]
                if not probe_line:
                    # During initial 1.5GB bootstrap on a fresh G4, Jupyter kernel is busy; just note quietly.
                    if cycle % 4 == 0:
                        wlog(f"[{tid}] keepalive/probe rc={cp.returncode} (kernel may be bootstrapping)")
                    continue

                data = json.loads(probe_line)
                st["offset"] = int(data.get("log_offset", st["offset"]))
                chunk = data.get("log_chunk", "")
                if chunk:
                    with t["local_log"].open("a", encoding="utf-8") as lf:
                        lf.write(chunk)

                if t["kind"] == "ppo":
                    hist = data.get("history") or []
                    if len(hist) > st["rounds_seen"]:
                        for entry in hist[st["rounds_seen"] :]:
                            rd = entry.get("round")
                            prom = entry.get("promoted")
                            wins = entry.get("wins")
                            nh = entry.get("nohit")
                            dmg = entry.get("avg_dmg")
                            wlog(f"[{tid}] Round {rd}: promoted={prom} wins={wins} nohit={nh} avg_dmg={dmg}")
                            if int(rd or 0) > 0:
                                notify_windows(
                                    f"HKRL {tid} Round {rd} 完成",
                                    f"promoted={prom} wins={wins}/30 nohit={nh} avg_dmg={dmg}",
                                )
                            if isinstance(nh, int) and nh > st["best_nohit"]:
                                st["best_nohit"] = nh
                        st["rounds_seen"] = len(hist)

                    if not data.get("failed"):
                        st["failed_notified"] = False
                    if not data.get("done"):
                        st["done_notified"] = False
                    if data.get("alive"):
                        st["dead_count"] = 0

                    if data.get("done") and not st["done_notified"]:
                        st["done_notified"] = True
                        notify_windows(
                            f"HKRL {tid} 全部轮次完成 (卡已保活)",
                            f"结果已同步 {t['drive_dest']}，G4 继续保活等待你确认",
                        )
                    elif data.get("failed") and not st["failed_notified"]:
                        st["failed_notified"] = True
                        notify_windows(
                            f"HKRL {tid} 异常退出 (卡已保活)",
                            f"{data.get('failed_text', '')} 请检查 {t['local_log'].name}",
                        )
                    elif not data.get("alive") and data.get("exists") and not data.get("done") and not data.get("failed"):
                        st["dead_count"] += 1
                        if st["dead_count"] == 3 and not st["failed_notified"]:
                            st["failed_notified"] = True
                            notify_windows(f"HKRL {tid} 进程未运行 (卡已保活)", f"请检查 {t['local_log'].name}")

                elif t["kind"] == "formal":
                    if not data.get("failed"):
                        st["failed_notified"] = False
                    if not data.get("done"):
                        st["done_notified"] = False
                    if data.get("alive"):
                        st["dead_count"] = 0

                    if data.get("done") and not st["done_notified"]:
                        st["done_notified"] = True
                        summary = data.get("summary") or {}
                        notify_windows(
                            "HKRL 100场正式验收完成 (卡已保活)",
                            f"wins={summary.get('wins')}/{summary.get('episodes_done')} nohit={summary.get('no_hit_wins')} avg_dmg={summary.get('avg_damage')}",
                        )
                    elif data.get("failed") and not st["failed_notified"]:
                        st["failed_notified"] = True
                        notify_windows("HKRL 100场验收异常 (卡已保活)", f"请检查 {t['local_log'].name}")
                    elif cycle % 4 == 0 and not st["low_cu_notified"]:
                        bal = check_cu_balance(acct)
                        if bal is not None and bal < 1.0:
                            st["low_cu_notified"] = True
                            notify_windows("HKRL colab-a 余额偏低", f"当前余额 {bal:.2f} CU（未自动释放，请指示是否备份释放）")
            except Exception as exc:
                wlog(f"[{tid}] poll exception: {exc}")

        time.sleep(30)


if __name__ == "__main__":
    main()

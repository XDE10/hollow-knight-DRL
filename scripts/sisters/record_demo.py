#!/usr/bin/env python3
"""One-click demo video & GIF recorder for Hollow Knight: Sisters of Battle.

Records two contrasting runs at 1.0x original game speed with Windows desktop
audio (WASAPI loopback — works with both speakers and Bluetooth headphones):
  1. Untrained / random policy getting defeated (`untrained_defeat`)
  2. Frozen native champion (`sisters_champion.bin`) achieving a full no-hit
     victory (`champion_nohit`)

For each run, produces:
  - 1.0x original-speed MP4 with audio
  - Sped-up highlight MP4 with pitch-preserved tempo audio
  - Palette-optimized GIF ready for README / slides

Example:
    python scripts/sisters/record_demo.py
    python scripts/sisters/record_demo.py --mode champion --champion-speed 4
    python scripts/sisters/record_demo.py --no-audio
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import wave

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))
if str(REPO_ROOT / "scripts/sisters") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/sisters"))

from hkrl.env import HKRLEnv
from hkrl.protocol import RewardEventKind as K
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config

from play import (
    DEFAULT_MODEL,
    DEFAULT_TASK_CFG,
    DEFAULT_TRAIN_CFG,
    _sha256,
    discover_game_root,
    ensure_default_model,
    find_game_exe,
    read_runtime_conf,
    wait_for_port,
)

# Verified fast full-no-hit seeds from Formal-100 (`models/sisters_champion/formal100_summary.json`)
DEFAULT_CHAMPION_SEEDS: tuple[int, ...] = (
    19900060,
    19900033,
    19900100,
    19900075,
    19900069,
    19900001,
    19900002,
    19900005,
)
DEFAULT_UNTRAINED_SEED = 19900060
DEFAULT_OUT_DIR = REPO_ROOT / "runs/demo_videos"


def enable_dpi_awareness() -> None:
    """Ensure Win32 window coordinates match physical pixels expected by ffmpeg gdigrab."""
    if os.name != "nt":
        return
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def find_ffmpeg() -> str:
    env_exe = os.environ.get("FFMPEG_PATH", "").strip()
    if env_exe and Path(env_exe).is_file():
        return str(Path(env_exe).resolve())
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise FileNotFoundError(
        "ffmpeg not found in PATH. Install ffmpeg or set FFMPEG_PATH before recording."
    )


def find_game_window(pid: int, timeout_s: float = 30.0) -> int:
    """Locate the visible Hollow Knight window belonging to `pid`."""
    import win32gui
    import win32process

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        found: list[int] = []

        def _cb(hwnd: int, _: Any) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            _, wpid = win32process.GetWindowThreadProcessId(hwnd)
            if wpid != pid:
                return
            l, t, r, b = win32gui.GetClientRect(hwnd)
            if (r - l) >= 320 and (b - t) >= 240:
                found.append(hwnd)

        win32gui.EnumWindows(_cb, None)
        if found:
            return found[0]
        hwnd_title = win32gui.FindWindow(None, "Hollow Knight")
        if hwnd_title and win32gui.IsWindowVisible(hwnd_title):
            l, t, r, b = win32gui.GetClientRect(hwnd_title)
            if (r - l) >= 320 and (b - t) >= 240:
                return hwnd_title
        time.sleep(0.25)
    raise RuntimeError("Could not find visible Hollow Knight window.")


def prepare_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Bring the game window cleanly onto the primary screen and return client (x, y, w, h)."""
    import win32con
    import win32gui

    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    wl, wt, wr, wb = win32gui.GetWindowRect(hwnd)
    win_w, win_h = wr - wl, wb - wt
    screen_w = ctypes.windll.user32.GetSystemMetrics(0)
    screen_h = ctypes.windll.user32.GetSystemMetrics(1)

    # Center or place comfortably inside primary monitor so gdigrab never clips off-screen.
    pos_x = max(0, min(40, screen_w - win_w))
    pos_y = max(0, min(40, screen_h - win_h))
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOP,
        pos_x,
        pos_y,
        win_w,
        win_h,
        win32con.SWP_SHOWWINDOW,
    )
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    time.sleep(0.4)

    cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
    x0, y0 = win32gui.ClientToScreen(hwnd, (cl, ct))
    x1, y1 = win32gui.ClientToScreen(hwnd, (cr, cb))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(screen_w, x1)
    y1 = min(screen_h, y1)
    w = max(2, (x1 - x0) & ~1)
    h = max(2, (y1 - y0) & ~1)
    return x0, y0, w, h


class WasapiLoopbackRecorder:
    """Captures Windows default playback audio (speakers or Bluetooth) into a synced WAV."""

    def __init__(self, wav_path: Path) -> None:
        self.wav_path = wav_path
        self.device_name: str = ""
        self.sample_rate: int = 48000
        self.channels: int = 2
        self._pa: Any = None
        self._stream: Any = None
        self._wf: wave.Wave_write | None = None
        self._lock = threading.Lock()
        self._started_at: float = 0.0
        self._written_frames: int = 0
        self._active = False
        self._bytes_per_frame = 4

    def start(self, t0: float) -> bool:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            print("[AUDIO] pyaudiowpatch not installed; recording video only.", flush=True)
            return False

        try:
            pa = pyaudio.PyAudio()
            lb = pa.get_default_wasapi_loopback()
            rate = int(round(float(lb["defaultSampleRate"])))
            ch = max(1, min(2, int(lb["maxInputChannels"])))
            self.device_name = str(lb["name"])
            self.sample_rate = rate
            self.channels = ch
            self._bytes_per_frame = ch * 2  # paInt16
            self._started_at = t0
            self._written_frames = 0

            wf = wave.open(str(self.wav_path), "wb")
            wf.setnchannels(ch)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            self._wf = wf
            self._pa = pa
            self._active = True

            def _callback(in_data: bytes, frame_count: int, _time_info: Any, _status: Any) -> tuple[None, int]:
                now = time.perf_counter()
                with self._lock:
                    if not self._active or self._wf is None:
                        return (None, pyaudio.paComplete)
                    # Pad exact silence if WASAPI paused during quiet segments to prevent A/V drift.
                    expected_end = int(round((now - self._started_at) * self.sample_rate))
                    gap = expected_end - frame_count - self._written_frames
                    if gap > int(0.06 * self.sample_rate):
                        self._wf.writeframes(b"\x00" * (gap * self._bytes_per_frame))
                        self._written_frames += gap
                    self._wf.writeframes(in_data)
                    self._written_frames += frame_count
                return (None, pyaudio.paContinue)

            self._stream = pa.open(
                format=pyaudio.paInt16,
                channels=ch,
                rate=rate,
                frames_per_buffer=1024,
                input=True,
                input_device_index=int(lb["index"]),
                stream_callback=_callback,
            )
            self._stream.start_stream()
            print(f"[AUDIO] Capturing WASAPI loopback: {self.device_name} ({rate}Hz, {ch}ch)", flush=True)
            return True
        except Exception as exc:
            print(f"[AUDIO] Failed to open WASAPI loopback ({exc}); falling back to video only.", flush=True)
            self.stop()
            return False

    def stop(self) -> bool:
        has_audio = False
        now = time.perf_counter()
        with self._lock:
            was_active = self._active
            self._active = False
            if was_active and self._wf is not None:
                expected = int(round((now - self._started_at) * self.sample_rate))
                pad = expected - self._written_frames
                if pad > 0:
                    self._wf.writeframes(b"\x00" * (pad * self._bytes_per_frame))
                    self._written_frames += pad
                has_audio = self._written_frames > 0
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._wf is not None:
            try:
                self._wf.close()
            except Exception:
                pass
            self._wf = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        return has_audio and self.wav_path.is_file() and self.wav_path.stat().st_size > 1024


class ScreenAndAudioRecorder:
    """Simultaneously records the Hollow Knight client rect (ffmpeg gdigrab) + WASAPI loopback."""

    def __init__(
        self,
        ffmpeg_exe: str,
        rect: tuple[int, int, int, int],
        fps: int,
        raw_video: Path,
        raw_audio: Path,
        capture_audio: bool,
    ) -> None:
        self.ffmpeg_exe = ffmpeg_exe
        self.rect = rect
        self.fps = fps
        self.raw_video = raw_video
        self.raw_audio = raw_audio
        self.capture_audio = capture_audio
        self._proc: subprocess.Popen[bytes] | None = None
        self._audio = WasapiLoopbackRecorder(raw_audio)
        self.audio_ok = False
        self.t0: float = 0.0

    def start(self) -> None:
        x, y, w, h = self.rect
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "gdigrab",
            "-framerate", str(self.fps),
            "-draw_mouse", "0",
            "-offset_x", str(x),
            "-offset_y", str(y),
            "-video_size", f"{w}x{h}",
            "-i", "desktop",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(self.raw_video),
        ]
        self.t0 = time.perf_counter()
        if self.capture_audio:
            self.audio_ok = self._audio.start(self.t0)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Brief warm-up so ffmpeg grabs initial frames before gameplay starts.
        time.sleep(0.4)

    def stop(self) -> tuple[bool, bool]:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write(b"q\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=15.0)
            except Exception:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except Exception:
                    self._proc.kill()
            self._proc = None
        has_audio = self._audio.stop() if self.audio_ok else False
        has_video = self.raw_video.is_file() and self.raw_video.stat().st_size > 1024
        return has_video, has_audio


def build_atempo_chain(speed: float) -> str:
    """Chain ffmpeg atempo filters (each stage must be in [0.5, 2.0])."""
    factors: list[float] = []
    rem = float(speed)
    while rem > 2.0 + 1e-6:
        factors.append(2.0)
        rem /= 2.0
    while rem < 0.5 - 1e-6:
        factors.append(0.5)
        rem /= 0.5
    factors.append(rem)
    return ",".join(f"atempo={f:.4f}" for f in factors)


def run_ffmpeg(ffmpeg_exe: str, args: list[str], desc: str) -> None:
    print(f"[POST] {desc} ...", flush=True)
    cp = subprocess.run([ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True)
    if cp.returncode != 0:
        err = cp.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed during {desc} (rc={cp.returncode}): {err}")


def render_deliverables(
    *,
    ffmpeg_exe: str,
    raw_video: Path,
    raw_audio: Path,
    has_audio: bool,
    trim_start_s: float,
    out_dir: Path,
    stem: str,
    fast_speed: float,
    gif_speed: float,
    gif_width: int,
    gif_fps: int,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_1x = out_dir / f"{stem}_1x.mp4"
    speed_tag = f"{int(fast_speed)}x" if abs(fast_speed - round(fast_speed)) < 1e-3 else f"{fast_speed:g}x"
    mp4_fast = out_dir / f"{stem}_{speed_tag}.mp4"
    gif_path = out_dir / f"{stem}.gif"

    # 1. Final 1.0x MP4 (mux video + synced audio, trim loading screen)
    ss_args = ["-ss", f"{trim_start_s:.3f}"] if trim_start_s > 0.05 else []
    if has_audio:
        run_ffmpeg(
            ffmpeg_exe,
            [
                *ss_args, "-i", str(raw_video),
                *ss_args, "-i", str(raw_audio),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart",
                str(mp4_1x),
            ],
            f"Muxing {mp4_1x.name}",
        )
    else:
        run_ffmpeg(
            ffmpeg_exe,
            [
                *ss_args, "-i", str(raw_video),
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                "-an", "-movflags", "+faststart",
                str(mp4_1x),
            ],
            f"Encoding {mp4_1x.name} (video only)",
        )

    # 2. Sped-up highlight MP4
    pts_expr = f"setpts=PTS/{fast_speed:.4f}"
    if has_audio:
        atempo = build_atempo_chain(fast_speed)
        run_ffmpeg(
            ffmpeg_exe,
            [
                "-i", str(mp4_1x),
                "-filter_complex", f"[0:v]{pts_expr}[v];[0:a]{atempo}[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart",
                str(mp4_fast),
            ],
            f"Generating {mp4_fast.name} ({speed_tag})",
        )
    else:
        run_ffmpeg(
            ffmpeg_exe,
            [
                "-i", str(mp4_1x),
                "-vf", pts_expr,
                "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                "-an", "-movflags", "+faststart",
                str(mp4_fast),
            ],
            f"Generating {mp4_fast.name} ({speed_tag}, video only)",
        )

    # 3. Palette-optimized GIF (2-pass via split)
    gif_vf = (
        f"setpts=PTS/{gif_speed:.4f},fps={gif_fps},"
        f"scale={gif_width}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=max_colors=256:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    run_ffmpeg(
        ffmpeg_exe,
        ["-i", str(mp4_1x), "-vf", gif_vf, "-loop", "0", str(gif_path)],
        f"Generating {gif_path.name} ({gif_speed:g}x, {gif_width}px, {gif_fps}fps)",
    )

    return {
        "mp4_1x": str(mp4_1x),
        "mp4_fast": str(mp4_fast),
        "gif": str(gif_path),
    }


def render_gif_only(
    *,
    ffmpeg_exe: str,
    raw_video: Path,
    trim_start_s: float,
    out_dir: Path,
    stem: str,
    gif_speed: float,
    gif_width: int,
    gif_fps: int,
) -> dict[str, str]:
    """Directly encode a trimmed, palette-optimized GIF without writing MP4s."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"{stem}.gif"
    ss_args = ["-ss", f"{trim_start_s:.3f}"] if trim_start_s > 0.05 else []
    gif_vf = (
        f"setpts=PTS/{gif_speed:.4f},fps={gif_fps},"
        f"scale={gif_width}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=max_colors=256:stats_mode=diff[p];"
        f"[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    run_ffmpeg(
        ffmpeg_exe,
        [*ss_args, "-i", str(raw_video), "-vf", gif_vf, "-loop", "0", str(gif_path)],
        f"Generating {gif_path.name} ({gif_speed:g}x, {gif_width}px, {gif_fps}fps)",
    )
    return {"gif": str(gif_path)}


def sample_untrained_action(rng: np.random.Generator, enable_macro: bool) -> dict[str, Any]:
    """Random untrained policy: moves, jumps, dashes, swings randomly (never heals)."""
    btns = np.zeros(9, dtype=np.int8)
    r = float(rng.random())
    if r < 0.25:
        btns[0] = 1  # jump_tap
        btns[1] = 1  # jump_hold
    elif r < 0.40:
        btns[2] = 1  # dash
    elif r < 0.65:
        btns[3] = 1  # attack
    elif r < 0.72:
        btns[4] = 1  # cast
    act: dict[str, Any] = {
        "movement_x": int(rng.integers(0, 3)),
        "aim_y": int(rng.integers(0, 3)),
        "buttons": btns,
        "duration": 2,  # 4 ticks (~0.08s)
    }
    if enable_macro:
        act["macro"] = 0
    return act


@dataclass
class EpisodeOutcome:
    seed: int
    win: bool
    no_hit: bool
    total_damage: int
    damage_taken: dict[str, int]
    game_sec: float
    wall_sec: float
    raw_video: Path
    raw_audio: Path
    has_audio: bool
    trim_start_s: float


def record_single_session(
    *,
    kind: str,
    game_root: Path,
    model_bin: Path | None,
    seeds: list[int],
    port: int,
    save_slot: int,
    width: int,
    height: int,
    fps: int,
    capture_audio: bool,
    ffmpeg_exe: str,
    tmp_dir: Path,
    require_no_hit: bool,
    target_count: int = 1,
    tail_seconds: float | None = None,
) -> list[EpisodeOutcome]:
    exe = find_game_exe(game_root)
    mod_dll = game_root / "hollow_knight_Data/Managed/Mods/HKRLEnvMod/HKRLEnvMod.dll"
    if not mod_dll.is_file():
        raise FileNotFoundError(f"HKRLEnvMod.dll not found at {mod_dll}")

    runtime_conf = read_runtime_conf(game_root)
    env_vars = dict(os.environ)
    for k in list(env_vars):
        if k.startswith("HKRL_NATIVE_") or k in ("HKRL_TACTICS_POLICY", "HKRL_TCP_ACTOR_PORT", "SDL_AUDIODRIVER"):
            env_vars.pop(k, None)

    ledger_dir = tempfile.TemporaryDirectory(prefix=f"hkrl_rec_{kind}_")
    env_vars.update(
        {
            "HKRL_HOST": "127.0.0.1",
            "HKRL_PORT": str(port),
            "HKRL_SAVE_SLOT": str(save_slot),
            "HKRL_SISTERS_NAIL_MULTIPLIER": "1",
            "SteamAppId": "367520",
            "SteamGameId": "367520",
        }
    )
    if model_bin is not None:
        env_vars["HKRL_NATIVE_ACTOR"] = str(model_bin.resolve())
        env_vars["HKRL_NATIVE_ACTOR_LEDGER"] = str(Path(ledger_dir.name) / "actor-ledger.txt")
    if "HKRL_AUTH_TOKEN" in runtime_conf and "HKRL_AUTH_TOKEN" not in env_vars:
        env_vars["HKRL_AUTH_TOKEN"] = runtime_conf["HKRL_AUTH_TOKEN"]

    args = [
        str(exe),
        "-screen-fullscreen", "0",
        "-screen-width", str(width),
        "-screen-height", str(height),
    ]

    cfg = load_train_config(str(DEFAULT_TRAIN_CFG))
    task = load_task_config(str(DEFAULT_TASK_CFG))
    if model_bin is not None:
        task.action.action_repeat = 16
        task.time_limit_seconds = 360
    else:
        task.action.action_repeat = 4
        task.time_limit_seconds = 120

    noop: dict[str, Any] = {
        "movement_x": 1,
        "aim_y": 1,
        "buttons": np.zeros(9, dtype=np.int8),
        "duration": 3,
    }
    if task.action.enable_macro_actions:
        noop["macro"] = 0

    label = f"{model_bin.name} (sha256={_sha256(model_bin)[:12]})" if model_bin else "UNTRAINED random policy"
    print(f"\n=== [HKRL RECORD: {kind}] Launching Hollow Knight ({label}) ===", flush=True)
    proc = subprocess.Popen(args, cwd=str(game_root), env=env_vars)

    game: HKRLEnv | None = None
    outcomes: list[EpisodeOutcome] = []
    try:
        if not wait_for_port(port, proc, timeout_s=90.0):
            raise RuntimeError(f"Hollow Knight did not listen on 127.0.0.1:{port} (exit={proc.poll()})")

        hwnd = find_game_window(proc.pid, timeout_s=30.0)
        rect = prepare_window_rect(hwnd)
        print(f"[WINDOW] Locked client area: x={rect[0]}, y={rect[1]}, {rect[2]}x{rect[3]}", flush=True)

        if "HKRL_AUTH_TOKEN" in env_vars:
            os.environ["HKRL_AUTH_TOKEN"] = env_vars["HKRL_AUTH_TOKEN"]
        game = HKRLEnv(make_transport(cfg, port=port), task, step_timeout_s=20.0)
        dmg_names = {1: "BOSS_BODY", 2: "ENEMY_PROJECTILE", 3: "SPIKES", 4: "OTHER"}

        for attempt_idx, seed in enumerate(seeds, start=1):
            raw_v = tmp_dir / f"{kind}_att{attempt_idx}_seed{seed}.mkv"
            raw_a = tmp_dir / f"{kind}_att{attempt_idx}_seed{seed}.wav"
            recorder = ScreenAndAudioRecorder(ffmpeg_exe, rect, fps, raw_v, raw_a, capture_audio)

            # Ensure game process & TCP connection are healthy before starting ffmpeg.
            if proc.poll() is not None:
                print(f"[{kind}] Hollow Knight exited (rc={proc.returncode}); relaunching...", flush=True)
                if game is not None:
                    try:
                        game.close()
                    except Exception:
                        pass
                    game = None
                proc = subprocess.Popen(args, cwd=str(game_root), env=env_vars)
                if not wait_for_port(port, proc, timeout_s=90.0):
                    raise RuntimeError(f"Hollow Knight did not listen on 127.0.0.1:{port} (exit={proc.poll()})")
                hwnd = find_game_window(proc.pid, timeout_s=30.0)
                rect = prepare_window_rect(hwnd)
                game = HKRLEnv(make_transport(cfg, port=port), task, step_timeout_s=20.0)

            # Re-raise window right before each attempt so it stays unobscured.
            rect = prepare_window_rect(hwnd)
            recorder.rect = rect
            print(f"[{kind} attempt {attempt_idx}/{len(seeds)}] seed={seed} — resetting & recording...", flush=True)
            recorder.start()
            recorder_stopped = False
            try:
                t_wall0 = time.monotonic()
                try:
                    obs, _ = game.reset(seed=seed, options={"reset_timeout_s": 90.0})
                except (ConnectionError, OSError, TimeoutError) as exc:
                    print(f"  -> TCP reset interrupted ({exc}); reconnecting and retrying seed={seed}...", flush=True)
                    recorder.stop()
                    recorder_stopped = True
                    raw_v.unlink(missing_ok=True)
                    raw_a.unlink(missing_ok=True)
                    if game is not None:
                        try:
                            game.close()
                        except Exception:
                            pass
                        game = None
                    if proc.poll() is not None:
                        proc = subprocess.Popen(args, cwd=str(game_root), env=env_vars)
                    if not wait_for_port(port, proc, timeout_s=90.0):
                        raise RuntimeError(f"Hollow Knight did not listen on 127.0.0.1:{port} (exit={proc.poll()})")
                    hwnd = find_game_window(proc.pid, timeout_s=30.0)
                    rect = prepare_window_rect(hwnd)
                    recorder = ScreenAndAudioRecorder(ffmpeg_exe, rect, fps, raw_v, raw_a, capture_audio)
                    game = HKRLEnv(make_transport(cfg, port=port), task, step_timeout_s=20.0)
                    recorder.start()
                    recorder_stopped = False
                    t_wall0 = time.monotonic()
                    obs, _ = game.reset(seed=seed, options={"reset_timeout_s": 90.0})

                fight_start_offset = max(0.0, time.perf_counter() - recorder.t0 - 1.2)
                hp0 = int(round(float(obs["player"][4])))
                soul0 = int(round(float(obs["player"][6])))
                dmg_counts: dict[str, int] = {}
                win = False
                game_sec = 0.0
                rng = np.random.default_rng(seed)

                while True:
                    action = noop if model_bin is not None else sample_untrained_action(rng, task.action.enable_macro_actions)
                    obs, _, term, trunc, info = game.step(action)
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

                end_offset = max(fight_start_offset, time.perf_counter() - recorder.t0)
                # Keep ~1.8s post-fight tail so defeat/victory animation completes naturally.
                time.sleep(1.8)
                has_video, has_audio = recorder.stop()
                recorder_stopped = True
            finally:
                if not recorder_stopped:
                    recorder.stop()
            if not has_video:
                raise RuntimeError(f"ffmpeg produced empty video file: {raw_v}")

            trim_start = fight_start_offset
            if tail_seconds is not None and tail_seconds > 0:
                trim_start = max(fight_start_offset, end_offset - float(tail_seconds))

            total_dmg = sum(dmg_counts.values())
            no_hit = bool(win and total_dmg == 0 and hp0 == 9 and soul0 == 0)
            outcome = EpisodeOutcome(
                seed=seed,
                win=win,
                no_hit=no_hit,
                total_damage=total_dmg,
                damage_taken=dmg_counts,
                game_sec=round(game_sec, 2),
                wall_sec=round(time.monotonic() - t_wall0, 1),
                raw_video=raw_v,
                raw_audio=raw_a,
                has_audio=has_audio,
                trim_start_s=trim_start,
            )
            print(
                f"  -> Result: win={win}, no_hit={no_hit}, dmg={total_dmg} ({dmg_counts}), "
                f"game_t={outcome.game_sec}s, audio={has_audio}",
                flush=True,
            )

            accepted = (
                (not win)
                if model_bin is None
                else (no_hit if require_no_hit else win)
            )
            if accepted:
                outcomes.append(outcome)
                if len(outcomes) >= max(1, target_count):
                    break
            else:
                # Discard rejected attempt and try next seed.
                print("  -> Did not meet target condition; trying next seed...", flush=True)
                raw_v.unlink(missing_ok=True)
                raw_a.unlink(missing_ok=True)
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

    if not outcomes:
        raise RuntimeError(
            f"[{kind}] Failed to capture a matching episode across seeds {seeds}. "
            "Try increasing --max-attempts or passing --allow-any-win."
        )
    return outcomes


def main() -> None:
    enable_dpi_awareness()

    ap = argparse.ArgumentParser(description="Record Sisters of Battle untrained-defeat vs champion-nohit videos & GIFs.")
    ap.add_argument("--mode", choices=("all", "untrained", "champion"), default="all")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to frozen native actor .bin")
    ap.add_argument("--game-root", type=Path, default=None, help="Path to Hollow Knight install dir")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for MP4s & GIFs")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--save-slot", type=int, default=3, choices=(1, 2, 3, 4))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30, help="Capture framerate (30 recommended on laptops)")
    ap.add_argument("--no-audio", action="store_true", help="Disable Windows WASAPI loopback audio recording")
    ap.add_argument("--untrained-seed", type=int, default=DEFAULT_UNTRAINED_SEED)
    ap.add_argument("--champion-seed", type=int, default=None, help="Force a specific first champion seed")
    ap.add_argument("--max-attempts", type=int, default=6, help="Max champion retries to capture a 0-hit win")
    ap.add_argument("--allow-any-win", action="store_true", help="Accept any champion victory even if hit")
    ap.add_argument("--untrained-speed", type=float, default=2.0, help="Fast MP4 speed for untrained clip")
    ap.add_argument("--champion-speed", type=float, default=4.0, help="Fast MP4 speed for champion clip")
    ap.add_argument("--untrained-gif-speed", type=float, default=1.5, help="GIF speed for untrained clip")
    ap.add_argument("--champion-gif-speed", type=float, default=5.0, help="GIF speed for champion clip")
    ap.add_argument("--untrained-clips", type=int, default=3, help="Number of short untrained-defeat GIF clips to record")
    ap.add_argument("--untrained-tail-sec", type=float, default=10.0, help="Seconds before death to keep in untrained GIFs")
    ap.add_argument("--untrained-full-video", action="store_true", help="Also export full 1x/fast MP4s for untrained runs (default: GIF only)")
    ap.add_argument("--gif-width", type=int, default=720, help="GIF width in pixels")
    ap.add_argument("--gif-fps", type=int, default=12, help="GIF framerate")
    ap.add_argument("--keep-raw", action="store_true", help="Keep temporary raw .mkv/.wav files")
    args = ap.parse_args()

    ffmpeg_exe = find_ffmpeg()
    game_root = discover_game_root(args.game_root)
    model_bin = ensure_default_model(args.model) if args.mode in ("all", "champion") else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_ctx = tempfile.TemporaryDirectory(prefix="hkrl_raw_", dir=str(args.output_dir), ignore_cleanup_errors=True)
    tmp_dir = Path(tmp_ctx.name)
    manifest_path = args.output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audio_enabled": not args.no_audio,
        "clips": {},
    }
    if manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(prev.get("clips"), dict):
                manifest["clips"].update(prev["clips"])
        except Exception:
            pass

    try:
        if args.mode in ("all", "untrained"):
            n_clips = max(1, int(args.untrained_clips))
            untrained_seeds = [args.untrained_seed + i for i in range(n_clips + 2)]
            gif_only = not args.untrained_full_video
            res_list_u = record_single_session(
                kind="untrained_defeat",
                game_root=game_root,
                model_bin=None,
                seeds=untrained_seeds,
                port=args.port,
                save_slot=args.save_slot,
                width=args.width,
                height=args.height,
                fps=args.fps,
                capture_audio=(not args.no_audio) and (not gif_only),
                ffmpeg_exe=ffmpeg_exe,
                tmp_dir=tmp_dir,
                require_no_hit=False,
                target_count=n_clips,
                tail_seconds=args.untrained_tail_sec if gif_only else None,
            )
            for idx, res_u in enumerate(res_list_u, start=1):
                stem = "untrained_defeat" if n_clips == 1 else f"untrained_defeat_{idx}"
                if gif_only:
                    files_u = render_gif_only(
                        ffmpeg_exe=ffmpeg_exe,
                        raw_video=res_u.raw_video,
                        trim_start_s=res_u.trim_start_s,
                        out_dir=args.output_dir,
                        stem=stem,
                        gif_speed=args.untrained_gif_speed,
                        gif_width=args.gif_width,
                        gif_fps=args.gif_fps,
                    )
                else:
                    files_u = render_deliverables(
                        ffmpeg_exe=ffmpeg_exe,
                        raw_video=res_u.raw_video,
                        raw_audio=res_u.raw_audio,
                        has_audio=res_u.has_audio,
                        trim_start_s=res_u.trim_start_s,
                        out_dir=args.output_dir,
                        stem=stem,
                        fast_speed=args.untrained_speed,
                        gif_speed=args.untrained_gif_speed,
                        gif_width=args.gif_width,
                        gif_fps=args.gif_fps,
                    )
                manifest["clips"][stem] = {
                    "seed": res_u.seed,
                    "win": res_u.win,
                    "total_damage": res_u.total_damage,
                    "damage_taken": res_u.damage_taken,
                    "game_sec": res_u.game_sec,
                    "has_audio": res_u.has_audio,
                    "files": files_u,
                }

        if args.mode in ("all", "champion"):
            assert model_bin is not None
            cand_seeds = list(DEFAULT_CHAMPION_SEEDS)
            if args.champion_seed is not None:
                cand_seeds = [args.champion_seed] + [s for s in cand_seeds if s != args.champion_seed]
            cand_seeds = cand_seeds[: max(1, args.max_attempts)]
            res_c = record_single_session(
                kind="champion_nohit",
                game_root=game_root,
                model_bin=model_bin,
                seeds=cand_seeds,
                port=args.port,
                save_slot=args.save_slot,
                width=args.width,
                height=args.height,
                fps=args.fps,
                capture_audio=not args.no_audio,
                ffmpeg_exe=ffmpeg_exe,
                tmp_dir=tmp_dir,
                require_no_hit=not args.allow_any_win,
                target_count=1,
            )[0]
            files_c = render_deliverables(
                ffmpeg_exe=ffmpeg_exe,
                raw_video=res_c.raw_video,
                raw_audio=res_c.raw_audio,
                has_audio=res_c.has_audio,
                trim_start_s=res_c.trim_start_s,
                out_dir=args.output_dir,
                stem="champion_nohit",
                fast_speed=args.champion_speed,
                gif_speed=args.champion_gif_speed,
                gif_width=args.gif_width,
                gif_fps=args.gif_fps,
            )
            manifest["clips"]["champion_nohit"] = {
                "seed": res_c.seed,
                "win": res_c.win,
                "no_hit": res_c.no_hit,
                "total_damage": res_c.total_damage,
                "damage_taken": res_c.damage_taken,
                "game_sec": res_c.game_sec,
                "has_audio": res_c.has_audio,
                "files": files_c,
            }
    finally:
        if not args.keep_raw:
            tmp_ctx.cleanup()
        else:
            print(f"[RAW KEPT] {tmp_dir}", flush=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n================ RECORDING COMPLETE ================", flush=True)
    for k, info in manifest["clips"].items():
        print(f"[{k}] seed={info['seed']} win={info['win']} dmg={info['total_damage']} audio={info['has_audio']}", flush=True)
        for fk, fp in info["files"].items():
            p = Path(fp)
            mb = p.stat().st_size / (1024 * 1024) if p.is_file() else 0.0
            print(f"  - {fk:8s}: {p} ({mb:.2f} MB)", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

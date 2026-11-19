"""Capture native player controls on save slot 3 and convert physics ticks to BC data."""

from __future__ import annotations

import argparse
import os
import struct
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from hkrl import protocol, spaces
from hkrl.env import HKRLEnv
from hkrl.training.local_session import exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.live_tuning import atomic_write_json
from hkrl.wrappers import NormalizeObservation


def frames(path: Path) -> Iterator[tuple[int, protocol.DecodedStepResponse]]:
    with path.open("rb") as stream:
        if stream.read(4) != b"HKD1":
            raise ValueError("unknown human recording format")
        while header := stream.read(6):
            if len(header) != 6:
                raise ValueError("incomplete input or frame header")
            keys, size = struct.unpack("<HI", header)
            if keys >= 1024 or not 0 < size <= 8 * 1024 * 1024:
                raise ValueError("invalid recorded input or frame size")
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError("incomplete observation frame")
            yield keys, protocol.decode_step_response(payload)


def primitive(keys: int, previous: int, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Project native button edges and holds onto legal duration-one policy actions."""
    axes = [
        1 + bool(keys & 2) - bool(keys & 1),
        1 + bool(keys & 8) - bool(keys & 4),
    ]
    pressed = keys & ~previous
    buttons = np.zeros(spaces.N_BUTTONS, dtype=np.int8)
    for name, active in (
        ("jump_tap", pressed & 16),
        ("jump_hold", keys & 16),
        ("dash", pressed & 32),
        ("attack", pressed & 64),
        ("nail_art_hold", keys & previous & 64),
        ("cast", pressed & 128),
        ("focus_hold", keys & 256),
        ("dream_nail", keys & 512),
    ):
        buttons[spaces.BUTTON_BITS[name]] = bool(active)
    allowed = mask[6 : 6 + spaces.N_BUTTONS]
    dropped = int(np.count_nonzero(buttons & ~allowed))
    return np.asarray([*axes, *(buttons * allowed), 0, 0], dtype=np.int64), dropped


def convert(path: Path, root: Path, env: HKRLEnv) -> Path:
    normal = NormalizeObservation(env)
    state_keys = ("global", "player", "entities", "entity_mask")
    saved: dict[str, list] = {
        key: []
        for key in (
            *state_keys,
            "actions",
            "masks",
            "rewards",
            "elapsed_ticks",
            "server_ticks",
            "raw_keys",
        )
    }
    previous = None
    previous_keys = 0
    first_hp = max_hp = peak_bosses = masked = 0
    damage: Counter[str] = Counter()
    observed_scales: Counter[float] = Counter()
    terminal = won = truncated = False
    episode_id = None
    first_bosses = 0
    first_multi_tick = None
    for keys, frame in frames(path):
        if (
            terminal
            or frame.observation is None
            or frame.error_code != protocol.StatusCode.OK
        ):
            raise ValueError("unexpected frame after terminal or missing valid state")
        obs = env._to_gym_observation(frame.observation)
        env._validate_action_mask(frame.action_mask)
        if frame.observation.global_state[2] != env.task.wire_id:
            raise ValueError("wrong task for Sisters demonstration")
        current_id = int(frame.observation.global_state[8])
        observed_scales[round(float(frame.observation.global_state[5]), 3)] += 1
        bosses = int(
            np.count_nonzero(
                (obs["entities"][:, 1] == 1) & obs["entity_mask"].astype(bool)
            )
        )
        if previous is None:
            episode_id = current_id
            first_hp, max_hp = map(int, obs["player"][4:6])
            first_bosses = bosses
        else:
            prev_frame, prev_obs, native = previous
            if (
                frame.server_tick != prev_frame.server_tick + 1
                or current_id != episode_id
            ):
                raise ValueError("discontinuous tick or mixed episodes")
            label, omitted = primitive(native, previous_keys, prev_frame.action_mask)
            masked += omitted
            for key in state_keys:
                saved[key].append(prev_obs[key])
            saved["raw_keys"].append(native)
            saved["actions"].append(label)
            saved["masks"].append(prev_frame.action_mask)
            saved["server_ticks"].append(prev_frame.server_tick)
            saved["elapsed_ticks"].append(1)
            saved["rewards"].append(env.reward_fn(frame.reward_events, dt=1))
            previous_keys = native
        if bosses >= 2 and first_multi_tick is None:
            first_multi_tick = frame.server_tick
        peak_bosses = max(peak_bosses, bosses)
        for event in frame.reward_events:
            if event.kind in (
                protocol.RewardEventKind.DAMAGE_TAKEN,
                protocol.RewardEventKind.DAMAGE_DEALT,
            ):
                damage[event.kind.name] += event.amount
        terminal = frame.terminated or frame.truncated
        truncated = frame.truncated
        if terminal:
            won = bool(
                frame.terminated
                and not truncated
                and obs["player"][4] > 0
                and any(
                    e.kind == protocol.RewardEventKind.BOSS_KILLED and e.entity_id == 0
                    for e in frame.reward_events
                )
                and not any(
                    e.kind == protocol.RewardEventKind.PLAYER_DEATH
                    for e in frame.reward_events
                )
            )
        previous = (frame, normal.observation(obs), keys)
    if not terminal or not saved["actions"]:
        raise ValueError("incomplete demonstration")
    qualified = (
        won
        and first_hp == max_hp
        and first_bosses == 1
        and peak_bosses >= 2
        and any(saved["raw_keys"])
    )
    dest = root / ("wins" if qualified else "attempts") / (path.stem + ".npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    record = {
        "teacher": "human",
        "steps": len(saved["actions"]),
        "episode_id": episode_id,
        "won": won,
        "truncated": truncated,
        "qualified": qualified,
        "damage": damage,
        "initial_hp": first_hp,
        "max_hp": max_hp,
        "max_concurrent": {"bosses": peak_bosses},
        "first_bosses": first_bosses,
        "first_multi_tick": first_multi_tick,
        "masked_button_labels": masked,
        "hits": [],
        "time_scale": observed_scales.most_common(1)[0][0],
    }
    with NamedTemporaryFile(dir=dest.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{k: np.asarray(v) for k, v in saved.items()})
        atomic_write_json(dest.with_suffix(".json"), record)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--mod-dir", type=Path)
    parser.add_argument(
        "--convert", type=Path, help="convert a completed raw capture offline"
    )
    args = parser.parse_args()
    task = load_task_config("configs/experiments/sisters_responsive.yaml")
    cfg = load_train_config("configs/train/sisters_platform.yaml")
    env = HKRLEnv(make_transport(cfg), task)
    root = args.output.resolve()
    if args.convert:
        print(convert(args.convert, root, env))
        return
    if (
        os.environ.get("HKRL_SAVE_SLOT") != "3"
        or args.mod_dir is None
        or args.episodes < 1
    ):
        parser.error(
            "recording requires the slot-3 launcher, mod directory and positive episode count"
        )
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = args.mod_dir / "hkrl-human-demo.path"
    with exclusive_game_lock(Path("runs/training/game.lock")):
        for _ in range(args.episodes):
            path = raw_dir / f"episode-{time.time_ns()}.hkdem"
            mine = False
            try:
                env.reset(options={"reset_timeout_s": 90})
                with marker.open("x", encoding="utf-8") as f:
                    mine = True
                    f.write(str(path))
                env.set_timescale(0.65)
                if marker.exists():
                    raise RuntimeError(
                        "game did not acknowledge recorder; check the installed mod"
                    )
                env.close()  # Releases synthetic input. InControl now belongs to the player.
                print(
                    "Native input active. Play the fight in the game window.",
                    flush=True,
                )
                deadline = time.monotonic() + 420
                while not path.is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("capture did not reach a terminal frame")
                    time.sleep(0.3)
                print(convert(path, root, env), flush=True)
            finally:
                if (
                    mine
                    and marker.exists()
                    and marker.read_text(encoding="utf-8") == str(path)
                ):
                    marker.unlink()
                try:
                    env.set_timescale(1.0)
                finally:
                    env.close()


if __name__ == "__main__":
    main()

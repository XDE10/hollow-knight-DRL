"""Record short/long jump, re-press in midair, and downward slash with the 48 actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hkrl import spaces
from hkrl.env import HKRLEnv
from hkrl.models.sisters_ddqn import ACTIONS, game_action, legal_actions
from hkrl.training.local_session import exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config

P = spaces.PLAYER_FEATURE_INDEX
FLAGS = spaces.PLAYER_ACTION_FLAG_BITS


def trial(env: HKRLEnv, move: str, speed: float) -> dict[str, Any]:
    obs, info = env.reset(options={"reset_timeout_s": 90})
    env.set_timescale(speed)
    trace: list[dict[str, Any]] = []
    started = None
    start_y = 0.0
    released = second_press = False
    for step in range(250):
        player = obs["player"]
        tick = info["server_tick"]
        if started is None and step and player[P["on_ground"]]:
            started, start_y = tick, float(player[P["pos_y"]])
        age = tick - started if started is not None else -1
        jump = age >= 0 and (age == 0 if move == "short" else age < 12)
        if move == "double" and age >= 12:
            if released and age >= 16 and player[P["double_jump_available"]] and not second_press:
                jump = second_press = True
            elif second_press and age < 25:
                jump = True
            else:
                released = True
        attack_allowed = info["action_mask"][6 + spaces.BUTTON_BITS["attack"]]
        down = move == "down" and age >= 4 and not player[P["on_ground"]] and attack_allowed
        nail = 2 if down else 0
        index = ACTIONS.index((1, int(jump), nail, 0))
        if not legal_actions(info["action_mask"])[index]:
            index = ACTIONS.index((1, 0, nail, 0))
        obs, _, terminated, truncated, info = env.step(game_action(index))
        player = obs["player"]
        trace.append(
            dict(
                tick=int(info["server_tick"]),
                index=index,
                x=float(player[P["pos_x"]]),
                y=float(player[P["pos_y"]]),
                vy=float(player[P["vel_y"]]),
                ground=bool(player[P["on_ground"]]),
                flags=int(player[P["action_flags"]]),
                hp=float(player[P["hp"]]),
            )
        )
        if terminated or truncated or (age >= 8 and player[P["on_ground"]]) or age > 85:
            break
    return dict(
        name=move,
        scale=speed,
        start_tick=started,
        peak=max(t["y"] for t in trace) - start_y if started is not None else None,
        clean=started is not None and len({t["hp"] for t in trace}) == 1,
        second=any(t["flags"] & (1 << FLAGS["double_jumping"]) for t in trace),
        down=any(t["flags"] & (1 << FLAGS["down_attacking"]) and not t["ground"] for t in trace),
        trace=trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    task = load_task_config("configs/experiments/sisters_ddqn.yaml")
    cfg = load_train_config("configs/train/sisters_platform.yaml")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with exclusive_game_lock(Path("runs/training/game.lock")):
        env = HKRLEnv(make_transport(cfg, port=args.port), task)
        try:
            for repeat, speed in ((1, 0.5), (4, 1.0)):
                task.action.action_repeat = repeat
                for move in ("short", "long", "double", "down") * 2:
                    row = dict(repeat=repeat, **trial(env, move, speed))
                    records.append(row)
                    args.output.write_text(json.dumps(records), encoding="utf-8")
                    print({k: v for k, v in row.items() if k != "trace"}, flush=True)
                    env.set_timescale(1.0)
        finally:
            try:
                env.set_timescale(1.0)
            finally:
                env.close()


if __name__ == "__main__":
    main()

"""Compare frozen full-fight and life-loss policies in alternating live Sisters fights."""

from __future__ import annotations

import argparse
import json
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fit_sisters_dynamics import ACTION_DIM, HISTORY, encode_actions
from pack_sisters_sequences import packed_state
from probe_sisters_replay import frame_geometry
from train_sisters_imagination import student

from hkrl import protocol, spaces
from hkrl.env import HKRLEnv
from hkrl.eval.evaluator import Evaluator, _aggregate
from hkrl.training.local_session import ResetTimeout, exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.wrappers import NormalizeObservation


class ImaginedPolicy:
    def __init__(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.raw = payload["catalogue"].numpy()
        self.codes = encode_actions(payload["catalogue"]).numpy()
        self.vocab = payload["vocab"].numpy()
        self.mean = payload["mean"].numpy()
        self.scale = payload["scale"].numpy()
        size = len(self.mean) + len(self.vocab) + 1
        self.actor = student(HISTORY * size + (HISTORY - 1) * ACTION_DIM, len(self.raw))
        self.actor.load_state_dict(payload["weights"])
        self.actor.eval()
        self.noop = np.asarray(spaces.canonical_noop_action_values(enable_macro=True))
        self.noop_code = encode_actions(torch.as_tensor(self.noop)).numpy()
        self.states: deque[np.ndarray] = deque(maxlen=HISTORY)
        self.actions: deque[np.ndarray] = deque(maxlen=HISTORY - 1)
        self.last_time = 0.0
        self.boss_id: int | None = None
        self.episode: int | None = None

    def reset(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.last_time = 0.0
        self.boss_id = None
        self.episode = None

    @torch.inference_mode()
    def act(self, obs: dict[str, np.ndarray], action_mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(obs["entity_mask"], dtype=bool).copy()
        entities = obs["entities"]
        e = spaces.ENTITY_FEATURE_INDEX
        bodies = np.flatnonzero(mask & (entities[:, e["entity_type"]] == protocol.EntityType.BOSS))
        raw = self.noop
        if len(bodies):
            # Only the single-body opening was present in training. Use the nearest body after it.
            xy = entities[bodies, e["hurtbox_center_x"] : e["hurtbox_center_y"] + 1]
            nearest = bodies[np.square(xy - obs["player"][:2]).sum(1).argmin()]
            mask[bodies] = False
            mask[nearest] = True
            geom, boss, _ = frame_geometry(
                obs["player"][None, None], entities[None, None], mask[None, None]
            )
            clock = float(obs["global"][spaces.GLOBAL_FEATURE_INDEX["time_in_episode"]])
            episode = int(obs["global"][spaces.GLOBAL_FEATURE_INDEX["episode_id"]])
            boss_id = int(entities[nearest, e["entity_id"]])
            if self.boss_id != boss_id or self.episode != episode or clock <= self.last_time:
                self.reset()
            elapsed = np.array([[clock - self.last_time if self.states else 0]], dtype=np.float32)
            state = packed_state(geom, boss, elapsed)[0, 0]
            fsm = int(entities[nearest, e["fsm_state_hash"]])
            index = int(np.searchsorted(self.vocab, fsm))
            one_hot = np.zeros(len(self.vocab) + 1, np.float32)
            one_hot[index if index < len(self.vocab) and self.vocab[index] == fsm else -1] = 1
            feature = np.r_[np.clip((state - self.mean) / self.scale, -12, 12), one_hot]
            if not self.states:
                self.states.extend([feature] * (HISTORY - 1))
                self.actions.extend([self.noop_code] * (HISTORY - 1))
            self.states.append(feature)
            history = np.concatenate([*self.states, *self.actions])
            pred = self.actor(torch.from_numpy(history)).numpy()
            legal = self.codes @ (~np.asarray(action_mask, bool)).astype(np.float32) == 0
            if legal.any():
                selected = np.where(legal, pred, -np.inf).argmax()
                raw = self.raw[selected]
                self.actions.append(self.codes[selected])
            else:
                self.actions.append(self.noop_code)
            self.boss_id, self.episode, self.last_time = boss_id, episode, clock
        else:
            self.reset()
        return dict(
            movement_x=int(raw[0]),
            aim_y=int(raw[1]),
            buttons=raw[2:11].copy(),
            duration=int(raw[11]),
            macro=int(raw[12]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="configs/experiments/sisters_counter.yaml")
    parser.add_argument("--config", default="configs/train/sisters_platform.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=33410)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("positive episode count required")
    torch.set_num_threads(1)
    task = load_task_config(args.task)
    cfg = load_train_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    labels = ("full_fight", "life_loss")
    policies = {name: ImaginedPolicy(args.models / f"{name}.pt") for name in labels}
    results: dict[str, list[Any]] = {name: [] for name in labels}
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    with exclusive_game_lock(Path("runs/training/game.lock")), ExitStack() as stack:
        streams = {
            name: stack.enter_context(
                (args.output / f"{name}-steps.jsonl").open("x", encoding="utf-8")
            )
            for name in labels
        }
        env = NormalizeObservation(ResetTimeout(HKRLEnv(make_transport(cfg), task), 90, 1))
        stack.callback(env.close)
        for i, seed in enumerate(seeds):
            for name in labels if i % 2 == 0 else labels[::-1]:
                policy = policies[name]
                policy.reset()
                stream = streams[name]
                evaluator = Evaluator(
                    policy,
                    tasks=[task],
                    seeds=[seed],
                    replay_sink=lambda record, sink=stream: sink.write(json.dumps(record) + "\n"),
                )
                result = evaluator._run_episode(env, seed, task=task, episode_index=i)
                results[name].append(result)
                stream.flush()
                print(name, i + 1, result.damage_dealt, result.won, flush=True)
                summary = dict(
                    metadata=dict(
                        episodes=i + 1,
                        task_ids=[task.task_id],
                        seeds=seeds[: i + 1],
                        checkpoint=str(args.models / f"{name}.pt"),
                        model=name,
                        game_seed_controlled=False,
                    ),
                    metrics={task.task_id: _aggregate(results[name])},
                )
                (args.output / f"{name}.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )


if __name__ == "__main__":
    main()

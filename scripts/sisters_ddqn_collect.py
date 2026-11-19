"""Local masked action selection, transition collection, and frozen Sisters evaluation."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from hkrl import protocol, spaces
from hkrl.env import HKRLEnv
from hkrl.eval.evaluator import Evaluator
from hkrl.models.sisters_ddqn import game_action, legal_actions
from hkrl.training.local_session import ResetTimeout, exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.registry import get
from hkrl.wrappers import NormalizeObservation

FIELDS = ("global", "player", "entities", "entity_mask")


class Policy:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @torch.inference_mode()
    def choose(self, obs: dict[str, np.ndarray], valid: np.ndarray) -> int:
        if not valid.any():
            raise ValueError("no legal 48-action choice")
        active = np.flatnonzero(obs["entity_mask"])
        end = int(active[-1]) + 1 if active.size else 1
        state = {
            k: torch.from_numpy(v[:end] if k in ("entities", "entity_mask") else v)[None]
            for k, v in obs.items()
            if k in FIELDS
        }
        q = self.model(state)[0].numpy()
        return int(np.where(valid, q, -np.inf).argmax())

    def act(self, obs: dict[str, np.ndarray], action_mask: np.ndarray) -> dict:
        return game_action(self.choose(obs, legal_actions(action_mask)))


def environment(port: int) -> NormalizeObservation:
    task = load_task_config("configs/experiments/sisters_ddqn.yaml")
    cfg = load_train_config("configs/train/sisters_platform.yaml")
    return NormalizeObservation(
        ResetTimeout(HKRLEnv(make_transport(cfg, port=port), task, step_timeout_s=20), 100, 2)
    )


class EvaluationEnv(gym.Wrapper):
    """Set original speed on every reset and record the observed speed per fight."""

    def __init__(self, port: int) -> None:
        super().__init__(environment(port))
        self.scales: list[list[float]] = []

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self.unwrapped.set_timescale(1.0)
        self.scales.append([])
        return obs, info

    def step(self, action: dict) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminal, truncated, info = self.env.step(action)
        scale = float(obs["global"][spaces.GLOBAL_FEATURE_INDEX["time_scale"]])
        self.scales[-1].append(scale)
        return obs, float(reward), terminal, truncated, info


def collect_worker(
    port: int,
    steps: int,
    policy: Policy,
    seed: int,
    epsilon: float,
    scale: float,
    benchmark: bool,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed + port)
    env = environment(port)
    arrays: dict[str, list] = defaultdict(list)
    episodes = resets = multi = 0
    start = time.monotonic()
    done = True
    try:
        for i in range(steps):
            if done:
                obs, info = env.reset(seed=seed + i)
                env.unwrapped.set_timescale(scale)
                resets += 1
            allowed = legal_actions(info["action_mask"])
            action = (
                policy.choose(obs, allowed)
                if benchmark or rng.random() >= epsilon
                else int(rng.choice(np.flatnonzero(allowed)))
            )
            next_obs, reward, terminated, truncated, after = env.step(game_action(action))
            if info["episode_id"] != after["episode_id"] or after["elapsed_ticks"] <= 0:
                raise ValueError("noncontiguous STEP response")
            for key in FIELDS:
                arrays["s_" + key].append(obs[key])
                arrays["n_" + key].append(next_obs[key])
            contacts = np.zeros(4, np.float32)
            dealt = 0.0
            win = False
            for event in after["reward_events"]:
                if event.kind == protocol.RewardEventKind.DAMAGE_TAKEN:
                    contacts[event.aux_int2 if 0 <= event.aux_int2 < 4 else 0] += event.amount
                elif event.kind == protocol.RewardEventKind.DAMAGE_DEALT:
                    dealt += event.amount
                elif event.kind == protocol.RewardEventKind.BOSS_KILLED:
                    win = True
            visible = next_obs["entities"][next_obs["entity_mask"].astype(bool)]
            bodies = int((visible[:, 1] == protocol.EntityType.BOSS).sum())
            multi += int(bodies >= 2)
            for k, v in dict(
                a=action,
                r=reward,
                term=terminated,
                trunc=truncated,
                mask=allowed,
                next_mask=legal_actions(after["action_mask"]),
                ticks=after["elapsed_ticks"],
                damage=dealt,
                contacts=contacts,
                bosses=bodies,
                win=win,
                episode=after["episode_id"],
                tick=after["server_tick"],
                port=port,
            ).items():
                arrays[k].append(v)
            done = terminated or truncated
            episodes += int(done)
            obs, info = next_obs, after
    finally:
        try:
            env.unwrapped.set_timescale(1.0)
        finally:
            env.close()
    elapsed = time.monotonic() - start
    packed = {k: np.asarray(v) for k, v in arrays.items()}
    return packed, dict(
        port=port,
        steps=steps,
        seconds=elapsed,
        ended=episodes,
        resets=resets,
        ticks=int(packed["ticks"].sum()),
        multi_steps=multi,
        median_scale=float(np.median(packed["n_global"][:, 5])),
        damage=float(packed["damage"].sum()),
        spikes=float(packed["contacts"][:, 3].sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bench", "collect", "eval"))
    parser.add_argument("--settings", default="configs/train/sisters_ddqn.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--ports", type=int, nargs="+", default=[5555, 5556, 5557])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.settings).read_text())
    if args.start < 0 or args.steps < len(args.ports) or len(set(args.ports)) != len(args.ports):
        parser.error("invalid collection size or ports")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    net = get("model", cfg["model"])(cfg["hidden"]).eval()
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if ck["steps"] != args.start:
            raise ValueError("checkpoint does not match the collection boundary")
        net.load_state_dict(ck["weights"])
    elif args.start or args.mode == "eval":
        parser.error("a trained checkpoint is required")
    policy = Policy(net)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_game_lock(Path("runs/training/game.lock")):
        if args.mode == "eval":
            if args.start != cfg["total_steps"]:
                raise ValueError("expected final checkpoint")
            task = load_task_config("configs/experiments/sisters_ddqn.yaml")
            seeds = list(range(33410, 33420))
            port = args.ports[0]
            evaluation_env = EvaluationEnv(port)
            with args.output.with_suffix(".jsonl").open("x", encoding="utf-8") as stream:
                evaluator = Evaluator(
                    policy,
                    tasks=[task],
                    seeds=seeds,
                    env_factory=lambda _: evaluation_env,
                    replay_sink=lambda row: stream.write(json.dumps(row) + "\n"),
                )
                metrics = evaluator.evaluate(episodes_per_task=10)
            observed_scales = [float(np.median(series)) for series in evaluation_env.scales]
            if len(observed_scales) != 10 or not np.allclose(observed_scales, 1.0):
                raise ValueError(f"original-speed evaluation failed: {observed_scales}")
            doc = dict(
                metadata=dict(
                    ports=[port],
                    observed_median_timescales=observed_scales,
                    episodes=10,
                    task_ids=[task.task_id],
                    seeds=seeds,
                    model=cfg["model"],
                    checkpoint=str(args.checkpoint),
                    game_seed_controlled=False,
                ),
                metrics=metrics,
            )
        else:
            start = time.monotonic()
            eps = max(
                cfg["epsilon_final"],
                1 - (1 - cfg["epsilon_final"]) * args.start / cfg["epsilon_decay_steps"],
            )
            with ThreadPoolExecutor(len(args.ports)) as pool:
                jobs = [
                    pool.submit(
                        collect_worker,
                        p,
                        args.steps // len(args.ports) + int(i < args.steps % len(args.ports)),
                        policy,
                        cfg["seed"] + args.start,
                        eps,
                        2.0,
                        args.mode == "bench",
                    )
                    for i, p in enumerate(args.ports)
                ]
                batches, stats = zip(*(j.result() for j in jobs), strict=True)
            elapsed = time.monotonic() - start
            combined = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
            doc = dict(
                start=args.start,
                steps=len(combined["a"]),
                epsilon=eps,
                seconds=elapsed,
                decisions_per_wall_second=args.steps / elapsed,
                workers=stats,
            )
            if args.mode == "collect":
                tmp = args.output.with_suffix(".part")
                with tmp.open("wb") as stream:
                    np.savez_compressed(stream, **combined)
                tmp.replace(args.output)
        args.output.with_suffix(".json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(json.dumps(doc), flush=True)


if __name__ == "__main__":
    main()

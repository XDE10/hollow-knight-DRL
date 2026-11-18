#!/usr/bin/env python3
"""Measure real HKRL models/optimizers using SYNTHETIC observations, without the game.

This checks compute capacity and checkpoint portability, not gameplay or win rate.
Sampling always runs on CPU; --device chooses the optimizer device.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hkrl.cli import _build_model, _obs_dims
from hkrl.spaces import action_mask_layout, make_action_space, make_observation_space
from hkrl.utils.config import TaskConfig, load_task_config, load_train_config
from hkrl.utils.registry import get
from hkrl.worker.game_worker import GameWorker


class SyntheticEnv:
    """A shape-correct test fixture. Actions do not control a game here."""

    def __init__(self, task: TaskConfig) -> None:
        self.task = task
        self.action_space = make_action_space(
            enable_macro=task.action.enable_macro_actions,
            n_macros=task.action.n_macro_actions,
        )
        self.observation_space = make_observation_space(
            max_entities=task.observation.max_entities, tier=task.observation.tier
        )
        self.mask = np.ones(
            len(
                action_mask_layout(
                    enable_macro=task.action.enable_macro_actions,
                    n_macros=task.action.n_macro_actions,
                )
            ),
            dtype=bool,
        )
        self.step_count = 0
        self.episode_id = 0

    def observation(self) -> dict[str, np.ndarray]:
        obs = {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }
        obs["entity_mask"][:8] = True
        obs["entities"][:8, 0] = np.arange(1, 9)
        obs["entities"][:8, 1] = 1
        obs["entities"][:8, 6:10] = np.sin(self.step_count / 20.0)
        obs["player"][:4] = np.cos(self.step_count / 20.0)
        return obs

    def info(self) -> dict[str, Any]:
        return {
            "action_mask": self.mask,
            "episode_id": self.episode_id,
            "task_id": self.task.wire_id,
            "elapsed_ticks": self.task.action.action_repeat,
        }

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.step_count = 0
        self.episode_id += 1
        return self.observation(), self.info()

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("worker produced an invalid action")
        self.step_count += 1
        return (
            self.observation(),
            float(np.sin(self.step_count / 10.0)),
            self.step_count >= 128,
            False,
            self.info(),
        )


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure(config_path: str, args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = load_train_config(config_path)
    cfg.rollout_steps = args.steps
    cfg.epochs = args.epochs
    cfg.minibatch_size = args.minibatch_size
    task = load_task_config("configs/tasks/mantis_lords.yaml")
    env = SyntheticEnv(task)
    model = _build_model(
        cfg,
        _obs_dims(env.observation_space),
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    model.eval()
    obs, info = env.reset()
    tensor_obs = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
    mask = torch.as_tensor(info["action_mask"][None])
    latencies = []
    state = model.initial_state(1)
    for i in range(120):
        start = time.perf_counter()
        with torch.inference_mode():
            *_, state = model.act(tensor_obs, rnn_state=state, action_mask=mask)
        if i >= 20:
            latencies.append((time.perf_counter() - start) * 1000)

    worker = GameWorker(env=env, model=model, config=cfg)
    start = time.perf_counter()
    worker.collect_rollout()
    collection_s = time.perf_counter() - start
    model.to(args.device)
    algo = get("algo", cfg.algorithm)(model=model, config=cfg)
    before = [p.detach().cpu().clone() for p in model.parameters()]
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(args.device)
    start = time.perf_counter()
    metrics = algo.update(worker.buffer)
    sync(args.device)
    update_s = time.perf_counter() - start
    changed = any(
        not torch.equal(a, b.detach().cpu())
        for a, b in zip(before, model.parameters(), strict=True)
    )
    if not changed or not all(np.isfinite(v) for v in metrics.values()):
        raise RuntimeError("optimizer failed to make a finite parameter update")

    out = Path(args.output).parent
    checkpoint = out / f"synthetic_{cfg.algorithm}_{args.device}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": algo.optimizer.state_dict(),
            "synthetic_only": True,
        },
        checkpoint,
    )
    restored = _build_model(
        cfg,
        _obs_dims(env.observation_space),
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "model_state_dict"
        ]
    )
    restored.eval()
    restored.act(tensor_obs, action_mask=mask)
    return {
        "config": config_path,
        "algorithm": cfg.algorithm,
        "parameters": sum(p.numel() for p in model.parameters()),
        "observation_shapes": _obs_dims(env.observation_space),
        "cpu_inference_median_ms": float(np.median(latencies)),
        "cpu_inference_p95_ms": float(np.percentile(latencies, 95)),
        "synthetic_collection_s": collection_s,
        "steps": args.steps,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "optimizer_device": args.device,
        "optimizer_update_s": update_s,
        "finite_metrics": True,
        "parameters_changed": changed,
        "checkpoint_reloaded_on_cpu": True,
        "metrics": metrics,
        "checkpoint": str(checkpoint),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--output", default="runs/validation/compute_cpu.json")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/train/ppo_mlp.yaml", "configs/train/ppo_attention_gru.yaml"],
    )
    args = parser.parse_args()
    if min(args.threads, args.steps, args.epochs, args.minibatch_size) <= 0:
        parser.error("numeric arguments must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "synthetic_only": True,
        "game_tested": False,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": args.threads,
        "results": [],
    }
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    for config_path in args.configs:
        result = measure(config_path, args)
        report["results"].append(result)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

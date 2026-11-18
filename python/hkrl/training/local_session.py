"""Resumable local CPU training from one or more synchronous game instances."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import sys
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from hkrl import protocol
from hkrl.env import EnvProtocolError
from hkrl.learner.checkpoint_payload import validate_checkpoint_payload, validate_model_state_dict
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.training.batch_io import save_rollout_batch
from hkrl.training.demonstration_replay import DemonstrationReplay
from hkrl.training.recurrent_buffer import RecurrentRolloutBuffer
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.live_tuning import atomic_write_json
from hkrl.utils.logging import JsonlSink
from hkrl.utils.registry import get
from hkrl.utils.seeding import seed_everything


@contextmanager
def exclusive_game_lock(path: Path) -> Iterator[None]:
    """OS-owned lock: released even if a training process crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another training/evaluation process owns the game") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another training/evaluation process owns the game") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def capture_rng(algo: Any) -> dict[str, Any]:
    numpy_state: Any = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
    }
    generator = getattr(algo, "_rng", None)
    if generator is not None:
        state["minibatch"] = generator.bit_generator.state
    return state


def restore_rng(state: dict[str, Any], algo: Any) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if "minibatch" in state:
        algo._rng.bit_generator.state = state["minibatch"]


class LocalSession:
    """Checkpoint model, Adam, RNG and counters; restart gameplay with a clean reset.

    Unity state is not serialized. Resume is an optimizer continuation, not an
    assertion of bit-exact game trajectories. Only CPU synchronous PPO is supported.
    """

    def __init__(self, root: Path, identity: dict[str, Any]) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        manifest = root / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != identity:
                raise ValueError("run configuration changed; use a new run directory")
        else:
            atomic_write_json(manifest, identity)
        self.registry = CheckpointRegistry(str(root / "checkpoints"))
        self.update = 0
        self.steps = 0
        self.wall_seconds = 0.0
        self.last_checkpoint: str | None = None
        self.last_phase = "initializing"

    def restore(self, model: Any, algo: Any) -> bool:
        meta = self.registry.latest()
        if meta is None:
            return False
        path = self.registry.resolve_path(meta)
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta.sha256:
            raise ValueError("checkpoint hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("run_identity") != self.identity:
            raise ValueError("checkpoint does not belong to this run")
        model.load_state_dict(state["model_state_dict"], strict=True)
        algo.optimizer.load_state_dict(state["optimizer_state_dict"])
        restore_rng(state["rng_state"], algo)
        self.update = int(state["update"])
        self.steps = int(state["step"])
        self.wall_seconds = float(state["wall_seconds"])
        self.last_checkpoint = str(path)
        return True

    def save(self, model: Any, algo: Any) -> None:
        meta = self.registry.publish(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": algo.optimizer.state_dict(),
                "rng_state": capture_rng(algo),
                "run_identity": self.identity,
                "policy_version": self.update,
                "update": self.update,
                "step": self.steps,
                "wall_seconds": self.wall_seconds,
            },
            policy_version=self.update,
            step=self.steps,
        )
        self.last_checkpoint = str(self.registry.resolve_path(meta))

    def status(self, phase: str, **extra: Any) -> dict[str, Any]:
        self.last_phase = phase
        record = {
            "status": phase,
            "pid": os.getpid(),
            "update": self.update,
            "total_steps": self.steps,
            "wall_seconds": self.wall_seconds,
            "last_checkpoint": self.last_checkpoint,
            "timestamp": time.time(),
            **extra,
        }
        atomic_write_json(self.root / "status.json", record)
        return record


class ResetTimeout(gym.Wrapper):
    def __init__(self, env: gym.Env, timeout: float, scene_load_retries: int = 0) -> None:
        super().__init__(env)
        if scene_load_retries < 0:
            raise ValueError("scene_load_retries must be nonnegative")
        self.timeout = timeout
        self.scene_load_retries = scene_load_retries
        self.retried_seeds: list[int | None] = []

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        options = {"reset_timeout_s": self.timeout, **(kwargs.pop("options", None) or {})}
        retries = 0
        while True:
            try:
                return self.env.reset(options=options, **kwargs)
            except EnvProtocolError as exc:
                if (
                    exc.status_code != protocol.StatusCode.SCENE_LOAD_FAILED
                    or retries >= self.scene_load_retries
                ):
                    raise
                retries += 1
                self.retried_seeds.append(kwargs.get("seed"))
                logging.getLogger(__name__).warning(
                    "Retrying scene load before episode start (seed=%s, attempt=%d): %s",
                    kwargs.get("seed"),
                    retries + 1,
                    exc,
                )


def run_session_from_args(args: argparse.Namespace) -> dict[str, Any]:
    from hkrl.cli import _build_model, _make_env_transport, _obs_dims
    from hkrl.env import HKRLEnv
    from hkrl.worker.game_worker import GameWorker
    from hkrl.worker.local_inference import LocalActionBatcher
    from hkrl.wrappers import NormalizeObservation

    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    if cfg.algorithm not in {"ppo", "recurrent_ppo"} or cfg.learner.device != "cpu":
        raise ValueError("local sessions require CPU ppo/recurrent_ppo")
    if args.updates <= 0 or args.checkpoint_every <= 0 or args.reset_timeout <= 0:
        raise ValueError("updates, checkpoint-every and reset-timeout must be positive")
    root = Path(args.run_dir).resolve()
    coordination = Path("runs/training").resolve()
    with exclusive_game_lock(coordination / "game.lock"), ExitStack() as resources:
        prior_manifest = root / "manifest.json"
        prior = (
            json.loads(prior_manifest.read_text(encoding="utf-8"))
            if prior_manifest.exists()
            else {}
        )
        requested_ports = getattr(args, "ports", None)
        if requested_ports is None and getattr(args, "port", None) is None:
            requested_ports = prior.get("sampling_ports")
        ports = requested_ports or [getattr(args, "port", None) or cfg.transport.port]
        if requested_ports and (
            cfg.transport.name != "tcp"
            or (getattr(args, "host", None) or cfg.transport.host) not in {"127.0.0.1", "localhost"}
            or any(not 1 <= p <= 65535 for p in ports)
            or len(set(ports)) != len(ports)
            or cfg.rollout_steps % len(ports)
            or (len(ports) > 1 and cfg.algorithm != "recurrent_ppo")
        ):
            raise ValueError(
                "--ports needs distinct loopback games and an evenly divisible recurrent rollout"
            )
        identity: dict[str, Any] = {
            "format": 1,
            "schema_version": protocol.SCHEMA_VERSION,
            "train_config": cfg.model_dump(mode="json"),
            "task_config": task.model_dump(mode="json"),
            "game_seed_controlled": False,
        }
        if requested_ports:
            identity["sampling_ports"] = list(ports)
        batch_inference = getattr(args, "batch_inference", False) or prior.get(
            "batch_inference", False
        )
        if batch_inference:
            if len(ports) < 2:
                raise ValueError("--batch-inference requires at least two local game ports")
            identity["batch_inference"] = True
        initial_weights = getattr(args, "init_weights", None)
        initial_bytes = Path(initial_weights).read_bytes() if initial_weights else None
        if initial_bytes is not None:
            identity["initial_weights_sha256"] = hashlib.sha256(initial_bytes).hexdigest()
        elif "initial_weights_sha256" in prior:
            identity["initial_weights_sha256"] = prior["initial_weights_sha256"]
        replay_path = getattr(args, "demo_replay", None)
        if replay_path is None and "demo_replay" in prior:
            replay_path = root / prior["demo_replay"]["directory"]
        replay = None
        if replay_path:
            if cfg.algorithm != "recurrent_ppo":
                raise ValueError("--demo-replay requires recurrent PPO")
            directory = Path(replay_path).resolve()
            replay = DemonstrationReplay(directory, cfg.sequence_length, cfg.burn_in)
            identity["demo_replay"] = {
                "directory": (
                    os.path.relpath(directory, root)
                    if directory.drive == root.drive
                    else str(directory)
                ),
                "training_files": replay.hashes,
            }
        session = LocalSession(root, identity)
        seed_everything(cfg.seed)
        torch.set_num_threads(1)
        games = []
        envs = []
        reset_wrappers: list[ResetTimeout] = []
        for port in ports:
            env_args = argparse.Namespace(**vars(args))
            if requested_ports:
                env_args.port = port
            game = HKRLEnv(
                transport=_make_env_transport(cfg, env_args),
                task=task,
                step_timeout_s=float(getattr(args, "step_timeout", 15.0)),
            )
            games.append(game)
            reset = ResetTimeout(game, float(args.reset_timeout), getattr(args, "reset_retries", 0))
            reset_wrappers.append(reset)
            env = NormalizeObservation(reset)
            resources.callback(env.close)
            envs.append(env)
        model = _build_model(
            cfg,
            _obs_dims(envs[0].observation_space),
            enable_macro=task.action.enable_macro_actions,
            n_macros=task.action.n_macro_actions,
            max_entities=task.observation.max_entities,
        )
        algo = get("algo", cfg.algorithm)(model=model, config=cfg)

        if replay is not None:
            algo.demonstrations = replay

        def report_progress(progress: dict[str, Any]) -> None:
            session.status("sampling", target_updates=args.updates, **progress)

        def report_recovery(progress: dict[str, Any]) -> None:
            with status_lock:
                record = session.status(
                    "recovering",
                    target_updates=args.updates,
                    **{key: value for key, value in progress.items() if key != "status"},
                )
            print(json.dumps(record, sort_keys=True), flush=True)

        sampling_cfg = cfg.model_copy(update={"rollout_steps": cfg.rollout_steps // len(ports)})
        batcher = LocalActionBatcher(model, len(ports)) if batch_inference else None
        if batcher is not None:
            resources.callback(batcher.close)
        workers = [
            GameWorker(
                env=worker_env,
                model=model,
                config=sampling_cfg,
                rollout_progress_sink=report_progress if len(ports) == 1 else None,
                heartbeat_sink=report_recovery,
                max_consecutive_failures=int(getattr(args, "sampling_retries", 2)),
                inference_batcher=batcher,
            )
            for worker_env in envs
        ]
        pool = (
            resources.enter_context(ThreadPoolExecutor(max_workers=len(ports)))
            if len(ports) > 1
            else None
        )
        paused = [False] * len(ports)
        status_lock = Lock()

        def sample(index: int) -> Any:
            if paused[index]:
                games[index].resume()
                paused[index] = False
            batch = workers[index].collect_rollout_with_recovery()
            paused[index] = True
            games[index].pause()
            return batch

        sink = JsonlSink(root / "metrics.jsonl")
        active = coordination / "active.json"
        atomic_write_json(active, {"pid": os.getpid(), "run_dir": str(root)})
        try:
            resumed = session.restore(model, algo)
            for worker in workers:
                worker.policy_version = session.update
            if not resumed:
                if "initial_weights_sha256" in identity:
                    if initial_bytes is None:
                        raise ValueError(
                            "no resumable checkpoint; supply the original --init-weights"
                        )
                    weights = torch.load(
                        io.BytesIO(initial_bytes), map_location="cpu", weights_only=True
                    )
                    if isinstance(weights, dict) and "model_state_dict" in weights:
                        weights = validate_checkpoint_payload(weights)["model_state_dict"]
                    model.load_state_dict(validate_model_state_dict(weights), strict=True)
                session.save(model, algo)
            # A previous stop is acknowledged by launching this session again.
            (root / "STOP").unlink(missing_ok=True)
            while session.update < args.updates:
                if (root / "STOP").exists():
                    break
                session.status("sampling", target_updates=args.updates)
                started = time.monotonic()
                passes_before = batcher.forward_passes if batcher else 0
                decisions_before = batcher.decisions if batcher else 0
                if pool is None:
                    batch = sample(0)
                    buffer = workers[0].buffer
                else:
                    futures = [pool.submit(sample, i) for i in range(len(workers))]
                    # Drain every game before cleanup or any learner mutation on failure.
                    wait(futures)
                    for future in futures:
                        future.result()
                    buffer = RecurrentRolloutBuffer.combine_envs(
                        [cast(RecurrentRolloutBuffer, worker.buffer) for worker in workers]
                    )
                    batch = buffer.to_batch(session.update)
                sample_seconds = time.monotonic() - started
                if getattr(args, "save_rollouts", False):
                    # Unique attempts survive an interrupted update and a replayed policy version.
                    save_rollout_batch(
                        root / "rollouts" / f"policy-{session.update:06d}-{time.time_ns()}.npz",
                        batch,
                    )
                session.status("updating", target_updates=args.updates)
                update_started = time.monotonic()
                metrics = {key: float(value) for key, value in algo.update(buffer).items()}
                if batcher:
                    metrics["mean_inference_batch_size"] = (
                        batcher.decisions - decisions_before
                    ) / max(1, batcher.forward_passes - passes_before)
                update_seconds = time.monotonic() - update_started
                session.update += 1
                session.steps += int(batch.rewards.size)
                session.wall_seconds += time.monotonic() - started
                for worker in workers:
                    worker.policy_version = session.update
                stopping = (root / "STOP").exists()
                if (
                    session.update % args.checkpoint_every == 0
                    or stopping
                    or session.update == args.updates
                ):
                    session.save(model, algo)
                record = session.status(
                    "checkpointed" if stopping else "running",
                    target_updates=args.updates,
                    sample_seconds=sample_seconds,
                    update_seconds=update_seconds,
                    sps=batch.rewards.size / max(sample_seconds, 1e-9),
                    rollout_reward=float(batch.rewards.sum()),
                    metrics=metrics,
                    sampling_failures=sum(w.worker_crash_count for w in workers),
                    discarded_rollout_steps=sum(w.discarded_rollout_steps for w in workers),
                )
                if getattr(args, "reset_retries", 0):
                    metrics["retried_scene_loads"] = float(
                        sum(len(wrapper.retried_seeds) for wrapper in reset_wrappers)
                    )
                sink.log_episode({"type": "update", **record})
                sink.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                if stopping:
                    break
            # Covers a stop arriving just after the periodic-checkpoint decision.
            meta = session.registry.latest()
            if meta is not None and meta.policy_version != session.update:
                session.save(model, algo)
            return session.status(
                "completed" if session.update >= args.updates else "stopped",
                target_updates=args.updates,
            )
        except BaseException as exc:
            # Never overwrite a valid checkpoint with a half-completed optimizer update.
            session.status(
                "failed",
                failed_phase=session.last_phase,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exception(type(exc), exc, exc.__traceback__, limit=12),
            )
            raise
        finally:
            try:
                with ExitStack() as resumes:
                    for game, is_paused in zip(games, paused, strict=True):
                        if is_paused:
                            resumes.callback(game.resume)
            finally:
                sink.close()
                active.unlink(missing_ok=True)

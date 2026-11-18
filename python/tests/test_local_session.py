from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from hkrl.training.local_session import LocalSession, exclusive_game_lock
from hkrl.utils.seeding import seed_everything


def make_pair() -> tuple[torch.nn.Linear, SimpleNamespace]:
    model = torch.nn.Linear(2, 1)
    return model, SimpleNamespace(
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        _rng=np.random.default_rng(42),
    )


def update(model: torch.nn.Linear, algo: SimpleNamespace) -> None:
    algo.optimizer.zero_grad()
    # Every restored RNG stream affects the next real Adam update.
    factor = random.random() + np.random.random() + algo._rng.random()
    model(torch.rand(3, 2)).square().mean().mul(factor).backward()
    algo.optimizer.step()


def test_resume_reproduces_next_adam_update_and_all_rng_streams(tmp_path: Path) -> None:
    seed_everything(7)
    model, algo = make_pair()
    update(model, algo)
    session = LocalSession(tmp_path, {"task": "sisters", "seed": 7})
    session.update, session.steps, session.wall_seconds = 1, 512, 42.0
    session.save(model, algo)
    update(model, algo)
    expected = {key: value.clone() for key, value in model.state_dict().items()}

    restored_model, restored_algo = make_pair()
    restored = LocalSession(tmp_path, {"task": "sisters", "seed": 7})
    assert restored.restore(restored_model, restored_algo)
    assert (restored.update, restored.steps, restored.wall_seconds) == (1, 512, 42.0)
    update(restored_model, restored_algo)
    for key, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert restored_algo.optimizer.state_dict()["state"][0]["step"].item() == 2


def test_session_rejects_changed_config_and_corrupted_weights(tmp_path: Path) -> None:
    session = LocalSession(tmp_path, {"seed": 1})
    model, algo = make_pair()
    session.save(model, algo)
    with pytest.raises(ValueError, match="configuration changed"):
        LocalSession(tmp_path, {"seed": 2})
    assert session.last_checkpoint
    with Path(session.last_checkpoint).open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        session.restore(model, algo)


def test_game_lock_blocks_second_owner_and_releases_on_error(tmp_path: Path) -> None:
    path = tmp_path / "game.lock"
    with pytest.raises(ValueError, match="test failure"), exclusive_game_lock(path):
        with pytest.raises(RuntimeError, match="owns the game"), exclusive_game_lock(path):
            pytest.fail("second owner entered")
        raise ValueError("test failure")
    with exclusive_game_lock(path):
        pass


@pytest.mark.parametrize("initial_format", ["none", "weights", "checkpoint"])
def test_safe_stop_then_resume_keeps_counters_and_pauses_updates(
    initial_format: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hkrl import cli
    from hkrl.training import local_session
    from hkrl.utils.config import TaskConfig, TrainConfig
    from hkrl.worker import game_worker

    order: list[str] = []
    archived: list[Path] = []
    root = tmp_path / "session"

    class FakeEnv(gym.Env):
        def __init__(self, **kwargs: object) -> None:
            self.observation_space = gym.spaces.Dict({})
            self.action_space = gym.spaces.Discrete(2)

        def pause(self) -> None:
            order.append("pause")

        def resume(self) -> None:
            order.append("resume")

        def close(self) -> None:
            order.append("close")

    class Worker:
        def __init__(self, **kwargs: object) -> None:
            self.buffer = None
            self.policy_version = 0
            self.worker_crash_count = 0
            self.discarded_rollout_steps = 0

        def collect_rollout_with_recovery(self) -> SimpleNamespace:
            order.append("sample")
            if self.policy_version == 0:
                (root / "STOP").touch()
            return SimpleNamespace(rewards=np.zeros(4))

    class Algo:
        def __init__(self, model: torch.nn.Linear, **kwargs: object) -> None:
            self.model = model
            self.optimizer = torch.optim.Adam(model.parameters())

        def update(self, buffer: object) -> dict[str, float]:
            assert order[-1] == "pause"
            order.append("update")
            self.optimizer.zero_grad()
            self.model(torch.ones(1, 2)).sum().backward()
            self.optimizer.step()
            return {"loss": 1.0}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        local_session, "load_train_config", lambda _: TrainConfig(learner={"device": "cpu"})
    )
    monkeypatch.setattr(
        local_session, "load_task_config", lambda _: TaskConfig(task_id="test", scene="Test")
    )
    monkeypatch.setattr(local_session, "get", lambda *_: Algo)
    monkeypatch.setattr("hkrl.env.HKRLEnv", FakeEnv)
    monkeypatch.setattr(cli, "_obs_dims", lambda _: {})
    monkeypatch.setattr(cli, "_make_env_transport", lambda *_: None)
    monkeypatch.setattr(cli, "_build_model", lambda *_, **__: torch.nn.Linear(2, 1))
    monkeypatch.setattr(game_worker, "GameWorker", Worker)

    def archive(path: Path, batch: SimpleNamespace) -> None:
        assert order[-1] == "pause"
        assert batch.rewards.size == 4
        archived.append(path)

    monkeypatch.setattr(local_session, "save_rollout_batch", archive)
    args = argparse.Namespace(
        config="test",
        task="test",
        run_dir=str(root),
        updates=2,
        save_rollouts=True,
        checkpoint_every=5,
        reset_timeout=90,
    )
    if initial_format != "none":
        teacher = torch.nn.Linear(2, 1)
        with torch.no_grad():
            teacher.weight.fill_(3.0)
        args.init_weights = str(tmp_path / "teacher.pt")
        weights = teacher.state_dict()
        payload = (
            {"model_state_dict": weights, "step": 10240, "optimizer_state_dict": {"old": True}}
            if initial_format == "checkpoint"
            else weights
        )
        torch.save(payload, args.init_weights)
    first = local_session.run_session_from_args(args)
    if initial_format != "none":
        initial = torch.load(root / "checkpoints/checkpoint_v000001.pt", weights_only=True)
        torch.testing.assert_close(initial["model_state_dict"]["weight"], teacher.weight)
        assert initial["step"] == 0 and not initial["optimizer_state_dict"]["state"]
        args.init_weights = None
    assert (first["status"], first["update"], first["total_steps"]) == ("stopped", 1, 4)
    assert order == ["sample", "pause", "update", "resume", "close"]
    second = local_session.run_session_from_args(args)
    assert (second["status"], second["update"], second["total_steps"]) == ("completed", 2, 8)
    state = torch.load(second["last_checkpoint"], weights_only=True)
    assert state["optimizer_state_dict"]["state"][0]["step"].item() == 2
    assert not (tmp_path / "runs/training/active.json").exists()
    assert len(archived) == 2 and archived[0] != archived[1]

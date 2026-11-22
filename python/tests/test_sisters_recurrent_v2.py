from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def trainer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_recurrent_ppo")


def test_native_draw_receipt_covers_every_trio_decision(trainer, tmp_path: Path) -> None:
    path = tmp_path / "episode.ppo"
    digest = bytes.fromhex("a0" * 32)
    clock = np.array([0.0, 0.04, 0.08, 0.2], dtype="f4")
    state = np.zeros((len(clock), 192), dtype="f4")
    state[1:, 23] = 1
    draws = np.zeros(3, dtype=trainer.DRAW_ROW)
    draws["time"] = clock[1:]
    draws["uniform"] = trainer.uniforms(1, 19, 3)
    draws["log_probability"] = -0.3
    raw = trainer.DRAW_HEADER.pack(0x314F5050, 1, digest, 19, 0.55)
    path.write_bytes(raw + draws.tobytes())
    result = trainer.read_native_draws(path, 1, digest.hex(), 19, 0.55, clock, state)
    np.testing.assert_array_equal(result["ppo_rows"], [1, 2, 3])
    draws["uniform"][0] = trainer.uniforms(2, 19, 1)[0]
    path.write_bytes(raw + draws.tobytes())
    with pytest.raises(ValueError, match="verified game RNG"):
        trainer.read_native_draws(path, 1, digest.hex(), 19, 0.55, clock, state)


def test_ppo_updates_recurrence_with_validated_world_auxiliary(trainer) -> None:
    episode_steps = 12
    catalogue = torch.tensor([[0, 0, 0], [1, 0, 8]])
    actor, critic = trainer.RecurrentActor(catalogue, 6), trainer.RecurrentValues(catalogue, 6)
    x = torch.rand(episode_steps, 578)
    state = actor.memory.weight_hh_l0.detach().clone()
    world = trainer.World(198, 2).requires_grad_(False)
    before = world.head.weight.detach().clone()
    with torch.no_grad():
        logits = (actor(x[None])[0] / 0.55).log_softmax(-1)
        values = critic(x[None])[2][0]
    rollout = trainer.Rollout(
        x,
        torch.arange(episode_steps) % 2,
        logits,
        values,
        torch.linspace(-1, 1, episode_steps),
        values + 0.2,
        torch.arange(2),
    )
    mask = torch.tensor([[True, False], [True, True], [True, True]])
    progress = trainer.optimize(actor, critic, [rollout], [world], mask, 0.55, 1)
    assert progress[0]["accepted"] and progress[0]["world_aux"] > 0
    assert not torch.equal(actor.memory.weight_hh_l0, state)
    torch.testing.assert_close(world.head.weight, before)


def test_gae_drops_future_advantages_at_gap_but_keeps_observed_bootstrap(trainer) -> None:
    clock = np.array([0.00, 0.04, 0.16, 0.20], dtype="f4")
    reward = np.array([1.0, 3.0, 2.0, 7.0], dtype="f4")
    value = np.array([2.0, 3.0, 4.0, 5.0], dtype="f4")
    before, _ = trainer.native_gae(reward, value, clock, False)
    reward[2:] += 100
    after, _ = trainer.native_gae(reward, value, clock, False)
    np.testing.assert_array_equal(before[:2], after[:2])
    assert before[1] == pytest.approx(3 + trainer.GAMMA**3 * 4 - 3)
    assert before[-1] == pytest.approx(7 - 5)
    timeout, _ = trainer.native_gae(reward, value, clock, True)
    assert len(timeout) == 3
    assert timeout[-1] == pytest.approx(reward[2] + trainer.GAMMA * 5 - 4)

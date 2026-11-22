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
    actions = torch.arange(episode_steps) % 2
    advantage = torch.linspace(-1, 1, episode_steps)
    actions[-1] = -1
    advantage[-1] = float("nan")
    values[-1] = float("nan")
    logits[-1] = float("nan")
    rollout = trainer.Rollout(
        x,
        actions,
        logits,
        values,
        advantage,
        values + 0.2,
        torch.arange(2),
        torch.arange(episode_steps) != episode_steps - 1,
    )
    mask = torch.tensor([[True, False], [True, True], [True, True]])
    progress = trainer.optimize(actor, critic, [rollout], [world], mask, 0.55, 1)
    assert progress[0]["accepted"] and progress[0]["world_aux"] > 0
    assert not torch.equal(actor.memory.weight_hh_l0, state)
    torch.testing.assert_close(world.head.weight, before)


@pytest.mark.parametrize("invalid_interval", [0.02, 0.12])
def test_gae_excludes_invalid_reward_and_stops_before_gap(trainer, invalid_interval) -> None:
    clock = np.array([0.0, 0.04, 0.04 + invalid_interval, 0.08 + invalid_interval], dtype="f4")
    reward = np.array([1.0, 3.0, 2.0, 7.0], dtype="f4")
    value = np.array([2.0, 3.0, 4.0, 5.0], dtype="f4")
    valid = np.array([True, False, True, True])
    terminated = np.array([False, False, False, True])
    before, target = trainer.native_gae(reward, value, clock, valid, terminated)
    reward[1:] += 100
    after, _ = trainer.native_gae(reward, value, clock, valid, terminated)
    assert before[0] == after[0] == pytest.approx(1 + trainer.GAMMA * 3 - 2)
    assert before[1] == after[1] == target[1] == 0
    assert before[-1] == pytest.approx(7 - 5)
    assert target[-1] == pytest.approx(7)


def test_timeout_bootstraps_only_from_a_valid_adjacent_observation(trainer) -> None:
    reward = np.array([1.0, 3.0, 99.0], dtype="f4")
    value = np.array([2.0, 3.0, 5.0], dtype="f4")
    clock = np.array([0.0, 0.04, 0.08], dtype="f4")
    terminated = np.zeros(3, dtype=bool)
    timeout, _ = trainer.native_gae(reward, value, clock, np.array([True, True, False]), terminated)
    assert timeout[1] == pytest.approx(3 + trainer.GAMMA * 5 - 3)
    assert timeout[2] == 0
    clock[-1] = 0.16
    gap, _ = trainer.native_gae(reward, value, clock, np.array([True, False, False]), terminated)
    assert gap[0] == pytest.approx(1 + trainer.GAMMA * 3 - 2)
    np.testing.assert_array_equal(gap[1:], 0)


def test_timeout_prefix_keeps_only_observed_native_rows(trainer) -> None:
    collector = importlib.import_module("collect_sisters_native_labels")
    clock = np.arange(6, dtype="f4") * 0.04
    decisions = {"control_time": clock, "timing_state": np.zeros((6, 192), dtype="f4")}
    draws = {"ppo_rows": np.arange(2, 6), "ppo_action": np.arange(4)}
    trace, ppo = collector.ppo_timeout_prefix(decisions, draws, float(clock[3]), float(clock[2]))
    np.testing.assert_array_equal(trace["control_time"], clock[:4])
    np.testing.assert_array_equal(ppo["ppo_rows"], [2, 3])
    np.testing.assert_array_equal(ppo["ppo_action"], [0, 1])
    with pytest.raises(ValueError, match="boundary observation"):
        collector.ppo_timeout_prefix(decisions, draws, float(clock[3]), float(clock[3]))


def test_teacher_losses_train_iql_without_behavior_cloning(trainer) -> None:
    entry = importlib.import_module("train_hkrl_wm_recurrent_rl")
    n = 40
    observation = np.zeros((n, 192), dtype="f4")
    observation[:, 23] = 1
    events = np.zeros((n, 8), dtype="f4")
    events[-1, 7] = 1
    terminal = np.arange(n) == n - 1
    controls = np.zeros((n, 3), dtype="i8")
    episode = entry.Episode(
        observation,
        controls,
        np.zeros(n, dtype="f4"),
        events,
        np.arange(n, dtype="f4") * 0.04,
        np.r_[np.arange(1, n), -1],
        terminal,
        np.zeros(n, dtype=bool),
        np.full(n, "teacher"),
    )
    fight = entry.prepare_fight(episode, {(0, 0, 0): 0})
    windows = entry.Windows([fight], np.random.default_rng(0), teacher_win_bc=True)
    *_, learn, behavior_clone = windows.batch(2, 16, teacher_only=False)
    assert learn.any()
    assert not behavior_clone.any()


def test_teacher_receipt_must_match_next_native_observation(trainer) -> None:
    pack = importlib.import_module("pack_sisters_teacher_corpus")
    timing = np.zeros((3, 192), dtype="f4")
    timing[1, 160:162] = [-1, 1]
    timing[1, [164, 166]] = 1
    keys = np.array([[-1, 1, 5], [0, 0, 0], [1, 0, 8]])
    assert pack.verified_native_commits(timing, keys) == 2
    timing[1, 166] = 0
    with pytest.raises(ValueError, match="submitted keys"):
        pack.verified_native_commits(timing, keys)


def test_critical_teacher_weight_changes_only_iql_behavior_cloning(trainer) -> None:
    entry = importlib.import_module("train_hkrl_wm_recurrent_rl")
    catalogue = torch.tensor([[0, 0, 0], [1, 0, 4]])
    actor, values = entry.RecurrentActor(catalogue, 6), entry.RecurrentValues(catalogue, 6)
    x = torch.zeros(1, 5, 578)
    actions = torch.tensor([[0, 1, 0, 1]])
    teacher = torch.tensor([[True, True, True, False]])
    batch = x, actions, torch.zeros(1, 4), torch.ones(1, 4), torch.ones_like(teacher), teacher
    _, plain = entry.iql_losses(actor, values, values, batch, 0)
    _, weighted = entry.iql_losses(actor, values, values, batch, 0, torch.tensor([1.0, 2.0]))
    with torch.no_grad():
        logits = actor(x[:, :-1])[teacher]
        expected = torch.nn.functional.cross_entropy(
            logits, actions[teacher], weight=torch.tensor([1.0, 2.0])
        )
    assert weighted["teacher_bc"] == pytest.approx(float(expected))
    assert {k: weighted[k] for k in ("q", "v", "awr")} == {k: plain[k] for k in ("q", "v", "awr")}


def test_command_history_ablation_survives_training_and_reload(trainer) -> None:
    entry = importlib.import_module("train_hkrl_wm_recurrent_rl")
    catalogue = torch.tensor([[0, 0, 0], [1, 0, 4]])
    actor = entry.RecurrentActor(catalogue, 6)
    entry.remove_actor_command_history(actor)
    original = torch.rand(1, 9, 578)
    changed = original.clone()
    for offset in (0, entry.WIDTH, 2 * entry.WIDTH + 1):
        changed[..., offset + entry.CONTROL : offset + entry.WIDTH - 1] += 10
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.01)
    actor(original).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(actor(original), actor(changed), rtol=0, atol=0)

    restored = entry.RecurrentActor(catalogue, 6)
    restored.load_state_dict(actor.state_dict())
    torch.testing.assert_close(restored(original), restored(changed), rtol=0, atol=0)

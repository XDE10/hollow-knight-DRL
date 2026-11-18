"""Landing avoidance, simultaneous counterattacks and recurrent demonstration replay."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from hkrl import protocol, spaces
from hkrl.eval.sisters_survival import SistersCounterPolicy
from hkrl.models.heads import HybridPolicyHead
from hkrl.models.recurrent_policy import (
    SingleTaskEntityAttentionGRU,
    SistersLandingGRU,
    SistersTimingGRU,
    landing_movement_bias,
)
from hkrl.models.sisters_timing import platform_preferences
from hkrl.training.demonstration_replay import DemonstrationReplay


def _arena() -> dict[str, torch.Tensor]:
    entities = torch.zeros(1, 5, 24)
    entities[0, :4, 1] = torch.tensor([1, 3, 4, 4])
    entities[0, :2, 2] = protocol.Team.ENEMY
    entities[0, 2:4, 14] = torch.tensor([-12.0, 12.0]) / spaces.ARENA_SCALE
    entities[0, 2:4, 16:18] = torch.tensor([4.0, 1.0]) / spaces.ARENA_SCALE
    return {
        "player": torch.zeros(1, 32),
        "global": torch.zeros(1, 9),
        "entities": entities,
        "entity_mask": torch.tensor([[True, True, True, True, False]]),
    }


def test_landing_bias_brakes_on_descent_and_respects_padding_and_symmetry() -> None:
    state = _arena()
    state["entities"][0, -1] = state["entities"][0, 2]
    obs = {key: value.unsqueeze(1).repeat_interleave(5, dim=1) for key, value in state.items()}
    player = obs["player"]
    player[0, :, 0] = torch.tensor([0, 8, -8, 8, 8]) / spaces.ARENA_SCALE
    player[0, :, 1] = torch.tensor([0, 0, 0, 9, 9]) / spaces.ARENA_SCALE
    player[0, :, 2] = torch.tensor([0, 6, -6, 6, 6]) / spaces.VEL_SCALE
    player[0, -1, 3] = -22 / spaces.VEL_SCALE
    bias = landing_movement_bias(obs)[0]
    assert torch.equal(bias[0], torch.zeros(3)) and torch.equal(bias[3], bias[0])
    assert bias[1, 2] < bias[1, 0] < bias[1, 1]
    torch.testing.assert_close(bias[2], bias[1].flip(-1))
    assert bias[4, 2] < 0
    obs["entity_mask"][..., 3] = False
    assert torch.equal(landing_movement_bias(obs), torch.zeros_like(bias).unsqueeze(0))


def test_timing_gru_separates_blade_height_and_checks_all_visible_threats() -> None:
    state = _arena()
    obs = {key: value.unsqueeze(1).repeat_interleave(4, dim=1) for key, value in state.items()}
    obs["player"][..., spaces.PLAYER_FEATURE_INDEX["on_ground"]] = 1
    rows = obs["entities"]
    # Equal approach times, with the high return passing overhead in the first two frames.
    rows[:, :, 1, 10] = -15 / spaces.VEL_SCALE
    rows[:, :, 1, 14] = 3 / spaces.ARENA_SCALE
    rows[:, :2, 1, 15] = 4 / spaces.ARENA_SCALE
    rows[:, 1, 4] = rows[:, 1, 1]
    rows[:, 1, 4, 15] = 0
    obs["entity_mask"][:, 1, 4] = True
    # Grounded, target landed on the left, player already facing it.
    rows[:, 3, 0, 5] = -1523407744
    rows[:, 3, 0, 14] = -2.5 / spaces.ARENA_SCALE
    obs["player"][:, 3, spaces.PLAYER_FEATURE_INDEX["facing"]] = 1
    obs["entity_mask"][:, 3, 1] = False
    dims = {k: tuple(v.shape[1:]) for k, v in state.items()}
    model = SistersTimingGRU(dims, rnn_hidden=16, entity_hidden=16, attention_heads=2)
    for parameter in model.parameters():
        parameter.data.zero_()
    mask = torch.ones(1, 4, model.policy.mask_dim, dtype=torch.bool)
    jump = spaces.BUTTON_BITS["jump_tap"]
    mask[:, 2, spaces.N_MOVEMENT_X + spaces.N_AIM_Y + jump] = False
    dist, _, _ = model(obs, action_mask=mask)
    assert dist.buttons.probs[0, 0, jump] < 0.1
    assert dist.buttons.probs[0, 1, jump] > 0.9
    assert dist.buttons.probs[0, 2, jump] == 0
    assert dist.buttons.probs[0, 3, spaces.BUTTON_BITS["attack"]] > 0.9
    assert dist.movement_x.probs[0, 3].argmax() == 1


def test_platform_allows_safe_edge_stance_and_air_recovery_on_either_side() -> None:
    state = _arena()
    obs = {key: value.repeat_interleave(5, dim=0) for key, value in state.items()}
    p = obs["player"]
    p[:, 0] = torch.tensor([-8, -10.8, 10.8, 10.8, 10.8]) / spaces.ARENA_SCALE
    p[:, 1] = 3.1 / spaces.ARENA_SCALE
    p[1:, 3] = -8 / spaces.VEL_SCALE
    p[0, 9] = 1
    obs["entity_mask"][-1, 3] = False
    head = HybridPolicyHead(4)
    mask = torch.ones(5, head.mask_dim, dtype=torch.bool)
    jump = spaces.BUTTON_BITS["jump_tap"]
    mask[3, spaces.N_MOVEMENT_X + spaces.N_AIM_Y + jump] = False
    dist = head(torch.zeros(5, 4), action_mask=mask)
    after = platform_preferences(dist, obs)
    assert torch.allclose(after.movement_x.probs[0], dist.movement_x.probs[0])
    assert after.buttons.probs[0, jump] == dist.buttons.probs[0, jump]
    assert (after.buttons.probs[1:3, jump] > 0.99).all()
    assert after.buttons.probs[3, jump] == 0
    assert torch.allclose(after.movement_x.probs[-1], dist.movement_x.probs[-1])
    assert torch.equal(after.buttons.probs[-1], dist.buttons.probs[-1])


def test_landing_gru_keeps_all_buttons_and_disallowed_movements() -> None:
    state = _arena()
    state["player"][0, 0] = 11 / spaces.ARENA_SCALE
    dims = {k: tuple(v.shape[1:]) for k, v in state.items()}
    options = dict(entity_hidden=8, rnn_hidden=8, enable_macro=False)
    original = SingleTaskEntityAttentionGRU(dims, **options).eval()
    model = SistersLandingGRU(dims, **options).eval()
    model.load_state_dict(original.state_dict(), strict=True)
    mask = torch.ones(1, len(spaces.action_mask_layout(enable_macro=False)), dtype=torch.bool)
    mask[:, 0] = False  # even the favored recovery direction is unavailable
    old, old_value, _ = original(state, action_mask=mask)
    new, value, _ = model(state, action_mask=mask)
    assert new.movement_x.probs[0, 0] == 0
    assert new.movement_x.probs[0, 2] < old.movement_x.probs[0, 2]
    torch.testing.assert_close(new.buttons.probs, old.buttons.probs)
    torch.testing.assert_close(value, old_value)
    act, logp, _, _ = model.act(state, action_mask=mask, deterministic=True)
    logp_again, _, _ = model.evaluate_actions(state, act, action_mask=mask)
    torch.testing.assert_close(logp, logp_again)


@pytest.mark.parametrize("side", [-1, 1])
def test_counter_teacher_can_jump_and_hit_while_braking(side: int) -> None:
    state = {k: v[0].numpy() for k, v in _arena().items()}
    player, entities = state["player"], state["entities"]
    player[0], player[2], player[9] = side * 8 / spaces.ARENA_SCALE, side * 6 / spaces.VEL_SCALE, 1
    entities[0, 8] = side * 4 / spaces.ARENA_SCALE  # attackable target past the ledge
    entities[1, 14] = side * 12 / spaces.ARENA_SCALE
    entities[1, 10] = -side * 7 / spaces.VEL_SCALE
    policy = SistersCounterPolicy(spaces.make_action_space(enable_macro=False))
    mask = np.ones(len(spaces.action_mask_layout(enable_macro=False)), dtype=bool)
    act = policy.act(state, mask)
    assert act["movement_x"] == 1
    assert (
        act["buttons"][spaces.BUTTON_BITS["attack"]]
        and act["buttons"][spaces.BUTTON_BITS["jump_tap"]]
    )
    state["entity_mask"][0] = False
    mask[spaces.N_MOVEMENT_X + spaces.N_AIM_Y + spaces.BUTTON_BITS["jump_tap"]] = False
    act = policy.act(state, mask)
    assert not act["buttons"][spaces.BUTTON_BITS["attack"]]
    assert not act["buttons"][spaces.BUTTON_BITS["jump_tap"]]


def test_demo_replay_excludes_held_out_episodes_and_trains_on_short_windows(tmp_path: Path) -> None:
    obs = _arena()
    length = 3
    mask = spaces.action_mask_layout(enable_macro=False)
    for num in range(1, 6):
        if num == 4:
            (tmp_path / "episode-004.npz").write_bytes(b"held-out")
            continue
        actions = np.tile(spaces.canonical_noop_action_values(enable_macro=False), (length, 1))
        actions[:, 0] = 2
        actions[:, 2 + spaces.BUTTON_BITS["attack"]] = 1
        np.savez(
            tmp_path / f"episode-{num:03d}.npz",
            **{k: v.numpy().repeat(length, axis=0) for k, v in obs.items()},
            actions=actions,
            masks=np.ones((length, len(mask)), dtype=bool),
            rewards=np.full(length, num, dtype=np.float32),
        )
    replay = DemonstrationReplay(tmp_path, length=2, burn_in=2)
    assert len(replay.windows) == 8 and "episode-004.npz" not in replay.hashes
    assert (
        replay.hashes["episode-001.npz"]
        == hashlib.sha256((tmp_path / "episode-001.npz").read_bytes()).hexdigest()
    )
    for number, episode in zip([1, 2, 3, 5], replay.episodes, strict=True):
        torch.testing.assert_close(
            episode["prev_action"][0],
            torch.tensor(spaces.canonical_noop_action_values(enable_macro=False)).float(),
        )
        torch.testing.assert_close(episode["prev_action"][1:], episode["actions"][:-1].float())
        assert episode["prev_reward"].flatten().tolist() == [0, number, number]
    model = SistersLandingGRU(
        {k: tuple(v.shape[1:]) for k, v in obs.items()},
        entity_hidden=8,
        rnn_hidden=8,
        enable_macro=False,
    )
    loss = replay.loss(model, np.random.default_rng(42), torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.policy.movement_x.weight.grad.abs().sum() > 0
    assert model.policy.buttons.weight.grad[spaces.BUTTON_BITS["attack"]].abs().sum() > 0

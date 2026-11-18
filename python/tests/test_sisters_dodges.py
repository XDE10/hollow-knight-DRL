from __future__ import annotations

import numpy as np
import pytest
import torch
from hkrl import spaces
from hkrl.eval.sisters_survival import SistersDodgeCounterPolicy
from hkrl.models.heads import HybridPolicyHead
from hkrl.training.demonstration_replay import dodge_imitation_loss
from hkrl.training.sisters_dodges import dodge_outcomes


def test_dodge_labels_align_actions_with_next_observation_and_real_injuries() -> None:
    n = 30
    actions = np.zeros((n, 13), np.int64)
    player = np.zeros((n, 32), np.float32)
    player[:, 9] = 1
    data = dict(actions=actions, player=player, elapsed_ticks=np.full(n, 4))
    actions[[0, 2, 8, 15, 29], 2] = 1
    actions[23, 4] = 1
    player[[1, 2, 9, 16], 11] = 1  # new jump, repeated input, unsafe jump, safe jump
    player[24, 13] = 1
    record = dict(steps=n, hits=[{"step": 4}], contacts=[{"step": 10, "amount": 1}])
    grade = dodge_outcomes(data, record)
    assert grade[[0, 2, 8, 15, 29], 0].tolist() == [2, -1, -1, 1, -1]
    assert grade[23, 1] == 1
    player[23, 15] = 1
    assert dodge_outcomes(data, record)[23, 1] == -1


@pytest.mark.parametrize("side", [-1, 1])
def test_teacher_evades_downward_attack_into_free_space_and_never_dashes_into_spikes(
    side: int,
) -> None:
    player = np.zeros(32, np.float32)
    player[[0, 9]] = [9 * side / spaces.ARENA_SCALE, 1]
    rows = np.zeros((4, 24), np.float32)
    rows[:, 1] = [1, 4, 4, 3]
    rows[0, 8:10] = [0, 0.3]
    rows[0, 14:16] = [(9 * side - side) / spaces.ARENA_SCALE, 0.3]
    rows[1:3, 14] = np.array([-13, 13]) / spaces.ARENA_SCALE
    rows[1:3, 16] = 4 / spaces.ARENA_SCALE
    obs = dict(player=player, entities=rows, entity_mask=np.array([1, 1, 1, 0], dtype=bool))
    mask = np.ones(len(spaces.action_mask_layout(enable_macro=False)), dtype=bool)
    policy = SistersDodgeCounterPolicy(spaces.make_action_space(enable_macro=False))
    act = policy.act(obs, mask)
    assert act["movement_x"] == 1 - side
    assert act["buttons"][spaces.BUTTON_BITS["dash"]] == 1
    mask[spaces.N_MOVEMENT_X + spaces.N_AIM_Y + spaces.BUTTON_BITS["dash"]] = False
    assert policy.act(obs, mask)["buttons"][spaces.BUTTON_BITS["dash"]] == 0


def test_dodge_replay_ignores_failed_positive_and_padding() -> None:
    head = HybridPolicyHead(4, enable_macro=False)
    x = torch.zeros(1, 4, 4)
    mask = torch.ones(1, 4, len(spaces.action_mask_layout(enable_macro=False)), dtype=torch.bool)
    actions = torch.tensor(spaces.canonical_noop_action_values(enable_macro=False)).repeat(1, 4, 1)
    actions[0, :3, 2] = 1
    grades = torch.tensor([[[2, 0], [-1, 0], [1, 0], [0, 0]]])
    valid = torch.tensor([[True, True, False, True]])
    dist = head(x, action_mask=mask)
    dist.buttons.logits.retain_grad()
    loss = dodge_imitation_loss(dist, actions, mask, valid, grades)
    loss.backward()
    gradients = dist.buttons.logits.grad[0, :, spaces.BUTTON_BITS["jump_tap"]]
    assert gradients[0] < 0 and gradients[3] > 0
    assert gradients[1] == gradients[2] == 0


def test_only_last_request_gets_credit_for_observed_jump() -> None:
    actions = np.zeros((20, 13), np.int64)
    actions[[0, 1], 2] = 1
    player = np.zeros((20, 32), np.float32)
    player[:, 9] = 1
    player[2, 11] = 1
    data = dict(actions=actions, player=player, elapsed_ticks=np.full(20, 3))
    record = dict(steps=20, contacts=[], hits=[])
    assert dodge_outcomes(data, record)[:2, 0].tolist() == [-1, 1]


def test_lunge_telegraph_on_second_boss_can_trigger_masked_jump() -> None:
    rows = np.zeros((2, 24), dtype=np.float32)
    rows[:, 1:3] = 1
    rows[:, 14] = np.array([21, -9]) / spaces.ARENA_SCALE
    rows[:, 8] = rows[:, 14]
    rows[1, 5] = SistersDodgeCounterPolicy.DASH_ANTIC
    player = np.zeros(32, np.float32)
    player[9] = 1
    obs = dict(player=player, entities=rows, entity_mask=np.array([True, True]))
    mask = np.ones(len(spaces.action_mask_layout(enable_macro=False)), dtype=bool)
    policy = SistersDodgeCounterPolicy(spaces.make_action_space(enable_macro=False))
    bit = spaces.BUTTON_BITS["jump_tap"]
    assert policy.act(obs, mask)["buttons"][bit]
    obs["entity_mask"][1] = False
    assert not policy.act(obs, mask)["buttons"][bit]
    obs["entity_mask"][1] = True
    mask[spaces.N_MOVEMENT_X + spaces.N_AIM_Y + bit] = False
    assert not policy.act(obs, mask)["buttons"][bit]


@pytest.mark.parametrize("side", [-1, 1])
def test_body_counter_stops_in_nail_range_and_holds_lunge_jump(side: int) -> None:
    from hkrl.eval.sisters_survival import SistersBodyCounterPolicy

    player = np.zeros(32, np.float32)
    player[9] = 1
    rows = np.zeros((1, 24), np.float32)
    rows[0, 1] = 1
    rows[0, 5] = SistersBodyCounterPolicy.LANDED
    rows[0, [8, 14]] = side * 2.4 / spaces.ARENA_SCALE
    obs = dict(player=player, entities=rows, entity_mask=np.array([True]))
    mask = np.ones(len(spaces.action_mask_layout(enable_macro=False)), dtype=bool)
    policy = SistersBodyCounterPolicy(spaces.make_action_space(enable_macro=False))
    act = policy.act(obs, mask)
    assert act["movement_x"] == 1
    assert act["buttons"][spaces.BUTTON_BITS["attack"]]
    rows[0, 5] = policy.DASH_ANTIC
    act = policy.act(obs, mask)
    assert act["buttons"][spaces.BUTTON_BITS["jump_hold"]] and act["duration"] == 3


def test_safe_nail_requires_a_complete_injury_free_window() -> None:
    from hkrl.training.sisters_dodges import safe_nail_windows

    data = dict(player=np.zeros((20, 32)), elapsed_ticks=np.full(20, 4))
    record = dict(hits=[dict(slash_step=7, step=8)], contacts=[])
    assert np.flatnonzero(safe_nail_windows(data, record)).tolist() == [3, 4, 5, 6]
    record["contacts"] = [dict(step=11, amount=1)]
    assert not safe_nail_windows(data, record).any()


def test_landed_counter_requires_visible_hit_target_and_safe_grounded_side_slash() -> None:
    from hkrl.training.sisters_dodges import landed_counter_labels

    n = 32
    actions = np.zeros((n, 13), dtype=int)
    actions[:, 1] = 1
    actions[:, 2 + spaces.BUTTON_BITS["attack"]] = 1
    player = np.zeros((n, 32))
    player[:, spaces.PLAYER_FEATURE_INDEX["on_ground"]] = 1
    entities = np.zeros((n, 2, 24))
    entities[:, :, 0] = [17, 42]
    entities[:, :, 1] = 1
    entities[:, :, 5] = 123
    entities[:, :, 8] = [-0.1, 0.1]
    data = dict(
        actions=actions,
        player=player,
        elapsed_ticks=np.full(n, 4),
        entities=entities,
        entity_mask=np.ones((n, 2), dtype=bool),
    )
    hits = [
        dict(slash_step=step, step=step + 1, target_id=target, dx_at_slash=side * 3.0)
        for step, target, side in ((7, 17, -1), (14, 42, 1), (21, 42, 1), (27, 17, -1))
    ]
    record = dict(hits=hits, contacts=[])
    # Another landed sister is visible during each of these two nonmatching hits.
    entities[20, 1, 5] = 111
    data["entity_mask"][26, 0] = False
    window, sides = landed_counter_labels(data, record, 123)
    assert np.flatnonzero(window).tolist() == [3, 4, 5, 6, 10, 11, 12, 13]
    assert np.flatnonzero(sides).tolist() == [6, 13]
    assert sides[[6, 13]].tolist() == [-1, 1]
    player[6, spaces.PLAYER_FEATURE_INDEX["on_ground"]] = 0
    record["contacts"] = [dict(step=16, amount=1)]
    assert not landed_counter_labels(data, record, 123)[1].any()

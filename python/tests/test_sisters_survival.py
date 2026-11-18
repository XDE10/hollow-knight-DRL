from __future__ import annotations

import numpy as np
import pytest
from hkrl import protocol, spaces
from hkrl.eval.evaluator import _metrics_from_info
from hkrl.eval.sisters_survival import SistersSurvivalPolicy


def _decision(
    bosses: list[tuple[float, float]],
    dangers: list[tuple[int, float, float]],
    *,
    padded: bool = False,
    jump_allowed: bool = True,
) -> dict:
    player = np.zeros(32, np.float32)
    player[9] = 1
    rows = []
    for kind, x, y in [(1, *xy) for xy in bosses] + dangers:
        row = np.zeros(24, np.float32)
        row[:3] = [len(rows) + 1, kind, 0 if kind == 4 else 1]
        row[[6, 8, 14]] = x / spaces.ARENA_SCALE
        row[[7, 9, 15]] = y / spaces.ARENA_SCALE
        row[16:18] = np.array([2, 1]) / spaces.ARENA_SCALE
        rows.append(row)
    obs = dict(player=player, entities=np.array(rows), entity_mask=np.ones(len(rows), dtype=bool))
    if padded:
        obs["entity_mask"][-1] = False
    mask = np.ones(len(spaces.action_mask_layout(enable_macro=False)), dtype=bool)
    mask[spaces.N_MOVEMENT_X + spaces.N_AIM_Y + spaces.BUTTON_BITS["jump_tap"]] = jump_allowed
    policy = SistersSurvivalPolicy(spaces.make_action_space(enable_macro=False))
    return policy.act(obs, mask)


def test_many_entities_survival_uses_second_projectile_and_respects_padding() -> None:
    bosses = [(7, 0), (15, 0), (20, 0)]
    harmless = [(3, -12, 0)]
    danger = [*harmless, (3, 2.5, 0)]
    straight = _decision(bosses, harmless)
    dodge = _decision(bosses, danger)
    assert straight["movement_x"] == 2
    assert (dodge["movement_x"], dodge["buttons"][0]) != (2, 0)
    masked = _decision(bosses, danger, padded=True)
    assert masked["movement_x"] == straight["movement_x"] and masked["buttons"][0] == 0
    blocked = _decision(bosses, danger, jump_allowed=False)
    assert blocked["buttons"][0] == 0


@pytest.mark.parametrize("side", [-1, 1])
def test_edge_allows_a_safe_jump_in_either_direction(side: int) -> None:
    action = _decision([(10 * side, 0)], [(4, 3 * side, -1.5)])
    assert action["movement_x"] == 1 + side
    assert action["buttons"][spaces.BUTTON_BITS["jump_tap"]] == 1


def test_contact_metrics_keep_simultaneous_instances_and_unknown_separate() -> None:
    ids = [12, 17, 35, 44, 53, 0]
    sources = [1, 1, 2, 2, 3, 0]
    events = [
        protocol.RewardEvent(protocol.RewardEventKind.DAMAGE_TAKEN, i, 1, aux_int2=k)
        for i, k in zip(ids, sources, strict=True)
    ]
    metrics = _metrics_from_info({"reward_events": events})
    assert metrics.damage_taken == 6
    assert [c[1] for c in metrics.contacts] == ids
    assert [c[0] for c in metrics.contacts] == [
        "boss_body",
        "boss_body",
        "enemy_projectile",
        "enemy_projectile",
        "spikes",
        "unknown",
    ]

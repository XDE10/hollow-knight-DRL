"""Soft action preferences for countering visible Sisters attacks in the privileged task."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical

from hkrl import protocol, spaces
from hkrl.models.heads import CompositeActionDistribution


def counter_preferences(
    dist: CompositeActionDistribution, obs: dict[str, Tensor]
) -> CompositeActionDistribution:
    """Bias only actions whose cues are present; the GRU still sets all base logits."""
    p, e = spaces.PLAYER_FEATURE_INDEX, spaces.ENTITY_FEATURE_INDEX
    player, entities = obs["player"], obs["entities"]
    if entities.shape[-1] <= e["hurtbox_size_y"] or not entities.shape[-2]:
        return dist
    active = obs["entity_mask"].bool()
    kind = entities[..., e["entity_type"]]
    body = active & (kind == protocol.EntityType.BOSS)
    blade = (
        active
        & (kind == protocol.EntityType.PROJECTILE)
        & (entities[..., e["team"]] == protocol.Team.ENEMY)
    )
    x, y = (player[..., p[name]] * spaces.ARENA_SCALE for name in ("pos_x", "pos_y"))
    dx = entities[..., e["hurtbox_center_x"]] * spaces.ARENA_SCALE - x.unsqueeze(-1)
    dy = entities[..., e["hurtbox_center_y"]] * spaces.ARENA_SCALE - y.unsqueeze(-1)
    vx = entities[..., e["vel_x"]] * spaces.VEL_SCALE
    vy = entities[..., e["vel_y"]] * spaces.VEL_SCALE
    state = entities[..., e["fsm_state_hash"]]
    grounded = player[..., p["on_ground"]].bool()
    coming = dx * vx < 0
    # Account for travel time and blade height at arrival. A high return can pass overhead.
    travel = dx.abs() / vx.abs().clamp_min(1)
    crossing = blade & coming & (travel < 0.38) & (vx.abs() > 1)
    crossing_y = dy + vy * travel
    low = (crossing & (crossing_y.abs() < 2.0)).any(-1)
    high = (crossing & (crossing_y > 2.8)).any(-1)
    # The slot-3 departure and landing states are measured by the game collector.
    lunge = (
        body
        & (dy.abs() < 2.4)
        & (((state == -737267072) & (dx.abs() < 15)) | (coming & (vx.abs() > 20) & (travel < 0.36)))
    )
    rise = (low | lunge.any(-1)) & grounded
    above = body & (dy > 2.5) & (dy < 13) & (dx.abs() < 2.8) & (vy < 1)
    # Near an edge, choose the open side when the drop is directly overhead.
    spikes = active & (kind == protocol.EntityType.HAZARD)
    centers = entities[..., e["hurtbox_center_x"]] * spaces.ARENA_SCALE
    mid = (centers * spikes).sum(-1) / spikes.sum(-1).clamp_min(1)
    away = torch.where(dx.abs() < 0.7, torch.sign(mid - x).unsqueeze(-1), -dx.sign())
    evade = (away * above).sum(-1).sign()
    landed = body & (state == -1523407744) & (dy.abs() < 1.8)
    target = dx.abs().masked_fill(~landed, torch.inf).argmin(-1, keepdim=True)
    offset = dx.gather(-1, target).squeeze(-1)
    can_counter = landed.any(-1) & grounded & ~rise & ~above.any(-1)
    reach = can_counter & (offset.abs() <= 3.5) & (offset.abs() >= 1.55)
    # Slot-3 Hero sprites use negative X scale while looking to the right.
    facing = -player[..., p["facing"]].sign()
    turn = reach & (facing != offset.sign())
    approach = can_counter & (offset.abs() > 3.5)
    back = can_counter & (offset.abs() < 1.55)
    direction = (turn | approach).float() * offset.sign() - back * offset.sign()
    direction = torch.where(above.any(-1), evade, direction)
    move_bias = torch.stack((-4 * direction, 3 * (reach & ~turn).float(), 4 * direction), dim=-1)
    # A body under the airborne Knight is a downward nail opportunity. Overlapping
    # threats can instead justify a downward spell when the game permits casting.
    underneath = (body & (dx.abs() < 3.5) & (dy < -0.7) & (dy > -5)).any(-1) & ~grounded
    crowd = ((body | blade) & (dx.abs() < 5.5) & (dy.abs() < 6)).sum(-1) >= 2
    dive = ~grounded & crowd & player[..., p["can_cast"]].bool()
    downward = underneath | dive
    aim_bias = torch.stack((5 * downward.float(), 4 * reach.float(), -3 * downward.float()), dim=-1)
    buttons = torch.zeros_like(dist.buttons.logits)
    bits = spaces.BUTTON_BITS
    buttons[..., bits["jump_tap"]] = 8 * rise.float() - 4 * (grounded & high & ~rise)
    holding = (player[..., p["vel_y"]] > 0) & (low | lunge.any(-1))
    buttons[..., bits["jump_hold"]] = 6 * (rise | holding).float()
    buttons[..., bits["attack"]] = 6 * (reach | underneath).float() - 4 * dive
    buttons[..., bits["cast"]] = 7 * dive.float()
    # Use a short hold for takeoff and release control promptly in the landing window.
    duration = torch.zeros_like(dist.duration.logits)
    duration[..., -1] = 4 * rise.float() - 3 * (can_counter | above.any(-1))
    macro = dist.macro
    if macro is not None:
        bias = torch.zeros_like(macro.logits)
        bias[..., 0] = 6 * (rise | reach | above.any(-1) | downward).float()
        macro = Categorical(logits=macro.logits + bias)
    return replace(
        dist,
        movement_x=Categorical(logits=dist.movement_x.logits + move_bias),
        aim_y=Categorical(logits=dist.aim_y.logits + aim_bias),
        buttons=Bernoulli(logits=dist.buttons.logits + buttons),
        duration=Categorical(logits=dist.duration.logits + duration),
        macro=macro,
    )

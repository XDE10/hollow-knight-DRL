"""Soft action preferences for countering visible Sisters attacks in the privileged task."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical

from hkrl import protocol, spaces
from hkrl.models.heads import CompositeActionDistribution


def platform_preferences(
    dist: CompositeActionDistribution, obs: dict[str, Tensor]
) -> CompositeActionDistribution:
    """Compare the three landing positions and preserve safe actions near either edge."""
    p, e = spaces.PLAYER_FEATURE_INDEX, spaces.ENTITY_FEATURE_INDEX
    player, entities = obs["player"], obs["entities"]
    if entities.shape[-1] <= e["hurtbox_size_y"] or entities.shape[-2] < 2:
        return dist
    visible = obs["entity_mask"].bool()
    kind = entities[..., e["entity_type"]]
    cx, cy, width, height = (
        entities[..., e[name]] * spaces.ARENA_SCALE
        for name in ("hurtbox_center_x", "hurtbox_center_y", "hurtbox_size_x", "hurtbox_size_y")
    )
    spikes = visible & (kind == protocol.EntityType.HAZARD) & (width > 0) & (height > 0)
    left = (cx + width / 2).masked_fill(~spikes, torch.inf).amin(-1)
    right = (cx - width / 2).masked_fill(~spikes, -torch.inf).amax(-1)
    paired = (spikes.sum(-1) == 2) & (left < right)
    # Finite fallbacks also keep masked rows safe during a minibatch backpropagation.
    left, right = (torch.where(paired, edge, 0) for edge in (left, right))
    top = (cy + height / 2).masked_fill(~spikes, -torch.inf).amax(-1)
    floor = torch.where(paired, top + 1.73, 0)
    x, y = (player[..., p[name]] * spaces.ARENA_SCALE for name in ("pos_x", "pos_y"))
    vx, vy = (player[..., p[name]] * spaces.VEL_SCALE for name in ("vel_x", "vel_y"))
    grounded = player[..., p["on_ground"]].bool()
    # Descent acceleration and walk speed measured in live slot-3 rollouts.
    gravity, walk_speed = 47.4, 8.3
    flight = (vy + (vy.square() + 2 * gravity * (y - floor).clamp_min(0)).sqrt()) / gravity
    horizon = torch.where(grounded, 0.17, flight.clamp(0.10, 0.7))
    lag = horizon.clamp(max=0.10)
    coast = x + lag * vx
    reach = (horizon - lag) * walk_speed
    landings = torch.stack((coast - reach, coast, coast + reach), -1)
    clear = torch.minimum(landings - left.unsqueeze(-1), right.unsqueeze(-1) - landings)
    gap = torch.minimum(x - left, right - x)
    edge_side = torch.sign(x - (right + left) / 2)
    dx, dy = cx - x.unsqueeze(-1), cy - y.unsqueeze(-1)
    enemy = (kind == protocol.EntityType.BOSS) | (
        (kind == protocol.EntityType.PROJECTILE) & (entities[..., e["team"]] == protocol.Team.ENEMY)
    )
    inboard = dx * edge_side.unsqueeze(-1) < 0
    close = (dx.abs() < width / 2 + 2.6) & (dy.abs() < height / 2 + 1.3)
    exposed = (visible & enemy & (width > 0) & inboard & close).any(-1)
    # Leave knockback room only while an inboard enemy can hit the Knight.
    margin = 0.8 + 2.4 * exposed.float()
    risk = ((margin.unsqueeze(-1) - clear) / 1.2).clamp(0, 3.5)
    risk = risk * paired.unsqueeze(-1)
    move = Categorical(logits=dist.movement_x.logits - 5 * risk)
    inward_landing = torch.where(edge_side < 0, clear[..., 2], clear[..., 0])
    recovery = paired & ~grounded & (vy < 1) & (gap < 0.9) & (inward_landing < 1.1)
    buttons = torch.zeros_like(dist.buttons.logits)
    buttons[..., spaces.BUTTON_BITS["jump_tap"]] = 8 * recovery.float()
    buttons[..., spaces.BUTTON_BITS["jump_hold"]] = 4 * recovery.float()
    macro = dist.macro
    if macro is not None:
        bias = torch.zeros_like(macro.logits)
        bias[..., 0] = 5 * (recovery | (risk.amax(-1) > 1)).float()
        macro = Categorical(logits=macro.logits + bias)
    return replace(
        dist, movement_x=move, buttons=Bernoulli(logits=dist.buttons.logits + buttons), macro=macro
    )


def counter_preferences(
    dist: CompositeActionDistribution,
    obs: dict[str, Tensor],
    *,
    blade_interception: bool = False,
    sustain_charge_jump: bool = False,
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
    if blade_interception:
        # The leading edge arrives before the centre. Measured horizontal acceleration
        # can reach 24 units/s² on the return; evaluate its earliest arrival.
        half_width = entities[..., e["hurtbox_size_x"]] * (spaces.ARENA_SCALE / 2)
        gap = (dx.abs() - half_width - 0.56).clamp_min(0)
        earliest = 2 * gap / (vx.abs() + (vx.square() + 48 * gap).sqrt()).clamp_min(1)
        low |= (
            blade
            & coming
            & (vx.abs() > 1)
            & (half_width > 0)
            & (earliest < 0.36)
            & ((dy + vy * earliest).abs() < 1.85)
        ).any(-1)
        # In live slot-3 trajectories the low thrower crouches close to the floor.
        # Her release is visible even when the pooled disc has not yet been observed.
        low_thrower = body & (half_width > 0) & (dy > 0) & (dy < 2.25)
        release = low_thrower & (state == 739257600) & (dx.abs() < 12)
        low |= release.any(-1) & grounded
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
    if blade_interception:
        # Give a nearby low windup room to resolve before committing to a jump.
        windup = low_thrower & (state == -1403342080) & (dx.abs() < 7)
        away_from_throw = -(windup * dx.sign()).sum(-1).sign()
        direction = torch.where(grounded & (away_from_throw != 0), away_from_throw, direction)
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
    primitive = rise | reach | above.any(-1) | downward
    if sustain_charge_jump and player.shape[-1] > p["applied_input_buttons"]:
        # Once the Knight clears the lance, the original low-height takeoff cue
        # disappears. Keep the same jump pressed until the crossing is clear.
        charging = body & ((state == -737267072) | ((state == -1374908928) & coming))
        below = (charging & (dx.abs() < 14) & (dy > -7) & (dy < 2.4)).any(-1)
        jump_bits = (1 << bits["jump_tap"]) | (1 << bits["jump_hold"])
        key_down = (player[..., p["applied_input_buttons"]].long() & jump_bits) != 0
        high_disc = (blade & (dx.abs() < 6) & (dy > 1.4) & (dy < 7)).any(-1)
        continue_jump = (
            below
            & ~grounded
            & player[..., p["jumping"]].bool()
            & (player[..., p["vel_y"]] > 0.05)
            & key_down
            & (dist.buttons.probs[..., bits["jump_hold"]] > 0)
            & ~player[..., p["invulnerable"]].bool()
            & ~above.any(-1)
            & ~high_disc
        )
        buttons[..., bits["jump_hold"]] += 7 * continue_jump.float()
        primitive |= continue_jump
    buttons[..., bits["attack"]] = 6 * (reach | underneath).float() - 4 * dive
    buttons[..., bits["cast"]] = 7 * dive.float()
    # Use a short hold for takeoff and release control promptly in the landing window.
    duration = torch.zeros_like(dist.duration.logits)
    duration[..., -1] = 4 * rise.float() - 3 * (can_counter | above.any(-1))
    macro = dist.macro
    if macro is not None:
        bias = torch.zeros_like(macro.logits)
        bias[..., 0] = 6 * primitive.float()
        macro = Categorical(logits=macro.logits + bias)
    return replace(
        dist,
        movement_x=Categorical(logits=dist.movement_x.logits + move_bias),
        aim_y=Categorical(logits=dist.aim_y.logits + aim_bias),
        buttons=Bernoulli(logits=dist.buttons.logits + buttons),
        duration=Categorical(logits=dist.duration.logits + duration),
        macro=macro,
    )

"""A slot-3 demonstration teacher that checks every visible threat before approaching."""

from __future__ import annotations

from typing import Any

import numpy as np

from hkrl import protocol, spaces
from hkrl.eval.scripted_policies import ScriptedAggroPolicy, _validated_action_mask


class SistersSurvivalPolicy(ScriptedAggroPolicy):
    """Rank legal short walk/stop/jump trajectories by predicted clearance."""

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        action = super().act(obs, action_mask)
        mask, _, _ = _validated_action_mask(self.action_space, action_mask)
        player = obs["player"]
        visible = obs["entities"][np.asarray(obs["entity_mask"], dtype=bool)]
        threats = visible[np.isin(visible[:, 1], (1, 3, 4)) & (visible[:, 2] != 2)]
        bosses = visible[visible[:, 1] == 1]
        if not len(bosses):
            action["buttons"][spaces.BUTTON_BITS["attack"]] = 0
            action["movement_x"] = int(np.flatnonzero(mask[:3])[0]) if not mask[1] else 1
        if not len(threats):
            return action

        e = spaces.ENTITY_FEATURE_INDEX
        p = spaces.PLAYER_FEATURE_INDEX
        pos = player[:2] * spaces.ARENA_SCALE
        grounded = bool(player[p["on_ground"]])
        horizon = np.array([0.12, 0.26, 0.42])[:, None]
        size = threats[:, e["hurtbox_size_x"] : e["hurtbox_size_y"] + 1] * spaces.ARENA_SCALE
        centers = threats[:, 14:16] * spaces.ARENA_SCALE
        centers = np.where(size.any(axis=1)[:, None], centers, threats[:, 6:8] * spaces.ARENA_SCALE)
        velocity = threats[:, 10:12] * spaces.VEL_SCALE
        hazard = threats[:, e["entity_type"]] == 4
        future_x = centers[:, 0] + horizon * velocity[:, 0]
        future_y = centers[:, 1] + horizon * velocity[:, 1]
        # Spikes sit below the platform. Include the drop into the pit when forecasting
        # a grounded walk. Characters already airborne can safely clear its rectangle.
        margin_x = np.where(hazard, 0.55, 1.25)
        margin_y = np.where(hazard, 2.1, 1.5)
        jump = spaces.BUTTON_BITS["jump_tap"]
        can_jump = mask[spaces.N_MOVEMENT_X + spaces.N_AIM_Y + jump] and (
            grounded or player[p["double_jump_available"]]
        )
        preferred = int(action["movement_x"])
        jump_speeds: tuple[bool, ...]
        if not grounded and player[p["vel_y"]] > 0:
            # Carry the existing arc across successive decisions.
            jump_speeds = (False,)
        else:
            jump_speeds = (False, True) if can_jump else (False,)
        candidates = []
        for move in np.flatnonzero(mask[:3]):
            x = pos[0] + (int(move) - 1) * 8.3 * horizon
            dx = np.maximum(0.0, np.abs(x - future_x) - size[:, 0] / 2)
            for leap in jump_speeds:
                vy = 16.0 if leap else float(player[p["vel_y"]] * spaces.VEL_SCALE)
                y = pos[1] + vy * horizon - 22.0 * horizon**2
                if grounded and not leap:
                    y = np.full_like(horizon, pos[1])
                dy = np.maximum(0.0, np.abs(y - future_y) - size[:, 1] / 2)
                overlap = np.maximum(0, 1 - dx / margin_x) * np.maximum(0, 1 - dy / margin_y)
                costs = overlap * np.where(hazard, 2.0, 1.0)
                risk = float(costs.max() + 0.25 * costs.sum(axis=1).max())
                preference = 0.04 * (move != preferred) + 0.025 * leap
                candidates.append((risk + preference, risk, int(move), leap))
        _, risk, move, leap = min(candidates)
        action["movement_x"] = move
        action["buttons"][jump] = int(leap)
        if risk > 0.25 or (preferred != 1 and move != preferred):
            action["buttons"][spaces.BUTTON_BITS["attack"]] = 0
        return action


class SistersCounterPolicy(ScriptedAggroPolicy):
    """Demonstrate nail attacks while leaving room to brake or land beside spikes."""

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        action = super().act(obs, action_mask)
        mask, _, _ = _validated_action_mask(self.action_space, action_mask)
        visible = obs["entities"][np.asarray(obs["entity_mask"], dtype=bool)]
        e = spaces.ENTITY_FEATURE_INDEX
        p = spaces.PLAYER_FEATURE_INDEX
        kinds = visible[:, e["entity_type"]]
        if not np.any(kinds == protocol.EntityType.BOSS):
            action["buttons"][spaces.BUTTON_BITS["attack"]] = 0
        if visible.shape[-1] <= e["hurtbox_size_y"]:
            return action
        player = obs["player"]
        pos = player[:2] * spaces.ARENA_SCALE
        movement = int(action["movement_x"])
        blades = visible[
            (kinds == protocol.EntityType.PROJECTILE)
            & (visible[:, e["team"]] == protocol.Team.ENEMY)
        ]
        if len(blades):
            relative = (
                blades[:, e["hurtbox_center_x"] : e["hurtbox_center_y"] + 1] * spaces.ARENA_SCALE
                - pos
            )
            speeds = blades[:, e["vel_x"]] * spaces.VEL_SCALE
            approaching = (relative[:, 0] * speeds < 0) & (abs(relative[:, 0]) < 7)
            if np.any(approaching & (abs(relative[:, 1]) < 1.25)):
                jump = spaces.BUTTON_BITS["jump_tap"]
                action["buttons"][jump] = int(
                    player[p["on_ground"]] and mask[spaces.N_MOVEMENT_X + spaces.N_AIM_Y + jump]
                )
        spikes = visible[kinds == protocol.EntityType.HAZARD]
        if len(spikes) == 2:
            centers = spikes[:, e["hurtbox_center_x"]] * spaces.ARENA_SCALE
            half = spikes[:, e["hurtbox_size_x"]] * spaces.ARENA_SCALE * 0.5
            left = int(np.argmin(centers))
            right = 1 - left
            lo, hi = centers[left] + half[left], centers[right] - half[right]
            # Allow lateral braking and jumping to occur in the same demonstration frame.
            horizon = 0.20 if player[p["on_ground"]] else 0.30
            projected = pos[0] + player[p["vel_x"]] * spaces.VEL_SCALE * horizon
            if lo < hi:
                if pos[0] < lo + 1.0 or pos[0] > hi - 1.0:
                    movement = 2 if pos[0] < (lo + hi) / 2 else 0
                elif (movement == 0 and projected < lo + 2.4) or (
                    movement == 2 and projected > hi - 2.4
                ):
                    movement = 1
        if mask[movement]:
            action["movement_x"] = movement
        return action


class SistersDodgeCounterPolicy(SistersCounterPolicy):
    """Demonstrate a jump over low attacks or a lateral escape from a descending sister."""

    # The float32 observation carries the mod's stable hash for the FSM state "Dash Antic".
    DASH_ANTIC = np.float32(-737267062)

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        action = super().act(obs, action_mask)
        mask, _, _ = _validated_action_mask(self.action_space, action_mask)
        active = obs["entities"][np.asarray(obs["entity_mask"], dtype=bool)]
        e, p = spaces.ENTITY_FEATURE_INDEX, spaces.PLAYER_FEATURE_INDEX
        if active.shape[1] <= e["hurtbox_size_y"]:
            return action
        player = obs["player"]
        x, y = player[:2] * spaces.ARENA_SCALE
        grounded = bool(player[p["on_ground"]])
        kind = active[:, e["entity_type"]]
        enemies = (kind == protocol.EntityType.BOSS) | (
            (kind == protocol.EntityType.PROJECTILE) & (active[:, e["team"]] == protocol.Team.ENEMY)
        )
        threats = active[enemies]
        if not len(threats):
            return action
        dx = threats[:, e["hurtbox_center_x"]] * spaces.ARENA_SCALE - x
        dy = threats[:, e["hurtbox_center_y"]] * spaces.ARENA_SCALE - y
        vx = threats[:, e["vel_x"]] * spaces.VEL_SCALE
        vy = threats[:, e["vel_y"]] * spaces.VEL_SCALE
        boss = threats[:, e["entity_type"]] == protocol.EntityType.BOSS
        drop = boss & (abs(dx) < 3.4) & (dy > 2.5) & (dy < 14) & (vy < 1)
        buttons = action["buttons"]
        legal = mask[spaces.N_MOVEMENT_X + spaces.N_AIM_Y :]
        jump, hold, dash, attack = (
            spaces.BUTTON_BITS[k] for k in ("jump_tap", "jump_hold", "dash", "attack")
        )
        if np.any(drop):
            spikes = active[kind == protocol.EntityType.HAZARD]
            if len(spikes) == 2:
                centers = spikes[:, e["hurtbox_center_x"]] * spaces.ARENA_SCALE
                half = spikes[:, e["hurtbox_size_x"]] * spaces.ARENA_SCALE / 2
                left, right = int(np.argmin(centers)), int(np.argmax(centers))
                clearance = (x - centers[left] - half[left], centers[right] - half[right] - x)
                closest = dx[np.flatnonzero(drop)[np.argmin(abs(dx[drop]))]]
                side = int(closest < 0) if abs(closest) > 0.6 else int(clearance[1] > clearance[0])
                if clearance[side] < 5:
                    side = 1 - side
                move = 2 * side
                if mask[move]:
                    action["movement_x"] = move
                    if grounded and legal[dash] and clearance[side] > 8:
                        buttons[dash], buttons[attack], buttons[jump] = 1, 0, 0
                    return action
        # Original collector saw horizontal lunges around 55 units/s and low blades
        # before they crossed the player. Give the jump time to clear their height.
        antic = boss & (threats[:, e["fsm_state_hash"]] == self.DASH_ANTIC) & (abs(dx) < 14)
        towards = (dx * vx < 0) & (abs(dx) < 2.5 + 0.31 * abs(vx))
        low_attack = np.any((abs(dy) < 2.2) & (antic | towards))
        if grounded and low_attack and legal[jump]:
            buttons[jump] = 1
        if low_attack and player[p["vel_y"]] > 0 and legal[hold]:
            buttons[hold] = 1
        below = boss & (abs(dx) < 3.0) & (dy < -1.0) & (dy > -5.2)
        if not grounded and np.any(below) and mask[spaces.N_MOVEMENT_X] and legal[attack]:
            action["aim_y"] = 0
            buttons[attack] = 1
        return action


class SistersBodyCounterPolicy(SistersDodgeCounterPolicy):
    """Hold the jump over a lunge and stop within nail reach after a sister lands."""

    # Observed after the downward strike in slot-3 traces, on 49 of 50 confirmed hits.
    LANDED = np.float32(-1523407744)

    def act(self, obs: Any, action_mask: Any | None = None) -> Any:
        action = super().act(obs, action_mask)
        mask, _, _ = _validated_action_mask(self.action_space, action_mask)
        e, p = spaces.ENTITY_FEATURE_INDEX, spaces.PLAYER_FEATURE_INDEX
        active = obs["entities"][np.asarray(obs["entity_mask"], dtype=bool)]
        if active.shape[1] <= e["hurtbox_size_y"]:
            return action
        player = obs["player"]
        pos = player[:2] * spaces.ARENA_SCALE
        bosses = active[active[:, e["entity_type"]] == protocol.EntityType.BOSS]
        if not len(bosses):
            return action
        delta = bosses[:, 14:16] * spaces.ARENA_SCALE - pos
        states = bosses[:, e["fsm_state_hash"]]
        buttons = action["buttons"]
        jump, hold, dash, attack = (
            spaces.BUTTON_BITS[k] for k in ("jump_tap", "jump_hold", "dash", "attack")
        )
        button_offset = spaces.N_MOVEMENT_X + spaces.N_AIM_Y
        if buttons[dash]:
            return action
        lunge = (states == self.DASH_ANTIC) & (np.abs(delta[:, 0]) < 14)
        if lunge.any() and buttons[jump] and mask[button_offset + hold]:
            buttons[hold] = 1
            duration_offset = button_offset + spaces.N_BUTTONS
            if mask[duration_offset + 3]:
                action["duration"] = 3
            return action
        if not player[p["on_ground"]] or buttons[jump]:
            return action

        # Preserve the parent's evasions when another body or blade is closing.
        moving_threats = active[
            (active[:, 1] == protocol.EntityType.BOSS)
            | (
                (active[:, 1] == protocol.EntityType.PROJECTILE)
                & (active[:, 2] == protocol.Team.ENEMY)
            )
        ]
        dx, dy = (moving_threats[:, 14:16] * spaces.ARENA_SCALE - pos).T
        vx = moving_threats[:, e["vel_x"]] * spaces.VEL_SCALE
        if lunge.any() or np.any(
            (dx * vx < 0) & (abs(dy) < 2.2) & (abs(dx) < 2.5 + 0.31 * abs(vx))
        ):
            return action
        nearest = int(np.argmin(np.linalg.norm(delta, axis=1)))
        offset, height = delta[nearest]
        if states[nearest] != self.LANDED or abs(height) > 1.7 or abs(offset) > 3.6:
            return action
        move = 1
        if abs(offset) < 1.75:
            move = 0 if offset > 0 else 2
            projected = pos[0] + (move - 1) * 2
            spikes = active[active[:, e["entity_type"]] == protocol.EntityType.HAZARD]
            if np.any(
                abs(spikes[:, 14] * spaces.ARENA_SCALE - projected)
                < spikes[:, 16] * spaces.ARENA_SCALE / 2 + 0.6
            ):
                move = 1
        if mask[move]:
            action["movement_x"] = move
        if mask[spaces.N_MOVEMENT_X + 1]:
            action["aim_y"] = 1
        buttons[attack] = int(mask[button_offset + attack])
        return action

"""A slot-3 demonstration teacher that checks every visible threat before approaching."""

from __future__ import annotations

from typing import Any

import numpy as np

from hkrl import spaces
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

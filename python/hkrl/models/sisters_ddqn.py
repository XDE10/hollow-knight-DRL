"""Masked 48-action dueling Q network for privileged Sisters observations."""

from __future__ import annotations

from itertools import product

import numpy as np
import torch
from torch import Tensor, nn

from hkrl import spaces
from hkrl.models.encoders import EntityEncoder, GlobalEncoder, PlayerEncoder
from hkrl.models.recurrent_policy import landing_movement_bias
from hkrl.utils.registry import register_model

# Movement (left, neutral, right), jump held, nail (none, side, down, up), dash.
ACTIONS = tuple(product(range(3), range(2), range(4), range(2)))
E = spaces.ENTITY_FEATURE_INDEX


def game_action(index: int) -> dict:
    move, jump, nail, dash = ACTIONS[index]
    buttons = np.zeros(spaces.N_BUTTONS, dtype=np.int8)
    buttons[spaces.BUTTON_BITS["jump_hold"]] = jump
    buttons[spaces.BUTTON_BITS["attack"]] = int(nail != 0)
    buttons[spaces.BUTTON_BITS["dash"]] = dash
    return dict(movement_x=move, aim_y=(1, 1, 0, 2)[nail], buttons=buttons, duration=0, macro=0)


def _required_controls() -> np.ndarray:
    required = np.zeros((len(ACTIONS), 19), bool)
    for i in range(len(ACTIONS)):
        a = game_action(i)
        required[i, a["movement_x"]] = True
        required[i, 3 + a["aim_y"]] = True
        required[i, 6:15] = a["buttons"]
        required[i, 15] = True
    return required


REQUIRED = _required_controls()


def legal_actions(mask: np.ndarray) -> np.ndarray:
    return ~np.any(REQUIRED & ~np.asarray(mask[:19], dtype=bool), axis=-1)


@register_model("sisters_dueling_q")
class SistersDuelingQ(nn.Module):
    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.context = GlobalEncoder(9, hidden, exclude_run_ids=True)
        self.player = PlayerEncoder(32, hidden)
        self.entities = EntityEncoder(24, hidden, n_ids=256)
        self.query = nn.Linear(hidden * 2, hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden * 3 + 3, 128), nn.LayerNorm(128), nn.SiLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, len(ACTIONS))

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        mask = obs["entity_mask"].bool()
        raw = obs["entities"]
        entities = self.entities(raw, raw[..., E["entity_type"]], raw[..., E["entity_id"]])
        context = torch.cat((self.context(obs["global"]), self.player(obs["player"])), -1)
        scores = (entities * self.query(context).unsqueeze(-2)).sum(-1) / entities.shape[-1] ** 0.5
        weights = scores.masked_fill(~mask, -1e9).softmax(-1) * mask
        pooled = (entities * weights.unsqueeze(-1)).sum(-2)
        hidden = self.trunk(torch.cat((context, pooled, landing_movement_bias(obs)), -1))
        advantages = self.advantage(hidden)
        return self.value(hidden) + advantages - advantages.mean(-1, keepdim=True)

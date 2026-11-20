"""Shared object features, primitive actions and networks for model-based actor–critic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sisters_world_data import buckets, objects, players
from torch import Tensor, nn
from torch.distributions import Categorical
from train_sisters_object_world import ObjectWorld

# Left/neutral/right crossed with aim and real controller commands. Both the
# learned simulator and the live game receive the same primitive decision.
COMMANDS = [
    (0, 0),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 8),
    (-1, 8),
    (1, 8),
    (0, 10),
    (-1, 10),
    (1, 10),
    (0, 16),
    (-1, 16),
    (1, 16),
    (0, 32),
]
CATALOG = [(move, aim, buttons) for move in (-1, 0, 1) for aim, buttons in COMMANDS]
STATE = 96 + 32
KEYS = 13  # axes (four signed indicators) and all nine button bits
WEIGHTS = (0.6, -2.0, -2.0, -3.0, -2.0, 0.0, 12.0, -10.0)
GAMMA = 0.985
SCHEMA = 1


def keys(raw: np.ndarray) -> np.ndarray:
    move, aim, buttons = raw.astype(np.int64).T
    return np.stack(
        [
            move < 0,
            move > 0,
            aim < 0,
            aim > 0,
            *[(buttons & (1 << bit)) != 0 for bit in range(9)],
        ],
        -1,
    ).astype("f4")


def action_vector(table: Tensor, index: Tensor, interval: float = 0.12) -> Tensor:
    command = table[index]
    return torch.cat((command, torch.full_like(command[..., :1], interval / 0.2)), -1)


def controller_action(index: int) -> dict:
    move, aim, buttons = CATALOG[index]
    return {
        "movement_x": move + 1,
        "aim_y": aim + 1,
        "buttons": np.array([(buttons >> i) & 1 for i in range(9)], np.int8),
        "duration": 2,
        "macro": 0,
    }


def legal_actions(mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            mask[m + 1]
            and mask[3 + a + 1]
            and mask[17]
            and all(mask[6 + i] for i in range(9) if b & (1 << i))
            for m, a, b in CATALOG
        ],
        dtype=bool,
    )


def applied_controls(
    t: np.ndarray, end: np.ndarray, clock: np.ndarray, raw: np.ndarray
) -> np.ndarray:
    k = keys(raw)
    integral = np.concatenate(
        [np.zeros((1, KEYS)), np.cumsum(k[:-1] * np.diff(clock)[:, None], 0)]
    )

    def area(at: np.ndarray) -> np.ndarray:
        pos = (np.searchsorted(clock, at, side="right") - 1).clip(0, len(clock) - 1)
        return integral[pos] + k[pos] * (at - clock[pos]).clip(min=0)[:, None]

    dt = (end - t).clip(min=0.001)
    duty = ((area(end) - area(t)) / dt[:, None]).clip(0, 1)
    return np.c_[duty, dt / 0.2].astype("f4")


def encoder_from(weights: Path) -> ObjectWorld:
    model = ObjectWorld()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    return model.eval().requires_grad_(False)


def pulse_controls(duty: np.ndarray) -> np.ndarray:
    """A short button press can start an entire attack during the control interval."""
    action = duty.copy()
    # Axes and sustained holds retain their duration. Pulses record presence.
    pulse = [4, 6, 7, 8, 12]
    action[:, pulse] = (action[:, pulse] > 1e-4).astype(np.float32)
    return action


def encounter_slots(batch: dict[str, Tensor]) -> Tensor:
    obj, mask, pl = batch["obj"], batch["mask"], batch["player"]
    rel = obj[..., 14:16] - pl[:, None, :2]
    distance = rel[..., 0].abs() + 0.5 * rel[..., 1].abs()
    geo = torch.cat(
        (
            torch.ones_like(obj[..., :1]),
            rel,
            obj[..., 10:12],
            obj[..., 16:18],
            obj[..., 12:13],
        ),
        -1,
    )
    boss = (obj[..., 1] - 0.2).abs() < 0.01
    disc = ((obj[..., 1] - 0.6).abs() < 0.01) & ((obj[..., 2] - 0.2).abs() < 0.01)
    selected = []
    for kind, slots in ((boss, 3), (disc, 2)):
        visible = kind & mask & (obj[..., 16] > 0)
        order = distance.masked_fill(~visible, torch.inf).argsort(-1)[:, :slots]
        observed = geo.gather(1, order[..., None].expand(-1, -1, 8))
        observed = observed * visible.gather(1, order)[..., None]
        selected.append(
            torch.nn.functional.pad(
                observed, (0, 0, 0, slots - observed.shape[1])
            ).flatten(1)
        )
    return torch.cat(selected, -1)


@torch.no_grad()
def encode(
    world: ObjectWorld,
    player: np.ndarray,
    raw: np.ndarray,
    ptr: np.ndarray,
    combat_inputs: bool = False,
) -> Tensor:
    device = next(world.parameters()).device
    counts = np.diff(ptr)
    width = max(1, int(counts.max()))
    size = len(player)
    entity = np.zeros((size, width, 24), "f4")
    cat = np.zeros((size, width, 3), "i8")
    mask = np.zeros((size, width), bool)
    frame = np.repeat(np.arange(size), counts)
    slot = np.arange(len(raw)) - np.repeat(ptr[:-1], counts)
    entity[frame, slot] = objects(raw)
    cat[frame, slot] = buckets(raw[:, 3:6])
    mask[frame, slot] = True
    batch = {
        k: torch.as_tensor(v, device=device)
        for k, v in {
            "player": players(player),
            "obj": entity,
            "cat": cat,
            "mask": mask,
        }.items()
    }
    latent, _ = world.encode(batch)
    inputs = [latent, batch["player"]]
    if combat_inputs:
        inputs.append(encounter_slots(batch))
    return torch.cat(inputs, -1)


class ActorCritic(nn.Module):
    def __init__(self, state_size: int = STATE) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 192),
            nn.LayerNorm(192),
            nn.Tanh(),
            nn.Linear(192, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, len(CATALOG))
        self.critic = nn.Linear(128, 1)

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Categorical, Tensor]:
        h = self.trunk(x)
        logits = self.actor(h)
        if mask is not None:
            if not mask.any(-1).all():
                raise ValueError("no legal primitive in observation")
            logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits), self.critic(h).squeeze(-1)


class Dynamics(nn.Module):
    def __init__(self, state_size: int = STATE) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + KEYS + 1, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.delta = nn.Linear(256, state_size)
        self.event = nn.Linear(256, len(WEIGHTS))

    def forward(self, state: Tensor, command: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(torch.cat([state, command], -1))
        return state + self.delta(h), self.event(h)


def lambda_returns(
    reward: Tensor, next_value: Tensor, discount: Tensor, lam: float = 0.95
) -> Tensor:
    carry = next_value[-1]
    result = []
    for t in reversed(range(len(reward))):
        carry = reward[t] + discount[t] * ((1 - lam) * next_value[t] + lam * carry)
        result.append(carry)
    return torch.stack(result[::-1])

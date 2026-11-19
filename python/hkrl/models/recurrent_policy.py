"""Entity-attention + recurrent ActorCritic (docs/model_architecture.md §2).

The project's high-ceiling model: encoders -> masked attention -> GRU/LSTM memory
-> hybrid heads + value. Threads ``entity_mask`` through attention and ``rnn_state``
through the recurrence. Trained with truncated BPTT (+ optional burn-in) by
training/recurrent_ppo.py.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from hkrl import protocol
from hkrl.models.base import ActorCritic, RnnState
from hkrl.models.encoders import EntityEncoder, GlobalEncoder, PlayerEncoder
from hkrl.models.entity_attention import EntityTransformerEncoder
from hkrl.models.heads import (
    ACTION_TENSOR_DIM_NO_MACRO,
    CompositeActionDistribution,
    HybridPolicyHead,
    ValueHead,
)
from hkrl.models.sisters_timing import counter_preferences, platform_preferences
from hkrl.spaces import ARENA_SCALE, DEFAULT_N_MACROS, ENTITY_FEATURE_INDEX, VEL_SCALE
from hkrl.utils.registry import register_model


@register_model("entity_attention_gru")
class EntityAttentionRecurrentAC(ActorCritic):
    """Encoders -> attention(entity_context) -> [global|player|context|prev_action|
    prev_reward] -> GRU/LSTM -> heads.

    ``rnn_type`` in {"gru", "lstm"}. ``initial_state`` returns the zeroed hidden
    (tuple for LSTM). Reset hidden at episode boundaries (and optionally between
    bosses in a linear sequence — config-controlled).
    """

    exclude_run_ids = False

    def __init__(
        self,
        obs_dims: dict[str, Any],
        entity_hidden: int = 128,
        attention_layers: int = 2,
        attention_heads: int = 4,
        rnn_type: str = "gru",
        rnn_hidden: int = 256,
        enable_macro: bool = True,
        n_macros: int = DEFAULT_N_MACROS,
        max_entities: int = 64,
    ) -> None:
        super().__init__()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        self.rnn_type = rnn_type
        self.rnn_hidden = rnn_hidden
        self.action_dim = ACTION_TENSOR_DIM_NO_MACRO + (1 if enable_macro else 0)
        global_dim = _flat_dim(obs_dims, "global")
        player_dim = _flat_dim(obs_dims, "player")
        entity_shape = _shape(obs_dims, "entities")
        entity_dim = entity_shape[-1]

        self.global_encoder = GlobalEncoder(
            global_dim, entity_hidden, exclude_run_ids=self.exclude_run_ids
        )
        self.player_encoder = PlayerEncoder(player_dim, entity_hidden)
        self.entity_encoder = EntityEncoder(entity_dim, entity_hidden, n_types=256, n_ids=4096)
        self.entity_attention = EntityTransformerEncoder(
            dim=entity_hidden,
            layers=attention_layers,
            heads=attention_heads,
        )
        self.prev_action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, entity_hidden),
            nn.Tanh(),
        )
        scale = torch.ones((self.action_dim,), dtype=torch.float32)
        scale[0] = 2.0
        scale[1] = 2.0
        scale[ACTION_TENSOR_DIM_NO_MACRO - 1] = 3.0
        if enable_macro:
            scale[-1] = float(max(n_macros, 1))
        self.register_buffer("_prev_action_scale", scale)
        memory_input_dim = entity_hidden * 4 + 1
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(memory_input_dim, rnn_hidden, batch_first=True)
        self.policy = HybridPolicyHead(
            rnn_hidden,
            enable_macro=enable_macro,
            n_macros=n_macros,
        )
        self.value = ValueHead(rnn_hidden)
        self.max_entities = max_entities

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RnnState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        hidden = torch.zeros((1, batch_size, self.rnn_hidden), device=device)
        if self.rnn_type == "lstm":
            return hidden, torch.zeros_like(hidden)
        return hidden

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[CompositeActionDistribution, Tensor, RnnState]:
        memory_input, is_sequence = self._memory_input(obs)
        batch_size = memory_input.shape[0]
        if rnn_state is None:
            rnn_state = self.initial_state(batch_size, device=memory_input.device)

        memory_out, next_state = self.rnn(memory_input, rnn_state)
        if not is_sequence:
            memory_out = memory_out[:, 0]
        dist = self.policy(memory_out, action_mask=action_mask)
        value = self.value(memory_out)
        return dist, value, next_state

    def act(
        self,
        obs: dict[str, Tensor],
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RnnState]:
        with torch.no_grad():
            dist, value, next_state = self.forward(
                obs,
                rnn_state=rnn_state,
                action_mask=action_mask,
            )
            action = dist.mode() if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value, next_state

    def evaluate_actions(
        self,
        obs: dict[str, Tensor],
        actions: Tensor,
        rnn_state: RnnState = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        dist, value, _ = self.forward(obs, rnn_state=rnn_state, action_mask=action_mask)
        return dist.log_prob(actions), dist.entropy(), value

    def _memory_input(self, obs: dict[str, Tensor]) -> tuple[Tensor, bool]:
        global_state = obs["global"]
        is_sequence = global_state.ndim == 3
        if global_state.ndim not in (2, 3):
            raise ValueError("obs['global'] must have shape (B, G) or (B, T, G)")

        if not is_sequence:
            prepared = {
                "global": obs["global"].unsqueeze(1),
                "player": obs["player"].unsqueeze(1),
                "entities": obs["entities"].unsqueeze(1),
                "entity_mask": obs["entity_mask"].unsqueeze(1),
            }
        else:
            prepared = obs

        global_seq = prepared["global"]
        player_seq = prepared["player"]
        entities_seq = prepared["entities"]
        mask_seq = prepared["entity_mask"]
        batch_size, seq_len = global_seq.shape[:2]
        entity_count = entities_seq.shape[-2]
        entity_dim = entities_seq.shape[-1]

        global_emb = self.global_encoder(global_seq.reshape(batch_size * seq_len, -1))
        player_emb = self.player_encoder(player_seq.reshape(batch_size * seq_len, -1))
        entities = entities_seq.reshape(batch_size * seq_len, entity_count, entity_dim)
        entity_mask = mask_seq.reshape(batch_size * seq_len, entity_count)
        entity_type = entities[..., 1] if entity_dim > 1 else torch.zeros_like(entity_mask)
        entity_id = entities[..., 0] if entity_dim > 0 else None
        entity_embs = self.entity_encoder(entities, entity_type=entity_type, entity_id=entity_id)
        entity_context = self.entity_attention(entity_embs, entity_mask)

        prev_action = _optional_sequence_feature(
            obs,
            "prev_action",
            batch_size=batch_size,
            seq_len=seq_len,
            feature_dim=self.action_dim,
            device=global_seq.device,
            dtype=global_seq.dtype,
        )
        prev_reward = _optional_sequence_feature(
            obs,
            "prev_reward",
            batch_size=batch_size,
            seq_len=seq_len,
            feature_dim=1,
            device=global_seq.device,
            dtype=global_seq.dtype,
        )
        prev_action_scale = cast(Tensor, self._prev_action_scale)
        scale = prev_action_scale.to(device=prev_action.device, dtype=prev_action.dtype)
        prev_action = prev_action / torch.clamp_min(scale, 1.0)
        prev_action_emb = self.prev_action_encoder(prev_action.reshape(batch_size * seq_len, -1))
        prev_reward = prev_reward.reshape(batch_size * seq_len, 1)

        memory_input = torch.cat(
            [global_emb, player_emb, entity_context, prev_action_emb, prev_reward],
            dim=-1,
        )
        return memory_input.reshape(batch_size, seq_len, -1), is_sequence


@register_model("single_task_entity_attention_gru")
class SingleTaskEntityAttentionGRU(EntityAttentionRecurrentAC):
    """Keep bookkeeping IDs on the rollout, outside the single-task policy."""

    exclude_run_ids = True


@register_model("sisters_landing_gru")
class SistersLandingGRU(SingleTaskEntityAttentionGRU):
    """A soft movement prior from the two observed spike strips. Same weight layout."""

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[CompositeActionDistribution, Tensor, RnnState]:
        dist, value, state = super().forward(obs, rnn_state, action_mask)
        bias = landing_movement_bias(obs)
        movement = Categorical(logits=dist.movement_x.logits + bias)
        return replace(dist, movement_x=movement), value, state


@register_model("sisters_timing_gru")
class SistersTimingGRU(SistersLandingGRU):
    """A residual GRU with soft preferences for verified Sisters counter windows."""

    blade_interception = False
    sustain_charge_jump = False

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[CompositeActionDistribution, Tensor, RnnState]:
        dist, value, state = super().forward(obs, rnn_state, action_mask)
        return (
            counter_preferences(
                dist,
                obs,
                blade_interception=self.blade_interception,
                sustain_charge_jump=self.sustain_charge_jump,
            ),
            value,
            state,
        )


@register_model("sisters_blade_gru")
class SistersBladeGRU(SistersTimingGRU):
    """Use wall throw cues and the approaching edge of each live disc."""

    blade_interception = True


@register_model("sisters_platform_gru")
class SistersPlatformGRU(SistersTimingGRU):
    """Compare projected landings with the observed platform and allow an air recovery."""

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[CompositeActionDistribution, Tensor, RnnState]:
        dist, value, state = super().forward(obs, rnn_state, action_mask)
        return platform_preferences(dist, obs), value, state


@register_model("sisters_primitive_gru")
class SistersPrimitiveGRU(SistersPlatformGRU):
    """Keep the frozen heads compatible while requesting one primitive tick at a time."""

    def forward(
        self, obs: dict[str, Tensor], rnn_state: RnnState = None, action_mask: Tensor | None = None
    ) -> tuple[CompositeActionDistribution, Tensor, RnnState]:
        dist, value, state = super().forward(obs, rnn_state, action_mask)

        def first_only(options: Categorical) -> Categorical:
            logits = torch.full_like(options.logits, torch.finfo(options.logits.dtype).min)
            logits[..., 0] = 0.0
            return Categorical(logits=logits)

        return (
            replace(
                dist,
                duration=first_only(dist.duration),
                macro=first_only(dist.macro) if dist.macro is not None else None,
            ),
            value,
            state,
        )


@register_model("sisters_charge_gru")
class SistersChargeGRU(SistersPlatformGRU):
    """Finish an already-held rising jump over an approaching horizontal charge."""

    sustain_charge_jump = True


def landing_movement_bias(obs: dict[str, Tensor]) -> Tensor:
    """Return additive left/neutral/right logits for normalized Sisters observations."""
    player, entities = obs["player"], obs["entities"]
    fields = ENTITY_FEATURE_INDEX
    if entities.shape[-1] <= fields["hurtbox_size_y"] or entities.shape[-2] == 0:
        return player.new_zeros((*player.shape[:-1], 3))
    spikes = obs["entity_mask"].bool() & (
        entities[..., fields["entity_type"]] == protocol.EntityType.HAZARD
    )
    cx, cy, width, height = (
        entities[..., fields[k]] * ARENA_SCALE
        for k in ("hurtbox_center_x", "hurtbox_center_y", "hurtbox_size_x", "hurtbox_size_y")
    )
    spikes = spikes & (width > 0) & (height > 0)
    # Only accept a pair of real, separated colliders. Padding cannot change the ledges.
    left = (cx + width / 2).masked_fill(~spikes, float("inf")).amin(-1)
    right = (cx - width / 2).masked_fill(~spikes, -float("inf")).amax(-1)
    top = (cy + height / 2).masked_fill(~spikes, -float("inf")).amax(-1)
    valid = (spikes.sum(-1) == 2) & (right > left)
    x, y = player[..., 0] * ARENA_SCALE, player[..., 1] * ARENA_SCALE
    vx, vy = player[..., 2] * VEL_SCALE, player[..., 3] * VEL_SCALE
    # Ground movement and a descending jump need braking distance. High jumps retain
    # their normal movement distribution while their projected feet clear the spikes.
    near_floor = ((top + 5.2 - y - 0.22 * vy.clamp(max=0)) / 2.8).clamp(0, 1)
    left_gap, right_gap = x - left, right - x
    brake_l = ((2.8 - left_gap - 0.20 * vx.clamp(max=0)) / 1.4).clamp(0, 3)
    brake_r = ((2.8 - right_gap + 0.20 * vx.clamp(min=0)) / 1.4).clamp(0, 3)
    recover_l = ((1.5 - left_gap) / 1.2).clamp(0, 2)
    recover_r = ((1.5 - right_gap) / 1.2).clamp(0, 2)
    bias = torch.stack(
        (
            -7 * brake_l + 6 * recover_r,
            1.5 * (brake_l + brake_r) - 3 * (recover_l + recover_r),
            -7 * brake_r + 6 * recover_l,
        ),
        dim=-1,
    )
    return bias * (near_floor * valid).unsqueeze(-1)


def _shape(obs_dims: dict[str, Any], key: str) -> tuple[int, ...]:
    if key not in obs_dims:
        raise KeyError(f"obs_dims missing {key!r}")
    value = obs_dims[key]
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, int):
        return (value,)
    return tuple(int(dim) for dim in value)


def _flat_dim(obs_dims: dict[str, Any], key: str) -> int:
    shape = _shape(obs_dims, key)
    if not shape:
        raise ValueError(f"obs_dims[{key!r}] must have at least one dimension")
    return math.prod(shape)


def _optional_sequence_feature(
    obs: dict[str, Tensor],
    key: str,
    *,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    shape = (batch_size, seq_len, feature_dim)
    value = obs.get(key)
    if value is None:
        return torch.zeros(shape, dtype=dtype, device=device)

    tensor = value.to(device=device, dtype=dtype)
    if feature_dim == 1:
        if tensor.shape == (batch_size,):
            tensor = tensor.reshape(batch_size, 1, 1)
        elif tensor.shape == (batch_size, seq_len):
            tensor = tensor.unsqueeze(-1)
        elif tensor.shape == (batch_size, 1) and seq_len == 1:
            tensor = tensor.reshape(batch_size, 1, 1)
    elif tensor.shape == (batch_size, feature_dim) and seq_len == 1:
        tensor = tensor.unsqueeze(1)

    if tensor.shape != shape:
        raise ValueError(f"obs[{key!r}] must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor

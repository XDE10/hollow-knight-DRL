"""CPU rollout inference for the built-in entity-attention GRU policies."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

from hkrl.models.base import ActorCritic
from hkrl.models.recurrent_policy import EntityAttentionRecurrentAC


class _GRUAction(nn.Module):
    def __init__(self, policy: EntityAttentionRecurrentAC) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        world: Tensor,
        player: Tensor,
        entities: Tensor,
        entity_mask: Tensor,
        prev_action: Tensor,
        prev_reward: Tensor,
        hidden: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        obs = {
            "global": world,
            "player": player,
            "entities": entities,
            "entity_mask": entity_mask,
            "prev_action": prev_action,
            "prev_reward": prev_reward,
        }
        return self.policy.act(obs, rnn_state=hidden, action_mask=mask)


class TracedGRUSampler:
    """Trace only acting. PPO keeps the eager network and the same parameter storage."""

    def __init__(self, model: ActorCritic) -> None:
        if (
            not isinstance(model, EntityAttentionRecurrentAC)
            or not isinstance(model.rnn, nn.GRU)
            or next(model.parameters()).device.type != "cpu"
        ):
            raise ValueError("traced local inference requires a CPU entity-attention GRU")
        self.policy = model
        self.graph: torch.jit.ScriptModule | None = None

    @torch.inference_mode()
    def act(
        self, obs: dict[str, Tensor], rnn_state: Tensor, action_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        args = (
            obs["global"],
            obs["player"],
            obs["entities"],
            obs["entity_mask"],
            obs["prev_action"],
            obs["prev_reward"].reshape(-1, 1),
            rnn_state,
            action_mask,
        )
        if self.graph is None:
            if self.policy.training:
                raise ValueError("trace the sampler only when the policy is in eval mode")
            encoder = self.policy.entity_attention.encoder
            nested = encoder.use_nested_tensor
            try:
                # Padding may be non-contiguous or an entire window may be empty.
                # Use the dense mask path instead of specializing on the first mask.
                encoder.use_nested_tensor = False
                with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
                    warnings.simplefilter("ignore", torch.jit.TracerWarning)
                    graph = torch.jit.trace(_GRUAction(self.policy), args, check_trace=False)
            finally:
                encoder.use_nested_tensor = nested
            parameters = dict(graph.policy.named_parameters())
            if any(
                parameters[k].data_ptr() != v.data_ptr() for k, v in self.policy.named_parameters()
            ):
                raise RuntimeError("traced sampler did not share all trainable weights")
            self.graph = graph
        return self.graph(*args)

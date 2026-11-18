"""Batched local sampling must preserve per-window memory, masks, and errors."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
import torch
from hkrl.models.base import ActorCritic, RnnState
from hkrl.worker.local_inference import ActionResult, LocalActionBatcher


@pytest.mark.parametrize("memory", ["gru", "lstm", "none"])
def test_independent_windows_share_one_forward_and_get_their_own_state(memory: str) -> None:
    started, resume = Event(), Event()
    calls: list[int] = []

    def forward(
        obs: dict[str, torch.Tensor], rnn_state: RnnState, action_mask: torch.Tensor
    ) -> ActionResult:
        data = obs["player"]
        calls.append(len(data))
        if len(calls) == 1:
            started.set()
            assert resume.wait(5)
        value = data * action_mask
        after = rnn_state
        if memory == "gru":
            after = rnn_state + data[None]
        if memory == "lstm":
            after = (rnn_state[0] + data[None], rnn_state[1] - data[None])
        return data, value + 1, value, after

    model = Mock(spec=ActorCritic)
    model.act.side_effect = forward
    batcher = LocalActionBatcher(model, 3)

    def request(i: int) -> ActionResult:
        x = torch.tensor([[i]], dtype=torch.float32)
        state: RnnState = None if memory == "none" else (x * 3)[None]
        if memory == "lstm":
            state = (state, state + 1)
        return batcher.act({"player": x}, state, (x % 2).bool())

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(request, 1)
            assert started.wait(5)
            futures = [first, pool.submit(request, 2), pool.submit(request, 3)]
            deadline = time.monotonic() + 5
            while batcher._queue.qsize() != 2 and time.monotonic() < deadline:
                time.sleep(0.002)
            resume.set()
            for i, future in enumerate(futures, 1):
                act, logp, value, hidden = future.result(timeout=5)
                assert act.item() == i and value.item() == i * (i % 2)
                assert logp.item() == value.item() + 1
                if memory == "gru":
                    assert hidden.shape == (1, 1, 1) and hidden.item() == 4 * i
                elif memory == "lstm":
                    assert [t.item() for t in hidden] == [4 * i, 2 * i + 1]
                else:
                    assert hidden is None
        assert calls == [1, 2] and batcher.decisions == 3
    finally:
        resume.set()
        batcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        request(1)


def test_inference_error_reaches_waiter_and_next_request_can_run() -> None:
    model = Mock(spec=ActorCritic)
    x = torch.ones(1, 1)
    model.act.side_effect = [ValueError("bad observation"), (x, x, x, None)]
    batcher = LocalActionBatcher(model, 2)
    try:
        with pytest.raises(ValueError, match="bad observation"):
            batcher.act({"player": x}, None, x.bool())
        assert batcher.act({"player": x}, None, x.bool())[2].item() == 1
    finally:
        batcher.close()


def test_traced_gru_preserves_windows_masks_rng_and_updated_ppo_weights() -> None:
    from hkrl.models.recurrent_policy import SistersPlatformGRU
    from hkrl.worker.traced_inference import TracedGRUSampler

    dims = {"global": (9,), "player": (32,), "entities": (6, 24), "entity_mask": (6,)}
    model = SistersPlatformGRU(dims, entity_hidden=16, rnn_hidden=16, attention_heads=2).eval()
    sampler = TracedGRUSampler(model)
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(719)
        for batch in (1, 3):
            obs = {key: torch.zeros(batch, *shape) for key, shape in dims.items()}
            obs["entity_mask"] = obs["entity_mask"].bool()
            obs["entity_mask"][0, [1, 4]] = True
            obs["entities"][:, 1, 1:3] = torch.tensor([1, 2])
            obs["prev_action"] = torch.zeros(batch, model.action_dim)
            obs["prev_reward"] = torch.randn(batch) if batch == 1 else torch.randn(batch, 1)
            state = torch.randn(1, batch, 16)
            mask = torch.ones(batch, model.policy.mask_dim, dtype=torch.bool)
            mask[:, -1] = False
            before = model(obs, rnn_state=state, action_mask=mask)[1].clone()
            for mutation in (False, True):
                if mutation:
                    model.value.v.bias.add_(2)
                    model.policy.movement_x.bias[0].add_(1)
                torch.manual_seed(317)
                expected = model.act(obs, rnn_state=state, action_mask=mask)
                torch.manual_seed(317)
                sampled = sampler.act(obs, state, mask)
                for actual, reference in zip(sampled, expected, strict=True):
                    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
                if mutation:
                    torch.testing.assert_close(sampled[2], before + 2)

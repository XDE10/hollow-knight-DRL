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

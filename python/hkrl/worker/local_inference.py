"""Coalesce ready game windows into a single local policy forward pass."""

from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread

import torch
from torch import Tensor

from hkrl.models.base import ActorCritic, RnnState
from hkrl.worker.traced_inference import TracedGRUSampler

ActionResult = tuple[Tensor, Tensor, Tensor, RnnState]


@dataclass
class _Request:
    obs: dict[str, Tensor]
    state: RnnState
    mask: Tensor
    reply: Future[ActionResult]


class LocalActionBatcher:
    """Each caller has one outstanding decision and keeps its own recurrent state."""

    def __init__(self, model: ActorCritic, windows: int, *, trace: bool = False) -> None:
        if windows < 2:
            raise ValueError("local inference batching requires at least two game windows")
        self.model = model
        self._act = TracedGRUSampler(model).act if trace else model.act
        self.windows = windows
        self._queue: Queue[_Request | None] = Queue()
        self._lock = Lock()
        self._closed = False
        self.forward_passes = 0
        self.decisions = 0
        self._thread = Thread(target=self._serve, name="hkrl-local-inference", daemon=True)
        self._thread.start()

    def act(self, obs: dict[str, Tensor], rnn_state: RnnState, action_mask: Tensor) -> ActionResult:
        request = _Request(obs, rnn_state, action_mask, Future())
        with self._lock:
            if self._closed:
                raise RuntimeError("local inference batcher is closed")
            self._queue.put(request)
        return request.reply.result()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join()

    def _serve(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            closing = False
            # A resetting or slow game must never hold up all other windows.
            deadline = time.monotonic() + 0.002
            while len(batch) < self.windows:
                try:
                    item = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
                except Empty:
                    break
                if item is None:
                    closing = True
                    break
                batch.append(item)
            try:
                results = self._infer(batch)
                self.forward_passes += 1
                self.decisions += len(batch)
                for request, result in zip(batch, results, strict=True):
                    request.reply.set_result(result)
            except BaseException as exc:
                for request in batch:
                    if not request.reply.done():
                        request.reply.set_exception(exc)
            if closing:
                return

    @torch.inference_mode()
    def _infer(self, batch: list[_Request]) -> list[ActionResult]:
        if len(batch) == 1:
            item = batch[0]
            return [self._act(item.obs, rnn_state=item.state, action_mask=item.mask)]
        obs = {key: torch.cat([r.obs[key] for r in batch]) for key in batch[0].obs}
        masks = torch.cat([r.mask for r in batch])
        states = [r.state for r in batch]
        state = states[0]
        if isinstance(state, tuple):
            state = tuple(torch.cat(parts, dim=1) for parts in zip(*states, strict=True))
        elif state is not None:
            state = torch.cat(states, dim=1)
        actions, logps, values, after = self._act(obs, rnn_state=state, action_mask=masks)
        results = []
        for i in range(len(batch)):
            one = slice(i, i + 1)
            hidden = after
            if isinstance(after, tuple):
                hidden = tuple(part[:, one] for part in after)
            elif after is not None:
                hidden = after[:, one]
            results.append((actions[one], logps[one], values[one], hidden))
        return results

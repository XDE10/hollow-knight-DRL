from __future__ import annotations

import importlib
from pathlib import Path

import torch


def test_greedy_horizontal_logits_ignore_how_many_keys_share_a_direction(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    model = importlib.import_module("train_sisters_descent_returns")
    scores = torch.tensor([[1.0, 2.0, 1.9, 2.1], [5.0, 3.0, 2.0, 1.0]], requires_grad=True)
    requested = torch.tensor([[-1, 1], [1, -1]])
    result = model.direction_logits(scores, torch.tensor([-1, -1, -1, 1]), requested)
    torch.testing.assert_close(result, torch.tensor([[2.0, 2.1], [1.0, 5.0]]))
    result.sum().backward()
    assert scores.grad.tolist() == [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]

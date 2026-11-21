from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def test_teacher_spell_windows_respect_handoffs_and_distinguish_requests(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    audit = importlib.import_module("audit_sisters_spell_requests").audit_keys
    t = np.arange(45, dtype="f4") * 0.04
    s = np.zeros((len(t), 192), "f4")
    s[:, 5] = 1
    s[:, 11] = s[:, 23] = 1
    s[12:, 5] = 2 / 3
    k = np.zeros((len(t), 3), int)
    k[5:8, 2] = k[22:25, 2] = 16
    k[5:8, 1] = k[22:25, 1] = -1
    taught = np.ones(len(t), bool)
    taught[26:31] = False
    chosen = k.copy()
    chosen[5] = 0
    chosen[3, 2] = 16
    spells = k.copy()
    out = audit(t, s, k, taught, chosen, spells, np.zeros_like(k))
    assert out["observed_cast_starts"] == 2 and out["network_same_tick"] == 1
    assert out["pre_has_eligible_tick"] == 2 and out["post_has_eligible_tick"] == 0
    assert out["debit_missed_oracle_aim_move"] == out["debit_down"] == 1
    assert out["outcome_window_missing"] == 1


def test_debit_windows_and_folded_downward_spell_scores(monkeypatch) -> None:
    import torch
    from torch import nn

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    training = importlib.import_module("train_sisters_spell_returns")
    t = np.arange(75, dtype="f4") * 0.04
    s = np.zeros((len(t), 192), "f4")
    s[:, 5] = s[:, 11] = s[:, 23] = 1
    k = np.zeros((len(t), 3), int)
    k[12:15, 1] = k[35:38, 1] = -1
    k[12:15, 2] = k[35:38, 2] = 16
    s[1:, 168] = (k[:-1, 2] & 16) != 0
    s[19:, 5] = 2 / 3
    teacher = np.ones(len(t), bool)
    teacher[39:44] = False
    p, n, eligible = training.spell_examples(t, s, k, teacher)
    assert np.flatnonzero(p).tolist() == [12]
    assert not n[12] and not n[34] and n[55] and not eligible[13]

    keys = torch.tensor([[0, -1, 16], [0, 0, 16], [1, -1, 8], [-1, -1, 16]])
    policy = training.TimingPolicy(keys, history=True)
    correction = nn.Linear(192, 1)
    h = torch.randn(16, 192)
    group = torch.tensor([True, False, False, True])
    old = policy.heads(h)[0].detach()
    training.fold_spell_correction(policy, correction, group)
    new = policy.heads(h)[0].detach()
    torch.testing.assert_close(new, old + correction(h).detach() * group)
    assert torch.equal(old[:, ~group], new[:, ~group])

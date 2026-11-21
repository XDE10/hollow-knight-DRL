from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch


def test_greedy_group_penalizes_a_winning_wrong_key(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.3], [1.0, 1.0, 1.0, 0.5]], requires_grad=True)
    penalty = mod.greedy_motion_loss(logits, torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0]))
    assert penalty[0] > penalty[1]
    penalty.sum().backward()
    assert logits.grad[0, -1] > 0


def test_real_teacher_movement_runs_world_and_iql_with_both_heldout_counts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    manifest = []
    for i, split in enumerate(("train", "holdout")):
        t = np.arange(28, dtype="f4") * 0.04
        s = np.zeros((len(t), 192), "f4")
        s[:, [23, 24, 44]] = 1
        s[:, 4] = 1
        keys = np.zeros((len(t), 3), "i2")
        keys[:, 0] = np.repeat([-1, 1, -1, 1], 7)
        s[1:, 160] = keys[:-1, 0]
        acted = np.ones(len(t), bool)
        acted[:3] = False
        path = tmp_path / f"f{i}.npz"
        np.savez(
            path,
            controls=keys,
            teacher_controls=keys,
            teacher_acted=acted,
            timing_state=s,
            control_time=t,
            time=t,
            elapsed_ticks=np.full(len(t), 2),
            events=np.zeros((len(t), 8), "f4"),
        )
        manifest.append(
            {
                "source": f"native-development/{i}",
                "file": path.name,
                "split": split,
                "won": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    mod.train(
        tmp_path,
        tmp_path / "fit",
        1,
        2,
        3,
        4,
        927,
        False,
        return_seconds=0.32,
        executed_teacher_advantage=True,
    )
    report = json.loads((tmp_path / "fit/training.json").read_bytes())
    checks = report["heldout_descent_executed_motion"]
    assert checks["change"]["rows"] == 3
    assert checks["hold"]["rows"] == 21
    assert (tmp_path / "fit/world_iql.bin").exists()

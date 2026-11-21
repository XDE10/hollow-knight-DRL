from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np


def test_spell_charges_only_reward_resources_present_at_the_real_handoff(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_timing_iql")
    rows = []
    for i, soul in enumerate((99, 32, 66)):
        game = "train" if i < 2 else "holdout"
        t = np.arange(9, dtype="f4") * 0.04
        s = np.zeros((len(t), mod.WIDTH), "f4")
        s[:, 4] = 1
        s[7:, 23] = 1
        s[7:, 5] = soul / 99
        a = np.zeros((len(t), 3), "i2")
        acted = np.ones(len(t), bool)
        acted[1] = False
        f = tmp_path / f"{i}.npz"
        np.savez(
            f,
            control_time=t,
            time=t,
            timing_state=s,
            controls=a,
            teacher_controls=a,
            teacher_acted=acted,
            elapsed_ticks=np.full(len(t), 2),
            events=np.zeros((len(t), 8), "f4"),
        )
        rows.append(
            {
                "file": f.name,
                "source": f"native-development/{i}",
                "split": game,
                "won": False,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
        )
        base, term = mod.opening_rewards(s, np.zeros(len(t)), np.zeros(len(t), bool))
        shaped, mark = mod.opening_rewards(s, np.zeros(len(t)), np.zeros(len(t), bool), True)
        assert sum(shaped - base) == 3 * (soul // 33)
        assert np.array_equal(term, mark) and np.flatnonzero(mark).tolist() == [6]
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    data, _, _ = mod.build_data(
        tmp_path, "cpu", mirror=False, opening_only=True, reserve_opening=True
    )
    age = data["train"][7].numpy()
    assert len(age) == 14 and np.isneginf(age[1]) and np.isneginf(age[7:]).all()
    assert np.isposinf(age[[0, 2, 3, 4, 5, 6]]).all()
    np.testing.assert_allclose(data["train"][5][0].item(), np.exp(-0.04 / 40), rtol=1e-6)
    assert data["train"][5][6].item() == 0

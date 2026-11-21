from __future__ import annotations

import hashlib
import importlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_imagined_keys")


def test_imagined_control_matches_native_release_and_availability(study):
    cat = torch.tensor([[0, 0, 0], [1, 0, 2], [1, 0, 3], [1, 0, 6], [0, 0, 8]])
    scores = torch.tensor([[5.0, 2.0, 4.0, 3.0, 1.0]])
    flags = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    first, other, ok = study.compound_pair(scores, cat, flags)
    assert ok.item() and first.item() == 0 and other.item() == 2
    assert study.continuation(scores, cat, other, "compound").item() == 1
    assert study.continuation(scores, cat, other, "horizontal").item() == 2
    assert study.continuation(scores, cat, other, "free").item() == 0
    flags[:, 5:7] = 1
    assert study.compound_pair(scores, cat, flags)[1].item() == 3
    flags[:, 9] = 1
    assert not study.compound_pair(scores, cat, flags)[2].item()


def test_future_box_queries_share_one_world_coordinate_system(study):
    z = torch.zeros(2, study.ENCODED + study.GEO)
    boxes = torch.zeros(2, 5, 5)
    # A body at the initial Knight position and a disc 2.5 units to its right.
    boxes[1, 0] = torch.tensor([0.0, 0.0, 0.1, 0.1, 1.0])
    boxes[1, 3] = torch.tensor([2.5, 0.0, 0.1, 0.1, 1.0])
    time = torch.ones(2, 2, dtype=torch.long)
    known = torch.tensor([[True, False], [False, False]])
    bag = {
        "data": [None] * 4 + [boxes, None, time, known],
        "mean": torch.zeros(z.shape[-1]),
        "scale": torch.ones(z.shape[-1]),
    }
    grid, mask = study.observed_fields(bag, torch.tensor([0]), z)
    assert grid.shape == (1, 2, 2, 9, 11) and mask.flatten().tolist() == [True, False]
    assert grid[0, 0, 0, 4, 5] == 1 and grid[0, 0, 1, 4, 7] == 1
    assert grid[0, 0, 1, 4, 5] < 0.2


def test_world_rollouts_do_not_cross_phase_or_native_clock_gap(study):
    clock = np.arange(12) * 0.04
    clock[8:] += 0.2
    trio = np.arange(12) >= 2
    assert list(study.rollout_starts(clock, trio, 3)) == [2, 3, 4, 8]


def test_world_shortlist_requires_output_and_randomized_motion(study):
    from train_sisters_compound_options import world_screen

    physical = [{"assigned_x_error": 0.5, "swapped_x_error": 0.6}] * 6

    def rows(v):
        return [{"vs_H": x} for x in v]

    stats = {
        "4.0": {"native_masks_lost": rows([-0.1] * 4 + [0] * 2), "boss_damage": rows([1] * 6)},
        "8.0": {"boss_damage": rows([0] * 6)},
    }
    assert world_screen(physical, stats)
    stats["8.0"]["boss_damage"] = rows([-1] * 6)
    assert not world_screen(physical, stats)


def test_held_draws_use_the_catalogue_that_was_executed(study, tmp_path):
    from train_sisters_timing_iql import TimingPolicy

    # Identical integer ID 0 means opposite movement in the two native policies.
    old = torch.tensor([[-1, 0, 0], [1, 0, 0]])
    new = old.flip(0)
    historical, current = TimingPolicy(old, True), TimingPolicy(new, True)
    t = np.arange(24, dtype="f4") * 0.04
    state = np.zeros((len(t), 192), dtype="f4")
    state[:, 4] = state[:, 23] = 1
    raw = io.BytesIO()
    np.savez(
        raw,
        control_time=t,
        time=t,
        elapsed_ticks=np.full(len(t), 2),
        events=np.zeros((len(t), 8), "f4"),
        timing_state=state,
        controls=np.tile(old[0].numpy(), (len(t), 1)),
        choice_rows=np.array([0]),
        choice_draw=np.array([0]),
        choice_pairs=np.array([[0, 1]]),
        choice_hold_stop=np.array([4]),
    )
    blob = raw.getvalue()
    archive = tmp_path / "choices.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("fight.npz", blob)
        z.writestr(
            "manifest.json",
            json.dumps(
                [
                    {
                        "source": f"native-development/{i}",
                        "file": "fight.npz",
                        "sha256": hashlib.sha256(blob).hexdigest(),
                        "split": kind,
                        "behavior_policy_sha256": "historical",
                    }
                    for i, kind in enumerate(("train", "holdout"))
                ]
            ),
        )
    correct = study.load_replay(
        archive, current, historical, old, "historical", torch.device("cpu")
    )
    assert len(correct["holdout"]["draws"]) == 1
    with pytest.raises(ValueError, match="issued keys"):
        study.load_replay(archive, current, historical, new, "historical", torch.device("cpu"))

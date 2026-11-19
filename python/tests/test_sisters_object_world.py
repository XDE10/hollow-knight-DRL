from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_native_controls_preserve_a_short_press(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from sisters_world_data import CONTROL, interval_actions

    t = np.array([0.0, 0.375, 0.5, 1.0])
    controls = np.zeros((4, 3), dtype=np.int16)
    controls[:, 0] = 1
    controls[1, 2] = 8
    action = interval_actions(t[:1], t[-1:], t, controls)
    bins = action[0, : 4 * CONTROL].reshape(4, CONTROL)
    np.testing.assert_array_equal(bins[:, 1], [1, 1, 1, 1])
    np.testing.assert_array_equal(bins[:, 6], [0, 0.5, 0, 0])


def test_object_ids_terminal_masks_and_game_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from sisters_world_data import WorldReplay

    rows = 12
    first = np.zeros((rows, 2, 24), np.float32)
    first[:, :, 0] = [104, 946]
    first[:, :, 1:3] = [1, 1]
    first[:, :, 12] = [111, 222]
    first[1::2] = first[1::2, ::-1].copy()
    t = np.arange(rows, dtype=np.float32) * 0.1
    ct = np.arange(rows * 5 + 1, dtype=np.float32) * 0.02
    events = np.zeros((rows, 8), np.float32)
    events[-1, 6] = 1
    for split in ("train", "holdout"):
        np.savez(
            tmp_path / f"{split}.npz",
            time=t,
            entities=first.reshape(-1, 24),
            entity_ptr=np.arange(rows + 1) * 2,
            player=np.zeros((rows, 32), np.float32),
            control_time=ct,
            elapsed_ticks=np.full(rows, 5),
            events=events,
            student_features=np.zeros((len(ct), 56), np.float32),
            controls=np.zeros((len(ct), 3), np.int16),
        )
    entries = [
        {"file": f"{s}.npz", "max_entities": 2, "won": True, "split": s}
        for s in ("train", "holdout")
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(entries))
    replay = WorldReplay(tmp_path, "cpu")
    for split in ("train", "holdout"):
        batch = replay.batch(replay.indices[split][-1:])
        assert batch["event"][0, -1, 6] == 1
        assert batch["observed"][0].tolist() == [True, True, True, True, False]
        expected_hp = batch["obj"][0, :, 12].tolist()
        assert batch["obj_target"][0, 0, :, 4].tolist() == expected_hp
        assert batch["obj_present"][0, -1].tolist() == [False, False]

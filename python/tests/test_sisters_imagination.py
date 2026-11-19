from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from hkrl import protocol, spaces


def scripts(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_imagination")


def test_only_future_value_changes_on_damage(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = scripts(monkeypatch)
    # Take a full hit, then land one. The observed first reward is shared.
    probs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    assert mod.imagined_returns(probs).tolist()[0] == pytest.approx([-12 + 0.98 * 6.3, -12])
    probs[0, 0] = torch.tensor([0.0, 0.0, 1.0])
    assert mod.imagined_returns(probs).tolist()[0] == pytest.approx([-100.0, -100.0])


def test_donor_continuation_is_disjoint_and_matches_boss_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = scripts(monkeypatch)
    valid = np.ones(640, dtype=bool)
    valid[319::320] = False
    fsm = torch.tensor(np.repeat([17, 22], 320))
    data = SimpleNamespace(valid=valid, fsm=fsm)
    anchors = np.array([4, 170, 312, 323, 460, 635])
    donor = mod.donors_for(data, anchors, np.random.default_rng(5))
    assert np.all(np.abs(donor - anchors) > mod.HORIZON + mod.HISTORY)
    np.testing.assert_array_equal(fsm[donor], fsm[anchors])
    assert valid[donor[:, None] + np.arange(1 - mod.HISTORY, mod.HORIZON)].all()


def test_live_history_matches_training_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = scripts(monkeypatch)
    packer = importlib.import_module("pack_sisters_sequences")
    live = importlib.import_module("eval_sisters_imagination")
    n = 10
    player = np.zeros((n, 1, 32), np.float32)
    player[..., :2] = np.arange(n, dtype=np.float32)[:, None, None] / 20
    player[..., spaces.PLAYER_FEATURE_INDEX["hp"]] = 1
    entities = np.zeros((n, 1, 3, 24), np.float32)
    e = spaces.ENTITY_FEATURE_INDEX
    entities[:, :, 1, e["entity_type"]] = protocol.EntityType.BOSS
    entities[:, :, 1, e["fsm_state_hash"]] = 37
    entities[:, :, 1, e["max_hp"]] = 1
    entities[:, :, 2, e["entity_type"]] = protocol.EntityType.PROJECTILE
    entities[:, :, 2, e["team"]] = protocol.Team.ENEMY
    mask = entities[..., e["entity_type"]] != 0
    mask[2:4, :, 2] = False
    g = np.zeros((n, 1, 9), np.float32)
    g[..., 4] = np.arange(n, dtype=np.float32)[:, None] / 8
    action = np.array([2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype=np.uint8)
    raw = tmp_path / "raw"
    raw.mkdir()
    np.savez(
        raw / "batch.npz",
        obs_player=player,
        obs_entities=entities,
        entity_mask=mask,
        obs_global=g,
        episode_ids=np.ones((n, 1)),
        dones=np.zeros((n, 1), bool),
        truncateds=np.zeros((n, 1), bool),
        actions=np.tile(action, (n, 1, 1)),
        action_masks=np.ones((n, 1, 31), bool),
    )
    packer.pack(raw, tmp_path / "pack.npz")
    replay = mod.Replay(tmp_path / "pack.npz", "cpu")
    correct = replay.history(torch.tensor([4])).numpy()[0]
    actor = mod.student(len(correct), 1)
    ckpt = tmp_path / "test.pt"
    torch.save(
        dict(
            weights=actor.state_dict(),
            catalogue=torch.tensor(action[None]),
            mean=torch.tensor(replay.mean),
            scale=torch.tensor(replay.scale),
            vocab=torch.tensor(replay.vocab),
        ),
        ckpt,
    )
    policy = live.ImaginedPolicy(ckpt)
    captured = []
    policy.actor.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].numpy().copy())
    )
    for i in range(5):
        obs = dict(
            player=player[i, 0], global_=g[i, 0], entities=entities[i, 0], entity_mask=mask[i, 0]
        )
        obs["global"] = obs.pop("global_")
        out = policy.act(obs, np.ones(31, bool))
        assert out["movement_x"] == 2
    np.testing.assert_allclose(captured[-1], correct, atol=1e-6)
    blocked = np.ones(31, bool)
    blocked[2] = False
    assert policy.act(obs, blocked)["movement_x"] == 1

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from hkrl import spaces
from hkrl.models.sisters_ddqn import ACTIONS, SistersDuelingQ, game_action, legal_actions


def load_learner() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "sisters_ddqn_learn.py"
    spec = importlib.util.spec_from_file_location("sisters_ddqn_learn", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controls_are_unique_and_respect_the_live_mask() -> None:
    wire = np.ones(19, dtype=bool)
    decoded = [game_action(i) for i in range(len(ACTIONS))]
    keys = {(a["movement_x"], a["aim_y"], tuple(a["buttons"])) for a in decoded}
    assert len(keys) == 48 and legal_actions(wire).all()
    assert all(a["duration"] == a["macro"] == 0 for a in decoded)
    wire[6 + spaces.BUTTON_BITS["attack"]] = False
    wire[0] = False
    for i, (move, _, attack, _) in enumerate(ACTIONS):
        assert legal_actions(wire)[i] == (move != 0 and attack == 0)


def test_all_entities_are_permutation_invariant_and_padding_is_ignored() -> None:
    torch.manual_seed(13)
    net = SistersDuelingQ(hidden=16).eval()
    obs = {
        "global": torch.randn(2, 9),
        "player": torch.randn(2, 32),
        "entities": torch.randn(2, 9, 24),
        "entity_mask": torch.tensor([[1, 1, 1, 0, 1, 1, 0, 0, 0], [0] * 9], dtype=torch.bool),
    }
    with torch.no_grad():
        q = net(obs)
        obs["entities"][~obs["entity_mask"]] = -2e4
        order = torch.randperm(9)
        obs["entities"] = obs["entities"][:, order]
        obs["entity_mask"] = obs["entity_mask"][:, order]
        torch.testing.assert_close(q, net(obs))
    assert q.shape == (2, 48) and q.isfinite().all()


def test_target_uses_online_legal_argmax_and_only_stops_at_true_terminal() -> None:
    module = load_learner()
    values = module.targets(
        torch.tensor([1.0, -1.0]),
        torch.tensor([False, True]),
        torch.tensor([0.5, 0.99]),
        torch.tensor([[200.0, 5.0, 9.0], [0.0, 1.0, 2.0]]),
        torch.tensor([[500.0, 50.0, 6.0], [5.0, 4.0, 3.0]]),
        torch.tensor([[False, True, True], [False, False, False]]),
    )
    torch.testing.assert_close(values, torch.tensor([4.0, -1.0]))


def test_replay_keeps_late_active_entities_across_all_shards(tmp_path: Path) -> None:
    module = load_learner()
    for number, slot in ((1, 2), (2, 63)):
        fields = {}
        for prefix in ("s_", "n_"):
            fields[prefix + "global"] = np.zeros((2, 9), dtype=np.float32)
            fields[prefix + "player"] = np.zeros((2, 32), dtype=np.float32)
            fields[prefix + "entities"] = np.zeros((2, 64, 24), dtype=np.float32)
            fields[prefix + "entity_mask"] = np.zeros((2, 64), dtype=bool)
        fields["n_entity_mask"][0, slot] = True
        fields["n_entities"][0, slot, 3] = number
        np.savez(
            tmp_path / f"chunk-{number:02d}.npz",
            **fields,
            a=np.zeros(2, dtype=int),
            r=np.zeros(2),
            term=np.zeros(2, dtype=bool),
            ticks=np.ones(2),
            mask=np.ones((2, 48), dtype=bool),
            next_mask=np.ones((2, 48), dtype=bool),
        )
    replay = module.replay(tmp_path, 2, 2, "cpu")
    assert replay["n_entities"].shape == (4, 64, 24)
    assert replay["n_entities"][0, 2, 3].item() == 1
    assert replay["n_entities"][2, 63, 3].item() == 2

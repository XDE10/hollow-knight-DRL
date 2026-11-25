from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch


def test_corrective_packaging_uses_the_existing_ppo_and_world_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    packing = importlib.import_module("package_sisters_corrective_ppo")
    trainer = importlib.import_module("train_sisters_recurrent_ppo")
    world = tmp_path / "world.pt"
    weights = trainer.World(193, 2).state_dict()
    torch.save({"actor_sha256": "offline", "horizons": trainer.HORIZONS, "world": [weights]}, world)
    world_sha = packing.sha256(world)
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "actor_sha256": "offline",
                "audit_sha256": "data",
                "world_sha256": world_sha,
                "measures": {
                    "equal_fight_means": {
                        "world_mae_xy": [[0.1, 0.1]],
                        "persistence_mae_xy": [[0.2, 0.2]],
                        "constant_velocity_mae_xy": [[0.2, 0.2]],
                        "shuffled_action_mae_xy": [[0.2, 0.2]],
                    }
                },
            }
        )
    )
    common = {
        "provenance": {"audit_sha256": "data", "hidden": 1},
        "catalogue": torch.tensor([[0, 0, 0], [1, 0, 0]]),
        "critic": {"weight": torch.tensor([3.0])},
    }
    wp5 = {
        **common,
        "state": {"weight": torch.tensor([1.0])},
        "ppo": {
            "parent_sha256": "wp5-earlier",
            "offline_sha256": "offline",
            "world_sha256": world_sha,
        },
    }
    wp6 = {
        **common,
        "state": {"weight": torch.tensor([2.0])},
        "corrective": {"parent_learner_sha256": "wp5"},
    }
    with pytest.raises(ValueError, match="share the frozen offline actor"):
        trainer.validate_behavior_lineage(wp6, "wp6", "offline", world_sha, "data")
    fitted = packing.package(wp6, "wp6", wp5, "wp5", "offline", world_sha, "data")
    assert fitted["ppo"] == wp5["ppo"]
    assert fitted["corrective"] == wp6["corrective"]
    assert fitted["ppo_resume"]["parent_sha256"] == "wp6"
    assert torch.equal(fitted["state"]["weight"], wp6["state"]["weight"])
    assert torch.equal(fitted["critic"]["weight"], wp5["critic"]["weight"])
    audit = trainer.validate_behavior_lineage(fitted, "packaged", "offline", world_sha, "data")
    _, mask = trainer.load_world(world, validation, "offline", audit, 193, 2)
    assert mask.all()
    with pytest.raises(ValueError, match="audit"):
        trainer.validate_behavior_lineage(fitted, "packaged", "offline", world_sha, "foreign-data")
    with pytest.raises(ValueError, match="share the frozen offline actor"):
        packing.package(wp6, "wp6", wp5, "wp5", "offline", "foreign-world", "data")
    with pytest.raises(ValueError, match="Critic differs"):
        packing.package(
            {**wp6, "critic": {"weight": torch.tensor([4.0])}},
            "wp6",
            wp5,
            "wp5",
            "offline",
            world_sha,
            "data",
        )

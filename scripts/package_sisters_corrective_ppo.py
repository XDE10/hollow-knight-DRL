"""Package corrective actor weights for the verified recurrent PPO lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from export_sisters_encoded_recurrent import export
from train_hkrl_wm_recurrent_rl import sha256
from train_sisters_recurrent_ppo import load_world, validate_behavior_lineage


def package(
    corrective: dict, corrected_sha: str, wp5: dict, wp5_sha: str,
    offline_sha: str, world_sha: str, audit_sha: str,
) -> dict:
    validate_behavior_lineage(wp5, wp5_sha, offline_sha, world_sha, audit_sha)
    if (
        corrective.get("ppo") is not None
        or corrective.get("corrective", {}).get("parent_learner_sha256") != wp5_sha
        or corrective["provenance"] != wp5["provenance"]
        or not torch.equal(corrective["catalogue"], wp5["catalogue"])
    ):
        raise ValueError("Corrective checkpoint does not descend from this verified PPO checkpoint")
    critic, previous = corrective["critic"], wp5["critic"]
    if critic.keys() != previous.keys() or any(not torch.equal(v, previous[k]) for k, v in critic.items()):
        raise ValueError("Corrective Critic differs from its PPO parent")
    packaged = {
        **corrective,
        "ppo": dict(wp5["ppo"]),
        "ppo_resume": {
            "parent_sha256": corrected_sha,
            "previous_ppo_checkpoint_sha256": wp5_sha,
            "optimizer_initialization": "fresh AdamW; no optimizer moments carried forward",
        },
    }
    validate_behavior_lineage(packaged, corrected_sha, offline_sha, world_sha, audit_sha)
    return packaged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("corrective", "previous", "native", "offline", "world",
                  "world-validation", "actor-revalidation", "training-audit", "opening", "output"):
        parser.add_argument("--" + label, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = vars(args).copy()
    paths.pop("output")
    shas = {key: sha256(path) for key, path in paths.items()}
    corrected = torch.load(args.corrective, map_location="cpu", weights_only=True)
    parent = torch.load(args.previous, map_location="cpu", weights_only=True)
    old_actor = torch.load(args.offline, map_location="cpu", weights_only=True)
    if not torch.equal(corrected["catalogue"], old_actor["catalogue"]):
        raise ValueError("Frozen offline action catalogue differs")
    wm = json.loads(args.actor_revalidation.read_bytes())
    if wm["world_sha256"] != shas["world"] or wm["actor_sha256"] != {
        "wp5": shas["previous"], "corrective": shas["corrective"]
    }:
        raise ValueError("Corrective world-model evidence names different weights")
    candidate = package(corrected, shas["corrective"], parent, shas["previous"],
                        shas["offline"], shas["world"], shas["training_audit"])
    catalogue = candidate["catalogue"]
    _, supported = load_world(args.world, args.world_validation, shas["offline"],
                              shas["training_audit"], 192 + candidate["provenance"]["hidden"], len(catalogue))
    if export(args.opening.read_bytes(), catalogue, candidate["state"], continuous=True) != args.native.read_bytes():
        raise ValueError("WP6 native actor does not match its unmodified learner tensors")
    args.output.mkdir(parents=True, exist_ok=False)
    dst = args.output / "learner.pt"
    torch.save(candidate, dst)
    reread = torch.load(dst, map_location="cpu", weights_only=True)
    for name in ("state", "critic"):
        if reread[name].keys() != corrected[name].keys() or any(
            not torch.equal(value, reread[name][key]) for key, value in corrected[name].items()
        ):
            raise ValueError(f"Packaging modified {name} tensors")
    validate_behavior_lineage(reread, sha256(dst), shas["offline"], shas["world"], shas["training_audit"])
    receipt = {
        "inputs": {key: {"path": path.as_posix(), "sha256": shas[key]} for key, path in paths.items()},
        "learner": {"path": dst.as_posix(), "sha256": sha256(dst)},
        "native": {"path": args.native.as_posix(), "sha256": shas["native"]},
        "actor_critic_tensors_equal": True,
        "native_reexport_byte_identical": True,
        "inherited_ppo": reread["ppo"],
        "resume": reread["ppo_resume"],
        "corrective": reread["corrective"],
        "original_world_validated_components": supported.tolist(),
    }
    (args.output / "packaging.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()


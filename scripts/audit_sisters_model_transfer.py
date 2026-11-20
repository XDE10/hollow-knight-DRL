"""Compare the frozen simulator with real on-policy rollouts from the same action interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sisters_latent_ac import CATALOG, WEIGHTS, Dynamics, action_vector, keys


@torch.inference_mode()
def audit(dynamics_path: Path, rollout_dir: Path) -> dict:
    torch.set_num_threads(1)
    weights = torch.tensor(WEIGHTS)
    actions = torch.tensor(keys(np.asarray(CATALOG)))
    payload = torch.load(dynamics_path, weights_only=True, map_location="cpu")
    models = []
    for state_dict in payload["models"]:
        model = Dynamics(state_dict["delta.weight"].shape[0])
        model.load_state_dict(state_dict)
        models.append(model.eval())
    bias = torch.stack(payload["biases"])
    windows = {1: [], 5: []}
    rewards = []
    fights = 0
    for path in sorted(rollout_dir.glob("rollout-*.pt")):
        record = torch.load(path, weights_only=True, map_location="cpu")
        batch = record["batch"]
        if sum(record["lengths"]) != len(batch["state"]):
            raise ValueError(f"Broken episode boundaries in {path}")
        cmd = action_vector(actions, batch["action"])
        logits = torch.stack([model(batch["state"], cmd)[1] for model in models])
        mean_reward = (((logits - bias[:, None]).sigmoid()) @ weights).mean(0)
        nail = cmd[:, 7] > 0.5
        rewards.append(torch.stack([mean_reward, batch["reward"], nail.float()]))
        start = 0
        for length in record["lengths"]:
            stop = start + length
            for horizon, chunks in windows.items():
                # The last state in a fight is a decision, not an observation after termination.
                if length <= horizon:
                    continue
                initial = batch["state"][start : stop - horizon]
                target = batch["state"][start + horizon : stop]
                forecasts = []
                for model in models:
                    predicted = initial
                    for offset in range(horizon):
                        predicted, _ = model(
                            predicted, cmd[start + offset : stop - horizon + offset]
                        )
                    forecasts.append((predicted - target).abs().mean(1))
                chunks.append(
                    torch.stack([*forecasts, (initial - target).abs().mean(1)])
                )
            start = stop
            fights += 1
    if not rewards:
        raise ValueError("No recorded on-policy rollouts")
    pred, actual, nail = torch.cat(rewards, 1)
    result = {"fights": fights, "decisions": len(actual), "reward": {}, "state": {}}
    for name, mask in (
        ("all", torch.ones_like(nail, dtype=torch.bool)),
        ("nail", nail.bool()),
        ("other", ~nail.bool()),
    ):
        if mask.any():
            result["reward"][name] = {
                "n": int(mask.sum()),
                "predicted": float(pred[mask].mean()),
                "observed": float(actual[mask].mean()),
            }
    for horizon, chunks in windows.items():
        if not chunks:
            result["state"][str(horizon)] = {
                "windows": 0,
                "member_mae": [],
                "unchanged_mae": None,
            }
            continue
        values = torch.cat(chunks, 1)
        result["state"][str(horizon)] = {
            "windows": values.shape[1],
            "member_mae": values[:-1].mean(1).tolist(),
            "unchanged_mae": float(values[-1].mean()),
        }
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dynamics", type=Path, required=True)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.dynamics, args.rollouts)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

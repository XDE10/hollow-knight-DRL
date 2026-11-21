"""Improve a frozen Sisters policy using randomized actions in genuine development fights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sisters_online_choices import choice_outcomes, late_catalogue
from torch import Tensor
from torch.nn import functional as F
from train_sisters_native_iql import REWARD_WEIGHTS
from train_sisters_timing_iql import (
    TimingPolicy,
    mlp,
    motion_history,
    movement_groups,
    phase_checkpoint,
    phase_components,
    save_native,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trials(
    directory: Path,
    parent_sha: str,
    actor: TimingPolicy,
    value: torch.nn.Module,
    catalogue: Tensor,
) -> tuple[dict[str, tuple[Tensor, ...]], dict[str, int]]:
    parts: dict[str, list[tuple[Tensor, ...]]] = {"train": [], "holdout": []}
    counts = {"train": 0, "holdout": 0}
    groups = movement_groups(catalogue)
    games = json.loads((directory / "manifest.json").read_bytes())
    if len({g["source"] for g in games}) != len(games):
        raise ValueError("Repeated online development fight")
    with torch.no_grad():
        for g in games:
            if g["behavior_policy_sha256"] != parent_sha or not g["source"].startswith(
                "native-development/"
            ):
                raise ValueError(
                    "Expected only the selected actor's development fights"
                )
            path = directory / g["file"]
            if sha(path) != g["sha256"]:
                raise ValueError("Online replay changed")
            with np.load(path, allow_pickle=False) as f:
                if "choice_hold_stop" in f:
                    raise ValueError("This estimator expects single-decision interventions")
                s, t, rows = f["timing_state"], f["control_time"], f["choice_rows"]
                if actor.past is not None:
                    s = motion_history(t, s)
                k, reward, nxt, discount = choice_outcomes(
                    t, f["time"], f["elapsed_ticks"], f["events"], rows, REWARD_WEIGHTS
                )
                if not len(k):
                    continue
                row = rows[k]
                pair = torch.tensor(f["choice_pairs"][k]).long()
                arm = torch.tensor(f["choice_draw"][k]).long()
                probability = torch.tensor(f["choice_probability"][k])
                if not torch.equal(
                    catalogue[pair[torch.arange(len(k)), arm]],
                    torch.tensor(f["controls"][row]),
                ):
                    raise ValueError(
                        "Reference learner did not issue the logged command"
                    )
                x = torch.from_numpy(s[row])
                hidden = actor.encode(x)
                logits = actor.heads(hidden)[0]
                gap = logits.gather(1, pair).diff(dim=1).neg().flatten()
                alternatives = logits.masked_fill(
                    groups[None, :] == groups[pair[:, :1]], -torch.inf
                )
                if (
                    not torch.allclose(
                        gap, torch.tensor(f["choice_gap"][k]), rtol=1e-4, atol=0.003
                    )
                    or (
                        logits.max(1).values - logits.gather(1, pair[:, :1]).flatten()
                        > 0.003
                    ).any()
                    or not torch.allclose(
                        alternatives.max(1).values,
                        logits.gather(1, pair[:, 1:]).flatten(),
                        atol=0.003,
                        rtol=1e-4,
                    )
                    or not torch.all(probability == 0.5)
                ):
                    raise ValueError(
                        "Online coin flip or reference logits differ from the game ledger"
                    )
                # The existing world-assisted IQL value supplies a discounted continuation.
                # Both experimental arms share the same state-only baseline.
                payoff = torch.tensor(reward) + 0.5 * (
                    torch.tensor(discount) * value(torch.from_numpy(s[nxt])).flatten()
                    - value(x).flatten()
                )
                split = g["split"]
                fight = torch.full((len(k),), counts[split], dtype=torch.long)
                parts[split].append(
                    (hidden, logits, pair, arm, payoff.clamp(-6, 6), probability, fight)
                )
                counts[split] += 1
    if min(counts.values()) < 2:
        raise ValueError("Need independent randomized fights in both splits")
    return {
        split: tuple(torch.cat(v) for v in zip(*items, strict=True))
        for split, items in parts.items()
    }, counts


def realized_gain(
    new: Tensor, old: Tensor, arm: Tensor, payoff: Tensor, prob: Tensor
) -> Tensor:
    """Inverse-propensity estimate of the difference between two action distributions."""
    j = torch.arange(len(arm))
    return (new[j, arm] - old[j, arm]) * payoff / prob


def fit(
    directory: Path,
    parent: Path,
    learner: Path,
    output: Path,
    seed: int = 635,
) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    native = parent.read_bytes()
    opener, _ = phase_components(native)
    saved = torch.load(learner, map_location="cpu", weights_only=True)
    cat = saved["catalogue"].long()
    if not np.array_equal(late_catalogue(native), cat.numpy()):
        raise ValueError("Mismatched action catalogue")
    actor = (
        TimingPolicy(cat, "past.weight" in saved["actor"]).eval().requires_grad_(False)
    )
    actor.load_state_dict(saved["actor"])
    value = mlp(saved["value"]["0.weight"].shape[1], 1).eval().requires_grad_(False)
    value.load_state_dict(saved["value"])
    data, fights = read_trials(directory, sha(parent), actor, value, cat)
    hidden, logits, pair, arm, payoff, probability, _ = data["train"]
    held_h, held_logits, held_pair, held_arm, held_pay, held_p, held_fight = data[
        "holdout"
    ]
    groups = movement_groups(cat)
    mean = hidden.double().mean(0).float()
    centered = hidden - mean
    # Direct SVD avoids squaring the condition number for repeated arena states.
    _, singular, axes = np.linalg.svd(centered.double().numpy(), full_matrices=False)
    scale = np.maximum(singular[:8] / np.sqrt(len(centered)), 0.1)
    projection = torch.from_numpy((axes[:8].T / scale).copy()).float()
    basis = centered @ projection
    held_basis = (held_h - mean) @ projection
    theta = torch.zeros((basis.shape[1], 12), requires_grad=True)
    bias = torch.zeros(12, requires_grad=True)
    opt = torch.optim.Adam((theta, bias), lr=0.015)
    old = logits.gather(1, pair).softmax(1)
    old_all = logits.softmax(1)
    for _ in range(160):
        updated = logits + (basis @ theta + bias)[:, groups]
        new = updated.gather(1, pair).softmax(1)
        advantage = realized_gain(new, old, arm, payoff, probability).mean()
        anchor = F.kl_div(updated.log_softmax(1), old_all, reduction="batchmean")
        loss = (
            -advantage
            + 0.20 * anchor
            + 0.01 * (theta.square().sum() + bias.square().sum())
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        held_old = F.one_hot(torch.zeros_like(held_arm), 2).float()
        adjustment = (held_basis @ theta + bias)[:, groups]
        estimates = []
        for scale in (0.0, 0.5, 1.0):
            score = held_logits + scale * adjustment
            action = score.argmax(1) if scale else held_pair[:, 0]
            supported = (held_pair == action[:, None]).any(1)
            prob = F.one_hot((action == held_pair[:, 1]).long(), 2).float()
            gain = realized_gain(prob, held_old, held_arm, held_pay, held_p)
            # Both arm returns were clipped to [-6, 6]. Outside this pair the
            # randomized data cannot identify the new action's return.
            gain = torch.where(supported, gain, -12.0)
            fight_gains = [
                gain[held_fight == i].mean().item() for i in range(fights["holdout"])
            ]
            estimates.append(
                {
                    "scale": scale,
                    "mean_fight_gain": float(np.mean(fight_gains)),
                    "fight_gains": fight_gains,
                    "changed_fraction": (action != held_pair[:, 0])
                    .float()
                    .mean()
                    .item(),
                    "unsupported_fraction": (~supported).float().mean().item(),
                }
            )
        eligible = [
            x
            for x in estimates
            if x["scale"] == 0
            or (
                0 < x["changed_fraction"] <= 0.15
                and sum(g > 0 for g in x["fight_gains"]) > len(x["fight_gains"]) / 2
            )
        ]
        selected = max(eligible, key=lambda x: x["mean_fight_gain"])
        scale = selected["scale"]
        result = {
            "method": "randomized pair policy gradient with world-assisted IQL continuation",
            "parent_sha256": sha(parent),
            "learner_sha256": sha(learner),
            "fights": fights,
            "trials": {"train": len(hidden), "holdout": len(held_h)},
            "horizon_game_seconds": 0.60,
            "selected": selected,
            "heldout_estimates": estimates,
            "opener_sha256": hashlib.sha256(opener).hexdigest(),
            "seed": seed,
            "return_clip": 6,
        }
        output.mkdir(parents=True, exist_ok=False)
        if scale:
            delta = projection @ theta
            actor.out.weight.add_(scale * delta[:, groups].T)
            actor.out.bias.add_(scale * (bias - mean @ delta)[groups])
            if not torch.allclose(
                actor.heads(held_h)[0],
                held_logits + scale * adjustment,
                atol=0.002,
                rtol=1e-4,
            ):
                raise ValueError(
                    "Folded native action head changed the learned correction"
                )
            path = output / "world_iql.bin"
            save_native(actor, cat.numpy(), path)
            path.write_bytes(phase_checkpoint(opener, path.read_bytes()))
        else:
            (output / "world_iql.bin").write_bytes(native)
        saved.update(actor=actor.state_dict(), opener_native=opener)
        saved.pop("late_native", None)
        torch.save(saved, output / "learner.pt")
        (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "parent", "learner", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(fit(args.data, args.parent, args.learner, args.output)), flush=True
    )

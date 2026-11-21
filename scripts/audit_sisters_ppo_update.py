"""Replay a completed primitive PPO fit and locate the evidence for changed greedy keys."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from replay_sisters_primitive_heads import models
from torch.nn import functional as F
from torch.optim.optimizer import register_optimizer_step_post_hook
from train_sisters_champion_router import digest
from train_sisters_primitive_heads import Outcome
from train_sisters_primitive_ppo import distribution, optimize, read_games
from train_sisters_timing_iql import option_components, phase_components


def capture_actor_step(rollout, ordinary, net, temperature, smooth, trial, optimizer, args, kwargs):
    if type(optimizer) is torch.optim.Adam:
        weight, bias = optimizer.param_groups[0]["params"]
        trial.append(measure(rollout, ordinary, net, weight, bias, temperature, smooth=smooth))


@torch.no_grad()
def measure(rollout, ordinary, net, weight, bias, temperature, *, smooth=False):
    x, allowed, chosen, initial_p, taken_p, advantage, _ = rollout
    logits = net.out(x)
    greedy = logits.argmax(1)
    updated = logits + F.linear(x, weight, bias)
    new_greedy = updated.argmax(1)
    moved = new_greedy != greedy
    explored = chosen != greedy
    new_taken = moved & (chosen == new_greedy)
    centered = advantage - advantage.mean()
    p = distribution(updated, allowed, temperature, smooth=smooth)
    ratio = p[torch.arange(len(x)), chosen] / taken_p.float()
    clipped = ((centered > 0) & (ratio > 1.15)) | ((centered < 0) & (ratio < 0.85))
    initial_p = initial_p.float()
    kl = (initial_p * (initial_p.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(1)
    safe = net.out(ordinary)
    changed_safe = (safe + F.linear(ordinary, weight, bias)).argmax(1) != safe.argmax(1)
    return {
        "flipped": int(moved.sum()),
        "new_key_sampled_positive_advantage": int((new_taken & (centered > 0)).sum()),
        "new_key_sampled_nonpositive_advantage": int((new_taken & (centered <= 0)).sum()),
        "new_key_not_sampled_at_this_state": int((moved & (chosen != new_greedy)).sum()),
        "old_key_sampled_negative_advantage": int((moved & ~explored & (centered < 0)).sum()),
        "exploratory_inputs": int(explored.sum()),
        "clipped_exploratory_inputs": int((explored & clipped).sum()),
        "kl": float(kl.mean()),
        "flipped_keys_share_of_kl": float(kl[moved].sum() / kl.sum().clamp_min(1e-12)),
        "ordinary_inputs_changed_fraction": float(changed_safe.float().mean()),
    }


def audit(
    parent: Path, behavior: Path, learner: Path, data: Path, trained: Path, output: Path
) -> None:
    fit = json.loads((trained / "training.json").read_bytes())
    original = parent.read_bytes()
    before = behavior.read_bytes()
    after = (trained / "world_iql.bin").read_bytes()
    for item, blob in (
        ("parent", original),
        ("behavior", before),
        ("policy", after),
        ("rollout", data.read_bytes()),
        ("critic_input", learner.read_bytes()),
    ):
        if fit[f"{item}_sha256"] != digest(blob):
            raise ValueError(f"Incorrect {item} for the saved update")
    torch.set_num_threads(1)
    torch.manual_seed(23670)
    nets, parents, cats, gate = models(original, before)
    critics = [Outcome(len(cat)) for cat in cats]
    saved = torch.load(learner, map_location="cpu", weights_only=True)
    for critic, name in zip(critics, ("primary", "alternate"), strict=True):
        critic.load_state_dict(saved[name]["critic"])
    temp = fit["temperature"]
    smooth = fit.get("smooth_key_sampling", False)
    rows, ordinary, _ = read_games(
        data, digest(before), nets, cats, gate, critics, temp, fit["clean_cost"], smooth=smooth
    )
    input_leaves = option_components(phase_components(before)[1])[:2]
    output_leaves = option_components(phase_components(after)[1])[:2]
    report = {}
    for i, name in enumerate(("primary", "alternate")):
        trial = []

        handle = register_optimizer_step_post_hook(
            partial(capture_actor_step, rows[i], ordinary[i], nets[i], temp, smooth, trial)
        )
        try:
            raw, _, stats = optimize(
                input_leaves[i],
                nets[i],
                parents[i],
                critics[i],
                rows[i],
                ordinary[i],
                temp,
                smooth=smooth,
            )
        finally:
            handle.remove()
        if raw != output_leaves[i] or stats != fit[name]:
            raise ValueError(f"Fit did not reproduce the saved {name} key head")
        n = stats["accepted_steps"]
        rejected = trial[n] if n < len(trial) else None
        if len(trial) != n + int(rejected is not None):
            raise ValueError("Unexpected number of attempted steps")
        report[name] = {"accepted_steps": n, "accepted": trial[:n], "rejected": rejected}
    output.write_text(
        json.dumps(
            {
                "scope": (
                    "Replay of the original optimizer. Unchosen keys can generalize across states. "
                    "The replay measures changes in model decisions on recorded game states."
                ),
                "policy_sha256": digest(after),
                "rollout_sha256": fit["rollout_sha256"],
                "heads": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("parent", "behavior", "learner", "data", "trained", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    audit(**vars(parser.parse_args()))

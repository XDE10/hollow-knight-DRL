"""Fit and validate action-conditioned world rollouts from a frozen recurrent IQL actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from train_hkrl_wm_recurrent_rl import (
    Fight,
    RecurrentActor,
    read_fights,
    recurrent_features,
    sha256,
)
from train_sisters_timing_iql import motion_history

STEPS = 9
HORIZONS = (2, 5, 9)
VELOCITY_SCALE = (1.5, 2.5)
POSITION_SCALE = (10, 8)


class World(nn.Module):
    def __init__(self, latent: int, actions: int) -> None:
        super().__init__()
        self.start = nn.Sequential(nn.Linear(latent + 6, 128), nn.Tanh())
        self.action = nn.Embedding(actions, 24)
        self.dynamics = nn.GRU(24, 128, batch_first=True)
        self.head = nn.Linear(128, 6)

    def forward(self, context: Tensor, hero: Tensor, keys: Tensor) -> Tensor:
        state = self.start(torch.cat((context, hero), dim=-1))[None]
        predicted, _ = self.dynamics(self.action(keys), state)
        predicted = self.head(predicted[:, [i - 1 for i in HORIZONS]])
        return predicted


@torch.inference_mode()
def build_game(actor: RecurrentActor, fight: Fight, stride: int) -> tuple[Tensor, ...] | None:
    tape = fight.episode
    blocks = []
    indices = np.asarray(HORIZONS)
    for lo, hi in fight.segments:
        if hi - lo <= STEPS:
            continue
        rows = np.arange(lo, hi)
        exact = (np.abs(np.diff(tape.clock[rows]) - 0.04) < 1e-4) & (fight.action[rows[:-1]] >= 0)
        broken = np.r_[0, np.cumsum(~exact)]
        starts = np.arange(0, len(rows) - STEPS)
        starts = starts[broken[starts] == broken[starts + STEPS]][::stride]
        if not len(starts):
            continue
        window = starts[:, None] + np.arange(STEPS)
        future = starts[:, None] + indices
        x = torch.from_numpy(motion_history(tape.clock, tape.observation, rows))[None]
        embedded, remembered = recurrent_features(actor.encoder, actor.memory, x, 0)
        latent = torch.cat((embedded[0, starts], remembered[0, starts]), dim=-1)
        hero = tape.observation[rows, :6]
        events = np.concatenate(
            (tape.events[rows, :1], tape.events[rows, 1:5].sum(1, keepdims=True)), axis=1
        )
        prefix = np.concatenate((np.zeros((1, 2), dtype="f4"), np.cumsum(events, axis=0)))
        scores = prefix[future] - prefix[starts, None]
        physical = hero[future, :4] - hero[starts, None, :4]
        blocks.append(
            (
                latent,
                torch.from_numpy(hero[starts]),
                torch.from_numpy(fight.action[rows[window]]),
                torch.from_numpy(physical),
                torch.from_numpy(scores),
            )
        )
    if not blocks:
        return None
    # Outside inference mode, these tensors can participate in a gradient-bearing forward pass.
    return tuple(torch.cat(x) for x in zip(*blocks, strict=True))


def fit(
    models: list[World], train: dict[str, tuple[Tensor, ...]], updates: int, seed: int
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    populations = [list(train.values()) for _ in models]
    membership = [
        rng.choice(len(train), max(1, round(0.85 * len(train))), replace=False) for _ in models
    ]
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001) for model in models]
    progress = []
    for step in range(1, updates + 1):
        losses = []
        for model, optimizer, pool, members in zip(
            models, optimizers, populations, membership, strict=True
        ):
            rows = []
            for key in rng.choice(members, 8):
                episode = pool[key]
                index = rng.integers(len(episode[0]), size=16)
                rows.append(tuple(tensor[index] for tensor in episode))
            context, hero, actions, motion, events = (
                torch.cat(col) for col in zip(*rows, strict=True)
            )
            predicted = model(context, hero, actions)
            position = F.smooth_l1_loss(predicted[..., :2], motion[..., :2], beta=0.04)
            velocity = F.smooth_l1_loss(predicted[..., 2:4], motion[..., 2:4], beta=0.1)
            attack = F.smooth_l1_loss(predicted[..., 4], torch.log1p(events[..., 0]), beta=0.3)
            hit = F.binary_cross_entropy_with_logits(
                predicted[..., 5], (events[..., 1] > 0).float()
            )
            loss = position + 0.20 * velocity + 0.10 * attack + 0.10 * hit
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        if step % 100 == 0 or step == updates:
            row = {"step": step, "loss": float(np.mean(losses))}
            progress.append(row)
            print(json.dumps(row), flush=True)
    return progress


@torch.inference_mode()
def assess(
    models: list[World], holdout: dict[str, tuple[Tensor, ...]], hit_prior: Tensor
) -> dict[str, Any]:
    records = {}
    times = 0.04 * torch.tensor(HORIZONS)
    units = torch.tensor(POSITION_SCALE)
    for source, (latent, hero, actions, motion, events) in holdout.items():
        actual = torch.stack([model(latent, hero, actions) for model in models])
        permuted = torch.stack(
            [model(latent, hero, actions.roll(len(actions) // 2, 0)) for model in models]
        )
        if not torch.isfinite(actual).all() or not torch.isfinite(permuted).all():
            raise ValueError(f"Nonfinite world rollout for {source}")
        target = motion[..., :2]
        constant = hero[:, None, 2:4] * torch.tensor(VELOCITY_SCALE) * times[None, :, None]
        hit = (events[..., 1] > 0).float()
        probability = actual[..., 5].sigmoid().mean(0)
        record = {
            "windows": len(latent),
            "persistence_mae_xy": (target.abs() * units).mean(0).tolist(),
            "constant_velocity_mae_xy": ((target - constant).abs() * units).mean(0).tolist(),
            "world_mae_xy": ((target - actual[..., :2].mean(0)).abs() * units).mean(0).tolist(),
            "shuffled_action_mae_xy": ((target - permuted[..., :2].mean(0)).abs() * units)
            .mean(0)
            .tolist(),
            "ensemble_spread_xy": (actual[..., :2].std(0, unbiased=False) * units).mean(0).tolist(),
            "world_hit_brier": (probability - hit).square().mean(0).tolist(),
            "prior_hit_brier": (hit_prior - hit).square().mean(0).tolist(),
            "world_damage_mae": (actual[..., 4].mean(0).expm1().clamp_min(0) - events[..., 0])
            .abs()
            .mean(0)
            .tolist(),
            "zero_damage_mae": events[..., 0].mean(0).tolist(),
        }
        records[source] = record
    means = {
        k: np.mean([r[k] for r in records.values()], axis=0).tolist()
        for k in next(iter(records.values()))
        if k != "windows"
    }
    return {"equal_fight_means": means, "fights": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=750)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.updates < 1 or args.threads < 1:
        raise ValueError("Use positive update and thread counts")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(10576007)
    saved = torch.load(args.actor, map_location="cpu", weights_only=True)
    if saved["provenance"]["audit_sha256"] != sha256(args.audit):
        raise ValueError("World replay differs from the actor's training corpus")
    catalogue = saved["catalogue"]
    hidden = saved["provenance"]["hidden"]
    actor = RecurrentActor(catalogue, hidden).eval()
    actor.load_state_dict(saved["state"])
    groups = read_fights(args.archive, args.audit, catalogue)
    prepared = {}
    for split, fights in groups.items():
        prepared[split] = {}
        for source, fight in fights.items():
            batch = build_game(actor, fight, stride=4 if split == "train" else 1)
            if batch is not None:
                prepared[split][source] = tuple(x.clone() for x in batch)
    del groups
    sizes = {
        k: {"fights": len(v), "windows": sum(len(e[0]) for e in v.values())}
        for k, v in prepared.items()
    }
    print(json.dumps(sizes), flush=True)
    if not all(v["fights"] > 1 for v in sizes.values()):
        raise ValueError("Too few independent world model episodes")
    hit_prior = torch.stack(
        [(v[-1][..., 1] > 0).float().mean(0) for v in prepared["train"].values()]
    ).mean(0)
    models = [World(192 + hidden, len(catalogue)) for _ in range(3)]
    progress = fit(models, prepared["train"], args.updates, 10576007)
    report = assess([m.eval() for m in models], prepared["holdout"], hit_prior)
    path = args.output / "world.pt"
    actor_hash = sha256(args.actor)
    torch.save(
        {
            "world": [m.state_dict() for m in models],
            "actor_sha256": actor_hash,
            "horizons": HORIZONS,
        },
        path,
    )
    evidence = {
        "actor_sha256": actor_hash,
        "audit_sha256": sha256(args.audit),
        "world_sha256": sha256(path),
        "position_units": "game units",
        "horizons_seconds": [0.04 * h for h in HORIZONS],
        "splits": sizes,
        "measures": report,
        "progress": progress,
    }
    (args.output / "validation.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["equal_fight_means"]), flush=True)


if __name__ == "__main__":
    main()

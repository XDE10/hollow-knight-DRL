"""Fit the local movement head from completed native Sisters gameplay recordings."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

FEATURES = 56
HIDDEN = 32
ACTIONS = 12
MIRRORED = [0, 2, 10, 13, 15, 23, 25, 33, 35, 43, 45, 50, 52]


def read_trace(path: Path, episode: int) -> np.ndarray:
    with path.open("rb") as stream:
        magic, identity, features = struct.unpack("<IQi", stream.read(16))
        if (magic, identity, features) != (0x31534B48, episode, FEATURES):
            raise ValueError(f"Trace does not match episode {episode} at {path}")
        data = np.frombuffer(stream.read(), dtype="<f4")
    return data.reshape(-1, FEATURES + 4)


def episode_batches(report_path: Path) -> list[tuple[np.ndarray, ...]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = read_trace(report_path.with_suffix(".trace"), report["episode_id"])
    if len(rows) < 60:
        return []
    states = rows[:, 1 : FEATURES + 1]
    present = states[:, [12, 22, 32, 42, 49]].max(axis=1) > 0
    if not present.any():
        return []
    end = min(len(states), np.flatnonzero(present)[-1] + 3)
    states = states[:end].copy()
    times = rows[:end, 0].copy()
    moves = rows[:end, FEATURES + 1].astype(np.int64) + 1
    buttons = rows[:end, FEATURES + 3].astype(np.int64)
    actions = 4 * moves + 2 * ((buttons & 3) != 0) + ((buttons & 4) != 0)

    rewards = np.zeros(len(times), dtype=np.float32)
    with gzip.open(
        report_path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8"
    ) as stream:
        for line in stream:
            start, _, _, _, events, elapsed = json.loads(line)
            index = np.searchsorted(times, start + 0.02 * elapsed, side="right") - 1
            if index < 0:
                continue
            for kind, entity, amount, _ in events:
                if kind == "DAMAGE_DEALT":
                    rewards[index] += amount / 60
                elif kind == "DAMAGE_TAKEN":
                    rewards[index] -= 3 * amount
                elif kind == "BOSS_KILLED" and entity == 0:
                    rewards[index] += 8

    index = np.arange(end)
    next_index = np.minimum(
        end - 1, np.maximum(index + 1, np.searchsorted(times, times + 0.40))
    )
    cumulative = np.r_[0, np.cumsum(rewards)]
    future = np.minimum(end, np.searchsorted(times, times + 1.10))
    local_return = cumulative[future] - cumulative[index]
    weights = np.clip(np.exp(np.clip(local_return * 0.27, -2, 1.2)), 0.18, 3.2)
    weights *= 1.35 if report["won"] else 1.0
    returns = (
        cumulative[np.minimum(end, next_index + 1)]
        - cumulative[np.minimum(end, index + 1)]
    )
    bootstrap = (next_index < end - 1).astype(np.float32)
    mirrored = states.copy()
    mirrored[:, MIRRORED] *= -1
    batches = []
    for observations, labels in (
        (states, actions),
        (mirrored, (2 - moves) * 4 + actions % 4),
    ):
        batches.append(
            (
                observations,
                labels,
                weights,
                returns,
                observations[next_index],
                bootstrap,
            )
        )
    return batches


def network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(FEATURES, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, ACTIONS)
    )


def train(reports: list[Path], output: Path, steps: int, seed: int) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    batches = [batch for report in reports for batch in episode_batches(report)]
    if not batches:
        raise ValueError(
            "At least one complete trace with active entities is required."
        )
    data: list[Tensor] = [torch.from_numpy(np.concatenate(v)) for v in zip(*batches)]
    states, actions, weights, returns, future, bootstrap = data
    actor, target = network(), network()
    target.load_state_dict(actor.state_dict())
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.002)
    for step in range(steps):
        ids = torch.randint(len(states), (512,))
        q = actor(states[ids])
        imitation = (
            F.cross_entropy(q, actions[ids], reduction="none") * weights[ids]
        ).mean()
        with torch.no_grad():
            selected = actor(future[ids]).argmax(dim=1, keepdim=True)
            next_value = target(future[ids]).gather(1, selected).squeeze(1)
            expected = returns[ids] + 0.93 * bootstrap[ids] * next_value
        value = F.smooth_l1_loss(q.gather(1, actions[ids, None]).squeeze(1), expected)
        loss = imitation + (0.18 if step < steps // 2 else 0.6) * value
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 3.0)
        optimizer.step()
        if step % 40 == 0:
            target.load_state_dict(actor.state_dict())

    output.mkdir(parents=True, exist_ok=False)
    torch.save(actor.state_dict(), output / "head.pt")
    path = output / "head.weights"
    flat = np.concatenate(
        [param.detach().numpy().reshape(-1) for param in actor.parameters()]
    )
    np.savetxt(path, flat, fmt="%.9g")
    metadata = {
        "steps": steps,
        "seed": seed,
        "rows": len(states),
        "episodes": len(reports),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "training.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", nargs="+", type=Path, help="Ordered per-game JSON files"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=600, type=int)
    parser.add_argument("--seed", default=381, type=int)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    train(args.reports, args.output, args.steps, args.seed)


if __name__ == "__main__":
    main()

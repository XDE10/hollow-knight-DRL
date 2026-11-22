"""Train the offline recurrent Sisters actor and IQL values on locked real trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch
from audit_sisters_mainline_replay import Episode, iter_episodes
from torch import Tensor, nn
from torch.nn import functional as F
from train_sisters_timing_iql import (
    TimingPolicy,
    motion_history,
    option_components,
    phase_components,
)

GAP = np.float32(0.081)
BURN_IN = 16
UNROLL = 64
GAMMA = 0.995


class RecurrentActor(nn.Module):
    def __init__(self, catalogue: Tensor, hidden: int) -> None:
        super().__init__()
        self.encoder = TimingPolicy(catalogue, history=True)
        self.memory = nn.GRU(192, hidden, batch_first=True)
        self.memory_projection = nn.Linear(hidden, 192)

    def forward(self, inputs: Tensor, burn: int = 0) -> Tensor:
        features = recurrent_features(self.encoder, self.memory, inputs, burn)
        encoded, remembered = features
        return self.encoder.heads(encoded + self.memory_projection(remembered))[0]


class RecurrentValues(nn.Module):
    def __init__(self, catalogue: Tensor, hidden: int) -> None:
        super().__init__()
        self.encoder = TimingPolicy(catalogue, history=True)
        self.encoder.out = nn.Identity()
        self.encoder.factors = nn.Identity()
        self.memory = nn.GRU(192, hidden, batch_first=True)
        width = 192 + hidden
        self.q1 = nn.Linear(width, len(catalogue))
        self.q2 = nn.Linear(width, len(catalogue))
        self.value = nn.Linear(width, 1)

    def forward(self, inputs: Tensor, burn: int = 0) -> tuple[Tensor, Tensor, Tensor]:
        features = torch.cat(recurrent_features(self.encoder, self.memory, inputs, burn), dim=-1)
        return self.q1(features), self.q2(features), self.value(features).squeeze(-1)


def recurrent_features(
    encoder: TimingPolicy,
    memory: nn.GRU,
    inputs: Tensor,
    burn: int,
) -> tuple[Tensor, Tensor]:
    state = None
    if burn:
        with torch.no_grad():
            prefix = encoder.encode(inputs[:, :burn].flatten(0, 1)).unflatten(
                0, (len(inputs), burn)
            )
            _, state = memory(prefix)
    learning = inputs[:, burn:]
    encoded = encoder.encode(learning.flatten(0, 1)).unflatten(0, learning.shape[:2])
    remembered, _ = memory(encoded, state)
    return encoded, remembered


@dataclass(frozen=True)
class Fight:
    episode: Episode
    action: np.ndarray
    train_rows: np.ndarray
    teacher: np.ndarray
    discount: np.ndarray
    segments: tuple[tuple[int, int], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_catalogue(parent: Path) -> Tensor:
    _, trio = phase_components(parent.read_bytes())
    left, right, _ = option_components(trio)
    catalogues = []
    for leaf in (left, right):
        n = struct.unpack_from("<i", leaf, 16)[0]
        catalogues.append(np.frombuffer(leaf, "<i4", count=n * 3, offset=20).reshape(n, 3))
    if not np.array_equal(*catalogues):
        raise ValueError("Frozen Trio action catalogues differ")
    return torch.from_numpy(catalogues[0].astype("i8"))


def prepare_fight(episode: Episode, lookup: dict[tuple[int, ...], int]) -> Fight:
    action = np.array([lookup.get(tuple(key), -1) for key in episode.action], dtype="i8")
    phase = episode.observation[:, 23] > 0.5
    delta = np.diff(episode.clock)
    boundaries = np.r_[0, np.flatnonzero((delta > GAP) | (phase[1:] != phase[:-1])) + 1, len(phase)]
    segments = tuple((int(lo), int(hi)) for lo, hi in pairwise(boundaries) if phase[lo])
    discount = np.zeros(len(phase), dtype="f4")
    has_next = (episode.next_row[:-1] >= 0) & (delta <= GAP) & phase[:-1] & phase[1:]
    discount[:-1][has_next] = GAMMA ** (delta[has_next] / 0.04)
    valid = (episode.terminated | (discount > 0)) & (action >= 0) & phase
    return Fight(episode, action, valid, episode.controller == "teacher", discount, segments)


def read_fights(archive: Path, audit: Path, catalogue: Tensor) -> dict[str, dict[str, Fight]]:
    report = json.loads(audit.read_bytes())
    if sha256(archive) != report["archive_sha256"]:
        raise ValueError("Replay archive differs from the verified report")
    expected = {row["provenance"]["source"]: row for row in report["episodes"]}
    reserved = {
        int(source.split("/")[1])
        for source, row in expected.items()
        if row.get("excluded") == "reserved acceptance seed"
    }
    lookup = {tuple(action): i for i, action in enumerate(catalogue.tolist())}
    groups: dict[str, dict[str, Fight]] = {"train": {}, "holdout": {}}
    with ZipFile(archive) as data:
        for item, episode in iter_episodes(data, reserved):
            source = item["source"]
            audited = expected.pop(source)
            if item != audited["provenance"]:
                raise ValueError(f"Recorded provenance changed for {source}")
            if isinstance(episode, str):
                if audited.get("excluded") != episode:
                    raise ValueError(f"Replay eligibility changed for {source}")
                continue
            if (
                "rows" not in audited
                or int(episode.valid_transition.sum()) != audited["rows"]["valid_transitions"]
            ):
                raise ValueError(f"Transition count changed for {source}")
            groups[item["split"]][source] = prepare_fight(episode, lookup)
    if expected or any(
        len(fights) != report["splits"][name]["episodes"] for name, fights in groups.items()
    ):
        raise ValueError("Replay fights differ from the audited splits")
    return groups


class Windows:
    def __init__(self, fights: list[Fight], rng: np.random.Generator) -> None:
        self.rng = rng
        self.pools: dict[tuple[bool, int], list[tuple[Fight, int, int]]] = {}
        for teacher in (True, False):
            for burn in (0, BURN_IN):
                pool = []
                for fight in fights:
                    valid = fight.train_rows & fight.teacher if teacher else fight.train_rows
                    for lo, hi in fight.segments:
                        if burn:
                            if valid[lo + burn : hi].any():
                                pool.append((fight, lo, hi))
                        elif valid[lo : min(hi, lo + UNROLL)].any():
                            pool.append((fight, lo, hi))
                if not pool:
                    raise ValueError("No eligible recurrent windows")
                self.pools[teacher, burn] = pool

    def batch(self, n: int, burn: int, teacher_only: bool) -> tuple[Tensor, ...]:
        rows = []
        for i in range(n):
            teacher = teacher_only or i < n // 2
            pool = self.pools[teacher, burn]
            fight, lo, hi = pool[int(self.rng.integers(len(pool)))]
            if burn:
                valid = fight.train_rows[lo + burn : hi]
                if teacher:
                    valid = valid & fight.teacher[lo + burn : hi]
                anchor = lo + burn + int(self.rng.choice(np.flatnonzero(valid)))
                start = max(lo, anchor - burn - int(self.rng.integers(UNROLL)))
            else:
                start = lo
            indices = np.arange(start, start + burn + UNROLL + 1)
            clipped = np.minimum(indices, hi - 1)
            scored = clipped[burn:-1]
            mask = (indices[burn:-1] < hi) & fight.train_rows[scored]
            tape = fight.episode
            rows.append(
                (
                    motion_history(tape.clock, tape.observation, clipped),
                    fight.action[scored].clip(0),
                    tape.reward[scored],
                    fight.discount[scored],
                    mask,
                    fight.teacher[scored] & mask,
                )
            )
        return tuple(torch.from_numpy(np.stack(column)) for column in zip(*rows, strict=True))


def iql_losses(
    actor: RecurrentActor,
    values: RecurrentValues,
    target: RecurrentValues,
    batch: tuple[Tensor, ...],
    burn: int,
) -> tuple[Tensor, dict[str, float]]:
    x, action, reward, discount, mask, teacher = batch
    logits = actor(x[:, :-1], burn)
    log_prob = logits.log_softmax(-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
    q1, q2, value = values(x, burn)
    with torch.no_grad():
        old_q1, old_q2, _ = target(x[:, :-1], burn)
        low_q = torch.minimum(old_q1, old_q2).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        td = reward + discount * value[:, 1:]
    gap = low_q - value[:, :-1]
    v_loss = (torch.where(gap >= 0, 0.7, 0.3) * gap.square())[mask].mean()
    picked = [q[:, :-1].gather(-1, action.unsqueeze(-1)).squeeze(-1) for q in (q1, q2)]
    q_loss = sum(F.smooth_l1_loss(q[mask], td[mask]) for q in picked)
    weight = (3.0 * gap.detach()[mask]).clamp(-12.0, float(np.log(20.0))).exp()
    awr = -(weight * log_prob[mask]).sum() / weight.sum().clamp_min(1e-6)
    bc = -log_prob[teacher].mean() if teacher.any() else log_prob.new_zeros(())
    total = q_loss + v_loss + awr + 0.5 * bc
    return total, {
        name: float(part.detach())
        for name, part in (("q", q_loss), ("v", v_loss), ("awr", awr), ("teacher_bc", bc))
    }


@torch.inference_mode()
def holdout(
    actor: RecurrentActor, values: RecurrentValues, fights: dict[str, Fight], device: torch.device
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for source, fight in fights.items():
        tape = fight.episode
        report: dict[str, Any] = {}
        score = {name: np.zeros(3) for name in ("teacher", "student", "unknown")}
        value_sum = value_square = 0.0
        count = 0
        for lo, hi in fight.segments:
            rows = np.arange(lo, hi)
            x = torch.from_numpy(motion_history(tape.clock, tape.observation, rows)).to(device)[
                None
            ]
            logits = actor(x)[0].cpu()
            value = values(x)[2][0].cpu().numpy()[fight.train_rows[rows]]
            if not torch.isfinite(logits).all() or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite held-out policy or value in {source}")
            log_prob = logits.log_softmax(-1)
            for key in score:
                mask = (tape.controller[rows] == key) & fight.train_rows[rows]
                actions = fight.action[rows[mask]]
                score[key] += (
                    len(actions),
                    (actions == logits[mask].argmax(-1).numpy()).sum(),
                    float(-log_prob[mask, actions].sum()),
                )
            count += len(value)
            value_sum += value.sum(dtype="f8")
            value_square += np.square(value, dtype="f8").sum()
        for key, (n, matches, loss) in score.items():
            if n:
                report[key] = {"rows": int(n), "top1": matches / n, "nll": loss / n}
        if count:
            mean = value_sum / count
            report["value_mean"] = mean
            report["value_std"] = np.sqrt(max(0.0, value_square / count - mean * mean))
        records[source] = report
    means = {}
    for key in ("teacher", "student", "unknown"):
        group = [row[key] for row in records.values() if key in row]
        if group:
            means[key] = {
                metric: float(np.mean([row[metric] for row in group])) for metric in ("top1", "nll")
            }
    return {"equal_fight_means": means, "fights": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Provides only the frozen opening and 75-key catalogue",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--warmup", type=int, default=384)
    parser.add_argument("--updates", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10576000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.warmup < 1 or args.updates < 1 or args.batch_size < 2 or args.threads < 1:
        raise ValueError("Use positive update and thread counts and at least two windows per batch")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    catalogue = native_catalogue(args.parent)
    groups = read_fights(args.archive, args.audit, catalogue)
    print(json.dumps({name: len(fights) for name, fights in groups.items()}), flush=True)
    windows = Windows(list(groups["train"].values()), np.random.default_rng(args.seed))
    actor = RecurrentActor(catalogue, args.hidden).to(device)
    values = RecurrentValues(catalogue, args.hidden).to(device)
    target = deepcopy(values).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [{"params": actor.parameters(), "lr": 3e-4}, {"params": values.parameters(), "lr": 3e-4}]
    )
    parameters = [*actor.parameters(), *values.parameters()]
    provenance = {
        "method": "recurrent-iql-v2",
        "randomly_initialized_trio": True,
        "audit_sha256": sha256(args.audit),
        "opening_parent_sha256": sha256(args.parent),
        "warmup": args.warmup,
        "iql_updates": args.updates,
        "hidden": args.hidden,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gap_seconds": float(GAP),
    }
    metrics = []
    fit_steps = args.warmup + args.updates
    for step in range(1, fit_steps + 1):
        burn = 0 if step % 4 == 1 else BURN_IN
        warm = step <= args.warmup
        batch = tuple(x.to(device) for x in windows.batch(args.batch_size, burn, warm))
        optimizer.zero_grad(set_to_none=True)
        if warm:
            x, selected, _, _, _, teacher = batch
            total = F.cross_entropy(actor(x[:, :-1], burn)[teacher], selected[teacher])
            measures = {"teacher_bc": float(total.detach())}
        else:
            total, measures = iql_losses(actor, values, target, batch, burn)
        total.backward()
        nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        if not warm:
            with torch.no_grad():
                for old, new in zip(target.parameters(), values.parameters(), strict=True):
                    old.lerp_(new, 0.01)
        if step % 64 == 0 or step in (1, args.warmup, fit_steps):
            row = {"step": step, "stage": "behavior-cloning" if warm else "iql", **measures}
            metrics.append(row)
            print(json.dumps(row), flush=True)
        if step % 256 == 0 or step in (args.warmup, fit_steps):
            destination = args.output / f"step-{step:05d}.pt"
            partial = destination.with_suffix(".partial")
            torch.save(
                {
                    "state": actor.state_dict(),
                    "critic": values.state_dict(),
                    "target": target.state_dict(),
                    "catalogue": catalogue,
                    "step": step,
                    "provenance": provenance,
                },
                partial,
            )
            partial.replace(destination)
    validation = holdout(actor.eval(), values.eval(), groups["holdout"], device)
    report = {
        **provenance,
        "checkpoint": destination.name,
        "checkpoint_sha256": sha256(destination),
        "splits": {name: len(fights) for name, fights in groups.items()},
        "metrics": metrics,
        "holdout": validation,
    }
    (args.output / "training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation["equal_fight_means"]), flush=True)


if __name__ == "__main__":
    main()

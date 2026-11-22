"""Train the offline recurrent Sisters actor and IQL values on locked real trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from hkrl.spaces import BUTTON_BITS
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
CRITICAL_KEYS = ("jump_tap", "dash", "cast")


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
        features = torch.cat(
            recurrent_features(self.encoder, self.memory, inputs, burn), dim=-1
        )
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

    @property
    def pure_teacher(self) -> bool:
        return bool(self.teacher.all())

    @property
    def won(self) -> bool:
        return bool(self.episode.events[-1, 6])


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
        catalogues.append(
            np.frombuffer(leaf, "<i4", count=n * 3, offset=20).reshape(n, 3)
        )
    if not np.array_equal(*catalogues):
        raise ValueError("Frozen Trio action catalogues differ")
    return torch.from_numpy(catalogues[0].astype("i8"))


def prepare_fight(episode: Episode, lookup: dict[tuple[int, ...], int]) -> Fight:
    action = np.array(
        [lookup.get(tuple(key), -1) for key in episode.action], dtype="i8"
    )
    phase = episode.observation[:, 23] > 0.5
    delta = np.diff(episode.clock)
    boundaries = np.r_[
        0, np.flatnonzero((delta > GAP) | (phase[1:] != phase[:-1])) + 1, len(phase)
    ]
    segments = tuple((int(lo), int(hi)) for lo, hi in pairwise(boundaries) if phase[lo])
    discount = np.zeros(len(phase), dtype="f4")
    has_next = (episode.next_row[:-1] >= 0) & (delta <= GAP) & phase[:-1] & phase[1:]
    discount[:-1][has_next] = GAMMA ** (delta[has_next] / 0.04)
    valid = (episode.terminated | (discount > 0)) & (action >= 0) & phase
    return Fight(
        episode, action, valid, episode.controller == "teacher", discount, segments
    )


def read_fights(
    archive: Path, audit: Path, catalogue: Tensor
) -> dict[str, dict[str, Fight]]:
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
                or int(episode.valid_transition.sum())
                != audited["rows"]["valid_transitions"]
            ):
                raise ValueError(f"Transition count changed for {source}")
            groups[item["split"]][source] = prepare_fight(episode, lookup)
    if expected or any(
        len(fights) != report["splits"][name]["episodes"]
        for name, fights in groups.items()
    ):
        raise ValueError("Replay fights differ from the audited splits")
    return groups


class Windows:
    def __init__(
        self,
        fights: list[Fight],
        rng: np.random.Generator,
        history: dict[int, np.ndarray] | None = None,
        *,
        teacher_win_bc: bool = False,
    ) -> None:
        self.rng = rng
        self.history = history or {}
        self.teacher_win_bc = teacher_win_bc
        self.pools: dict[tuple[bool, int], list[tuple[Fight, int, int]]] = {}
        for teacher in (True, False):
            for burn in (0, BURN_IN):
                pool = []
                for fight in fights:
                    valid = (
                        fight.train_rows & fight.teacher
                        if teacher
                        else fight.train_rows
                    )
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
            cached = self.history.get(id(fight))
            rows.append(
                (
                    cached[clipped]
                    if cached is not None
                    else motion_history(tape.clock, tape.observation, clipped),
                    fight.action[scored].clip(0),
                    tape.reward[scored],
                    fight.discount[scored],
                    mask,
                    fight.teacher[scored]
                    & mask
                    & (fight.won or not self.teacher_win_bc),
                )
            )
        return tuple(
            torch.from_numpy(np.stack(column)) for column in zip(*rows, strict=True)
        )


def iql_losses(
    actor: RecurrentActor,
    values: RecurrentValues,
    target: RecurrentValues,
    batch: tuple[Tensor, ...],
    burn: int,
    teacher_class_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    x, action, reward, discount, mask, teacher = batch
    logits = actor(x[:, :-1], burn)
    log_prob = logits.log_softmax(-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
    q1, q2, value = values(x, burn)
    with torch.no_grad():
        old_q1, old_q2, _ = target(x[:, :-1], burn)
        low_q = (
            torch.minimum(old_q1, old_q2).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        )
        td = reward + discount * value[:, 1:]
    gap = low_q - value[:, :-1]
    v_loss = (torch.where(gap >= 0, 0.7, 0.3) * gap.square())[mask].mean()
    picked = [q[:, :-1].gather(-1, action.unsqueeze(-1)).squeeze(-1) for q in (q1, q2)]
    q_loss = sum(F.smooth_l1_loss(q[mask], td[mask]) for q in picked)
    weight = (3.0 * gap.detach()[mask]).clamp(-12.0, float(np.log(20.0))).exp()
    awr = -(weight * log_prob[mask]).sum() / weight.sum().clamp_min(1e-6)
    bc = teacher_bc_loss(log_prob[teacher], action[teacher], teacher_class_weight)
    total = q_loss + v_loss + awr + 0.5 * bc
    return total, {
        name: float(part.detach())
        for name, part in (
            ("q", q_loss),
            ("v", v_loss),
            ("awr", awr),
            ("teacher_bc", bc),
        )
    }


def teacher_bc_loss(
    log_prob: Tensor, action: Tensor, class_weight: Tensor | None
) -> Tensor:
    if not log_prob.numel():
        return log_prob.new_zeros(())
    if class_weight is None:
        return -log_prob.mean()
    weight = class_weight[action]
    return -(weight * log_prob).sum() / weight.sum()


@torch.inference_mode()
def holdout(
    actor: RecurrentActor,
    values: RecurrentValues,
    fights: dict[str, Fight],
    device: torch.device,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    actions_with_keys = actor.encoder.basis.detach().cpu().numpy()[:, 6:].astype(bool)
    key_columns = {"jump": 0, "dash": 2, "attack": 3, "cast": 4, "focus": 5}
    for source, fight in fights.items():
        tape = fight.episode
        report: dict[str, Any] = {}
        score = {name: np.zeros(3) for name in ("teacher", "student", "unknown")}
        key_counts = np.zeros((3, actions_with_keys.shape[1]), dtype=np.int64)
        closest_fsm = np.zeros((2, 16), dtype=np.int64)
        value_sum = value_square = 0.0
        count = 0
        for lo, hi in fight.segments:
            rows = np.arange(lo, hi)
            x = torch.from_numpy(motion_history(tape.clock, tape.observation, rows)).to(
                device
            )[None]
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
            mentor_rows = fight.teacher[rows] & fight.train_rows[rows]
            wanted = fight.action[rows[mentor_rows]]
            guessed = logits[mentor_rows].argmax(-1).numpy()
            expected_keys = actions_with_keys[wanted]
            predicted_keys = actions_with_keys[guessed]
            key_counts += np.stack(
                (
                    expected_keys.sum(0),
                    predicted_keys.sum(0),
                    (expected_keys & predicted_keys).sum(0),
                )
            )
            # The nearest boss slot contains a masked one-hot FSM value, zero for unknown.
            fsm = tape.observation[rows[mentor_rows], 44:60] > 0.5
            closest_fsm += np.stack(
                (fsm.sum(0), (fsm & (wanted == guessed)[:, None]).sum(0))
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
        if key_counts.any():
            report["teacher_key_counts"] = {
                name: {"labeled": int(n), "predicted": int(p), "matches": int(m)}
                for name, column in key_columns.items()
                for n, p, m in [key_counts[:, column]]
            }
            report["teacher_nearest_boss_fsm"] = {
                str(i): {"rows": int(n), "top1": float(m / n)}
                for i, (n, m) in enumerate(closest_fsm.T)
                if n
            }
        records[source] = report
    means = {}
    for key in ("teacher", "student", "unknown"):
        group = [row[key] for row in records.values() if key in row]
        if group:
            means[key] = {
                metric: float(np.mean([row[metric] for row in group]))
                for metric in ("top1", "nll")
            }
    mentor_wins = [
        records[source]["teacher"]
        for source, fight in fights.items()
        if fight.pure_teacher and fight.won and "teacher" in records[source]
    ]
    if mentor_wins:
        means["teacher_wins"] = {
            key: float(np.mean([r[key] for r in mentor_wins]))
            for key in ("top1", "nll")
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
    parser.add_argument(
        "--teacher-curriculum",
        action="store_true",
        help="Use only pure-teacher wins in BC and all pure-teacher outcomes in IQL",
    )
    parser.add_argument(
        "--critical-teacher-bc-weight",
        type=float,
        default=1.0,
        help="Weight teacher BC rows that tap jump, dash, or cast (teacher curriculum only)",
    )
    parser.add_argument("--cache-history", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument("--validation-patience", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 1 or args.updates < 1 or args.batch_size < 2 or args.threads < 1:
        raise ValueError(
            "Use positive update and thread counts and at least two windows per batch"
        )
    if (
        args.checkpoint_interval < 1
        or args.validation_interval < 0
        or args.validation_patience < 0
    ):
        raise ValueError("Invalid checkpoint or validation interval")
    if args.validation_patience and not args.validation_interval:
        raise ValueError("Early stopping needs an independent validation interval")
    if (
        not math.isfinite(args.critical_teacher_bc_weight)
        or args.critical_teacher_bc_weight < 1
    ):
        raise ValueError("Critical teacher BC weight must be finite and at least 1")
    if args.critical_teacher_bc_weight != 1 and not args.teacher_curriculum:
        raise ValueError(
            "Critical teacher BC weighting requires the teacher curriculum"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    catalogue = native_catalogue(args.parent)
    teacher_class_weight = None
    if args.critical_teacher_bc_weight != 1:
        critical_key_mask = sum(1 << BUTTON_BITS[key] for key in CRITICAL_KEYS)
        critical_actions = (
            catalogue[:, 2].to(device).bitwise_and(critical_key_mask).ne(0)
        )
        teacher_class_weight = torch.where(
            critical_actions, args.critical_teacher_bc_weight, 1.0
        )
    groups = read_fights(args.archive, args.audit, catalogue)
    if (
        args.teacher_curriculum
        and args.validation_interval
        and not any(
            fight.pure_teacher and fight.won and fight.train_rows.any()
            for fight in groups["holdout"].values()
        )
    ):
        raise ValueError(
            "Periodic validation needs a held-out pure teacher win with Trio decisions"
        )
    training = list(groups["train"].values())
    if args.teacher_curriculum:
        training = [fight for fight in training if fight.pure_teacher]
    bc_training = (
        [fight for fight in training if fight.won]
        if args.teacher_curriculum
        else training
    )
    cohorts = {
        "bc_fights": len(bc_training),
        "iql_fights": len(training),
        "iql_wins": sum(fight.won for fight in training),
        "archive_train_fights": len(groups["train"]),
        "holdout_fights": len(groups["holdout"]),
    }
    print(json.dumps(cohorts), flush=True)
    history = (
        {
            id(fight): motion_history(fight.episode.clock, fight.episode.observation)
            for fight in training
        }
        if args.cache_history
        else None
    )
    rng = np.random.default_rng(args.seed)
    windows = Windows(training, rng, history, teacher_win_bc=args.teacher_curriculum)
    warm_windows = (
        Windows(bc_training, rng, history) if args.teacher_curriculum else windows
    )
    actor = RecurrentActor(catalogue, args.hidden).to(device)
    values = RecurrentValues(catalogue, args.hidden).to(device)
    target = deepcopy(values).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": 3e-4},
            {"params": values.parameters(), "lr": 3e-4},
        ]
    )
    parameters = [*actor.parameters(), *values.parameters()]
    provenance = {
        "method": "recurrent-iql-v2",
        "randomly_initialized_trio": True,
        "audit_sha256": sha256(args.audit),
        "opening_parent_sha256": sha256(args.parent),
        "warmup": args.warmup,
        "planned_iql_updates": args.updates,
        "training_cohorts": cohorts,
        "teacher_curriculum": args.teacher_curriculum,
        "teacher_win_bc": args.teacher_curriculum,
        "critical_teacher_bc_weight": args.critical_teacher_bc_weight,
        "trainer_source_sha256": sha256(Path(__file__)),
        "archive_sha256": sha256(args.archive),
        "hidden": args.hidden,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gap_seconds": float(GAP),
    }
    metrics = []
    validation_steps = []
    validations_without_gain = 0
    best_nll = float("inf")
    best = None
    fit_steps = args.warmup + args.updates
    for step in range(1, fit_steps + 1):
        burn = 0 if step % 4 == 1 else BURN_IN
        warm = step <= args.warmup
        pool = warm_windows if warm else windows
        batch = tuple(x.to(device) for x in pool.batch(args.batch_size, burn, warm))
        optimizer.zero_grad(set_to_none=True)
        if warm:
            x, selected, _, _, _, teacher = batch
            total = F.cross_entropy(
                actor(x[:, :-1], burn)[teacher],
                selected[teacher],
                weight=teacher_class_weight,
            )
            measures = {"teacher_bc": float(total.detach())}
        else:
            total, measures = iql_losses(
                actor, values, target, batch, burn, teacher_class_weight
            )
        total.backward()
        nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        if not warm:
            with torch.no_grad():
                for old, new in zip(
                    target.parameters(), values.parameters(), strict=True
                ):
                    old.lerp_(new, 0.01)
        if step % 64 == 0 or step in (1, args.warmup, fit_steps):
            row = {
                "step": step,
                "stage": "behavior-cloning" if warm else "iql",
                **measures,
            }
            metrics.append(row)
            print(json.dumps(row), flush=True)
        validate = bool(
            args.validation_interval
            and (
                step in (args.warmup, fit_steps, min(args.warmup, 384))
                or (not warm and (step - args.warmup) % args.validation_interval == 0)
            )
        )
        if (
            step % args.checkpoint_interval == 0
            or step in (args.warmup, fit_steps)
            or validate
        ):
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
        if validate:
            current = holdout(actor, values, groups["holdout"], device)
            means = current["equal_fight_means"]
            metric_name = "teacher_wins" if args.teacher_curriculum else "teacher"
            nll = means[metric_name]["nll"]
            validation_steps.append({"step": step, "nll": nll, "means": means})
            print(json.dumps({"step": step, "validation": means}), flush=True)
            if not warm:
                if nll < best_nll - 0.001:
                    best_nll = nll
                    best = destination, current, step
                    validations_without_gain = 0
                else:
                    validations_without_gain += 1
                    if (
                        args.validation_patience
                        and validations_without_gain >= args.validation_patience
                    ):
                        break
    if best is None:
        validation = holdout(actor.eval(), values.eval(), groups["holdout"], device)
        selected_step = step
    else:
        destination, validation, selected_step = best
    report = {
        **provenance,
        "checkpoint": destination.name,
        "checkpoint_sha256": sha256(destination),
        "splits": {name: len(fights) for name, fights in groups.items()},
        "metrics": metrics,
        "iql_updates": step - args.warmup,
        "selected_iql_update": selected_step - args.warmup,
        "validation_steps": validation_steps,
        "holdout": validation,
    }
    (args.output / "training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation["equal_fight_means"]), flush=True)


if __name__ == "__main__":
    main()

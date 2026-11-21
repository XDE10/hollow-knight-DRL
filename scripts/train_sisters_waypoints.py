"""Distill a causal waypoint policy with control-conditioned offline returns."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from train_sisters_memory_residual import external_assignments, measured_targets
from train_sisters_native_iql import attribute_events
from train_sisters_timing_iql import (
    PHYSICAL,
    TimingPolicy,
    mlp,
    motion_history,
    phase_checkpoint,
    phase_components,
    save_native,
)


def waypoint_labels(
    t: np.ndarray, s: np.ndarray, debit: np.ndarray, switches: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Achieved motion is a training target only, with exogenous hits and handoffs excluded."""
    rows = np.arange(len(t))
    end = np.searchsorted(t, t + 0.32 - 1e-5)
    endpoint = end.clip(max=len(t) - 1)
    new = np.r_[True, (np.diff(t) > 0.081) | (s[1:, 23] != s[:-1, 23])]
    new[switches] = True
    block = np.cumsum(new)
    # A delayed source event and its earlier HP change can both exclude a window.
    # Only the game's event ledger contributes a numerical cost to the return.
    contact = (debit>0) | np.r_[np.diff(s[:,4]) < -.05, False]
    payments = np.r_[0, np.cumsum(contact)]
    travel = (s[endpoint, :2] - s[:, :2]) * [10.0, 8.0]
    # Left / stationary / right crossed with down / level / up, in game units.
    bucket = (travel > 0.55).astype("i8") - (travel < -0.55) + 1
    goal = bucket[:, 0] * 3 + bucket[:, 1]
    valid = (
        (end < len(t))
        & (t[endpoint] - t <= 0.361)
        & (s[:, 23] > 0.5)
        & (block[endpoint] == block)
        & (payments[endpoint] == payments[rows])
        & (s[:, 9] < 0.5)
    )
    return goal, valid


def conditional_shuffle(
    s: np.ndarray, goals: np.ndarray, valid: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Match velocity and grounding within a fight while breaking the future-goal pairing."""
    out = goals.copy()
    group = (
        np.digitize(s[:, 2], [-0.25, 0.25]) * 3 + np.digitize(s[:, 3], [-0.18, 0.18])
    ) * 2 + (s[:, 7] > 0.5)
    for key in np.unique(group[valid]):
        rows = np.flatnonzero(valid & (group == key))
        out[rows] = goals[rng.permutation(rows)]
    return out


class GoalActor(nn.Module):
    def __init__(self, actions: int) -> None:
        super().__init__()
        self.high = nn.Sequential(nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, 9))
        self.behavior = nn.Sequential(nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, 9))
        self.motor = nn.Sequential(
            nn.Linear(201, 128), nn.ReLU(), nn.Linear(128, actions)
        )
        nn.init.zeros_(self.motor[-1].weight)
        nn.init.zeros_(self.motor[-1].bias)

    def command(self, h: Tensor, base: Tensor, goal: Tensor) -> Tensor:
        code = F.one_hot(goal, 9).to(h.dtype)
        return base + self.motor(torch.cat((h, code), 1)).clamp(-8, 8)

    def policy(self, h: Tensor, base: Tensor) -> Tensor:
        # Both inputs are available before the real game selects its next action.
        frequency = self.behavior(h).softmax(1)
        supported = frequency >= 0.10 * frequency.amax(1, keepdim=True)
        mixture = self.high(h).masked_fill(~supported, -torch.inf).softmax(1)
        goals = torch.arange(9, device=h.device).repeat(len(h))
        actions = self.command(
            h.repeat_interleave(9, 0), base.repeat_interleave(9, 0), goals
        )
        return (mixture[:, :, None] * actions.softmax(1).reshape(len(h), 9, -1)).sum(1)


def load(
    path: Path, actor: TimingPolicy, ck: dict, device: torch.device, limit: int | None
) -> dict[str, list[dict]]:
    value = mlp(ck["value"]["0.weight"].shape[1], 1).to(device).eval()
    value.load_state_dict(ck["value"])
    dynamics = (
        mlp(ck["value"]["0.weight"].shape[1] + 11, len(PHYSICAL)).to(device).eval()
    )
    dynamics.load_state_dict(ck["world"])
    catalogue = {tuple(v): i for i, v in enumerate(ck["catalogue"].tolist())}
    result = {"train": [], "holdout": []}
    seen = set()
    with zipfile.ZipFile(path) as archive, torch.no_grad():
        for game in json.loads(archive.read("manifest.json")):
            group = game["split"]
            if limit and len(result[group]) >= limit:
                continue
            name = game["source"]
            blob = archive.read(game["file"])
            if (
                not name.startswith("native-development/")
                or name in seen
                or hashlib.sha256(blob).hexdigest() != game["sha256"]
            ):
                raise ValueError("Unexpected or changed development fight")
            seen.add(name)
            with np.load(io.BytesIO(blob)) as data:
                t, s, keys = (
                    data["control_time"],
                    data["timing_state"],
                    data["controls"].astype("i8"),
                )
                if not np.any(s[:, 23] > 0.5):
                    continue
                events = attribute_events(
                    t,
                    data["time"],
                    data["elapsed_ticks"],
                    data["events"],
                    health_masks=s[:, 4] * 9,
                )
                switches = external_assignments(data)
                goal, valid = waypoint_labels(t, s, events[:, 1:5].sum(1), switches)
                alt = conditional_shuffle(
                    s, goal, valid, np.random.default_rng(int(name.rsplit("/", 1)[1]))
                )
                teacher = (
                    data["teacher_acted"].astype(bool)
                    if "teacher_acted" in data
                    else np.zeros(len(t), bool)
                )
                entries = np.flatnonzero(np.r_[False, teacher[1:] & ~teacher[:-1]])
                entered_from_student = np.zeros(len(t), bool)
                for j in entries:
                    before = np.searchsorted(t, t[j] - 0.72)
                    if t[j] - t[before] < 0.63 or teacher[before:j].any():
                        continue
                    stop = np.searchsorted(t, t[j] + 0.72)
                    entered_from_student[j:stop] = teacher[j:stop]
                history = torch.from_numpy(motion_history(t, s)).to(device)
                action = torch.from_numpy(
                    np.column_stack(
                        (keys[:, :2], (keys[:, 2, None] >> np.arange(9)) & 1)
                    ).astype("f4")
                ).to(device)
                pres = []
                vals = []
                model_vals = []
                hook = actor.dense[1].register_forward_hook(
                    lambda _m, _i, o, storage=pres: storage.append(
                        o.detach().cpu().half()
                    )
                )
                try:
                    for j in range(0, len(t), 512):
                        sl = slice(j, min(j + 512, len(t)))
                        x = history[sl]
                        actor.encode(x)
                        vals.append(value(x).flatten().cpu().numpy())
                        following = history[
                            torch.arange(j, j + len(x), device=device)
                            .add(1)
                            .clamp_max(len(t) - 1)
                        ].clone()
                        following[:, PHYSICAL] = x[:, PHYSICAL] + dynamics(
                            torch.cat((x, action[sl]), 1)
                        )
                        model_vals.append(value(following).flatten().cpu().numpy())
                finally:
                    hook.remove()
                advantage, measured, futures, known = measured_targets(
                    t,
                    s,
                    events,
                    np.concatenate(vals),
                    np.concatenate(model_vals),
                    switches,
                )
                labels = np.fromiter(
                    (catalogue.get(tuple(k), -1) for k in keys), dtype="i8"
                )
                rows = np.flatnonzero((s[:, 23] > 0.5) & (labels >= 0) & measured)
                if not len(rows):
                    continue
                result[group].append(
                    {
                        "source": name,
                        "teacher_free": game.get("teacher_labels", True) is False,
                        "pre": torch.cat(pres)[rows],
                        "goal": torch.from_numpy(goal[rows]),
                        "shuffled": torch.from_numpy(alt[rows]),
                        "valid": torch.from_numpy(valid[rows]),
                        "label": torch.from_numpy(labels[rows]),
                        "advantage": torch.from_numpy(advantage[rows]),
                        "teacher": torch.from_numpy(teacher[rows]),
                        "takeover": torch.from_numpy(entered_from_student[rows]),
                        "clean_takeover": torch.from_numpy(
                            (entered_from_student & known[:, 0] & (futures[:, 0] == 0))[
                                rows
                            ]
                        ),
                        "soon_hurt": torch.from_numpy(
                            (known[:, 0] & (futures[:, 0] > 0))[rows]
                        ),
                    }
                )
    return result


def logits(actor: TimingPolicy, pre: Tensor) -> tuple[Tensor, Tensor]:
    hidden = actor.dense[3](actor.dense[2](pre))
    return hidden, actor.heads(hidden)[0]


def fit(
    base: TimingPolicy,
    games: list[dict],
    goal_key: str,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[TimingPolicy, GoalActor]:
    torch.manual_seed(seed)
    buf = {
        k: torch.cat([g[k] for g in games]).to(device)
        for k in ("pre", "label", "goal", "shuffled", "valid", "advantage", "takeover")
    }
    valid = buf["valid"].nonzero().flatten()
    recovery = (buf["takeover"] & buf["valid"]).nonzero().flatten()
    if not len(valid):
        raise ValueError("No controllable native trajectories")
    scale = buf["advantage"].abs().median().clamp_min(1.0)
    motor = GoalActor(base.out.out_features).to(device)
    opt = torch.optim.AdamW(motor.parameters(), lr=0.0007)
    student = copy.deepcopy(base).requires_grad_(False)
    student.dense[2].requires_grad_(True)
    student.out.requires_grad_(True)
    student.factors.requires_grad_(True)
    refine = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=0.00018
    )

    def sample(pool: Tensor, n: int) -> Tensor:
        return pool[torch.randint(len(pool), (n,), device=device)]

    for _ in range(steps):
        j = torch.cat(
            (sample(valid, 384), sample(recovery if len(recovery) else valid, 128))
        )
        h, original = logits(base, buf["pre"][j].float())
        goal = torch.zeros_like(j) if goal_key == "direct" else buf[goal_key][j]
        weight = (buf["advantage"][j] / scale).clamp(-2.0, 1.5).exp()
        cost = F.cross_entropy(motor.high(h), goal, reduction="none") + F.cross_entropy(
            motor.command(h, original, goal), buf["label"][j], reduction="none"
        )
        loss = (weight * cost).sum() / weight.sum() + F.cross_entropy(
            motor.behavior(h), goal
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(motor.parameters(), 1.0)
        opt.step()
    motor.eval().requires_grad_(False)
    all_rows = torch.arange(len(buf["pre"]), device=device)
    for _ in range(steps):
        j = torch.cat(
            (sample(all_rows, 384), sample(recovery if len(recovery) else valid, 128))
        )
        pre = buf["pre"][j].float()
        h, original = logits(base, pre)
        # No achieved goal, future health, teacher label or realized return is an actor input.
        target = (
            motor.command(h, original, torch.zeros_like(j)).softmax(1)
            if goal_key == "direct"
            else motor.policy(h, original)
        )
        target = 0.85 * target + 0.15 * original.softmax(1)
        _, actual = logits(student, pre)
        loss = -(target * actual.log_softmax(1)).sum(1).mean()
        refine.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        refine.step()
    return student.eval().requires_grad_(False), motor


def evaluate(
    base: TimingPolicy,
    actor: TimingPolicy,
    games: list[dict],
    cat: Tensor,
    device: torch.device,
) -> list[dict]:
    result = []
    with torch.no_grad():
        for g in games:
            pre = g["pre"].float().to(device)
            y = g["label"].to(device)
            old = logits(base, pre)[1]
            new = logits(actor, pre)[1]
            a, b = old.argmax(1), new.argmax(1)
            mx = cat[y, 0]
            d = {
                "source": g["source"],
                "teacher_free": g["teacher_free"],
                "rows": len(y),
                "changed": int((a != b).sum()),
            }
            for label in ("teacher", "takeover", "clean_takeover", "soon_hurt"):
                m = g[label].to(device)
                d[label] = {
                    "rows": int(m.sum()),
                    "base_keys": int(((a == y) & m).sum()),
                    "keys": int(((b == y) & m).sum()),
                    "base_direction": int(((cat[a, 0] == mx) & m).sum()),
                    "direction": int(((cat[b, 0] == mx) & m).sum()),
                    "issued_key_log_shift_sum": float(
                        (
                            (new.log_softmax(1) - old.log_softmax(1))
                            .gather(1, y[:, None])
                            .flatten()
                            * m
                        ).sum()
                    ),
                }
            delta = (
                (new.log_softmax(1) - old.log_softmax(1))
                .gather(1, y[:, None])
                .flatten()
            )
            d["return_weighted_log_shift"] = float(
                (g["advantage"].to(device).clamp(-5, 5) * delta).mean()
            )
            result.append(d)
    return result


def nominate(real: list[dict], shuffled: list[dict]) -> bool:
    matched = [
        (a, b)
        for a, b in zip(real, shuffled, strict=True)
        if a["takeover"]["rows"] >= 5
    ]
    if len(matched) < 3:
        return False
    take = [(a["takeover"], b["takeover"]) for a, b in matched]
    diff = sum(a["direction"] - a["base_direction"] for a, b in take)
    vs = sum(a["direction"] - b["direction"] for a, b in take)
    ordinary = sum(
        g["teacher"]["direction"] - g["teacher"]["base_direction"] for g in real
    )
    support = [g for g in real if g["teacher_free"]]
    return (
        diff > 0
        and vs > 0
        and sum(a["direction"] > b["direction"] for a, b in take) > len(take) / 2
        and ordinary >= -0.01 * sum(g["teacher"]["rows"] for g in real)
        and len(support) >= 3
        and np.mean([g["return_weighted_log_shift"] for g in support]) > 0
    )


def train(
    archive: Path,
    parent: Path,
    learner: Path,
    output: Path,
    steps: int,
    seed: int,
    limit_games: int | None,
) -> None:
    if output.exists() or steps <= 0:
        raise ValueError("Expected fresh output and positive updates")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    ck = torch.load(learner, weights_only=True, map_location="cpu")
    cat = ck["catalogue"].long().to(device)
    base = TimingPolicy(cat, True).to(device).eval().requires_grad_(False)
    base.load_state_dict(ck["actor"])
    opening, _ = phase_components(parent.read_bytes())
    data = load(archive, base, ck, device, limit_games)
    print("Cached whole games", {k: len(v) for k, v in data.items()}, flush=True)
    output.mkdir(parents=True)
    metrics = {}
    for key in ("goal", "shuffled", "direct"):
        actor, motor = fit(base, data["train"], key, steps, seed, device)
        metrics[key] = evaluate(base, actor, data["holdout"], cat, device)
        if key == "goal":
            save_native(actor, cat.cpu().numpy(), output / "trio.bin")
            (output / "world_iql.bin").write_bytes(
                phase_checkpoint(opening, (output / "trio.bin").read_bytes())
            )
            ck_out = {**ck, "actor": actor.cpu().state_dict()}
            torch.save(ck_out, output / "learner.pt")
            torch.save(motor.cpu().state_dict(), output / "waypoints.pt")

    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    report = {
        "method": "Offline return-weighted hierarchical policy and causal distillation, with an identical shuffled-goal control.",
        "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "seed": seed,
        "updates_per_stage": steps,
        "training_fights": len(data["train"]),
        "heldout_fights": len(data["holdout"]),
        "limit_games": limit_games,
        "eligible": bool(
            not limit_games
            and all(
                nominate(metrics["goal"], metrics[k]) for k in ("shuffled", "direct")
            )
        ),
        "heldout": metrics,
        "parent_sha256": digest(parent),
        "learner_input_sha256": digest(learner),
        "archive_sha256": digest(archive),
        "policy_sha256": digest(output / "world_iql.bin"),
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Finished", report["eligible"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "parent", "learner", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=3447)
    p.add_argument("--limit-games", type=int)
    train(**vars(p.parse_args()))

"""Fit an arena policy with entity timings, teacher labels and offline game returns."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from sisters_future_occupancy import TARGET_SHAPE, future_occupancy
from torch import Tensor, nn
from torch.nn import functional as F
from train_sisters_native_iql import REWARD_WEIGHTS, attribute_events, quantile_loss

WIDTH = 192
FOES, DISCS, CONTROL = 24, 132, 160
MIRROR = [0, 2, 6, CONTROL]
for offset in (24, 60, 96):
    MIRROR.extend(offset + j for j in (1, 3, 5, 11, 12, 15, 18))
for offset in (132, 146):
    MIRROR.extend(offset + j for j in (1, 3, 5, 11, 12))
# Predict measured physical variables only. History and key ages are deterministic.
PHYSICAL = [0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20]
for offset in (24, 60, 96, 132, 146):
    PHYSICAL.extend(offset + j for j in range(10))


class TimingPolicy(nn.Module):
    def __init__(self, catalogue: Tensor) -> None:
        super().__init__()
        catalogue = catalogue.long()
        self.boss = nn.Linear(36, 48)
        self.disc = nn.Linear(14, 24)
        self.dense = nn.Sequential(
            nn.Linear(WIDTH + 72, 256), nn.ReLU(), nn.Linear(256, 192), nn.ReLU()
        )
        self.out = nn.Linear(192, len(catalogue))
        # Per-factor supervision shares the same representation; these heads are
        # also folded into the exported joint logits.
        self.factors = nn.Linear(192, 15)
        bits = (
            (catalogue[:, 2, None] >> torch.arange(9, device=catalogue.device)) & 1
        ).float()
        basis = torch.cat(
            [
                F.one_hot(catalogue[:, 0] + 1, 3),
                F.one_hot(catalogue[:, 1] + 1, 3),
                bits,
            ],
            dim=1,
        )
        self.register_buffer("basis", basis.float())

    @staticmethod
    def pool(slots: Tensor, layer: nn.Linear) -> Tensor:
        return (F.relu(layer(slots)) * slots[:, :, :1]).amax(1)

    def encode(self, state: Tensor) -> Tensor:
        bosses = state[:, FOES:DISCS].reshape(-1, 3, 36)
        discs = state[:, DISCS:CONTROL].reshape(-1, 2, 14)
        bosses = bosses * bosses[:, :, :1]
        discs = discs * discs[:, :, :1]
        boss = self.pool(bosses, self.boss)
        disc = self.pool(discs, self.disc)
        return self.dense(
            torch.cat(
                [
                    state[:, :FOES],
                    bosses.flatten(1),
                    discs.flatten(1),
                    state[:, CONTROL:],
                    boss,
                    disc,
                ],
                1,
            )
        )

    def heads(self, x: Tensor) -> tuple[Tensor, Tensor]:
        factors = self.factors(x)
        return self.out(x) + factors @ self.basis.T, factors

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.heads(self.encode(state))


def save_native(actor: TimingPolicy, catalogue: np.ndarray, path: Path) -> None:
    def array(t: Tensor) -> bytes:
        return t.detach().cpu().numpy().astype("<f4").tobytes()

    with path.open("wb") as out:
        out.write(struct.pack("<5i", 0x324C5149, WIDTH, 256, 192, len(catalogue)))
        out.write(catalogue.astype("<i4").tobytes())
        for layer in (actor.boss, actor.disc, actor.dense[0], actor.dense[2]):
            out.write(array(layer.weight))
            out.write(array(layer.bias))
        out.write(array(actor.out.weight + actor.basis @ actor.factors.weight))
        out.write(array(actor.out.bias + actor.basis @ actor.factors.bias))


def teacher_history_age(
    times: np.ndarray, applied: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Seconds since the last earlier input where the teacher did not control the game."""
    different = np.any(applied != labels, axis=1)
    last_mismatch = np.maximum.accumulate(np.where(different, times, -np.inf))
    return times - np.r_[-np.inf, last_mismatch[:-1]]


def measured_returns(
    times: np.ndarray,
    rewards: np.ndarray,
    discounts: np.ndarray,
    valid: np.ndarray,
    seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate observed rewards on one tape. Stop at death, gaps or the horizon.

    The returned index selects the successor of the last included transition.
    A gap ends a window before the gap and preserves bootstrapping at its start.
    """
    n = len(times)
    rows = np.arange(n)
    first_missing = np.minimum.accumulate(np.where(valid, n, rows)[::-1])[::-1]
    end = np.minimum(
        np.searchsorted(times.astype("f8"), times.astype("f8") + seconds), first_missing
    )
    total = np.zeros(n, dtype="f4")
    remaining = np.ones(n, dtype="f4")
    last = rows.copy()
    for k in range(int(np.max(end - rows, initial=0))):
        active = (rows + k < end) & (remaining > 0)
        j = rows[active] + k
        total[active] += remaining[active] * rewards[j]
        remaining[active] *= discounts[j]
        last[active] = j
    return total, remaining, last


def build_data(
    directory: Path,
    device: str,
    mirror: bool = True,
    forecast: bool = False,
    trio_only: bool = False,
    return_seconds: float = 0.0,
) -> tuple[dict, np.ndarray, dict]:
    games = json.loads((directory / "manifest.json").read_bytes())
    if len({g["source"] for g in games}) != len(games) or any(
        not g["source"].startswith("native-development/")
        or g["split"] not in ("train", "holdout")
        for g in games
    ):
        raise ValueError("Expected unique development games with explicit splits")
    parts: dict[str, list] = {"train": [], "holdout": []}
    fields: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
    windows: dict[str, list[tuple]] = {"train": [], "holdout": []}
    stats: dict[str, dict] = {}
    for game in games:
        path = directory / game["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != game["sha256"]:
            raise ValueError(f"Changed tape {game['source']}")
        with np.load(path, allow_pickle=False) as f:
            s = f["timing_state"].astype("f4")
            t, act, teacher = f["control_time"], f["controls"], f["teacher_controls"]
            if (
                s.shape != (len(t), WIDTH)
                or not np.isfinite(s).all()
                or (np.diff(t) <= 0).any()
            ):
                raise ValueError("Invalid native state or timestamp")
            ev = attribute_events(t, f["time"], f["elapsed_ticks"], f["events"])
        terminal = ev[:, 6:].any(1)
        limit = np.flatnonzero(terminal)
        n = int(limit[0]) + 1 if len(limit) else len(t)
        s, t, act, teacher, ev, terminal = [
            v[:n] for v in (s, t, act, teacher, ev, terminal)
        ]
        dt = np.r_[np.diff(t), 0.04].astype("f4")
        d = (0.995 ** (dt / 0.04)) * (~terminal)
        good = (dt <= 0.24) & (np.r_[np.ones(n - 1, bool), False] | terminal)
        next_state = np.r_[s[1:], s[-1:]]
        reward = ev @ REWARD_WEIGHTS
        agreement = teacher_history_age(t, act, teacher)
        selected = s[:, 23] > 0.5 if trio_only else slice(None)
        if trio_only and not selected.any():
            continue
        if forecast:
            fields[game["split"]].append(future_occupancy(t, s)[selected])
        if return_seconds:
            total, remaining, last = measured_returns(
                t, reward, d, good, return_seconds
            )
            windows[game["split"]].append(
                (total[selected], remaining[selected], next_state[last[selected]])
            )
        parts[game["split"]].append(
            tuple(
                x[selected]
                for x in (s, act, teacher, next_state, reward, d, good, agreement)
            )
        )
    if any(not games for games in parts.values()):
        raise ValueError("Need both fight splits")
    result = {}
    catalogue = None
    for split, items in parts.items():
        s, a, label, ns, r, d, g, age = [
            np.concatenate(x) for x in zip(*items, strict=True)
        ]
        stats[split] = {"fights": len(items), "decisions": len(s)}
        if split == "train" and mirror:
            ms, mns, ma, ml = s.copy(), ns.copy(), a.copy(), label.copy()
            ms[:, MIRROR] *= -1
            mns[:, MIRROR] *= -1
            ma[:, 0] *= -1
            ml[:, 0] *= -1
            s, a, label, ns = [
                np.concatenate(pair)
                for pair in ((s, ms), (a, ma), (label, ml), (ns, mns))
            ]
            r, d, g, age = [np.tile(x, 2) for x in (r, d, g, age)]
        if split == "train":
            catalogue = np.unique(np.concatenate((a, label)).astype("i8"), axis=0)
        lookup = {tuple(cmd): i for i, cmd in enumerate(catalogue)}
        yi = np.array([lookup.get(tuple(cmd), -1) for cmd in a], dtype="i8")
        li = np.array([lookup.get(tuple(cmd), -1) for cmd in label], dtype="i8")
        result[split] = tuple(
            torch.from_numpy(x).to(device) for x in (s, yi, li, ns, r, d, g, age)
        )
        if forecast:
            volumes = np.concatenate(fields[split])
            if split == "train" and mirror:
                volumes = np.concatenate((volumes, volumes[..., ::-1]))
            flat = volumes.reshape(len(volumes), -1)
            result[split] += (torch.from_numpy(flat).to(device),)
        if return_seconds:
            total, remaining, successor = [
                np.concatenate(values) for values in zip(*windows[split], strict=True)
            ]
            if split == "train" and mirror:
                mirrored = successor.copy()
                mirrored[:, MIRROR] *= -1
                successor = np.concatenate((successor, mirrored))
                total, remaining = np.tile(total, 2), np.tile(remaining, 2)
            result[split] += tuple(
                torch.from_numpy(x).to(device) for x in (total, remaining, successor)
            )
    return result, catalogue, stats


def mlp(n: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n, 256),
        nn.ReLU(),
        nn.Linear(256, 192),
        nn.ReLU(),
        nn.Linear(192, out),
    )


def previous_commands(states: Tensor) -> Tensor:
    """Decode the keys actually applied on the previous native commit."""
    bits = states[:, CONTROL + 4 : CONTROL + 13].long()
    flags = (bits << torch.arange(9, device=states.device)).sum(dim=1)
    return torch.cat((states[:, CONTROL : CONTROL + 2].long(), flags[:, None]), 1)


def phase_checkpoint(opener: bytes, trio: bytes) -> bytes:
    for component in (opener, trio):
        if len(component) < 20 or struct.unpack_from("<4i", component) != (
            0x324C5149,
            WIDTH,
            256,
            192,
        ):
            raise ValueError("Expected a native timing actor for each battle phase")
    return struct.pack("<3i", 0x334C5149, len(opener), len(trio)) + opener + trio


def initialize_actor(actor: TimingPolicy, commands: np.ndarray, path: Path) -> int:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    source = TimingPolicy(saved["catalogue"])
    source.load_state_dict(saved["actor"])
    for name in ("boss", "disc", "dense", "factors"):
        getattr(actor, name).load_state_dict(getattr(source, name).state_dict())
    previous = {tuple(row): j for j, row in enumerate(saved["catalogue"].tolist())}
    matched = 0
    with torch.no_grad():
        for j, cmd in enumerate(commands):
            index = previous.get(tuple(cmd))
            if index is not None:
                actor.out.weight[j].copy_(source.out.weight[index])
                actor.out.bias[j].copy_(source.out.bias[index])
                matched += 1
    return matched


def train(
    data_dir: Path,
    output: Path,
    bc_steps: int,
    world_steps: int,
    rl_steps: int,
    batch: int,
    seed: int,
    mirror: bool = True,
    switch_weight: float = 1.0,
    agreement_seconds: float = 0.0,
    forecast_weight: float = 0.0,
    opener_path: Path | None = None,
    initial_learner: Path | None = None,
    return_seconds: float = 0.0,
) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    opener = opener_path.read_bytes() if opener_path is not None else None
    if opener is not None:
        phase_checkpoint(opener, opener)
    data, commands, counts = build_data(
        data_dir,
        device,
        mirror,
        forecast_weight > 0,
        opener is not None,
        return_seconds,
    )
    output.mkdir(exist_ok=False, parents=True)
    cat = torch.as_tensor(commands, device=device)
    s, y, expert, ns, r, d, good, age = data["train"][:8]
    supervised = (age >= agreement_seconds).nonzero().flatten()
    switch = (cat[expert] != previous_commands(s)).any(1)
    corrective_weight = 1 + (switch_weight - 1) * switch.float()
    valid = good.nonzero().flatten()
    dynamics_rows = (good & (d > 0)).nonzero().flatten()
    actor = TimingPolicy(cat).to(device)
    inherited = (
        initialize_actor(actor, commands, initial_learner)
        if initial_learner is not None
        else 0
    )
    forecast_head = None
    if forecast_weight > 0:
        forecast_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, int(np.prod(TARGET_SHAPE))),
            nn.Sigmoid(),
        ).to(device)
    counts_act = expert[supervised].bincount(minlength=len(cat)).float().clamp_min(1)
    rare = (counts_act.max() / counts_act).pow(0.175).clamp_max(4)
    bits = actor.basis[:, 6:]
    bit_pos = bits[expert[supervised]].mean(0)
    bit_weight = ((1 - bit_pos) / bit_pos.clamp_min(1e-4)).sqrt().clamp(1, 12)
    action_encoding = torch.cat((cat[:, :2], bits), 1).float()
    report = {
        "counts": counts,
        "seed": seed,
        "batch": batch,
        "actions": len(cat),
        "mirrored_training": mirror,
        "teacher_switch_weight": switch_weight,
        "teacher_agreement_seconds": agreement_seconds,
        "future_box_loss_weight": forecast_weight,
        "opener_sha256": hashlib.sha256(opener).hexdigest()
        if opener is not None
        else None,
        "initial_learner_sha256": hashlib.sha256(
            initial_learner.read_bytes()
        ).hexdigest()
        if initial_learner
        else None,
        "inherited_action_rows": inherited,
        "supervised_fraction": len(supervised) / len(s),
        "train_switch_fraction": switch.float().mean().item(),
        "bc_steps": bc_steps,
        "world_steps": world_steps,
        "rl_steps": rl_steps,
        "measured_return_seconds": return_seconds,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "manifest_sha256": hashlib.sha256(
            (data_dir / "manifest.json").read_bytes()
        ).hexdigest(),
    }

    def update(opt: torch.optim.Optimizer, loss: Tensor) -> None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def indices(n: int) -> Tensor:
        return torch.randint(n, (batch,), device=device)

    def factor_loss(factor: Tensor, label: Tensor) -> Tensor:
        return (
            F.cross_entropy(factor[:, :3], cat[label, 0] + 1)
            + F.cross_entropy(factor[:, 3:6], cat[label, 1] + 1)
            + F.binary_cross_entropy_with_logits(
                factor[:, 6:], bits[label], pos_weight=bit_weight
            )
        )

    def save(name: str) -> None:
        was_training = actor.training
        actor.eval()
        with torch.no_grad():
            hs, _, hl = data["holdout"][:3]
            pred = torch.cat([actor(x)[0].argmax(1) for x in hs.split(1024)])
            report[name + "_holdout_exact"] = (pred == hl).float().mean().item()
            known = hl >= 0
            changes = (cat[hl.clamp_min(0)] != previous_commands(hs)).any(1)
            for group, mask in (("switch", changes), ("hold", ~changes)):
                selected = known & mask
                if selected.any():
                    report[name + "_holdout_" + group + "_known_exact"] = (
                        (pred[selected] == hl[selected]).float().mean().item()
                    )
            report["holdout_label_coverage"] = known.float().mean().item()
            path = output / (name + ".bin")
            save_native(actor, commands, path)
            if opener is not None:
                path.write_bytes(phase_checkpoint(opener, path.read_bytes()))
        actor.train(was_training)
        print(name, report[name + "_holdout_exact"], flush=True)

    def forecast_loss(encoded: Tensor, rows: Tensor) -> Tensor:
        if forecast_head is None:
            return encoded.new_zeros(())
        target = data["train"][8][rows]
        known = target.isfinite()
        target = target.nan_to_num()
        weights = known * (1 + 4 * target)
        return (
            (forecast_head(encoded) - target).square() * weights
        ).sum() / weights.sum().clamp_min(1)

    parameters = list(actor.parameters())
    if forecast_head is not None:
        parameters.extend(forecast_head.parameters())
    opt = torch.optim.AdamW(parameters, lr=0.00065)
    for step in range(bc_steps):
        i = supervised[indices(len(supervised))]
        encoded = actor.encode(s[i])
        logits, factor = actor.heads(encoded)
        w = rare[expert[i]] * corrective_weight[i]
        loss = (
            F.cross_entropy(logits, expert[i], reduction="none") * w
        ).sum() / w.sum()
        update(
            opt,
            loss
            + 0.30 * factor_loss(factor, expert[i])
            + forecast_weight * forecast_loss(encoded, i),
        )
        if (step + 1) % 2000 == 0:
            print("teacher", step + 1, flush=True)
    save("bc")
    ref = copy.deepcopy(actor).requires_grad_(False)
    world = mlp(WIDTH + 11, len(PHYSICAL)).to(device)
    wopt = torch.optim.AdamW(world.parameters(), lr=0.0008)
    for _ in range(world_steps):
        i = dynamics_rows[indices(len(dynamics_rows))]
        delta = world(torch.cat((s[i], action_encoding[y[i]]), 1))
        update(
            wopt,
            F.smooth_l1_loss(delta, ns[i][:, PHYSICAL] - s[i][:, PHYSICAL], beta=0.025),
        )
    world.eval().requires_grad_(False)
    hs, hy, _, hn, _, hd, hg, _ = data["holdout"][:8]
    vi = ((hy >= 0) & hg & (hd > 0)).nonzero().flatten()[::4]
    with torch.no_grad():
        delta = torch.cat(
            [
                world(torch.cat((hs[j], action_encoding[hy[j]]), 1))
                for j in vi.split(1024)
            ]
        )
        actual = hn[vi][:, PHYSICAL] - hs[vi][:, PHYSICAL]
        report.update(
            world_mae=(delta - actual).abs().mean().item(),
            persistence_mae=actual.abs().mean().item(),
        )
    q1, q2 = mlp(WIDTH, len(cat)).to(device), mlp(WIDTH, len(cat)).to(device)
    t1, t2 = (
        copy.deepcopy(q1).requires_grad_(False),
        copy.deepcopy(q2).requires_grad_(False),
    )
    val = mlp(WIDTH, 1).to(device)
    qopt = torch.optim.AdamW([*q1.parameters(), *q2.parameters()], lr=0.00035)
    vopt = torch.optim.AdamW(val.parameters(), lr=0.00035)
    aopt = torch.optim.AdamW(parameters, lr=0.00009)
    for step in range(rl_steps):
        i = valid[indices(len(valid))]
        x, u, observed = s[i], y[i], ns[i]
        with torch.no_grad():
            key = u[:, None]
            old = torch.minimum(t1(x), t2(x)).gather(1, key).flatten()
            imagined = observed.clone()
            imagined[:, PHYSICAL] = x[:, PHYSICAL] + world(
                torch.cat((x, action_encoding[u]), 1)
            )
            target = r[i] + d[i] * (
                0.75 * val(observed).flatten() + 0.25 * val(imagined).flatten()
            )
            if return_seconds:
                payout, continuation, successor = data["train"][-3:]
                measured = payout[i] + continuation[i] * val(successor[i]).flatten()
                # Replay continuations come from a mixture of policies. Retain the
                # one-step IQL target alongside their measured short-term outcomes.
                target = 0.5 * (target + measured)
        v = val(x).flatten()
        update(vopt, quantile_loss(old - v, 0.7))
        update(
            qopt,
            sum(
                F.smooth_l1_loss(q(x).gather(1, key).flatten(), target)
                for q in (q1, q2)
            ),
        )
        with torch.no_grad():
            for slow, fast in ((t1, q1), (t2, q2)):
                for a, b in zip(slow.parameters(), fast.parameters(), strict=True):
                    a.lerp_(b, 0.005)
            aw = (6 * (old - v.detach()).clamp(-1, 1)).exp().clamp_max(12) * rare[u]
            prior = ref(x)[0].softmax(1)
        if step >= min(1000, rl_steps // 4):
            encoded = actor.encode(x)
            logits, factor = actor.heads(encoded)
            loss = (F.cross_entropy(logits, u, reduction="none") * aw).sum() / aw.sum()
            if agreement_seconds > 0:
                j = supervised[indices(len(supervised))]
                supervised_logits, factor = actor(s[j])
            else:
                j, supervised_logits = i, logits
            lab = expert[j]
            w = rare[lab] * corrective_weight[j]
            supervision = (
                F.cross_entropy(supervised_logits, lab, reduction="none") * w
            ).sum() / w.sum()
            update(
                aopt,
                loss
                + 0.7 * F.kl_div(logits.log_softmax(1), prior, reduction="batchmean")
                + 0.85 * supervision
                + 0.2 * factor_loss(factor, lab)
                + forecast_weight * forecast_loss(encoded, i),
            )
        if (step + 1) % 2000 == 0:
            print("iql", step + 1, flush=True)
    save("world_iql")
    if return_seconds:
        with torch.no_grad():
            payout, continuation, successor = data["holdout"][-3:]
            rows = (hg & (hy >= 0)).nonzero().flatten()
            errors = []
            for j in rows.split(1024):
                observed = payout[j] + continuation[j] * val(successor[j]).flatten()
                estimate = (
                    torch.minimum(q1(hs[j]), q2(hs[j])).gather(1, hy[j, None]).flatten()
                )
                errors.append(
                    torch.stack(
                        (
                            (observed - estimate).abs(),
                            (observed - val(hs[j]).flatten()).abs(),
                        )
                    )
                )
            report["heldout_return_q_mae"], report["heldout_return_value_mae"] = (
                torch.cat(errors, dim=1).mean(dim=1).tolist()
            )
    if forecast_head is not None:
        with torch.no_grad():
            estimated = torch.cat(
                [forecast_head(actor.encode(x)) for x in hs.split(1024)]
            )
            target = data["holdout"][8]
            report["future_box_holdout_mae"] = (
                (estimated - target).abs().nanmean().item()
            )
            report["future_box_zero_baseline_mae"] = target.abs().nanmean().item()
    torch.save(
        {
            "actor": actor.cpu().state_dict(),
            "catalogue": cat.cpu(),
            "world": world.cpu().state_dict(),
            "q1": q1.cpu().state_dict(),
            "q2": q2.cpu().state_dict(),
            "value": val.cpu().state_dict(),
            **({"opener_native": opener} if opener is not None else {}),
            **(
                {"future_boxes": forecast_head.cpu().state_dict()}
                if forecast_head is not None
                else {}
            ),
        },
        output / "learner.pt",
    )
    (output / "training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bc-steps", type=int, default=11000)
    parser.add_argument("--world-steps", type=int, default=2500)
    parser.add_argument("--rl-steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--teacher-agreement-seconds", type=float, default=0.0)
    parser.add_argument("--forecast-weight", type=float, default=0.0)
    parser.add_argument(
        "--return-seconds",
        type=float,
        default=0.0,
        help="Mix short real trajectories into the IQL target, measured in game seconds.",
    )
    parser.add_argument(
        "--opener-policy",
        type=Path,
        help="Retain this frozen timing actor for the opener; fit only on trio states.",
    )
    parser.add_argument(
        "--initial-learner",
        type=Path,
        help="Initialize the learned actor from a prior timing checkpoint.",
    )
    args = parser.parse_args()
    if min(args.bc_steps, args.world_steps, args.rl_steps, args.batch) < 1:
        parser.error("Positive budgets required")
    if not np.isfinite(args.switch_weight) or args.switch_weight < 1:
        parser.error("Switch weight must be finite and at least one")
    if (
        not np.isfinite(args.teacher_agreement_seconds)
        or args.teacher_agreement_seconds < 0
    ):
        parser.error("Teacher agreement duration must be finite and nonnegative")
    if not np.isfinite(args.forecast_weight) or args.forecast_weight < 0:
        parser.error("Forecast weight must be finite and nonnegative")
    if not 0 <= args.return_seconds <= 4:
        parser.error("Measured return horizon must be between zero and four seconds")
    train(
        args.data,
        args.output,
        args.bc_steps,
        args.world_steps,
        args.rl_steps,
        args.batch,
        args.seed,
        not args.no_mirror,
        args.switch_weight,
        args.teacher_agreement_seconds,
        args.forecast_weight,
        args.opener_policy,
        args.initial_learner,
        args.return_seconds,
    )

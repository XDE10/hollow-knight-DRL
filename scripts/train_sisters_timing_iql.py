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

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        bosses = state[:, FOES:DISCS].reshape(-1, 3, 36)
        discs = state[:, DISCS:CONTROL].reshape(-1, 2, 14)
        bosses = bosses * bosses[:, :, :1]
        discs = discs * discs[:, :, :1]
        boss = self.pool(bosses, self.boss)
        disc = self.pool(discs, self.disc)
        x = self.dense(
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
        factors = self.factors(x)
        return self.out(x) + factors @ self.basis.T, factors


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


def build_data(
    directory: Path, device: str, mirror: bool = True
) -> tuple[dict, np.ndarray, dict]:
    games = json.loads((directory / "manifest.json").read_bytes())
    if len({g["source"] for g in games}) != len(games) or any(
        not g["source"].startswith("native-development/")
        or g["split"] not in ("train", "holdout")
        for g in games
    ):
        raise ValueError("Expected unique development games with explicit splits")
    parts: dict[str, list] = {"train": [], "holdout": []}
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
        parts[game["split"]].append((s, act, teacher, next_state, reward, d, good))
    if any(not games for games in parts.values()):
        raise ValueError("Need both fight splits")
    result = {}
    catalogue = None
    for split, items in parts.items():
        s, a, label, ns, r, d, g = [np.concatenate(x) for x in zip(*items, strict=True)]
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
            r, d, g = [np.tile(x, 2) for x in (r, d, g)]
        if split == "train":
            catalogue = np.unique(np.concatenate((a, label)).astype("i8"), axis=0)
        lookup = {tuple(cmd): i for i, cmd in enumerate(catalogue)}
        yi = np.array([lookup.get(tuple(cmd), -1) for cmd in a], dtype="i8")
        li = np.array([lookup.get(tuple(cmd), -1) for cmd in label], dtype="i8")
        result[split] = tuple(
            torch.from_numpy(x).to(device) for x in (s, yi, li, ns, r, d, g)
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
) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    data, commands, counts = build_data(data_dir, device, mirror)
    output.mkdir(exist_ok=False, parents=True)
    cat = torch.as_tensor(commands, device=device)
    s, y, expert, ns, r, d, good = data["train"]
    switch = (cat[expert] != previous_commands(s)).any(1)
    corrective_weight = 1 + (switch_weight - 1) * switch.float()
    valid = good.nonzero().flatten()
    dynamics_rows = (good & (d > 0)).nonzero().flatten()
    actor = TimingPolicy(cat).to(device)
    counts_act = expert.bincount(minlength=len(cat)).float().clamp_min(1)
    rare = (counts_act.max() / counts_act).pow(0.175).clamp_max(4)
    bits = actor.basis[:, 6:]
    bit_pos = bits[expert].mean(0)
    bit_weight = ((1 - bit_pos) / bit_pos.clamp_min(1e-4)).sqrt().clamp(1, 12)
    action_encoding = torch.cat((cat[:, :2], bits), 1).float()
    report = {
        "counts": counts,
        "seed": seed,
        "batch": batch,
        "actions": len(cat),
        "mirrored_training": mirror,
        "teacher_switch_weight": switch_weight,
        "train_switch_fraction": switch.float().mean().item(),
        "bc_steps": bc_steps,
        "world_steps": world_steps,
        "rl_steps": rl_steps,
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
            save_native(actor, commands, output / (name + ".bin"))
        actor.train(was_training)
        print(name, report[name + "_holdout_exact"], flush=True)

    opt = torch.optim.AdamW(actor.parameters(), lr=0.00065)
    for step in range(bc_steps):
        i = indices(len(s))
        logits, factor = actor(s[i])
        w = rare[expert[i]] * corrective_weight[i]
        loss = (
            F.cross_entropy(logits, expert[i], reduction="none") * w
        ).sum() / w.sum()
        update(opt, loss + 0.30 * factor_loss(factor, expert[i]))
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
    hs, hy, _, hn, _, hd, hg = data["holdout"]
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
    aopt = torch.optim.AdamW(actor.parameters(), lr=0.00009)
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
            logits, factor = actor(x)
            loss = (F.cross_entropy(logits, u, reduction="none") * aw).sum() / aw.sum()
            lab = expert[i]
            w = rare[lab] * corrective_weight[i]
            supervision = (
                F.cross_entropy(logits, lab, reduction="none") * w
            ).sum() / w.sum()
            update(
                aopt,
                loss
                + 0.7 * F.kl_div(logits.log_softmax(1), prior, reduction="batchmean")
                + 0.85 * supervision
                + 0.2 * factor_loss(factor, lab),
            )
        if (step + 1) % 2000 == 0:
            print("iql", step + 1, flush=True)
    save("world_iql")
    torch.save(
        {
            "actor": actor.cpu().state_dict(),
            "catalogue": cat.cpu(),
            "world": world.cpu().state_dict(),
            "q1": q1.cpu().state_dict(),
            "q2": q2.cpu().state_dict(),
            "value": val.cpu().state_dict(),
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
    args = parser.parse_args()
    if min(args.bc_steps, args.world_steps, args.rl_steps, args.batch) < 1:
        parser.error("Positive budgets required")
    if not np.isfinite(args.switch_weight) or args.switch_weight < 1:
        parser.error("Switch weight must be finite and at least one")
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
    )

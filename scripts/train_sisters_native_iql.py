"""Offline IQL on timestamped native control decisions, with one-step model value expansion."""

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

PHYSICAL = 56
STATE = 3 * PHYSICAL + 12
PULSES = (1, 4, 8, 16, 32, 256)
HOLDS = (2, 128, 32)
MIRROR_AXES = (0, 2, 10, 13, 15, 23, 25, 33, 35, 43, 45, 50, 52)
REWARD_WEIGHTS = np.array([0.45 / 21, -1.5, -1.5, -2.0, -1.5, 0, 8, -6], "f4")


def decision_states(raw: np.ndarray, t: np.ndarray, buttons: np.ndarray) -> np.ndarray:
    """Features at decision i only use observed frames and inputs issued before i."""
    index = np.arange(len(t))
    # All ages use actual game time, including stalls.
    prev = np.r_[0, buttons[:-1]]
    prev2 = np.r_[0, prev[:-1]]
    extra = []
    for bit in PULSES:
        start = ((prev & bit) != 0) & ((prev2 & bit) == 0)
        last = np.maximum.accumulate(np.where(start, np.r_[t[0], t[:-1]], -100.0))
        extra.append(np.minimum(t - last, 1.0))
    for bit in HOLDS:
        held = (prev & bit) != 0
        start = held & ((prev2 & bit) == 0)
        last = np.maximum.accumulate(np.where(start, np.r_[t[0], t[:-1]], -100.0))
        extra.extend([held, np.where(held, np.minimum(t - last, 1.0), 0)])
    return np.column_stack(
        (raw, raw[(index - 1).clip(0)], raw[(index - 4).clip(0)], *extra)
    ).astype("f4")


def attribute_events(
    t: np.ndarray, obs_t: np.ndarray, ticks: np.ndarray, events: np.ndarray
):
    # Each snapshot row carries events from the ensuing Gym interval. The last
    # native decision strictly before its end gets the reward, once per event.
    end = np.r_[obs_t[1:], obs_t[-1] + ticks[-1] * 0.02]
    slot = np.searchsorted(t, end - 1e-6, side="left") - 1
    valid = (slot >= 0) & (end >= t[0])
    credited = np.zeros((len(t), 8), "f4")
    np.add.at(credited, slot[valid], events[valid])
    return credited


def load_data(folder: Path, device: str):
    games = json.loads((folder / "manifest.json").read_text())
    if len({x["source"] for x in games}) != len(games):
        raise ValueError("Duplicate development episode")
    sets = {name: [] for name in ("train", "holdout")}
    for game in games:
        if "official" in game["source"] or "exam-" in game["source"]:
            raise ValueError("Formal evaluation cannot be used for training")
        with np.load(folder / game["file"], allow_pickle=False) as f:
            t, a = f["control_time"], f["controls"].astype("i8")
            corrected = "teacher_controls" in f
            teacher = f["teacher_controls"].astype("i8") if corrected else a
            if teacher.shape != a.shape:
                raise ValueError(
                    "Teacher labels and executed actions have different shapes"
                )
            if len(t) < 2 or not np.all(np.diff(t) > 0):
                raise ValueError("Invalid native control clock")
            state = decision_states(f["student_features"], t, a[:, 2])
            events = attribute_events(t, f["time"], f["elapsed_ticks"], f["events"])
        end = events[:, 6:].any(1)
        if end.any() and np.where(end)[0][0] < len(t) - 1:
            stop = np.where(end)[0][0] + 1
            state, a, teacher, events, t, end = (
                x[:stop] for x in (state, a, teacher, events, t, end)
            )
        step = np.r_[np.diff(t), 0.04].astype("f4")
        # Unknown successors at truncated recordings are excluded from TD fitting.
        good = (step <= 0.24) & (np.r_[np.ones(len(t) - 1, bool), False] | end)
        successor = np.r_[state[1:], state[-1:]]
        reward = events @ REWARD_WEIGHTS
        discount = (0.995 ** (step / 0.04)) * (~end)
        sets[game["split"]].append(
            (
                state,
                a,
                successor,
                reward,
                discount,
                good,
                np.full(len(t), game["won"], bool),
                teacher,
                np.full(len(t), corrected, bool),
            )
        )

    data = {}
    catalogue = None
    for split, parts in sets.items():
        s, a, nxt, r, d, good, won, teacher, corrected = [
            np.concatenate(x) for x in zip(*parts)
        ]
        if split == "train":
            m_s, m_nxt, m_a = s.copy(), nxt.copy(), a.copy()
            flip = [i + j for i in (0, 56, 112) for j in MIRROR_AXES]
            m_s[:, flip] *= -1
            m_nxt[:, flip] *= -1
            m_a[:, 0] *= -1
            s, nxt, a = (np.concatenate(v) for v in ((s, m_s), (nxt, m_nxt), (a, m_a)))
            m_teacher = teacher.copy()
            m_teacher[:, 0] *= -1
            teacher = np.concatenate((teacher, m_teacher))
            r, d, good, won, corrected = (
                np.tile(x, 2) for x in (r, d, good, won, corrected)
            )
            catalogue = np.unique(np.concatenate([a, teacher]), axis=0)
        index = {tuple(cmd): i for i, cmd in enumerate(catalogue)}
        label = np.array([index.get(tuple(cmd), -1) for cmd in a], "i8")
        expert = np.array([index.get(tuple(cmd), -1) for cmd in teacher], "i8")
        data[split] = tuple(
            torch.as_tensor(x, device=device)
            for x in (s, label, nxt, r, d, good, won, expert, corrected)
        )
    counts = {
        k: {"fights": len(v), "native_decisions": sum(len(x[0]) for x in v)}
        for k, v in sets.items()
    }
    return data, catalogue, counts


def mlp(inputs: int, outputs: int):
    return nn.Sequential(
        nn.Linear(inputs, 160),
        nn.ReLU(),
        nn.Linear(160, 128),
        nn.ReLU(),
        nn.Linear(128, outputs),
    )


class Policy(nn.Module):
    def __init__(self, catalogue: Tensor):
        super().__init__()
        buttons, codes = torch.unique(catalogue[:, 2], sorted=True, return_inverse=True)
        self.register_buffer(
            "columns",
            torch.stack((catalogue[:, 0] + 1, catalogue[:, 1] + 4, codes + 6), dim=1),
        )
        self.net = mlp(STATE, 6 + len(buttons))

    def forward(self, state: Tensor) -> Tensor:
        raw = self.net(state)
        return raw[:, self.columns].sum(-1)


def export(actor: Policy, catalogue: np.ndarray, output: Path):
    # Fold the categorical factors into one linear output per observed primitive.
    weights = [p.detach().cpu().numpy() for p in actor.net.parameters()]
    col = actor.columns.cpu().numpy()
    weights[-2] = weights[-2][col].sum(1)
    weights[-1] = weights[-1][col].sum(1)
    with output.open("wb") as f:
        f.write(struct.pack("<5i", 0x314C5149, STATE, 160, 128, len(col)))
        f.write(catalogue.astype("<i4").tobytes())
        for w in weights:
            f.write(w.astype("<f4").tobytes())


def quantile_loss(error: Tensor, tau: float):
    return (torch.where(error >= 0, tau, 1 - tau) * error.square()).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bc-steps", type=int, default=5500)
    p.add_argument("--world-steps", type=int, default=2200)
    p.add_argument("--iql-steps", type=int, default=4500)
    p.add_argument("--batch", type=int, default=768)
    p.add_argument("--seed", type=int, default=419)
    p.add_argument(
        "--corrective",
        action="store_true",
        help="Use corrective labels, weight rare combat inputs and train a compute-matched BC control",
    )
    a = p.parse_args()
    if min(a.bc_steps, a.world_steps, a.iql_steps, a.batch) < 1:
        p.error("Positive training budgets required")
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    data, commands, counts = load_data(a.data, device)
    cat = torch.as_tensor(commands, device=device)
    s, y, nxt, r, d, good, won, expert, corrected = data["train"]
    if corrected.any().item() != a.corrective:
        raise ValueError(
            "Use --corrective exactly when demonstration correction data is supplied"
        )
    valid = good.nonzero().flatten()
    # Input to dynamics contains interpretable keys, not ordinal class numbers.
    cmd = torch.stack(
        [cat[:, 0], cat[:, 1], *[(cat[:, 2] >> i) & 1 for i in range(9)]], 1
    ).float()
    actor = Policy(cat).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=0.00075)
    freq = expert.bincount(minlength=len(cat)).float().clamp_min(1)
    balance = (freq.max() / freq).pow(0.23).clamp_max(4)
    if a.corrective:
        buttons = cat[:, 2]
        balance *= torch.where((buttons & 4) != 0, 8.0, 1.0)
        balance *= torch.where((buttons & (16 | 256)) != 0, 2.0, 1.0)
    label_weight = balance[expert] * (1 + won.float()) * (1 + 2 * corrected.float())
    report = {
        "seed": a.seed,
        "device": device,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "counts": counts,
        "actions": len(cat),
        "manifest_sha256": hashlib.sha256(
            (a.data / "manifest.json").read_bytes()
        ).hexdigest(),
        "bc_steps": a.bc_steps,
        "world_steps": a.world_steps,
        "iql_steps": a.iql_steps,
        "corrective": a.corrective,
        "batch": a.batch,
    }

    def log(**kw):
        print(json.dumps(kw), flush=True)

    def sample(tensor):
        return torch.randint(len(tensor), (a.batch,), device=device)

    def update(optimizer, loss):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def save(stage, policy):
        export(policy, commands, out / (stage + ".bin"))
        with torch.no_grad():
            vs, vy = data["holdout"][:2]
            correct = sum(
                (policy(x).argmax(1) == z).sum().item()
                for x, z in zip(vs.split(2048), vy.split(2048))
            )
        report[stage + "_holdout_exact"] = correct / len(vy)
        log(stage=stage, accuracy=correct / len(vy))

    for step in range(a.bc_steps):
        i = sample(s)
        weight = label_weight[i]
        loss = F.cross_entropy(actor(s[i]), expert[i], reduction="none")
        update(opt, (weight * loss).sum() / weight.sum())
        if (step + 1) % 1000 == 0:
            log(stage="behaviour", step=step + 1)
    save("bc", actor)
    ref = copy.deepcopy(actor).requires_grad_(False)
    world = mlp(STATE + 11, PHYSICAL).to(device)
    wopt = torch.optim.AdamW(world.parameters(), lr=0.001)
    for _ in range(a.world_steps):
        i = valid[sample(valid)]
        i = i[d[i] > 0]
        delta = world(torch.cat((s[i], cmd[y[i]]), 1))
        update(
            wopt,
            F.smooth_l1_loss(delta, nxt[i, :PHYSICAL] - s[i, :PHYSICAL], beta=0.03),
        )
    world.eval().requires_grad_(False)
    vs, vy, vnext, _, vd, vg = data["holdout"][:6]
    vi = ((vy >= 0) & vg & (vd > 0)).nonzero().flatten()[::4]
    with torch.no_grad():
        predictions = torch.cat(
            [
                vs[i, :PHYSICAL] + world(torch.cat((vs[i], cmd[vy[i]]), 1))
                for i in vi.split(2048)
            ]
        )
        report["world_next_mae"] = (
            (predictions - vnext[vi, :PHYSICAL]).abs().mean().item()
        )
        report["persistence_mae"] = (
            (vs[vi, :PHYSICAL] - vnext[vi, :PHYSICAL]).abs().mean().item()
        )
    # Use holdout only for reporting. The fraction of model value expansion is fixed.
    q1, q2 = (mlp(STATE, len(cat)).to(device) for _ in range(2))
    v = mlp(STATE, 1).to(device)
    t1, t2 = (copy.deepcopy(q).requires_grad_(False) for q in (q1, q2))
    qopt = torch.optim.AdamW([*q1.parameters(), *q2.parameters()], lr=0.00035)
    vopt = torch.optim.AdamW(v.parameters(), lr=0.00035)
    aopt = torch.optim.AdamW(actor.parameters(), lr=0.00012)
    clone = copy.deepcopy(actor) if a.corrective else None
    clone_opt = (
        torch.optim.AdamW(clone.parameters(), lr=0.00012) if clone is not None else None
    )
    for step in range(a.iql_steps):
        i = valid[sample(valid)]
        x, action = s[i], y[i]
        key = action[:, None]
        with torch.no_grad():
            old_q = torch.minimum(t1(x), t2(x)).gather(1, key).squeeze(1)
            next_m = nxt[i].clone()
            next_m[:, :PHYSICAL] = x[:, :PHYSICAL] + world(
                torch.cat((x, cmd[action]), 1)
            )
            boot = 0.75 * v(nxt[i]).flatten() + 0.25 * v(next_m).flatten()
            target = r[i] + d[i] * boot
        value = v(x).flatten()
        update(vopt, quantile_loss(old_q - value, 0.7))
        update(
            qopt,
            sum(
                F.smooth_l1_loss(q(x).gather(1, key).flatten(), target)
                for q in (q1, q2)
            ),
        )
        with torch.no_grad():
            for target_net, cur in ((t1, q1), (t2, q2)):
                for tp, cp in zip(target_net.parameters(), cur.parameters()):
                    tp.lerp_(cp, 0.005)
            advantage = (old_q - value.detach()).clamp(-1, 1)
            # A corrective teacher action was not necessarily executed. Only
            # real controls have a measured transition and an IQL advantage.
            weight = (
                (advantage * (12 if a.corrective else 2)).exp().clamp_max(20)
                * balance[action]
                * (1 + won[i].float())
            )
            prior = ref(x).softmax(1)
        logits = actor(x)
        ce = F.cross_entropy(logits, action, reduction="none")
        kl = F.kl_div(logits.log_softmax(1), prior, reduction="batchmean")
        policy_loss = (ce * weight).sum() / weight.sum() + 0.7 * kl
        if clone is not None and clone_opt is not None:
            w = label_weight[i]
            teacher_ce = F.cross_entropy(logits, expert[i], reduction="none")
            extra = w * corrected[i]
            policy_loss += (teacher_ce * extra).sum() / w.sum()
            # Same architecture, initialization, mini-batches, update count and
            # prior regularization, without Bellman or advantage weighting.
            clone_logits = clone(x)
            clone_ce = F.cross_entropy(clone_logits, expert[i], reduction="none")
            clone_kl = F.kl_div(
                clone_logits.log_softmax(1), prior, reduction="batchmean"
            )
            update(clone_opt, (clone_ce * w).sum() / w.sum() + 0.7 * clone_kl)
        update(aopt, policy_loss)
        if (step + 1) % 1000 == 0:
            log(stage="iql", step=step + 1, mean_abs_adv=advantage.abs().mean().item())
    save("world_iql", actor)
    if clone is not None:
        save("matched_bc", clone)
    torch.save(
        {
            "catalogue": cat.cpu(),
            "world": world.cpu().state_dict(),
            "actor": actor.cpu().state_dict(),
            "q1": q1.cpu().state_dict(),
            "q2": q2.cpu().state_dict(),
            "value": v.cpu().state_dict(),
        },
        out / "learner.pt",
    )
    report["actor_delta_l2"] = (
        sum(
            (x.cpu() - z.cpu()).square().sum().item()
            for x, z in zip(actor.parameters(), ref.parameters())
        )
        ** 0.5
    )
    (out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    log(**report)


if __name__ == "__main__":
    main()

"""Continue the frozen nine-win router with an offline deep actor-critic on actual randomized blocks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from sisters_control_blocks import neural_block_starts
from torch import nn
from torch.nn import functional as F
from train_sisters_option_ppo import option_rewards, verified_events
from train_sisters_timing_iql import (
    ROUTER_RESIDUAL_MAGIC,
    TimingPolicy,
    motion_history,
    option_components,
    phase_checkpoint,
    phase_components,
    save_native,
)

RAW = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    13,
    19,
    25,
    26,
    27,
    28,
    34,
    61,
    62,
    97,
    98,
    132,
    133,
    134,
    135,
    146,
    147,
    148,
    149,
]
WORLD = [0, 1, 2, 3, 25, 26, 27, 28, 61, 62, 97, 98, 133, 134, 135, 147, 148, 149]
PRESENCE = [
    -1,
    -1,
    -1,
    -1,
    24,
    24,
    24,
    24,
    60,
    60,
    96,
    96,
    132,
    132,
    132,
    146,
    146,
    146,
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def features(
    embedding: np.ndarray, state: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    projection = weights[: 192 * 12].reshape(192, 12).astype("f8")
    mean, scale = weights[192 * 12 : 193 * 12], weights[193 * 12 : 194 * 12]
    coefficient = weights[194 * 12 :]
    h = np.clip((embedding.astype("f8") @ projection - mean) / scale, -6, 6)
    return np.column_stack((h / 3, state[:, RAW])).astype("f4"), h @ coefficient[
        1:
    ] + coefficient[0]


class Change(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).flatten().tanh()


class ActionValue(nn.Module):
    """The Q and next-observation heads share the learned state-action representation."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(40, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU()
        )
        self.value = nn.Linear(48, 2)
        self.transition = nn.Linear(48, 2 * len(WORLD))

    def forward(self, x):
        h = self.encoder(x)
        return self.value(h), self.transition(h).reshape(-1, 2, len(WORLD))


def packet(actor: Change, bound: float, parent: bytes) -> bytes:
    op, base = phase_components(parent)
    option_components(base)
    arrays = [np.array([bound], "<f4")] + [
        p.detach().cpu().numpy().astype("<f4").ravel() for p in actor.net.parameters()
    ]
    extended = (
        struct.pack("<2i", ROUTER_RESIDUAL_MAGIC, len(base))
        + base
        + np.concatenate(arrays).tobytes()
    )
    return phase_checkpoint(op, extended)


def data_rows(data: Path, spec: dict, parent: bytes, learner: Path, clean_cost: float):
    opening, route = phase_components(parent)
    primitive, alternate, weights = option_components(route)
    if (
        digest(phase_checkpoint(opening, primitive)) != spec["actor_sha256"]
        or digest(phase_checkpoint(opening, alternate))
        != spec["alternate_policy_sha256"]
        or spec["candidate_probability"] != 0.5
        or spec["block_game_seconds"] != 0.8
    ):
        raise ValueError("Randomized controllers differ from the winning parent")
    ck = torch.load(learner, map_location="cpu", weights_only=True)
    net = TimingPolicy(ck["catalogue"], True).eval().requires_grad_(False)
    net.load_state_dict(ck["actor"])
    with tempfile.TemporaryDirectory() as temp:
        export = Path(temp) / "actor.bin"
        save_native(net, ck["catalogue"].numpy(), export)
        raw = export.read_bytes()
    n = len(ck["catalogue"]) * 193 * 4
    if raw[:-n] != primitive[:-n] or not np.allclose(
        np.frombuffer(raw[-n:], "<f4"),
        np.frombuffer(primitive[-n:], "<f4"),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("The embedded parent encoder or action catalogue changed")
    result = []
    seen = set()
    with zipfile.ZipFile(data) as z, torch.inference_mode():
        for g in json.loads(z.read("manifest.json")):
            seed = int(g["source"].split("/")[-1])
            split = g["split"]
            if (
                g["source"] != f"native-development/{seed}"
                or seed in seen
                or g["teacher_labels"]
                or seed
                not in spec["train_seeds" if split == "train" else "heldout_seeds"]
                or g["student_policy_sha256"] != spec["actor_sha256"]
                or g["alternate_policy_sha256"] != spec["alternate_policy_sha256"]
                or g["candidate_probability"] != 0.5
            ):
                raise ValueError("Unexpected development trajectory")
            seen.add(seed)
            raw = z.read(g["file"])
            if digest(raw) != g["sha256"]:
                raise ValueError("Changed block trajectory")
            with np.load(io.BytesIO(raw)) as tape:
                s, t = tape["timing_state"], tape["control_time"]
                ev = verified_events(
                    t, s[:, 4] * 9, tape["time"], tape["elapsed_ticks"], tape["events"]
                )
                terminal = np.flatnonzero(ev[:, 6:].any(1))
                if len(terminal) != 1 or bool(ev[:, 6].sum()) != bool(g["won"]):
                    raise ValueError("Missing real fight terminal")
                starts = neural_block_starts(t)
                arms = tape["candidate_draw"].astype(int)
                if not np.array_equal(
                    starts, tape["block_starts"]
                ) or not np.array_equal(
                    np.repeat(arms[starts], np.diff(np.r_[starts, len(t)])), arms
                ):
                    raise ValueError("Within-block arm changes")
                rows = starts[(s[starts, 23] > 0.5) & (starts <= terminal[0])]
                if not len(rows):
                    continue
                successor = np.r_[rows[1:], terminal[0]]
                history = torch.from_numpy(motion_history(t, s))
                emb = net.encode(history[rows]).numpy()
                x, margin = features(emb, s[rows], weights)
                nx, _ = features(
                    net.encode(history[successor]).numpy(), s[successor], weights
                )
                reward, discount, _ = option_rewards(
                    t, ev[:, 1:5].sum(1), ev, rows, clean_cost
                )
                target = s[successor][:, WORLD] - s[rows][:, WORLD]
                present = np.array(PRESENCE)
                valid = (
                    (present < 0)
                    | ((s[successor][:, present] > 0.5) & (s[rows][:, present] > 0.5))
                ) & (discount[:, None] > 0)
                result.append(
                    {
                        "seed": np.full(len(rows), seed),
                        "train": np.full(len(rows), split == "train"),
                        "x": x,
                        "nx": nx,
                        "margin": margin.astype("f4"),
                        "a": arms[rows],
                        "reward": reward.astype("f4"),
                        "gamma": discount.astype("f4"),
                        "dy": target,
                        "valid": valid,
                    }
                )
    if seen != set(spec["train_seeds"] + spec["heldout_seeds"]):
        raise ValueError("Incomplete assigned games")
    return {k: np.concatenate([g[k] for g in result]) for k in result[0]}


def fit_critic(batch, train_mask, device, seed, updates):
    torch.manual_seed(seed)
    ix = (
        torch.flatnonzero(train_mask)
        if hasattr(torch, "flatnonzero")
        else torch.where(train_mask)[0]
    )
    q1, q2 = ActionValue().to(device), ActionValue().to(device)
    tar1, tar2 = (
        copy.deepcopy(q1).requires_grad_(False),
        copy.deepcopy(q2).requires_grad_(False),
    )
    v = nn.Sequential(
        nn.Linear(40, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)
    ).to(device)
    optim = torch.optim.AdamW(
        [*q1.parameters(), *q2.parameters(), *v.parameters()],
        lr=0.0007,
        weight_decay=0.001,
    )
    for _ in range(updates):
        i = ix[torch.randint(len(ix), (96,), device=device)]
        x = batch["x"][i]
        a = batch["a"][i]
        rows = torch.arange(len(i), device=device)
        with torch.no_grad():
            qref = torch.minimum(tar1(x)[0], tar2(x)[0])[rows, a]
            target = (
                batch["reward"][i] + batch["gamma"][i] * v(batch["nx"][i]).flatten()
            )
        gap = qref - v(x).flatten()
        loss = (torch.where(gap > 0, 0.7, 0.3) * gap.square()).mean()
        m = batch["valid"][i]
        for q in (q1, q2):
            values, dy = q(x)
            loss = loss + F.smooth_l1_loss(values[rows, a], target)
            error = F.smooth_l1_loss(dy[rows, a], batch["dy"][i], reduction="none")
            loss = loss + 0.4 * (error * m).sum() / m.sum().clamp_min(1)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [*q1.parameters(), *q2.parameters(), *v.parameters()], 3.0
        )
        optim.step()
        with torch.no_grad():
            for online, target_net in ((q1, tar1), (q2, tar2)):
                for p, t in zip(
                    online.parameters(), target_net.parameters(), strict=True
                ):
                    t.lerp_(p, 0.02)
    with torch.no_grad():
        values = torch.minimum(q1(batch["x"])[0], q2(batch["x"])[0])
    return values, {
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "value": v.state_dict(),
    }


def train(
    data: Path,
    experiment: Path,
    parent: Path,
    learner: Path,
    output: Path,
    updates: int = 900,
    device: str = "cpu",
    clean_cost: float = 0.0,
) -> None:
    if output.exists() or updates < 1 or clean_cost < 0:
        raise ValueError("Expected a new run and positive training budget")
    torch.set_num_threads(1)
    spec = json.loads(experiment.read_bytes())
    base = parent.read_bytes()
    arrays = data_rows(data, spec, base, learner, clean_cost)
    b = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
    tr = b["train"]
    groups = sorted(spec["train_seeds"])
    scores = torch.zeros((len(tr), 2), device=device)
    for fold in range(3):
        held = tr & torch.isin(b["seed"], torch.tensor(groups[fold::3], device=device))
        q, _ = fit_critic(b, tr & ~held, device, 75121 + fold, updates)
        scores[held] = q[held]
    q, critics = fit_critic(b, tr, device, 75128, updates)
    scores[~tr] = q[~tr]
    torch.manual_seed(75132)
    actor = Change().to(device)
    bound = max(
        0.025, float(np.quantile(np.abs(arrays["margin"][arrays["train"]]), 0.5))
    )
    zero_packet = packet(actor, bound, base)
    with torch.no_grad():
        if torch.count_nonzero(actor(b["x"])):
            raise ValueError("Residual must exactly preserve the initial selector")
    prior = (b["margin"] / bound).clamp(-10, 10)
    p0 = prior.sigmoid().clamp(0.0001, 0.9999)
    # The cross-fitted Q contrast optimizes the actor. Keep the parent where actual support is weak.
    advantage = (scores[:, 1] - scores[:, 0]).clamp(-4, 4)
    scale = advantage[tr].abs().median().clamp_min(0.2)
    advantage = (advantage / scale).clamp(-3, 3)
    opt = torch.optim.AdamW(actor.parameters(), lr=0.0015, weight_decay=0.01)
    for _ in range(180):
        p = (prior[tr] + actor(b["x"][tr])).sigmoid().clamp(0.0001, 0.9999)
        old = p0[tr]
        kl = p * (p.log() - old.log()) + (1 - p) * (torch.log1p(-p) - torch.log1p(-old))
        loss = -(p * advantage[tr] - 0.8 * kl).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        delta = actor(b["x"]).cpu().numpy() * bound
    margin = arrays["margin"]
    ref = margin > 0
    # Resolve the maximum allowed training-state drift without looking at the historical holdout.
    gain = next(
        float(g)
        for g in np.linspace(1.0, 0.0, 101)
        if (((margin + g * delta) > 0) != ref)[arrays["train"]].mean() <= 0.12
    )
    # All serialized bounds must be positive. A zero gain encodes exact parent equivalence.
    if gain == 0:
        nn.init.zeros_(actor.net[-1].weight)
        nn.init.zeros_(actor.net[-1].bias)
    deployed_bound = bound * (gain if gain else 1.0)
    route = margin + gain * delta > 0
    changed = route != ref
    diagnostics = []
    for seed in sorted(set(arrays["seed"])):
        m = arrays["seed"] == seed
        actual = arrays["a"][m]
        reward = arrays["reward"][m]
        ips = (
            2
            * ((route[m] == actual).astype(float) - (ref[m] == actual).astype(float))
            * reward
        )
        diagnostics.append(
            {
                "seed": int(seed),
                "train": bool(arrays["train"][m][0]),
                "blocks": int(m.sum()),
                "changed": int(changed[m].sum()),
                "immediate_reward_ips_change": float(ips.mean()),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        out = Path(temp) / "candidate"
        out.mkdir()
        native = packet(actor, deployed_bound, base)
        (out / "world_iql.bin").write_bytes(native)
        (out / "parent-equivalent.bin").write_bytes(zero_packet)
        torch.save(
            {"actor": actor.cpu().state_dict(), "critics": critics, "bound": deployed_bound},
            out / "learner.pt",
        )
        report = {
            "method": "Offline neural actor-critic from the exact nine-win two-policy selector. Real 0.8-second returns, twin Q, expectile V, auxiliary action-conditioned motion and cross-fitted actor improvement.",
            "device": device,
            "critic_updates_per_fit": updates,
            "folds": 3,
            "actor_steps": 180,
            "actor_parameters": sum(p.numel() for p in actor.parameters()),
            "training_fights": len(groups),
            "training_blocks": int(arrays["train"].sum()),
            "historical_holdout_blocks": int((~arrays["train"]).sum()),
            "training_changes": int(changed[arrays["train"]].sum()),
            "historical_holdout_changes": int(changed[~arrays["train"]].sum()),
            "raw_bound": bound,
            "applied_fraction": gain,
            "episodes": diagnostics,
            "clean_cost": clean_cost,
            "parent_sha256": digest(base),
            "initial_equivalent_sha256": digest(zero_packet),
            "policy_sha256": digest(native),
            "archive_sha256": digest(data.read_bytes()),
            "source_learner_sha256": digest(learner.read_bytes()),
        }
        (out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        out.rename(output)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "experiment", "parent", "learner", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=900)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-cost", type=float, default=0.0)
    train(**vars(parser.parse_args()))

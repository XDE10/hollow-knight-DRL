"""Improve the nine-win policy's native keys using measured combat returns and shadow labels."""

from __future__ import annotations

import argparse
import io
import json
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_sisters_champion_router import WORLD, digest, features
from train_sisters_timing_iql import (
    OPTION_MAGIC,
    TimingPolicy,
    option_components,
    phase_checkpoint,
    phase_components,
)


def read_leaf(raw: bytes):
    magic, width, h1, h2, n = struct.unpack_from("<5i", raw)
    if (magic, width, h1, h2) != (0x344C5149, 192, 256, 192) or n < 2:
        raise ValueError("Expected a native timing-history actor")
    cat = np.frombuffer(raw, "<i4", count=n * 3, offset=20).reshape(n, 3).copy()
    model = TimingPolicy(torch.from_numpy(cat), True).eval().requires_grad_(False)
    offset = 20 + 12 * n
    for layer in (
        model.boss,
        model.disc,
        model.past,
        model.dense[0],
        model.dense[2],
        model.out,
    ):
        for weight in (layer.weight, layer.bias):
            a = np.frombuffer(raw, "<f4", count=weight.numel(), offset=offset)
            weight.copy_(torch.from_numpy(a.copy()).reshape_as(weight))
            offset += 4 * weight.numel()
    if offset != len(raw):
        raise ValueError("Unexpected primitive weights")
    model.factors.weight.zero_()
    model.factors.bias.zero_()
    return model, cat


def replace_head(raw, delta, gain):
    w = delta.weight.detach().cpu().numpy().astype("<f4")
    b = delta.bias.detach().cpu().numpy().astype("<f4")
    adjustment = np.r_[w.ravel(), b] * np.float32(gain)
    if not np.isfinite(adjustment).all():
        raise ValueError("Invalid primitive head")
    tail = np.frombuffer(raw[-4 * len(adjustment) :], "<f4")
    return (
        raw[: -4 * len(adjustment)] + (tail + adjustment).astype("<f4").tobytes()
        if gain
        else raw
    )


def load_data(path, models, cats, gate, device):
    by_arm = [[], []]
    lookup = [{tuple(c): i for i, c in enumerate(cat)} for cat in cats]
    with zipfile.ZipFile(path) as z, torch.inference_mode():
        manifest = json.loads(z.read("manifest.json"))
        for game in manifest["games"]:
            raw = z.read(game["file"])
            if digest(raw) != game["sha256"]:
                raise ValueError("Window data changed")
            with np.load(io.BytesIO(raw)) as d:

                def encode(net, key):
                    return net.encode(torch.from_numpy(d[key]).to(device))

                gate_hist = encode(models[0], "gate_history").cpu().numpy()
                _, score = features(gate_hist, d["gate_history"][:, :192], gate)
                used = (score[d["gate_index"]] > 0) & d["gate_trio"]
                ev = d["events"]
                mask = ev[:, 1:5].sum(1)
                reward = (
                    0.45 / 21 * ev[:, 0]
                    - 1.9 * mask
                    - 0.5 * ev[:, 3]
                    + 12 * (ev[:, 6] - ev[:, 7])
                ).clip(-16, 16)
                for arm, net in enumerate(models):
                    take = np.flatnonzero(used == arm)
                    if not len(take):
                        continue
                    h = (
                        net.encode(torch.from_numpy(d["history"][take]).to(device))
                        .cpu()
                        .numpy()
                    )
                    actual = np.array(
                        [lookup[arm].get(tuple(c), -1) for c in d["issued"][take]]
                    )
                    label = np.array(
                        [lookup[arm].get(tuple(c), -1) for c in d["label"][take]]
                    )
                    by_arm[arm].append(
                        {
                            "h": h,
                            "a": actual,
                            "label": label,
                            "hurt": mask[take] > 0,
                            "reward": reward[take],
                            "student": d["student_only"][take],
                            "train": np.full(len(take), game["split"] == "train"),
                            "seed": np.full(
                                len(take), int(game["source"].split("/")[-1])
                            ),
                            "world": d["world_delta"][take],
                            "visible": d["world_mask"][take],
                        }
                    )
    return [
        {
            k: torch.from_numpy(np.concatenate([g[k] for g in arm])).to(device)
            for k in arm[0]
        }
        for arm in by_arm
    ]


class Outcome(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(192, 64), nn.LayerNorm(64), nn.SiLU())
        self.action = nn.Embedding(actions, 64)
        self.state_value = nn.Linear(64, 1)
        self.factual = nn.Sequential(
            nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 1 + len(WORLD))
        )

    def forward(self, h, actions):
        state = self.encoder(h)
        after = self.factual(state + self.action(actions))
        return self.state_value(state).flatten(), after[:, 0], after[:, 1:]


def fit(leaf, net, d, steps, device):
    h, a = d["h"], d["a"]
    train = d["train"] & (a >= 0)
    indices = torch.where(train)[0]
    base = net.out(h).detach()
    value = Outcome(base.shape[1]).to(device)
    opt = torch.optim.AdamW(value.parameters(), lr=0.0008)
    # Targets describe the recorded next 0.8 seconds under the recorded ensuing inputs.
    for _ in range(steps):
        ix = indices[torch.randint(len(indices), (256,), device=device)]
        v, q, dy = value(h[ix], a[ix])
        m = d["visible"][ix]
        err = F.smooth_l1_loss(dy, d["world"][ix], reduction="none")
        loss = (
            F.smooth_l1_loss(v, d["reward"][ix])
            + F.smooth_l1_loss(q, d["reward"][ix])
            + 0.5 * (m * err).sum() / m.sum().clamp_min(1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), 2.0)
        opt.step()
    with torch.no_grad():
        v, q, dy = value(h, a.clamp_min(0))
        # Favor observed advantageous actions, shrink noisy single-window returns toward the learned Q.
        adv = (0.5 * (q + d["reward"]) - v).clamp(-3, 3)
        weight = adv.exp().clamp(0.08, 4.0)
    delta = nn.Linear(192, base.shape[1]).to(device)
    nn.init.zeros_(delta.weight)
    nn.init.zeros_(delta.bias)
    if replace_head(leaf, delta, 0.0) != leaf:
        raise ValueError("Zero-update actor differs from original")
    actor_opt = torch.optim.AdamW(delta.parameters(), lr=0.0015)
    teacher = (d["label"] >= 0) & d["student"] & d["hurt"]
    shadow = torch.where(train & teacher)[0]
    if len(shadow) < 16:
        raise ValueError(
            "Missing shadow labels on student-controlled risky transitions"
        )
    for _ in range(400):
        ordinary = indices[torch.randint(len(indices), (320,), device=device)]
        danger = shadow[torch.randint(len(shadow), (96,), device=device)]
        pred = base[ordinary] + delta(h[ordinary])
        actor_loss = (
            weight[ordinary] * F.cross_entropy(pred, a[ordinary], reduction="none")
        ).mean()
        anchor = F.kl_div(
            pred.log_softmax(1), base[ordinary].softmax(1), reduction="batchmean"
        )
        assist = F.cross_entropy(base[danger] + delta(h[danger]), d["label"][danger])
        loss = (
            actor_loss
            + 0.8 * assist
            + 6.0 * anchor
            + 0.005 * delta(h[ordinary]).square().mean()
        )
        actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(delta.parameters(), 1.0)
        actor_opt.step()
    with torch.no_grad():
        old = base.argmax(1)
        diff = delta(h)
        safe = train & ~d["hurt"]
        gain = next(
            float(k)
            for k in np.linspace(1, 0, 101)
            if ((base + k * diff).argmax(1)[safe] != old[safe]).float().mean() <= 0.015
        )
        new = (base + gain * diff).argmax(1)
        scores = {}
        for split, mask in (
            ("train", d["train"]),
            ("development_holdout", ~d["train"]),
        ):
            keys = mask & teacher
            valid = mask & (a >= 0)
            normal = valid & ~d["hurt"]
            mv = mask[:, None] & d["visible"]
            scores[split] = {
                "sampled_inputs": int(mask.sum()),
                "safe_inputs": int(normal.sum()),
                "changed_safe": int(((new != old) & normal).sum()),
                "changed_inputs": int(((new != old) & mask).sum()),
                "affected_pre_hit_inputs": int(((new != old) & mask & d["hurt"]).sum()),
                "shadow_pre_hit": int(keys.sum()),
                "parent_matches": int(((old == d["label"]) & keys).sum()),
                "new_matches": int(((new == d["label"]) & keys).sum()),
                "value_mae": float((v[valid] - d["reward"][valid]).abs().mean()),
                "world_mae": float(
                    ((dy - d["world"]).abs() * mv).sum() / mv.sum().clamp_min(1)
                ),
                "persistence_mae": float(
                    (d["world"].abs() * mv).sum() / mv.sum().clamp_min(1)
                ),
            }
    return (
        replace_head(leaf, delta, gain),
        {"gain": gain, "scores": scores},
        {
            "critic": value.cpu().state_dict(),
            "delta": delta.cpu().state_dict(),
            "gain": gain,
        },
    )


def train(
    data: Path, parent: Path, output: Path, device: str = "cpu", steps: int = 700
):
    if output.exists() or steps <= 0:
        raise ValueError("Expected a new training run")
    torch.set_num_threads(1)
    torch.manual_seed(191121)
    base = parent.read_bytes()
    opening, router = phase_components(base)
    h, r, gate = option_components(router)
    models, cats = zip(*[read_leaf(x) for x in (h, r)], strict=True)
    models = [net.to(device) for net in models]
    arms = load_data(data, models, cats, gate, device)
    fitted = [
        fit(leaf, net, d, steps, device)
        for leaf, net, d in zip((h, r), models, arms, strict=True)
    ]
    leaves = [x[0] for x in fitted]
    new_route = (
        struct.pack("<3i", OPTION_MAGIC, *map(len, leaves))
        + b"".join(leaves)
        + gate.tobytes()
    )
    result = phase_checkpoint(opening, new_route)
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "fit"
        stage.mkdir()
        (stage / "world_iql.bin").write_bytes(result)
        torch.save(
            {"primary": fitted[0][2], "alternate": fitted[1][2]}, stage / "learner.pt"
        )
        report = {
            "method": "Offline advantage-weighted primitive actor update from real short-window returns, an action-conditioned outcome critic and auxiliary shadow-key supervision. Original router, encoders and opening frozen.",
            "parent_sha256": digest(base),
            "policy_sha256": digest(result),
            "archive_sha256": digest(data.read_bytes()),
            "device": device,
            "critic_steps": steps,
            "actor_steps": 400,
            "primary": fitted[0][1],
            "alternate": fitted[1][1],
            "required_safe_train_drift_max": 0.015,
        }
        (stage / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "parent", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=700)
    train(**vars(parser.parse_args()))

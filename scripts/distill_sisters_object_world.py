"""Distill action-conditioned ensemble forecasts into a small recurrent movement residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sisters_world_data import (
    BINS,
    CONTROL,
    HORIZON,
    PLAYER_TARGET,
    TARGET,
    WorldReplay,
)
from torch import Tensor, nn
from torch.nn import functional as F
from train_sisters_object_world import ObjectWorld

OPTIONS = 12


class MovementStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.history = nn.GRU(56, 32, batch_first=True)
        self.head = nn.Linear(32, OPTIONS)

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.history(x)
        q = self.head(h[:, -1])
        return q - q.mean(-1, keepdim=True)


def reference(path: Path, device: str) -> nn.Module:
    flat = torch.as_tensor(np.loadtxt(path, dtype=np.float32), device=device)
    net = nn.Sequential(nn.Linear(56, 32), nn.ReLU(), nn.Linear(32, OPTIONS)).to(device)
    pos = 0
    with torch.no_grad():
        for parameter in net.parameters():
            end = pos + parameter.numel()
            parameter.copy_(flat[pos:end].reshape_as(parameter))
            pos = end
    if pos != len(flat):
        raise ValueError("unexpected baseline checkpoint size")
    return net.eval()


@torch.inference_mode()
def alternatives(b: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    base = b["action"]
    batch = len(base)
    a = base[:, None].expand(-1, OPTIONS, -1, -1).clone()
    code = torch.arange(OPTIONS, device=base.device)
    move = code // 4 - 1
    jump = code % 4 // 2
    dash = code % 2
    inputs = a[:, :, 0, : BINS * CONTROL].reshape(batch, OPTIONS, BINS, CONTROL)
    inputs[:, :, :, 0] = (move == -1)[None, :, None]
    inputs[:, :, :, 1] = (move == 1)[None, :, None]
    inputs[:, :, :, 4] = jump[None, :, None]
    inputs[:, :, :, 5] = dash[None, :, None]
    p = b["student"][:, -1]
    legal = ((dash == 0)[None] | (p[:, 8] > 0.5)[:, None]) & (
        (jump == 0)[None]
        | (p[:, 6] > 0.5)[:, None]
        | (p[:, 7] > 0.5)[:, None]
        | (base[:, 0, 4] > 0.5)[:, None]
    )
    return a, legal


@torch.inference_mode()
def returns(world: ObjectWorld, b: dict[str, Tensor], action: Tensor) -> Tensor:
    batch, choices = action.shape[:2]
    z, tokens = world.encode(b)
    traj = world.imagine(
        z[:, None].expand(-1, choices, -1).reshape(batch * choices, -1),
        action.reshape(batch * choices, HORIZON, -1),
    )
    traj = traj.reshape(batch, choices, HORIZON, -1)
    calibrated = (world.events(traj) - world.calibration).sigmoid()
    hit, body, disc, spike, other, heal, win, death = calibrated.unbind(-1)
    hero = b["player"][:, None, None, PLAYER_TARGET] + world.player_readout(traj)
    obj = tokens[:, None, None].expand(-1, choices, HORIZON, -1, -1)
    combined = torch.cat(
        [traj[:, :, :, None].expand(-1, -1, -1, obj.shape[3], -1), obj], -1
    )
    estimated = world.geometry(world.object_readout(combined))
    geometry = b["obj"][:, None, None, :, TARGET] + estimated[..., :-1]
    dx = (geometry[..., 5] - hero[:, :, :, None, 0]).abs() * 12
    dy = (geometry[..., 6] - hero[:, :, :, None, 1]).abs() * 9
    width = b["obj"][:, None, None, :, 16] * 2.5
    height = b["obj"][:, None, None, :, 17] * 2.5
    threat = b["mask"] & (
        ((b["obj"][:, :, 1] - 0.2).abs() < 0.01)
        | ((b["obj"][:, :, 1] - 0.6).abs() < 0.01)
        | ((b["obj"][:, :, 1] - 0.8).abs() < 0.01)
    )
    clearance = torch.maximum(dx - width - 0.52, dy - height - 1.05)
    overlap = (
        (0.15 - clearance).mul(2.0).sigmoid()
        * estimated[..., -1].sigmoid()
        * threat[:, None, None]
    )
    collision = overlap.amax(-1)
    reward = (
        4 * hit
        - 12 * body
        - 13 * disc
        - 18 * spike
        - 12 * other
        + 3 * heal
        + 12 * win
        - 28 * death
        - 1.8 * collision
    )
    x = hero[..., 0] * 12 + 30
    y = hero[..., 1] * 9 + 7.4
    fall = (
        torch.maximum(21.35 - x, x - 38.65).add(0.18).mul(4).sigmoid()
        * (8.6 - y).mul(3).sigmoid()
    )
    reward -= 3 * fall
    discount = torch.cumprod(
        torch.cat(
            [torch.ones_like(death[:, :, :1]), 0.96 * (1 - death[:, :, :-1])], -1
        ),
        -1,
    )
    return (reward * discount).sum(-1)


@torch.inference_mode()
def collect_labels(
    models: list[ObjectWorld],
    data: WorldReplay,
    anchor: nn.Module,
    split: str,
    count: int,
) -> tuple[Tensor, ...]:
    x, base_q, mean, spread, masks, labels, outcomes = [], [], [], [], [], [], []
    indices = data.indices[split]
    ids = indices[torch.randperm(len(indices), device=indices.device)[:count]]
    for group in ids.split(96):
        b = data.batch(group)
        actions, legal = alternatives(b)
        q = torch.stack([returns(model, b, actions) for model in models])
        q -= ((q * legal).sum(-1) / legal.sum(-1).clamp_min(1))[..., None]
        ref = anchor(b["student"][:, -1])
        x.append(b["student"].cpu())
        base_q.append(ref.cpu())
        mean.append(q.mean(0).cpu())
        spread.append(q.std(0, unbiased=False).cpu())
        masks.append(legal.cpu())
        labels.append(b["label"].cpu())
        outcomes.append(b["won"].cpu())
    return tuple(
        torch.cat(v) for v in (x, base_q, mean, spread, masks, labels, outcomes)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=40000)
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()
    device = "cuda"
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(8041)
    data = WorldReplay(args.data, device)
    anchor = reference(args.base, device)
    models = []
    for file in sorted(args.models.glob("world-*.pt")):
        model = ObjectWorld().to(device)
        saved = torch.load(file, map_location=device, weights_only=True)
        saved.setdefault("calibration", model.pos_weight.log())
        model.load_state_dict(saved)
        models.append(model.eval())
    if len(models) < 2:
        raise ValueError("Multiple independently trained dynamics models are required")
    args.output.mkdir(parents=True, exist_ok=True)
    calibration = collect_labels(models, data, anchor, "holdout", 4000)
    _, _, cal_q, cal_u, mask, _, _ = calibration
    scale = float(cal_q[mask].abs().quantile(0.78).clamp_min(0.025))
    threshold = float(cal_u[mask].quantile(0.65).clamp_min(0.015))
    print(
        json.dumps(
            {
                "stage": "calibrated",
                "ensemble": len(models),
                "scale": scale,
                "uncertainty": threshold,
            }
        ),
        flush=True,
    )
    replay = collect_labels(models, data, anchor, "train", args.samples)
    x, reference_q, q, u, mask, _actual, _won = (item.to(device) for item in replay)
    for gated in (True, False):
        torch.manual_seed(803)
        model = MovementStudent().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0018)
        certainty = 1 / (1 + (u / threshold).square()) if gated else torch.ones_like(u)
        advantage = (q / scale).clamp(-3, 3) * certainty
        teacher = (
            (reference_q * 0.9 + 1.8 * advantage).masked_fill(~mask, -1e4).softmax(-1)
        )
        for i in range(args.steps):
            ids = torch.randint(len(x), (384,), device=device)
            residual = model(x[ids])
            logits = reference_q[ids] * 0.9 + residual
            masked = logits.masked_fill(~mask[ids], -1e4)
            distill = F.kl_div(
                masked.log_softmax(-1), teacher[ids], reduction="batchmean"
            )
            loss = distill + 0.018 * (residual.square().mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            opt.step()
        label = "gated" if gated else "uniform"
        torch.save(model.state_dict(), args.output / f"{label}.pt")
        np.savetxt(
            args.output / f"{label}.weights",
            np.concatenate(
                [v.detach().cpu().numpy().ravel() for v in model.parameters()]
            ),
            fmt="%.9g",
        )
        with torch.no_grad():
            learned = model(x[:2000])
            change = (
                (
                    (reference_q[:2000] * 0.9 + learned)
                    .masked_fill(~mask[:2000], -1e4)
                    .argmax(-1)
                    != (reference_q[:2000] * 0.9)
                    .masked_fill(~mask[:2000], -1e4)
                    .argmax(-1)
                )
                .float()
                .mean()
            )
        report = {
            "ensemble": len(models),
            "windows": len(x),
            "scale": scale,
            "uncertainty": threshold,
            "mean_gate": float(certainty[mask].mean()),
            "loss": float(loss.detach()),
            "preference_changes": float(change),
        }
        (args.output / f"{label}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({label: report}), flush=True)


if __name__ == "__main__":
    main()

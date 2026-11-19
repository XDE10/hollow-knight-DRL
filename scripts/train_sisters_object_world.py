"""Train an action-conditioned full-entity ensemble on real Sisters game trajectories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sisters_world_data import (
    BINS,
    CONTROL,
    PLAYER_TARGET,
    STATE_BUCKETS,
    TARGET,
    WorldReplay,
)
from torch import Tensor, nn
from torch.nn import functional as F

POS_WEIGHT = [2.8, 10, 12, 25, 15, 12, 28, 22]


class ObjectWorld(nn.Module):
    def __init__(self, object_history: bool = False) -> None:
        super().__init__()
        self.object_history = object_history
        self.tags = nn.ModuleList(
            [nn.Embedding(STATE_BUCKETS + 1, 8) for _ in range(3)]
        )
        self.entity = nn.Sequential(
            nn.Linear(52 if object_history else 48, 64), nn.LayerNorm(64), nn.SiLU()
        )
        self.hero = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.SiLU())
        self.interaction = nn.MultiheadAttention(64, 4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.initial = nn.Linear(64, 96)
        self.cell = nn.GRUCell(BINS * CONTROL + 1, 96)
        self.player_readout = nn.Linear(96, len(PLAYER_TARGET))
        self.events = nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 8))
        self.occupancy = nn.Linear(96, 60)
        self.object_readout = nn.Sequential(nn.Linear(160, 64), nn.SiLU())
        self.geometry = nn.Linear(64, len(TARGET) + 1)
        self.fsm = nn.Linear(64, STATE_BUCKETS + 1)
        self.register_buffer("pos_weight", torch.tensor(POS_WEIGHT))
        self.register_buffer("calibration", torch.tensor(POS_WEIGHT).log())

    def encode(self, b: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        tags = [layer(b["cat"][:, :, i]) for i, layer in enumerate(self.tags)]
        continuous = [b["obj"], b["obj_history"]] if self.object_history else [b["obj"]]
        objects = self.entity(torch.cat([*continuous, *tags], dim=-1))
        tokens = torch.cat([self.hero(b["player"])[:, None], objects], dim=1)
        mask = torch.cat([torch.ones_like(b["mask"][:, :1]), b["mask"]], dim=1)
        mixed = self.interaction(
            tokens, tokens, tokens, key_padding_mask=~mask, need_weights=False
        )[0]
        encoded = self.norm(tokens + mixed)
        return self.initial(encoded[:, 0]).tanh(), encoded[:, 1:]

    def imagine(self, initial: Tensor, actions: Tensor) -> Tensor:
        z = initial
        trajectory = []
        for step in actions.unbind(1):
            z = self.cell(step, z)
            trajectory.append(z)
        return torch.stack(trajectory, dim=1)

    def forward(self, b: dict[str, Tensor]) -> dict[str, Tensor]:
        z, tokens = self.encode(b)
        states = self.imagine(z, b["action"])
        time = states.size(1)
        paired = torch.cat(
            [
                states[:, :, None].expand(-1, -1, tokens.shape[1], -1),
                tokens[:, None].expand(-1, time, -1, -1),
            ],
            -1,
        )
        objects = self.object_readout(paired)
        shape = self.geometry(objects)
        return {
            "hero": b["player"][:, None, PLAYER_TARGET] + self.player_readout(states),
            "entity": b["obj"][:, None, :, TARGET] + shape[..., :-1],
            "present": shape[..., -1],
            "fsm": self.fsm(objects),
            "event": self.events(states),
            "occupancy": self.occupancy(states),
        }

    def loss(
        self, b: dict[str, Tensor], out: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        observed = b["observed"]
        hero_err = F.smooth_l1_loss(
            out["hero"], b["player_target"], beta=0.06, reduction="none"
        ).mean(-1)
        hero = (hero_err * observed).sum() / observed.sum().clamp_min(1)
        valid = b["obj_present"] & b["mask"][:, None] & observed[:, :, None]
        mobile = ((b["obj"][:, :, 1] - 0.2).abs() < 0.01) | (
            (b["obj"][:, :, 1] - 0.6).abs() < 0.01
        )
        weights = valid * (1 + 3 * mobile[:, None])
        err = F.smooth_l1_loss(
            out["entity"], b["obj_target"], beta=0.08, reduction="none"
        ).mean(-1)
        geom = (weights * err).sum() / weights.sum().clamp_min(1)
        alive = F.binary_cross_entropy_with_logits(
            out["present"], b["obj_present"].float(), reduction="none"
        )
        tracked = b["mask"][:, None] & observed[:, :, None]
        alive = (alive * tracked).sum() / tracked.sum().clamp_min(1)
        classes = F.cross_entropy(
            out["fsm"].flatten(0, 2), b["fsm"].reshape(-1), reduction="none"
        ).view_as(valid)
        fsm = (classes * weights).sum() / weights.sum().clamp_min(1)
        occ = F.binary_cross_entropy_with_logits(
            out["occupancy"], b["occupancy"], reduction="none"
        ).mean(-1)
        occupancy = (occ * observed).sum() / observed.sum().clamp_min(1)
        event = F.binary_cross_entropy_with_logits(
            out["event"], (b["event"] > 0).float(), pos_weight=self.pos_weight
        )
        loss = (
            3 * hero
            + 2 * geom
            + 0.16 * alive
            + 0.13 * fsm
            + 0.18 * occupancy
            + 1.6 * event
        )
        return loss, {
            "hero": hero,
            "geometry": geom,
            "presence": alive,
            "event": event,
            "fsm": fsm,
        }


@torch.inference_mode()
def validate(model: ObjectWorld, data: WorldReplay) -> dict[str, object]:
    model.eval()
    metrics: dict[str, list[float]] = {}
    probs, truth = [], []
    ids = data.indices["holdout"]
    for start in range(0, len(ids), 256):
        b = data.batch(ids[start : start + 256])
        out = model(b)
        _, parts = model.loss(b, out)
        for k, v in parts.items():
            metrics.setdefault(k, []).append(float(v))
        probs.append(out["event"].flatten(0, 1).cpu())
        truth.append((b["event"] > 0).flatten(0, 1).cpu())
    logits = torch.cat(probs)
    target = torch.cat(truth)
    rates = (target.sum(0) + 0.5) / (len(target) + 1)
    lo = torch.full((8,), -18.0)
    hi = torch.full((8,), 18.0)
    for _ in range(24):
        middle = (lo + hi) * 0.5
        high = (logits - middle).sigmoid().mean(0) > rates
        lo = torch.where(high, middle, lo)
        hi = torch.where(high, hi, middle)
    model.calibration.copy_(((lo + hi) * 0.5).to(model.calibration.device))
    pred = (logits - model.calibration.cpu()).sigmoid()
    ap = []
    for j in range(8):
        order = pred[:, j].argsort(descending=True)
        y = target[order, j].float()
        precision = y.cumsum(0) / torch.arange(1, len(y) + 1)
        ap.append(float((precision * y).sum() / y.sum().clamp_min(1)))
    return {
        "losses": {k: sum(v) / len(v) for k, v in metrics.items()},
        "events_ap": ap,
        "event_frequency": target.float().mean(0).tolist(),
        "holdout_windows": len(ids),
        "calibration_bias": model.calibration.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=1600)
    p.add_argument("--batch", type=int, default=160)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 41, 73])
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--object-history",
        action="store_true",
        help="Include causal FSM ages and acceleration.",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Colab GPU unavailable")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    data = WorldReplay(args.data, args.device)
    hardware = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
    print(
        json.dumps(
            {
                "hardware": hardware,
                "episodes": data.episodes,
                "windows": {k: len(v) for k, v in data.indices.items()},
            }
        ),
        flush=True,
    )
    for seed in args.seeds:
        path = args.output / f"world-{seed}.pt"
        if path.exists():
            continue
        torch.manual_seed(seed)
        model = ObjectWorld(object_history=args.object_history).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=7e-4)
        start = time.monotonic()
        for i in range(args.updates):
            batch = data.batch(data.sample("train", args.batch))
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"
            ):
                out = model(batch)
                loss, _ = model.loss(batch, out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            if i % 200 == 0 or i + 1 == args.updates:
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "step": i + 1,
                            "loss": float(loss.detach()),
                            "seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
        report = validate(model, data)
        partial = path.with_suffix(".tmp")
        torch.save(model.state_dict(), partial)
        partial.replace(path)
        (args.output / f"metrics-{seed}.json").write_text(
            json.dumps(
                {"hardware": hardware, "updates": args.updates, **report}, indent=2
            )
        )
        print(json.dumps({"seed": seed, "holdout_ap": report["events_ap"]}), flush=True)


if __name__ == "__main__":
    main()

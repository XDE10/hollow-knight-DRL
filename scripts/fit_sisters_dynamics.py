"""Train and test an action-conditioned open-loop model on packed single-sister sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

HISTORY = 4
HORIZONS = (1, 5, 15)
ACTION_DIM = 3 + 3 + 9 + 4 + 12
ARENA_SCALE = 30.0  # hkrl.spaces.ARENA_SCALE in the two archived runs


def starts(valid: np.ndarray, horizon: int) -> np.ndarray:
    """Indices whose entire history and requested future remain in the same batch/episode."""
    edges = HISTORY - 1 + horizon
    if horizon < 0:
        raise ValueError("negative rollout horizon")
    if len(valid) <= edges:
        return np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(valid, edges)
    result = np.flatnonzero(windows.all(axis=-1)) + HISTORY - 1
    return result[result + horizon < len(valid)]


def encode_actions(raw: Tensor) -> Tensor:
    raw = raw.long()
    return torch.cat(
        (
            F.one_hot(raw[..., 0], 3),
            F.one_hot(raw[..., 1], 3),
            raw[..., 2:11],
            F.one_hot(raw[..., 11], 4),
            F.one_hot(raw[..., 12], 12),
        ),
        dim=-1,
    ).float()


class Replay:
    def __init__(self, path: Path, device: str, train: Replay | None = None) -> None:
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        self.valid = data["valid"]
        self.raw = data["actions"]
        self.mask = data["mask"].astype(bool)
        self.hp = data["hp"]
        self.vocab = np.unique(data["fsm"]) if train is None else train.vocab
        indices = np.searchsorted(self.vocab, data["fsm"])
        unknown = len(self.vocab)
        indices[self.vocab[indices.clip(max=unknown - 1)] != data["fsm"]] = unknown
        self.fsm = torch.as_tensor(indices, device=device).long()
        self.xy = torch.as_tensor(data["state"][:, :4] * ARENA_SCALE, device=device)
        self.mean = data["state"].mean(0) if train is None else train.mean
        self.scale = data["state"].std(0).clip(0.025) if train is None else train.scale
        normalized = np.clip((data["state"] - self.mean) / self.scale, -12, 12)
        self.obs = torch.cat(
            (torch.as_tensor(normalized, device=device), F.one_hot(self.fsm, unknown + 1).float()),
            dim=-1,
        )
        self.actions = encode_actions(torch.as_tensor(self.raw, device=device))
        # An event label requires both HP observations. Terminal-only labels use the
        # transition flags even when the end-of-fight observation was not archived.
        drops = np.zeros_like(self.hp)
        drops[:-1] = self.hp[1:] < self.hp[:-1]
        self.drops = torch.as_tensor(drops, device=device)
        self.links = torch.as_tensor(self.valid, device=device)
        self.terminal = torch.as_tensor(data["terminated"], device=device)

    def history(self, ix: Tensor) -> Tensor:
        rows = ix[:, None] + torch.arange(1 - HISTORY, 1, device=ix.device)
        return torch.cat((self.obs[rows].flatten(1), self.actions[rows[:, :-1]].flatten(1)), dim=-1)


class Dynamics(nn.Module):
    def __init__(self, obs_size: int, fsm_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size * HISTORY + ACTION_DIM * (HISTORY - 1), 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Linear(160, 128),
            nn.Tanh(),
        )
        self.transition = nn.GRU(ACTION_DIM, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 7 + fsm_count))
        self.register_buffer("step_scale", torch.tensor([1.0, 1.0, 4.0, 4.0]))

    def forward(
        self, history: Tensor, actions: Tensor, origin: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent, _ = self.transition(actions, self.encoder(history)[None])
        output = self.head(latent)
        xy = origin[:, None] + (output[..., :4] * self.step_scale).cumsum(dim=1)
        return xy, output[..., 4:7], output[..., 7:]


def fit(data: Replay, seed: int, updates: int, batch: int) -> Dynamics:
    device = data.xy.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Dynamics(data.obs.shape[-1], len(data.vocab) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    candidates = starts(data.valid, 0)
    for step in range(updates):
        ix = torch.as_tensor(rng.choice(candidates, batch), device=device)
        horizon = (1, 5, 15)[step % 3]
        offset = torch.arange(horizon, device=device)
        rows = (ix[:, None] + offset).clamp(max=len(data.valid) - 1)
        following = (rows + 1).clamp(max=len(data.valid) - 1)
        link = data.links[rows]
        valid = link.long().cumprod(dim=1).bool()
        reachable = F.pad(valid[:, :-1], (1, 0), value=True)
        known_terminal = reachable & (link | data.terminal[rows])
        xy, logits, fsm = model(data.history(ix), data.actions[rows], data.xy[ix])
        error = (xy - data.xy[following]) / model.step_scale
        state_loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(-1)
        drop_loss = F.binary_cross_entropy_with_logits(
            logits[..., :2], data.drops[rows], reduction="none"
        )
        fsm_loss = F.cross_entropy(fsm.transpose(1, 2), data.fsm[following], reduction="none")
        terminal_loss = F.binary_cross_entropy_with_logits(
            logits[..., 2], data.terminal[rows].float(), reduction="none"
        )
        loss = (
            (state_loss + 3 * drop_loss[..., 0] + 6 * drop_loss[..., 1] + 0.25 * fsm_loss) * valid
        ).sum() / valid.sum().clamp(min=1)
        loss = loss + (terminal_loss * known_terminal).sum() / known_terminal.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def forecast(
    models: list[Dynamics],
    data: Replay,
    anchors: np.ndarray,
    horizon: int,
    alternatives: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    samples: dict[str, list[np.ndarray]] = {"xy": [], "event": [], "spread": [], "fsm": []}
    for start in range(0, len(anchors), 512):
        ix = torch.as_tensor(anchors[start : start + 512], device=data.xy.device)
        rows = ix[:, None] + torch.arange(horizon, device=ix.device)
        actions = data.actions[rows]
        if alternatives is not None:
            actions[:, 0] = encode_actions(
                torch.as_tensor(alternatives[start : start + 512], device=ix.device)
            )
        history = data.history(ix)
        pred = [model(history, actions, data.xy[ix]) for model in models]
        mean_xy = torch.stack([p[0][:, -1] for p in pred]).mean(0)
        # Probability of at least one HP drop over the window, conditional-hazard approximation.
        risk = torch.stack([-torch.expm1(F.logsigmoid(-p[1][..., :2]).sum(1)) for p in pred])
        fsm = torch.stack([p[2][:, -1].softmax(-1) for p in pred]).mean(0).argmax(-1)
        for name, value in (
            ("xy", mean_xy),
            ("event", risk.mean(0)),
            ("spread", risk.std(0, unbiased=False)),
            ("fsm", fsm),
        ):
            samples[name].append(value.cpu().numpy())
    return {k: np.concatenate(v) for k, v in samples.items()}


def score(models: list[Dynamics], data: Replay) -> list[dict[str, Any]]:
    from sklearn.metrics import average_precision_score

    results = []
    xy = data.xy.cpu().numpy()
    fsm = data.fsm.cpu().numpy()
    for horizon in HORIZONS:
        ix = starts(data.valid, horizon)
        result = forecast(models, data, ix, horizon)
        hp = data.hp[ix[:, None] + np.arange(horizon + 1)]
        truth = (np.diff(hp, axis=1) < 0).any(axis=1)
        end = xy[ix + horizon]
        error = np.linalg.norm((result["xy"] - end).reshape(-1, 2, 2), axis=-1)
        persistence = np.linalg.norm((xy[ix] - end).reshape(-1, 2, 2), axis=-1)
        results.append(
            dict(
                horizon=horizon,
                windows=len(ix),
                mean_xy_error=error.mean(0).tolist(),
                persistence=persistence.mean(0).tolist(),
                event_ap=[
                    float(average_precision_score(truth[:, i], result["event"][:, i]))
                    for i in range(2)
                ],
                prevalence=truth.mean(0).tolist(),
                fsm_accuracy=float(np.mean(result["fsm"] == fsm[ix + horizon])),
                fsm_persistence=float(np.mean(fsm[ix] == fsm[ix + horizon])),
            )
        )
    return results


def alternatives(
    data: Replay, ix: np.ndarray, kind: str, supported: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Change one primitive choice. Preserve duration/aim and require recorded support."""
    raw = data.raw[ix].copy()
    keep = raw[:, 12] == 0
    if kind == "reverse":
        keep &= raw[:, 0] != 1
        raw[:, 0] = 2 - raw[:, 0]
        keep &= data.mask[ix, raw[:, 0]]
    else:
        bit = {"jump": 0, "dash": 2, "attack": 3}[kind]
        old = raw[:, 2 + bit].copy()
        if kind != "attack":
            keep &= old == 0
            if kind == "jump":
                keep &= raw[:, 3] == 0  # a held jump is already a jump command
        raw[:, 2 + bit] = 1 - old
        keep &= (old == 1) | data.mask[ix, 6 + bit]
    key = np.dtype((np.void, raw.shape[1]))
    keep &= np.isin(raw.view(key).ravel(), supported.view(key).ravel())
    return ix[keep], raw[keep]


def action_checks(models: list[Dynamics], train: Replay, data: Replay) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    rows, counts = np.unique(train.raw, axis=0, return_counts=True)
    supported = np.ascontiguousarray(rows[counts >= 20])
    ix = starts(data.valid, 1)
    rng = np.random.default_rng(20261119)
    hurt = ix[data.hp[ix + 1, 0] < data.hp[ix, 0]]
    draws = rng.choice(hurt, min(500, len(hurt)), replace=False)
    output: dict[str, Any] = {"damage_frames_drawn": len(draws), "hurt_frames": {}}
    for kind in ("jump", "dash", "reverse"):
        subset, alt = alternatives(data, draws, kind, supported)
        if not len(subset):
            output["hurt_frames"][kind] = {"eligible": 0}
            continue
        actual = forecast(models, data, subset, 1)
        changed = forecast(models, data, subset, 1, alt)
        delta = changed["xy"][:, :2] - actual["xy"][:, :2]
        output["hurt_frames"][kind] = dict(
            eligible=len(subset),
            mean_displacement=float(np.linalg.norm(delta, axis=-1).mean()),
            mean_risk_change=float((changed["event"][:, 0] - actual["event"][:, 0]).mean()),
            mean_alternative_risk_spread=float(changed["spread"][:, 0].mean()),
        )
    for kind in ("reverse", "attack"):
        subset, alt = alternatives(data, ix, kind, supported)
        if not len(subset):
            output[kind] = {"eligible": 0}
            continue
        actual = forecast(models, data, subset, 1)
        changed = forecast(models, data, subset, 1, alt)
        if kind == "reverse":
            truth = data.xy.cpu().numpy()[subset + 1, 0]
            shift = changed["xy"][:, 0] - actual["xy"][:, 0]
            sign = alt[:, 0].astype(float) - 1.0
            output[kind] = dict(
                eligible=len(subset),
                factual_x_mae=float(np.abs(actual["xy"][:, 0] - truth).mean()),
                reversed_x_mae=float(np.abs(changed["xy"][:, 0] - truth).mean()),
                correct_direction=float((shift * sign > 0).mean()),
            )
        else:
            truth = data.hp[subset + 1, 1] < data.hp[subset, 1]
            output[kind] = dict(
                eligible=len(subset),
                positives=int(truth.sum()),
                factual_ap=float(average_precision_score(truth, actual["event"][:, 1])),
                toggled_ap=float(average_precision_score(truth, changed["event"][:, 1])),
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "holdout", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 41, 73])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.train.resolve() == args.holdout.resolve():
        parser.error("use separate training and evaluation runs")
    if len(set(args.seeds)) != len(args.seeds) or min(args.batch, args.updates) < 1:
        parser.error("supply distinct seeds and a positive batch size and update count")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    train = Replay(args.train, args.device)
    holdout = Replay(args.holdout, args.device, train)
    models = []
    for seed in args.seeds:
        model = fit(train, seed, args.updates, args.batch)
        torch.save(
            dict(
                weights=model.state_dict(),
                seed=seed,
                updates=args.updates,
                mean=torch.tensor(train.mean),
                scale=torch.tensor(train.scale),
                vocab=torch.tensor(train.vocab),
            ),
            args.output / f"model-{seed}.pt",
        )
        models.append(model)
        print(f"Finished seed {seed}", flush=True)
    rows = score(models, holdout)
    actions = action_checks(models, train, holdout)
    bounds = {
        "player_and_boss_beat_position_persistence": all(
            np.all(np.array(row["mean_xy_error"]) < row["persistence"]) for row in rows
        ),
        "both_events_beat_prevalence": all(
            np.all(np.array(row["event_ap"]) > row["prevalence"]) for row in rows
        ),
        "factual_movement_beats_reversal": bool(
            actions["reverse"]["eligible"]
            and actions["reverse"]["factual_x_mae"] < actions["reverse"]["reversed_x_mae"]
            and actions["reverse"]["correct_direction"] > 0.5
        ),
        "factual_attack_beats_toggle": bool(
            actions["attack"]["eligible"]
            and actions["attack"]["positives"]
            and actions["attack"]["factual_ap"] > actions["attack"]["toggled_ap"]
        ),
    }
    report = dict(
        device=args.device,
        hardware=(torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU"),
        updates=args.updates,
        batch=args.batch,
        seeds=args.seeds,
        training_rows=len(train.raw),
        holdout_rows=len(holdout.raw),
        results=rows,
        action_checks=actions,
        checks=bounds,
        passed=all(bounds.values()),
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

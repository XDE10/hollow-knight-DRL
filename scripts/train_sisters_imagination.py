"""Distill two primitive policies from the same frozen short-horizon ensemble forecasts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from fit_sisters_dynamics import ACTION_DIM, HISTORY, Dynamics, Replay, encode_actions, starts
from torch import Tensor, nn
from torch.nn import functional as F

HORIZON = 12


def student(width: int, choices: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, 128),
        nn.LayerNorm(128),
        nn.SiLU(),
        nn.Linear(128, 96),
        nn.SiLU(),
        nn.Linear(96, choices),
    )


def imagined_returns(prob: Tensor) -> Tensor:
    """Two returns per sequence, with physical terminal and optional HP-loss discount."""
    hurt, hit, terminal = prob.unbind(-1)
    reward = 6.3 * hit - 12 * hurt - 100 * terminal
    live = 0.98 * (1 - terminal)
    discount = torch.stack((live, live * (1 - hurt)), dim=-1)
    prefix = torch.cat((torch.ones_like(discount[..., :1, :]), discount[..., :-1, :]), dim=-2)
    return (prefix.cumprod(dim=-2) * reward[..., None]).sum(-2)


def donors_for(data: Replay, anchors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample common follow-up controls from a separate window in the same boss FSM."""
    future = starts(data.valid, HORIZON)
    fsm = data.fsm.cpu().numpy()
    chosen = np.empty_like(anchors)
    for state in np.unique(fsm[anchors]):
        where = np.flatnonzero(fsm[anchors] == state)
        pool = future[fsm[future] == state]
        if len(pool) < 2 * (HORIZON + HISTORY) + 2:
            pool = future
        # Uniform draw excluding indices within either window's history/future.
        lo = np.searchsorted(pool, anchors[where] - HORIZON - HISTORY)
        hi = np.searchsorted(pool, anchors[where] + HORIZON + HISTORY, side="right")
        index = rng.integers(len(pool) - (hi - lo))
        chosen[where] = pool[index + np.where(index >= lo, hi - lo, 0)]
    return chosen


def train_policies(
    train: Replay, models: list[Dynamics], output: Path, sample_count: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = train.xy.device
    eligible = starts(train.valid, 0)
    primitive = eligible[train.raw[eligible, 12] == 0]
    choices, inverse, counts = np.unique(
        train.raw[primitive], axis=0, return_counts=True, return_inverse=True
    )
    selected = np.argsort(-counts, kind="stable")[:64]
    catalogue = choices[selected]
    lookup = np.full(len(choices), -1)
    lookup[selected] = np.arange(len(selected))
    labels = lookup[inverse]
    in_catalogue = labels >= 0
    examples = primitive[in_catalogue]
    labels = torch.as_tensor(labels[in_catalogue], device=device)
    codes = encode_actions(torch.as_tensor(catalogue, device=device))
    masks = torch.as_tensor(train.mask, device=device)
    width = train.obs.shape[-1] * HISTORY + ACTION_DIM * (HISTORY - 1)
    prior = student(width, len(selected)).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=0.001)
    for _ in range(650):
        sample = rng.integers(len(examples), size=384)
        ix = torch.as_tensor(examples[sample], device=device)
        loss = F.cross_entropy(prior(train.history(ix)), labels[sample])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    anchors = rng.choice(eligible, size=min(sample_count, len(eligible)), replace=False)
    donor = donors_for(train, anchors, rng)
    targets, all_hist = [], []
    with torch.inference_mode():
        for start in range(0, len(anchors), 48):
            ix = torch.as_tensor(anchors[start : start + 48], device=device)
            history = train.history(ix)
            donor_ix = torch.as_tensor(donor[start : start + 48], device=device)
            suffix = train.actions[donor_ix[:, None] + torch.arange(1, HORIZON, device=device)]
            batch, count = len(ix), len(codes)
            actions = torch.cat(
                (
                    codes[None, :, None].expand(batch, -1, -1, -1),
                    suffix[:, None].expand(-1, count, -1, -1),
                ),
                dim=2,
            ).flatten(0, 1)
            returns = []
            for model in models:
                initial = model.encoder(history).repeat_interleave(count, dim=0)[None]
                latent, _ = model.transition(actions, initial)
                logit = model.head(latent)[..., 4:7]
                returns.append(imagined_returns(logit.sigmoid()).reshape(batch, count, 2))
            values = torch.stack(returns)
            robust = values.mean(0) - values.std(0, unbiased=False)
            legal = (~masks[ix]).float() @ codes.T == 0
            keep = legal.any(1)
            preferences = robust / 0.8 + 0.6 * prior(history).log_softmax(-1)[..., None]
            weights = preferences[keep].masked_fill(~legal[keep, :, None], -torch.inf).softmax(1)
            all_hist.append(history[keep])
            targets.append(weights)
    x = torch.cat(all_hist)
    y = torch.cat(targets)
    # Clone the common initialization and use the identical minibatch schedule.
    schedule = rng.integers(len(x), size=(900, 384))
    for i, name in enumerate(("full_fight", "life_loss")):
        actor = copy.deepcopy(prior)
        opt = torch.optim.AdamW(actor.parameters(), lr=0.0006)
        for batch_idx in schedule:
            prediction = actor(x[batch_idx])
            loss = -(y[batch_idx, :, i] * prediction.log_softmax(-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.save(
            dict(
                weights=actor.cpu().state_dict(),
                catalogue=torch.tensor(catalogue),
                mean=torch.tensor(train.mean),
                scale=torch.tensor(train.scale),
                vocab=torch.tensor(train.vocab),
                life_loss=bool(i),
                horizon=HORIZON,
            ),
            output / f"{name}.pt",
        )
    info = dict(
        training_anchors=len(x),
        horizon=HORIZON,
        choices=len(catalogue),
        catalogue_primitive_coverage=float(in_catalogue.mean()),
        attack_choices=int(catalogue[:, 5].sum()),
        seed=seed,
        ensemble=len(models),
    )
    (output / "policy_training.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchors", type=int, default=24000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = Replay(args.train, args.device)
    ensemble = []
    for path in sorted(args.models.glob("model-*.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("vocab", "mean", "scale"):
            if not np.array_equal(ckpt[key].numpy(), getattr(dataset, key)):
                raise ValueError(f"{path} uses a different replay normalization or FSM vocabulary")
        model = Dynamics(dataset.obs.shape[-1], len(dataset.vocab) + 1).to(args.device)
        model.load_state_dict(ckpt["weights"])
        ensemble.append(model.eval())
    if not ensemble:
        parser.error("no trained model files")
    train_policies(dataset, ensemble, args.output, args.anchors, seed=83)

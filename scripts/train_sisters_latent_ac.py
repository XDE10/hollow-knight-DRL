"""Fit real transitions, then train a discrete actor and critic inside the learned model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sisters_latent_ac import (
    CATALOG,
    GAMMA,
    SCHEMA,
    WEIGHTS,
    ActorCritic,
    Dynamics,
    action_vector,
    applied_controls,
    encode,
    encoder_from,
    keys,
    lambda_returns,
    pulse_controls,
)
from torch import Tensor
from torch.nn import functional as F


def read_replay(
    folder: Path, encoder, device: str, combat_inputs: bool = False
) -> tuple[dict, dict]:
    games = json.loads((folder / "manifest.json").read_text())
    table = keys(np.asarray(CATALOG))
    buffers = {
        k: {
            n: []
            for n in ("s", "next", "a", "event", "label", "mc", "observed", "successor")
        }
        for k in ("train", "holdout")
    }
    used = set()
    counts = {k: 0 for k in buffers}
    for game in games:
        split = game["split"]
        if (
            game["source"] in used
            or "official" in game["source"]
            or "exam-" in game["source"]
        ):
            raise ValueError("duplicate or formal evaluation fight in replay")
        used.add(game["source"])
        with np.load(folder / game["file"], allow_pickle=False) as arc:
            raw, ptr, player = arc["entities"], arc["entity_ptr"], arc["player"]
            vectors = []
            for start in range(0, len(player), 384):
                stop = min(len(player), start + 384)
                chunk = ptr[start : stop + 1]
                vectors.append(
                    encode(
                        encoder,
                        player[start:stop],
                        raw[chunk[0] : chunk[-1]],
                        chunk - chunk[0],
                        combat_inputs,
                    ).cpu()
                )
            state = torch.cat(vectors).numpy()
            times = arc["time"]
            end = np.r_[times[1:], times[-1] + 0.02 * arc["elapsed_ticks"][-1]]
            dt = end - times
            controls = applied_controls(
                times, end, arc["control_time"], arc["controls"]
            )
            if combat_inputs:
                controls = pulse_controls(controls)
            ev = (arc["events"] > 0).astype("f4")
            reward = ev @ np.array(WEIGHTS, "f4")
            mc = np.zeros(len(ev), "f4")
            bootstrap = 0.0
            for j in reversed(range(len(ev))):
                bootstrap = (
                    reward[j]
                    + GAMMA ** (dt[j] / 0.12) * (1 - max(ev[j, 6:])) * bootstrap
                )
                mc[j] = bootstrap
            # Mask bootstrapping across missing controls, host stalls and fight boundaries.
            covered = (times >= arc["control_time"][0]) & (
                end <= arc["control_time"][-1] + 0.065
            )
            good = covered & (dt > 0.025) & (dt <= 0.36)
            terminal = ev[:, 6:].any(1)
            good &= np.r_[np.ones(len(ev) - 1, bool), terminal[-1]]
            successor = good & np.r_[good[1:], False] & ~terminal
            observed = np.arange(len(ev)) < len(ev) - 1
            error = np.square(controls[:, None, :-1] - table[None])
            if combat_inputs:
                # A brief offensive edge is meaningful even when aim changes mid-interval.
                error[:, :, [4, 6, 7, 8]] *= 3
            labels = error.sum(-1).argmin(-1)
            values = {
                "s": state,
                "next": np.r_[state[1:], state[-1:]],
                "a": controls,
                "event": ev,
                "label": labels,
                "mc": mc,
                "observed": observed,
                "successor": successor,
            }
            if not good.any():
                continue
            for k, v in values.items():
                buffers[split][k].append(v[good])
            counts[split] += 1
    datasets = {
        s: {k: torch.as_tensor(np.concatenate(v), device=device) for k, v in b.items()}
        for s, b in buffers.items()
    }
    train = datasets["train"]
    mean = train["s"].mean(0)
    std = train["s"].std(0).clamp_min(0.04)
    for d in datasets.values():
        for k in ("s", "next"):
            d[k] = ((d[k] - mean) / std).clamp(-8, 8)
    metadata = {
        "mean": mean,
        "std": std,
        "episode_counts": counts,
        "replay_manifest_sha256": hashlib.sha256(
            (folder / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    return datasets, metadata


def sample(data: dict, n: int) -> Tensor:
    return torch.randint(len(data["s"]), (n,), device=data["s"].device)


def log(**row) -> None:
    print(json.dumps(row), flush=True)


def fit_world(
    data: dict, holdout: dict, steps: int, batch: int
) -> tuple[list, list, list]:
    models, biases, reports = [], [], []
    rate = data["event"].mean(0)
    pos = ((1 - rate) / rate.clamp_min(1e-5)).sqrt().clamp(1, 24)
    for seed in (71, 127, 257):
        torch.manual_seed(seed)
        model = Dynamics(data["s"].shape[-1]).to(pos.device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        for step in range(steps):
            i = sample(data, batch)
            predicted, evt = model(data["s"][i], data["a"][i])
            keep = data["observed"][i]
            one = F.smooth_l1_loss(predicted[keep], data["next"][i][keep])
            loss = one + F.binary_cross_entropy_with_logits(
                evt, data["event"][i], pos_weight=pos
            )
            # Roll a second transition using the recorded next action, within the same fight.
            paired = data["successor"][i]
            ii = i[paired] + 1
            if len(ii):
                nxt, ev2 = model(predicted[paired], data["a"][ii])
                err = F.smooth_l1_loss(nxt, data["next"][ii], reduction="none").mean(1)
                loss += 0.45 * (
                    (err * data["observed"][ii]).mean()
                    + F.binary_cross_entropy_with_logits(
                        ev2, data["event"][ii], pos_weight=pos
                    )
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        model.eval().requires_grad_(False)
        with torch.no_grad():
            calibration = sample(data, 10000)
            _, logits = model(data["s"][calibration], data["a"][calibration])
            lo, hi = torch.full_like(rate, -20), torch.full_like(rate, 20)
            for _ in range(22):
                mid = (lo + hi) / 2
                high = (logits - mid).sigmoid().mean(0) > rate
                lo, hi = torch.where(high, mid, lo), torch.where(high, hi, mid)
            bias = (lo + hi) / 2
            # Genuine multi-step open-loop errors, using held-out actions only.
            eligible = torch.arange(len(holdout["s"]) - 5, device=pos.device)
            for j in range(4):
                eligible = eligible[holdout["successor"][eligible + j]]
            eligible = eligible[holdout["observed"][eligible + 4]]
            i = eligible[:: max(1, len(eligible) // 4096)][:4096]
            initial = holdout["s"][i]
            state = initial
            metrics = {}
            for j in range(5):
                state, _ = model(state, holdout["a"][i + j])
                if j in (0, 4):
                    target = holdout["next"][i + j]
                    metrics[str(j + 1)] = {
                        "mae": (state - target).abs().mean().item(),
                        "unchanged_mae": (initial - target).abs().mean().item(),
                    }
            log(stage="world", seed=seed, holdout=metrics)
        models.append(model)
        biases.append(bias)
        reports.append(metrics)
    return models, biases, reports


def warm_start(
    policy: ActorCritic, data: dict, steps: int, batch: int, balance: bool = False
) -> None:
    opt = torch.optim.AdamW(policy.parameters(), lr=0.001)
    counts = data["label"].bincount(minlength=len(CATALOG)).float().clamp_min(1)
    weights = (
        (counts.max() / counts).pow(0.45).clamp_max(6)
        if balance
        else torch.ones_like(counts)
    )
    for _ in range(steps):
        i = sample(data, batch)
        dist, value = policy(data["s"][i])
        importance = weights[data["label"][i]]
        loss = -(
            importance * dist.log_prob(data["label"][i])
        ).sum() / importance.sum() + 0.18 * F.smooth_l1_loss(value, data["mc"][i])
        opt.zero_grad()
        loss.backward()
        opt.step()


def optimize_ppo(
    policy,
    optimizer,
    state,
    act,
    old_logp,
    target,
    advantage,
    mask=None,
    reference=None,
) -> dict:
    advantage = (advantage - advantage.mean()) / advantage.std(
        unbiased=False
    ).clamp_min(1e-4)
    before = None
    for _ in range(3):
        for idx in torch.randperm(len(state), device=state.device).split(256):
            dist, value = policy(state[idx], mask[idx] if mask is not None else None)
            logp = dist.log_prob(act[idx])
            ratio = (logp - old_logp[idx]).exp()
            surrogate = torch.minimum(
                ratio * advantage[idx], ratio.clamp(0.85, 1.15) * advantage[idx]
            )
            actor_loss = -surrogate.mean()
            if before is None:
                before = float(actor_loss.detach())
            loss = (
                actor_loss
                + 0.45 * F.smooth_l1_loss(value, target[idx])
                - 0.008 * dist.entropy().mean()
            )
            if reference is not None:
                with torch.no_grad():
                    prior, _ = reference(state[idx])
                loss += 0.14 * torch.distributions.kl_divergence(dist, prior).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
    return {
        "actor_loss_first": before,
        "value_loss": float(F.smooth_l1_loss(value.detach(), target[idx])),
        "entropy": float(dist.entropy().mean().detach()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("data", "encoder", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--world-steps", type=int, default=1500)
    p.add_argument("--warm-steps", type=int, default=1000)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument(
        "--combat-inputs",
        action="store_true",
        help="Retain button edges and explicit masked body/disc geometry.",
    )
    p.add_argument("--batch", type=int, default=192)
    args = p.parse_args()
    if any(
        v <= 0
        for v in (
            args.world_steps,
            args.warm_steps,
            args.updates,
            args.horizon,
            args.batch,
        )
    ):
        p.error("positive training budgets required")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7121)
    torch.set_num_threads(1)
    encoder = encoder_from(args.encoder).to(args.device)
    data, meta = read_replay(args.data, encoder, args.device, args.combat_inputs)
    d = data["train"]
    hardware = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
    log(
        stage="replay",
        hardware=hardware,
        episodes=meta["episode_counts"],
        transitions={k: len(v["s"]) for k, v in data.items()},
    )
    models, biases, validation = fit_world(
        d, data["holdout"], args.world_steps, args.batch
    )
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    common = {
        "schema": SCHEMA,
        "catalog": CATALOG,
        "encoder": encoder.cpu().state_dict(),
        "normalization": (meta["mean"].cpu(), meta["std"].cpu()),
        "reward": WEIGHTS,
        "combat_inputs": args.combat_inputs,
        "replay_manifest_sha256": meta["replay_manifest_sha256"],
    }
    policy = ActorCritic(d["s"].shape[-1]).to(args.device)
    warm_start(policy, d, args.warm_steps, args.batch, balance=args.combat_inputs)
    prior = copy.deepcopy(policy).eval().requires_grad_(False)

    def save_policy(label):
        torch.save(
            dict(
                **common,
                actor_critic={
                    k: v.detach().cpu() for k, v in policy.state_dict().items()
                },
                stage=label,
            ),
            out / (label + ".pt"),
        )

    save_policy("before_rl")
    torch.save(
        {"models": [m.state_dict() for m in models], "biases": biases},
        out / "dynamics.pt",
    )
    opt = torch.optim.Adam(policy.parameters(), lr=0.00022)
    table = torch.as_tensor(keys(np.asarray(CATALOG)), device=args.device)
    weights = table.new_tensor(WEIGHTS)
    for update in range(args.updates):
        with torch.no_grad():
            state = d["s"][sample(d, args.batch)]
            chosen = torch.randint(len(models), (len(state),), device=state.device)
            ids = torch.arange(len(state), device=state.device)
            states = []
            actions = []
            old_probs = []
            values = []
            rewards = []
            discounts = []
            next_values = []
            for _ in range(args.horizon):
                dist, value = policy(state)
                action = dist.sample()
                command = action_vector(table, action)
                predicted, events = zip(*(m(state, command) for m in models))
                z = torch.stack(predicted)
                probabilities = (
                    torch.stack(events) - torch.stack(biases)[:, None]
                ).sigmoid()
                next_state = z[chosen, ids].clamp(-8, 8)
                event = probabilities[chosen, ids]
                uncertainty = (probabilities @ weights).std(0) + 0.2 * z.std(0).mean(-1)
                states.append(state)
                actions.append(action)
                old_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(event @ weights - 0.5 * uncertainty)
                discounts.append(GAMMA * (1 - event[:, 6:].sum(-1).clamp(0, 1)))
                next_values.append(policy(next_state)[1])
                state = next_state
            target = lambda_returns(
                torch.stack(rewards), torch.stack(next_values), torch.stack(discounts)
            )
            advantages = target - torch.stack(values)
        stats = optimize_ppo(
            policy,
            opt,
            torch.cat(states),
            torch.cat(actions),
            torch.cat(old_probs),
            target.flatten(),
            advantages.flatten(),
            reference=prior,
        )
        # Real observed returns anchor the critic's scale, including terminal defeats.
        idx = sample(d, args.batch)
        _, value = policy(d["s"][idx])
        opt.zero_grad()
        (0.12 * F.smooth_l1_loss(value, d["mc"][idx])).backward()
        opt.step()
        if update % 25 == 0 or update + 1 == args.updates:
            log(stage="imagined_ppo", update=update + 1, **stats)
    save_policy("imagined_ppo")
    with torch.no_grad():
        h = data["holdout"]["s"][::3]
        before = prior(h)[0]
        after = policy(h)[0]
        delta = float(torch.distributions.kl_divergence(before, after).mean())
    report = {
        "config": params,
        "hardware": hardware,
        "episodes": meta["episode_counts"],
        "heldout_model": validation,
        "imagined_policy_steps": args.batch * args.updates * args.horizon,
        "heldout_actor_kl": delta,
        "manifest_sha256": meta["replay_manifest_sha256"],
    }
    (out / "training.json").write_text(json.dumps(report, indent=2))
    log(stage="finished", heldout_actor_kl=delta)


if __name__ == "__main__":
    main()

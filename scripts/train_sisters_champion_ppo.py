"""Continue the nine-win selector with clipped PPO on its own real game trajectories."""

from __future__ import annotations

import argparse
import copy
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from replay_sisters_champion_router import load_arms, load_router, predict_arms
from sisters_control_blocks import neural_block_starts
from torch import nn
from train_sisters_champion_router import digest, features, packet
from train_sisters_option_ppo import discounted_scan, option_rewards, verified_events


def value_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(40, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)
    )


def probabilities(margin, correction, bound, temperature):
    return 0.04 + 0.92 * ((margin + bound * correction) / temperature).sigmoid()


def advantages(rewards, discounts, traces, old_value):
    td = rewards + discounts * np.r_[old_value[1:], 0.0] - old_value
    return discounted_scan(td, discounts * traces)


def batches(
    archive, behavior_hash, nets, gating, change, bound, temperature, value, clean_cost
):
    games, outcomes = [], []
    with zipfile.ZipFile(archive) as z, torch.inference_mode():
        spec = json.loads(z.read("batch.json"))
        manifest = json.loads(z.read("manifest.json"))
        if (
            spec["actor_sha256"] != behavior_hash
            or spec["temperature"] != temperature
            or len(set(spec["seeds"])) != len(manifest)
            or [g["unity_reset_seed"] for g in manifest] != spec["seeds"]
        ):
            raise ValueError("Unexpected behavior policy or game seeds")
        for g in manifest:
            raw = z.read(g["file"])
            if (
                digest(raw) != g["sha256"]
                or g["behavior_policy_sha256"] != behavior_hash
                or g["source"] != f"native-development/{g['unity_reset_seed']}"
                or g["teacher_labels"]
                or g["teacher_fraction"] != 0
                or g["split"] != "train"
                or g["router_temperature"] != temperature
                or g["sampler_draw_seed"]
                != spec["draw_seeds"][str(g["unity_reset_seed"])]
            ):
                raise ValueError("Trajectory differs from its native receipt")
            with np.load(io.BytesIO(raw), allow_pickle=False) as tape:
                t, s = tape["control_time"], tape["timing_state"]
                rows, arm = tape["route_rows"], tape["route_draw"].astype("i8")
                ev = verified_events(
                    t, s[:, 4] * 9, tape["time"], tape["elapsed_ticks"], tape["events"]
                )
                loss = ev[:, 1:5].sum(1)
                won = bool(ev[:, 6].sum())
                if (
                    len(np.flatnonzero(ev[:, 6:].any(1))) != 1
                    or won != g["won"]
                    or loss.sum() != sum(g["damage_taken"].values())
                    or ev[:, 0].sum() != g["damage"]
                ):
                    raise ValueError("Outcomes disagree with the actual game")
                starts = neural_block_starts(t)
                active = s[starts, 23] > 0.5
                if not np.array_equal(starts[active], rows):
                    raise ValueError("Wrong control interval")
                outcome = {
                    k: g[k]
                    for k in (
                        "source",
                        "won",
                        "first_cleared",
                        "damage",
                        "damage_taken",
                        "game_speed",
                    )
                }
                outcome["decisions"] = len(rows)
                outcomes.append(outcome)
                if not len(rows):
                    continue
                logits, h = predict_arms(nets, t, s)
                x, margin = features(h[rows], s[rows], gating)
                x_t = torch.from_numpy(x)
                margin_t = torch.from_numpy(margin).float()
                predicted = margin + bound * change(x_t).numpy()
                p = tape["route_probability"]
                logged = tape["route_margin"]
                uniform = tape["route_uniform"]
                if (
                    not np.allclose(predicted, logged, rtol=0, atol=3e-5)
                    or not np.allclose(
                        probabilities(
                            margin_t, change(x_t), bound, temperature
                        ).numpy(),
                        p,
                        rtol=0,
                        atol=3e-4,
                    )
                    or not np.array_equal(arm, uniform < p)
                ):
                    raise ValueError(
                        "Recorded decisions differ from the frozen network"
                    )
                assignment = np.zeros(len(starts), dtype=bool)
                assignment[active] = arm
                choice = np.repeat(assignment, np.diff(np.r_[starts, len(t)]))
                trio = s[:, 23] > 0.5
                if not np.array_equal(
                    np.where(choice[:, None], logits[1], logits[0])[trio],
                    tape["controls"][trio],
                ):
                    raise ValueError(
                        "Neural controller disagrees with the executed primitive keys"
                    )
                r, gamma, trace = option_rewards(t, loss, ev, rows, clean_cost)
                old = value(x_t).flatten().numpy()
                adv = advantages(r, gamma, trace, old)
                games.append((x, margin, p, arm, adv, adv + old))
                outcome["verified_trio_inputs"] = int(trio.sum())
    if len(games) < 2:
        raise ValueError("Too few fights reached the trainable controller")
    return [
        torch.from_numpy(np.concatenate(a)).float() for a in zip(*games, strict=True)
    ], outcomes


def train(
    data: Path,
    parent: Path,
    behavior: Path,
    learner: Path,
    primary: Path,
    alternate: Path,
    output: Path,
    temperature: float,
    clean_cost: float = 0.0,
):
    if output.exists() or clean_cost < 0 or not 0 < temperature < 1000:
        raise ValueError("Expected a fresh update and finite rollout parameters")
    torch.set_num_threads(1)
    gating, bound, actor = load_router(parent, behavior)
    ck = torch.load(learner, weights_only=True, map_location="cpu")
    value = value_net()
    value.load_state_dict(ck["critics"]["value"])
    base = parent.read_bytes()
    if packet(actor, bound, base) != behavior.read_bytes():
        raise ValueError("Initial online actor differs from the real behavior policy")
    behavior_hash = digest(behavior.read_bytes())
    tensors, outcomes = batches(
        data,
        behavior_hash,
        load_arms(primary, alternate),
        gating,
        actor,
        bound,
        temperature,
        value,
        clean_cost,
    )
    x, margin, p0, a, adv, target = tensors
    actor.train().requires_grad_(True)
    before_training = copy.deepcopy(actor.state_dict())
    normalized = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(0.05)
    taken = torch.where(a.bool(), p0, 1 - p0)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    epochs = 0
    for _ in range(18):
        p = probabilities(margin, actor(x), bound, temperature)
        ratio = torch.where(a.bool(), p, 1 - p) / taken
        surrogate = torch.minimum(
            ratio * normalized, ratio.clamp(0.85, 1.15) * normalized
        )
        entropy = -(p * p.log() + (1 - p) * torch.log1p(-p))
        optimizer.zero_grad(set_to_none=True)
        loss = -(surrogate + 0.002 * entropy).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        previous = copy.deepcopy(actor.state_dict())
        optimizer.step()
        with torch.no_grad():
            p = probabilities(margin, actor(x), bound, temperature)
            kl = (p0 * (p0 / p).log() + (1 - p0) * ((1 - p0) / (1 - p)).log()).mean()
            drift = ((p > 0.5) != (p0 > 0.5)).float().mean()
            if kl > 0.025 or drift > 0.16:
                actor.load_state_dict(previous)
                break
        epochs += 1
    critic = torch.optim.AdamW(value.parameters(), lr=0.001)
    for _ in range(48):
        critic.zero_grad(set_to_none=True)
        loss = nn.functional.smooth_l1_loss(value(x).flatten(), target)
        loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), 2.0)
        critic.step()
    with torch.no_grad():
        p = probabilities(margin, actor(x), bound, temperature)
        kl = (p0 * (p0 / p).log() + (1 - p0) * ((1 - p0) / (1 - p)).log()).mean()
    deployed = packet(actor, bound, base)
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        (stage / "world_iql.bin").write_bytes(deployed)
        torch.save(
            {
                "actor": actor.state_dict(),
                "critics": {"value": value.state_dict()},
                "bound": bound,
            },
            stage / "learner.pt",
        )
        report = {
            "method": "On-policy deep PPO on the nine-win controller with elapsed-time GAE and a learned value baseline.",
            "parent_sha256": digest(base),
            "behavior_sha256": behavior_hash,
            "rollout_sha256": digest(data.read_bytes()),
            "value_input_sha256": digest(learner.read_bytes()),
            "policy_sha256": digest(deployed),
            "accepted_epochs": epochs,
            "decisions": len(x),
            "mean_kl": float(kl),
            "changed_greedy_blocks": int(((p > 0.5) != (p0 > 0.5)).sum()),
            "actor_parameter_change_l2": {
                k: float((v - before_training[k]).norm())
                for k, v in actor.state_dict().items()
            },
            "clean_cost": clean_cost,
            "temperature": temperature,
            "episodes": outcomes,
        }
        (stage / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "data",
        "parent",
        "behavior",
        "learner",
        "primary",
        "alternate",
        "output",
    ):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--clean-cost", type=float, default=0.0)
    train(**vars(parser.parse_args()))

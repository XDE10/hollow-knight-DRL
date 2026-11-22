"""Verify native on-policy samples and update the recurrent actor on measured game rewards."""

from __future__ import annotations

import argparse
import json
import struct
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from audit_sisters_mainline_replay import read_episode
from torch import Tensor, nn
from torch.nn import functional as F
from train_hkrl_wm_recurrent_rl import (
    GAMMA,
    RecurrentActor,
    RecurrentValues,
    prepare_fight,
    recurrent_features,
    sha256,
)
from train_sisters_recurrent_world import HORIZONS, STEPS, World
from train_sisters_timing_iql import motion_history

from hkrl.training.gae import compute_gae

DRAW_HEADER = struct.Struct("<IQ32sid")
DRAW_ROW = np.dtype(
    [("time", "<f4"), ("action", "<i4"), ("uniform", "<f8"), ("log_probability", "<f8")]
)
MASK = (1 << 64) - 1
INCREMENT = 0x9E3779B97F4A7C15


def uniforms(episode: int, seed: int, n: int) -> np.ndarray:
    state = (seed & 0xFFFFFFFF) ^ ((episode * INCREMENT) & MASK)
    result = np.empty(n, dtype="f8")
    for i in range(n):
        state = (state + INCREMENT) & MASK
        x = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
        result[i] = ((x ^ (x >> 31)) >> 11) / (1 << 53)
    return result


def read_native_draws(
    path: Path,
    episode: int,
    behavior: str,
    seed: int,
    temperature: float,
    clock: np.ndarray,
    observation: np.ndarray,
) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    if len(raw) < DRAW_HEADER.size or DRAW_HEADER.unpack_from(raw) != (
        0x314F5050,
        episode,
        bytes.fromhex(behavior),
        seed,
        temperature,
    ):
        raise ValueError("PPO draw header does not match the frozen actor or assigned RNG")
    packet = np.frombuffer(raw, dtype=DRAW_ROW, offset=DRAW_HEADER.size)
    rows = np.flatnonzero(observation[:, 23] > 0.5)
    if (
        not np.array_equal(packet["time"], clock[rows])
        or not np.array_equal(packet["uniform"], uniforms(episode, seed, len(rows)))
        or (packet["action"] < 0).any()
        or not np.isfinite(packet["log_probability"]).all()
        or (packet["log_probability"] > 0).any()
    ):
        raise ValueError("PPO draws must cover every Trio decision with the verified game RNG")
    return {
        "ppo_rows": rows,
        **{
            f"ppo_{field}": packet[field].copy()
            for field in ("action", "uniform", "log_probability")
        },
    }


@dataclass(frozen=True)
class Rollout:
    inputs: Tensor
    action: Tensor
    old_log_distribution: Tensor
    old_value: Tensor
    advantage: Tensor
    returns: Tensor
    world_rows: Tensor


@torch.no_grad()
def verified_rollouts(
    folders: list[Path],
    plan: dict,
    actor: RecurrentActor,
    critic: RecurrentValues,
    catalogue: Tensor,
) -> tuple[list[Rollout], list[dict]]:
    found = {}
    for folder in folders:
        for index, item in enumerate(json.loads((folder / "manifest.json").read_bytes())):
            receipt = json.loads((folder / f"episode-{index:03d}.json").read_bytes())
            seed = receipt["unity_reset_seed"]
            source = folder / item["file"]
            if (
                receipt["seed"] != seed
                or seed in found
                or seed not in plan["seeds"]
                or receipt["behavior_policy_sha256"] != plan["actor_sha256"]
                or receipt["mod_sha256"] != plan["mod_sha256"]
                or receipt["teacher_fraction"] != 0
                or receipt["game_speed"] != 1
                or not all(float(k) <= 1.001 for k in receipt["speed_counts"])
                or receipt["ppo_temperature"] != plan["temperature"]
                or receipt["sampler_draw_seed"] != plan["draw_seeds"][str(seed)]
                or sha256(source) != receipt["tape_sha256"]
                or item["tape_sha256"] != receipt["tape_sha256"]
            ):
                raise ValueError(
                    f"PPO fight {seed} does not match its registered rollout assignment"
                )
            found[seed] = source, receipt
    if set(found) != set(plan["seeds"]) or len(found) != len(plan["seeds"]):
        raise ValueError("The complete registered training cohort is required")
    lookup = {tuple(a): i for i, a in enumerate(catalogue.tolist())}
    rollouts = []
    evidence = []
    for seed in plan["seeds"]:
        source, receipt = found[seed]
        with np.load(source, allow_pickle=False) as tape:
            arrays = {k: tape[k] for k in tape.files}
        episode = read_episode(
            {k: v for k, v in arrays.items() if not k.startswith("ppo_")},
            {"won": receipt["win"], "truncated": receipt["truncated"]},
        )
        fight = prepare_fight(episode, lookup)
        n = len(episode.clock)
        selected = arrays["ppo_rows"] < n
        rows = arrays["ppo_rows"][selected]
        actions = arrays["ppo_action"][selected]
        if (
            not np.array_equal(rows, np.flatnonzero(episode.observation[:, 23] > 0.5))
            or (actions >= len(catalogue)).any()
            or (actions < 0).any()
            or not np.array_equal(catalogue.numpy()[actions], episode.action[rows])
        ):
            raise ValueError(f"Sampled PPO keys disagree with applied keys for fight {seed}")
        stats = {
            "seed": seed,
            "win": receipt["win"],
            "truncated": receipt["truncated"],
            "decisions": len(rows),
            "max_logp_error": 0.0,
            "masks_lost": sum(receipt["damage_taken"].values()),
        }
        evidence.append(stats)
        if not len(rows):
            continue
        observations = []
        logits = []
        values = []
        for lo, hi in fight.segments:
            x = torch.from_numpy(
                motion_history(episode.clock, episode.observation, np.arange(lo, hi))
            )
            observations.append(x)
            logits.append((actor(x[None])[0].double() / plan["temperature"]).log_softmax(-1))
            values.append(critic(x[None])[2][0])
        old = torch.cat(logits)
        value = torch.cat(values).numpy()
        selected_logp = old[torch.arange(len(rows)), torch.from_numpy(actions).long()].numpy()
        log_error = float(np.max(np.abs(selected_logp - arrays["ppo_log_probability"][selected])))
        cumulative = old.exp().cumsum(-1).numpy()
        upper = cumulative[np.arange(len(rows)), actions]
        lower = upper - np.exp(selected_logp)
        u = arrays["ppo_uniform"][selected]
        if log_error > 2e-4 or (u < lower - 2e-5).any() or (u >= upper + 2e-5).any():
            raise ValueError(
                f"PPO behavior log likelihood does not reproduce for {seed}: {log_error}"
            )
        stats["max_logp_error"] = log_error

        # Native sampling logs each decision. At a time limit the last pre-action
        # observation is the bootstrap; the unresolved action itself is discarded.
        truncation = bool(episode.truncated[-1])
        count = len(rows) - int(truncation)
        if count <= 0:
            continue
        done = np.zeros(count, dtype=bool)
        truncated = done.copy()
        done[-1] = True
        truncated[-1] = truncation
        exponent = np.r_[np.diff(episode.clock[rows]) / 0.04, 1.0][:count]
        last = np.asarray([value[-1] if truncation else 0.0], dtype="f4")
        advantage, returns = compute_gae(
            episode.reward[rows[:count], None],
            value[:count, None],
            done[:, None],
            truncated[:, None],
            last,
            gamma=GAMMA,
            discount_exponents=exponent[:, None],
        )
        offset = 0
        for x, log_dist, val, (lo, hi) in zip(
            observations, logits, values, fight.segments, strict=True
        ):
            length = min(hi - lo, count - offset)
            if length <= 0:
                break
            section = slice(offset, offset + length)
            exact = np.abs(np.diff(episode.clock[lo : lo + length]) - 0.04) < 1e-4
            consecutive = np.r_[0, np.cumsum(~exact)]
            possible = np.arange(max(0, length - STEPS))
            world_rows = possible[consecutive[possible] == consecutive[possible + STEPS]][::4]
            rollouts.append(
                Rollout(
                    x[:length],
                    torch.from_numpy(actions[section]).long(),
                    log_dist[:length].float(),
                    val[:length],
                    torch.from_numpy(advantage[section, 0]),
                    torch.from_numpy(returns[section, 0]),
                    torch.from_numpy(world_rows),
                )
            )
            offset += length
    if not rollouts:
        raise ValueError("The cohort contains no verified Trio actions")
    return rollouts, evidence


def load_world(
    path: Path, validation: Path, offline: Path, latent: int, actions: int
) -> tuple[list[World], Tensor]:
    report = json.loads(validation.read_bytes())
    expected = sha256(offline)
    if report["actor_sha256"] != expected or report["world_sha256"] != sha256(path):
        raise ValueError("World model must descend from the frozen offline actor")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["actor_sha256"] != expected:
        raise ValueError("Unexpected world-weight lineage")
    m = report["measures"]["equal_fight_means"]
    error = np.asarray(m["world_mae_xy"])
    validated = error < np.minimum(m["persistence_mae_xy"], m["constant_velocity_mae_xy"])
    validated &= error < np.asarray(m["shuffled_action_mae_xy"])
    if not validated.any():
        raise ValueError("No forecast component improves on both baselines and shuffled actions")
    models = [World(latent, actions) for _ in data["world"]]
    for model, state in zip(models, data["world"], strict=True):
        model.load_state_dict(state)
        model.eval().requires_grad_(False)
    return models, torch.from_numpy(validated)


def optimize(
    actor: RecurrentActor,
    critic: RecurrentValues,
    rollouts: list[Rollout],
    worlds: list[World],
    validated: Tensor,
    temperature: float,
    updates: int,
) -> list[dict]:
    optimizer = torch.optim.AdamW(
        [{"params": actor.parameters(), "lr": 7e-5}, {"params": critic.parameters(), "lr": 2e-4}]
    )
    all_advantages = torch.cat([r.advantage for r in rollouts])
    mean, scale = all_advantages.mean(), all_advantages.std(unbiased=False).clamp_min(0.05)
    samples = len(all_advantages)
    progress = []
    for step in range(updates):
        before_actor, before_critic = deepcopy(actor.state_dict()), deepcopy(critic.state_dict())
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros(3)
        for rollout in rollouts:
            features, hidden = recurrent_features(
                actor.encoder, actor.memory, rollout.inputs[None], 0
            )
            logits = actor.encoder.heads(features + actor.memory_projection(hidden))[0][0]
            log_p = (logits / temperature).log_softmax(-1)
            chosen = rollout.action[:, None]
            ratio = (
                (log_p.gather(1, chosen) - rollout.old_log_distribution.gather(1, chosen))
                .flatten()
                .exp()
            )
            advantage = ((rollout.advantage - mean) / scale).clamp(-5, 5)
            ppo = (
                -torch.minimum(ratio * advantage, ratio.clamp(0.9, 1.1) * advantage).sum() / samples
            )
            value = critic(rollout.inputs[None])[2][0]
            clipped = rollout.old_value + (value - rollout.old_value).clamp(-0.2, 0.2)
            value_loss = (
                0.5
                * torch.maximum(
                    (value - rollout.returns).square(), (clipped - rollout.returns).square()
                ).sum()
                / samples
            )
            anchor = (
                rollout.old_log_distribution.exp() * (rollout.old_log_distribution - log_p)
            ).sum() / samples
            aux = ppo.new_zeros(())
            if len(rollout.world_rows):
                start = rollout.world_rows
                seq = start[:, None] + torch.arange(STEPS)
                future = start[:, None] + torch.tensor(HORIZONS)
                target = rollout.inputs[future, :2] - rollout.inputs[start, None, :2]
                latent = torch.cat((features[0, start], hidden[0, start]), -1)
                forecasts = torch.stack(
                    [
                        model(latent, rollout.inputs[start, :6], rollout.action[seq])[..., :2]
                        for model in worlds
                    ]
                )
                aux = (
                    0.05
                    * F.smooth_l1_loss(
                        forecasts.mean(0)[:, validated], target[:, validated], beta=0.04
                    )
                    * len(start)
                    * 4
                    / samples
                )
            (ppo + value_loss + 0.02 * anchor + aux).backward()
            objective += torch.stack((ppo.detach(), value_loss.detach(), aux.detach()))
        nn.utils.clip_grad_norm_(
            [*actor.parameters(), *critic.parameters()], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            kl = 0.0
            for rollout in rollouts:
                new = (actor(rollout.inputs[None])[0] / temperature).log_softmax(-1)
                kl += (
                    float(
                        (
                            rollout.old_log_distribution.exp()
                            * (rollout.old_log_distribution - new)
                        ).sum()
                    )
                    / samples
                )
        accepted = np.isfinite(kl) and kl <= 0.015
        progress.append(
            {
                "step": step + 1,
                "ppo": float(objective[0]),
                "critic": float(objective[1]),
                "world_aux": float(objective[2]),
                "kl": kl,
                "accepted": bool(accepted),
            }
        )
        if not accepted:
            actor.load_state_dict(before_actor)
            critic.load_state_dict(before_critic)
            break
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("offline", "behavior", "native", "world", "world-validation", "plan", "output"):
        parser.add_argument("--" + path, type=Path, required=True)
    parser.add_argument("--rollout", type=Path, nargs="+", required=True)
    parser.add_argument("--updates", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(3)
    plan = json.loads(args.plan.read_bytes())
    if (
        sha256(args.native) != plan["actor_sha256"]
        or sha256(args.behavior) != plan["learner_sha256"]
        or args.updates < 1
    ):
        raise ValueError("The native PPO policy and Python weights must match their frozen plan")
    saved = torch.load(args.behavior, map_location="cpu", weights_only=True)
    catalogue = saved["catalogue"]
    hidden = saved["provenance"]["hidden"]
    actor, critic = RecurrentActor(catalogue, hidden), RecurrentValues(catalogue, hidden)
    actor.load_state_dict(saved["state"])
    critic.load_state_dict(saved["critic"])
    worlds, validated = load_world(
        args.world, args.world_validation, args.offline, 192 + hidden, len(catalogue)
    )
    rollouts, evidence = verified_rollouts(args.rollout, plan, actor, critic, catalogue)
    progress = optimize(
        actor, critic, rollouts, worlds, validated, plan["temperature"], args.updates
    )
    args.output.mkdir(parents=True, exist_ok=False)
    asset = args.output / "learner.pt"
    accepted = sum(row["accepted"] for row in progress)
    if accepted:
        torch.save(
            {
                **saved,
                "state": actor.state_dict(),
                "critic": critic.state_dict(),
                "ppo": {
                    "parent_sha256": plan["learner_sha256"],
                    "world_sha256": sha256(args.world),
                    "plan_sha256": sha256(args.plan),
                    "accepted_updates": accepted,
                },
            },
            asset,
        )
    result = {
        "progress": progress,
        "verified_fights": evidence,
        "validated_world_position_components": validated.tolist(),
        "checkpoint_sha256": sha256(asset) if accepted else None,
    }
    (args.output / "training.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Run direct, rule-free primitive actions and optionally update PPO from real on-policy battles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from hkrl import protocol
from hkrl.env import HKRLEnv
from hkrl.training.local_session import exclusive_game_lock
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config
from sisters_latent_ac import (
    CATALOG,
    GAMMA,
    SCHEMA,
    WEIGHTS,
    ActorCritic,
    controller_action,
    encode,
    lambda_returns,
    legal_actions,
)
from train_sisters_latent_ac import optimize_ppo
from train_sisters_object_world import ObjectWorld


def event_vector(events) -> tuple[np.ndarray, float, Counter]:
    vector = np.zeros(8, np.float32)
    damage = 0.0
    lost = Counter()
    for e in events:
        kind = protocol.RewardEventKind(e.kind).name
        if kind == "DAMAGE_DEALT":
            damage += e.amount
            vector[0] = 1
        elif kind == "DAMAGE_TAKEN":
            idx = int(e.aux_int2)
            vector[idx if idx in (1, 2, 3) else 4] = 1
            lost[protocol.DamageSource(idx).name if idx in (1, 2, 3) else "OTHER"] += (
                e.amount
            )
        elif kind == "BOSS_KILLED" and e.entity_id == 0:
            vector[6] = 1
        elif kind == "PLAYER_DEATH":
            vector[7] = 1
        elif kind == "HEAL":
            vector[5] = 1
    return vector, damage, lost


def verify_native_off(port: int) -> int:
    import psutil

    listeners = [
        c
        for c in psutil.net_connections("tcp")
        if c.status == "LISTEN" and c.laddr.port == port
    ]
    if len(listeners) != 1 or listeners[0].laddr.ip != "127.0.0.1":
        raise RuntimeError(f"Expected one local game server at {port}")
    pid = listeners[0].pid
    proc = psutil.Process(pid)
    if proc.name().lower() != "hollow_knight.exe" or proc.environ().get(
        "HKRL_TACTICS_POLICY"
    ):
        raise RuntimeError(
            "Learned actor requires a game with the native tactical controller disabled"
        )
    if float(proc.environ().get("HKRL_SISTERS_NAIL_MULTIPLIER", "1")) != 1:
        raise RuntimeError("Expected original nail damage")
    return pid


class LivePolicy:
    def __init__(self, path: Path):
        self.payload = torch.load(path, map_location="cpu", weights_only=True)
        if self.payload["schema"] != SCHEMA or self.payload["catalog"] != CATALOG:
            raise ValueError("checkpoint incompatible with action catalogue")
        self.encoder = ObjectWorld().eval().requires_grad_(False)
        self.encoder.load_state_dict(self.payload["encoder"])
        self.mean, self.std = self.payload["normalization"]
        self.policy = ActorCritic(len(self.mean)).eval()
        self.policy.load_state_dict(self.payload["actor_critic"])

    @torch.no_grad()
    def state(self, obs):
        p = obs["player"][None]
        e = obs["entities"][obs["entity_mask"].astype(bool)]
        z = encode(
            self.encoder,
            p,
            e,
            np.asarray([0, len(e)]),
            self.payload.get("combat_inputs", False),
        )
        return ((z - self.mean) / self.std).clamp(-8, 8)[0]


def episode(
    env, agent: LivePolicy, seed: int, training: bool, scale: float
) -> tuple[dict, dict]:
    obs, info = env.reset(seed=seed, options={"reset_timeout_s": 90})
    eid = info["episode_id"]
    if list(obs["player"][4:7]) != [9, 9, 0]:
        raise RuntimeError("Unexpected starting health or soul")
    opening = obs["entities"][obs["entity_mask"].astype(bool)]
    opening = opening[opening[:, 1] == 1]
    if len(opening) != 1 or list(opening[0, 12:14]) != [500, 500]:
        raise RuntimeError("Unexpected arena or opening health")
    env.set_timescale(scale)
    collected = {
        k: []
        for k in (
            "state",
            "mask",
            "action",
            "logp",
            "value",
            "reward",
            "next_value",
            "discount",
        )
    }
    damage = 0.0
    lost = Counter()
    count = Counter()
    trio = False
    elapsed = 0
    won = False
    health_floor = 9
    scale_counts = Counter()
    state = agent.state(obs)
    for _ in range(12000):
        mask = torch.from_numpy(legal_actions(info["action_mask"]))
        with torch.no_grad():
            dist, value = agent.policy(state, mask)
            a = dist.sample() if training else dist.probs.argmax()
            logp = dist.log_prob(a)
        nxt, _, term, trunc, info = env.step(controller_action(int(a)))
        if info["episode_id"] != eid or info["elapsed_ticks"] <= 0:
            raise RuntimeError("episode discontinuity")
        dt = 0.02 * info["elapsed_ticks"]
        elapsed += dt
        scale_counts[round(float(nxt["global"][5]), 2)] += 1
        events, dealt, hits = event_vector(info["reward_events"])
        damage += dealt
        lost.update(hits)
        count[int(a)] += 1
        won |= bool(events[6])
        health_floor = min(health_floor, int(nxt["player"][4]))
        ents = nxt["entities"][nxt["entity_mask"].astype(bool)]
        bosses = ents[ents[:, 1] == 1]
        if len(bosses) == 3:
            if any(b[13] != 750 for b in bosses):
                raise RuntimeError("Unexpected trio health")
            trio = True
        next_state = agent.state(nxt)
        if training:
            with torch.no_grad():
                next_value = agent.policy(next_state)[1]
            values = {
                "state": state,
                "mask": mask,
                "action": a,
                "logp": logp,
                "value": value,
                "reward": torch.tensor(float(events @ WEIGHTS), dtype=torch.float32),
                "next_value": next_value,
                "discount": torch.tensor(0.0 if term else GAMMA ** (dt / 0.12)),
            }
            for k, v in values.items():
                collected[k].append(v)
        state = next_state
        if term or trunc:
            break
    else:
        raise RuntimeError("episode did not finish")
    if not scale_counts or scale_counts.most_common(1)[0][0] != scale:
        raise RuntimeError(f"Unexpected game speed: {scale_counts}")
    result = {
        "seed": seed,
        "episode_id": int(eid),
        "won": bool(won and term and not trunc and nxt["player"][4] > 0),
        "first_cleared": trio,
        "damage": damage,
        "lost": dict(lost),
        "minimum_hp": health_floor,
        "truncated": bool(trunc),
        "seconds": round(elapsed, 2),
        "actions": dict(count),
        "decisions": sum(count.values()),
        "scale_counts": dict(scale_counts),
    }
    if training:
        data = {k: torch.stack(v) for k, v in collected.items()}
        data["target"] = lambda_returns(
            data["reward"], data["next_value"], data["discount"]
        )
        data["advantage"] = data["target"] - data["value"]
    else:
        data = {}
    return result, data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--scale", type=float, default=1)
    p.add_argument("--update-every", type=int, default=4)
    args = p.parse_args()
    if (
        args.episodes < 1
        or args.update_every < 1
        or args.scale <= 0
        or (not args.train and args.scale != 1)
    ):
        p.error("invalid episode count, update interval or evaluation speed")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cfg = load_train_config("configs/train/sisters_platform.yaml")
    task = load_task_config("configs/experiments/sisters_counter.yaml")
    task.action.action_repeat = 5
    task.time_limit_seconds = 480
    agent = LivePolicy(args.checkpoint)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    pid = verify_native_off(args.port)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    (out / "run.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "native_tactics": False,
                "trained": args.train,
                "action_selection": "sampled" if args.train else "greedy",
                "pid": pid,
                "port": args.port,
                "seed": args.seed,
                "episodes": args.episodes,
                "scale": args.scale,
            }
        )
    )
    opt = torch.optim.Adam(agent.policy.parameters(), lr=0.00012)
    batch = []
    updates = []
    results = []
    with exclusive_game_lock(Path(f"runs/training/sisters-actor-{args.port}.lock")):
        env = HKRLEnv(make_transport(cfg, port=args.port), task, step_timeout_s=20)
        try:
            for i in range(args.episodes):
                result, data = episode(
                    env, agent, args.seed + i, args.train, args.scale
                )
                result["policy_update"] = len(updates)
                results.append(result)
                (out / f"episode-{i:03d}.json").write_text(json.dumps(result))
                print(json.dumps(result), flush=True)
                if args.train:
                    batch.append(data)
                    if len(batch) == args.update_every or i + 1 == args.episodes:
                        b = {k: torch.cat([game[k] for game in batch]) for k in data}
                        with torch.no_grad():
                            behavior, _ = agent.policy(b["state"], b["mask"])
                            mismatch = (
                                (behavior.log_prob(b["action"]) - b["logp"])
                                .abs()
                                .max()
                                .item()
                            )
                            if mismatch > 1e-3:
                                raise RuntimeError(
                                    "Logged action probability differs from the collecting policy"
                                )
                            old_actor = agent.policy.actor.weight.clone()
                            old_critic = agent.policy.critic.weight.clone()
                        # Retain the actual on-policy batch with episode boundaries for independent replay.
                        torch.save(
                            {
                                "batch": b,
                                "lengths": [len(game["state"]) for game in batch],
                            },
                            out / f"rollout-{len(updates):03d}.pt",
                        )
                        stats = optimize_ppo(
                            agent.policy,
                            opt,
                            b["state"],
                            b["action"],
                            b["logp"],
                            b["target"],
                            b["advantage"],
                            mask=b["mask"],
                        )
                        updates.append(
                            dict(
                                steps=len(b["state"]),
                                behavior_logp_max_error=mismatch,
                                actor_head_change=(
                                    agent.policy.actor.weight.detach() - old_actor
                                )
                                .norm()
                                .item(),
                                critic_head_change=(
                                    agent.policy.critic.weight.detach() - old_critic
                                )
                                .norm()
                                .item(),
                                **stats,
                            )
                        )
                        batch = []
                        payload = dict(
                            agent.payload,
                            actor_critic=agent.policy.state_dict(),
                            stage="real_ppo",
                            parent_sha256=digest,
                            real_updates=updates,
                        )
                        temp = out / "real_ppo.tmp"
                        torch.save(payload, temp)
                        temp.replace(out / "real_ppo.pt")
                        (out / "updates.json").write_text(json.dumps(updates, indent=2))
        finally:
            try:
                env.set_timescale(1)
            finally:
                env.close()
    sources = Counter()
    for row in results:
        sources.update(row["lost"])
    summary = {
        "episodes": len(results),
        "first_cleared": sum(row["first_cleared"] for row in results),
        "wins": sum(row["won"] for row in results),
        "no_hit_wins": sum(
            row["won"] and row["minimum_hp"] == 9 and not sum(row["lost"].values())
            for row in results
        ),
        "mean_damage": sum(row["damage"] for row in results) / len(results),
        "masks_lost": dict(sources),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

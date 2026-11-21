"""Update a six-way native maneuver policy from complete on-policy arena fights."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from sisters_directional_options import (
    GROUPS,
    HEAD_MAGIC,
    HOLD_SECONDS,
    INTERVAL_SECONDS,
    behavior_probabilities,
    menu_logits,
    menu_slots,
)
from train_sisters_option_ppo import (
    discounted_scan,
    option_rewards,
    ridge_value,
    sha,
    start_policy,
    verified_events,
    verify_option_inputs,
)
from train_sisters_timing_iql import (
    DIRECTIONAL_MAGIC,
    TimingPolicy,
    motion_history,
    phase_checkpoint,
    phase_components,
    policy_gradient_components,
)


def initial_head(risk: Path, interval: float = INTERVAL_SECONDS) -> bytes:
    base = start_policy(risk, interval)
    return (
        struct.pack("<i", HEAD_MAGIC)
        + base[4 : 8 + 193 * 4]
        + np.zeros((GROUPS, 193), "<f4").tobytes()
    )


def categorical(logits: torch.Tensor) -> torch.Tensor:
    valid = logits.isfinite()
    return 0.95 * logits.softmax(-1) + 0.05 * valid / valid.sum(-1, keepdim=True)


def fit_selector(
    features: np.ndarray,
    logits: np.ndarray,
    probability: np.ndarray,
    chosen: np.ndarray,
    advantages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x, baseline, p0, advantage = (
        torch.as_tensor(v, dtype=torch.float64)
        for v in (features, logits, probability, advantages)
    )
    indices = torch.arange(len(x))
    a = torch.as_tensor(chosen, dtype=torch.long)
    layer = torch.nn.Linear(x.shape[1], GROUPS, dtype=torch.float64)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    opt = torch.optim.Adam(layer.parameters(), lr=0.012)
    advantage = (
        (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(0.05)
    ).clamp(-5, 5)
    old_taken = p0[indices, a]
    epochs = 0
    final_kl = 0.0
    for _ in range(12):
        p = categorical(baseline + layer(x))
        ratio = p[indices, a] / old_taken
        clipped = torch.minimum(ratio * advantage, ratio.clamp(0.85, 1.15) * advantage)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(1)
        loss = -(clipped + 0.002 * entropy).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer.parameters(), 0.5)
        before = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        opt.step()
        with torch.no_grad():
            p = categorical(baseline + layer(x))
            kl = (
                (p0 * (p0.clamp_min(1e-12).log() - p.clamp_min(1e-12).log()))
                .sum(1)
                .mean()
                .item()
            )
            if not np.isfinite(kl) or kl > 0.035:
                layer.load_state_dict(before)
                break
            final_kl = kl
            epochs += 1
    with torch.no_grad():
        updated = baseline + layer(x)
        return (
            layer.weight.numpy().copy(),
            layer.bias.numpy().copy(),
            {
                "epochs": epochs,
                "kl": final_kl,
                "changes_to_greedy_option": int(
                    (baseline.argmax(1) != updated.argmax(1)).sum()
                ),
            },
        )


def train(
    data: Path,
    parent: Path,
    learner: Path,
    sampler: Path,
    action_world: Path,
    output: Path,
    clean_cost: float = 0.0,
) -> None:
    if output.exists() or not np.isfinite(clean_cost) or clean_cost < 0:
        raise ValueError("Expected a fresh output directory and a nonnegative cost")
    torch.set_num_threads(1)
    ck = torch.load(learner, map_location="cpu", weights_only=True)
    cat = ck["catalogue"].long()
    actor = TimingPolicy(cat, True).eval().requires_grad_(False)
    actor.load_state_dict(ck["actor"])
    opener, leaf = phase_components(parent.read_bytes())
    head = sampler.read_bytes()
    wrapper = struct.pack("<2i", DIRECTIONAL_MAGIC, len(leaf)) + leaf
    policy_gradient_components(wrapper + head)
    weights = np.frombuffer(head, "<f4", offset=8).reshape(1 + GROUPS, 193).astype("f8")
    risk, selector = weights[0], weights[1:]
    world = json.loads((action_world / "training.json").read_bytes())
    proj = np.load(action_world / "risk-projection.npy", allow_pickle=False).astype(
        "f8"
    )
    if (
        proj.shape != (192, 10)
        or sha(action_world / "risk-projection.npy") != world["projection_sha256"]
        or sha(learner) != world["learner_sha256"]
    ):
        raise ValueError("The frozen action world differs from the parent actor")
    outcomes = []
    games = []
    digest = sha(sampler)
    parent_hash = sha(parent)
    seen, draw_seeds = set(), set()
    with zipfile.ZipFile(data) as z, torch.inference_mode():
        for g in json.loads(z.read("manifest.json")):
            source = g["source"]
            if (
                source in seen
                or g["sampler_draw_seed"] in draw_seeds
                or not source.startswith("native-development/")
                or g["split"] != "train"
                or g["teacher_labels"]
                or g.get("directional_groups") != GROUPS
                or g["sampler_sha256"] != digest
                or g["behavior_policy_sha256"] != parent_hash
            ):
                raise ValueError(
                    "Unexpected directional behavior policy, repeated game or sampler seed"
                )
            seen.add(source)
            draw_seeds.add(g["sampler_draw_seed"])
            raw = z.read(g["file"])
            if hashlib.sha256(raw).hexdigest() != g["sha256"]:
                raise ValueError("Rollout checksum mismatch")
            with np.load(io.BytesIO(raw)) as d:
                s, t, rows = d["timing_state"], d["control_time"], d["choice_rows"]
                ev = verified_events(
                    t, s[:, 4] * 9, d["time"], d["elapsed_ticks"], d["events"]
                )
                if bool(ev[:, 6].sum()) != bool(g["won"]) or ev[:, 6:].sum() != 1:
                    raise ValueError("Missing arena terminal")
                loss = ev[:, 1:5].sum(1)
                outcomes.append(
                    {
                        "source": source,
                        "won": bool(g["won"]),
                        "masks_lost": float(loss.sum()),
                        "boss_damage": float(ev[:, 0].sum()),
                        "decisions": len(rows),
                    }
                )
                if not len(rows):
                    continue
                h = motion_history(t, s)
                emb = actor.encode(torch.from_numpy(h[rows]))
                scores = actor.heads(emb)[0].numpy()
                x = emb.numpy().astype("f8")
                slots, primary = menu_slots(scores, cat.numpy(), s[rows])
                draw = d["menu_draw"].astype("i8")
                idx = np.arange(len(rows))
                logits = menu_logits(x, selector, slots, primary)
                prob = behavior_probabilities(logits)
                stops = np.searchsorted(t, t[rows] + HOLD_SECONDS)
                if (
                    not np.array_equal(slots, d["menu_slots"])
                    or not np.array_equal(primary, d["menu_primary"])
                    or not np.allclose(prob, d["menu_probability"], rtol=0, atol=3e-5)
                    or not np.allclose(
                        x @ risk[1:] + risk[0], d["menu_risk"], rtol=0.001, atol=0.001
                    )
                    or not np.array_equal(stops, d["choice_hold_stop"])
                    or (slots[idx, draw] < 0).any()
                    or not np.array_equal(
                        cat.numpy()[slots[idx, draw]], d["controls"][rows]
                    )
                ):
                    raise ValueError(
                        "Native proposals, sampled probabilities or issued keys differ"
                    )
                pairs = np.column_stack((slots[idx, primary], slots[idx, draw]))
                outcomes[-1]["verified_continuation_inputs"] = verify_option_inputs(
                    actor,
                    cat.numpy(),
                    h,
                    d["controls"],
                    rows,
                    stops,
                    pairs,
                    (primary != draw).astype("i8"),
                )
                rew, discount, trace = option_rewards(t, loss, ev, rows, clean_cost)
                games.append(
                    (
                        x,
                        logits,
                        d["menu_probability"].astype("f8"),
                        draw,
                        rew,
                        discount,
                        trace,
                    )
                )
    if len(games) < 4:
        raise ValueError("Too few complete fights with an eligible action")
    x = np.vstack([g[0] for g in games])
    ids = np.repeat(np.arange(len(games)), [len(g[0]) for g in games])
    center = x.mean(0)
    basis = proj
    features = (x - center) @ basis
    scale = np.sqrt((features**2).mean(0)).clip(0.05)
    features /= scale
    returns = np.concatenate([discounted_scan(g[4], g[5]) for g in games])
    uniform = np.ones(len(x))
    advantages = []
    for i, (_, _, _, _, rew, discount, trace) in enumerate(games):
        other = ids != i
        beta = ridge_value(features[other], returns[other], uniform[other])
        value = beta[0] + features[~other] @ beta[1:]
        advantages.append(
            discounted_scan(
                rew + discount * np.r_[value[1:], 0.0] - value, discount * trace
            )
        )
    critic = ridge_value(features, returns, uniform)
    logits, probability, actions = (
        np.concatenate([g[i] for g in games]) for i in (1, 2, 3)
    )
    slopes, biases, update = fit_selector(
        features, logits, probability, actions, np.concatenate(advantages)
    )
    native_slope = (slopes / scale) @ basis.T
    new = (
        selector + np.column_stack((biases - native_slope @ center, native_slope))
    ).astype("<f4")
    projected = x @ (new[:, 1:] - selector[:, 1:]).T + new[:, 0] - selector[:, 0]
    np.testing.assert_allclose(
        projected, features @ slopes.T + biases, rtol=1e-4, atol=1e-4
    )
    new_head = head[: 8 + 193 * 4] + new.tobytes()
    deployed = phase_checkpoint(opener, wrapper + new_head)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        out = Path(temp) / "candidate"
        out.mkdir()
        (out / "world_iql.bin").write_bytes(deployed)
        (out / "sampler.bin").write_bytes(new_head)
        torch.save(
            {
                k: torch.as_tensor(v)
                for k, v in {
                    "selector": new,
                    "critic": critic,
                    "basis": basis,
                    "center": center,
                    "scale": scale,
                }.items()
            },
            out / "learner.pt",
        )
        report = dict(
            method="Categorical on-policy PPO over native maneuver options with elapsed-time GAE",
            decisions=len(x),
            sampled_groups=np.bincount(actions, minlength=GROUPS).tolist(),
            rollout_games=len(outcomes),
            episodes=outcomes,
            first_hit_cost=clean_cost,
            **update,
            base_sha256=parent_hash,
            sampler_input_sha256=digest,
            sampler_sha256=sha(out / "sampler.bin"),
            policy_sha256=sha(out / "world_iql.bin"),
            rollout_sha256=sha(data),
            projection_sha256=world["projection_sha256"],
        )
        (out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        out.rename(output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "parent", "learner", "sampler", "action-world", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--clean-cost", type=float, default=0.0)
    train(**vars(parser.parse_args()))

"""Update the local native option policy with PPO on one batch of real stochastic fights."""

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
from train_sisters_native_iql import attribute_events
from train_sisters_timing_iql import (
    POLICY_GRADIENT_MAGIC,
    TimingPolicy,
    motion_history,
    phase_checkpoint,
    phase_components,
    policy_gradient_components,
)

DISCOUNT_PER_SECOND = 0.985
TRACE_PER_SECOND = 0.92
SAMPLING_MAGIC = 0x31504753


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def start_policy(risk: Path, interval: float = 0.9) -> bytes:
    packet = risk.read_bytes()
    if (
        len(packet) != 4 + 193 * 4
        or struct.unpack_from("<i", packet)[0] != 0x314B5352
        or not 0.32 <= interval <= 8.4
        or not np.isfinite(np.frombuffer(packet, "<f4", offset=4)).all()
    ):
        raise ValueError("Invalid native risk weights or decision interval")
    return (
        struct.pack("<if", SAMPLING_MAGIC, interval)
        + packet[4:]
        + np.zeros(193, "<f4").tobytes()
    )


def verified_events(
    clock: np.ndarray,
    hp: np.ndarray,
    starts: np.ndarray,
    ticks: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Localize the game's events to inputs while conserving its per-source totals."""
    events = attribute_events(clock, starts, ticks, observed, health_masks=hp)
    if not np.allclose(events.sum(0), observed.sum(0), rtol=0, atol=1e-5):
        raise ValueError(
            "Native control timing does not cover the entire game's events"
        )
    return events


def option_rewards(
    clock: np.ndarray,
    loss: np.ndarray,
    events: np.ndarray,
    rows: np.ndarray,
    clean_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Own rewards through the next option, including the verified arena terminal exactly once."""
    terminal = np.flatnonzero(events[:, 6:].any(1))
    if (
        len(terminal) != 1
        or not len(rows)
        or rows[0] < 0
        or np.any(np.diff(rows) <= 0)
        or rows[-1] > terminal[0]
    ):
        raise ValueError("Expected a complete fight and ordered real option onsets")
    end = int(terminal[0]) + 1
    reward = (
        0.45 / 21 * events[:, 0]
        - 1.9 * loss
        - 0.5 * events[:, 3]
        + 12 * (events[:, 6] - events[:, 7])
    )
    debits = np.flatnonzero(loss > 0)
    if clean_cost and len(debits):
        reward[debits[0]] -= clean_cost
    rewards = []
    discounts = []
    traces = []
    for i, (a, b) in enumerate(zip(rows, np.r_[rows[1:], end], strict=True)):
        dt = clock[a:b].astype("f8") - clock[a]
        rewards.append(float(np.dot(reward[a:b], DISCOUNT_PER_SECOND**dt)))
        if i == len(rows) - 1:
            discounts.append(0.0)
            traces.append(0.0)
        else:
            elapsed = float(clock[b] - clock[a])
            discounts.append(DISCOUNT_PER_SECOND**elapsed)
            traces.append(TRACE_PER_SECOND**elapsed)
    return np.array(rewards), np.array(discounts), np.array(traces)


def discounted_scan(reward: np.ndarray, discount: np.ndarray) -> np.ndarray:
    result = np.empty_like(reward)
    tail = 0.0
    for i in range(len(reward) - 1, -1, -1):
        tail = reward[i] + discount[i] * tail
        result[i] = tail
    return result


def ridge_value(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))
    w = weight / weight.mean()
    penalty = np.diag(np.r_[0.001, np.full(x.shape[1], 20.0)])
    return np.linalg.solve((design.T * w) @ design + penalty, design.T @ (w * y))


def verify_option_inputs(
    actor: TimingPolicy,
    catalogue: np.ndarray,
    history: np.ndarray,
    controls: np.ndarray,
    rows: np.ndarray,
    stops: np.ndarray,
    pairs: np.ndarray,
    drawn: np.ndarray,
) -> int:
    """Replay the actual held keys, including permitted fresh jump, dash and cast pulses."""
    control_rows = np.concatenate(
        [np.arange(a + 1, b) for a, b in zip(rows, stops, strict=True)]
    )
    if not len(control_rows):
        return 0
    assignment = np.repeat(np.arange(len(rows)), stops - rows - 1)
    onset = catalogue[pairs[assignment, drawn[assignment]]]
    scores = actor(torch.from_numpy(history[control_rows]))[0].detach().numpy()
    greedy = scores.argmax(1)
    bits = catalogue[:, 2]
    held = (catalogue[:, 0] == onset[:, :1]) & (((bits ^ onset[:, 2:3]) & 2) == 0)
    held &= (bits & 21 & ~bits[greedy, None]) == 0
    allowed = np.where(held, scores, -np.inf).argmax(1)
    chosen = np.where((drawn[assignment] == 1) & held.any(1), allowed, greedy)
    if not np.array_equal(catalogue[chosen], controls[control_rows]):
        raise ValueError("Actual native keys changed inside an option hold")
    return len(control_rows)


def train(
    data: Path,
    parent: Path,
    learner: Path,
    sampler: Path,
    action_world: Path,
    output: Path,
    clean_cost: float = 0.0,
) -> None:
    if output.exists() or clean_cost < 0:
        raise ValueError("Expected fresh output and a nonnegative first-hit cost")
    torch.set_num_threads(1)
    ck = torch.load(learner, map_location="cpu", weights_only=True)
    cat = ck["catalogue"].long()
    actor = TimingPolicy(cat, True).eval().requires_grad_(False)
    actor.load_state_dict(ck["actor"])
    base = parent.read_bytes()
    opener, leaf = phase_components(base)
    old = sampler.read_bytes()
    old_branch = struct.pack("<2i", POLICY_GRADIENT_MAGIC, len(leaf)) + leaf + old
    policy_gradient_components(old_branch)
    risk, old_head = (
        np.frombuffer(old, dtype="<f4", offset=8).reshape(2, 193).astype("f8")
    )
    proj = np.load(action_world / "risk-projection.npy", allow_pickle=False).astype(
        "f8"
    )
    world_doc = json.loads((action_world / "training.json").read_bytes())
    if (
        proj.shape != (192, 10)
        or world_doc["learner_sha256"] != sha(learner)
        or world_doc["projection_sha256"] != sha(action_world / "risk-projection.npy")
    ):
        raise ValueError("World features differ from the frozen game policy")
    games = []
    seen = set()
    digest = sha(sampler)
    parent_hash = sha(parent)
    outcomes = []
    with zipfile.ZipFile(data) as z, torch.inference_mode():
        for g in json.loads(z.read("manifest.json")):
            source = g["source"]
            if (
                source in seen
                or g.get("split") != "train"
                or not source.startswith("native-development/")
                or g["teacher_labels"]
                or g["sampler_sha256"] != digest
                or g["behavior_policy_sha256"] != parent_hash
            ):
                raise ValueError("Duplicate game or different rollout policy")
            seen.add(source)
            raw = z.read(g["file"])
            if hashlib.sha256(raw).hexdigest() != g["sha256"]:
                raise ValueError("Changed trajectory")
            with np.load(io.BytesIO(raw)) as d:
                s, t = d["timing_state"], d["control_time"]
                rows = d["choice_rows"]
                hp = s[:, 4] * 9
                events = verified_events(
                    t, hp, d["time"], d["elapsed_ticks"], d["events"]
                )
                if bool(events[:, 6].sum()) != bool(g["won"]):
                    raise ValueError("Arena terminal mismatch")
                loss = events[:, 1:5].sum(1)
                outcomes.append(
                    {
                        "source": source,
                        "won": bool(g["won"]),
                        "masks_lost": float(loss.sum()),
                        "boss_damage": float(events[:, 0].sum()),
                        "decisions": len(rows),
                    }
                )
                if not len(rows):
                    continue
                arm = d["choice_draw"].astype("i8")
                p = d["choice_probability"]
                history = motion_history(t, s)
                h = actor.encode(torch.from_numpy(history[rows]))
                scores = actor.heads(h)[0].numpy()
                pairs = d["choice_pairs"]
                stops = np.searchsorted(t, t[rows] + np.float32(0.32))
                if not np.array_equal(stops, d["choice_hold_stop"]):
                    raise ValueError(
                        "Recorded option duration differs from the native action"
                    )
                outcomes[-1]["verified_continuation_inputs"] = verify_option_inputs(
                    actor,
                    cat.numpy(),
                    history,
                    d["controls"],
                    rows,
                    stops,
                    pairs,
                    arm,
                )
                x = h.numpy().astype("f8")
                logits = x @ old_head[1:] + old_head[0]
                expected = (1 / (1 + np.exp(-logits.clip(-50, 50)))).clip(0.05, 0.95)
                if (
                    not np.allclose(p, expected, rtol=0, atol=3e-5)
                    or not np.allclose(
                        d["choice_risk"], x @ risk[1:] + risk[0], rtol=0.001, atol=0.001
                    )
                    or not np.allclose(
                        scores[np.arange(len(rows))[:, None], pairs],
                        d["choice_scores"],
                        rtol=0.001,
                        atol=0.001,
                    )
                    or not np.array_equal(
                        cat.numpy()[pairs[np.arange(len(rows)), arm]],
                        d["controls"][rows],
                    )
                ):
                    raise ValueError(
                        "Recorded probabilities, proposals or received inputs differ from the policy"
                    )
                rewards, discounts, traces = option_rewards(
                    t, loss, events, rows, clean_cost
                )
                games.append((source, x, logits, p, arm, rewards, discounts, traces))
    if len(games) < 4:
        raise ValueError("Need at least four complete independent on-policy fights")
    x = np.vstack([g[1] for g in games])
    ids = np.concatenate([np.full(len(g[1]), i) for i, g in enumerate(games)])
    weights = np.ones(len(x))
    center = np.average(x, axis=0, weights=weights)
    _, _, vt = np.linalg.svd(
        (x - center) * np.sqrt(weights[:, None]), full_matrices=False
    )
    basis = np.column_stack((proj, vt[:8].T))
    features = (x - center) @ basis
    scale = np.sqrt(np.average(features**2, axis=0, weights=weights)).clip(0.05)
    features /= scale
    mc = np.concatenate([discounted_scan(g[5], g[6]) for g in games])
    advantages = []
    for i, g in enumerate(games):
        other = ids != i
        critic = ridge_value(features[other], mc[other], weights[other])
        val = critic[0] + features[~other] @ critic[1:]
        td = g[5] + g[6] * np.r_[val[1:], 0.0] - val
        advantages.append(discounted_scan(td, g[6] * g[7]))
    critic = ridge_value(features, mc, weights)
    adv = np.concatenate(advantages)
    adv -= np.average(adv, weights=weights)
    adv /= max(np.sqrt(np.average(adv**2, weights=weights)), 0.05)
    feats, old_log, p0, a, w, adv_t = (
        torch.as_tensor(array, dtype=torch.float32)
        for array in (
            features,
            np.concatenate([g[2] for g in games]),
            np.concatenate([g[3] for g in games]),
            np.concatenate([g[4] for g in games]),
            weights / weights.sum(),
            adv.clip(-5, 5),
        )
    )
    delta = torch.nn.Linear(feats.shape[1], 1)
    torch.nn.init.zeros_(delta.weight)
    torch.nn.init.zeros_(delta.bias)
    optimizer = torch.optim.Adam(delta.parameters(), lr=0.012)
    old_taken = torch.where(a > 0, p0, 1 - p0)
    epochs = 0
    for _ in range(12):
        p = (old_log + delta(feats).flatten()).sigmoid().clamp(0.05, 0.95)
        ratio = torch.where(a > 0, p, 1 - p) / old_taken
        surrogate = torch.minimum(ratio * adv_t, ratio.clamp(0.85, 1.15) * adv_t)
        entropy = -(p * p.log() + (1 - p) * torch.log1p(-p))
        loss = -(w * (surrogate + 0.002 * entropy)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(delta.parameters(), 0.5)
        before = {k: v.detach().clone() for k, v in delta.state_dict().items()}
        optimizer.step()
        with torch.no_grad():
            np1 = (old_log + delta(feats).flatten()).sigmoid().clamp(0.05, 0.95)
            kl = (
                w * (p0 * (p0 / np1).log() + (1 - p0) * ((1 - p0) / (1 - np1)).log())
            ).sum()
            if float(kl) > 0.035:
                delta.load_state_dict(before)
                break
        epochs += 1
    slope = basis @ (delta.weight.detach().numpy().ravel() / scale)
    new = old_head + np.r_[delta.bias.item() - center @ slope, slope]
    new = new.astype("<f4")
    final_logits = x @ new[1:] + new[0]
    np.testing.assert_allclose(
        final_logits,
        old_log.numpy() + delta(feats).detach().numpy().ravel(),
        rtol=0.0001,
        atol=0.0001,
    )
    new_head = old[: 8 + 193 * 4] + new.tobytes()
    deployed = phase_checkpoint(
        opener, struct.pack("<2i", POLICY_GRADIENT_MAGIC, len(leaf)) + leaf + new_head
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output.name + "-", dir=output.parent
    ) as work:
        staging = Path(work) / "candidate"
        staging.mkdir()
        (staging / "world_iql.bin").write_bytes(deployed)
        (staging / "sampler.bin").write_bytes(new_head)
        torch.save(
            {
                "risk": torch.tensor(risk),
                "selector": torch.from_numpy(new.copy()),
                "critic": torch.tensor(critic),
                "basis": torch.tensor(basis),
                "center": torch.tensor(center),
                "scale": torch.tensor(scale),
            },
            staging / "learner.pt",
        )
        report = {
            "method": "Real on-policy PPO over native action options with a fight-cross-fitted critic, elapsed-time GAE and frozen world features.",
            "games": len(games),
            "decisions": len(x),
            "accepted_ppo_epochs": epochs,
            "first_hit_cost": clean_cost,
            "episodes": outcomes,
            "rollout_games": len(outcomes),
            "changes_to_greedy_option": int(
                np.sum((final_logits > 0) != (old_log.numpy() > 0))
            ),
            "base_sha256": sha(parent),
            "sampler_input_sha256": digest,
            "sampler_sha256": sha(staging / "sampler.bin"),
            "policy_sha256": sha(staging / "world_iql.bin"),
            "rollout_sha256": sha(data),
            "projection_sha256": world_doc["projection_sha256"],
        }
        (staging / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "parent", "learner", "sampler", "action-world", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--clean-cost", type=float, default=0.0)
    train(**vars(parser.parse_args()))

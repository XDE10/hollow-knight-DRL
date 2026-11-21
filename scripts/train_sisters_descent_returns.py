"""Train native keys from randomized short horizontal maneuvers and measured arena rewards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from sisters_online_choices import choice_outcomes, late_catalogue
from torch import Tensor
from torch.nn import functional as F
from train_sisters_native_iql import REWARD_WEIGHTS
from train_sisters_timing_iql import (
    TimingPolicy,
    mlp,
    motion_history,
    phase_checkpoint,
    phase_components,
    save_native,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_logits(scores: Tensor, moves: Tensor, requested: Tensor) -> Tensor:
    """Use the best primitive in each direction, as the deployed greedy actor does."""
    member = moves[None, None, :] == requested[:, :, None]
    return scores[:, None, :].masked_fill(~member, -torch.inf).amax(-1)


def fit(
    archive: Path, parent: Path, deployed: Path, output: Path, updates: int, seed: int
) -> None:
    if updates < 1 or output.exists():
        raise ValueError("Use a positive budget and a new output directory")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prev = torch.load(parent, weights_only=True, map_location="cpu")
    native = deployed.read_bytes()
    cat = prev["catalogue"].long()
    if not np.array_equal(late_catalogue(native), cat.numpy()):
        raise ValueError("Parent action catalogue differs from its deployed weights")
    ref = TimingPolicy(cat, history=True).to(device).eval().requires_grad_(False)
    ref.load_state_dict(prev["actor"])
    val = (
        mlp(prev["value"]["0.weight"].shape[1], 1)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    val.load_state_dict(prev["value"])
    moves = cat[:, 0].to(device)
    lookup = {tuple(key): i for i, key in enumerate(cat.tolist())}
    parts = {"train": [], "holdout": []}
    anchors = {"train": [], "holdout": []}
    counts = {}
    with zipfile.ZipFile(archive) as z, torch.inference_mode():
        manifest = json.loads(z.read("manifest.json"))
        if len({g["source"] for g in manifest}) != len(manifest):
            raise ValueError("Duplicate development fight")
        for game in manifest:
            raw = z.read(game["file"])
            if (
                sha(deployed) != game.get("behavior_policy_sha256")
                or game.get("teacher_labels", True)
                or hashlib.sha256(raw).hexdigest() != game["sha256"]
                or not game["source"].startswith("native-development/")
            ):
                raise ValueError(
                    "Expected this frozen neural actor and genuine development data"
                )
            with np.load(io.BytesIO(raw), allow_pickle=False) as d:
                t, s, keys = d["control_time"], d["timing_state"], d["controls"]
                starts, stops = d["choice_rows"], d["choice_hold_stop"]
                pairs, arms, probability = (
                    d["choice_pairs"],
                    d["choice_draw"],
                    d["choice_probability"],
                )
                if not (
                    len(starts) == len(stops) == len(pairs) == len(arms)
                ) or not np.all(probability == 0.5):
                    raise ValueError("Invalid randomized directions")
                if (np.diff(t[starts]) < 1.499).any() or not np.array_equal(
                    stops, np.searchsorted(t, t[starts] + np.float32(0.16))
                ):
                    raise ValueError(
                        "Changed duration or cooldown of a direction intervention"
                    )
                kept, payout, next_row, discount = choice_outcomes(
                    t,
                    d["time"],
                    d["elapsed_ticks"],
                    d["events"],
                    starts,
                    REWARD_WEIGHTS,
                    horizon=0.6,
                )
                feat = torch.as_tensor(motion_history(t, s), device=device)
                hidden = torch.cat([ref.encode(v) for v in feat.split(512)])
                original = ref.heads(hidden)[0]
                start = starts[kept]
                v0 = val(feat[start]).flatten()
                future = val(feat[next_row]).flatten()
                reward = (
                    torch.as_tensor(payout, device=device)
                    + torch.as_tensor(discount, device=device) * future
                    - v0
                )
                measured = reward.clamp(-6, 6).cpu().numpy()
                occupied = np.zeros(len(s), bool)
                for start_row, end in zip(starts, stops, strict=True):
                    occupied[start_row:end] = True
                neutral = np.flatnonzero((s[:, 23] > 0.5) & ~occupied)[::8]
                split, source = game["split"], game["source"]
                if len(neutral):
                    anchors[split].append(hidden[neutral].cpu().numpy().copy())
                for j, index in enumerate(kept):
                    row, end = int(starts[index]), int(stops[index])
                    # Exclude discontinuities in an intended 0.16-second movement.
                    if (
                        end <= row
                        or end >= len(t)
                        or np.diff(t[row : end + 1]).max() > 0.081
                    ):
                        continue
                    pair = pairs[index]
                    arm = int(arms[index])
                    actions = keys[row:end]
                    alt = cat[pair, 0].numpy()
                    if (
                        alt[0] == alt[1]
                        or not np.array_equal(actions[0], cat[pair[arm]].numpy())
                        or not np.all(actions[:, 0] == alt[arm])
                    ):
                        raise ValueError(
                            "Recorded movement differs from the randomized arm"
                        )
                    raw_logits = original[row]
                    if (
                        raw_logits.max() - raw_logits[pair[0]] > 0.003
                        or abs(
                            float(raw_logits[pair[0]] - raw_logits[pair[1]])
                            - float(d["choice_gap"][index])
                        )
                        > 0.003
                    ):
                        raise ValueError(
                            "Exported neural action scores differ from the arena"
                        )
                    ids = np.array([lookup[tuple(a)] for a in actions], dtype=np.int64)
                    allowed = moves[None, :] == int(alt[arm])
                    expected = (
                        original[row:end]
                        .masked_fill(~allowed, -torch.inf)
                        .max(1)
                        .values
                    )
                    if (
                        expected
                        - original[row:end]
                        .gather(1, torch.as_tensor(ids, device=device)[:, None])
                        .flatten()
                        > 0.003
                    ).any():
                        raise ValueError(
                            "Secondary keys did not follow the frozen network"
                        )
                    parts[split].append(
                        (
                            source,
                            hidden[row:end].cpu().numpy().copy(),
                            ids,
                            alt.copy(),
                            arm,
                            float(measured[j]),
                        )
                    )
                counts[source] = sum(x[0] == source for x in parts[split])
    if min(map(len, parts.values())) < 2 or not all(anchors.values()):
        raise ValueError(
            "Need complete randomized maneuvers and ordinary states in both splits"
        )

    actor = copy.deepcopy(ref)
    actor.out.requires_grad_(True)
    opt = torch.optim.AdamW(actor.out.parameters(), lr=7e-5)
    rng = np.random.default_rng(seed)
    games = sorted({b[0] for b in parts["train"]})
    by_fight = [[b for b in parts["train"] if b[0] == g] for g in games]
    anchor = torch.as_tensor(np.concatenate(anchors["train"]), device=device)
    payoffs = np.array([b[-1] for b in parts["train"]])
    center = float(np.median(payoffs))
    scale = max(0.25, float(np.std(payoffs)))
    for _ in range(updates):
        ids = rng.integers(len(games), size=96)
        blocks = [by_fight[i][rng.integers(len(by_fight[i]))] for i in ids]
        onset = torch.as_tensor(np.stack([b[1][0] for b in blocks]), device=device)
        directions = torch.as_tensor(np.stack([b[3] for b in blocks]), device=device)
        arms = torch.tensor([b[4] for b in blocks], device=device)
        credit = torch.tensor(
            [(b[-1] - center) / scale for b in blocks], device=device
        ).clamp(-3, 3)
        scores = actor.heads(onset)[0]
        probability = (
            direction_logits(scores, moves, directions)
            .softmax(1)
            .gather(1, arms[:, None])
            .flatten()
        )
        # A fair coin chose each observed maneuver. Its return is measured in real combat.
        policy_loss = -(2 * probability * credit).mean()
        j = [rng.integers(len(b[2])) for b in blocks]
        h = torch.as_tensor(
            np.stack([b[1][k] for b, k in zip(blocks, j, strict=True)]), device=device
        )
        a = torch.tensor(
            [b[2][k] for b, k in zip(blocks, j, strict=True)], device=device
        )
        weight = credit.clamp(-1, 1).exp()
        imitation = (
            F.cross_entropy(actor.heads(h)[0], a, reduction="none") * weight
        ).sum() / weight.sum()
        ordinary = anchor[torch.randint(len(anchor), (192,), device=device)]
        old = ref.heads(ordinary)[0].softmax(1)
        kl = F.kl_div(
            actor.heads(ordinary)[0].log_softmax(1), old, reduction="batchmean"
        )
        loss = policy_loss + 0.15 * imitation + 2 * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.out.parameters(), 1.0)
        opt.step()
    outcomes = {}
    with torch.inference_mode():
        held = parts["holdout"]
        for source in sorted({b[0] for b in held}):
            trials = [b for b in held if b[0] == source]
            feat = torch.as_tensor(np.stack([b[1][0] for b in trials]), device=device)
            chosen = moves[actor.heads(feat)[0].argmax(1)].cpu().numpy()
            pair = np.stack([b[3] for b in trials])
            assigned = np.array([b[4] for b in trials])
            pay = np.array([b[-1] for b in trials])
            other = chosen == pair[:, 1]
            supported = other | (chosen == pair[:, 0])
            gain = 2 * (2 * assigned - 1) * pay * other
            # The option experiment provides no counterfactual for a third direction.
            gain[~supported] = -12.0
            outcomes[source] = {
                "trials": len(trials),
                "other_direction": int(other.sum()),
                "unsupported": int((~supported).sum()),
                "estimated_option_gain": float(gain.mean()),
            }
        h = torch.as_tensor(np.concatenate(anchors["holdout"]), device=device)
        before = moves[ref.heads(h)[0].argmax(1)]
        after = moves[actor.heads(h)[0].argmax(1)]
        drift = float((after != before).float().mean())
    gains = [v["estimated_option_gain"] for v in outcomes.values()]
    eligible = (
        len(gains) >= 3
        and np.mean(gains) > 0
        and sum(g > 0 for g in gains) > len(gains) / 2
        and drift <= 0.05
    )
    output.mkdir(parents=True, exist_ok=False)
    model = output / "world_iql.bin"
    save_native(actor, cat.numpy(), model)
    model.write_bytes(phase_checkpoint(phase_components(native)[0], model.read_bytes()))
    prev["actor"] = {k: v.cpu() for k, v in actor.state_dict().items()}
    torch.save(prev, output / "learner.pt")
    report = {
        "method": "inverse-propensity neural policy update on actual randomized maneuvers with world-assisted IQL continuation",
        "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "seed": seed,
        "updates": updates,
        "trials": counts,
        "per_held_fight": outcomes,
        "held_anchor_direction_drift": drift,
        "trial_eligible": bool(eligible),
        "scope": "Held-out estimates compare the two randomized movements with their 0.16-second commitment. The exported raw-key network has no forced hold and requires independent full-game evaluation.",
        "parent_sha256": sha(deployed),
        "data_sha256": sha(archive),
        "policy_sha256": sha(model),
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "parent", "deployed", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--updates", type=int, default=350)
    p.add_argument("--seed", type=int, default=9761)
    args = p.parse_args()
    fit(args.archive, args.parent, args.deployed, args.output, args.updates, args.seed)

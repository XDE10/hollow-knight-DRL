"""Fit a neural trio recovery policy from actions actually taken in randomized teacher blocks."""

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
from sisters_control_blocks import control_blocks
from torch.nn import functional as F
from train_sisters_native_iql import REWARD_WEIGHTS
from train_sisters_timing_iql import (
    TimingPolicy,
    mlp,
    motion_history,
    movement_groups,
    phase_checkpoint,
    phase_components,
    save_native,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(
    archive: Path, parent: Path, deployed: Path, output: Path, updates: int, seed: int
) -> None:
    if updates < 1 or output.exists():
        raise ValueError("Use positive update count and a new output directory")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    saved = torch.load(parent, weights_only=True, map_location="cpu")
    cat = saved["catalogue"].cpu()
    actor = TimingPolicy(cat, history=True).to(device)
    actor.load_state_dict(saved["actor"])
    reference = copy.deepcopy(actor).eval().requires_grad_(False)
    value = mlp(saved["value"]["0.weight"].shape[1], 1).to(device)
    value.load_state_dict(saved["value"])
    value.eval().requires_grad_(False)
    groups = movement_groups(cat).cpu().numpy()
    lookup = {tuple(row): i for i, row in enumerate(cat.tolist())}
    blocks: dict[str, list] = {"train": [], "holdout": []}
    anchors: dict[str, list] = {"train": [], "holdout": []}
    coverage: dict[str, int] = {}
    with zipfile.ZipFile(archive) as z, torch.no_grad():
        spec = json.loads(z.read("experiment.json"))
        manifest = json.loads(z.read("manifest.json"))
        if (
            spec["actor_sha256"] != digest(deployed)
            or spec["learner_sha256"] != digest(parent)
            or spec["teacher_probability"] != 0.5
        ):
            raise ValueError(
                "Expected the original neural arm and fair controller draws"
            )
        if set(spec["train_seeds"]) & set(spec["heldout_seeds"]):
            raise ValueError("Overlapping fight splits")
        expected = {
            f"native-development/{n}": split
            for split, seeds in (
                ("train", spec["train_seeds"]),
                ("holdout", spec["heldout_seeds"]),
            )
            for n in seeds
        }
        if (
            len(manifest) != len(expected)
            or {g["source"]: g["split"] for g in manifest} != expected
        ):
            raise ValueError("The complete preassigned cohort is required")
        opening, _ = phase_components(deployed.read_bytes())
        for game in manifest:
            raw = z.read(game["file"])
            if hashlib.sha256(raw).hexdigest() != game["sha256"]:
                raise ValueError("Changed development replay")
            with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                s, t = f["timing_state"].astype("f4"), f["control_time"]
                x = motion_history(t, s)
                actual, teacher, student = (
                    f["controls"],
                    f["teacher_controls"],
                    f["student_controls"],
                )
                acted = f["teacher_acted"]
                if not np.array_equal(
                    actual, np.where(acted[:, None], teacher, student)
                ):
                    raise ValueError(
                        "The recorded controls do not match the assigned arm"
                    )
                drawn = control_blocks(
                    t,
                    acted,
                    s[:, 23] > 0.5,
                    f["time"],
                    f["elapsed_ticks"],
                    f["events"],
                    REWARD_WEIGHTS,
                )
                label_ids = np.array([lookup.get(tuple(row), -1) for row in teacher])
                student_ids = np.array([lookup.get(tuple(row), -1) for row in student])
                split = game["split"]
                for block in drawn:
                    i, k = block.start, block.stop
                    if np.any(np.diff(t[i : min(k + 1, len(t))]) > 0.081):
                        continue
                    rows = np.arange(i, k)
                    # Agreement states remain anchors, regardless of block reward.
                    same = rows[
                        (student[rows] == teacher[rows]).all(1) & (s[rows, 9] < 0.5)
                    ]
                    if len(same):
                        anchors[split].append((game["source"], x[same[::4]].copy()))
                    if not block.teacher:
                        continue
                    rows = rows[
                        (label_ids[rows] >= 0)
                        & (student_ids[rows] >= 0)
                        & (s[rows, 9] < 0.5)
                    ]
                    rows = rows[groups[label_ids[rows]] != groups[student_ids[rows]]]
                    if not len(rows):
                        continue
                    successor = min(k, len(t) - 1)
                    v = (
                        value(torch.as_tensor(x[[i, successor]], device=device))
                        .flatten()
                        .cpu()
                        .numpy()
                    )
                    advantage = block.reward + block.discount * v[1] - v[0]
                    blocks[split].append(
                        (
                            game["source"],
                            x[rows].copy(),
                            label_ids[rows],
                            float(advantage),
                        )
                    )
                coverage[game["source"]] = sum(
                    g[0] == game["source"] for g in blocks[split]
                )
    if not blocks["train"] or not anchors["train"]:
        raise ValueError(
            "Training requires recorded correction blocks and agreement states"
        )
    # Each teacher-controlled block is a unit. AWR gives more mass to sequences
    # whose measured reward and frozen continuation exceed the starting value.
    train_blocks = blocks["train"]
    advantage = np.array([b[3] for b in train_blocks])
    mass = np.exp(
        np.clip(
            (advantage - np.median(advantage)) / max(advantage.std(), 0.25), -1.4, 1.4
        )
    )
    mass /= mass.sum()
    anchor = torch.as_tensor(
        np.concatenate([a for _, a in anchors["train"]]), device=device
    )
    actor.requires_grad_(False)
    for layer in (actor.dense[2], actor.out, actor.factors):
        layer.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (p for p in actor.parameters() if p.requires_grad), lr=2e-5
    )
    for step in range(updates):
        batch = [
            train_blocks[i] for i in rng.choice(len(train_blocks), size=128, p=mass)
        ]
        picks = [rng.integers(len(b[2])) for b in batch]
        x = torch.as_tensor(
            np.stack([b[1][j] for b, j in zip(batch, picks, strict=True)]),
            device=device,
        )
        y = torch.as_tensor(
            [b[2][j] for b, j in zip(batch, picks, strict=True)], device=device, dtype=torch.long
        )
        a = anchor[torch.randint(len(anchor), (256,), device=device)]
        logits = actor(x)[0]
        with torch.no_grad():
            base_x = reference(x)[0].softmax(1)
            base_a = reference(a)[0].softmax(1)
        loss = F.cross_entropy(logits, y) + F.kl_div(
            logits.log_softmax(1), base_x, reduction="batchmean"
        )
        loss += 4 * F.kl_div(actor(a)[0].log_softmax(1), base_a, reduction="batchmean")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
        optimizer.step()
    fights = {}
    with torch.no_grad():
        for source in (s for s, split in expected.items() if split == "holdout"):
            pairs = [b for b in blocks["holdout"] if b[0] == source]
            agreement_states = [a for s, a in anchors["holdout"] if s == source]
            if not pairs or not agreement_states:
                fights[source] = {
                    "correction_rows": 0,
                    "parent_matches": 0,
                    "candidate_matches": 0,
                    "anchor_rows": 0,
                    "anchor_motion_drift": None,
                }
                continue
            x = torch.as_tensor(np.concatenate([b[1] for b in pairs]), device=device)
            y = groups[np.concatenate([b[2] for b in pairs])]

            def predict(model: TimingPolicy, states: torch.Tensor) -> np.ndarray:
                return groups[
                    torch.cat([model(part)[0].argmax(1) for part in states.split(512)])
                    .cpu()
                    .numpy()
                ]

            old, new = predict(reference, x), predict(actor, x)
            agreed = torch.as_tensor(np.concatenate(agreement_states), device=device)
            fights[source] = {
                "correction_rows": len(y),
                "parent_matches": int((old == y).sum()),
                "candidate_matches": int((new == y).sum()),
                "anchor_rows": len(agreed),
                "anchor_motion_drift": float(
                    np.mean(predict(reference, agreed) != predict(actor, agreed))
                ),
            }
    output.mkdir(parents=True, exist_ok=False)
    policy = output / "world_iql.bin"
    save_native(actor, cat.numpy(), policy)
    policy.write_bytes(phase_checkpoint(opening, policy.read_bytes()))
    saved["actor"] = {k: v.cpu() for k, v in actor.state_dict().items()}
    torch.save(saved, output / "learner.pt")
    required = (3 * len(fights) + 3) // 4
    report = {
        "method": "return-weighted cloning of randomized, actually executed correction blocks",
        "eligibility": "neural-vs-neural development trial, not an arena win-rate estimate",
        "trial_eligible": sum(
            v["candidate_matches"] > v["parent_matches"] for v in fights.values()
        )
        >= required
        and all(
            v["anchor_motion_drift"] is not None and v["anchor_motion_drift"] <= 0.10
            for v in fights.values()
        ),
        "seed": seed,
        "updates": updates,
        "block_count": coverage,
        "heldout_fights": fights,
        "parent_sha256": digest(deployed),
        "replay_sha256": digest(archive),
        "policy_sha256": digest(policy),
        "learner_sha256": digest(output / "learner.pt"),
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "parent", "deployed", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--seed", type=int, default=3127)
    args = parser.parse_args()
    train(
        args.archive, args.parent, args.deployed, args.output, args.updates, args.seed
    )

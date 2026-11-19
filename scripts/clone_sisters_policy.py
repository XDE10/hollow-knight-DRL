"""Behavior cloning for the Sisters GRU using masked, episode-local demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hkrl import spaces
from hkrl.cli import _build_model
from hkrl.training.demonstration_replay import (
    DEMO_OBS,
    compact_obs,
    read_demonstration,
    split_demonstrations,
)
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.live_tuning import atomic_write_json
from hkrl.utils.seeding import seed_everything
from torch import Tensor

OBS = DEMO_OBS
ATTACK = 2 + spaces.BUTTON_BITS["attack"]


def load_demo(
    path: Path, *, survival: bool = False, human: bool = False
) -> dict[str, Tensor]:
    data = read_demonstration(path, path.read_bytes())
    actions = data["actions"]
    direction = actions[:, 0]
    turns = torch.zeros_like(direction, dtype=torch.bool)
    last = 1
    for i, label in enumerate(direction.tolist()):
        turns[i] = label != 1 and last != 1 and label != last
        if label != 1:
            last = label
    data["turns"] = turns
    data["weight"] = torch.ones(len(actions))
    record = json.loads(path.with_suffix(".json").read_text())
    if record["steps"] != len(actions):
        raise ValueError(f"demonstration metadata length mismatch: {path}")
    if human:
        if not (
            record.get("teacher") == "human"
            and record.get("qualified") is True
            and record.get("won") is True
            and record.get("truncated") is False
            and record.get("first_bosses") == 1
            and record["max_concurrent"]["bosses"] >= 2
            and "server_ticks" in data
            and "raw_keys" in data
            and (data["server_ticks"].diff() == 1).all()
        ):
            raise ValueError(f"expected a complete winning human demonstration: {path}")
        return data
    if survival:
        jumping = actions[:, 2 + spaces.BUTTON_BITS["jump_tap"]].bool()
        data["weight"] += 2 * jumping
        data["weight"] += turns
        return data
    for hit in record["hits"]:
        if hit["approached"]:
            end = hit["step"]
            data["weight"][max(0, hit["slash_step"] - 9) : end] = 2
    data["weight"] += 3 * turns
    return data


def padded_clips(
    episodes: list[dict[str, Tensor]], length: int, burn: int
) -> list[dict[str, Tensor]]:
    clips = []
    for ep in episodes:
        for start in range(0, len(ep["actions"]), length):
            lo = max(0, start - burn)
            hi = min(start + length, len(ep["actions"]))
            keep = hi - lo
            clip = {}
            for k in (*OBS, "actions", "masks", "weight"):
                data = ep[k][lo:hi]
                # Tail padding is excluded from the loss. Repeat a valid mask/action.
                clip[k] = torch.cat(
                    (data, data[-1:].expand((length + burn - keep, *data.shape[1:])))
                )
            clip["loss_mask"] = torch.arange(length + burn).ge(
                start - lo
            ) & torch.arange(length + burn).lt(keep)
            clips.append(clip)
    return clips


@torch.no_grad()
def validate(model: Any, episodes: list[dict[str, Tensor]]) -> dict[str, float]:
    model.eval()
    predictions = []
    labels = []
    turns = []
    likelihood = 0.0
    for ep in episodes:
        device = next(model.parameters()).device
        data = {k: v.unsqueeze(0).to(device) for k, v in ep.items()}
        dist, _, _ = model(compact_obs(data), action_mask=data["masks"])
        predictions.append(dist.mode()[0].cpu())
        labels.append(ep["actions"])
        turns.append(ep["turns"])
        likelihood -= float(dist.log_prob(data["actions"]).sum())
    pred = torch.cat(predictions)
    true = torch.cat(labels)
    turn = torch.cat(turns)
    p_attack = pred[:, ATTACK].bool()
    t_attack = true[:, ATTACK].bool()

    def recall(mask: Tensor) -> float:
        return (
            float((pred[mask, 0] == true[mask, 0]).float().mean())
            if mask.any()
            else 0.0
        )

    return {
        "left_recall": recall(true[:, 0] == 0),
        "right_recall": recall(true[:, 0] == 2),
        "turn_recall": recall(turn),
        "attack_f1": 2
        * float((p_attack & t_attack).sum())
        / max(int(p_attack.sum() + t_attack.sum()), 1),
        "jump_f1": jump_f1(pred, true),
        "nll": likelihood / len(true),
    }


def jump_f1(pred: Tensor, true: Tensor) -> float:
    column = 2 + spaces.BUTTON_BITS["jump_tap"]
    requested = pred[:, column].bool()
    target = true[:, column].bool()
    return (
        2
        * float((requested & target).sum())
        / max(int(requested.sum() + target.sum()), 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument(
        "--objective", choices=("attack", "survival", "human"), default="attack"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    human = args.objective == "human"
    survival = args.objective == "survival"
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    seed_everything(cfg.seed)
    torch.set_num_threads(1)
    train_files, val_files = split_demonstrations(args.demos)
    if not val_files or not train_files:
        raise ValueError("need separate complete training and validation episodes")
    summary: dict[str, Any] = {}
    for split, paths in (("train", train_files), ("validation", val_files)):
        hits = [
            h
            for p in paths
            for h in json.loads(p.with_suffix(".json").read_text())["hits"]
        ]
        counts = Counter(h["side"] for h in hits if h["approached"])
        turns = Counter(h["side"] for h in hits if h["turned_and_approached"])
        if (
            not human
            and not survival
            and min(turns["from_left"], turns["from_right"]) < 1
        ):
            raise ValueError(f"{split}: missing a verified turn and hit on one side")
        summary[split] = {
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
            "approaches": dict(counts),
            "turn_and_hit": dict(turns),
            "matched_hits": len(hits),
        }
    train = [load_demo(p, survival=survival, human=human) for p in train_files]
    val = [load_demo(p, survival=survival, human=human) for p in val_files]
    if survival:
        for split, episodes in (("train", train), ("validation", val)):
            jumps = torch.cat(
                [
                    ep["actions"][:, 2 + spaces.BUTTON_BITS["jump_tap"]]
                    for ep in episodes
                ]
            )
            if not (jumps.any() and (jumps == 0).any()):
                raise ValueError(
                    f"{split}: survival demonstrations must contain jumps and ground actions"
                )
            summary[split]["jump_requests"] = int(jumps.sum())
    dims = {k: tuple(train[0][k].shape[1:]) for k in OBS[:4]}
    model = _build_model(
        cfg,
        dims,
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    rng = np.random.default_rng(cfg.seed)
    clips = padded_clips(train, cfg.sequence_length, cfg.burn_in)
    best = -float("inf")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "bc-weights.pt"
    if target.exists():
        raise FileExistsError(target)
    report = {
        "objective": args.objective,
        "splits": summary,
        "training_steps": sum(len(ep["actions"]) for ep in train),
        "validation_steps": sum(len(ep["actions"]) for ep in val),
        "initial": validate(model, val),
        "seed": cfg.seed,
        "epochs": args.epochs,
    }
    print(json.dumps(report), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = rng.permutation(len(clips))
        for start in range(0, len(indices), 6):
            batch = [clips[int(i)] for i in indices[start : start + 6]]
            data = {
                k: torch.stack([row[k] for row in batch]).to(args.device)
                for k in batch[0]
            }
            obs = compact_obs(data)
            # Break the shortcut of repeating the teacher's previous direction.
            # The labels, observed geometry and directional supervision stay paired.
            if not human:
                prev = obs["prev_action"].clone()
                prev[..., 0].masked_fill_(
                    torch.rand(prev.shape[:2], device=prev.device) < 0.65, 1
                )
                obs["prev_action"] = prev
            dist, _, _ = model(obs, action_mask=data["masks"])
            actions = data["actions"]
            weights = data["weight"] * data["loss_mask"]
            logp = dist.log_prob(actions)
            if not human:
                focus_button = "jump_tap" if survival else "attack"
                button_logp = dist.buttons.log_prob(
                    actions[..., 2 : 2 + spaces.N_BUTTONS].float()
                )[..., spaces.BUTTON_BITS[focus_button]]
                logp = (
                    logp + 3 * dist.movement_x.log_prob(actions[..., 0]) + button_logp
                )
            loss = -(weights * logp).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = validate(model, val)
        score = (
            metrics["left_recall"]
            + metrics["right_recall"]
            + metrics["turn_recall"]
            + metrics["jump_f1" if survival else "attack_f1"]
        ) / 4
        if human:
            score = -metrics["nll"]
        print(json.dumps(dict(epoch=epoch, score=score, **metrics)), flush=True)
        if score > best:
            best = score
            temp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), temp)
            temp.replace(target)
            report["best"] = dict(epoch=epoch, score=score, **metrics)
    report["weights_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    atomic_write_json(args.output / "bc-report.json", report)


if __name__ == "__main__":
    main()

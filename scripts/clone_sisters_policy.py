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
from torch import Tensor

from hkrl import spaces
from hkrl.cli import _build_model
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.live_tuning import atomic_write_json
from hkrl.utils.seeding import seed_everything

OBS = ("global", "player", "entities", "entity_mask", "prev_action", "prev_reward")
ATTACK = 2 + spaces.BUTTON_BITS["attack"]


def load_demo(path: Path) -> dict[str, Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        data = {k: torch.from_numpy(archive[k]) for k in archive.files}
    actions = data["actions"].long()
    data["actions"] = actions
    data["prev_action"] = torch.cat(
        (torch.tensor([spaces.canonical_noop_action_values(enable_macro=True)]), actions[:-1])
    ).float()
    data["prev_reward"] = torch.cat((torch.zeros(1), data["rewards"][:-1])).float()[:, None]
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
    assert record["steps"] == len(actions)
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
            clip["loss_mask"] = torch.arange(length + burn).ge(start - lo) & torch.arange(
                length + burn
            ).lt(keep)
            clips.append(clip)
    return clips


def compact_obs(data: dict[str, Tensor]) -> dict[str, Tensor]:
    obs = {k: data[k] for k in OBS}
    # All slots removed here are padding in every example of this batch.
    valid = obs["entity_mask"].any(dim=(0, 1))
    if not valid.any():
        valid[0] = True
    obs["entities"] = obs["entities"][:, :, valid]
    obs["entity_mask"] = obs["entity_mask"][:, :, valid]
    return obs


@torch.no_grad()
def validate(model: Any, episodes: list[dict[str, Tensor]]) -> dict[str, float]:
    model.eval()
    predictions = []
    labels = []
    turns = []
    likelihood = 0.0
    for ep in episodes:
        data = {k: v.unsqueeze(0) for k, v in ep.items()}
        dist, _, _ = model(compact_obs(data), action_mask=data["masks"])
        predictions.append(dist.mode()[0])
        labels.append(ep["actions"])
        turns.append(ep["turns"])
        likelihood -= float(dist.log_prob(data["actions"]).sum())
    pred = torch.cat(predictions)
    true = torch.cat(labels)
    turn = torch.cat(turns)
    p_attack = pred[:, ATTACK].bool()
    t_attack = true[:, ATTACK].bool()

    def recall(mask: Tensor) -> float:
        return float((pred[mask, 0] == true[mask, 0]).float().mean()) if mask.any() else 0.0

    return dict(
        left_recall=recall(true[:, 0] == 0),
        right_recall=recall(true[:, 0] == 2),
        turn_recall=recall(turn),
        attack_f1=2
        * float((p_attack & t_attack).sum())
        / max(int(p_attack.sum() + t_attack.sum()), 1),
        nll=likelihood / len(true),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=36)
    args = parser.parse_args()
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)
    seed_everything(cfg.seed)
    torch.set_num_threads(1)
    files = sorted(args.demos.glob("episode-*.npz"))
    train_files = [p for i, p in enumerate(files) if i % 4 != 3]
    val_files = [p for i, p in enumerate(files) if i % 4 == 3]
    if not val_files or not train_files:
        raise ValueError("need separate complete training and validation episodes")
    summary: dict[str, Any] = {}
    for split, paths in (("train", train_files), ("validation", val_files)):
        hits = [h for p in paths for h in json.loads(p.with_suffix(".json").read_text())["hits"]]
        counts = Counter(h["side"] for h in hits if h["approached"])
        turns = Counter(h["side"] for h in hits if h["turned_and_approached"])
        if min(turns["from_left"], turns["from_right"]) < 1:
            raise ValueError(f"{split}: missing a verified turn and hit on one side")
        summary[split] = dict(
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            approaches=dict(counts),
            turn_and_hit=dict(turns),
            matched_hits=len(hits),
        )
    train = [load_demo(p) for p in train_files]
    val = [load_demo(p) for p in val_files]
    dims = {k: tuple(train[0][k].shape[1:]) for k in OBS[:4]}
    model = _build_model(
        cfg,
        dims,
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
    rng = np.random.default_rng(cfg.seed)
    clips = padded_clips(train, cfg.sequence_length, cfg.burn_in)
    best = -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "bc-weights.pt"
    if target.exists():
        raise FileExistsError(target)
    report = dict(
        splits=summary,
        training_steps=sum(len(ep["actions"]) for ep in train),
        validation_steps=sum(len(ep["actions"]) for ep in val),
        initial=validate(model, val),
        seed=cfg.seed,
        epochs=args.epochs,
    )
    print(json.dumps(report), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = rng.permutation(len(clips))
        for start in range(0, len(indices), 6):
            batch = [clips[int(i)] for i in indices[start : start + 6]]
            data = {k: torch.stack([row[k] for row in batch]) for k in batch[0]}
            obs = compact_obs(data)
            # Break the shortcut of repeating the teacher's previous direction.
            # The labels, observed geometry and directional supervision stay paired.
            prev = obs["prev_action"].clone()
            prev[..., 0].masked_fill_(torch.rand(prev.shape[:2]) < 0.65, 1)
            obs["prev_action"] = prev
            dist, _, _ = model(obs, action_mask=data["masks"])
            actions = data["actions"]
            weights = data["weight"] * data["loss_mask"]
            attacks = dist.buttons.log_prob(actions[..., 2 : 2 + spaces.N_BUTTONS].float())[
                ..., spaces.BUTTON_BITS["attack"]
            ]
            logp = dist.log_prob(actions) + 3 * dist.movement_x.log_prob(actions[..., 0]) + attacks
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
            + metrics["attack_f1"]
        ) / 4
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

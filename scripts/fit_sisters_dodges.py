"""Warm-start a Sisters checkpoint on verified evasions before continued local PPO."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hkrl import spaces
from hkrl.cli import _build_model
from hkrl.eval.sisters_survival import SistersBodyCounterPolicy
from hkrl.learner.checkpoint_payload import (
    validate_checkpoint_payload,
    validate_model_state_dict,
)
from hkrl.training.demonstration_replay import (
    DemonstrationReplay,
    compact_obs,
    read_demonstration,
    split_demonstrations,
)
from hkrl.training.sisters_dodges import (
    DODGE_BUTTONS,
    dodge_outcomes,
    landed_counter_labels,
    safe_nail_windows,
)
from hkrl.utils.config import load_task_config, load_train_config
from hkrl.utils.live_tuning import atomic_write_json
from hkrl.utils.seeding import seed_everything
from torch import Tensor


def prepare(
    source: Path, target: Path, *, landed_counters: bool = False
) -> dict[str, Any]:
    training, heldout = split_demonstrations(source)
    if not heldout:
        raise ValueError("need whole training and held-out episodes")
    summary: dict[str, Any] = {}
    target.mkdir(parents=True, exist_ok=True)
    for path in (*training, *heldout):
        payload = path.read_bytes()
        metadata = path.with_suffix(".json").read_bytes()
        record = json.loads(metadata)
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        grades = dodge_outcomes(arrays, record)
        if landed_counters:
            arrays["safe_nail"], arrays["landed_side"] = landed_counter_labels(
                arrays, record, SistersBodyCounterPolicy.LANDED
            )
        elif record.get("teacher") == "body_counter":
            arrays["safe_nail"] = safe_nail_windows(arrays, record)
        dest = target / path.name
        with dest.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f, **arrays, dodge_outcomes=grades)
        dest.with_suffix(".tmp").replace(dest)
        summary[path.name] = {
            "split": "train" if path in training else "heldout",
            "observations_sha256": hashlib.sha256(payload).hexdigest(),
            "events_sha256": hashlib.sha256(metadata).hexdigest(),
            "hits": len(record["hits"]),
            **(
                {"landed_slashes": int(np.count_nonzero(arrays["landed_side"]))}
                if landed_counters
                else {}
            ),
            "safe_nail_steps": int(arrays["safe_nail"].sum())
            if "safe_nail" in arrays
            else None,
            "survival_ticks": record["elapsed_ticks"],
            "spike_hp": sum(
                c["amount"] for c in record["contacts"] if c["source"] == "SPIKES"
            ),
            "successful_dodges": (grades > 0).sum(0).tolist(),
            "followed_by_hit": (grades == 2).sum(0).tolist(),
        }
    if landed_counters:
        for split in ("train", "heldout"):
            if not any(
                r["landed_slashes"] for r in summary.values() if r["split"] == split
            ):
                raise ValueError(
                    f"no verified grounded side counters in {split} episodes"
                )
    return summary


@torch.no_grad()
def validate(model: Any, episodes: list[dict[str, Tensor]]) -> dict[str, Any]:
    model.eval()
    predictions = []
    for ep in episodes:
        batch = {k: v.unsqueeze(0) for k, v in ep.items()}
        dist, _, _ = model(compact_obs(batch), action_mask=batch["masks"])
        predictions.append(dist.mode()[0])
    pred = torch.cat(predictions)
    actions, grades = (
        torch.cat([ep[k] for ep in episodes]) for k in ("actions", "dodge_outcomes")
    )
    metrics: dict[str, Any] = {
        "move_accuracy": float((pred[:, 0] == actions[:, 0]).float().mean())
    }
    if "landed_side" in episodes[0]:
        sides = torch.cat([ep["landed_side"] for ep in episodes])
        copied = (
            (pred[:, 0] == actions[:, 0])
            & (pred[:, 1] == 1)
            & pred[:, 2 + spaces.BUTTON_BITS["attack"]].bool()
        )
        metrics["landed_counter"] = {
            name: {
                "decisions": int((sides == side).sum()),
                "matched": int((copied & (sides == side)).sum()),
            }
            for name, side in (("target_left", -1), ("target_right", 1))
        }
    for name in ("attack", *DODGE_BUTTONS):
        bit = 2 + spaces.BUTTON_BITS[name]
        known = torch.ones_like(actions[:, bit], dtype=torch.bool)
        truth = actions[:, bit].bool()
        if name in DODGE_BUTTONS:
            grade = grades[:, DODGE_BUTTONS.index(name)]
            truth, known = grade > 0, grade >= 0
        predicted = pred[:, bit].bool() & known
        matched = int((truth & predicted).sum())
        metrics[name] = {
            "positive": int(truth.sum()),
            "predicted": int(predicted.sum()),
            "matched": matched,
            "f1": 2 * matched / max(1, int(truth.sum() + predicted.sum())),
        }
    metrics["score"] = (
        metrics["move_accuracy"]
        + sum(metrics[k]["f1"] for k in ("attack", *DODGE_BUTTONS))
    ) / 4
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "task", "demos", "output", "init-weights"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=320)
    parser.add_argument(
        "--landed-counters",
        action="store_true",
        help="Focus counter replay on grounded horizontal nail hits against the landed sister.",
    )
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        work = Path(staging) / "result"
        work.mkdir()
        fit(args, work)
        work.rename(args.output)


def fit(args: argparse.Namespace, work: Path) -> None:
    cfg, task = load_train_config(args.config), load_task_config(args.task)
    seed_everything(cfg.seed)
    torch.set_num_threads(1)
    target = work / "demos"
    manifest = prepare(args.demos, target, landed_counters=args.landed_counters)
    replay = DemonstrationReplay(target, cfg.sequence_length, cfg.burn_in)
    holdouts = [
        read_demonstration(p, p.read_bytes()) for p in split_demonstrations(target)[1]
    ]
    model = _build_model(
        cfg,
        {
            k: tuple(holdouts[0][k].shape[1:])
            for k in ("global", "player", "entities", "entity_mask")
        },
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    payload = args.init_weights.read_bytes()
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    model.load_state_dict(
        validate_model_state_dict(
            validate_checkpoint_payload(state)["model_state_dict"]
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00022)
    rng = np.random.default_rng(cfg.seed)
    initial = validate(model, holdouts)
    best = initial
    weights = work / "bc-weights.pt"
    torch.save(model.state_dict(), weights)
    report = {
        "episodes": manifest,
        "initial": initial,
        "best": best,
        "best_update": 0,
        "initial_sha256": hashlib.sha256(payload).hexdigest(),
        "updates": args.updates,
        "landed_counters": args.landed_counters,
    }
    for step in range(1, args.updates + 1):
        model.train()
        loss = replay.loss(model, rng, torch.device("cpu"))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0 or step == args.updates:
            metrics = validate(model, holdouts)
            print(json.dumps({"update": step, **metrics}), flush=True)
            if metrics["score"] > best["score"]:
                best = metrics
                tmp = weights.with_suffix(".tmp")
                torch.save(model.state_dict(), tmp)
                tmp.replace(weights)
                report.update(best=best, best_update=step)
    report["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    atomic_write_json(work / "report.json", report)


if __name__ == "__main__":
    main()

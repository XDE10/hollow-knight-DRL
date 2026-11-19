"""One idempotent Double DQN update on accumulated, real online Sisters transitions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from hkrl.models import sisters_ddqn  # noqa: F401
from hkrl.utils.registry import get

FEATURES = ("global", "player", "entities", "entity_mask")


def targets(
    reward: Tensor,
    terminal: Tensor,
    discount: Tensor,
    online_q: Tensor,
    target_q: Tensor,
    valid: Tensor,
) -> Tensor:
    """Select legal next actions with the online net and value them with the target net."""
    if not torch.all(valid.any(1) | terminal):
        raise ValueError("nonterminal state without any legal next action")
    choice = online_q.masked_fill(~valid, -torch.inf).argmax(1, keepdim=True)
    bootstrap = target_q.gather(1, choice).squeeze(1)
    return reward + torch.where(terminal, 0.0, discount * bootstrap)


def replay(root: Path, number: int, chunk: int, device: str) -> dict[str, Tensor]:
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    limit = 1
    for i in range(1, number + 1):
        with np.load(root / f"chunk-{i:02d}.npz", allow_pickle=False) as batch:
            action = batch["a"]
            if len(action) != chunk or not batch["mask"][np.arange(len(action)), action].all():
                raise ValueError("incomplete shard or illegal historical action")
            for prefix in ("s_", "n_"):
                mask = batch[prefix + "entity_mask"].astype(bool)
                columns = np.flatnonzero(mask.any(0))
                if columns.size:
                    limit = max(limit, int(columns[-1]) + 1)
                for key in FEATURES:
                    arrays[prefix + key].append(batch[prefix + key])
            for key in ("a", "r", "term", "ticks", "next_mask"):
                arrays[key].append(batch[key])
    result = {}
    for key, entries in arrays.items():
        if key.endswith(("entities", "entity_mask")):
            entries = [a[:, :limit] for a in entries]
        result[key] = torch.as_tensor(np.concatenate(entries), device=device)
    return result


def atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(root: Path, settings: Path, number: int, device: str) -> None:
    cfg = yaml.safe_load(settings.read_text())
    count = cfg["chunk"] * number
    if not 1 <= number <= cfg["total_steps"] // cfg["chunk"]:
        raise ValueError("round outside training budget")
    torch.manual_seed(cfg["seed"])
    model = get("model", cfg["model"])(cfg["hidden"]).to(device)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    full = root / "learner.pt"
    previous = 0
    updates = 0
    if full.exists():
        ck = torch.load(full, map_location=device, weights_only=True)
        model.load_state_dict(ck["online"])
        target.load_state_dict(ck["target"])
        optimizer.load_state_dict(ck["optimizer"])
        previous = ck["steps"]
        updates = ck["updates"]
        if ck["settings"] != cfg:
            raise ValueError("learner config changed")
    if previous not in (count - cfg["chunk"], count):
        raise ValueError("non-contiguous learner round")
    if previous < count:
        data = replay(root, number, cfg["chunk"], device)
        rng = torch.Generator(device=device).manual_seed(cfg["seed"] + number)
        losses = []
        for _ in range(cfg["updates_per_chunk"]):
            ix = torch.randint(count, (cfg["batch"],), generator=rng, device=device)
            current = {k: data["s_" + k][ix] for k in FEATURES}
            after = {k: data["n_" + k][ix] for k in FEATURES}
            q = model(current).gather(1, data["a"][ix, None].long()).squeeze(1)
            with torch.no_grad():
                discount = cfg["gamma"] ** (data["ticks"][ix].float() / cfg["discount_ticks"])
                y = targets(
                    data["r"][ix].float() / cfg["reward_scale"],
                    data["term"][ix].bool(),
                    discount,
                    model(after),
                    target(after),
                    data["next_mask"][ix].bool(),
                )
            loss = F.smooth_l1_loss(q, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if updates % cfg["target_interval"] == 0:
                target.load_state_dict(model.state_dict())
            losses.append(loss.detach())
        avg = torch.stack(losses).mean().item()
        meta = dict(
            steps=count,
            updates=updates,
            loss=avg,
            device=str(device),
            hardware=torch.cuda.get_device_name() if device.startswith("cuda") else "CPU",
        )
        atomic_save(
            dict(
                steps=count,
                updates=updates,
                settings=cfg,
                online=model.state_dict(),
                target=target.state_dict(),
                optimizer=optimizer.state_dict(),
                meta=meta,
            ),
            full,
        )
    else:
        meta = ck["meta"]
    policy = root / f"policy-{number:02d}.pt"
    atomic_save(
        dict(steps=count, weights={k: v.cpu() for k, v in model.state_dict().items()}),
        policy,
    )
    doc = dict(**meta, policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
    out = root / f"policy-{number:02d}.zip"
    tmp = out.with_suffix(".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(policy, policy.name)
        z.writestr("metrics.json", json.dumps(doc))
    tmp.replace(out)
    print(json.dumps(doc), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.root, args.settings, args.round, args.device)

"""Improve a frozen IQL actor's downward spell requests with observed combat returns."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_sisters_native_iql import REWARD_WEIGHTS, attribute_events
from train_sisters_timing_iql import (
    TimingPolicy,
    measured_returns,
    mlp,
    motion_history,
    phase_checkpoint,
    phase_components,
    save_native,
)


def spell_examples(
    t: np.ndarray, state: np.ndarray, keys: np.ndarray, teacher: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select debit-supported downward starts and unambiguous teacher abstentions."""
    active = (keys[:, 2] & 16) != 0
    previous_active = state[:, 168] > 0.5  # previously committed quickCast bit
    focus = (keys[:, 2] & 32) != 0
    live = (state[:, 23] > 0.5) & teacher
    segment = np.r_[0, np.cumsum((np.diff(t) > 0.081) | ~live[1:] | ~live[:-1])]
    ready = (state[:, 11] > 0.5) & (state[:, 5] * 99 >= 32.99)
    eligible = live & ready & ~previous_active
    ends = np.searchsorted(t.astype("f8"), t.astype("f8") + 0.4001, side="right") - 1
    complete = (segment[ends] == segment) & (t[ends] - t >= 0.319)
    starts = active & ~previous_active
    up_to = lambda x: np.r_[0, np.cumsum(x)]
    count_cast, count_focus = up_to(starts), up_to(focus)
    soul = np.rint(state[:, 5] * 99)
    count_debit = up_to(np.r_[False, np.diff(soul) <= -22])
    idx = np.arange(len(t))
    clear = (count_focus[ends + 1] == count_focus[idx]) & complete & eligible
    no_later_cast = count_cast[ends + 1] == count_cast[idx + 1]
    positive = (
        clear
        & starts
        & (keys[:, 1] == -1)
        & no_later_cast
        & (count_debit[ends + 1] > count_debit[idx + 1])
    )
    # Samples inside a recently used spell can have stale locks or costs. Do not
    # label buffered requests or a teacher spell due in the next 0.40 s as bad.
    left = np.searchsorted(t, t - 0.401)
    negative = (
        clear & (segment[left] == segment) & (count_cast[ends + 1] == count_cast[left])
    )
    return positive, negative, eligible


def fold_spell_correction(
    actor: TimingPolicy, correction: nn.Linear, group: torch.Tensor
) -> None:
    """Add the same state-dependent linear score to just the downward cast catalogue."""
    with torch.no_grad():
        actor.out.weight[group] += correction.weight
        actor.out.bias[group] += correction.bias


def fit(
    archive: Path, parent: Path, deployed: Path, output: Path, updates: int, seed: int
) -> None:
    if updates < 1 or output.exists():
        raise ValueError("Expected a positive budget and unused output directory")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(parent, map_location="cpu", weights_only=True)
    cat = saved["catalogue"]
    actor = TimingPolicy(cat, history=True).to(device).eval().requires_grad_(False)
    actor.load_state_dict(saved["actor"])
    opening, late = phase_components(deployed.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "native"
        save_native(actor, cat.numpy(), p)
        exported = p.read_bytes()
        head_bytes = (actor.out.in_features + 1) * len(cat) * 4
        if (
            len(exported) != len(late)
            or exported[:-head_bytes] != late[:-head_bytes]
            or not np.allclose(
                np.frombuffer(exported[-head_bytes:], "<f4"),
                np.frombuffer(late[-head_bytes:], "<f4"),
                rtol=1e-6,
                atol=1e-6,
            )
        ):
            raise ValueError("Parent actor and deployed three-Sister weights differ")
    value = (
        mlp(saved["value"]["0.weight"].shape[1], 1)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    value.load_state_dict(saved["value"])
    group = ((cat[:, 2] & 16) != 0) & (cat[:, 1] == -1)
    g = group.to(device)
    blocks: dict[str, list] = {"train": [], "holdout": []}
    with zipfile.ZipFile(archive) as z, torch.inference_mode():
        manifest = json.loads(z.read("manifest.json"))
        seen: set[str] = set()
        for game in manifest:
            ident = game["source"]
            if ident in seen or not ident.startswith("native-development/"):
                raise ValueError("Duplicate or non-development game")
            seen.add(ident)
            raw = z.read(game["file"])
            if hashlib.sha256(raw).hexdigest() != game["sha256"]:
                raise ValueError("Changed development example")
            with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                k, t, s = f["controls"], f["control_time"], f["timing_state"]
                teach = (
                    f["teacher_acted"]
                    if "teacher_acted" in f
                    else np.zeros(len(k), bool)
                )
                if (
                    teach.dtype != np.bool_
                    or teach.shape != t.shape
                    or (
                        teach.any()
                        and not np.array_equal(f["teacher_controls"][teach], k[teach])
                    )
                ):
                    raise ValueError("Unverified teacher controls")
                pos, neg, _ = spell_examples(t, s, k, teach)
                ev = attribute_events(
                    t,
                    f["time"],
                    f["elapsed_ticks"],
                    f["events"],
                    health_masks=9 * s[:, 4],
                )
                term = ev[:, 6:].any(1)
                dt = np.r_[np.diff(t), 0.04]
                valid = (dt <= 0.081) & (np.r_[np.ones(len(t) - 1, bool), False] | term)
                bounds = np.flatnonzero(teach[1:] != teach[:-1]) + 1
                if "block_starts" in f:
                    bounds = np.union1d(bounds, f["block_starts"])
                if "choice_rows" in f:
                    stops = (
                        f["choice_hold_stop"]
                        if "choice_hold_stop" in f
                        else f["choice_rows"] + 1
                    )
                    bounds = np.union1d(bounds, np.union1d(f["choice_rows"], stops))
                    bounds = bounds[bounds < len(t)]
                ret, cont, last = measured_returns(
                    t,
                    ev @ REWARD_WEIGHTS,
                    0.995 ** (dt / 0.04) * ~term,
                    valid,
                    0.8,
                    bounds,
                )
                features = torch.from_numpy(motion_history(t, s)).to(device)
                hidden = torch.cat([actor.encode(b) for b in features.split(512)])
                logits = actor.heads(hidden)[0]
                inside, outside = (
                    logits.masked_fill(~g, -torch.inf).max(1),
                    logits.masked_fill(g, -torch.inf).max(1),
                )
                margin = inside.values - outside.values
                v = (
                    torch.cat([value(b) for b in features.split(512)])
                    .flatten()
                    .cpu()
                    .numpy()
                )
                adv = ret + cont * v[np.minimum(last + 1, len(v) - 1)] - v
                phase = (s[:, 23] > 0.5) & valid
                trained = (k[:, 1] == -1) & ((k[:, 2] & 16) != 0)
                # Include every actual spell/teacher-start/hard abstention, and a
                # regular subsample of ordinary states for the KL and drift checks.
                ordinary = phase & (np.arange(len(t)) % 6 == 0)
                hard = neg & (margin.cpu().numpy() >= -1)
                keep = phase & (
                    ordinary
                    | pos
                    | hard
                    | trained
                    | (neg & (np.arange(len(t)) % 8 == 0))
                )
                if game["split"] == "holdout":
                    keep = phase
                    ordinary = phase
                rows = np.flatnonzero(keep)
                if not len(rows):
                    continue
                blocks[game["split"]].append(
                    (
                        ident,
                        hidden[rows].cpu(),
                        margin[rows].cpu(),
                        torch.from_numpy(
                            np.stack([pos, neg, trained, ordinary, hard], 1)[rows]
                        ),
                        torch.from_numpy(adv[rows].astype("f4")),
                        cat[inside.indices[rows].cpu()],
                        cat[outside.indices[rows].cpu()],
                        torch.from_numpy(k[rows]),
                    )
                )
    if not all(blocks.values()):
        raise ValueError("Missing training or held-out fights")
    _, h, m, marks, adv, _, _, _ = zip(*blocks["train"], strict=True)
    h, m, marks, adv = [torch.cat(v).to(device) for v in (h, m, marks, adv)]
    positives, _, actual, ordinary, hard = marks.T
    sets = [x.nonzero().flatten() for x in (positives, hard, ordinary)]
    if min(map(len, sets)) < 2:
        raise ValueError("Insufficient examples for a supported correction")
    fix = nn.Linear(h.shape[1], 1).to(device)
    nn.init.zeros_(fix.weight)
    nn.init.zeros_(fix.bias)
    opt = torch.optim.AdamW(fix.parameters(), lr=0.00015)
    for _ in range(updates):
        p, n, a = [v[torch.randint(len(v), (160,), device=device)] for v in sets]
        ids = torch.cat((p, n))
        delta = fix(h[ids]).flatten()
        weight = adv[ids].clamp(-1.5, 1.5).exp()
        imitation = F.binary_cross_entropy_with_logits(
            m[ids] + delta, positives[ids].float(), weight=weight
        )
        # Value-weighted improvement on actual native actions independently of
        # teacher labels. The frozen continuation comes from world-assisted IQL.
        observed = fix(h[a]).flatten()
        loss_rl = F.binary_cross_entropy_with_logits(
            m[a] + observed, actual[a].float(), weight=adv[a].clamp(-1.5, 1.5).exp()
        )
        prior = m[a].sigmoid()
        anchor = F.binary_cross_entropy_with_logits(m[a] + observed, prior)
        loss = (
            imitation
            + 0.35 * loss_rl
            + 2 * anchor
            + 0.06 * (delta.square().mean() + observed.square().mean())
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    per = []
    with torch.no_grad():
        for ident, h0, m0, marks0, _, inside, outside, key in blocks["holdout"]:
            p, n, _, a, _ = marks0.T
            before = m0 > 0
            after = (m0 + fix(h0.to(device)).flatten().cpu()) > 0
            compare = (inside[:, 0] == key[:, 0]) & (inside[:, 1] == key[:, 1])
            changed = before != after

            def nb(x: torch.Tensor) -> int:
                return int(x.sum())

            per.append(
                {
                    "source": ident,
                    "confirmed": nb(p),
                    "baseline_matched": nb(p & before & compare),
                    "matched": nb(p & after & compare),
                    "baseline_extra": nb(n & before),
                    "extra": nb(n & after),
                    "ordinary": nb(a),
                    "direction_drift": nb(
                        a & changed & (inside[:, 0] != outside[:, 0])
                    ),
                }
            )
    total = {k: sum(d[k] for d in per) for k in per[0] if k != "source"}
    gate = (
        total["matched"] > total["baseline_matched"]
        and total["extra"] <= total["baseline_extra"]
        and total["direction_drift"] <= 0.01 * total["ordinary"]
    )
    output.mkdir(parents=True, exist_ok=False)
    actor.cpu()
    fold_spell_correction(actor, fix.cpu(), group)
    policy = output / "world_iql.bin"
    # Preserve even the original GPU rounding of unaffected native key rows.
    native_weights = np.frombuffer(late[-head_bytes:], "<f4").copy()
    native_weights[: -len(cat)].reshape(len(cat), -1)[group.numpy()] += (
        fix.weight.detach().numpy()
    )
    native_weights[-len(cat) :][group.numpy()] += fix.bias.detach().numpy()
    policy.write_bytes(
        phase_checkpoint(opening, late[:-head_bytes] + native_weights.tobytes())
    )
    saved["actor"] = actor.state_dict()
    torch.save(saved, output / "learner.pt")

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    report = {
        "method": "World-assisted IQL continuation and measured short-horizon rewards for a downward-spell logit correction",
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "seed": seed,
        "updates": updates,
        "training_counts": {
            "confirmed": len(sets[0]),
            "hard_abstentions": len(sets[1]),
            "ordinary": len(sets[2]),
        },
        "archive_sha256": sha(archive),
        "parent_sha256": sha(deployed),
        "policy_sha256": sha(policy),
        "per_held_game": per,
        "holdout": total,
        "trial_eligible": bool(gate),
        "scope": "Observed teacher states; matching at the issued tick and clear abstentions screen candidates. Future arena fights determine win and no-hit counts.",
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("archive", "parent", "deployed", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--updates", type=int, default=450)
    p.add_argument("--seed", type=int, default=9812)
    args = p.parse_args()
    fit(args.archive, args.parent, args.deployed, args.output, args.updates, args.seed)

"""Train bounded recurrent control residuals against native rewards and a frozen world model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_sisters_native_iql import REWARD_WEIGHTS, attribute_events
from train_sisters_timing_iql import (
    PHYSICAL,
    TimingPolicy,
    mlp,
    motion_history,
    phase_components,
)

HIDDEN = 24
SPAN = 3.2
MAGIC = 0x384C5149


class MemoryActor(nn.Module):
    def __init__(self, actions: int, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.memory = nn.GRU(192, HIDDEN, batch_first=True)
        self.delta = nn.Linear(HIDDEN, actions)
        self.value = nn.Linear(HIDDEN, 1)
        self.events = nn.Linear(HIDDEN, 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, raw, encoded, base):
        x = encoded if self.variant == "encoded" else raw
        if self.variant == "instant":
            h = self.memory(x.flatten(0, 1)[:, None])[0].reshape(*x.shape[:2], HIDDEN)
        else:
            h = self.memory(x)[0]
        logits = base + 3.0 * self.delta(h).tanh()
        return logits, self.value(h).squeeze(-1), self.events(h)


def segments(time, state):
    blocks, part = [], []
    for i, t in enumerate(time):
        if state[i, 23] <= 0.5:
            if part:
                blocks.append(part)
                part = []
            continue
        if part and (
            t >= time[part[0]] + np.float32(SPAN)
            or t - time[part[-1]] > np.float32(0.081)
        ):
            blocks.append(part)
            part = []
        part.append(i)
    if part:
        blocks.append(part)
    return blocks


def measured_targets(t, s, ev, value, one_step):
    n = len(t)
    ix = np.arange(n)
    boundary = np.r_[True, (np.diff(t) > 0.081) | (s[:-1, 23] != s[1:, 23])]
    segment = np.cumsum(boundary)
    terminal = np.flatnonzero(ev[:, 6:].any(1))
    if len(terminal) > 1:
        raise ValueError("Repeated terminal")
    terminal_end = terminal[0] + 1 if len(terminal) else n + 1
    reward = ev @ REWARD_WEIGHTS
    discount = np.exp(np.log(0.995) * (t - t[0]) / 0.04)
    prefix = np.r_[0, np.cumsum(reward * discount)]
    hp = np.rint(s[:, 4] * 9)
    debit = np.r_[np.maximum(-np.diff(hp), 0), 0]
    losses = np.r_[0, np.cumsum((debit > 0) | ev[:, 1:5].any(1))]
    damage = np.r_[0, np.cumsum(ev[:, 0])]

    def end_at(seconds):
        end = np.minimum(np.searchsorted(t, t + seconds - 1e-5), terminal_end)
        end = end.clip(0, n)
        last = (end - 1).clip(0, n - 1)
        complete = (
            (end > ix)
            & (ix < terminal_end)
            & (segment[last] == segment)
            & (
                (end == terminal_end)
                | (
                    (end < n)
                    & (segment[end.clip(0, n - 1)] == segment)
                    & (t[end.clip(0, n - 1)] >= t + seconds - 1e-4)
                )
            )
        )
        return end, complete

    end, valid = end_at(4.0)
    k = end.clip(0, n - 1)
    g = (prefix[end] - prefix[ix]) / discount + np.where(
        end == terminal_end, 0, discount[k] / discount * value[k]
    )
    residual = 0.75 * (g - value) + 0.25 * (
        reward + 0.995 * one_step * (~ev[:, 6:].any(1)) - value
    )
    e0, m0 = end_at(0.72)
    e1, m1 = end_at(2.4)
    targets = np.column_stack(
        [
            losses[e0] > losses[ix],
            (damage[e1] > damage[ix]) & (losses[e1] == losses[ix]),
        ]
    )
    return residual.astype("f4"), valid, targets.astype("f4"), np.column_stack([m0, m1])


def load_replay(path, ck, device):
    cat = ck["catalogue"].long()
    actor = TimingPolicy(cat, True).to(device).eval()
    actor.load_state_dict(ck["actor"])
    width = ck["value"]["0.weight"].shape[1]
    val = mlp(width, 1).to(device).eval()
    val.load_state_dict(ck["value"])
    world = mlp(width + 11, len(PHYSICAL)).to(device).eval()
    world.load_state_dict(ck["world"])
    codes = {tuple(x): i for i, x in enumerate(cat.tolist())}
    container = {s: [] for s in ("train", "holdout")}
    seen = set()
    with zipfile.ZipFile(path) as z, torch.inference_mode():
        for m in json.loads(z.read("manifest.json")):
            raw = z.read(m["file"])
            source = m["source"]
            if source in seen or hashlib.sha256(raw).hexdigest() != m["sha256"]:
                raise ValueError("Duplicate or changed fight")
            seen.add(source)
            with np.load(io.BytesIO(raw)) as d:
                t = d["control_time"]
                s = d["timing_state"]
                a = d["controls"].astype(int)
                pieces = segments(t, s)
                if not pieces:
                    continue
                h = torch.from_numpy(motion_history(t, s)).to(device)
                acts = np.array([codes.get(tuple(x), -1) for x in a], dtype="i8")
                u = np.column_stack(
                    [a[:, :2], (a[:, 2, None] >> np.arange(9)) & 1]
                ).astype("f4")
                ux = torch.from_numpy(u).to(device)
                enc = []
                log = []
                v = []
                v_model = []
                for i in range(0, len(s), 512):
                    j = slice(i, min(i + 512, len(s)))
                    x = h[j]
                    hidden = actor.encode(x)
                    enc.append(hidden.cpu().numpy())
                    log.append(actor.heads(hidden)[0].cpu().numpy())
                    v.append(val(x).flatten().cpu().numpy())
                    nex = h[
                        torch.arange(i, i + len(x), device=device)
                        .add(1)
                        .clamp_max(len(h) - 1)
                    ].clone()
                    nex[:, PHYSICAL] = x[:, PHYSICAL] + world(torch.cat((x, ux[j]), 1))
                    v_model.append(val(nex).flatten().cpu().numpy())
                enc = np.vstack(enc)
                log = np.vstack(log)
                v = np.concatenate(v)
                v_model = np.concatenate(v_model)
                ev = attribute_events(
                    t,
                    d["time"],
                    d["elapsed_ticks"],
                    d["events"],
                    health_masks=s[:, 4] * 9,
                )
                target, eligible, events, valid_event = measured_targets(
                    t.astype("f8"),
                    s,
                    ev,
                    v,
                    0.75 * np.r_[v[1:], v[-1]] + 0.25 * v_model,
                )
                teacher = (
                    d["teacher_acted"].astype(bool).reshape(-1)
                    if "teacher_acted" in d
                    else np.zeros(len(s), bool)
                )
                neural_only = "teacher_controls" not in d
                changed = np.r_[False, np.any(a[1:] != a[:-1], axis=1)]
                rows = (
                    s,
                    enc,
                    log,
                    acts,
                    target,
                    eligible & (acts >= 0),
                    events,
                    valid_event,
                    teacher,
                    changed,
                )
                for piece in pieces:
                    group = [np.asarray(col[piece]) for col in rows]
                    container[m["split"]].append((source, neural_only, group))

    def tensorize(entries):
        labels = np.array([e[0] for e in entries])
        neural = np.array([e[1] for e in entries])
        columns = []
        longest = max(len(e[2][0]) for e in entries)
        for k in range(10):
            first = entries[0][2][k]
            arr = np.zeros((len(entries), longest, *first.shape[1:]), dtype=first.dtype)
            for i, (_, _, game) in enumerate(entries):
                arr[i, : len(game[k])] = game[k]
            columns.append(torch.from_numpy(arr).to(device))
        return labels, neural, columns

    return {k: tensorize(v) for k, v in container.items()}


def masked_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


def update(net, optim, data, batch):
    raw, emb, base, a, target, valid, event, known, teacher, changed = [
        x[batch] for x in data
    ]
    logits, value, events = net(raw, emb, base)
    # Baseline and residual critic both see only observations available before the chosen key.
    advantage = (target - value.detach()).clamp(-4, 4)
    weight = torch.exp(advantage / 2).clamp(0.15, 5) * (
        1 + 0.6 * teacher + 0.5 * changed
    )
    key_logp = (
        F.log_softmax(logits, -1).gather(-1, a.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    )
    prior = F.kl_div(
        F.log_softmax(logits, -1), F.softmax(base, -1), reduction="none"
    ).sum(-1)
    nll = masked_mean(-weight * key_logp, valid)
    critic = masked_mean(
        F.smooth_l1_loss(value, target.clamp(-12, 12), reduction="none"), valid
    )
    event_loss = masked_mean(
        F.binary_cross_entropy_with_logits(events, event, reduction="none"), known
    )
    loss = nll + 0.8 * masked_mean(prior, valid) + 0.35 * critic + 2.0 * event_loss
    optim.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 2.0)
    optim.step()
    return float(loss.detach())


@torch.inference_mode()
def heldout(net, cached, cat):
    labels, neural, c = cached
    per = {}
    for source in np.unique(labels):
        indices = np.flatnonzero(labels == source)
        sums = np.zeros(12)
        brier = np.zeros(2)
        counts = np.zeros(2)
        for chunk in np.array_split(indices, max(1, (len(indices) + 31) // 32)):
            raw, emb, base, a, target, valid, event, known, teacher, _ = [
                x[chunk] for x in c
            ]
            logits, _, e = net(raw, emb, base)
            pred = logits.argmax(-1)
            old = base.argmax(-1)
            keys = cat[a.clamp_min(0)]
            issued = teacher & valid
            trust = valid & (target.abs() > 0.1)
            gain = (
                (F.log_softmax(logits, -1) - F.log_softmax(base, -1))
                .gather(-1, a.clamp_min(0).unsqueeze(-1))
                .squeeze(-1)
            )
            sums[:9] += [
                valid.sum().item(),
                (valid & (pred != old)).sum().item(),
                issued.sum().item(),
                (issued & (cat[pred, 0] == keys[:, :, 0])).sum().item(),
                (issued & (cat[old, 0] == keys[:, :, 0])).sum().item(),
                (issued & (pred == a)).sum().item(),
                (issued & (old == a)).sum().item(),
                ((gain * target.clamp(-4, 4)) * trust).sum().item(),
                trust.sum().item(),
            ]
            brier += (((e.sigmoid() - event) ** 2) * known).sum((0, 1)).cpu().numpy()
            counts += known.sum((0, 1)).cpu().numpy()
        per[source] = {
            "teacher_free": bool(neural[indices].all()),
            "factual_inputs": int(sums[0]),
            "changed_inputs": int(sums[1]),
            "issued_teacher_inputs": int(sums[2]),
            "correct_teacher_direction": int(sums[3]),
            "base_correct_teacher_direction": int(sums[4]),
            "matched_teacher_key": int(sums[5]),
            "base_matched_teacher_key": int(sums[6]),
            "return_weighted_shift": float(sums[7] / max(1, sums[8])),
            "brier": (brier / np.maximum(counts, 1)).tolist(),
            "event_windows": counts.astype(int).tolist(),
        }
    return per


def train(archive, parent, deployed, output, updates, batch, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(parent, weights_only=True, map_location="cpu")
    cat = ck["catalogue"].long().to(device)
    games = load_replay(archive, ck, device)
    train_ids, _, train_data = games["train"]
    print(
        "REPLAY",
        len(set(train_ids)),
        len(set(games["holdout"][0])),
        len(train_ids),
        flush=True,
    )
    _, inv, n = np.unique(train_ids, return_inverse=True, return_counts=True)
    probabilities = 1 / n[inv]
    probabilities /= probabilities.sum()
    reports = {}
    stats = {}
    raw_net = None
    for name in ("raw", "instant", "encoded"):
        torch.manual_seed(seed)
        net = MemoryActor(len(cat), name).to(device)
        optim = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
        rng = np.random.default_rng(seed)
        for step in range(updates):
            ix = rng.choice(len(train_ids), batch, replace=True, p=probabilities)
            loss = update(net, optim, train_data, ix)
            if step % 300 == 0:
                print(name, step, round(loss, 4), flush=True)
        reports[name] = heldout(net, games["holdout"], cat)
        stats[name] = net.cpu().state_dict()
        if name == "raw":
            raw_net = net
    opening, late = phase_components(deployed.read_bytes())
    buf = [struct.pack("<2i", MAGIC, len(late)), late]
    for obj in (
        raw_net.memory.weight_ih_l0,
        raw_net.memory.weight_hh_l0,
        raw_net.memory.bias_ih_l0,
        raw_net.memory.bias_hh_l0,
        raw_net.delta.weight,
        raw_net.delta.bias,
    ):
        buf.append(obj.detach().numpy().astype("<f4").tobytes())
    tail = b"".join(buf)
    blob = struct.pack("<3i", 0x334C5149, len(opening), len(tail)) + opening + tail
    output.mkdir(parents=True, exist_ok=False)
    (output / "world_iql.bin").write_bytes(blob)
    torch.save(
        {"variants": stats, "catalogue": cat.cpu(), "hidden": HIDDEN, "span": SPAN},
        output / "learner.pt",
    )
    neural = [
        k
        for k, r in reports["raw"].items()
        if r["teacher_free"] and min(r["event_windows"]) > 0
    ]

    def mismatch(group):
        return sum(r["changed_inputs"] for r in group.values()) / max(
            1, sum(r["factual_inputs"] for r in group.values())
        )

    def teacher_score(group, k):
        return sum(r[k] for r in group.values()) / max(
            1, sum(r["issued_teacher_inputs"] for r in group.values())
        )

    a = reports["raw"]
    delta = np.array(
        [np.subtract(reports["instant"][s]["brier"], a[s]["brier"]) for s in neural]
    )
    changed = mismatch(a)
    direction_change = teacher_score(a, "correct_teacher_direction") - teacher_score(
        a, "base_correct_teacher_direction"
    )
    eligible = bool(
        len(neural) >= 3
        and np.all(delta.mean(0) > 0)
        and 0.001 <= changed <= 0.25
        and direction_change >= -0.03
    )
    result = {
        "method": "Recurrent advantage-weighted offline actor-critic on native actions, with real four-second returns and a frozen world-assisted one-step target. Whole continuous blocks are replayed from zero state. Only the raw-state actor is eligible for deployment.",
        "hardware": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "updates_per_variant": updates,
        "variants": reports,
        "batch": batch,
        "seed": seed,
        "hidden": HIDDEN,
        "span_seconds": SPAN,
        "input_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "parent_sha256": hashlib.sha256(deployed.read_bytes()).hexdigest(),
        "learner_input_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(blob).hexdigest(),
        "teacher_free_held_fights": neural,
        "raw_vs_instant_mean_brier_reduction": delta.mean(0).tolist(),
        "changed_native_action_rate": changed,
        "issued_teacher_direction_accuracy_change": direction_change,
        "trial_eligible": eligible,
    }
    (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FIT", eligible, delta.mean(0), changed, direction_change, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "parent", "deployed", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--seed", type=int, default=71964)
    train(**vars(p.parse_args()))

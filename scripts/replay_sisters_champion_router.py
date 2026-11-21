"""Replay the continued router and both frozen primitive actors on recorded states."""

from __future__ import annotations

import argparse
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch
from sisters_control_blocks import neural_block_starts
from train_sisters_champion_router import Change, features
from train_sisters_timing_iql import (
    TimingPolicy,
    motion_history,
    option_components,
    phase_components,
    router_residual_components,
)


def load_router(parent: Path, checkpoint: Path):
    old_open, old_route = phase_components(parent.read_bytes())
    opening, route = phase_components(checkpoint.read_bytes())
    old, packed = router_residual_components(route)
    if (opening, old) != (old_open, old_route):
        raise ValueError("Original winning components changed")
    _, _, gating = option_components(old)
    change = Change().eval().requires_grad_(False)
    cursor = 1
    for layer in change.net.parameters():
        end = cursor + layer.numel()
        layer.copy_(torch.from_numpy(packed[cursor:end].copy()).reshape_as(layer))
        cursor = end
    if cursor != len(packed):
        raise ValueError("Unexpected MLP weights")
    return gating, float(packed[0]), change


def load_arms(primary: Path, alternate: Path):
    nets = []
    for path in (primary, alternate):
        ck = torch.load(path, weights_only=True, map_location="cpu")
        model = TimingPolicy(ck["catalogue"], True).eval().requires_grad_(False)
        model.load_state_dict(ck["actor"])
        nets.append((model, ck["catalogue"]))
    return nets


def predict_arms(nets, clock, state):
    history = torch.from_numpy(motion_history(clock, state))
    logits = []
    primary_emb = None
    for actor, cat in nets:
        h = torch.cat([actor.encode(part) for part in history.split(512)])
        logits.append(cat[actor.heads(h)[0].argmax(1)].numpy())
        if primary_emb is None:
            primary_emb = h.numpy()
    return logits, primary_emb


def replay(
    parent: Path,
    checkpoint: Path,
    primary: Path,
    alternate: Path,
    data: Path,
    output: Path,
) -> None:
    torch.set_num_threads(1)
    gating, bound, change = load_router(parent, checkpoint)
    nets = load_arms(primary, alternate)
    output.mkdir(exist_ok=True, parents=True)
    reports = []
    with zipfile.ZipFile(data) as archive, torch.inference_mode():
        for g in json.loads(archive.read("manifest.json")):
            if g["split"] != "holdout":
                continue
            with np.load(io.BytesIO(archive.read(g["file"]))) as tape:
                t, s = tape["control_time"], tape["timing_state"]
                logits, primary_emb = predict_arms(nets, t, s)
                starts = neural_block_starts(t)
                lengths = np.diff(np.r_[starts, len(t)])
                x, margin = features(primary_emb[starts], s[starts], gating)
                pred = margin + bound * change(torch.from_numpy(x)).numpy()
                active = s[starts, 23] > 0.5
                assigned = np.repeat(active & (pred > 0), lengths) & (s[:, 23] > 0.5)
                original = np.repeat(active & (margin > 0), lengths) & (s[:, 23] > 0.5)
                expected = np.where(assigned[:, None], logits[1], logits[0])
                opening = s[:, 23] <= 0.5
                expected[opening] = tape["controls"][opening]
                prefix = g["source"].split("/")[-1]
                with (output / (prefix + ".parity")).open("wb") as f:
                    f.write(struct.pack("<i", len(s)))
                    for now, state, keys, arm in zip(
                        t, s, expected, assigned, strict=True
                    ):
                        f.write(
                            struct.pack("<f", now)
                            + state.astype("<f4").tobytes()
                            + struct.pack("<4i", *keys, arm)
                        )
                reports.append(
                    {
                        "source": g["source"],
                        "decisions": len(t),
                        "changed_route_ticks": int((assigned != original).sum()),
                        "changed_native_inputs": int(
                            (
                                (assigned != original) & (logits[1] != logits[0]).any(1)
                            ).sum()
                        ),
                    }
                )
    (output / "neural-reference.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "checkpoint", "primary", "alternate", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    replay(**vars(parser.parse_args()))

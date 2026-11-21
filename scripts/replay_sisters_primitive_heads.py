"""Replay both updated native key heads, with the nine-win opening and routing held fixed."""

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
from train_sisters_champion_router import features
from train_sisters_primitive_heads import read_leaf
from train_sisters_timing_iql import motion_history, option_components, phase_components


def models(parent: bytes, policy: bytes):
    pre, old_router = phase_components(parent)
    current, router = phase_components(policy)
    a, b, gate = option_components(router)
    old_a, old_b, old_gate = option_components(old_router)
    if pre != current or not np.array_equal(gate, old_gate):
        raise ValueError("Opening or learned routing changed")
    nets = []
    old_nets = []
    catalogues = []
    for before, after in zip((old_a, old_b), (a, b), strict=True):
        net, cat = read_leaf(after)
        baseline, _ = read_leaf(before)
        n = len(cat) * 193 * 4
        if before[:-n] != after[:-n] or len(before) != len(after):
            raise ValueError("The original action catalogue or encoder changed")
        nets.append(net)
        old_nets.append(baseline)
        catalogues.append(cat)
    return nets, old_nets, catalogues, gate


def actions(t, s, nets, catalogues, gate):
    geometry = torch.from_numpy(motion_history(t, s))
    h = [
        torch.cat([net.encode(chunk) for chunk in geometry.split(512)]) for net in nets
    ]
    starts = neural_block_starts(t)
    _, scores = features(h[0][starts].numpy(), s[starts], gate)
    arm = np.repeat(
        (scores > 0) & (s[starts, 23] > 0.5), np.diff(np.r_[starts, len(t)])
    ) & (s[:, 23] > 0.5)
    indices = [net.out(x).argmax(1).numpy() for net, x in zip(nets, h, strict=True)]
    keys = np.where(arm[:, None], catalogues[1][indices[1]], catalogues[0][indices[0]])
    return keys, arm, h


def replay(parent, checkpoint, data, output):
    torch.set_num_threads(1)
    nets, baselines, cats, gate = models(parent.read_bytes(), checkpoint.read_bytes())
    output.mkdir(exist_ok=True, parents=True)
    reports = []
    with zipfile.ZipFile(data) as z, torch.inference_mode():
        for g in json.loads(z.read("manifest.json")):
            if g["split"] != "holdout":
                continue
            with np.load(io.BytesIO(z.read(g["file"]))) as d:
                s, t = d["timing_state"], d["control_time"]
                keys, arm, h = actions(t, s, nets, cats, gate)
                base = [
                    cat[net.out(x).argmax(1).numpy()]
                    for net, cat, x in zip(baselines, cats, h, strict=True)
                ]
                changed = (s[:, 23] > 0.5) & (
                    keys != np.where(arm[:, None], base[1], base[0])
                ).any(1)
                keys[s[:, 23] <= 0.5] = d["controls"][s[:, 23] <= 0.5]
                with (output / (g["source"].split("/")[-1] + ".parity")).open(
                    "wb"
                ) as fp:
                    fp.write(struct.pack("<i", len(t)))
                    for time, state, key, a in zip(t, s, keys, arm, strict=True):
                        fp.write(
                            struct.pack("<f", time)
                            + state.astype("<f4").tobytes()
                            + struct.pack("<4i", *key, a)
                        )
                reports.append(
                    {
                        "source": g["source"],
                        "decisions": len(s),
                        "changed_native_inputs": int(changed.sum()),
                    }
                )
    (output / "neural-reference.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "checkpoint", "data", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    replay(**vars(parser.parse_args()))

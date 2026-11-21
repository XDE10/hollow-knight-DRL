"""Generate an independent native-key reference tape for a frozen directional policy."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from sisters_directional_options import HOLD_SECONDS, menu_logits, menu_slots
from train_sisters_timing_iql import (
    TimingPolicy,
    motion_history,
    phase_components,
    policy_gradient_components,
)


def replay(checkpoint: Path, learner: Path, trajectory: Path, output: Path) -> None:
    torch.set_num_threads(1)
    ck = torch.load(learner, map_location="cpu", weights_only=True)
    actor = TimingPolicy(ck["catalogue"], True).eval().requires_grad_(False)
    actor.load_state_dict(ck["actor"])
    catalogue = ck["catalogue"].numpy()
    bits = catalogue[:, 2]
    _, leaf = phase_components(checkpoint.read_bytes())
    _, head = policy_gradient_components(leaf)
    interval = np.float32(struct.unpack_from("<f", head, 4)[0])
    weights = np.frombuffer(head, "<f4", offset=8).reshape(7, 193).astype("f8")
    with np.load(trajectory) as data:
        times, states, controls = (
            data["control_time"],
            data["timing_state"],
            data["controls"].copy(),
        )
    history = torch.from_numpy(motion_history(times, states))
    with torch.inference_mode():
        h = torch.cat([actor.encode(part) for part in history.split(384)])
        scores = actor.heads(h)[0].numpy()
    x = h.numpy().astype("f8")
    risk = x @ weights[0, 1:] + weights[0, 0]
    slots, primary = menu_slots(scores, catalogue, states)
    groups = menu_logits(x, weights[1:], slots, primary).argmax(1)
    best = scores.argmax(1)
    next_draw = release = np.float32(0)
    onset = 0
    count = alternate = changed = 0
    for i, time in enumerate(times):
        if states[i, 23] <= 0.5:
            continue
        key = best[i]
        if time < release:
            allowed = (catalogue[:, 0] == catalogue[onset, 0]) & (
                ((bits ^ bits[onset]) & 2) == 0
            )
            allowed &= (bits & 21 & ~bits[key]) == 0
            if allowed.any():
                key = np.where(allowed, scores[i], -np.inf).argmax()
        elif (
            time >= next_draw
            and states[i, 9] <= 0.5
            and risk[i] >= 0
            and (slots[i] >= 0).sum() >= 2
        ):
            count += 1
            next_draw = time + interval
            if groups[i] != primary[i]:
                alternate += 1
                onset = key = slots[i, groups[i]]
                release = time + HOLD_SECONDS
        controls[i] = catalogue[key]
        changed += int(key != best[i])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "development.parity").open("wb") as f:
        f.write(struct.pack("<i", len(times)))
        for t, s, key in zip(times, states, controls, strict=True):
            f.write(
                struct.pack("<f", t)
                + s.astype("<f4").tobytes()
                + struct.pack("<4i", *key, 0)
            )
    result = {"decisions": count, "alternatives": alternate, "changes": changed}
    (output / "option-reference.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("checkpoint", "learner", "trajectory", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    replay(**vars(parser.parse_args()))

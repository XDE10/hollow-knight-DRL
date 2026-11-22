"""Export a trained encoded-GRU Sisters actor for the native Unity action loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import Tensor
from train_sisters_timing_iql import (
    CONTINUOUS_MEMORY_MAGIC,
    ENCODED_MEMORY_MAGIC,
    TimingPolicy,
    memory_components,
    option_components,
    phase_checkpoint,
    phase_components,
    save_native,
)


def export(
    parent: bytes,
    catalogue: Tensor,
    state: Mapping[str, Tensor],
    *,
    continuous: bool = False,
) -> bytes:
    opening, trio = phase_components(parent)
    left, right, _ = option_components(trio)
    for arm in (left, right):
        count = struct.unpack_from("<i", arm, 16)[0]
        keys = np.frombuffer(arm, "<i4", count=3 * count, offset=20).reshape(-1, 3)
        if not np.array_equal(keys, catalogue.numpy()):
            raise ValueError("The trained actor must retain the parent's original Trio catalogue")
    encoder = TimingPolicy(catalogue, history=True)
    encoder_state = {key: state[f"encoder.{key}"] for key in encoder.state_dict()}
    if not torch.equal(encoder_state["basis"], encoder.basis):
        raise ValueError("The encoded actor's factor basis does not match the catalogue")
    encoder.load_state_dict(encoder_state, strict=True)
    projection = state["memory_projection.weight"]
    if projection.ndim != 2 or projection.shape[0] != 192:
        raise ValueError("Expected the recurrent projection to use the 192-feature encoder")
    hidden = projection.shape[1]
    if not 1 <= hidden <= 256:
        raise ValueError("Unsupported native recurrent hidden size")
    gates = 3 * hidden
    layouts = {
        "memory.weight_ih_l0": (gates, 192),
        "memory.weight_hh_l0": (gates, hidden),
        "memory.bias_ih_l0": (gates,),
        "memory.bias_hh_l0": (gates,),
        "memory_projection.bias": (192,),
    }
    for name, layout in layouts.items():
        if tuple(state[name].shape) != layout:
            raise ValueError(f"Invalid layout for {name}")
    if any(
        value.dtype != torch.float32 or not torch.isfinite(value).all()
        for value in (
            *encoder_state.values(),
            projection,
            *(state[key] for key in layouts),
        )
    ):
        raise ValueError("Native actor weights must be finite float32 tensors")

    with torch.no_grad():
        effective_head = encoder.out.weight + encoder.basis @ encoder.factors.weight
        recurrence_weights = effective_head @ projection
        encoder.out.bias.add_(effective_head @ state["memory_projection.bias"])
    with TemporaryDirectory() as temp:
        leaf_path = Path(temp) / "trio.bin"
        save_native(encoder, catalogue.numpy(), leaf_path)
        leaf = leaf_path.read_bytes()
    values = [
        *(state[key] for key in layouts if key.startswith("memory.")),
        recurrence_weights,
        torch.zeros(len(catalogue), dtype=torch.float32),
    ]
    data = b"".join(v.detach().numpy().astype("<f4", copy=False).tobytes() for v in values)
    magic = CONTINUOUS_MEMORY_MAGIC if continuous else ENCODED_MEMORY_MAGIC
    component = struct.pack("<3i", magic, len(leaf), hidden) + leaf + data
    memory_components(component)
    return phase_checkpoint(opening, component)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--learner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(args.learner, map_location="cpu", weights_only=True)
    continuous = saved.get("provenance", {}).get("method") == "recurrent-iql-v2"
    data = export(
        args.parent.read_bytes(), saved["catalogue"], saved["state"], continuous=continuous
    )
    with args.output.open("xb") as stream:
        stream.write(data)
    opening, _ = phase_components(data)
    print(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
                "opening_sha256": hashlib.sha256(opening).hexdigest(),
                "actions": len(saved["catalogue"]),
            }
        )
    )


if __name__ == "__main__":
    main()

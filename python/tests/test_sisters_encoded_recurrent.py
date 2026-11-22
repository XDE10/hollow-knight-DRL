from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import torch


def test_encoded_gru_export_preserves_opening_and_unfolded_logits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    export = importlib.import_module("export_sisters_encoded_recurrent")
    timing = importlib.import_module("train_sisters_timing_iql")
    native = importlib.import_module("train_sisters_primitive_heads")
    cats = torch.tensor([[-1, 0, 0], [0, 0, 8], [1, 0, 4]])
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(41)
        encoder = timing.TimingPolicy(cats, history=True)
        memory = torch.nn.GRU(192, 5, batch_first=True)
        projection = torch.nn.Linear(5, 192)
        initial_path = tmp_path / "leaf.bin"
        timing.save_native(encoder, cats.numpy(), initial_path)
        opening = initial_path.read_bytes()
        gate = np.ones(
            timing.WIDTH * timing.OPTION_FEATURES + 3 * timing.OPTION_FEATURES + 1, dtype="<f4"
        )
        two_arms = (
            struct.pack("<3i", timing.OPTION_MAGIC, len(opening), len(opening))
            + opening
            + opening
            + gate.tobytes()
        )
        parent = timing.phase_checkpoint(opening, two_arms)
        state = {
            f"{name}.{key}": tensor
            for name, module in (
                ("encoder", encoder),
                ("memory", memory),
                ("memory_projection", projection),
            )
            for key, tensor in module.state_dict().items()
        }
        binary = export.export(parent, cats, state)
        frozen_opening, trio = timing.phase_components(binary)
        assert frozen_opening == opening
        continuous_opening, continuous_trio = timing.phase_components(
            export.export(parent, cats, state, continuous=True)
        )
        assert continuous_opening == opening
        assert struct.unpack_from("<i", continuous_trio)[0] == timing.CONTINUOUS_MEMORY_MAGIC
        assert continuous_trio[4:] == trio[4:]
        assert struct.unpack_from("<i", trio, 8)[0] == memory.hidden_size
        materialized, parameters = timing.memory_components(trio)
        leaf, exported_cats = native.read_leaf(materialized)
        np.testing.assert_array_equal(exported_cats, cats.numpy())
        states = torch.randn(7, timing.WIDTH + 2 * timing.HISTORY_BLOCK)
        for slot in (24, 60, 96, 132, 146):
            states[:, slot] = 1.0
        features = encoder.encode(states)
        hidden, _ = memory(features.unsqueeze(0))
        expected, _ = encoder.heads(features + projection(hidden[0]))
        tail = torch.from_numpy(parameters[-len(cats) * 6 :].copy())
        hidden_weights = tail[: len(cats) * 5].reshape(len(cats), 5)
        folded = leaf(states)[0] + hidden[0] @ hidden_weights.T + tail[-len(cats) :]
        torch.testing.assert_close(folded, expected, rtol=2e-5, atol=2e-6)

        broken = bytearray(trio)
        struct.pack_into("<i", broken, 8, 257)
        with pytest.raises(ValueError, match="weight count"):
            timing.memory_components(bytes(broken))
        with pytest.raises(ValueError, match="original Trio catalogue"):
            export.export(parent, cats.flip(0), state)

"""WP18 Scaled Deep Residual + Multi-Layer Continuous GRU Neural Actor, Critic & Exporter.

Binary Magic: 0x3E4C5149 (exact numerical parity with C# SistersScaledRecurrent.cs).
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

OBS_WIDTH = 192
HISTORY_BLOCK = OBS_WIDTH + 1
FOES, DISCS, CONTROL = 24, 132, 160
SCALED_MAGIC = 0x3E4C5149
PHASE_MAGIC = 0x334C5149


class PreLNResFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.fc1(self.ln(x)))
        if self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return x + self.fc2(h)


class ScaledBackbone(nn.Module):
    def __init__(
        self,
        d_boss: int = 96,
        d_disc: int = 48,
        d_past: int = 64,
        d_model: int = 384,
        d_ffn: int = 512,
        n_blocks: int = 2,
        n_gru_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = {
            "d_boss": d_boss,
            "d_disc": d_disc,
            "d_past": d_past,
            "d_model": d_model,
            "d_ffn": d_ffn,
            "n_blocks": n_blocks,
            "d_gru": d_model,
            "n_gru_layers": n_gru_layers,
        }
        self.boss1 = nn.Linear(36, d_boss)
        self.boss2 = nn.Linear(d_boss, d_boss)
        self.disc1 = nn.Linear(14, d_disc)
        self.disc2 = nn.Linear(d_disc, d_disc)
        self.past = nn.Linear(HISTORY_BLOCK, d_past)

        enc_in = OBS_WIDTH + 2 * d_boss + 2 * d_disc + 2 * d_past
        self.stem = nn.Linear(enc_in, d_model)
        self.stem_ln = nn.LayerNorm(d_model, eps=1e-5)
        self.blocks = nn.ModuleList([PreLNResFFN(d_model, d_ffn, dropout=dropout) for _ in range(n_blocks)])
        self.grus = nn.ModuleList([nn.GRU(d_model, d_model, batch_first=True) for _ in range(n_gru_layers)])
        self.gru_lns = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-5) for _ in range(n_gru_layers)])

    def encode_spatial(self, flat_x: Tensor) -> Tensor:
        raw = flat_x[:, :OBS_WIDTH]
        bosses = raw[:, FOES:DISCS].reshape(-1, 3, 36)
        discs = raw[:, DISCS:CONTROL].reshape(-1, 2, 14)
        b_mask = (bosses[:, :, :1] != 0).float()
        d_mask = (discs[:, :, :1] != 0).float()
        bosses_c = bosses * b_mask
        discs_c = discs * d_mask

        b_feat = F.relu(self.boss2(F.silu(self.boss1(bosses_c)))) * b_mask
        d_feat = F.relu(self.disc2(F.silu(self.disc1(discs_c)))) * d_mask
        boss_max = b_feat.amax(dim=1)
        boss_sum = b_feat.sum(dim=1)
        disc_max = d_feat.amax(dim=1)
        disc_sum = d_feat.sum(dim=1)

        past = flat_x[:, OBS_WIDTH:].reshape(-1, HISTORY_BLOCK)
        p_foes = past[:, FOES:DISCS].reshape(-1, 3, 36)
        p_discs = past[:, DISCS:CONTROL].reshape(-1, 2, 14)
        p_clean = torch.cat(
            (
                past[:, :FOES],
                (p_foes * (p_foes[:, :, :1] != 0).float()).flatten(1),
                (p_discs * (p_discs[:, :, :1] != 0).float()).flatten(1),
                past[:, CONTROL:],
            ),
            dim=1,
        )
        past_emb = (F.relu(self.past(p_clean)) * (past[:, -1:] > 0).float()).reshape(len(flat_x), -1)

        enc_in = torch.cat(
            (
                raw[:, :FOES],
                bosses_c.flatten(1),
                discs_c.flatten(1),
                raw[:, CONTROL:OBS_WIDTH],
                boss_max,
                boss_sum,
                disc_max,
                disc_sum,
                past_emb,
            ),
            dim=1,
        )
        h = F.silu(self.stem_ln(self.stem(enc_in)))
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward_sequence(
        self, inputs: Tensor, burn: int = 0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (spatial_enc, post_gru_emb, concat_gru_h) for steps [burn:]."""
        states: list[Tensor | None] = [None] * len(self.grus)
        if burn:
            with torch.no_grad():
                pre = self.encode_spatial(inputs[:, :burn].flatten(0, 1)).unflatten(0, (len(inputs), burn))
                cur = pre
                for l, (gru, ln) in enumerate(zip(self.grus, self.gru_lns, strict=True)):
                    gru.flatten_parameters()
                    out, h_n = gru(cur, None)
                    states[l] = h_n
                    cur = ln(cur + out)

        learn = inputs[:, burn:]
        spatial = self.encode_spatial(learn.flatten(0, 1)).unflatten(0, learn.shape[:2])
        cur = spatial
        h_all: list[Tensor] = []
        for l, (gru, ln) in enumerate(zip(self.grus, self.gru_lns, strict=True)):
            gru.flatten_parameters()
            out, _ = gru(cur, states[l])
            h_all.append(out)
            cur = ln(cur + out)
        return spatial, cur, torch.cat(h_all, dim=-1)


class ScaledRecurrentActor(nn.Module):
    def __init__(self, catalogue: Tensor, **backbone_kwargs: Any) -> None:
        super().__init__()
        catalogue = catalogue.long()
        self.backbone = ScaledBackbone(**backbone_kwargs)
        d_model = self.backbone.cfg["d_model"]
        self.out = nn.Linear(d_model, len(catalogue))
        self.factors = nn.Linear(d_model, 15)
        bits = ((catalogue[:, 2, None] >> torch.arange(9, device=catalogue.device)) & 1).float()
        basis = torch.cat(
            [
                F.one_hot(catalogue[:, 0] + 1, 3),
                F.one_hot(catalogue[:, 1] + 1, 3),
                bits,
            ],
            dim=1,
        )
        self.register_buffer("basis", basis.float())

    @property
    def cfg(self) -> dict[str, int]:
        return self.backbone.cfg

    def heads(self, z: Tensor) -> tuple[Tensor, Tensor]:
        fac = self.factors(z)
        return self.out(z) + fac @ self.basis.T, fac

    def forward_full(self, inputs: Tensor, burn: int = 0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        spatial, emb, h_cat = self.backbone.forward_sequence(inputs, burn)
        logits, fac = self.heads(emb)
        return logits, fac, emb, h_cat

    def forward(self, inputs: Tensor, burn: int = 0) -> Tensor:
        _, emb, _ = self.backbone.forward_sequence(inputs, burn)
        return self.heads(emb)[0]


class ScaledRecurrentValues(nn.Module):
    def __init__(self, catalogue: Tensor, **backbone_kwargs: Any) -> None:
        super().__init__()
        self.backbone = ScaledBackbone(**backbone_kwargs)
        d_model = self.backbone.cfg["d_model"]
        width = 2 * d_model
        self.q1 = nn.Linear(width, len(catalogue))
        self.q2 = nn.Linear(width, len(catalogue))
        self.value = nn.Linear(width, 1)

    def forward(self, inputs: Tensor, burn: int = 0) -> tuple[Tensor, Tensor, Tensor]:
        spatial, emb, _ = self.backbone.forward_sequence(inputs, burn)
        feat = torch.cat((spatial, emb), dim=-1)
        return self.q1(feat), self.q2(feat), self.value(feat).squeeze(-1)


def serialize_scaled_actor(actor: ScaledRecurrentActor, catalogue: np.ndarray) -> bytes:
    def arr(t: Tensor) -> bytes:
        return t.detach().cpu().contiguous().numpy().astype("<f4").tobytes()

    cfg = actor.cfg
    bb = actor.backbone
    parts: list[bytes] = [
        struct.pack(
            "<11i",
            SCALED_MAGIC,
            OBS_WIDTH,
            cfg["d_boss"],
            cfg["d_disc"],
            cfg["d_past"],
            cfg["d_model"],
            cfg["d_ffn"],
            cfg["n_blocks"],
            cfg["d_gru"],
            cfg["n_gru_layers"],
            len(catalogue),
        ),
        np.asarray(catalogue, dtype="<i4").tobytes(),
    ]
    for layer in (bb.boss1, bb.boss2, bb.disc1, bb.disc2, bb.past, bb.stem, bb.stem_ln):
        parts.append(arr(layer.weight))
        parts.append(arr(layer.bias))
    for blk in bb.blocks:
        for layer in (blk.ln, blk.fc1, blk.fc2):
            parts.append(arr(layer.weight))
            parts.append(arr(layer.bias))
    for gru, ln in zip(bb.grus, bb.gru_lns, strict=True):
        parts.append(arr(gru.weight_ih_l0))
        parts.append(arr(gru.weight_hh_l0))
        parts.append(arr(gru.bias_ih_l0))
        parts.append(arr(gru.bias_hh_l0))
        parts.append(arr(ln.weight))
        parts.append(arr(ln.bias))
    folded_w = actor.out.weight + actor.basis @ actor.factors.weight
    folded_b = actor.out.bias + actor.basis @ actor.factors.bias
    parts.append(arr(folded_w))
    parts.append(arr(folded_b))
    return b"".join(parts)


def extract_opening_bytes(parent_bin: Path) -> bytes:
    raw = parent_bin.read_bytes()
    magic, first_size, trio_size = struct.unpack_from("<3i", raw, 0)
    if magic != PHASE_MAGIC or 12 + first_size + trio_size != len(raw):
        raise ValueError(f"Expected 2-phase 0x334C5149 binary: {parent_bin}")
    return raw[12 : 12 + first_size]


def export_two_phase_scaled(
    trio_actor: ScaledRecurrentActor,
    catalogue: np.ndarray,
    output_path: Path,
    opening_bytes: bytes | None = None,
    opening_parent_bin: Path | None = None,
    opening_actor: ScaledRecurrentActor | None = None,
) -> dict[str, Any]:
    if opening_actor is not None:
        first_blob = serialize_scaled_actor(opening_actor, catalogue)
    elif opening_bytes is not None:
        first_blob = opening_bytes
    elif opening_parent_bin is not None:
        first_blob = extract_opening_bytes(opening_parent_bin)
    else:
        raise ValueError("Provide opening_actor, opening_bytes, or opening_parent_bin")
    trio_blob = serialize_scaled_actor(trio_actor, catalogue)
    payload = struct.pack("<3i", PHASE_MAGIC, len(first_blob), len(trio_blob)) + first_blob + trio_blob
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return {
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "opening_sha256": hashlib.sha256(first_blob).hexdigest(),
        "trio_sha256": hashlib.sha256(trio_blob).hexdigest(),
        "bytes": len(payload),
        "actions": len(catalogue),
        "trio_cfg": trio_actor.cfg,
    }

"""WP22 V2 Entity Self-Attention + 4-Lag Causal Motion History Recurrent Actor & Critic.

Binary Magic: 0x3F4C5149 (exact numerical parity with C# SistersScaledRecurrent.cs MagicV2).
Uses 100% existing 192-dim SistersTimingState trajectories with ZERO recollection:
  - 4-Lag causal motion history (0.08s, 0.16s, 0.32s, 0.48s) -> captures teacher's dt>=0.055s
    fast/slow boomerang disc discrimination without noisy single-step Ax!
  - 6-Token Pre-LN Multi-Head Self-Attention (Hero + 3 Boss + 2 Disc) + Pre-LN Token FFN
    -> directly models Boss x Disc trap geometry and twin-disc crossings before pooling!
  - Pre-LN ResFFN spatial fusion + Residual GRU temporal memory + Folded Factorized Action Head.
"""
from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wp18_scaled_model import (
    CONTROL,
    DISCS,
    FOES,
    HISTORY_BLOCK,
    OBS_WIDTH,
    PHASE_MAGIC,
    PreLNResFFN,
    extract_opening_bytes,
    serialize_scaled_actor,
)

ATTN_MAGIC = 0x3F4C5149
LAGS_V2 = (0.08, 0.16, 0.32, 0.48)
NUM_TOKENS = 6


def motion_history_4lag(times: np.ndarray, state: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Attach the last observation available at each V2 lag (0.08, 0.16, 0.32, 0.48s) on uninterrupted tape."""
    clock = times.astype("f8")
    all_rows = np.arange(len(clock))
    if rows is None:
        rows = all_rows
    segment = np.maximum.accumulate(
        np.where(np.r_[True, np.diff(clock) > 0.24001], all_rows, 0)
    )
    blocks = [state[rows]]
    for lag in LAGS_V2:
        j = np.searchsorted(clock, clock[rows] - lag + 0.00001, side="right") - 1
        age = clock[rows] - clock[j.clip(0)]
        known = (j >= segment[rows]) & (age <= lag + 0.24001) & ((rows - j) <= 64)
        block = np.zeros((len(rows), HISTORY_BLOCK), dtype="f4")
        block[known, :OBS_WIDTH] = state[j[known]]
        block[known, -1] = age[known]
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


class AttnRecurrentBackbone(nn.Module):
    def __init__(
        self,
        d_boss: int = 96,
        d_disc: int = 80,
        d_past: int = 48,
        d_tok: int = 96,
        n_heads: int = 4,
        d_model: int = 256,
        d_ffn: int = 512,
        n_blocks: int = 3,
        n_gru_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_tok % n_heads != 0:
            raise ValueError(f"d_tok={d_tok} must be divisible by n_heads={n_heads}")
        self.cfg = {
            "magic": ATTN_MAGIC,
            "d_boss": d_boss,
            "d_disc": d_disc,
            "d_past": d_past,
            "d_tok": d_tok,
            "n_heads": n_heads,
            "d_model": d_model,
            "d_ffn": d_ffn,
            "n_blocks": n_blocks,
            "d_gru": d_model,
            "n_gru_layers": n_gru_layers,
        }
        self.d_tok = d_tok
        self.n_heads = n_heads
        self.head_dim = d_tok // n_heads
        self.dropout = dropout

        self.boss1 = nn.Linear(36, d_boss)
        self.boss2 = nn.Linear(d_boss, d_tok)
        self.disc1 = nn.Linear(14, d_disc)
        self.disc2 = nn.Linear(d_disc, d_tok)
        self.past = nn.Linear(HISTORY_BLOCK, d_past)

        hero_in = 24 + 32 + len(LAGS_V2) * d_past
        self.hero_tok = nn.Linear(hero_in, d_tok)

        # 6-Token Pre-LN Multi-Head Self-Attention + Pre-LN Token FFN
        self.attn_ln = nn.LayerNorm(d_tok, eps=1e-5)
        self.attn_qkv = nn.Linear(d_tok, 3 * d_tok)
        self.attn_out = nn.Linear(d_tok, d_tok)
        self.tok_ffn_ln = nn.LayerNorm(d_tok, eps=1e-5)
        self.tok_ffn1 = nn.Linear(d_tok, 2 * d_tok)
        self.tok_ffn2 = nn.Linear(2 * d_tok, d_tok)

        enc_in = OBS_WIDTH + len(LAGS_V2) * d_past + 5 * d_tok
        self.stem = nn.Linear(enc_in, d_model)
        self.stem_ln = nn.LayerNorm(d_model, eps=1e-5)
        self.blocks = nn.ModuleList([PreLNResFFN(d_model, d_ffn, dropout=dropout) for _ in range(n_blocks)])
        self.grus = nn.ModuleList([nn.GRU(d_model, d_model, batch_first=True) for _ in range(n_gru_layers)])
        self.gru_lns = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-5) for _ in range(n_gru_layers)])

    def encode_spatial(self, flat_x: Tensor) -> Tensor:
        B = len(flat_x)
        raw = flat_x[:, :OBS_WIDTH]
        bosses = raw[:, FOES:DISCS].reshape(B, 3, 36)
        discs = raw[:, DISCS:CONTROL].reshape(B, 2, 14)
        b_valid = bosses[:, :, 0] != 0
        d_valid = discs[:, :, 0] != 0
        b_mask = b_valid.unsqueeze(-1).float()
        d_mask = d_valid.unsqueeze(-1).float()
        bosses_c = bosses * b_mask
        discs_c = discs * d_mask

        # 4-Lag history
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
        past_emb = (F.relu(self.past(p_clean)) * (past[:, -1:] > 0).float()).reshape(B, -1)

        # Build 6 entity tokens: [Hero, Boss0, Boss1, Boss2, Disc0, Disc1]
        hero_in = torch.cat((raw[:, :FOES], raw[:, CONTROL:OBS_WIDTH], past_emb), dim=1)
        h_tok = F.silu(self.hero_tok(hero_in)).unsqueeze(1)  # (B, 1, d_tok)
        b_tok = F.relu(self.boss2(F.silu(self.boss1(bosses_c)))) * b_mask  # (B, 3, d_tok)
        d_tok = F.relu(self.disc2(F.silu(self.disc1(discs_c)))) * d_mask  # (B, 2, d_tok)

        toks = torch.cat((h_tok, b_tok, d_tok), dim=1)  # (B, 6, d_tok)
        h_valid = torch.ones((B, 1), dtype=torch.bool, device=flat_x.device)
        tok_valid = torch.cat((h_valid, b_valid, d_valid), dim=1)  # (B, 6)
        tok_valid_f = tok_valid.unsqueeze(-1).float()

        # Pre-LN Multi-Head Self-Attention
        ln_t = self.attn_ln(toks)
        qkv = self.attn_qkv(ln_t).reshape(B, NUM_TOKENS, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # (B, H, 6, Dh)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        key_mask = ~tok_valid[:, None, None, :]  # (B, 1, 1, 6)
        scores = scores.masked_fill(key_mask, -1e9)
        attn = scores.softmax(dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout, training=True)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, NUM_TOKENS, self.d_tok)
        proj = self.attn_out(ctx)
        if self.dropout > 0.0 and self.training:
            proj = F.dropout(proj, p=self.dropout, training=True)
        toks = (toks + proj) * tok_valid_f

        # Pre-LN Token FFN
        ffn_h = F.silu(self.tok_ffn1(self.tok_ffn_ln(toks)))
        if self.dropout > 0.0 and self.training:
            ffn_h = F.dropout(ffn_h, p=self.dropout, training=True)
        toks = (toks + self.tok_ffn2(ffn_h)) * tok_valid_f

        # Exact masked Max+Sum pooling (0 when no entities active, matching C#!)
        hero_out = toks[:, 0]
        boss_out = toks[:, 1:4]
        disc_out = toks[:, 4:6]

        any_boss = b_valid.any(dim=1, keepdim=True)
        boss_max = torch.where(
            any_boss,
            boss_out.masked_fill(~b_valid.unsqueeze(-1), -1e9).amax(dim=1),
            torch.zeros_like(hero_out),
        )
        boss_sum = boss_out.sum(dim=1)

        any_disc = d_valid.any(dim=1, keepdim=True)
        disc_max = torch.where(
            any_disc,
            disc_out.masked_fill(~d_valid.unsqueeze(-1), -1e9).amax(dim=1),
            torch.zeros_like(hero_out),
        )
        disc_sum = disc_out.sum(dim=1)

        enc_in = torch.cat(
            (
                raw[:, :FOES],
                bosses_c.flatten(1),
                discs_c.flatten(1),
                raw[:, CONTROL:OBS_WIDTH],
                past_emb,
                hero_out,
                boss_max,
                boss_sum,
                disc_max,
                disc_sum,
            ),
            dim=1,
        )
        h = F.silu(self.stem_ln(self.stem(enc_in)))
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward_sequence(self, inputs: Tensor, burn: int = 0) -> tuple[Tensor, Tensor, Tensor]:
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


class AttnRecurrentActor(nn.Module):
    def __init__(self, catalogue: Tensor, **backbone_kwargs: Any) -> None:
        super().__init__()
        catalogue = catalogue.long()
        self.backbone = AttnRecurrentBackbone(**backbone_kwargs)
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


class AttnRecurrentValues(nn.Module):
    def __init__(self, catalogue: Tensor, **backbone_kwargs: Any) -> None:
        super().__init__()
        self.backbone = AttnRecurrentBackbone(**backbone_kwargs)
        d_model = self.backbone.cfg["d_model"]
        width = 2 * d_model
        self.q1 = nn.Linear(width, len(catalogue))
        self.q2 = nn.Linear(width, len(catalogue))
        self.value = nn.Linear(width, 1)

    def forward(self, inputs: Tensor, burn: int = 0) -> tuple[Tensor, Tensor, Tensor]:
        spatial, emb, _ = self.backbone.forward_sequence(inputs, burn)
        feat = torch.cat((spatial, emb), dim=-1)
        return self.q1(feat), self.q2(feat), self.value(feat).squeeze(-1)


def serialize_attn_actor(actor: AttnRecurrentActor, catalogue: np.ndarray) -> bytes:
    def arr(t: Tensor) -> bytes:
        return t.detach().cpu().contiguous().numpy().astype("<f4").tobytes()

    cfg = actor.cfg
    bb = actor.backbone
    parts: list[bytes] = [
        struct.pack(
            "<13i",
            ATTN_MAGIC,
            OBS_WIDTH,
            cfg["d_boss"],
            cfg["d_disc"],
            cfg["d_past"],
            cfg["d_tok"],
            cfg["n_heads"],
            cfg["d_model"],
            cfg["d_ffn"],
            cfg["n_blocks"],
            cfg["d_gru"],
            cfg["n_gru_layers"],
            len(catalogue),
        ),
        np.asarray(catalogue, dtype="<i4").tobytes(),
    ]
    for layer in (
        bb.boss1, bb.boss2, bb.disc1, bb.disc2, bb.past,
        bb.hero_tok, bb.attn_ln, bb.attn_qkv, bb.attn_out,
        bb.tok_ffn_ln, bb.tok_ffn1, bb.tok_ffn2,
        bb.stem, bb.stem_ln,
    ):
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


def export_two_phase_attn(
    trio_actor: AttnRecurrentActor,
    catalogue: np.ndarray,
    output_path: Path,
    opening_bytes: bytes | None = None,
    opening_parent_bin: Path | None = None,
    opening_actor: Any | None = None,
) -> dict[str, Any]:
    if opening_actor is not None:
        first_blob = (
            serialize_attn_actor(opening_actor, catalogue)
            if isinstance(opening_actor, AttnRecurrentActor)
            else serialize_scaled_actor(opening_actor, catalogue)
        )
    elif opening_bytes is not None:
        first_blob = opening_bytes
    elif opening_parent_bin is not None:
        first_blob = extract_opening_bytes(opening_parent_bin)
    else:
        raise ValueError("Provide opening_actor, opening_bytes, or opening_parent_bin")
    trio_blob = serialize_attn_actor(trio_actor, catalogue)
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

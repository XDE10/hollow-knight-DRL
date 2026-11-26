"""Integrity and architecture tests for the shipped Sisters of Battle champion."""

from __future__ import annotations

import hashlib
import json
import py_compile
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models/sisters_champion"
SCRIPTS_DIR = REPO_ROOT / "scripts/sisters"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_champion_receipts_and_optional_binary() -> None:
    """Verify Formal-100 receipts and local champion binary header/SHA-256 if present."""
    summary_path = MODEL_DIR / "summary.json"
    formal_path = MODEL_DIR / "formal100_summary.json"
    receipts_tar = MODEL_DIR / "formal100_receipts.tar.gz"
    wm_val = MODEL_DIR / "world_model/validation.json"

    for p in (summary_path, formal_path, receipts_tar, wm_val):
        assert p.is_file() and p.stat().st_size > 0, f"Missing champion artifact: {p}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selected_winner"] == "r06_best_holdout"

    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    assert formal["episodes_done"] == 100
    assert formal["wins"] == 100
    assert formal["no_hit_wins"] == 65

    # When the Release binary is present locally, verify exact SHA-256 and V2 dual-phase header
    champ_bin = MODEL_DIR / "sisters_champion.bin"
    if champ_bin.is_file():
        assert _sha256(champ_bin) == summary["winner_bin_sha256"]
        with champ_bin.open("rb") as f:
            magic, first_size, trio_size = struct.unpack("<3I", f.read(12))
            (first_magic,) = struct.unpack("<I", f.read(4))
            f.seek(12 + first_size)
            (trio_magic,) = struct.unpack("<I", f.read(4))
        assert magic == 0x334C5149  # PHASE_MAGIC
        assert first_size > 0 and trio_size > 0
        assert first_magic == 0x3F4C5149  # ATTN_MAGIC (Phase-1 opener)
        assert trio_magic == 0x3F4C5149  # ATTN_MAGIC (Phase-2 Trio actor)


def test_attn_recurrent_actor_forward_and_binary_export(tmp_path: Path) -> None:
    """Verify V2 AttnRecurrentActor forward pass and dual-phase .bin serialization."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from models import (  # type: ignore[import-not-found]
        ATTN_MAGIC,
        PHASE_MAGIC,
        AttnRecurrentActor,
        export_two_phase_attn,
    )

    torch.manual_seed(42)
    catalogue = torch.zeros((16, 3), dtype=torch.int64)
    actor = AttnRecurrentActor(
        catalogue,
        d_boss=16,
        d_disc=8,
        d_past=16,
        d_tok=32,
        n_heads=2,
        d_model=64,
        d_ffn=128,
        n_blocks=1,
        n_gru_layers=2,
        dropout=0.0,
    ).eval()

    x = torch.randn(1, 5, 964)
    with torch.no_grad():
        logits = actor(x)
    assert logits.shape == (1, 5, 16)
    assert torch.isfinite(logits).all()

    out_bin = tmp_path / "test_champion.bin"
    info = export_two_phase_attn(
        actor, np.zeros((16, 3), dtype=np.int32), out_bin, opening_actor=actor
    )
    assert out_bin.is_file() and info["bytes"] == out_bin.stat().st_size
    with out_bin.open("rb") as f:
        pmagic, fsz, tsz, amagic = struct.unpack("<4I", f.read(16))
    assert pmagic == PHASE_MAGIC and amagic == ATTN_MAGIC and fsz > 0 and tsz > 0


@pytest.mark.parametrize(
    "script_name",
    [
        "play.py",
        "models.py",
        "collect_episode.py",
        "eval_parallel.py",
        "train_offline_wm_iql.py",
        "train_online_ppo.py",
    ],
)
def test_sisters_scripts_compile(script_name: str) -> None:
    path = SCRIPTS_DIR / script_name
    assert path.is_file(), f"Missing script: {path}"
    py_compile.compile(str(path), doraise=True)


def test_sisters_play_default_configs_load_cleanly() -> None:
    from hkrl.utils.config import load_task_config, load_train_config

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import play  # type: ignore[import-not-found]

    train_cfg = load_train_config(play.DEFAULT_TRAIN_CFG)
    task_cfg = load_task_config(play.DEFAULT_TASK_CFG)

    assert train_cfg.model.name == "sisters_platform_gru"
    assert train_cfg.rollout_steps == 510
    assert task_cfg.task_id == "sisters_of_battle"
    assert task_cfg.reward.boss_damage == 0.30
    assert task_cfg.reward.player_damage == -12.0

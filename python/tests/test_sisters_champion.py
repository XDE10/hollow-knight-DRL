"""Integrity, portability, and architecture tests for Sisters of Battle champion."""

from __future__ import annotations

import hashlib
import json
import py_compile
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hkrl.utils.config import load_task_config, load_train_config, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models/sisters_champion"
SCRIPTS_DIR = REPO_ROOT / "scripts/sisters"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_all_repo_yaml_configs_load_cleanly() -> None:
    """Every YAML under configs/ must resolve with zero missing default parents."""
    yaml_files = sorted((REPO_ROOT / "configs").rglob("*.yaml"))
    assert len(yaml_files) >= 2
    for yf in yaml_files:
        raw = load_yaml(yf)
        assert isinstance(raw, dict) and raw
    train_cfg = load_train_config(REPO_ROOT / "configs/sisters_train.yaml")
    task_cfg = load_task_config(REPO_ROOT / "configs/sisters_task.yaml")
    assert train_cfg.transport.port == 5555
    assert task_cfg.task_id == "sisters_of_battle"


def test_zero_hardcoded_personal_or_machine_paths() -> None:
    """Guard against machine-specific, personal, Colab, or hardcoded-save paths leaking."""
    forbidden = re.compile(
        r"Mr\.Z|CodexArtifacts|HKRL-Handoff|/content/hk|/mnt/[a-z]/|D:\\Steam|D:/Steam"
        r"|D:\\DevTools|C:\\ffmpeg|mydrive:|WP18-ScaledDRL|HKRL-wp18|user3\.dat",
        re.IGNORECASE,
    )
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines()
    violations: list[str] = []
    for rel in tracked:
        if rel.endswith((".tar.gz", ".bin", ".pt", ".png", ".jpg")):
            continue
        if rel == "python/tests/test_sisters_champion.py":
            continue
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if forbidden.search(txt):
            violations.append(rel)
    assert not violations, f"Found hardcoded personal/machine paths in: {violations}"


def test_public_docs_contain_no_internal_team_wording() -> None:
    """Ensure public-facing README and docs remain suitable for a general open-source audience."""
    forbidden = re.compile(r"组员|小组成员|本小组|答辩|评委")
    for rel in ("README.md", "docs/setup_guide.md", "docs/project_guide.md"):
        p = REPO_ROOT / rel
        assert p.is_file(), f"Missing doc: {rel}"
        txt = p.read_text(encoding="utf-8")
        assert not forbidden.search(txt), f"Found internal team wording in {rel}"


def test_save_slot_auto_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify save slot resolution works cleanly across any slot 1-4 with zero hardcoded slot 3."""
    scripts_root = str(REPO_ROOT / "scripts")
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    from install_mod import discover_save_slot, resolve_port  # type: ignore[import-not-found]

    monkeypatch.delenv("HKRL_SAVE_SLOT", raising=False)
    monkeypatch.delenv("HKRL_PORT", raising=False)

    # Empty save dir falls back to slot 1
    assert discover_save_slot(save_dir=tmp_path) == (1, [])

    # Single existing save slot (e.g. slot 2) is auto-detected
    (tmp_path / "user2.dat").write_bytes(b"save2")
    assert discover_save_slot(save_dir=tmp_path) == (2, [2])

    # Multiple save slots prefer a modded slot if present, else lowest existing slot
    (tmp_path / "user1.dat").write_bytes(b"save1")
    assert discover_save_slot(save_dir=tmp_path) == (1, [1, 2])
    (tmp_path / "user2.modded.json").write_text("{}", encoding="utf-8")
    assert discover_save_slot(save_dir=tmp_path) == (2, [1, 2])

    # Explicit / env / runtime_conf overrides always take priority
    assert discover_save_slot(explicit=4, save_dir=tmp_path) == (4, [4])
    assert discover_save_slot(runtime_conf={"HKRL_SAVE_SLOT": "3"}, save_dir=tmp_path) == (3, [3])
    monkeypatch.setenv("HKRL_SAVE_SLOT", "4")
    assert discover_save_slot(runtime_conf={"HKRL_SAVE_SLOT": "3"}, save_dir=tmp_path) == (4, [4])

    # Port resolution honors explicit -> env -> runtime_conf -> default
    assert resolve_port(None, {"HKRL_PORT": "5601"}, default=5555) == 5601
    monkeypatch.setenv("HKRL_PORT", "5602")
    assert resolve_port(None, {"HKRL_PORT": "5601"}, default=5555) == 5602
    assert resolve_port(5603, {"HKRL_PORT": "5601"}, default=5555) == 5603


def test_steam_vdf_parser_discovers_cross_drive_libraries(tmp_path: Path) -> None:
    """Verify Steam libraryfolders.vdf parsing discovers games on arbitrary drives."""
    scripts_root = str(REPO_ROOT / "scripts")
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    from install_mod import parse_steam_library_paths  # type: ignore[import-not-found]

    vdf = tmp_path / "libraryfolders.vdf"
    vdf.write_text(
        '"libraryfolders"\n{\n  "0"\n  {\n    "path"\t\t"C:\\\\Program Files (x86)\\\\Steam"\n'
        '  }\n  "1"\n  {\n    "path"\t\t"E:\\\\SteamLibrary"\n  }\n}\n',
        encoding="utf-8",
    )
    roots = parse_steam_library_paths(vdf)
    assert len(roots) == 2
    assert any("SteamLibrary" in str(r) for r in roots)


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
    "rel_path",
    [
        "scripts/install_mod.py",
        "scripts/sisters/play.py",
        "scripts/sisters/record_demo.py",
        "scripts/sisters/models.py",
        "scripts/sisters/collect_episode.py",
        "scripts/sisters/eval_parallel.py",
        "scripts/sisters/train_offline_wm_iql.py",
        "scripts/sisters/train_online_ppo.py",
    ],
)
def test_all_user_scripts_compile(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"Missing script: {path}"
    py_compile.compile(str(path), doraise=True)


def test_assess_sisters_goals_and_demo_imports() -> None:
    """Verify key helper scripts resolve configs and cross-script imports cleanly."""
    lib_dir = str(REPO_ROOT / "scripts/sisters/_lib")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))

    import assess_sisters_goals  # type: ignore[import-not-found]
    import play  # type: ignore[import-not-found]
    import record_demo  # type: ignore[import-not-found]

    assert assess_sisters_goals.SPEC.is_file()
    assert play.DEFAULT_TRAIN_CFG.is_file()
    assert play.DEFAULT_TASK_CFG.is_file()
    assert callable(record_demo.discover_game_root)

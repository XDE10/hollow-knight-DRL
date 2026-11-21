from __future__ import annotations

import hashlib
import importlib
import json
import zipfile
from pathlib import Path

import pytest


def test_merge_preserves_complete_fight_identity_and_heldout_split(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    merge = importlib.import_module("merge_sisters_development").merge

    def game(name, seed, split):
        f = tmp_path / name
        with zipfile.ZipFile(f, "w") as z:
            z.writestr("g.npz", name)
            z.writestr(
                "manifest.json",
                json.dumps(
                    [
                        {
                            "source": f"native-development/{seed}",
                            "file": "g.npz",
                            "split": split,
                            "sha256": hashlib.sha256(name.encode()).hexdigest(),
                        }
                    ]
                ),
            )
        return f

    a, b = game("a.zip", 1, "holdout"), game("b.zip", 2, "train")
    out = tmp_path / "out.zip"
    manifest_hash = merge([a, b], out)
    with zipfile.ZipFile(out) as z:
        data = z.read("manifest.json")
        assert hashlib.sha256(data).hexdigest() == manifest_hash
        rows = json.loads(data)
        assert [g["split"] for g in rows] == ["holdout", "train"]
        assert [z.read(g["file"]) for g in rows] == [b"a.zip", b"b.zip"]
    fresh = tmp_path / "reject.zip"
    with pytest.raises(ValueError, match="Invalid development fight"):
        merge([a, a], fresh)
    assert not fresh.exists()

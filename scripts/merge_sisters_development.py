"""Merge archived native development fights while preserving their original train/held-out splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


def merge(archives: list[Path], output: Path) -> str:
    if output.exists() or not archives:
        raise ValueError("Provide source archives and a fresh destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    games: list[dict] = []
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        merged = Path(folder) / "combined.zip"
        with zipfile.ZipFile(merged, "x") as dst:
            for archive in archives:
                with zipfile.ZipFile(archive) as src:
                    for game in json.loads(src.read("manifest.json")):
                        key = game["source"]
                        if (
                            key in seen
                            or not key.startswith("native-development/")
                            or game["split"] not in ("train", "holdout")
                        ):
                            raise ValueError(
                                f"Invalid development fight or split: {key}"
                            )
                        blob = src.read(game["file"])
                        if hashlib.sha256(blob).hexdigest() != game["sha256"]:
                            raise ValueError(f"Inconsistent replay checksum: {key}")
                        seen.add(key)
                        name = f"fight-{len(games):03d}.npz"
                        dst.writestr(zipfile.ZipInfo(name), blob)
                        games.append({**game, "file": name})
            manifest = (json.dumps(games, indent=2) + "\n").encode()
            dst.writestr(zipfile.ZipInfo("manifest.json"), manifest)
        merged.rename(output)
    return hashlib.sha256(manifest).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(merge(args.archives, args.output))

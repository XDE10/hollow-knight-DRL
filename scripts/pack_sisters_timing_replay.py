"""Package completed native timing demonstrations, splitting by complete development fight."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def pack(manifests: list[Path], output: Path) -> None:
    games: dict[int, tuple[Path, dict]] = {}
    for manifest in manifests:
        for i, game in enumerate(json.loads(manifest.read_bytes())):
            meta = json.loads((manifest.parent / f"episode-{i:03d}.json").read_bytes())
            seed = meta["seed"]
            if (
                seed in games
                or game["source"] != f"native-development/{seed}"
                or meta["win"] != game["won"]
            ):
                raise ValueError("Repeated seed or unmatched development record")
            games[seed] = (manifest.parent / game["file"], game)
    entries = []
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial")
    with zipfile.ZipFile(partial, "x") as archive:
        for seed, (path, game) in sorted(games.items()):
            with np.load(path, allow_pickle=False) as raw:
                features = raw["timing_state"]
                t = raw["control_time"]
                if (
                    features.shape != (len(t), 192)
                    or not np.isfinite(features).all()
                    or not np.all(np.diff(t) > 0)
                ):
                    raise ValueError(f"Invalid decision trace in {path}")
                if not any(features[:, 24]):
                    raise ValueError(f"No observed Sisters in {path}")
                buf = io.BytesIO()
                np.savez_compressed(
                    buf,
                    **{
                        k: raw[k]
                        for k in (
                            "control_time",
                            "timing_state",
                            "controls",
                            "teacher_controls",
                            "time",
                            "events",
                            "elapsed_ticks",
                        )
                    },
                )
                data = buf.getvalue()
            name = f"fight-{seed}.npz"
            archive.writestr(name, data)
            entries.append(
                {
                    "source": game["source"],
                    "file": name,
                    "split": "holdout" if (seed % 1000) % 6 == 1 else "train",
                    "won": game["won"],
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        if {g["split"] for g in entries} != {"train", "holdout"}:
            raise ValueError("Missing fight split")
        archive.writestr("manifest.json", json.dumps(entries, indent=2) + "\n")
    partial.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pack(args.manifests, args.output)

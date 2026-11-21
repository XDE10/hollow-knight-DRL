"""Fit a class-balanced ridge risk score from frozen neural features and later native HP losses."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch
from train_sisters_timing_iql import TimingPolicy, motion_history


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fit(archives: list[Path], learner: Path, parent: Path, output: Path) -> None:
    saved = torch.load(learner, weights_only=True, map_location="cpu")
    net = TimingPolicy(saved["catalogue"], True).eval()
    net.load_state_dict(saved["actor"])
    rows = {"train": [], "holdout": []}
    seen = set()
    with torch.inference_mode():
        for path in archives:
            with zipfile.ZipFile(path) as z:
                for game in json.loads(z.read("manifest.json")):
                    raw = z.read(game["file"])
                    if game["source"] in seen or sha(raw) != game["sha256"]:
                        raise ValueError("Duplicate or changed development fight")
                    seen.add(game["source"])
                    with np.load(io.BytesIO(raw)) as d:
                        t, state = d["control_time"].astype("f8"), d["timing_state"]
                        hp = np.rint(state[:, 4] * 9)
                        losses = np.r_[0, np.cumsum(np.maximum(hp[:-1] - hp[1:], 0))]
                        gaps = np.r_[0, np.cumsum(np.diff(t) > 0.081)]
                        end = np.searchsorted(t, t + 0.72 - 1e-5).clip(0, len(t) - 1)
                        valid = (
                            (state[:, 23] > 0.5)
                            & (state[:, 9] < 0.5)
                            & (hp > 0)
                            & (gaps[end] == gaps)
                            & (abs(t[end] - t - 0.72) < 0.081)
                        )
                        start = np.flatnonzero(valid)[::3]
                        if not len(start):
                            continue
                        history = torch.from_numpy(motion_history(t, state)[start])
                        hidden = torch.cat(
                            [net.encode(batch) for batch in history.split(512)]
                        ).numpy()
                        target = (losses[end[start]] > losses[start]).astype("f8")
                        rows[game["split"]].append((game["source"], hidden, target))
    x = np.concatenate([g[1] for g in rows["train"]]).astype("f8")
    y = np.concatenate([g[2] for g in rows["train"]])
    if not 0 < np.mean(y) < 1:
        raise ValueError("Risk fitting requires observed losses and safe trajectories")
    center, scale = x.mean(0), x.std(0).clip(0.10)
    matrix = np.column_stack([np.ones(len(x)), (x - center) / scale])
    weight = np.where(y > 0, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    ridge = np.diag(np.r_[0.01, np.full(x.shape[1], 250.0)])
    beta = np.linalg.solve(
        matrix.T @ (matrix * weight[:, None]) + ridge, matrix.T @ (y * weight)
    )
    w = beta[1:] / scale
    b = beta[0] - w @ center
    cut = np.quantile(x @ w + b, 0.75)
    coeff = np.r_[b - cut, w].astype("<f4")
    blob = struct.pack("<i", 0x314B5352) + coeff.tobytes()
    held = []
    for source, features, target in rows["holdout"]:
        flagged = features @ coeff[1:] + coeff[0] >= 0
        held.append(
            {
                "source": source,
                "states": len(target),
                "observed_mask_losses_within_072s": int(target.sum()),
                "flagged": int(flagged.sum()),
                "flagged_followed_by_loss": int(target[flagged].sum()),
            }
        )
    output.mkdir(exist_ok=True, parents=True)
    (output / "risk.bin").write_bytes(blob)
    report = {
        "model": "Class-balanced ridge on frozen neural features. The decision threshold was the training 75th percentile. The labels are later observed losses under the development behavior.",
        "parent": sha(parent.read_bytes()),
        "primary_learner": sha(learner.read_bytes()),
        "readout_sha256": sha(blob),
        "input_archives": [sha(path.read_bytes()) for path in archives],
        "training_sources": [g[0] for g in rows["train"]],
        "holdout": held,
        "training_rows": len(y),
        "training_positive": int(y.sum()),
        "predefined_threshold_quantile": 0.75,
        "ridge": 250,
        "unadjusted_threshold": cut,
    }
    (output / "risk.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", nargs="+", required=True, type=Path)
    for arg in ("learner", "parent", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    torch.set_num_threads(1)
    fit(**vars(parser.parse_args()))

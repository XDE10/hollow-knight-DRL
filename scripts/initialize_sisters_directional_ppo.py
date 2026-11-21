"""Warm-start the native selector on shadow actions in development training games."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from sisters_directional_options import (
    INTERVAL_SECONDS,
    behavior_probabilities,
    menu_logits,
    menu_slots,
)
from train_sisters_directional_ppo import initial_head
from train_sisters_option_ppo import sha
from train_sisters_timing_iql import TimingPolicy, motion_history


def fit(
    x: np.ndarray,
    logits: np.ndarray,
    matches: np.ndarray,
    ids: np.ndarray,
    selected: np.ndarray,
    *,
    intercept_only: bool = False,
) -> np.ndarray:
    """Fit exact shadow suggestions expressible in the menu, with fight-level weights."""
    eligible = matches.any(1) & selected
    features = x[eligible]
    mean = features.mean(0)
    _, _, vh = np.linalg.svd(features - mean, full_matrices=False)
    basis = vh[: 0 if intercept_only else 12].T
    scale = ((features - mean) @ basis).std(0).clip(0.05)
    f = torch.from_numpy((features - mean) @ basis / scale)
    baseline = torch.from_numpy(logits[eligible])
    target = torch.from_numpy(matches[eligible].argmax(1))
    game = ids[eligible]
    weight = 1.0 / np.bincount(game)[game]
    w = torch.from_numpy(weight / weight.sum())
    weight_matrix = torch.zeros(
        (basis.shape[1] + 1, 6), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.Adam([weight_matrix], lr=0.035)
    design = torch.cat([torch.ones((len(f), 1), dtype=f.dtype), f], 1)
    for _ in range(140):
        logits_fit = baseline + design @ weight_matrix
        ce = torch.nn.functional.cross_entropy(logits_fit, target, reduction="none")
        objective = (ce * w).sum() + 0.003 * weight_matrix.square().sum()
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    raw = weight_matrix.detach().numpy().T
    slopes = (raw[:, 1:] / scale) @ basis.T
    heads = np.column_stack((raw[:, 0] - slopes @ mean, slopes)).astype("<f4")
    # Calibrate on the fit games only, before comparing unseen games. Keep exploration near T.
    initial = behavior_probabilities(logits[selected])
    adjustment = x[selected] @ heads[:, 1:].T + heads[:, 0]
    for factor in np.linspace(1.0, 0.0, 51):
        p = behavior_probabilities(logits[selected] + factor * adjustment)
        kl = (initial * (np.log(initial.clip(1e-12)) - np.log(p.clip(1e-12)))).sum(1)
        if kl.mean() <= 0.12 and np.quantile(kl, 0.95) <= 0.45:
            return heads * np.float32(factor)
    raise ValueError("Could not initialize a bounded exploration policy")


def metrics(p: np.ndarray, matches: np.ndarray, primary: np.ndarray) -> dict:
    i = np.arange(len(p))
    chosen = p.argmax(1)
    old = matches[i, primary]
    new = matches[i, chosen]
    supported = matches.any(1)
    nll = -np.log((p * matches).sum(1)[supported].clip(1e-12)).mean()
    return {
        "decisions": len(p),
        "expressible": int(matches.any(1).sum()),
        "exact": int(new.sum()),
        "baseline_exact": int(old.sum()),
        "gained": int((new & ~old).sum()),
        "lost": int((old & ~new).sum()),
        "alternate": int((chosen != primary).sum()),
        "matched_nll": float(nll),
        "primary_probability": float(p[i, primary].mean()),
    }


def initialize(data: Path, learner: Path, risk: Path, output: Path) -> None:
    torch.set_num_threads(1)
    ck = torch.load(learner, weights_only=True, map_location="cpu")
    cat = ck["catalogue"].numpy()
    actor = TimingPolicy(ck["catalogue"], True).eval().requires_grad_(False)
    actor.load_state_dict(ck["actor"])
    initial = initial_head(risk)
    risk_weights = np.frombuffer(initial, "<f4", offset=8, count=193).astype("f8")
    neutral = np.zeros((6, 193), "f8")
    chunks, sources = [], []
    with zipfile.ZipFile(data) as z, torch.inference_mode():
        for g in json.loads(z.read("manifest.json")):
            if g["split"] != "train":
                continue
            body = z.read(g["file"])
            if hashlib.sha256(body).hexdigest() != g["sha256"]:
                raise ValueError("Changed shadow trajectory")
            with np.load(io.BytesIO(body)) as d:
                if "teacher_acted" not in d or "teacher_controls" not in d:
                    continue
                t, s = d["control_time"], d["timing_state"]
                student = ~d["teacher_acted"].astype(bool) & (s[:, 23] > 0.5)
                segment = np.maximum.accumulate(
                    np.where(
                        np.r_[True, (~student[:-1]) | (np.diff(t) > 0.24)],
                        np.arange(len(t)),
                        0,
                    )
                )
                ix = np.flatnonzero(
                    student & (s[:, 9] <= 0.5) & (t - t[segment] >= 2.0)
                )
                if not len(ix):
                    continue
                h = torch.from_numpy(motion_history(t, s)[ix])
                emb = torch.cat([actor.encode(b) for b in h.split(384)])
                scores = actor.heads(emb)[0].numpy()
                x = emb.numpy().astype("f8")
                slots, primary = menu_slots(scores, cat, s[ix])
                active = (x @ risk_weights[1:] + risk_weights[0] >= 0) & (
                    (slots >= 0).sum(1) >= 2
                )
                keep = []
                unlock = np.float32(0)
                for j in np.flatnonzero(active):
                    if t[ix[j]] >= unlock:
                        keep.append(j)
                        unlock = t[ix[j]] + np.float32(INTERVAL_SECONDS)
                if not keep:
                    continue
                slots, primary, x = slots[keep], primary[keep], x[keep]
                target = d["teacher_controls"][ix[keep]]
                matches = (slots >= 0) & (cat[slots] == target[:, None]).all(2)
                chunks.append(
                    (
                        x,
                        menu_logits(x, neutral, slots, primary),
                        matches,
                        primary,
                        np.full(len(x), len(sources), dtype="i8"),
                    )
                )
                sources.append(g["source"])
    x, prior, matches, primary, ids = (
        np.concatenate(a) for a in zip(*chunks, strict=True)
    )
    internal = np.array(
        [
            int.from_bytes(hashlib.sha256(k.encode()).digest()[:4], "little") % 4 == 0
            for k in sources
        ]
    )[ids]
    if len(set(ids[internal])) < 4 or len(set(ids[~internal])) < 8:
        raise ValueError("Insufficient development games for a held-fight check")
    heads = fit(x, prior, matches, ids, ~internal)
    constant = fit(x, prior, matches, ids, ~internal, intercept_only=True)
    baseline = metrics(
        behavior_probabilities(prior[internal]), matches[internal], primary[internal]
    )
    intercept = metrics(
        behavior_probabilities(prior[internal] + constant[:, 0]),
        matches[internal],
        primary[internal],
    )
    held = metrics(
        behavior_probabilities(
            prior[internal] + x[internal] @ heads[:, 1:].T + heads[:, 0]
        ),
        matches[internal],
        primary[internal],
    )
    selected = (
        held["matched_nll"] + 0.03
        < min(intercept["matched_nll"], baseline["matched_nll"])
        and held["gained"] >= held["lost"]
    )
    # The internal check selects an initialization only. All later reinforcement gradients use real returns.
    final = (
        fit(x, prior, matches, ids, np.ones_like(internal))
        if selected
        else neutral.astype("<f4")
    )
    result = initial[: 8 + 193 * 4] + final.tobytes()
    report = {
        "scope": "Development training partition only. Shadow actions were not executed. No future damage filter.",
        "archive_sha256": sha(data),
        "learner_sha256": sha(learner),
        "risk_sha256": sha(risk),
        "games": len(sources),
        "sources": sources,
        "internal_test_games": len(set(ids[internal])),
        "internal_test": held,
        "uniform_initialization": baseline,
        "constant_initialization": intercept,
        "selected": bool(selected),
        "sampler_sha256": hashlib.sha256(result).hexdigest(),
        "decisions": len(x),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        out = Path(temp) / "initialization"
        out.mkdir()
        (out / "sampler.bin").write_bytes(result)
        (out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        out.rename(output)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "learner", "risk", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    initialize(**vars(parser.parse_args()))

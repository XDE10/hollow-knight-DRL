"""Audit neural spell requests against executed teacher inputs, observed SOUL and clock support."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from train_sisters_timing_iql import TimingPolicy, motion_history


def clock_segments(t: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Bound every before/after comparison to a single uninterrupted teacher segment."""
    resets = np.r_[True, (np.diff(t) > 0.081) | (~eligible[:-1]) | (~eligible[1:])]
    return resets.cumsum()


def audit_keys(
    t: np.ndarray,
    s: np.ndarray,
    key: np.ndarray,
    teacher: np.ndarray,
    actual_choice: np.ndarray,
    cast_choice: np.ndarray,
    other_choice: np.ndarray,
) -> dict[str, int]:
    actual = (key[:, 2] & 16) != 0
    requested = (actual_choice[:, 2] & 16) != 0
    live = teacher & (s[:, 23] > 0.5)
    seg = clock_segments(t, live)
    contiguous = np.r_[False, seg[1:] == seg[:-1]] & live
    eligible = contiguous & ~np.roll(actual, 1)
    onsets = eligible & actual
    pool = eligible & ~actual & requested
    soul = np.rint(s[:, 5] * 99).astype(int)
    ready = (s[:, 11] > 0.5) & (soul >= 33)
    counts: Counter[str] = Counter()
    counts.update(
        {
            "teacher_rows": int(contiguous.sum()),
            "observed_cast_starts": int(onsets.sum()),
            "network_same_tick": int((onsets & requested).sum()),
            "extra_request_ticks": int(pool.sum()),
            "extra_not_ready": int((pool & ~ready).sum()),
            "extra_non_down": int((pool & (actual_choice[:, 1] != -1)).sum()),
            "extra_with_noncast_horizontal_restored": int(
                (pool & (other_choice[:, 0] == key[:, 0])).sum()
            ),
        }
    )
    for i in np.flatnonzero(onsets):
        # Left observations precede the teacher's intervention. Right observations
        # already include its actions/physics; quantify the asymmetry explicitly.
        for side, lo, hi in (
            ("pre", t[i] - 0.0801, t[i] - 0.001),
            ("post", t[i] + 0.001, t[i] + 0.0801),
        ):
            rows = np.arange(
                np.searchsorted(t, lo), np.searchsorted(t, hi, side="right")
            )
            usable = rows[(seg[rows] == seg[i]) & eligible[rows] & ready[rows]]
            counts[side + "_has_eligible_tick"] += int(len(usable) > 0)
            counts[side + "_has_request"] += int(requested[usable].any())
        # The old 192-value tape omits the spell FSM. Only report a compatible
        # debit, bounded by other casts, focus requests, controller switches and gaps.
        stop = np.searchsorted(t, t[i] + 0.4001, side="right")
        rows = np.arange(i + 1, stop)
        if not len(rows) or t[rows[-1]] - t[i] < 0.319 or (seg[rows] != seg[i]).any():
            counts["outcome_window_missing"] += 1
            continue
        if onsets[rows].any() or (key[i:stop, 2] & 32).any():
            counts["outcome_window_ambiguous"] += 1
            continue
        counts["outcome_window_complete"] += 1
        delta = np.diff(soul[i:stop])
        debit = np.flatnonzero(delta <= -22)
        if not len(debit):
            counts["no_matching_debit"] += 1
            continue
        counts["with_matching_debit"] += 1
        down = key[i, 1] == -1
        counts["debit_down" if down else "debit_other"] += 1
        matched_aim = cast_choice[i, 1] == key[i, 1]
        matched_move = cast_choice[i, 0] == key[i, 0]
        counts["debit_oracle_aim"] += int(matched_aim)
        counts["debit_oracle_aim_move"] += int(matched_aim and matched_move)
        counts["debit_actual_aim_move"] += int(
            requested[i] and matched_aim and matched_move
        )
        if not requested[i]:
            counts["debit_missed"] += 1
            counts["debit_missed_oracle_aim_move"] += int(matched_aim and matched_move)
    return dict(counts)


def audit(archives: list[Path], model: Path) -> dict:
    torch.set_num_threads(1)
    saved = torch.load(model, weights_only=True, map_location="cpu")
    cat = saved["catalogue"]
    cast = (cat[:, 2] & 16) != 0
    actor = TimingPolicy(cat, history=True).eval()
    actor.load_state_dict(saved["actor"])
    seen: set[str] = set()
    rows = []
    total: Counter[str] = Counter()
    with torch.inference_mode():
        for path in archives:
            with zipfile.ZipFile(path) as z:
                for game in json.loads(z.read("manifest.json")):
                    source = game["source"]
                    if source in seen or not source.startswith("native-development/"):
                        raise ValueError("Expected unique development fights")
                    seen.add(source)
                    if not game["won"] or game["split"] != "holdout":
                        continue
                    raw = z.read(game["file"])
                    if hashlib.sha256(raw).hexdigest() != game["sha256"]:
                        raise ValueError("Development recording differs from manifest")
                    with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                        if "teacher_acted" not in f:
                            continue
                        t, s, k, teach = (
                            f[n]
                            for n in (
                                "control_time",
                                "timing_state",
                                "controls",
                                "teacher_acted",
                            )
                        )
                        if teach.dtype != np.bool_ or not np.array_equal(
                            k[teach], f["teacher_controls"][teach]
                        ):
                            raise ValueError("Need actual teacher assignments")
                        if not (teach & (s[:, 23] > 0.5)).any():
                            continue
                        states = torch.from_numpy(motion_history(t, s))
                        scores = torch.cat(
                            [actor(batch)[0] for batch in states.split(256)]
                        )
                        actual = cat[scores.argmax(1)]
                        spell = cat[scores.masked_fill(~cast, -torch.inf).argmax(1)]
                        plain = cat[scores.masked_fill(cast, -torch.inf).argmax(1)]
                        counts = audit_keys(
                            t, s, k, teach, actual.numpy(), spell.numpy(), plain.numpy()
                        )
                    total.update(counts)
                    rows.append({"source": source, "counts": counts})
    return {
        "scope": "Neural requests on observed teacher histories in won held-out development fights. SOUL loss of at least 22 within 0.40 game seconds is a proxy. A positive result does not identify a counterfactual cast or damage prevented.",
        "learner_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "counts": dict(total),
        "per_fight": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", type=Path, nargs="+", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(
        json.dumps(audit(args.archives, args.model), indent=2) + "\n"
    )

"""Select real combat transitions for a small update to the nine-win primitive policy."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from sisters_control_blocks import neural_block_starts
from train_sisters_champion_router import PRESENCE, WORLD
from train_sisters_option_ppo import verified_events
from train_sisters_timing_iql import motion_history


def select_game(tape, rng):
    s, t = tape["timing_state"], tape["control_time"]
    events = verified_events(
        t, s[:, 4] * 9, tape["time"], tape["elapsed_ticks"], tape["events"]
    )
    terminals = np.flatnonzero(events[:, 6:].any(1))
    if len(terminals) != 1:
        return None
    terminal = terminals[0]
    if terminal < 1:
        return None
    t, s, events = t[: terminal + 1], s[: terminal + 1], events[: terminal + 1]
    ix = np.arange(len(t))
    stop = np.minimum(np.searchsorted(t, t + 0.8), terminal)
    gap = np.cumsum(np.r_[0, np.diff(t) > 0.24])
    eligible = (s[:, 23] > 0.5) & (s[:, 9] < 0.5) & (ix < stop) & (gap == gap[stop])
    if not eligible.any():
        return None
    vulnerable = eligible & (
        ((s[:, 24] > 0.5) & (abs(s[:, 25]) < 0.48) & (abs(s[:, 26]) < 0.8))
        | ((s[:, 132] > 0.5) & (abs(s[:, 133]) < 0.6) & (abs(s[:, 134]) < 0.42))
    )
    masks = events[:, 1:5].sum(1)
    hurt = np.flatnonzero(masks > 0)
    lead = np.searchsorted(
        t, t[hurt, None] - np.array([0.16, 0.32, 0.52, 0.72])
    ).flatten()

    def sample(valid, count):
        indices = np.flatnonzero(valid)
        return rng.choice(indices, min(count, len(indices)), replace=False)

    rows = np.unique(
        np.r_[sample(eligible, 144), sample(vulnerable, 112), lead[eligible[lead]]]
    )
    starts = neural_block_starts(t)
    gate = starts[np.searchsorted(starts, rows, side="right") - 1]
    hist = motion_history(t, s)
    unique_gate, gate_index = np.unique(gate, return_inverse=True)
    summary = np.vstack((np.zeros(8), events.cumsum(0)))
    # A decision owns the intervals after its input has arrived. Include the terminal once.
    end = stop[rows]
    sums = summary[end + (end == terminal)] - summary[rows]
    world_index = np.array(PRESENCE)
    valid = (
        (world_index < 0)
        | ((s[rows][:, world_index] > 0.5) & (s[end][:, world_index] > 0.5))
    ) & (end[:, None] != terminal)
    has_teacher = "teacher_controls" in tape and "teacher_acted" in tape
    acted = (
        tape["teacher_acted"][: len(t)].astype(bool)
        if has_teacher
        else np.zeros(len(t), bool)
    )
    totals = np.r_[0, acted.cumsum()]
    return {
        "rows": rows,
        "clock": t[rows],
        "history": hist[rows],
        "gate_history": hist[unique_gate],
        "gate_index": gate_index,
        "gate_trio": s[gate, 23] > 0.5,
        "elapsed": t[end] - t[rows],
        "terminal": end == terminal,
        "events": sums.astype("f4"),
        "world_delta": s[end][:, WORLD] - s[rows][:, WORLD],
        "world_mask": valid,
        "issued": tape["controls"][rows],
        "label": tape["teacher_controls"][rows]
        if has_teacher
        else np.full((len(rows), 3), -99, "i4"),
        "has_label": np.full(len(rows), has_teacher),
        "student_only": (totals[end + (end == terminal)] == totals[rows])
        if has_teacher
        else np.ones(len(rows), bool),
    }


def pack(source: Path, dest: Path):
    if dest.exists():
        raise FileExistsError(dest)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    dest.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        new = Path(tmp) / "windows.zip"
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(new, "w") as out:
            summary = []
            for game in json.loads(src.read("manifest.json")):
                if game["split"] not in ("train", "holdout"):
                    raise ValueError("Unregistered development split")
                seed = int(game["source"].split("/")[-1])
                if game["source"] != f"native-development/{seed}":
                    raise ValueError("Unexpected episode origin")
                raw = src.read(game["file"])
                if hashlib.sha256(raw).hexdigest() != game["sha256"]:
                    raise ValueError("Changed development tape")
                with np.load(io.BytesIO(raw)) as tape:
                    data = select_game(tape, np.random.default_rng(seed + 9432))
                if data is None:
                    continue
                stream = io.BytesIO()
                np.savez_compressed(stream, **data)
                content = stream.getvalue()
                name = f"{seed}.npz"
                out.writestr(name, content)
                summary.append(
                    {
                        "source": game["source"],
                        "split": game["split"],
                        "rows": len(data["rows"]),
                        "file": name,
                        "won": game["won"],
                        "source_sha256": game["sha256"],
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            if len({g["source"] for g in summary}) != len(summary):
                raise ValueError("Duplicate trajectories")
            out.writestr(
                "manifest.json", json.dumps({"source_sha256": digest, "games": summary})
            )
        new.rename(dest)
    print(len(summary), sum(g["rows"] for g in summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    pack(**vars(parser.parse_args()))

"""Object-aligned training windows from game snapshots and timestamped controller inputs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

HORIZON = 5
BINS = 4
CONTROL = 12
TARGET = [6, 7, 10, 11, 12, 14, 15]
PLAYER_TARGET = [0, 1, 2, 3, 4, 6]
STATE_BUCKETS = 127


def buckets(raw: np.ndarray) -> np.ndarray:
    v = raw.astype(np.int64)
    return (((v >> 7) ^ (v >> 18) ^ v) % STATE_BUCKETS + 1).astype(np.int64)


def objects(values: np.ndarray) -> np.ndarray:
    v = values.copy()
    v[:, 0] = 0  # identity aligns targets within each fight
    v[:, 1:3] /= 5
    v[:, 3:6] = 0  # categorical embeddings are carried separately
    for i in (6, 14):
        v[:, i] = (v[:, i] - 30) / 12
    for i in (7, 15):
        v[:, i] = (v[:, i] - 7.4) / 9
    v[:, 8:10] /= [12, 9]
    v[:, 10:12] /= [45, 55]
    v[:, 12:14] /= 750
    v[:, 16:18] /= [5, 5]
    v[:, 20:] /= [1, 10, 10, 50]
    return v.clip(-5, 5)


def players(raw: np.ndarray) -> np.ndarray:
    v = raw.copy()
    v[:, :4] = (v[:, :4] - [30, 7.4, 0, 0]) / [12, 9, 20, 25]
    v[:, 4:8] /= [9, 9, 99, 99]
    v[:, [25, 27, 28, 29]] = 0
    v[:, 26] /= 127
    v[:, 31] /= 511
    return v.clip(-5, 5)


def controls(raw: np.ndarray) -> np.ndarray:
    m, aim, b = raw.T
    return np.stack(
        [
            m < 0,
            m > 0,
            aim < 0,
            aim > 0,
            (b & 3) != 0,
            (b & 4) != 0,
            (b & 8) != 0,
            (b & 16) != 0,
            (b & 32) != 0,
            (b & 64) != 0,
            (b & 128) != 0,
            (b & 256) != 0,
        ],
        axis=-1,
    ).astype(np.float32)


def interval_actions(
    t: np.ndarray, end: np.ndarray, ct: np.ndarray, raw: np.ndarray
) -> np.ndarray:
    """Integrate the complete piecewise-constant input in four equal subintervals."""
    keys = controls(raw)
    cs = np.concatenate(
        (np.zeros((1, CONTROL)), np.cumsum(keys[:-1] * np.diff(ct)[:, None], 0))
    )
    knots = t[:, None] + (end - t)[:, None] * np.linspace(0, 1, BINS + 1)[None, :]
    idx = (np.searchsorted(ct, knots, side="right") - 1).clip(0, len(ct) - 1)
    area = cs[idx] + keys[idx] * np.maximum(0, knots - ct[idx])[:, :, None]
    action = (
        np.diff(area, axis=1) / np.maximum(0.0001, np.diff(knots, axis=1))[:, :, None]
    )
    return np.concatenate(
        (action.clip(0, 1).reshape(len(t), -1), ((end - t) / 0.20)[:, None].clip(0, 8)),
        1,
    ).astype(np.float32)


class WorldReplay:
    def __init__(self, folder: Path, device: str):
        entries = json.loads((folder / "manifest.json").read_text())
        max_entities = max(int(e["max_entities"]) for e in entries)
        parts: dict[str, list[np.ndarray]] = {}
        self.starts: dict[str, list[np.ndarray]] = {"train": [], "holdout": []}
        self.episodes: dict[str, int] = {"train": 0, "holdout": 0}
        offset = 0
        for entry in entries:
            with np.load(folder / entry["file"], allow_pickle=False) as arc:
                raw_t = arc["time"]
                ct = arc["control_time"]
                # The final action has real events, even when the return scene makes its
                # next spatial observation unavailable. Supervise those events separately.
                t = np.r_[
                    raw_t, raw_t[-1] + 0.02 * float(arc["elapsed_ticks"][-1])
                ].astype(np.float32)
                observed = np.ones(len(t), bool)
                observed[-1] = False
                n = len(t)
                hero = arc["player"]
                pl = players(np.concatenate([hero, hero[-1:]]))
                raw = arc["entities"]
                ptr = np.r_[arc["entity_ptr"], len(raw)]
                obj = np.zeros((n, max_entities, 24), np.float32)
                eid = np.full((n, max_entities), -1, np.int32)
                obj_cat = np.zeros((n, max_entities, 3), np.int16)
                all_obj = objects(raw)
                cats = buckets(raw[:, 3:6])
                for i, (l, r) in enumerate(itertools.pairwise(ptr)):
                    k = r - l
                    obj[i, :k] = all_obj[l:r]
                    eid[i, :k] = raw[l:r, 0]
                    obj_cat[i, :k] = cats[l:r]
                native = np.searchsorted(ct, t, side="right") - 1
                hi = np.minimum(np.arange(n) + 1, n - 1)
                act = interval_actions(t, t[hi], ct, arc["controls"])
                hist = (native[:, None] + np.arange(-3, 1)[None, :]).clip(
                    0, len(ct) - 1
                )
                sf = arc["student_features"][hist]
                b = arc["controls"][hist[:, -1]].astype(int)
                labels = (
                    (b[:, 0] + 1) * 4 + 2 * ((b[:, 2] & 3) != 0) + ((b[:, 2] & 4) != 0)
                )
                ev = np.concatenate([arc["events"], np.zeros((1, 8), np.float32)])
                occupancy = np.zeros((n, 60), np.float32)
                kind = raw[:, 1].astype(int)
                group = np.where(
                    kind == 1,
                    0,
                    np.where(
                        (kind == 3) & (raw[:, 2] == 1), 1, np.where(kind == 4, 2, 3)
                    ),
                )
                xzone = np.digitize(raw[:, 14], [20, 26, 34, 40])
                yz = np.digitize(raw[:, 15], [8.8, 13])
                frame_ids = np.repeat(np.arange(n), np.diff(ptr))
                occupancy[frame_ids, group * 15 + xzone * 3 + yz] = 1
                # Cache alignment with each of the next H observations once, by local object ID.
                matches = np.full((n, HORIZON, max_entities), -1, np.int16)
                for h in range(HORIZON):
                    count = n - h - 1
                    if count <= 0:
                        break
                    prev = eid[:count]
                    nxt = eid[h + 1 :]
                    for first in range(0, count, 512):
                        a = prev[first : first + 512]
                        other = nxt[first : first + 512]
                        eq = (a[:, :, None] == other[:, None, :]) & (a[:, :, None] >= 0)
                        matches[first : first + len(a), h] = np.where(
                            eq.any(-1), eq.argmax(-1), -1
                        )
                block = {
                    "player": pl,
                    "obj": obj,
                    "cat": obj_cat,
                    "mask": eid >= 0,
                    "action": act,
                    "event": ev,
                    "match": matches,
                    "student": sf,
                    "label": labels,
                    "occupancy": occupancy,
                    "observed": observed,
                    "phase": (
                        np.sum(np.isclose(obj[:, :, 1], 0.2) & (eid >= 0), 1) >= 2
                    ),
                    "outcome": np.full(n, float(entry["won"]), np.float32),
                }
                for key, value in block.items():
                    parts.setdefault(key, []).append(value)
                # Exclude missing control coverage, extreme host stalls and all cross-game windows.
                last = max(0, n - HORIZON)
                valid = np.arange(last)
                gaps = np.diff(t)
                if last:
                    ok = (
                        np.lib.stride_tricks.sliding_window_view(gaps, HORIZON).max(-1)
                        <= 0.55
                    )
                    covered = (t[HORIZON:] <= ct[-1] + 0.06) | (
                        (valid + HORIZON == n - 1) & (t[:last] <= ct[-1])
                    )
                    valid = valid[ok & (native[:last] >= 3) & covered]
                self.starts[entry["split"]].append(valid + offset)
                self.episodes[entry["split"]] += 1
                offset += n
        self.data = {
            k: torch.as_tensor(np.concatenate(v), device=device)
            for k, v in parts.items()
        }
        self.indices = {
            k: torch.as_tensor(np.concatenate(v), device=device)
            for k, v in self.starts.items()
        }
        self.device = device
        self.offset = torch.arange(HORIZON, device=device)
        train = self.indices["train"]
        interesting = (
            self.data["event"][train[:, None] + self.offset][:, :, 1:].amax(dim=(1, 2))
            > 0
        )
        self.rare = train[interesting]

    def sample(self, split: str, size: int) -> torch.Tensor:
        pool = self.indices[split]
        if split == "train" and len(self.rare):
            rare_count = size // 4
            return torch.cat(
                (
                    pool[
                        torch.randint(
                            len(pool), (size - rare_count,), device=self.device
                        )
                    ],
                    self.rare[
                        torch.randint(len(self.rare), (rare_count,), device=self.device)
                    ],
                )
            )
        return pool[torch.randint(len(pool), (size,), device=self.device)]

    def batch(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        d = self.data
        window = ids[:, None] + self.offset
        after = window + 1
        match = d["match"][ids].long()
        future = d["obj"][after][..., TARGET]
        obj_target = future.gather(
            2, match.clamp_min(0)[..., None].expand(-1, -1, -1, len(TARGET))
        )
        fsm = d["cat"][after][:, :, :, 2].long().gather(2, match.clamp_min(0))
        return {
            "player": d["player"][ids],
            "obj": d["obj"][ids],
            "cat": d["cat"][ids].long(),
            "mask": d["mask"][ids],
            "action": d["action"][window],
            "event": d["event"][window],
            "player_target": d["player"][after][..., PLAYER_TARGET],
            "obj_target": obj_target,
            "obj_present": match >= 0,
            "fsm": fsm,
            "student": d["student"][ids],
            "observed": d["observed"][after],
            "occupancy": d["occupancy"][after],
            "label": d["label"][ids],
            "won": d["outcome"][ids],
            "phase": d["phase"][ids],
        }

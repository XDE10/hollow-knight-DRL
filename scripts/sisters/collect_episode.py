#!/usr/bin/env python3
"""Single-episode native actor collector with optional Radiant 1-hit early stop."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (_HERE / "_lib", _REPO / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_sisters_native_labels as collector  # noqa: E402
from hkrl import protocol  # noqa: E402


class ObservedEnv(collector.HKRLEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.folder = Path(sys.argv[sys.argv.index("--output") + 1]).parent
        self.initial = {
            "episode_id": info["episode_id"],
            "seed": seed,
            "initial_masks": int(obs["player"][4]),
            "initial_max_masks": int(obs["player"][5]),
            "initial_soul": int(obs["player"][6]),
        }
        self.radiant_max_trio_hits = int(os.environ.get("HKRL_RADIANT_MAX_TRIO_HITS", "0"))
        self.p1_only = os.environ.get("HKRL_P1_ONLY", "0") == "1"
        self.save_step_obs = os.environ.get("HKRL_SAVE_STEP_OBS", "0") == "1"
        self.trio_started = False
        self.trio_steps = 0
        self.trio_hits = 0
        self.samples = []
        if self.save_step_obs:
            self.record(obs, info)
        (self.folder / "fight-started.json").write_text(json.dumps(self.initial) + "\n", encoding="utf-8")
        return obs, info

    def record(self, obs, info):
        self.samples.append(
            (
                obs["global"].copy(),
                obs["player"].copy(),
                obs["entities"].copy(),
                obs["entity_mask"].copy(),
                int(info.get("elapsed_ticks", 0)),
            )
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.save_step_obs:
            self.record(obs, info)
        if (self.radiant_max_trio_hits > 0 or self.p1_only) and not (terminated or truncated):
            if not self.trio_started:
                visible = obs["entities"][obs["entity_mask"].astype(bool)]
                bosses = visible[visible[:, 1] == protocol.EntityType.BOSS]
                if len(bosses) >= 3:
                    self.trio_started = True
                    if self.p1_only:
                        truncated = True
                        (self.folder / "p1-cleared.json").write_text(
                            json.dumps({"p1_cleared": True, "trio_detected": True}) + "\n",
                            encoding="utf-8",
                        )
            if self.trio_started and self.radiant_max_trio_hits > 0 and not (terminated or truncated):
                self.trio_steps += 1
                for ev in info.get("reward_events", ()):
                    if ev.kind == protocol.RewardEventKind.DAMAGE_TAKEN:
                        self.trio_hits += int(ev.amount)
                if self.trio_steps >= 2 and self.trio_hits >= self.radiant_max_trio_hits:
                    truncated = True
                    (self.folder / "radiant-stop.json").write_text(
                        json.dumps(
                            {
                                "radiant_max_trio_hits": self.radiant_max_trio_hits,
                                "trio_hits": self.trio_hits,
                                "trio_steps": self.trio_steps,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
        if (terminated or truncated) and self.save_step_obs and self.samples:
            names = ("global_state", "player", "entities", "entity_mask", "elapsed_ticks")
            np.savez_compressed(
                self.folder / "step-observations.npz",
                **{name: np.stack(column) for name, column in zip(names, zip(*self.samples), strict=True)},
            )
        return obs, reward, terminated, truncated, info


collector.HKRLEnv = ObservedEnv
collector.main()

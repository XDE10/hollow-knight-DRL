from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "scripts"))
import collect_sisters_native_labels as collector
from hkrl import protocol


class ObservedEnv(collector.HKRLEnv):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.folder = Path(sys.argv[sys.argv.index("--output") + 1]).parent
        self.initial = {
            "episode_id": info["episode_id"], "seed": seed,
            "initial_masks": int(obs["player"][4]),
            "initial_max_masks": int(obs["player"][5]),
            "initial_soul": int(obs["player"][6]),
        }
        self.radiant_max_trio_hits = int(os.environ.get("HKRL_RADIANT_MAX_TRIO_HITS", "0"))
        self.trio_started = False
        self.trio_steps = 0
        self.trio_hits = 0
        self.samples = []
        self.record(obs, info)
        (self.folder / "fight-started.json").write_text(json.dumps(self.initial) + "\n")
        return obs, info

    def record(self, obs, info):
        self.samples.append((
            obs["global"].copy(), obs["player"].copy(),
            obs["entities"].copy(), obs["entity_mask"].copy(),
            int(info.get("elapsed_ticks", 0)),
        ))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.record(obs, info)
        if self.radiant_max_trio_hits > 0 and not (terminated or truncated):
            if not self.trio_started:
                visible = obs["entities"][obs["entity_mask"].astype(bool)]
                bosses = visible[visible[:, 1] == protocol.EntityType.BOSS]
                if len(bosses) >= 3:
                    self.trio_started = True
            if self.trio_started:
                self.trio_steps += 1
                for ev in info.get("reward_events", ()):
                    if ev.kind == protocol.RewardEventKind.DAMAGE_TAKEN:
                        self.trio_hits += int(ev.amount)
                if self.trio_steps >= 2 and self.trio_hits >= self.radiant_max_trio_hits:
                    truncated = True
                    (self.folder / "radiant-stop.json").write_text(
                        json.dumps({
                            "radiant_max_trio_hits": self.radiant_max_trio_hits,
                            "trio_hits": self.trio_hits,
                            "trio_steps": self.trio_steps,
                        }) + "\n",
                        encoding="utf-8",
                    )
        if terminated or truncated:
            names = ("global_state", "player", "entities", "entity_mask", "elapsed_ticks")
            np.savez_compressed(
                self.folder / "step-observations.npz",
                **{name: np.stack(column) for name, column in zip(names, zip(*self.samples), strict=True)},
            )
        return obs, reward, terminated, truncated, info


collector.HKRLEnv = ObservedEnv
collector.main()


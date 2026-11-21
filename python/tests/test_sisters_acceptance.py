from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def test_successive_frozen_cohorts_and_damage_rules(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("assess_sisters_goals")
    actor, dll = tmp_path / "actor", tmp_path / "mod"
    actor.write_bytes(b"test actor")
    dll.write_bytes(b"test binary")
    first = mod.freeze(1, actor, dll, 1100, None)

    def games(plan, clean, wins):
        return [
            dict(
                seed=s,
                win=i < wins,
                no_hit_win=i < clean,
                minimum_hp=9 if i < clean else 8,
                hp=9 if i < wins else 0,
                decisions=500,
                truncated=False,
                damage_taken={} if i < clean else {"BOSS_BODY": 1},
                speed_counts={"1.0": 80, "0.5": 1},
                policy_sha256=plan["policy_sha256"],
                mod_sha256=plan["mod_sha256"],
            )
            for i, s in enumerate(mod.seed_range(plan))
        ]

    valid = games(first, 7, 98)
    seeded = mod.freeze(1, actor, dll, 1100, None, verify_unity_seed=True)
    with pytest.raises(ValueError, match="Unity reset seed"):
        mod.score(seeded, valid)
    assert mod.score(seeded, [dict(g, unity_reset_seed=g["seed"]) for g in valid])["passed"]
    with pytest.raises(ValueError, match="Int32"):
        mod.freeze(1, actor, dll, 2**31 - 10, None, verify_unity_seed=True)

    assert mod.score(first, valid)["next_goal"] == 2
    assert not mod.score(first, games(first, 7, 97))["passed"]
    assert not mod.score(first, valid[:-1])["passed"]
    report = tmp_path / "goal-1.json"
    report.write_text(json.dumps({"plan": first, "episodes": valid}))
    with pytest.raises(ValueError, match="new cohort"):
        mod.freeze(2, actor, dll, 1199, report)
    second = mod.freeze(2, actor, dll, 1200, report)
    assert mod.score(second, games(second, 95, 98))["passed"]
    assert not mod.score(second, games(second, 94, 100))["passed"]
    assert not mod.score(second, games(second, 95, 97))["passed"]
    invalid = games(second, 95, 98)
    invalid[0]["damage_taken"] = {"SPIKES": 1}
    with pytest.raises(ValueError, match="damage evidence"):
        mod.score(second, invalid)
    with pytest.raises(ValueError, match="Repeated"):
        mod.score(first, valid + valid[:1])

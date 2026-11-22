from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


def games(plan: dict, seeds: range, clean: int, wins: int) -> list[dict]:
    return [
        dict(
            seed=seed,
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
        for i, seed in enumerate(seeds)
    ]


def test_successive_frozen_cohorts_and_damage_rules(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("assess_sisters_goals")
    actor, dll = tmp_path / "actor", tmp_path / "mod"
    actor.write_bytes(b"test actor")
    dll.write_bytes(b"test binary")
    first = mod.freeze(1, actor, dll, 1100, None)
    first_seeds = mod.seed_range(first)
    valid = games(first, first_seeds, 7, 98)
    seeded = mod.freeze(1, actor, dll, 1100, None, verify_unity_seed=True)
    with pytest.raises(ValueError, match="Unity reset seed"):
        mod.score(seeded, valid)
    assert mod.score(seeded, [dict(g, unity_reset_seed=g["seed"]) for g in valid])["passed"]
    with pytest.raises(ValueError, match="Int32"):
        mod.freeze(1, actor, dll, 2**31 - 10, None, verify_unity_seed=True)

    assert mod.score(first, valid)["next_goal"] == 2
    assert not mod.score(first, games(first, first_seeds, 7, 97))["passed"]
    assert not mod.score(first, valid[:-1])["passed"]
    report = tmp_path / "goal-1.json"
    report.write_text(json.dumps({"plan": first, "episodes": valid}))
    with pytest.raises(ValueError, match="new cohort"):
        mod.freeze(2, actor, dll, 1199, report)
    second = mod.freeze(2, actor, dll, 1200, report)
    second_seeds = mod.seed_range(second)
    assert mod.score(second, games(second, second_seeds, 95, 98))["passed"]
    assert not mod.score(second, games(second, second_seeds, 94, 100))["passed"]
    assert not mod.score(second, games(second, second_seeds, 95, 97))["passed"]
    invalid = games(second, second_seeds, 95, 98)
    invalid[0]["damage_taken"] = {"SPIKES": 1}
    with pytest.raises(ValueError, match="damage evidence"):
        mod.score(second, invalid)
    with pytest.raises(ValueError, match="Repeated"):
        mod.score(first, valid + valid[:1])


def test_current_acceptance_passes_at_95_wins_without_no_hit_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("assess_sisters_goals")
    actor, dll, plan_path = tmp_path / "actor", tmp_path / "mod", tmp_path / "plan.json"
    actor.write_bytes(b"test actor")
    dll.write_bytes(b"test binary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assess_sisters_goals.py",
            "freeze",
            "--checkpoint",
            str(actor),
            "--game-mod",
            str(dll),
            "--first-seed",
            "3100",
            "--output",
            str(plan_path),
        ],
    )
    mod.main()
    plan = json.loads(plan_path.read_bytes())
    seeds = mod.seed_range(plan)
    assert len(seeds) == 100
    result = mod.score(plan, games(plan, seeds, 0, 95))
    assert result["passed"]
    assert result["no_hit_wins"] == 0
    assert result["next_goal"] is None
    assert not mod.score(plan, games(plan, seeds, 0, 94))["passed"]
    assert not mod.score(plan, games(plan, seeds, 0, 95)[:-1])["passed"]

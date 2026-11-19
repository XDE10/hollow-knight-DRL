"""Causal ordering, raw input preservation and expert admission for native captures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from hkrl import protocol, spaces
from hkrl.env import HKRLEnv
from hkrl.utils.config import load_task_config

ROOT = Path(__file__).parents[2]


def script(name: str = "record_sisters_human"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_jump_dash_attack_and_held_keys_obey_mask() -> None:
    module = script()
    mask = np.ones(31, dtype=bool)
    keys = 1 | 4 | 16 | 32 | 64
    action, omitted = module.primitive(keys, 0, mask)
    assert tuple(action[:2]) == (0, 0)
    for name in ("jump_tap", "jump_hold", "dash", "attack"):
        assert action[2 + spaces.BUTTON_BITS[name]] == 1
    assert omitted == 0
    mask[6 + spaces.BUTTON_BITS["nail_art_hold"]] = False
    held, omitted = module.primitive(keys, keys, mask)
    assert held[2 + spaces.BUTTON_BITS["jump_hold"]] == 1
    assert held[2 + spaces.BUTTON_BITS["attack"]] == held[2 + spaces.BUTTON_BITS["dash"]] == 0
    assert omitted == 1


@pytest.mark.parametrize(
    "win,gap,expected", [(True, 1, "wins"), (False, 1, "attempts"), (True, 2, None)]
)
def test_native_episode_uses_next_events_and_keeps_only_full_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, win: bool, gap: int, expected: str | None
) -> None:
    module = script()
    task = load_task_config(ROOT / "configs/experiments/sisters_responsive.yaml")
    env = HKRLEnv(None, task)

    def frame(tick, body_count, events, done):
        entities = np.zeros((body_count, 24))
        entities[:, 1] = 1
        player = np.zeros(32)
        player[4:6] = 9
        global_state = np.array([0, 0, 2010, 0, 0, 0.65, 0.02, 0, 13])
        obs = protocol.DecodedObservation(global_state, player, entities, np.ones(body_count))
        return protocol.DecodedStepResponse(
            schema_version=protocol.SCHEMA_VERSION,
            env_id=0,
            tick_id=tick,
            server_tick=tick,
            observation=obs,
            reward_events=events,
            action_mask=np.ones(31, dtype=bool),
            terminated=done,
            truncated=False,
            lifecycle_state=protocol.LifecycleState.RUNNING,
            error_code=protocol.StatusCode.OK,
        )

    hit = protocol.RewardEvent(protocol.RewardEventKind.DAMAGE_DEALT, amount=21)
    end = protocol.RewardEvent(protocol.RewardEventKind.BOSS_KILLED, entity_id=0 if win else 15)
    stream = [
        (64, frame(1, 1, [], False)),
        (2, frame(1 + gap, 3, [hit], False)),
        (0, frame(2 + gap, 0, [end], True)),
    ]
    monkeypatch.setattr(module, "frames", lambda _: iter(stream))
    if expected is None:
        with pytest.raises(ValueError, match="discontinuous"):
            module.convert(Path("episode-1.hkdem"), tmp_path, env)
        return
    path = module.convert(Path("episode-1.hkdem"), tmp_path, env)
    assert path.parent.name == expected
    assert json.loads(path.with_suffix(".json").read_text())["qualified"] == win
    clone = script("clone_sisters_policy")
    if win:
        assert len(clone.load_demo(path, human=True)["actions"]) == 2
    else:
        with pytest.raises(ValueError, match="complete winning human demonstration"):
            clone.load_demo(path, human=True)
    with np.load(path, allow_pickle=False) as result:
        assert list(result["raw_keys"]) == [64, 2]
        assert list(result["server_ticks"]) == [1, 2]
        assert result["rewards"][0] == pytest.approx(6.3)
        assert result["actions"][0, 2 + spaces.BUTTON_BITS["attack"]] == 1


def test_partial_capture_never_accepted(tmp_path: Path) -> None:
    path = tmp_path / "episode.hkdem"
    path.write_bytes(b"HKD1\x01")
    with pytest.raises(ValueError, match="incomplete"):
        list(script().frames(path))

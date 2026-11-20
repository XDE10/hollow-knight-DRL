from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def ac(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("sisters_latent_ac")


def test_terminal_return_does_not_bootstrap(ac):
    r = torch.tensor([[1.0, 1.0], [3.0, 4.0]])
    nxt = torch.full_like(r, 100.0)
    disc = torch.tensor([[0.0, 0.5], [0.0, 0.0]])
    torch.testing.assert_close(
        ac.lambda_returns(r, nxt, disc, 1), torch.tensor([[1.0, 3.0], [3.0, 4.0]])
    )
    torch.testing.assert_close(
        ac.lambda_returns(r, nxt, disc, 0), torch.tensor([[1.0, 51.0], [3.0, 4.0]])
    )


def test_hold_duration_and_masks_match_live_primitive(ac):
    from hkrl import spaces

    valid = np.ones(len(spaces.action_mask_layout()), bool)
    valid[6 + spaces.BUTTON_BITS["cast"]] = False
    options = ac.legal_actions(valid)
    for i, (move, aim, buttons) in enumerate(ac.CATALOG):
        a = ac.controller_action(i)
        assert a["movement_x"] == move + 1 and a["aim_y"] == aim + 1
        assert sum(int(bit) << j for j, bit in enumerate(a["buttons"])) == buttons
        assert a["macro"] == 0 and spaces.DURATION_TICKS[a["duration"]] == 4
        assert options[i] == (buttons & 16 == 0)
    valid[17] = False
    assert not ac.legal_actions(valid).any()


def test_live_and_replay_encoded_states_match_without_padding_leakage(ac):
    from train_sisters_object_world import ObjectWorld

    model = ObjectWorld().eval()
    p = np.zeros((2, 32), np.float32)
    e = np.zeros((4, 24), np.float32)
    e[:, 1] = [1, 1, 1, 3]
    e[:, 6] = [20, 28, 34, 31]
    both = ac.encode(model, p, e, np.array([0, 1, 4]))
    one = ac.encode(model, p[:1], e[:1], np.array([0, 1]))
    torch.testing.assert_close(both[0], one[0], rtol=1e-5, atol=1e-6)


def test_interval_integrates_buttons_on_both_sides_of_a_change(ac):
    clock = np.array([0.0, 0.1, 0.2])
    raw = np.array([[-1, 0, 3], [1, 0, 2], [1, 0, 2]])
    actual = ac.applied_controls(np.array([0.05]), np.array([0.15]), clock, raw)[0]
    expected = (ac.keys(raw[:1]) + ac.keys(raw[1:2]))[0] / 2
    np.testing.assert_allclose(actual[:-1], expected, atol=1e-6)
    assert np.isclose(actual[-1], 0.5)


def test_ppo_updates_actor_and_critic_on_nonconstant_advantages(ac):
    from train_sisters_latent_ac import optimize_ppo

    torch.manual_seed(138)
    state = torch.randn(100, ac.STATE)
    policy = ac.ActorCritic()
    mask = torch.ones((100, len(ac.CATALOG)), dtype=torch.bool)
    mask[:, ::3] = False
    with torch.no_grad():
        dist, v = policy(state, mask)
        a = dist.sample()
        old = dist.log_prob(a)
    adv = torch.linspace(-2, 2, 100)
    initial = [p.detach().clone() for p in (policy.actor.weight, policy.critic.weight)]
    optimize_ppo(
        policy, torch.optim.Adam(policy.parameters()), state, a, old, v + adv, adv, mask=mask
    )
    assert not torch.equal(initial[0], policy.actor.weight)
    assert not torch.equal(initial[1], policy.critic.weight)


def test_short_attacks_survive_binning_without_turning_holds_into_pulses(ac):
    duty = np.zeros((2, ac.KEYS + 1), np.float32)
    duty[0, [5, 7, 9]] = [0.22, 0.18, 0.18]
    changed = ac.pulse_controls(duty)
    assert changed[0, 7] == 1 and changed[1, 7] == 0
    np.testing.assert_array_equal(changed[:, [5, 9]], duty[:, [5, 9]])
    assert duty[0, 7] < 0.2


def test_encounter_slots_are_causal_and_padding_is_empty(ac):
    import torch
    from train_sisters_object_world import ObjectWorld

    model = ObjectWorld().eval()
    p = np.zeros((1, 32), np.float32)
    p[0, 0:2] = [30, 7.4]
    e = np.zeros((3, 24), np.float32)
    e[:, 1] = [1, 1, 3]
    e[:, 2] = 1
    e[:, 14] = [38, 31, 29]
    e[:, 15] = 7.4
    e[:, 16:18] = 1
    z = ac.encode(model, p, e, np.array([0, 3]), True)[0, ac.STATE :].reshape(5, 8)
    torch.testing.assert_close(z[:, 0], torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(z[:, 1], torch.tensor([1.0, 8.0, 0.0, -1.0, 0.0]) / 12)
    shuffled = ac.encode(model, p, e[[1, 2, 0]], np.array([0, 3]), True)
    torch.testing.assert_close(shuffled[0, ac.STATE :], z.flatten())


def test_real_reward_preserves_spike_and_disc_damage_when_healing(ac):
    from types import SimpleNamespace

    from hkrl.protocol import DamageSource
    from hkrl.protocol import RewardEventKind as Kind
    from run_sisters_latent_ac import event_vector

    def event(kind, amount, source=0, entity=1):
        return SimpleNamespace(kind=kind, amount=amount, aux_int2=int(source), entity_id=entity)

    observed, damage, lost = event_vector(
        [
            event(Kind.DAMAGE_DEALT, 42),
            event(Kind.DAMAGE_TAKEN, 1, DamageSource.ENEMY_PROJECTILE),
            event(Kind.DAMAGE_TAKEN, 2, DamageSource.SPIKES),
            event(Kind.HEAL, 1),
            event(Kind.BOSS_KILLED, 1, entity=4),
        ]
    )
    assert damage == 42 and lost == {"ENEMY_PROJECTILE": 1, "SPIKES": 2}
    assert observed.tolist() == [1, 0, 1, 1, 0, 1, 0, 0]
    assert event_vector([event(Kind.BOSS_KILLED, 1, entity=0)])[0][6] == 1


def test_transfer_audit_never_predicts_across_episode_reset(ac, tmp_path):
    from audit_sisters_model_transfer import audit

    model = ac.Dynamics()
    for parameter in model.parameters():
        parameter.detach().zero_()
    world = tmp_path / "world.pt"
    torch.save({"models": [model.state_dict()], "biases": [torch.zeros(8)]}, world)
    record = {
        "state": torch.randn(11, ac.STATE),
        "action": torch.tensor([4] + [0] * 10),
        "reward": torch.arange(11).float(),
    }
    torch.save({"batch": record, "lengths": [4, 7]}, tmp_path / "rollout-000.pt")
    report = audit(world, tmp_path)
    assert report["fights"] == 2 and report["decisions"] == 11
    assert report["state"]["1"]["windows"] == 9 and report["state"]["5"]["windows"] == 2
    assert report["state"]["5"]["member_mae"][0] == pytest.approx(
        report["state"]["5"]["unchanged_mae"]
    )
    assert report["reward"]["all"]["observed"] == 5.0 and report["reward"]["nail"]["n"] == 1

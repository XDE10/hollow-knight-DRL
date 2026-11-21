from __future__ import annotations

import hashlib
import importlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def ppo(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("train_sisters_option_ppo")


def test_mask_loss_inside_same_interval_and_terminal(ppo):
    clock = np.arange(6) * 0.04
    hp = np.array([3, 2, 2, 2, 1, 1])
    starts = np.array([0.0, 0.12])
    events = np.zeros((2, 8))
    events[0, 1] = 1
    events[1, 2] = 1
    events[1, 7] = 1
    events[1, 3] = 1
    timed = ppo.verified_events(clock, hp, starts, np.array([6, 6]), events)
    np.testing.assert_array_equal(timed.sum(0), events.sum(0))
    # The native HP drop can localize the one source in the first interval.
    # The ambiguous second interval keeps both charges at its closing control.
    np.testing.assert_array_equal(timed[:, 1:5].sum(1), [1, 0, 0, 0, 0, 2])


def test_elapsed_returns_and_first_hit_before_control(ppo):
    clock = np.arange(6, dtype="f8")
    loss = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    events = np.zeros((6, 8))
    events[2, 0], events[-1, 6] = 21, 1
    rows = np.array([1, 4])
    rewards, gamma, traces = ppo.option_rewards(clock, loss, events, rows, 6.0)
    g = ppo.DISCOUNT_PER_SECOND
    np.testing.assert_allclose(rewards, [0.45 * g, (12 - 1.9) * g])
    np.testing.assert_allclose(gamma, [g**3, 0.0])
    np.testing.assert_allclose(traces, [ppo.TRACE_PER_SECOND**3, 0.0])
    np.testing.assert_allclose(
        ppo.discounted_scan(rewards, gamma), [rewards[0] + g**3 * rewards[1], rewards[1]]
    )
    loss[0] = 0
    costs, _, _ = ppo.option_rewards(clock, loss, events, rows, 6.0)
    assert rewards[1] - costs[1] == pytest.approx(6 * g)


def test_sampled_choice_receipt_binds_behavior_and_seed(tmp_path, ppo):
    choices = importlib.import_module("sisters_online_choices")
    risk = tmp_path / "risk"
    risk.write_bytes(struct.pack("<i", 0x314B5352) + bytes(193 * 4))
    sampler = ppo.start_policy(risk)
    clock = np.arange(50, dtype="f4") * 0.04
    state = np.zeros((len(clock), 192), "f4")
    state[:, 23] = 1
    actions = np.array([[-1, 0, 8], [1, 0, 1]], "i4")
    applied = np.tile(actions[0], (len(clock), 1))
    applied[30] = actions[1]
    typ = np.dtype([*choices.ROW.descr, ("scores", "<f4", (2,)), ("risk", "<f4")])
    rows = np.array(
        [
            (clock[3], [0, 1], 0, 0.3, 1.0, [2.0, 1.0], 0.1),
            (clock[30], [0, 1], 1, 0.65, 1.0, [2.0, 1.0], 0.2),
        ],
        dtype=typ,
    )
    path = tmp_path / "tape"
    path.write_bytes(
        struct.pack("<IQf", 0x34434B48, 123, 0.9)
        + hashlib.sha256(sampler).digest()
        + rows.tobytes()
    )
    actual = choices.read_choices(path, 123, clock, state, applied, actions, sampler)
    np.testing.assert_allclose(actual["choice_probability"], [0.3, 0.65])
    with pytest.raises(ValueError, match="Wrong online episode"):
        choices.read_choices(path, 124, clock, state, applied, actions, sampler)
    with pytest.raises(ValueError, match="differed"):
        choices.read_choices(path, 123, clock, state, applied, actions, sampler[:-1] + b"x")
    with pytest.raises(ValueError, match="Missing"):
        choices.read_choices(path, 123, clock, state, applied, actions)


@pytest.mark.parametrize("directional", [False, True])
def test_update_and_export_with_synthetic_on_policy_fights(tmp_path: Path, ppo, directional):
    timing = importlib.import_module("train_sisters_timing_iql")
    trainer = importlib.import_module("train_sisters_directional_ppo") if directional else ppo
    menu = importlib.import_module("sisters_directional_options")
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(22)
        catalogue = torch.tensor([[-1, 0, 8], [1, 0, 1]])
        actor = timing.TimingPolicy(catalogue, True).eval().requires_grad_(False)
    template = tmp_path / "leaf.bin"
    timing.save_native(actor, catalogue.numpy(), template)
    base = tmp_path / "base.bin"
    base.write_bytes(timing.phase_checkpoint(template.read_bytes(), template.read_bytes()))
    learner = tmp_path / "learner.pt"
    torch.save({"catalogue": catalogue, "actor": actor.state_dict()}, learner)
    risk = tmp_path / "risk.bin"
    risk.write_bytes(struct.pack("<i", 0x314B5352) + bytes(193 * 4))
    sampler = tmp_path / "sampler.bin"
    sampler.write_bytes(trainer.initial_head(risk) if directional else ppo.start_policy(risk))
    projection = tmp_path / "risk-projection.npy"
    np.save(projection, np.eye(192, 10, dtype="f4"))
    (tmp_path / "training.json").write_text(
        json.dumps({"learner_sha256": ppo.sha(learner), "projection_sha256": ppo.sha(projection)})
    )
    manifest = []
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for seed in range(4):
            times = np.arange(150, dtype="f4") * 0.04
            state = np.random.default_rng(seed).normal(0, 0.2, (len(times), 192)).astype("f4")
            state[:, 4], state[:, 23] = 1, 1
            if directional:
                state[:, 7], state[:, 164] = 1, 0
            rows = np.array([25, 60, 100])
            stops = np.searchsorted(
                times, times[rows] + (menu.HOLD_SECONDS if directional else np.float32(0.32))
            )
            all_scores = actor(torch.from_numpy(timing.motion_history(times, state)))[0].numpy()
            samples = all_scores[rows]
            pairs = np.argsort(-samples, axis=1)
            draws = (np.arange(len(rows)) + seed) % 2
            greedy = all_scores.argmax(1)
            applied = catalogue.numpy()[greedy].astype("i2")
            applied[rows] = catalogue.numpy()[pairs[np.arange(len(rows)), draws]]
            for j, start in enumerate(rows):
                if draws[j] and pairs[j, 1] == 0:
                    applied[start + 1 : stops[j]] = catalogue.numpy()[0]
            observed = np.zeros((len(times), 8), "f4")
            observed[-1, 6 if seed % 2 else 7] = 1
            if directional:
                slots, primary = menu.menu_slots(samples, catalogue.numpy(), state[rows])
                sampled = np.argmax(slots == pairs[np.arange(len(rows)), draws, None], axis=1)
                logits = menu.menu_logits(
                    np.zeros((len(rows), 192)), np.zeros((6, 193)), slots, primary
                )
                trace = {
                    "menu_draw": sampled,
                    "menu_primary": primary,
                    "menu_slots": slots,
                    "menu_risk": np.zeros(len(rows)),
                    "menu_probability": menu.behavior_probabilities(logits),
                }
            else:
                trace = {
                    "choice_draw": draws,
                    "choice_probability": np.full(len(rows), 0.5),
                    "choice_pairs": pairs,
                    "choice_scores": np.take_along_axis(samples, pairs, axis=1),
                    "choice_risk": np.zeros(len(rows)),
                }
            buf = io.BytesIO()
            np.savez(
                buf,
                control_time=times,
                timing_state=state,
                controls=applied,
                time=times,
                events=observed,
                elapsed_ticks=np.full(len(times), 2),
                choice_rows=rows,
                choice_hold_stop=stops,
                **trace,
            )
            name = f"fight-{seed}.npz"
            z.writestr(name, buf.getvalue())
            manifest.append(
                {
                    "source": f"native-development/{seed}",
                    "split": "train",
                    "teacher_labels": False,
                    "sampler_sha256": ppo.sha(sampler),
                    **(
                        {"sampler_draw_seed": seed + 31, "directional_groups": 6}
                        if directional
                        else {}
                    ),
                    "behavior_policy_sha256": ppo.sha(base),
                    "won": bool(seed % 2),
                    "file": name,
                    "sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
                }
            )
        z.writestr("manifest.json", json.dumps(manifest))
    output = tmp_path / "result"
    trainer.train(archive, base, learner, sampler, tmp_path, output)
    metrics = json.loads((output / "training.json").read_bytes())
    assert metrics["rollout_games"] == 4 and metrics["decisions"] == 12
    assert (
        metrics["epochs" if directional else "accepted_ppo_epochs"] > 0
        and metrics["sampler_input_sha256"] != metrics["sampler_sha256"]
    )
    _, trio = timing.phase_components((output / "world_iql.bin").read_bytes())
    leaf, head = timing.policy_gradient_components(trio)
    assert leaf == template.read_bytes() and head == (output / "sampler.bin").read_bytes()


def test_on_policy_pack_preserves_training_role_and_rejects_mixed_heads(tmp_path, ppo):
    pack = importlib.import_module("pack_sisters_timing_replay").pack
    entries = []
    for i, seed in enumerate((1000, 1001)):
        meta = {
            "behavior_policy_sha256": "b" * 64,
            "sampler_sha256": "c" * 64,
            "sampler_draw_seed": seed + 30,
            "teacher_labels": False,
        }
        observation = np.zeros((3, 192), "f4")
        observation[:, 24] = 1
        np.savez(
            tmp_path / f"g{i}.npz",
            timing_state=observation,
            control_time=np.arange(3) * 0.04,
            controls=np.zeros((3, 3), "i2"),
            events=np.zeros((1, 8)),
            elapsed_ticks=[6],
            time=[0.0],
            choice_rows=np.array([], dtype="i4"),
            choice_pairs=np.empty((0, 2), "i4"),
            choice_draw=np.array([], dtype="i4"),
            choice_probability=[],
            choice_gap=[],
        )
        entries.append(
            {**meta, "source": f"native-development/{seed}", "won": True, "file": f"g{i}.npz"}
        )
        (tmp_path / f"episode-{i:03d}.json").write_text(
            json.dumps({**meta, "seed": seed, "win": True, "speed_counts": {"1": 3}})
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    archive = tmp_path / "same.zip"
    pack([manifest], archive, True, on_policy_batch=True)
    with zipfile.ZipFile(archive) as z:
        assert [g["split"] for g in json.loads(z.read("manifest.json"))] == ["train", "train"]
    entries[1]["sampler_sha256"] = "d" * 64
    info = tmp_path / "episode-001.json"
    info.write_text(json.dumps({**json.loads(info.read_bytes()), "sampler_sha256": "d" * 64}))
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="one sampled controller"):
        pack([manifest], tmp_path / "mixed.zip", True, on_policy_batch=True)

from __future__ import annotations

import importlib
import io
import json
import pathlib
import struct
import zipfile

import numpy as np
import pytest


def test_random_neural_receipt_including_repeated_draws(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    parse = importlib.import_module("sisters_control_blocks").neural_assignments
    clock = np.arange(50, dtype="f4") / 25
    a, b = "01" * 32, "02" * 32
    header = struct.pack("<IQ32s32s", 0x314D5241, 7, bytes.fromhex(a), bytes.fromhex(b))
    tape = tmp_path / "7.arms"
    draws = [(0, 1), (20, 1), (40, 0)]
    tape.write_bytes(header + b"".join(struct.pack("<fB", clock[i], arm) for i, arm in draws))
    starts, assigned = parse(tape, 7, clock, a, b)
    assert starts.tolist() == [0, 20, 40]
    assert assigned.tolist() == [True] * 40 + [False] * 10
    with pytest.raises(ValueError, match="receipt"):
        parse(tape, 7, clock, b, a)
    tape.write_bytes(tape.read_bytes()[:-5])
    with pytest.raises(ValueError, match="draws"):
        parse(tape, 7, clock, a, b)


def test_neural_block_archive_keeps_interventions_and_fight_split(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    pack = importlib.import_module("pack_sisters_timing_replay").pack
    x = np.zeros((30, 192), "f4")
    x[:, 24] = 1
    clock = np.arange(30, dtype="f4") / 25
    assigned = np.arange(30) >= 20
    data = dict(
        control_time=clock,
        timing_state=x,
        controls=np.zeros((30, 3)),
        time=clock[::3],
        events=np.zeros((10, 8)),
        elapsed_ticks=np.full(10, 6),
        block_starts=[0, 20],
        candidate_draw=assigned,
    )
    fields = dict(
        student_policy_sha256="a" * 64,
        alternate_policy_sha256="b" * 64,
        candidate_probability=0.5,
        teacher_labels=False,
    )
    games = []
    for i in [0, 1]:
        meta = dict(seed=i, win=False, speed_counts={"1": 30}, **fields)
        games.append(dict(source=f"native-development/{i}", won=False, file=f"{i}.npz", **fields))
        (tmp_path / f"episode-{i:03d}.json").write_text(json.dumps(meta))
        np.savez(tmp_path / f"{i}.npz", **data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(games))
    contract = tmp_path / "experiment.json"
    contract.write_text(
        json.dumps(
            dict(
                actor_sha256="a" * 64,
                alternate_policy_sha256="b" * 64,
                candidate_probability=0.5,
                train_seeds=[1],
                heldout_seeds=[0],
            )
        )
    )
    output = tmp_path / "complete.zip"
    pack([manifest], output, True, contract)
    with zipfile.ZipFile(output) as z:
        rows = json.loads(z.read("manifest.json"))
        assert [r["split"] for r in rows] == ["holdout", "train"]
        for row in rows:
            assert row["alternate_policy_sha256"] == "b" * 64
            with np.load(io.BytesIO(z.read(row["file"]))) as d:
                assert np.array_equal(d["candidate_draw"], assigned)
    assigned[4] = True
    np.savez(tmp_path / "0.npz", **data)
    with pytest.raises(ValueError, match="Inconsistent neural draws"):
        pack([manifest], tmp_path / "invalid.zip", True, contract)


def test_contiguous_same_arm_draws_and_terminal_use_only_owned_rewards(monkeypatch):
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    take = importlib.import_module("sisters_control_blocks").control_blocks
    t = (np.arange(49) / 25).astype("f4")
    # Use the first native decision after each float32 deadline, like the recorder.
    second = np.searchsorted(t, np.float32(t[0] + np.float32(0.8)))
    third = np.searchsorted(t, np.float32(t[second] + np.float32(0.8)))
    assigned = np.ones(len(t), dtype=bool)
    assigned[third:] = False
    starts = np.arange(17, dtype="f4") * 0.12
    events = np.zeros((len(starts), 8), dtype="f4")
    events[6, 1] = 10  # [0.72, 0.84) straddles the draw; neither arm is charged.
    events[8, 0] = 1
    events[15, 7] = 1
    weights = np.array([1, -1, -1, -1, -1, 0, 1, -1])
    args = (t, assigned, np.ones(len(t), bool), starts, np.full(len(starts), 6), events, weights)
    blocks = take(*args)
    assert [(b.start, b.teacher) for b in blocks] == [(0, True), (second, True), (third, False)]
    assert blocks[0].reward == 0
    assert 0 < blocks[1].reward < 1 and blocks[1].discount > 0
    assert -1 < blocks[2].reward < 0 and blocks[2].discount == 0
    assigned[3] = False
    with pytest.raises(ValueError, match="within its randomized block"):
        take(*args)


def test_partial_nonterminal_last_block_is_excluded(monkeypatch):
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    take = importlib.import_module("sisters_control_blocks").control_blocks
    t = np.arange(12, dtype="f4") * 0.04
    assert (
        take(
            t,
            np.ones(len(t), bool),
            np.ones(len(t), bool),
            np.array([0.0, 0.12, 0.24]),
            np.full(3, 6),
            np.zeros((3, 8)),
            np.ones(8),
        )
        == []
    )


def test_recovery_fit_preserves_opener_and_refuses_mixed_splits(tmp_path, monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    mod = importlib.import_module("train_sisters_recovery_blocks")
    torch.manual_seed(9)
    cat = torch.tensor([[-1, 0, 0], [1, 0, 0]])
    network = mod.TimingPolicy(cat, history=True)
    parent = tmp_path / "parent.pt"
    torch.save(
        {"catalogue": cat, "actor": network.state_dict(), "value": mod.mlp(578, 1).state_dict()},
        parent,
    )
    deployed = tmp_path / "parent.bin"
    mod.save_native(network, cat.numpy(), deployed)
    opening = deployed.read_bytes()
    deployed.write_bytes(mod.phase_checkpoint(opening, opening))
    t = np.arange(64, dtype="f4") * 0.04
    assigned = np.zeros(len(t), dtype=bool)
    assigned[: np.searchsorted(t, np.float32(0.8))] = True
    s = np.zeros((len(t), 192), dtype="f4")
    s[:, 4] = s[:, 23] = s[:, 24] = 1
    student = np.tile([1, 0, 0], (len(t), 1)).astype("i2")
    teacher = student.copy()
    teacher[::2, 0] = -1
    starts = np.arange(22, dtype="f4") * 0.12
    body = io.BytesIO()
    np.savez_compressed(
        body,
        control_time=t,
        timing_state=s,
        teacher_acted=assigned,
        student_controls=student,
        teacher_controls=teacher,
        controls=np.where(assigned[:, None], teacher, student),
        time=starts,
        elapsed_ticks=np.full(len(starts), 6),
        events=np.zeros((len(starts), 8), "f4"),
    )
    data = body.getvalue()
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    spec = {
        "actor_sha256": mod.digest(deployed),
        "learner_sha256": mod.digest(parent),
        "teacher_probability": 0.5,
        "train_seeds": [1],
        "heldout_seeds": [2],
    }
    entries = [
        {"source": f"native-development/{i}", "split": split, "file": f"{i}.npz", "sha256": digest}
        for i, split in [(1, "train"), (2, "holdout")]
    ]
    archive = tmp_path / "development.zip"

    def write():
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("experiment.json", json.dumps(spec))
            z.writestr("manifest.json", json.dumps(entries))
            for row in entries:
                z.writestr(row["file"], data)

    write()
    out = tmp_path / "fit"
    mod.train(archive, parent, deployed, out, updates=1, seed=9)
    assert mod.phase_components((out / "world_iql.bin").read_bytes())[0] == opening
    report = json.loads((out / "training.json").read_bytes())
    assert set(report["heldout_fights"]) == {"native-development/2"}
    spec["heldout_seeds"] = [1]
    write()
    with pytest.raises(ValueError, match="Overlapping"):
        mod.train(archive, parent, deployed, tmp_path / "bad", updates=1, seed=9)

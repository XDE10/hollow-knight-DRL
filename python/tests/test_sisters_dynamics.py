from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

module = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "fit_sisters_dynamics.py")
)


def test_history_and_future_do_not_cross_workers_resets_or_file_end() -> None:
    valid = np.ones(19, bool)
    valid[[7, 14, 18]] = False
    assert module["starts"](valid, 2).tolist() == [3, 4, 5, 11, 12]
    assert module["starts"](valid, 0).tolist() == [3, 4, 5, 6, 7, 11, 12, 13, 14, 18]


def test_alternative_requires_legal_primitive_and_observed_combination() -> None:
    raw = np.zeros((4, 13), dtype=np.uint8)
    raw[1, -1] = 1
    raw[2, 4] = 1
    mask = np.ones((4, 31), bool)
    mask[3, 2] = False
    supported = raw[:1].copy()
    supported[:, 0] = 2
    ix, alt = module["alternatives"](
        SimpleNamespace(raw=raw, mask=mask), np.arange(4), "reverse", supported
    )
    assert ix.tolist() == [0]
    np.testing.assert_array_equal(alt, supported)
    assert not raw[:, 0].any()


def test_open_loop_uses_no_future_observation(tmp_path: Path) -> None:
    n = 24
    rng = np.random.default_rng(11)
    state = rng.normal(size=(n, 20)).astype(np.float32)
    actions = np.zeros((n, 13), np.uint8)
    data = dict(
        state=state,
        valid=np.r_[np.ones(n - 1, bool), False],
        hp=np.ones((n, 2), np.float32),
        mask=np.ones((n, 31), bool),
        terminated=np.zeros(n, bool),
        fsm=np.zeros(n, np.int64),
        actions=actions,
    )
    path = tmp_path / "train.npz"
    np.savez(path, **data)
    train = module["Replay"](path, "cpu")
    data["state"][7:] += 90.0
    data["fsm"][7:] = 4321
    np.savez(tmp_path / "hold.npz", **data)
    hold = module["Replay"](tmp_path / "hold.npz", "cpu", train)
    assert hold.fsm[-1].item() == len(train.vocab)
    torch.manual_seed(0)
    model = module["Dynamics"](train.obs.shape[-1], len(train.vocab) + 1).eval()
    for key, values in module["forecast"]([model], train, np.array([6]), 15).items():
        np.testing.assert_array_equal(
            values, module["forecast"]([model], hold, np.array([6]), 15)[key]
        )


@pytest.mark.parametrize("kind,bit", [("jump", 0), ("dash", 2), ("attack", 3)])
def test_button_intervention_preserves_other_inputs(kind: str, bit: int) -> None:
    raw = np.zeros((1, 13), dtype=np.uint8)
    raw[:, 1] = 2
    mask = np.ones((1, 31), bool)
    allowed = raw.copy()
    allowed[:, 2 + bit] = 1
    data = SimpleNamespace(raw=raw, mask=mask)
    _, alternative = module["alternatives"](data, np.array([0]), kind, allowed)
    np.testing.assert_array_equal(alternative, allowed)
    mask[0, 6 + bit] = False
    assert len(module["alternatives"](data, np.array([0]), kind, allowed)[0]) == 0

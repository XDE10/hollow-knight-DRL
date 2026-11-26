"""Config loader and repository configuration integrity tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hkrl.utils.config import (
    SecurityConfig,
    TrainConfig,
    load_task_config,
    load_train_config,
    load_yaml,
    resolve_auth_token,
    validate_bind_address,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_yaml_composes_defaults_with_deep_override(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "nested" / "child.yaml"
    child.parent.mkdir()

    _write_yaml(
        base,
        {
            "algorithm": "recurrent_ppo",
            "model": {"name": "entity_attention_gru", "rnn_hidden": 256},
            "transport": {"host": "127.0.0.1", "port": 5555},
        },
    )
    _write_yaml(
        child,
        {
            "defaults": ["../base.yaml"],
            "algorithm": "ppo",
            "model": {"name": "mlp"},
        },
    )

    assert load_yaml(child) == {
        "algorithm": "ppo",
        "model": {"name": "mlp", "rnn_hidden": 256},
        "transport": {"host": "127.0.0.1", "port": 5555},
    }


def test_load_yaml_rejects_default_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    _write_yaml(first, {"defaults": ["second.yaml"], "algorithm": "ppo"})
    _write_yaml(second, {"defaults": ["first.yaml"], "algorithm": "recurrent_ppo"})

    with pytest.raises(ValueError, match="cyclic config defaults"):
        load_yaml(first)


def test_sisters_repo_configs_load_cleanly() -> None:
    train_cfg = load_train_config(REPO_ROOT / "configs/sisters_train.yaml")
    task_cfg = load_task_config(REPO_ROOT / "configs/sisters_task.yaml")

    assert train_cfg.algorithm == "recurrent_ppo"
    assert train_cfg.transport.name == "tcp"
    assert train_cfg.transport.host == "127.0.0.1"
    assert train_cfg.transport.port == 5555

    assert task_cfg.task_id == "sisters_of_battle"
    assert task_cfg.wire_id == 2010
    assert task_cfg.scene == "GG_Mantis_Lords_V"
    assert task_cfg.action.enable_macro_actions is True
    assert task_cfg.action.n_macro_actions == 11


def test_security_helpers_validate_loopback_and_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    assert validate_bind_address("127.0.0.1:5555", bind_scope="localhost") == "127.0.0.1:5555"
    with pytest.raises(ValueError):
        validate_bind_address("0.0.0.0:5555", bind_scope="localhost")

    cfg = TrainConfig(security=SecurityConfig(require_token=True, auth_token_env="HKRL_TEST_TOKEN"))
    monkeypatch.setenv("HKRL_TEST_TOKEN", "secret-token")
    assert resolve_auth_token(cfg) == "secret-token"

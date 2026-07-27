"""API-key precedence: config-set key SUPERSEDES env (operator ruling dm#201).

Cloud API keys are the RULED exception to behavior-env elimination: they stay
env-INHERITABLE by default (nothing configured -> the exported key works, no
migration forced). BUT a key redefined via `abstractcore --config` always
wins — the old "environment variables always win" injection meant a rotated
console key applied only to lanes that happened to lack an export (the
key-precedence inversion, env-conflict report angle A #3).
"""

from __future__ import annotations

import json
import logging

import pytest

from abstractcore.config.manager import ConfigurationManager, api_key_fingerprint


def _write_config(tmp_path, api_keys: dict) -> str:
    config_file = tmp_path / "abstractcore.json"
    config_file.write_text(json.dumps({"api_keys": api_keys}))
    return str(config_file)


def test_env_key_inherited_when_config_has_none(tmp_path, monkeypatch):
    # The ruled DEFAULT: keys exist for other apps too; env inherits.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-inherited")
    _ = ConfigurationManager(config_dir=str(tmp_path), apply_env=True)
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-env-inherited"


def test_config_key_supersedes_env_key(tmp_path, monkeypatch, caplog):
    # The inversion fix: config-set key wins over the exported one.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-stale")
    config_file = _write_config(tmp_path, {"openai": "sk-config-rotated"})
    with caplog.at_level(logging.WARNING):
        _ = ConfigurationManager(config_file=config_file, apply_env=True)
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-config-rotated", "config-set key must supersede env (dm#201)"
    # Shadow warning names the env var + fingerprints, never key material.
    shadow = [r.getMessage() for r in caplog.records if "SHADOWED" in r.getMessage()]
    assert shadow and "OPENAI_API_KEY" in shadow[0]
    assert "sk-env-stale" not in shadow[0] and "sk-config-rotated" not in shadow[0]
    assert api_key_fingerprint("sk-env-stale") in shadow[0]


def test_config_key_fills_absent_env_without_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_file = _write_config(tmp_path, {"anthropic": "sk-ant-configured"})
    with caplog.at_level(logging.WARNING):
        _ = ConfigurationManager(config_file=config_file, apply_env=True)
    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-configured"
    assert not any("SHADOWED" in r.getMessage() for r in caplog.records)


def test_identical_config_and_env_key_no_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-same")
    config_file = _write_config(tmp_path, {"openrouter": "sk-same"})
    with caplog.at_level(logging.WARNING):
        _ = ConfigurationManager(config_file=config_file, apply_env=True)
    assert not any("SHADOWED" in r.getMessage() for r in caplog.records)


def test_shadow_warning_fires_once_per_state(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-stale")
    config_file = _write_config(tmp_path, {"openai": "sk-config-rotated"})
    manager = ConfigurationManager(config_file=config_file, apply_env=True)
    # set_api_key re-runs the injection; same state must not re-warn.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-stale")
    with caplog.at_level(logging.WARNING):
        manager._apply_api_keys_to_env()
        manager._apply_api_keys_to_env()
    # The state already warned once at construction; re-running the injection
    # with the same (env var, config key, env key) state must stay silent.
    assert sum("SHADOWED" in r.getMessage() for r in caplog.records) == 0


def test_openai_field_wins_over_openai_compatible_for_shared_env_var(tmp_path, monkeypatch):
    # openai and openai_compatible share OPENAI_API_KEY: the openai field wins
    # (historical injection order preserved).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_file = _write_config(tmp_path, {"openai": "sk-openai", "openai_compatible": "sk-compat"})
    _ = ConfigurationManager(config_file=config_file, apply_env=True)
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-openai"

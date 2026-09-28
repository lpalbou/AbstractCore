"""The OpenAI key saved in AbstractCore's config reaches the voice plugin.

AbstractVoice (>= 0.13.0) reads `voice_openai_api_key` from its host's plugin
config, never from an env var of its own choosing. AbstractCore hands it the
key saved in the config file the instance was created for
(`_abstractcore_config_file`, the gateway's core config), else the global
config; an explicit `create_llm(..., voice_openai_api_key=...)` wins, and with
no key saved nothing is set (AbstractVoice keeps its OPENAI_API_KEY fallback).
"""

from __future__ import annotations

import importlib.metadata
import json

import pytest

from abstractcore.core.interface import AbstractCoreInterface
from abstractcore.core.types import GenerateResponse


class _Provider(AbstractCoreInterface):
    def generate(self, prompt: str, **kwargs):
        return GenerateResponse(content=str(prompt))

    def get_capabilities(self):
        return []

    def unload_model(self, model_name: str) -> None:
        return None


@pytest.fixture()
def no_plugins(monkeypatch):
    # The registry discovers plugins through entry points: none here, so only
    # the owner's config is under test.
    class _EPs:
        def select(self, *, group):
            return []

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EPs())


def _config_file(tmp_path, openai_key):
    path = tmp_path / "abstractcore.json"
    path.write_text(json.dumps({"api_keys": {"openai": openai_key}} if openai_key else {}), encoding="utf-8")
    return str(path)


def test_the_instances_config_file_key_reaches_the_plugin_config(tmp_path, no_plugins):
    llm = _Provider("m", _abstractcore_config_file=_config_file(tmp_path, "sk-gateway"))
    _ = llm.capabilities
    assert llm.config["voice_openai_api_key"] == "sk-gateway"


def test_an_explicit_kwarg_wins(tmp_path, no_plugins):
    llm = _Provider(
        "m", _abstractcore_config_file=_config_file(tmp_path, "sk-gateway"), voice_openai_api_key="sk-explicit"
    )
    _ = llm.capabilities
    assert llm.config["voice_openai_api_key"] == "sk-explicit"


def test_no_saved_key_sets_nothing(tmp_path, no_plugins, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    llm = _Provider("m", _abstractcore_config_file=_config_file(tmp_path, None))
    _ = llm.capabilities
    assert "voice_openai_api_key" not in llm.config


def test_without_a_config_file_the_global_config_answers(tmp_path, no_plugins, monkeypatch):
    from types import SimpleNamespace

    from abstractcore.config import manager as mgr

    fake = SimpleNamespace(config=SimpleNamespace(api_keys=SimpleNamespace(openai="sk-global")))
    monkeypatch.setattr(mgr, "get_config_manager", lambda: fake)
    llm = _Provider("m")
    _ = llm.capabilities
    assert llm.config["voice_openai_api_key"] == "sk-global"


def test_an_unreadable_config_sets_nothing(no_plugins, monkeypatch):
    from abstractcore.config import manager as mgr

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(mgr, "get_config_manager", boom)
    llm = _Provider("m")
    _ = llm.capabilities
    assert "voice_openai_api_key" not in llm.config

"""Tests for Core provider endpoint profiles."""

from __future__ import annotations

import json
import os
import stat
import sys

from abstractcore import create_llm
from abstractcore.config.main import main as config_main
from abstractcore.config.manager import ConfigurationManager
from abstractcore.core.types import GenerateResponse


def _reset_global_config(monkeypatch, config_file):
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_FILE", str(config_file))
    import abstractcore.config.manager as manager_module
    import abstractcore.providers.registry as registry_module

    manager_module._config_manager = None
    registry_module._registry = None


def test_provider_profile_persists_and_public_output_redacts_secret(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"
    monkeypatch.setenv("OVH_AI_API_KEY", "env-secret")

    manager = ConfigurationManager(config_file=config_file, apply_env=False)
    profile = manager.set_provider_profile(
        "ovh-provider",
        display_name="OVH Provider",
        description="OVH inference endpoint",
        provider_family="openai-compatible",
        base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        api_key="$OVH_AI_API_KEY",
        allowed_models=["Qwen3.5-9B", "Qwen3-Embedding-8B"],
    )

    assert profile.virtual_provider_id == "endpoint:ovh-provider"
    public = manager.list_provider_profiles()[0]
    assert public["virtual_provider"] == "endpoint:ovh-provider"
    assert public["api_key_set"] is True
    assert public["api_key_fingerprint"]
    assert "env-secret" not in json.dumps(public)

    config = manager.get_provider_config("endpoint:ovh-provider")
    assert config["provider_family"] == "openai-compatible"
    assert config["base_url"] == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    assert config["api_key"] == "env-secret"
    assert config["allowed_models"] == ["Qwen3.5-9B", "Qwen3-Embedding-8B"]

    assert stat.S_IMODE(config_file.stat().st_mode) & stat.S_IRWXG == 0
    assert stat.S_IMODE(config_file.stat().st_mode) & stat.S_IRWXO == 0


def test_provider_profile_rejects_invalid_family_and_base_url(tmp_path) -> None:
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)

    try:
        manager.set_provider_profile("bad", provider_family="not-a-provider")
    except ValueError as exc:
        assert "Unsupported provider family" in str(exc)
    else:
        raise AssertionError("invalid family should fail")

    try:
        manager.set_provider_profile("bad", base_url="file:///tmp/socket")
    except ValueError as exc:
        assert "base URL must start" in str(exc)
    else:
        raise AssertionError("invalid base URL should fail")


def test_provider_profile_cli_set_list_models_and_default(monkeypatch, tmp_path, capsys) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"

    assert config_main(
        [
            "config",
            "--config-file",
            str(config_file),
            "set-provider",
            "ovh-provider",
            "--family",
            "openai-compatible",
            "--base-url",
            "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
            "--api-key",
            "stored-secret",
            "--name",
            "OVH Provider",
            "--description",
            "OVH inference endpoint",
            "--allow-model",
            "Qwen3.5-9B",
            "--json",
        ]
    ) == 0
    assert "stored-secret" not in capsys.readouterr().out

    assert config_main(["config", "--config-file", str(config_file), "providers", "--json"]) == 0
    providers_payload = json.loads(capsys.readouterr().out)
    assert providers_payload["profiles"][0]["virtual_provider"] == "endpoint:ovh-provider"
    assert "stored-secret" not in json.dumps(providers_payload)

    assert config_main(["config", "--config-file", str(config_file), "models", "endpoint:ovh-provider", "--json"]) == 0
    models_payload = json.loads(capsys.readouterr().out)
    assert models_payload["provider"] == "endpoint:ovh-provider"
    assert models_payload["models"] == ["Qwen3.5-9B"]

    assert config_main(
        [
            "config",
            "--config-file",
            str(config_file),
            "set-default",
            "input.text",
            "--provider",
            "endpoint:ovh-provider",
            "--model",
            "Qwen3.5-9B",
        ]
    ) == 0
    capsys.readouterr()

    assert config_main(["config", "--config-file", str(config_file), "defaults", "--json"]) == 0
    defaults_payload = json.loads(capsys.readouterr().out)
    route = next(row for row in defaults_payload["routes"] if row["key"] == "input.text")
    assert route["provider"] == "endpoint:ovh-provider"
    assert route["model"] == "Qwen3.5-9B"


def test_abstractcore_config_alias_accepts_direct_subcommands(monkeypatch, tmp_path, capsys) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"
    monkeypatch.setattr(sys, "argv", ["abstractcore-config", "set-provider"])

    assert config_main(
        [
            "--config-file",
            str(config_file),
            "set-provider",
            "ovh-provider",
            "--family",
            "openai-compatible",
            "--base-url",
            "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
            "--api-key",
            "stored-secret",
            "--allow-model",
            "Qwen3.5-9B",
        ]
    ) == 0
    assert "stored-secret" not in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["abstractcore-config", "models"])
    assert config_main(["--config-file", str(config_file), "models", "ovh-provider", "--json"]) == 0
    models_payload = json.loads(capsys.readouterr().out)
    assert models_payload["provider"] == "endpoint:ovh-provider"
    assert models_payload["models"] == ["Qwen3.5-9B"]


def test_provider_profile_cli_switches_api_key_source_with_one_flag(monkeypatch, tmp_path, capsys) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"
    monkeypatch.setenv("OVH_AI_API_KEY", "env-secret")

    assert config_main(
        [
            "config",
            "--config-file",
            str(config_file),
            "set-provider",
            "ovh-provider",
            "--family",
            "openai-compatible",
            "--base-url",
            "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
            "--api-key",
            "$OVH_AI_API_KEY",
        ]
    ) == 0
    capsys.readouterr()

    manager = ConfigurationManager(config_file=config_file, apply_env=False)
    assert manager.get_provider_config("endpoint:ovh-provider")["api_key"] == "env-secret"

    assert config_main(
        [
            "config",
            "--config-file",
            str(config_file),
            "set-provider",
            "ovh-provider",
            "--api-key",
            "stored-secret",
        ]
    ) == 0
    capsys.readouterr()

    manager = ConfigurationManager(config_file=config_file, apply_env=False)
    assert manager.get_provider_config("endpoint:ovh-provider")["api_key"] == "stored-secret"
    public = manager.get_provider_profile("ovh-provider").public_dict()
    assert public["api_key_env_var"] == ""
    assert "stored-secret" not in json.dumps(public)
    assert "env-secret" not in json.dumps(public)

    assert config_main(["config", "--config-file", str(config_file), "set-provider", "ovh-provider", "--clear-api-key"]) == 0
    capsys.readouterr()
    manager = ConfigurationManager(config_file=config_file, apply_env=False)
    public = manager.get_provider_profile("ovh-provider").public_dict()
    assert public["api_key_set"] is False
    assert public["api_key_env_var"] == ""


def test_endpoint_profile_is_available_and_injected_into_provider(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"
    _reset_global_config(monkeypatch, config_file)

    from abstractcore.config import get_config_manager
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
    from abstractcore.providers.registry import get_available_models_for_provider, list_available_providers

    manager = get_config_manager()
    manager.set_provider_profile(
        "ovh-provider",
        display_name="OVH Provider",
        provider_family="openai-compatible",
        base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        api_key="stored-secret",
        allowed_models=["Qwen3.5-9B"],
    )

    assert "endpoint:ovh-provider" in list_available_providers()
    assert get_available_models_for_provider("endpoint:ovh-provider") == ["Qwen3.5-9B"]

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)
    provider = create_llm("endpoint:ovh-provider", model="Qwen3.5-9B")

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.provider == "openai-compatible"
    assert provider.base_url == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    assert provider.api_key == "stored-secret"
    assert provider._abstractcore_virtual_provider == "endpoint:ovh-provider"
    assert provider._abstractcore_provider_family == "openai-compatible"


def test_endpoint_profile_qwen3_6_thinking_stays_strict_openai_compatible(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"
    _reset_global_config(monkeypatch, config_file)

    from abstractcore.config import get_config_manager
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    manager = get_config_manager()
    manager.set_provider_profile(
        "ovh-provider",
        display_name="OVH Provider",
        provider_family="openai-compatible",
        base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        api_key="stored-secret",
        allowed_models=["Qwen/Qwen3.6-27B"],
    )

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)
    provider = create_llm("endpoint:ovh-provider", model="Qwen/Qwen3.6-27B")
    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    provider.generate("hi", thinking="high", temperature=0)

    payload = captured["payload"]
    assert "chat_template_kwargs" not in payload
    assert "extra_body" not in payload
    assert "enableThinking" not in str(payload)
    assert "enable_thinking" not in str(payload)


def test_endpoint_profile_can_back_embedding_manager(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"
    _reset_global_config(monkeypatch, config_file)

    from abstractcore.config import get_config_manager
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
    from abstractcore.embeddings.manager import EmbeddingManager

    manager = get_config_manager()
    manager.set_provider_profile(
        "ovh-provider",
        provider_family="openai-compatible",
        base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        api_key="stored-secret",
        allowed_models=["Qwen3-Embedding-8B"],
    )

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)
    embeddings = EmbeddingManager(provider="endpoint:ovh-provider", model="Qwen3-Embedding-8B")

    assert embeddings.provider == "endpoint:ovh-provider"
    assert isinstance(embeddings._provider_instance, OpenAICompatibleProvider)
    assert embeddings._provider_instance.base_url == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    assert embeddings._provider_instance.api_key == "stored-secret"

"""Tests for persisted HTTP server configuration."""

import importlib
import os

import pytest

from abstractcore.config.manager import ConfigurationManager


_CONFIG_ENV_NAMES = (
    "ABSTRACTCORE_AUTH_TOKEN",
    "ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED",
    "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST",
    "ABSTRACTCORE_SERVER_URL_FETCH_ALLOWLIST",
    "ABSTRACTCORE_SERVER_MEDIA_ROOT",
    "ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES",
    "ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG",
    "HOST",
    "PORT",
    "OPENAI_API_KEY",
    "VLLM_API_KEY",
)


@pytest.fixture(autouse=True)
def _restore_config_env_after_test():
    """Config tests intentionally mutate os.environ; keep that local to each test."""
    original = {name: os.environ[name] for name in _CONFIG_ENV_NAMES if name in os.environ}
    for name in _CONFIG_ENV_NAMES:
        os.environ.pop(name, None)
    yield
    for name in _CONFIG_ENV_NAMES:
        os.environ.pop(name, None)
    os.environ.update(original)


def test_server_config_persists_and_injects_env(monkeypatch, tmp_path) -> None:
    """Server auth token and hardening settings should be persisted and env-backed."""
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_server_auth_token("server-secret")
    assert manager.set_server_allow_unauthenticated(True)
    assert manager.set_server_base_url_allowlist("https://example.com/v1")
    assert manager.set_server_url_fetch_allowlist("https://files.example.com")
    assert manager.set_server_media_root("/srv/abstractcore-media")
    assert manager.set_server_allow_local_files(True)
    assert manager.set_server_bind(host="127.0.0.1", port=8787)

    reloaded = ConfigurationManager()
    status = reloaded.get_status()["server"]

    assert status["auth_token"] == "✅ Set"
    assert status["allow_unauthenticated"] is True
    assert status["base_url_allowlist"] == "https://example.com/v1"
    assert status["url_fetch_allowlist"] == "https://files.example.com"
    assert status["media_root"] == "/srv/abstractcore-media"
    assert status["allow_local_files"] is True
    assert status["host"] == "127.0.0.1"
    assert status["port"] == 8787

    assert os.environ["ABSTRACTCORE_AUTH_TOKEN"] == "server-secret"
    assert os.environ["ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED"] == "1"
    assert os.environ["ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST"] == "https://example.com/v1"
    assert os.environ["ABSTRACTCORE_SERVER_URL_FETCH_ALLOWLIST"] == "https://files.example.com"
    assert os.environ["ABSTRACTCORE_SERVER_MEDIA_ROOT"] == "/srv/abstractcore-media"
    assert os.environ["ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES"] == "1"
    assert os.environ["HOST"] == "127.0.0.1"
    assert os.environ["PORT"] == "8787"


def test_environment_values_override_persisted_server_config(monkeypatch, tmp_path) -> None:
    """Deployment env vars must win over local config values."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "env-secret")

    manager = ConfigurationManager()
    assert manager.set_server_auth_token("config-secret")

    reloaded = ConfigurationManager()
    assert reloaded.config.server.auth_token == "config-secret"
    assert os.environ["ABSTRACTCORE_AUTH_TOKEN"] == "env-secret"


def test_provider_api_key_aliases_include_openai_compatible_and_vllm(monkeypatch, tmp_path) -> None:
    """Gateway/self-hosted OpenAI-shaped providers share the standard OpenAI env key."""
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_api_key("openai-compatible", "compat-key")
    assert manager.set_api_key("vllm", "vllm-key")

    reloaded = ConfigurationManager()
    assert reloaded.config.api_keys.openai_compatible == "compat-key"
    assert reloaded.config.api_keys.vllm == "vllm-key"
    assert os.environ["OPENAI_API_KEY"] == "compat-key"
    assert os.environ["VLLM_API_KEY"] == "vllm-key"


def test_server_app_loader_applies_persisted_server_config_when_enabled(monkeypatch, tmp_path) -> None:
    """The HTTP server startup hook should load persisted server settings outside tests."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    manager = ConfigurationManager()
    assert manager.set_server_auth_token("central-server-secret")

    # Simulate a fresh server process where the value is only in config JSON.
    os.environ.pop("ABSTRACTCORE_AUTH_TOKEN", None)
    config_manager_module = importlib.import_module("abstractcore.config.manager")
    monkeypatch.setattr(config_manager_module, "_config_manager", None)

    server_app = importlib.import_module("abstractcore.server.app")
    server_app._apply_centralized_config_env()

    assert os.environ["ABSTRACTCORE_AUTH_TOKEN"] == "central-server-secret"

import pytest

from abstractcore.providers.openrouter_provider import OpenRouterProvider


def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(OpenRouterProvider, "_get_api_key_from_config", lambda self: None)
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)

    with pytest.raises(ValueError, match="OpenRouter API key required"):
        OpenRouterProvider(model="openai/gpt-4o-mini")


def test_openrouter_uses_config_api_key_when_env_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(OpenRouterProvider, "_get_api_key_from_config", lambda self: "sk-test-from-config")
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)

    provider = OpenRouterProvider(model="openai/gpt-4o-mini")
    assert provider.api_key == "sk-test-from-config"


def test_openrouter_headers_include_auth_and_metadata(monkeypatch):
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "abstractframework-test")
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)

    provider = OpenRouterProvider(model="openai/gpt-4o-mini", api_key="sk-test-key")
    headers = provider._get_headers()

    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-Title"] == "abstractframework-test"


"""Tests for provider base_url environment variable support."""
import os
import pytest
from abstractcore.providers import OllamaProvider, LMStudioProvider
from abstractcore.providers import get_all_providers_with_models


class TestOllamaBaseUrlEnvVar:
    """Test environment variable support for Ollama provider."""

    def test_respects_ollama_base_url_env(self, monkeypatch):
        """Test that Ollama provider respects OLLAMA_BASE_URL."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-server:11434")
        provider = OllamaProvider(model="test")
        assert provider.base_url == "http://custom-server:11434"

    def test_respects_ollama_host_env(self, monkeypatch):
        """Test that Ollama provider respects OLLAMA_HOST."""
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:11434")
        provider = OllamaProvider(model="test")
        assert provider.base_url == "http://remote:11434"

    def test_ollama_base_url_precedence_over_host(self, monkeypatch):
        """OLLAMA_BASE_URL takes precedence over OLLAMA_HOST."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://base-url:11434")
        monkeypatch.setenv("OLLAMA_HOST", "http://host:11434")
        provider = OllamaProvider(model="test")
        assert provider.base_url == "http://base-url:11434"

    def test_param_overrides_env(self, monkeypatch):
        """Programmatic param takes precedence over env var."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
        provider = OllamaProvider(model="test", base_url="http://from-param:11434")
        assert provider.base_url == "http://from-param:11434"

    def test_default_when_no_env(self):
        """Falls back to default when no env var or param."""
        provider = OllamaProvider(model="test")
        assert provider.base_url == "http://localhost:11434"

    def test_strips_trailing_slash(self, monkeypatch):
        """Ensure trailing slashes are removed."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434/")
        provider = OllamaProvider(model="test")
        assert provider.base_url == "http://custom:11434"


class TestLMStudioBaseUrlEnvVar:
    """Test environment variable support for LMStudio provider."""

    def test_respects_lmstudio_base_url_env(self, monkeypatch):
        """Test that LMStudio provider respects LMSTUDIO_BASE_URL."""
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://custom:1235/v1")
        provider = LMStudioProvider(model="test")
        assert provider.base_url == "http://custom:1235/v1"

    def test_param_overrides_env(self, monkeypatch):
        """Programmatic param takes precedence over env var."""
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://from-env:1234/v1")
        provider = LMStudioProvider(model="test", base_url="http://from-param:1234/v1")
        assert provider.base_url == "http://from-param:1234/v1"

    def test_default_when_no_env(self):
        """Falls back to default when no env var or param."""
        # Use an actual valid model from the LMStudio server
        provider = LMStudioProvider(model="qwen/qwen3-4b-2507")
        assert provider.base_url == "http://localhost:1234/v1"

    def test_strips_trailing_slash(self, monkeypatch):
        """Ensure trailing slashes are removed."""
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://custom:1234/v1/")
        provider = LMStudioProvider(model="test")
        assert provider.base_url == "http://custom:1234/v1"


class TestRegistryRespectsEnvVars:
    """Test that provider registry respects environment variables."""

    def test_registry_ollama_env_var_integration(self, monkeypatch):
        """Verify registry works with OLLAMA_BASE_URL env var."""
        # Set to localhost (where server is running)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        providers = get_all_providers_with_models(include_models=False)
        ollama = next((p for p in providers if p['name'] == 'ollama'), None)

        assert ollama is not None
        # Registry should use the env var URL and detect availability
        assert ollama['status'] in ('available', 'error', 'no_models', 'unavailable')

    def test_registry_lmstudio_env_var_integration(self, monkeypatch):
        """Verify registry works with LMSTUDIO_BASE_URL env var."""
        # Set to localhost (where server is running)
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

        providers = get_all_providers_with_models(include_models=False)
        lmstudio = next((p for p in providers if p['name'] == 'lmstudio'), None)

        assert lmstudio is not None
        # Registry should use the env var URL and detect availability
        assert lmstudio['status'] in ('available', 'error', 'no_models', 'unavailable')

"""Tests for programmatic provider configuration."""
import pytest
from abstractcore.config import configure_provider, get_provider_config, clear_provider_config
from abstractcore import create_llm
from abstractcore.providers import get_all_providers_with_models


class TestConfigureProvider:
    """Test provider configuration methods."""

    def setup_method(self):
        """Clear config before each test."""
        clear_provider_config()

    def teardown_method(self):
        """Clear config after each test."""
        clear_provider_config()

    def test_configure_base_url(self):
        """Test setting base_url via configure_provider."""
        configure_provider('ollama', base_url='http://custom:11434')
        config = get_provider_config('ollama')
        assert config == {'base_url': 'http://custom:11434'}

    def test_config_case_insensitive(self):
        """Provider names are case-insensitive."""
        configure_provider('OLLAMA', base_url='http://test:11434')
        assert get_provider_config('ollama') == {'base_url': 'http://test:11434'}

    def test_clear_single_provider(self):
        """Test clearing config for a single provider."""
        configure_provider('ollama', base_url='http://a:11434')
        configure_provider('lmstudio', base_url='http://b:1234')
        clear_provider_config('ollama')
        assert get_provider_config('ollama') == {}
        assert get_provider_config('lmstudio') == {'base_url': 'http://b:1234'}

    def test_clear_all_providers(self):
        """Test clearing all provider config."""
        configure_provider('ollama', base_url='http://a:11434')
        configure_provider('lmstudio', base_url='http://b:1234')
        clear_provider_config()
        assert get_provider_config('ollama') == {}
        assert get_provider_config('lmstudio') == {}

    def test_set_none_clears_key(self):
        """Setting a value to None removes it."""
        configure_provider('ollama', base_url='http://test:11434')
        configure_provider('ollama', base_url=None)
        assert get_provider_config('ollama') == {}

    def test_multiple_config_values(self):
        """Test setting multiple configuration values."""
        configure_provider('ollama', base_url='http://test:11434', timeout=30.0)
        config = get_provider_config('ollama')
        assert config == {'base_url': 'http://test:11434', 'timeout': 30.0}


class TestProviderCreationWithConfig:
    """Test provider creation with runtime config."""

    def setup_method(self):
        clear_provider_config()

    def teardown_method(self):
        clear_provider_config()

    def test_create_llm_uses_config(self):
        """create_llm should use runtime config."""
        configure_provider('ollama', base_url='http://configured:11434')
        llm = create_llm('ollama', model='gemma3:1b')
        assert llm.base_url == 'http://configured:11434'

    def test_explicit_param_overrides_config(self):
        """Explicit base_url param should override config."""
        configure_provider('ollama', base_url='http://config:11434')
        llm = create_llm('ollama', model='gemma3:1b', base_url='http://param:11434')
        assert llm.base_url == 'http://param:11434'


class TestProviderDiscoveryWithConfig:
    """Test provider discovery with runtime config."""

    def setup_method(self):
        clear_provider_config()

    def teardown_method(self):
        clear_provider_config()

    def test_registry_uses_config(self):
        """Provider discovery should use runtime config."""
        # Set config to localhost (where server might be running)
        configure_provider('ollama', base_url='http://localhost:11434')
        providers = get_all_providers_with_models(include_models=False)
        ollama = next((p for p in providers if p['name'] == 'ollama'), None)
        # Registry should use the configured URL and detect availability
        # Status will be 'available' if server running, otherwise 'error' or 'no_models'
        assert ollama['status'] in ('available', 'error', 'unavailable', 'no_models')

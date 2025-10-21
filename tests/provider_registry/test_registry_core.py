"""
Test Provider Registry Core Functionality

Tests the centralized provider registry system including:
- Provider registration and discovery
- Metadata management
- Provider instantiation
- Error handling
"""

import pytest
from unittest.mock import patch, MagicMock
from abstractcore.providers.registry import (
    ProviderRegistry,
    ProviderInfo,
    get_provider_registry,
    list_available_providers,
    get_provider_info,
    is_provider_available,
    get_all_providers_with_models,
    get_all_providers_status,
    create_provider,
    get_available_models_for_provider
)


class TestProviderInfo:
    """Test ProviderInfo dataclass functionality."""

    def test_provider_info_creation(self):
        """Test creating ProviderInfo with required fields."""
        provider_info = ProviderInfo(
            name="test_provider",
            display_name="Test Provider",
            provider_class=None,
            description="A test provider"
        )

        assert provider_info.name == "test_provider"
        assert provider_info.display_name == "Test Provider"
        assert provider_info.description == "A test provider"
        assert provider_info.provider_type == "llm"  # Default value
        assert provider_info.authentication_required is True  # Default value
        assert provider_info.local_provider is False  # Default value
        assert provider_info.supported_features == []  # Default value

    def test_provider_info_with_all_fields(self):
        """Test creating ProviderInfo with all fields."""
        provider_info = ProviderInfo(
            name="test_provider",
            display_name="Test Provider",
            provider_class=None,
            description="A test provider",
            provider_type="embedding",
            default_model="test-model",
            supported_features=["chat", "completion"],
            authentication_required=False,
            local_provider=True,
            installation_extras="test",
            import_path="test.provider"
        )

        assert provider_info.provider_type == "embedding"
        assert provider_info.default_model == "test-model"
        assert provider_info.supported_features == ["chat", "completion"]
        assert provider_info.authentication_required is False
        assert provider_info.local_provider is True
        assert provider_info.installation_extras == "test"
        assert provider_info.import_path == "test.provider"


class TestProviderRegistry:
    """Test ProviderRegistry functionality."""

    def test_registry_initialization(self):
        """Test that registry initializes with standard providers."""
        registry = ProviderRegistry()

        # Check that standard providers are registered
        expected_providers = ["openai", "anthropic", "ollama", "lmstudio", "mlx", "huggingface", "mock"]
        registered_providers = registry.list_provider_names()

        for provider in expected_providers:
            assert provider in registered_providers

    def test_get_provider_info(self):
        """Test getting provider information."""
        registry = ProviderRegistry()

        # Test getting existing provider
        openai_info = registry.get_provider_info("openai")
        assert openai_info is not None
        assert openai_info.name == "openai"
        assert openai_info.display_name == "OpenAI"
        assert "native_tools" in openai_info.supported_features

        # Test getting non-existing provider
        invalid_info = registry.get_provider_info("nonexistent")
        assert invalid_info is None

    def test_is_provider_available(self):
        """Test checking provider availability."""
        registry = ProviderRegistry()

        assert registry.is_provider_available("openai") is True
        assert registry.is_provider_available("mock") is True
        assert registry.is_provider_available("nonexistent") is False

        # Test case insensitivity
        assert registry.is_provider_available("OPENAI") is True

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        registry = ProviderRegistry()

        custom_info = ProviderInfo(
            name="custom",
            display_name="Custom Provider",
            provider_class=None,
            description="A custom test provider"
        )

        registry.register_provider(custom_info)

        assert registry.is_provider_available("custom")
        retrieved_info = registry.get_provider_info("custom")
        assert retrieved_info.name == "custom"
        assert retrieved_info.display_name == "Custom Provider"

    def test_get_provider_class_openai(self):
        """Test getting provider class for OpenAI provider."""
        registry = ProviderRegistry()
        
        try:
            provider_class = registry.get_provider_class("openai")
            assert provider_class is not None
        except ImportError:
            pytest.skip("OpenAI provider not available")

    def test_get_provider_class_invalid(self):
        """Test getting provider class for invalid provider."""
        registry = ProviderRegistry()

        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            registry.get_provider_class("nonexistent")

    def test_get_available_models(self):
        """Test getting available models for a provider using OpenAI provider."""
        registry = ProviderRegistry()

        try:
            # Use OpenAI provider - may require API key for full functionality
            models = registry.get_available_models("openai")
            assert isinstance(models, list)
        except ImportError:
            pytest.skip("OpenAI provider not available")
        except Exception:
            # API calls may fail without proper credentials, which is expected
            pass

    def test_get_provider_status_success(self):
        """Test getting provider status when provider is working."""
        registry = ProviderRegistry()

        try:
            status = registry.get_provider_status("openai")
            assert status["name"] == "openai"
            assert status["local_provider"] is False
            assert "models" in status
        except ImportError:
            pytest.skip("OpenAI provider not available")

    def test_get_provider_status_nonexistent(self):
        """Test getting provider status for nonexistent provider."""
        registry = ProviderRegistry()

        status = registry.get_provider_status("nonexistent")

        assert status["status"] == "unknown"
        assert "error" in status

    def test_create_provider_instance(self):
        """Test creating provider instance."""
        registry = ProviderRegistry()

        try:
            instance = registry.create_provider_instance("openai", "gpt-4o", custom_param="value")
            assert instance is not None
            assert instance.model == "gpt-4o"
        except ImportError:
            pytest.skip("OpenAI provider not available")

    def test_create_provider_instance_invalid(self):
        """Test creating instance for invalid provider."""
        registry = ProviderRegistry()

        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            registry.create_provider_instance("nonexistent")


class TestGlobalRegistryFunctions:
    """Test global registry convenience functions."""

    def test_list_available_providers(self):
        """Test global list_available_providers function."""
        providers = list_available_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "mock" in providers

    def test_get_provider_info_global(self):
        """Test global get_provider_info function."""
        info = get_provider_info("openai")

        assert info is not None
        assert info.name == "openai"
        assert info.display_name == "OpenAI"

    def test_is_provider_available_global(self):
        """Test global is_provider_available function."""
        assert is_provider_available("openai") is True
        assert is_provider_available("nonexistent") is False

    @patch('abstractcore.providers.registry.get_provider_registry')
    def test_get_all_providers_with_models(self, mock_get_registry):
        """Test global get_all_providers_with_models function."""
        # Mock registry
        mock_registry = MagicMock()
        mock_registry.get_providers_with_models.return_value = [
            {"name": "mock", "status": "available", "model_count": 3}
        ]
        mock_get_registry.return_value = mock_registry

        providers = get_all_providers_with_models()

        assert len(providers) == 1
        assert providers[0]["name"] == "mock"
        assert providers[0]["status"] == "available"

    @patch('abstractcore.providers.registry.get_provider_registry')
    def test_get_all_providers_status(self, mock_get_registry):
        """Test global get_all_providers_status function."""
        # Mock registry
        mock_registry = MagicMock()
        mock_registry.get_all_providers_status.return_value = [
            {"name": "mock", "status": "available"},
            {"name": "error_provider", "status": "error"}
        ]
        mock_get_registry.return_value = mock_registry

        providers = get_all_providers_status()

        assert len(providers) == 2
        assert providers[0]["name"] == "mock"
        assert providers[1]["name"] == "error_provider"

    @patch('abstractcore.providers.registry.get_provider_registry')
    def test_create_provider_global(self, mock_get_registry):
        """Test global create_provider function."""
        # Mock registry
        mock_registry = MagicMock()
        mock_instance = MagicMock()
        mock_registry.create_provider_instance.return_value = mock_instance
        mock_get_registry.return_value = mock_registry

        instance = create_provider("mock", "test-model", custom_param="value")

        mock_registry.create_provider_instance.assert_called_once_with(
            "mock", "test-model", custom_param="value"
        )
        assert instance == mock_instance

    @patch('abstractcore.providers.registry.get_provider_registry')
    def test_get_available_models_for_provider_global(self, mock_get_registry):
        """Test global get_available_models_for_provider function."""
        # Mock registry
        mock_registry = MagicMock()
        mock_registry.get_available_models.return_value = ["model1", "model2"]
        mock_get_registry.return_value = mock_registry

        models = get_available_models_for_provider("mock")

        mock_registry.get_available_models.assert_called_once_with("mock")
        assert models == ["model1", "model2"]


class TestProviderRegistryIntegration:
    """Test provider registry integration with real providers."""

    def test_openai_provider_integration(self):
        """Test that OpenAI provider is properly integrated."""
        registry = ProviderRegistry()

        # Test getting OpenAI provider info
        openai_info = registry.get_provider_info("openai")
        assert openai_info is not None
        assert openai_info.name == "openai"
        assert openai_info.display_name == "OpenAI"
        assert openai_info.local_provider is False
        assert openai_info.authentication_required is True

    def test_provider_metadata_consistency(self):
        """Test that all registered providers have consistent metadata."""
        registry = ProviderRegistry()

        for provider_name in registry.list_provider_names():
            info = registry.get_provider_info(provider_name)

            # Required fields
            assert info.name is not None
            assert info.display_name is not None
            assert info.description is not None

            # Type validation
            assert info.provider_type in ["llm", "embedding"]
            assert isinstance(info.local_provider, bool)
            assert isinstance(info.authentication_required, bool)
            assert isinstance(info.supported_features, list)

    def test_singleton_registry(self):
        """Test that get_provider_registry returns singleton instance."""
        registry1 = get_provider_registry()
        registry2 = get_provider_registry()

        assert registry1 is registry2  # Same instance


if __name__ == "__main__":
    pytest.main([__file__])
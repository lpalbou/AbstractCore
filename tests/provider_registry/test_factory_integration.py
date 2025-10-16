"""
Test Provider Registry Factory Integration

Tests the integration between the provider registry and the factory create_llm function,
ensuring that the factory uses the centralized registry properly.
"""

import pytest
from unittest.mock import patch, MagicMock
from abstractcore.core.factory import create_llm
from abstractcore.exceptions import ModelNotFoundError, AuthenticationError, ProviderAPIError


class TestFactoryRegistryIntegration:
    """Test factory integration with provider registry."""

    @patch('abstractcore.providers.registry.create_provider')
    def test_create_llm_uses_registry(self, mock_create_provider):
        """Test that create_llm uses the centralized registry."""
        # Mock the registry function
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        result = create_llm("mock", model="test-model", custom_param="value")

        # Verify registry function was called with correct parameters
        mock_create_provider.assert_called_once_with("mock", "test-model", custom_param="value")
        assert result == mock_instance

    @patch('abstractcore.providers.registry.create_provider')
    def test_create_llm_without_model(self, mock_create_provider):
        """Test create_llm without specifying model (uses default)."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        result = create_llm("mock")

        # Should pass None as model, registry will use default
        mock_create_provider.assert_called_once_with("mock", None)
        assert result == mock_instance

    @patch('abstractcore.providers.registry.create_provider')
    def test_create_llm_with_kwargs(self, mock_create_provider):
        """Test create_llm passes through all kwargs to registry."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        result = create_llm(
            "mock",
            model="test-model",
            max_tokens=8192,
            temperature=0.7,
            timeout=30,
            custom_param="value"
        )

        mock_create_provider.assert_called_once_with(
            "mock",
            "test-model",
            max_tokens=8192,
            temperature=0.7,
            timeout=30,
            custom_param="value"
        )
        assert result == mock_instance

    @patch('abstractcore.providers.registry.create_provider')
    def test_create_llm_error_passthrough(self, mock_create_provider):
        """Test that create_llm passes through registry errors correctly."""
        # Test ModelNotFoundError
        mock_create_provider.side_effect = ModelNotFoundError("Model not found")

        with pytest.raises(ModelNotFoundError, match="Model not found"):
            create_llm("mock", "nonexistent-model")

        # Test AuthenticationError
        mock_create_provider.side_effect = AuthenticationError("Auth failed")

        with pytest.raises(AuthenticationError, match="Auth failed"):
            create_llm("openai", "gpt-4")

        # Test ProviderAPIError
        mock_create_provider.side_effect = ProviderAPIError("API error")

        with pytest.raises(ProviderAPIError, match="API error"):
            create_llm("anthropic", "claude-3")

    @patch('abstractcore.providers.registry.create_provider')
    def test_create_llm_mlx_model_detection(self, mock_create_provider):
        """Test that MLX model detection still works with registry."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        # Test MLX model detection when provider is huggingface
        result = create_llm("huggingface", model="mlx-community/Qwen3-4B")

        # Should have changed provider to mlx due to model name
        mock_create_provider.assert_called_once_with("mlx", "mlx-community/Qwen3-4B")
        assert result == mock_instance

    @patch('abstractcore.providers.registry.create_provider')
    def test_create_llm_gguf_model_detection(self, mock_create_provider):
        """Test that GGUF model detection still works with registry."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        # Test GGUF model detection when provider is mlx
        result = create_llm("mlx", model="model-name.gguf")

        # Should have changed provider to huggingface due to GGUF extension
        mock_create_provider.assert_called_once_with("huggingface", "model-name.gguf")
        assert result == mock_instance

    @patch('abstractcore.providers.registry.create_provider')
    def test_create_llm_no_model_detection_change(self, mock_create_provider):
        """Test that normal models don't trigger provider change."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        # Normal model should not change provider
        result = create_llm("openai", model="gpt-4")

        mock_create_provider.assert_called_once_with("openai", "gpt-4")
        assert result == mock_instance


class TestFactoryBackwardCompatibility:
    """Test that factory changes maintain backward compatibility."""

    @patch('abstractcore.providers.registry.create_provider')
    def test_factory_signature_unchanged(self, mock_create_provider):
        """Test that create_llm signature is unchanged."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        # Test that all previous calling patterns still work

        # Pattern 1: provider only
        create_llm("mock")

        # Pattern 2: provider and model
        create_llm("mock", "test-model")

        # Pattern 3: provider with kwargs
        create_llm("mock", max_tokens=8192)

        # Pattern 4: provider, model, and kwargs
        create_llm("mock", "test-model", max_tokens=8192, temperature=0.7)

        # All should have worked without errors
        assert mock_create_provider.call_count == 4

    @patch('abstractcore.providers.registry.create_provider')
    def test_factory_return_type_unchanged(self, mock_create_provider):
        """Test that create_llm return type is unchanged."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        result = create_llm("mock")

        # Should return the same instance from the registry
        assert result == mock_instance

    @patch('abstractcore.providers.registry.create_provider')
    def test_factory_error_types_unchanged(self, mock_create_provider):
        """Test that create_llm raises the same error types as before."""
        # Test that provider API errors are still raised
        mock_create_provider.side_effect = ModelNotFoundError("Test error")

        with pytest.raises(ModelNotFoundError):
            create_llm("mock", "nonexistent")

        # Test that authentication errors are still raised
        mock_create_provider.side_effect = AuthenticationError("Auth error")

        with pytest.raises(AuthenticationError):
            create_llm("openai", "gpt-4")


class TestFactoryWithMockProvider:
    """Test factory integration using the real mock provider."""

    def test_create_mock_provider_real(self):
        """Test creating mock provider without mocking (integration test)."""
        # This test uses the actual registry and mock provider
        instance = create_llm("mock", "test-model")

        # Verify we got a real provider instance
        assert instance is not None
        assert hasattr(instance, 'generate')
        assert hasattr(instance, 'list_available_models')

        # Verify the model was set correctly
        assert instance.model == "test-model"

    def test_create_mock_provider_with_kwargs(self):
        """Test creating mock provider with additional kwargs."""
        instance = create_llm("mock", "test-model", max_tokens=8192, timeout=30)

        assert instance.model == "test-model"
        assert instance.max_tokens == 8192
        # Test that kwargs were passed to the provider constructor
        # (timeout is a valid parameter for BaseProvider)

    def test_mock_provider_functionality(self):
        """Test that mock provider created through factory works correctly."""
        instance = create_llm("mock")

        # Test basic generation
        response = instance.generate("Test prompt")
        assert response is not None
        assert hasattr(response, 'content')

        # Test model listing
        models = instance.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0


class TestFactoryDocumentationExamples:
    """Test that examples from documentation still work."""

    @patch('abstractcore.providers.registry.create_provider')
    def test_documentation_examples(self, mock_create_provider):
        """Test examples that appear in the documentation."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        # Example from documentation
        create_llm("openai", model="gpt-4o-mini")
        create_llm("anthropic", model="claude-3-haiku")
        create_llm("ollama", model="qwen3-coder:30b")

        # All should have worked
        assert mock_create_provider.call_count == 3

    @patch('abstractcore.providers.registry.create_provider')
    def test_token_configuration_examples(self, mock_create_provider):
        """Test token configuration examples from documentation."""
        mock_instance = MagicMock()
        mock_create_provider.return_value = mock_instance

        # Examples with token configuration
        create_llm(
            "openai",
            model="gpt-4o",
            max_tokens=32000,
            max_output_tokens=4096,
            temperature=0.7,
            timeout=30
        )

        create_llm(
            "ollama",
            model="qwen3-coder:30b",
            max_tokens=8192,
            base_url="http://192.168.1.100:11434"
        )

        # Should have passed all parameters through
        assert mock_create_provider.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
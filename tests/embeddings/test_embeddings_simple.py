"""
Simple tests for embeddings module without complex mocking.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from abstractcore.embeddings.models import (
    get_model_config,
    get_default_model,
    list_available_models,
    list_available_providers,
    list_direct_embedding_providers,
    list_endpoint_backed_providers,
    list_provider_details,
    list_providers_requiring_local_model_files,
)
from abstractcore.embeddings.manager import EmbeddingManager


class TestEmbeddingModels:
    """Test embedding model configurations."""

    def test_get_model_config_valid(self):
        """Test getting valid model configurations."""
        config = get_model_config("embeddinggemma")
        assert config.name == "embeddinggemma"
        assert config.model_id == "google/embeddinggemma-300m"
        assert config.dimension == 768
        assert config.supports_matryoshka is True
        assert 256 in config.matryoshka_dims

    def test_get_model_config_invalid(self):
        """Test getting invalid model configuration."""
        with pytest.raises(ValueError, match="Model 'invalid' not supported"):
            get_model_config("invalid")

    def test_get_default_model(self):
        """Test getting default model."""
        default = get_default_model()
        assert default == "all-minilm-l6-v2"

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        assert "embeddinggemma" in models
        assert "all-minilm-l6-v2" in models
        assert "granite-30m" in models
        assert len(models) >= 6  # We now have 8 models

    def test_list_available_providers(self):
        """Test listing supported embedding providers."""
        providers = list_available_providers()
        assert "huggingface" in providers
        assert "openai-compatible" in providers
        assert "vllm" in providers
        assert "anthropic" not in providers

    def test_list_provider_details(self):
        """Test provider discovery metadata."""
        details = list_provider_details("lmstudio")
        assert details == [
            {
                "id": "lmstudio",
                "provider": "lmstudio",
                "label": "LMStudio",
                "transport": "openai_compatible_http",
                "base_url_configurable": True,
                "base_url_env_vars": ["LMSTUDIO_BASE_URL"],
                "default_base_url": "http://localhost:1234/v1",
            }
        ]
        assert "huggingface" in list_providers_requiring_local_model_files()
        assert "lmstudio" in list_endpoint_backed_providers()
        assert "vllm" in list_direct_embedding_providers()


class TestEmbeddingManagerBasic:
    """Test basic EmbeddingManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_fails_without_sentence_transformers(self):
        """Test that initialization fails when sentence-transformers is not available."""
        with patch('abstractcore.embeddings.manager.sentence_transformers', None):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                EmbeddingManager(cache_dir=self.cache_dir)

    def test_text_hash(self):
        """Test text hashing functionality."""
        # Create a minimal mock to test the hashing method
        with patch('abstractcore.embeddings.manager.sentence_transformers') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.SentenceTransformer.return_value = mock_model

            manager = EmbeddingManager(cache_dir=self.cache_dir)

            # Test that different texts produce different hashes
            hash1 = manager._text_hash("Hello world")
            hash2 = manager._text_hash("Hello world!")
            assert hash1 != hash2

            # Test that same text produces same hash
            hash3 = manager._text_hash("Hello world")
            assert hash1 == hash3

    def test_dimension_methods(self):
        """Test dimension-related methods."""
        with patch('abstractcore.embeddings.manager.sentence_transformers') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.SentenceTransformer.return_value = mock_model

            # Test without Matryoshka
            manager = EmbeddingManager(cache_dir=self.cache_dir)
            assert manager.get_dimension() == 768

            # Test with Matryoshka
            manager = EmbeddingManager(
                model="embeddinggemma",
                cache_dir=self.cache_dir,
                output_dims=256
            )
            assert manager.get_dimension() == 256

    def test_openrouter_embedding_manager_skips_chat_model_validation(self):
        """OpenRouter embedding model IDs are not always listed in the chat model catalogue."""
        from abstractcore.providers.openrouter_provider import OpenRouterProvider

        captured = {}

        def fake_init(self, model, **kwargs):
            self.model = model
            captured["model"] = model
            captured["kwargs"] = kwargs

        with patch.object(OpenRouterProvider, "__init__", fake_init):
            manager = EmbeddingManager(
                model="openai/text-embedding-3-small",
                provider="openrouter",
                cache_dir=self.cache_dir,
                provider_kwargs={"api_key": "sk-test"},
            )

        assert manager.model_id == "openai/text-embedding-3-small"
        assert captured["model"] == "openai/text-embedding-3-small"
        assert captured["kwargs"]["api_key"] == "sk-test"
        assert captured["kwargs"]["validate_model"] is False

    def test_vllm_embedding_manager_uses_vllm_provider(self):
        """vLLM is a first-class embedding provider, not a runtime-only catalog row."""
        from abstractcore.providers.vllm_provider import VLLMProvider

        captured = {}

        def fake_init(self, model, **kwargs):
            self.model = model
            captured["model"] = model
            captured["kwargs"] = kwargs

        with patch.object(VLLMProvider, "__init__", fake_init):
            manager = EmbeddingManager(
                model="embedding-model",
                provider="vllm",
                cache_dir=self.cache_dir,
                provider_kwargs={"base_url": "http://127.0.0.1:8000/v1"},
            )

        assert manager.model_id == "embedding-model"
        assert captured["model"] == "embedding-model"
        assert captured["kwargs"]["base_url"] == "http://127.0.0.1:8000/v1"

    def test_cache_operations(self):
        """Test cache operations."""
        with patch('abstractcore.embeddings.manager.sentence_transformers') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.SentenceTransformer.return_value = mock_model

            manager = EmbeddingManager(cache_dir=self.cache_dir)

            # Test cache stats
            stats = manager.get_cache_stats()
            assert "persistent_cache_size" in stats
            assert "embedding_dimension" in stats
            assert stats["embedding_dimension"] == 768

            # Test cache clearing
            manager.clear_cache()
            stats_after = manager.get_cache_stats()
            assert stats_after["memory_cache_info"]["currsize"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Simple tests for embeddings module without complex mocking.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from abstractllm.embeddings.models import get_model_config, get_default_model, list_available_models
from abstractllm.embeddings.manager import EmbeddingManager


class TestEmbeddingModels:
    """Test embedding model configurations."""

    def test_get_model_config_valid(self):
        """Test getting valid model configurations."""
        config = get_model_config("embeddinggemma")
        assert config.name == "embeddinggemma"
        assert config.model_id == "google/embeddinggemma-1.1"
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
        assert default == "embeddinggemma"

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        assert "embeddinggemma" in models
        assert "stella-400m" in models
        assert len(models) >= 4


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
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                EmbeddingManager(cache_dir=self.cache_dir)

    def test_text_hash(self):
        """Test text hashing functionality."""
        # Create a minimal mock to test the hashing method
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

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
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

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

    def test_cache_operations(self):
        """Test cache operations."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

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
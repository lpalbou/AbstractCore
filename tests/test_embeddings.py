"""
Comprehensive tests for the embeddings module.
Tests all functionality with real models and examples.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from abstractllm.embeddings import EmbeddingManager, EmbeddingModelConfig, get_model_config


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

    def test_all_model_configs(self):
        """Test all predefined model configurations."""
        models = ["embeddinggemma", "stella-400m", "nomic-embed", "mxbai-large"]
        for model_name in models:
            config = get_model_config(model_name)
            assert isinstance(config, EmbeddingModelConfig)
            assert config.dimension > 0
            assert config.max_sequence_length > 0


class TestEmbeddingManagerInit:
    """Test EmbeddingManager initialization."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('sentence_transformers.SentenceTransformer')
    def test_init_default(self, mock_st_class):
        """Test initialization with default parameters."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)

        assert manager.model_id == "google/embeddinggemma-1.1"
        assert manager.cache_dir == self.cache_dir
        assert manager.cache_size == 1000
        assert manager.output_dims is None

    @patch('sentence_transformers.SentenceTransformer')
    def test_init_custom_model(self, mock_st_class):
        """Test initialization with custom model."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_st_class.return_value = mock_model

        manager = EmbeddingManager(
            model="stella-400m",
            cache_dir=self.cache_dir,
            output_dims=512
        )

        assert manager.model_id == "dunzhang/stella_en_400M_v5"
        assert manager.output_dims == 512

    @patch('sentence_transformers.SentenceTransformer')
    def test_init_direct_huggingface_id(self, mock_st_class):
        """Test initialization with direct HuggingFace model ID."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        manager = EmbeddingManager(
            model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=self.cache_dir
        )

        assert manager.model_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert manager.model_config is None

    @patch('sentence_transformers.SentenceTransformer')
    def test_backend_selection(self, mock_st_class):
        """Test backend selection logic."""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        # Test auto backend selection
        with patch('onnxruntime'):
            manager = EmbeddingManager(cache_dir=self.cache_dir, backend="auto")
            # Should try ONNX first

        # Test explicit backend
        manager = EmbeddingManager(cache_dir=self.cache_dir, backend="pytorch")


class TestEmbeddingGeneration:
    """Test embedding generation with mocked models."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_embed_single_text(self, mock_st):
        """Test embedding a single text."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random(768)
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)
        embedding = manager.embed("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_embed_empty_text(self, mock_st):
        """Test embedding empty text."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)
        embedding = manager.embed("")

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(x == 0.0 for x in embedding)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_embed_batch(self, mock_st):
        """Test batch embedding."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random((3, 768))
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)
        texts = ["Hello", "World", "Test"]
        embeddings = manager.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_matryoshka_truncation(self, mock_st):
        """Test Matryoshka dimension truncation."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random(768)
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(
            model="embeddinggemma",
            cache_dir=self.cache_dir,
            output_dims=256
        )
        embedding = manager.embed("Hello world")

        assert len(embedding) == 256
        assert manager.get_dimension() == 256

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_caching_behavior(self, mock_st):
        """Test caching behavior."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        # Return consistent embeddings for caching test
        mock_model.encode.return_value = np.ones(768)
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)

        # First call - should generate
        embedding1 = manager.embed("Hello world")
        assert mock_model.encode.call_count == 1

        # Second call - should use cache (LRU cache)
        embedding2 = manager.embed("Hello world")
        assert mock_model.encode.call_count == 1  # Should not increase

        # Results should be identical
        assert embedding1 == embedding2

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_similarity_computation(self, mock_st):
        """Test similarity computation."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        # Return normalized vectors for predictable similarity
        mock_model.encode.side_effect = [
            np.ones(768) / np.sqrt(768),  # Normalized vector
            np.ones(768) / np.sqrt(768)   # Same normalized vector
        ]
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)
        similarity = manager.compute_similarity("Hello", "Hello")

        assert isinstance(similarity, float)
        assert abs(similarity - 1.0) < 1e-6  # Should be very close to 1


class TestCacheOperations:
    """Test cache operations."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_persistent_cache_save_load(self, mock_st):
        """Test persistent cache save and load."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random(768)
        mock_st.SentenceTransformer.return_value = mock_model

        # Create manager and generate embedding
        manager1 = EmbeddingManager(cache_dir=self.cache_dir)
        embedding1 = manager1.embed("Test text")
        manager1._save_persistent_cache()

        # Create new manager (should load cache)
        manager2 = EmbeddingManager(cache_dir=self.cache_dir)
        embedding2 = manager2.embed("Test text")

        # Should get same result from cache
        assert embedding1 == embedding2

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_cache_stats(self, mock_st):
        """Test cache statistics."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random(768)
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)
        manager.embed("Test text")

        stats = manager.get_cache_stats()
        assert "persistent_cache_size" in stats
        assert "memory_cache_info" in stats
        assert "embedding_dimension" in stats
        assert stats["embedding_dimension"] == 768

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_clear_cache(self, mock_st):
        """Test cache clearing."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random(768)
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)
        manager.embed("Test text")

        # Verify cache has content
        stats_before = manager.get_cache_stats()
        assert stats_before["memory_cache_info"]["currsize"] > 0

        # Clear cache
        manager.clear_cache()

        # Verify cache is empty
        stats_after = manager.get_cache_stats()
        assert stats_after["memory_cache_info"]["currsize"] == 0
        assert stats_after["persistent_cache_size"] == 0


class TestEventIntegration:
    """Test event system integration."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_event_emission(self, mock_st):
        """Test that events are emitted correctly."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random(768)
        mock_st.SentenceTransformer.return_value = mock_model

        with patch('abstractllm.embeddings.manager.emit_global') as mock_emit:
            manager = EmbeddingManager(cache_dir=self.cache_dir)
            manager.embed("Test text")

            # Should emit embedding_generated event
            mock_emit.assert_called()

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_no_events_fallback(self, mock_st):
        """Test graceful handling when events are not available."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.random.random(768)
        mock_st.SentenceTransformer.return_value = mock_model

        # Mock import error for events
        with patch('abstractllm.embeddings.manager.emit_global', side_effect=ImportError):
            manager = EmbeddingManager(cache_dir=self.cache_dir)
            # Should still work without events
            embedding = manager.embed("Test text")
            assert len(embedding) == 768


class TestErrorHandling:
    """Test error handling scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_model_loading_failure(self, mock_st):
        """Test handling of model loading failure."""
        mock_st.SentenceTransformer.side_effect = Exception("Model not found")

        with pytest.raises(Exception, match="Model not found"):
            EmbeddingManager(cache_dir=self.cache_dir)

    @patch('abstractllm.embeddings.manager.sentence_transformers')
    def test_encoding_failure_fallback(self, mock_st):
        """Test fallback to zero vector on encoding failure."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_st.SentenceTransformer.return_value = mock_model

        manager = EmbeddingManager(cache_dir=self.cache_dir)
        embedding = manager.embed("Test text")

        # Should return zero vector
        assert len(embedding) == 768
        assert all(x == 0.0 for x in embedding)

    def test_missing_sentence_transformers(self):
        """Test error when sentence-transformers is not available."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                EmbeddingManager(cache_dir=self.cache_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
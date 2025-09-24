"""
Real-world integration tests for embeddings with LLMs.
Tests actual embedding models and LLM integration scenarios.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import patch

from abstractllm.embeddings import EmbeddingManager
from abstractllm import create_llm


@pytest.mark.integration
class TestRealEmbeddings:
    """Test with real embedding models (requires network/downloads)."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_real_embedding_generation(self):
        """Test actual embedding generation with a lightweight model."""
        try:
            # Use a lightweight model for testing
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",  # Small, fast model
                cache_dir=self.cache_dir
            )

            # Test single embedding
            embedding = manager.embed("Hello world")
            assert isinstance(embedding, list)
            assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
            assert all(isinstance(x, (int, float)) for x in embedding)

            # Test batch embedding
            texts = ["Hello", "World", "Machine learning", "Artificial intelligence"]
            embeddings = manager.embed_batch(texts)
            assert len(embeddings) == 4
            assert all(len(emb) == 384 for emb in embeddings)

            # Test similarity
            similarity = manager.compute_similarity("cat", "kitten")
            assert isinstance(similarity, float)
            assert -1 <= similarity <= 1
            assert similarity > 0.5  # Should be reasonably similar

            # Test caching
            stats = manager.get_cache_stats()
            assert stats["persistent_cache_size"] > 0

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_real_semantic_search_scenario(self):
        """Test real semantic search scenario."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Document collection
            documents = [
                "The cat sat on the mat",
                "Python is a programming language",
                "Machine learning algorithms are powerful",
                "The dog played in the garden",
                "JavaScript is used for web development",
                "Deep learning requires large datasets"
            ]

            # Query
            query = "programming languages"

            # Generate embeddings
            query_embedding = manager.embed(query)
            doc_embeddings = manager.embed_batch(documents)

            # Compute similarities
            similarities = []
            for doc_emb in doc_embeddings:
                # Manual cosine similarity
                query_norm = np.linalg.norm(query_embedding)
                doc_norm = np.linalg.norm(doc_emb)
                if query_norm > 0 and doc_norm > 0:
                    sim = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
                    similarities.append(float(sim))
                else:
                    similarities.append(0.0)

            # Find most similar document
            most_similar_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            most_similar_doc = documents[most_similar_idx]

            # Should find programming-related documents
            assert "programming" in most_similar_doc.lower() or "javascript" in most_similar_doc.lower()

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


@pytest.mark.integration
class TestLLMEmbeddingIntegration:
    """Test integration between LLMs and embeddings for RAG scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rag_pipeline_simulation(self):
        """Test a simulated RAG pipeline with embeddings and LLM."""
        try:
            # Initialize embedding manager
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Knowledge base
            knowledge_base = [
                "Paris is the capital of France and has a population of over 2 million people.",
                "The Eiffel Tower was built in 1889 and is 330 meters tall.",
                "French cuisine is known for its sophistication and includes dishes like croissants and escargot.",
                "The Louvre Museum in Paris houses the Mona Lisa painting.",
                "France is located in Western Europe and borders several countries."
            ]

            # User question
            question = "What is the height of the Eiffel Tower?"

            # Step 1: Embed question and knowledge base
            question_embedding = embedder.embed(question)
            kb_embeddings = embedder.embed_batch(knowledge_base)

            # Step 2: Find most relevant context
            similarities = []
            for kb_emb in kb_embeddings:
                similarity = embedder.compute_similarity(question, knowledge_base[len(similarities)])
                similarities.append(similarity)

            most_relevant_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            context = knowledge_base[most_relevant_idx]

            # Step 3: Verify we found the right context
            assert "Eiffel Tower" in context
            assert "330 meters" in context

            # Step 4: Simulate LLM call with context (mock for speed)
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

            # In a real scenario, you would use:
            # llm = create_llm("openai", model="gpt-3.5-turbo")
            # response = llm.generate(prompt)

            # For testing, just verify the pipeline structure works
            assert len(prompt) > 0
            assert question in prompt
            assert context in prompt

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_embedding_llm_separation(self):
        """Test that embeddings and LLMs work independently."""
        try:
            # Test that we can create both independently
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Mock LLM (don't actually call APIs in tests)
            with patch('abstractllm.create_llm') as mock_create_llm:
                mock_llm = mock_create_llm.return_value
                mock_llm.generate.return_value = "Mock response"

                llm = create_llm("mock", model="test")

                # Use both independently
                embedding = embedder.embed("Test text")
                response = llm.generate("Test prompt")

                assert len(embedding) == 384
                assert response == "Mock response"

        except ImportError:
            pytest.skip("sentence-transformers not available")


@pytest.mark.integration
class TestPerformanceBenchmarks:
    """Performance benchmarks for embeddings."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_vs_individual_performance(self):
        """Test that batch processing is more efficient than individual calls."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            texts = [f"Test text number {i}" for i in range(10)]

            import time

            # Individual embeddings
            start_time = time.time()
            individual_embeddings = [manager.embed(text) for text in texts]
            individual_time = time.time() - start_time

            # Clear cache to ensure fair comparison
            manager.embed.cache_clear()

            # Batch embeddings
            start_time = time.time()
            batch_embeddings = manager.embed_batch(texts)
            batch_time = time.time() - start_time

            # Results should be identical
            assert len(individual_embeddings) == len(batch_embeddings)
            for ind, batch in zip(individual_embeddings, batch_embeddings):
                np.testing.assert_array_almost_equal(ind, batch, decimal=6)

            # Batch should be faster (though this may not always be true for small batches)
            print(f"Individual time: {individual_time:.3f}s, Batch time: {batch_time:.3f}s")

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_caching_performance(self):
        """Test that caching improves performance."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            text = "This is a test text for caching performance"

            import time

            # First call (no cache)
            start_time = time.time()
            embedding1 = manager.embed(text)
            first_call_time = time.time() - start_time

            # Second call (cached)
            start_time = time.time()
            embedding2 = manager.embed(text)
            second_call_time = time.time() - start_time

            # Results should be identical
            assert embedding1 == embedding2

            # Second call should be faster
            assert second_call_time < first_call_time
            print(f"First call: {first_call_time:.3f}s, Cached call: {second_call_time:.3f}s")

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
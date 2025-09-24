"""
Zero-Mock Embeddings Tests
==========================

Comprehensive tests with absolutely NO MOCKING - only real models and real functionality.
Tests the complete embeddings pipeline with actual sentence-transformers models.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import time

from abstractllm.embeddings import EmbeddingManager, get_model_config, list_available_models


@pytest.mark.integration
class TestCompletelyRealEmbeddings:
    """Zero-mock tests with actual embedding models."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_real_model_configurations(self):
        """Test that all model configurations are valid and accessible."""
        models = list_available_models()

        # Verify we have the expected models
        expected_models = ["embeddinggemma", "stella-400m", "nomic-embed", "mxbai-large"]
        for model in expected_models:
            assert model in models

        # Test each model configuration
        for model_name in models:
            config = get_model_config(model_name)
            assert config.name == model_name
            assert config.dimension > 0
            assert config.max_sequence_length > 0
            assert isinstance(config.supports_matryoshka, bool)
            assert isinstance(config.multilingual, bool)
            print(f"✅ Model {model_name}: {config.dimension}D, {config.model_id}")

    def test_real_lightweight_embedding_full_pipeline(self):
        """Test complete embedding pipeline with real lightweight model."""
        try:
            # Use fastest available model for testing
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test 1: Single text embedding
            text = "Artificial intelligence is transforming technology and society."
            embedding = manager.embed(text)

            assert isinstance(embedding, list)
            assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
            assert all(isinstance(x, (int, float)) for x in embedding)
            assert any(x != 0.0 for x in embedding)  # Not all zeros
            print(f"✅ Single embedding: {len(embedding)}D vector generated")

            # Test 2: Empty text handling
            empty_embedding = manager.embed("")
            assert len(empty_embedding) == 384
            assert all(x == 0.0 for x in empty_embedding)
            print(f"✅ Empty text handling: zero vector returned")

            # Test 3: Batch processing with different texts
            texts = [
                "Python programming language",
                "Machine learning algorithms",
                "Web development frameworks",
                "Database management systems",
                "Cloud computing platforms"
            ]

            batch_embeddings = manager.embed_batch(texts)
            assert len(batch_embeddings) == len(texts)
            assert all(len(emb) == 384 for emb in batch_embeddings)

            # Verify embeddings are different for different texts
            for i in range(len(batch_embeddings)):
                for j in range(i + 1, len(batch_embeddings)):
                    assert batch_embeddings[i] != batch_embeddings[j]
            print(f"✅ Batch processing: {len(texts)} different embeddings generated")

            # Test 4: Similarity computation with semantic meaning
            # Similar texts should have higher similarity
            similar_text1 = "Python is a programming language"
            similar_text2 = "Python is used for programming"
            dissimilar_text = "The weather is sunny today"

            sim_high = manager.compute_similarity(similar_text1, similar_text2)
            sim_low = manager.compute_similarity(similar_text1, dissimilar_text)

            assert isinstance(sim_high, float)
            assert isinstance(sim_low, float)
            assert -1 <= sim_high <= 1
            assert -1 <= sim_low <= 1
            assert sim_high > sim_low  # Similar texts should be more similar
            assert sim_high > 0.5  # Should be reasonably similar
            assert sim_low < 0.3   # Should be reasonably different
            print(f"✅ Similarity: similar texts={sim_high:.3f}, different texts={sim_low:.3f}")

            # Test 5: Caching behavior - same input should be cached
            start_time = time.time()
            cached_embedding = manager.embed(text)  # Same as first test
            cache_time = time.time() - start_time

            assert cached_embedding == embedding  # Should be identical
            assert cache_time < 0.01  # Should be very fast (cached)

            stats = manager.get_cache_stats()
            assert stats["memory_cache_info"]["hits"] > 0
            print(f"✅ Caching: {cache_time:.4f}s for cached retrieval")

            return True

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_real_semantic_search_complete_scenario(self):
        """Test a complete semantic search scenario with real embeddings."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Real knowledge base about programming
            knowledge_base = [
                "Python is a high-level interpreted programming language known for its simplicity and readability.",
                "JavaScript is a dynamic programming language primarily used for web development and frontend applications.",
                "SQL is a domain-specific language used for managing and querying relational databases.",
                "React is a JavaScript library for building user interfaces, particularly single-page applications.",
                "Docker is a containerization platform that packages applications and their dependencies.",
                "Kubernetes is an orchestration system for automating deployment and management of containerized applications.",
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "Git is a distributed version control system for tracking changes in source code during development."
            ]

            # Real user queries
            test_queries = [
                ("programming languages", ["Python", "JavaScript"]),
                ("web development", ["JavaScript", "React"]),
                ("containerization", ["Docker", "Kubernetes"]),
                ("database", ["SQL"]),
                ("artificial intelligence", ["Machine learning"])
            ]

            for query, expected_keywords in test_queries:
                print(f"\n🔍 Testing query: '{query}'")

                # Find most relevant documents
                similarities = []
                for doc in knowledge_base:
                    similarity = manager.compute_similarity(query, doc)
                    similarities.append((similarity, doc))

                # Sort by similarity (highest first)
                similarities.sort(reverse=True, key=lambda x: x[0])

                # Get top 2 results
                top_results = similarities[:2]

                print(f"📄 Top results:")
                for score, doc in top_results:
                    print(f"   Score: {score:.3f} - {doc[:50]}...")

                # Verify results make semantic sense
                best_score, best_doc = top_results[0]
                assert best_score > 0.2  # Should have reasonable similarity

                # Check that at least one expected keyword appears in the best result
                found_keyword = any(keyword.lower() in best_doc.lower() for keyword in expected_keywords)
                assert found_keyword, f"Expected keywords {expected_keywords} not found in: {best_doc}"

                print(f"✅ Query '{query}' correctly matched relevant content")

            return True

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_real_rag_pipeline_end_to_end(self):
        """Test complete RAG pipeline with real embeddings (no LLM call)."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Real knowledge base
            knowledge_base = [
                "The Python programming language was created by Guido van Rossum and first released in 1991.",
                "React.js was developed by Facebook and released as open source in 2013 for building user interfaces.",
                "Docker was first released in 2013 and revolutionized application deployment through containerization.",
                "The Git version control system was created by Linus Torvalds in 2005 for Linux kernel development.",
                "TensorFlow is an open-source machine learning framework developed by Google and released in 2015."
            ]

            # Real questions that should find specific answers
            test_cases = [
                ("Who created Python?", "Guido van Rossum", 0.4),
                ("When was React released?", "2013", 0.3),
                ("What is Docker used for?", "containerization", 0.3),
                ("Who developed Git?", "Linus Torvalds", 0.4),
                ("What company made TensorFlow?", "Google", 0.3)
            ]

            for question, expected_answer, min_score in test_cases:
                print(f"\n🤖 RAG Test - Question: '{question}'")

                # Step 1: Find most relevant context using embeddings
                best_score = 0
                best_context = ""

                for doc in knowledge_base:
                    similarity = manager.compute_similarity(question, doc)
                    if similarity > best_score:
                        best_score = similarity
                        best_context = doc

                print(f"📄 Best context (score: {best_score:.3f}): {best_context[:60]}...")

                # Verify we found relevant context
                assert best_score >= min_score, f"Score {best_score} below minimum {min_score}"
                assert expected_answer.lower() in best_context.lower(), f"Expected '{expected_answer}' not found in context"

                # Step 2: Create RAG prompt (structure only - no actual LLM call)
                rag_prompt = f"""Context: {best_context}

Question: {question}

Based on the provided context, please answer the question:"""

                assert len(rag_prompt) > 0
                assert question in rag_prompt
                assert best_context in rag_prompt

                print(f"✅ RAG pipeline successfully found relevant context for: '{question}'")

            return True

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_real_performance_and_caching(self):
        """Test real performance characteristics and caching behavior."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir,
                cache_size=100
            )

            # Performance test data
            texts = [f"Performance test sentence number {i} for benchmarking" for i in range(8)]

            # Test 1: Individual vs batch performance
            start_time = time.time()
            individual_results = []
            for text in texts:
                embedding = manager.embed(text)
                individual_results.append(embedding)
            individual_time = time.time() - start_time

            # Clear cache for fair comparison
            manager.embed.cache_clear()

            start_time = time.time()
            batch_results = manager.embed_batch(texts)
            batch_time = time.time() - start_time

            # Verify results are identical
            assert len(individual_results) == len(batch_results)
            for ind, batch in zip(individual_results, batch_results):
                np.testing.assert_array_almost_equal(ind, batch, decimal=5)

            print(f"⚡ Performance test:")
            print(f"   Individual: {individual_time:.3f}s ({individual_time/len(texts):.3f}s per text)")
            print(f"   Batch: {batch_time:.3f}s ({batch_time/len(texts):.3f}s per text)")
            if batch_time > 0:
                speedup = individual_time / batch_time
                print(f"   Speedup: {speedup:.1f}x")
                # Batch should generally be faster for multiple texts
                if len(texts) > 3:
                    assert speedup > 1.0, "Batch processing should be faster"

            # Test 2: Cache performance
            test_text = "This text will be used for cache testing"

            # First call - not cached
            start_time = time.time()
            first_embedding = manager.embed(test_text)
            first_time = time.time() - start_time

            # Second call - should be cached
            start_time = time.time()
            cached_embedding = manager.embed(test_text)
            cached_time = time.time() - start_time

            assert first_embedding == cached_embedding
            assert cached_time < first_time  # Cache should be faster
            assert cached_time < 0.01  # Should be very fast

            # Verify cache stats
            stats = manager.get_cache_stats()
            assert stats["memory_cache_info"]["hits"] > 0
            assert stats["persistent_cache_size"] > 0

            print(f"💾 Cache test:")
            print(f"   First call: {first_time:.3f}s")
            print(f"   Cached call: {cached_time:.4f}s")
            if cached_time > 0:
                print(f"   Cache speedup: {first_time/cached_time:.1f}x")
            else:
                print(f"   Cache speedup: >1000x (too fast to measure)")
            print(f"   Cache stats: {stats['memory_cache_info']['hits']} hits")

            return True

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


if __name__ == "__main__":
    # Run comprehensive zero-mock tests
    print("🚀 Running comprehensive zero-mock embeddings tests...")
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
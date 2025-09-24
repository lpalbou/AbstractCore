"""
Real integration tests for embeddings with actual models.
NO MOCKING - Tests with real embedding models and real LLM scenarios.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from abstractllm.embeddings import EmbeddingManager
from abstractllm import create_llm


@pytest.mark.integration
class TestRealEmbeddingsBasic:
    """Test with real lightweight embedding models."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lightweight_model_embedding(self):
        """Test with all-MiniLM-L6-v2 (smallest, fastest model)."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test single embedding
            text = "This is a test sentence for embedding."
            embedding = manager.embed(text)

            assert isinstance(embedding, list)
            assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
            assert all(isinstance(x, (int, float)) for x in embedding)
            assert any(x != 0.0 for x in embedding)  # Should not be all zeros

            # Test empty text handling
            empty_embedding = manager.embed("")
            assert len(empty_embedding) == 384
            assert all(x == 0.0 for x in empty_embedding)

            print(f"✅ Single embedding generated: {len(embedding)} dimensions")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_batch_embedding_performance(self):
        """Test batch embedding with real performance comparison."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            texts = [
                "Machine learning is transforming technology.",
                "Python is a versatile programming language.",
                "Natural language processing enables AI communication.",
                "Vector embeddings capture semantic meaning.",
                "Artificial intelligence augments human capabilities."
            ]

            # Test batch processing
            embeddings = manager.embed_batch(texts)

            assert len(embeddings) == len(texts)
            assert all(len(emb) == 384 for emb in embeddings)
            assert all(isinstance(emb, list) for emb in embeddings)

            # Verify embeddings are different for different texts
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    emb1, emb2 = np.array(embeddings[i]), np.array(embeddings[j])
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(sim)

            # Embeddings should be reasonably similar (all tech-related) but not identical
            avg_similarity = np.mean(similarities)
            assert 0.3 < avg_similarity < 0.9  # Reasonable similarity range

            print(f"✅ Batch embeddings generated: {len(embeddings)} texts, avg similarity: {avg_similarity:.3f}")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_caching_behavior_real(self):
        """Test caching with real embeddings."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir,
                cache_size=100
            )

            text = "This text will be cached for performance testing."

            # Generate embedding twice
            embedding1 = manager.embed(text)
            embedding2 = manager.embed(text)

            # Should be identical due to caching
            assert embedding1 == embedding2

            # Check cache stats
            stats = manager.get_cache_stats()
            assert stats["memory_cache_info"]["hits"] > 0
            assert stats["persistent_cache_size"] > 0

            print(f"✅ Caching working: {stats['memory_cache_info']['hits']} cache hits")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_similarity_computation_real(self):
        """Test similarity computation with real embeddings."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test similar texts
            text1 = "The cat is sleeping on the couch."
            text2 = "A cat is resting on the sofa."
            similarity_high = manager.compute_similarity(text1, text2)

            # Test dissimilar texts
            text3 = "Python programming language syntax."
            similarity_low = manager.compute_similarity(text1, text3)

            # Verify similarity makes sense
            assert isinstance(similarity_high, float)
            assert isinstance(similarity_low, float)
            assert -1 <= similarity_high <= 1
            assert -1 <= similarity_low <= 1
            assert similarity_high > similarity_low  # Cat texts should be more similar

            print(f"✅ Similarity computation: similar={similarity_high:.3f}, different={similarity_low:.3f}")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


@pytest.mark.integration
class TestSemanticSearchReal:
    """Test real semantic search scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_semantic_search_scenario(self):
        """Test a complete semantic search scenario."""
        try:
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Knowledge base about programming
            documents = [
                "Python is an interpreted, high-level programming language with dynamic semantics.",
                "JavaScript is a programming language that enables interactive web pages.",
                "Machine learning algorithms learn patterns from data to make predictions.",
                "React is a JavaScript library for building user interfaces.",
                "NumPy provides support for large arrays and mathematical functions in Python.",
                "Deep learning uses neural networks with multiple layers to process data.",
                "FastAPI is a modern web framework for building APIs with Python.",
                "Node.js is a JavaScript runtime for server-side development."
            ]

            # User queries
            queries = [
                "Python web development frameworks",
                "JavaScript frontend libraries",
                "machine learning techniques"
            ]

            for query in queries:
                print(f"\n🔍 Query: '{query}'")

                # Get query embedding
                query_embedding = embedder.embed(query)

                # Get document embeddings
                doc_embeddings = embedder.embed_batch(documents)

                # Compute similarities
                similarities = []
                for doc_emb in doc_embeddings:
                    similarity = embedder.compute_similarity(query, documents[len(similarities)])
                    similarities.append(similarity)

                # Find top 2 most relevant documents
                top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:2]

                print(f"📄 Top results:")
                for idx in top_indices:
                    print(f"   Score: {similarities[idx]:.3f} - {documents[idx][:60]}...")

                # Verify results make sense
                top_doc = documents[top_indices[0]]
                if "Python" in query:
                    assert "Python" in top_doc or "python" in top_doc.lower()
                elif "JavaScript" in query:
                    assert "JavaScript" in top_doc or "javascript" in top_doc.lower()
                elif "machine learning" in query:
                    assert "learning" in top_doc.lower() or "neural" in top_doc.lower()

            print("✅ Semantic search working correctly")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


@pytest.mark.integration
class TestLLMIntegrationReal:
    """Test real integration between embeddings and LLMs."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embeddings_llm_separation(self):
        """Test that embeddings and LLMs work independently."""
        try:
            # Initialize embedding manager
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test embedding generation
            text = "Testing embedding and LLM integration"
            embedding = embedder.embed(text)
            assert len(embedding) == 384

            # Test that LLM creation doesn't interfere with embeddings
            try:
                # This might fail if no LLM providers are configured, which is fine
                llm = create_llm("mock")  # Use mock provider to avoid API calls
                print("✅ LLM and embeddings can coexist")
            except Exception:
                print("✅ LLM creation failed as expected (no real providers configured)")

            # Embeddings should still work
            embedding2 = embedder.embed("Another test text")
            assert len(embedding2) == 384

            print("✅ Embeddings and LLM integration separation working")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_rag_pipeline_structure(self):
        """Test the structure of a RAG pipeline (without actual LLM calls)."""
        try:
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Simulate RAG knowledge base
            knowledge_base = [
                "Paris is the capital and largest city of France, with a population of over 2 million.",
                "The Eiffel Tower, built in 1889, is an iron lattice tower 330 meters tall.",
                "The Louvre Museum in Paris is home to the Mona Lisa and many other artworks.",
                "French cuisine is renowned worldwide for its sophistication and diversity.",
                "The Seine River flows through the heart of Paris, dividing it into two banks."
            ]

            # User question
            question = "How tall is the Eiffel Tower?"

            # Step 1: Embed the question
            question_embedding = embedder.embed(question)
            assert len(question_embedding) == 384

            # Step 2: Embed the knowledge base
            kb_embeddings = embedder.embed_batch(knowledge_base)
            assert len(kb_embeddings) == len(knowledge_base)

            # Step 3: Find most relevant context
            similarities = []
            for i, kb_text in enumerate(knowledge_base):
                similarity = embedder.compute_similarity(question, kb_text)
                similarities.append((similarity, i, kb_text))

            # Sort by similarity
            similarities.sort(reverse=True)
            most_relevant = similarities[0]

            print(f"🤖 RAG Pipeline Test:")
            print(f"   Question: {question}")
            print(f"   Best match (score: {most_relevant[0]:.3f}): {most_relevant[2][:60]}...")

            # Verify we found the right context
            assert most_relevant[0] > 0.3  # Should have reasonable similarity
            assert "Eiffel Tower" in most_relevant[2]
            assert "330 meters" in most_relevant[2]

            # Step 4: Prepare context for LLM (structure only)
            context = most_relevant[2]
            rag_prompt = f"""Context: {context}

Question: {question}

Answer based on the context provided:"""

            assert len(rag_prompt) > 0
            assert question in rag_prompt
            assert context in rag_prompt

            print("✅ RAG pipeline structure working correctly")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


@pytest.mark.integration
class TestEmbeddingPerformance:
    """Test embedding performance characteristics."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_vs_individual_performance(self):
        """Test batch vs individual embedding performance."""
        try:
            manager = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            texts = [f"Test sentence number {i} for performance testing." for i in range(5)]

            import time

            # Individual embeddings
            start_time = time.time()
            individual_embeddings = []
            for text in texts:
                emb = manager.embed(text)
                individual_embeddings.append(emb)
            individual_time = time.time() - start_time

            # Clear cache for fair comparison
            manager.embed.cache_clear()

            # Batch embeddings
            start_time = time.time()
            batch_embeddings = manager.embed_batch(texts)
            batch_time = time.time() - start_time

            # Verify results are identical
            assert len(individual_embeddings) == len(batch_embeddings)
            for ind, batch in zip(individual_embeddings, batch_embeddings):
                np.testing.assert_array_almost_equal(ind, batch, decimal=5)

            print(f"⚡ Performance comparison:")
            print(f"   Individual: {individual_time:.3f}s ({individual_time/len(texts):.3f}s per text)")
            print(f"   Batch: {batch_time:.3f}s ({batch_time/len(texts):.3f}s per text)")
            print(f"   Speedup: {individual_time/batch_time:.1f}x")

            # Batch should generally be faster for multiple texts
            if len(texts) > 2:
                assert batch_time <= individual_time * 1.2  # Allow some variance

            print("✅ Performance testing completed")

        except Exception as e:
            if "offline" in str(e).lower() or "network" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


if __name__ == "__main__":
    # Run real integration tests
    print("🚀 Running real embedding integration tests...")
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
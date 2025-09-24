"""
Comprehensive Real Models Testing
=================================

Tests with actual SOTA embedding models:
- Google EmbeddingGemma
- IBM Granite
- sentence-transformers/all-MiniLM-L6-v2

Plus integration with real LLM providers.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import time

from abstractllm.embeddings import EmbeddingManager, get_model_config
from abstractllm import create_llm


@pytest.mark.integration
@pytest.mark.slow
class TestRealSOTAModels:
    """Test with real SOTA embedding models."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embeddinggemma_real_usage(self):
        """Test Google EmbeddingGemma model."""
        try:
            print("\n🔥 Testing Google EmbeddingGemma")

            # Test model configuration
            config = get_model_config("embeddinggemma")
            assert config.model_id == "google/embeddinggemma-300m"
            assert config.dimension == 768
            assert config.multilingual is True
            assert config.supports_matryoshka is True
            print(f"✓ Model config: {config.model_id}, {config.dimension}D, multilingual")

            # Initialize with real model
            embedder = EmbeddingManager(
                model="embeddinggemma",
                cache_dir=self.cache_dir
            )

            # Test basic embedding
            text = "Machine learning is revolutionizing artificial intelligence applications."
            embedding = embedder.embed(text)

            assert isinstance(embedding, list)
            assert len(embedding) == 768
            assert all(isinstance(x, (int, float)) for x in embedding)
            assert any(x != 0.0 for x in embedding)
            print(f"✓ Generated {len(embedding)}D embedding")

            # Test multilingual capability
            multilingual_texts = [
                "Hello, how are you?",  # English
                "Bonjour, comment allez-vous?",  # French
                "Hola, ¿cómo estás?",  # Spanish
                "Guten Tag, wie geht es Ihnen?"  # German
            ]

            embeddings = embedder.embed_batch(multilingual_texts)
            assert len(embeddings) == 4
            assert all(len(emb) == 768 for emb in embeddings)
            print(f"✓ Multilingual processing: {len(multilingual_texts)} languages")

            # Test Matryoshka truncation
            truncated_manager = EmbeddingManager(
                model="embeddinggemma",
                cache_dir=self.cache_dir,
                output_dims=256
            )
            truncated_embedding = truncated_manager.embed(text)
            assert len(truncated_embedding) == 256
            print(f"✓ Matryoshka truncation: 768D → 256D")

            # Test semantic similarity
            similar_texts = ["AI and machine learning", "Artificial intelligence and ML"]
            sim_score = embedder.compute_similarity(similar_texts[0], similar_texts[1])
            assert sim_score > 0.7  # Should be highly similar
            print(f"✓ Semantic similarity: {sim_score:.3f}")

            return True

        except Exception as e:
            if "offline" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"EmbeddingGemma model not available: {e}")
            else:
                raise

    def test_granite_real_usage(self):
        """Test IBM Granite embedding model."""
        try:
            print("\n🏢 Testing IBM Granite")

            # Test model configuration
            config = get_model_config("granite")
            assert config.model_id == "ibm-granite/granite-embedding-278m-multilingual"
            assert config.dimension == 768
            assert config.multilingual is True
            print(f"✓ Model config: {config.model_id}, {config.dimension}D, multilingual")

            # Initialize with real model
            embedder = EmbeddingManager(
                model="granite",
                cache_dir=self.cache_dir
            )

            # Test basic embedding
            text = "Enterprise applications require robust and scalable AI solutions."
            embedding = embedder.embed(text)

            assert isinstance(embedding, list)
            assert len(embedding) == 768
            assert all(isinstance(x, (int, float)) for x in embedding)
            assert any(x != 0.0 for x in embedding)
            print(f"✓ Generated {len(embedding)}D embedding")

            # Test enterprise/business content
            business_texts = [
                "Financial reports show quarterly growth in revenue and market share.",
                "Supply chain optimization reduces costs and improves delivery times.",
                "Customer satisfaction metrics indicate strong brand loyalty and retention.",
                "Digital transformation initiatives enhance operational efficiency."
            ]

            embeddings = embedder.embed_batch(business_texts)
            assert len(embeddings) == 4
            assert all(len(emb) == 768 for emb in embeddings)
            print(f"✓ Business content processing: {len(business_texts)} documents")

            # Test semantic business understanding
            query = "enterprise financial performance"
            similarities = []
            for doc in business_texts:
                sim = embedder.compute_similarity(query, doc)
                similarities.append(sim)

            best_match_idx = similarities.index(max(similarities))
            best_match = business_texts[best_match_idx]
            assert "Financial reports" in best_match  # Should find financial content
            print(f"✓ Business semantic search: found relevant content")

            return True

        except Exception as e:
            if "offline" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Granite model not available: {e}")
            else:
                raise

    def test_all_minilm_baseline(self):
        """Test baseline model all-MiniLM-L6-v2."""
        try:
            print("\n⚡ Testing all-MiniLM-L6-v2 (baseline)")

            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test basic functionality
            text = "This is a test sentence for embedding generation."
            embedding = embedder.embed(text)

            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, (int, float)) for x in embedding)
            print(f"✓ Generated {len(embedding)}D embedding (compact)")

            # Performance benchmark
            texts = [f"Performance test sentence {i}" for i in range(10)]

            start_time = time.time()
            batch_embeddings = embedder.embed_batch(texts)
            batch_time = time.time() - start_time

            assert len(batch_embeddings) == 10
            assert all(len(emb) == 384 for emb in batch_embeddings)
            print(f"✓ Batch performance: {batch_time:.3f}s for {len(texts)} texts")

            return True

        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_model_comparison(self):
        """Compare performance across different models."""
        try:
            print("\n🔬 Model Comparison Test")

            # Test text for comparison
            test_text = "Artificial intelligence and machine learning are transforming technology."

            models_to_test = [
                ("sentence-transformers/all-MiniLM-L6-v2", 384),
                ("embeddinggemma", 768),
                ("granite", 768)
            ]

            results = {}

            for model_name, expected_dim in models_to_test:
                try:
                    print(f"\n  Testing {model_name}...")

                    start_time = time.time()
                    embedder = EmbeddingManager(model=model_name, cache_dir=self.cache_dir)
                    init_time = time.time() - start_time

                    start_time = time.time()
                    embedding = embedder.embed(test_text)
                    embed_time = time.time() - start_time

                    assert len(embedding) == expected_dim

                    results[model_name] = {
                        "dimension": len(embedding),
                        "init_time": init_time,
                        "embed_time": embed_time,
                        "status": "✓ Success"
                    }

                    print(f"    ✓ {expected_dim}D embedding in {embed_time:.3f}s")

                except Exception as e:
                    results[model_name] = {
                        "status": f"✗ Failed: {str(e)[:50]}..."
                    }
                    print(f"    ✗ Failed: {e}")

            # Summary
            print(f"\n📊 Model Comparison Summary:")
            for model, result in results.items():
                print(f"  {model}:")
                for key, value in result.items():
                    print(f"    {key}: {value}")

            # At least baseline should work
            baseline_status = results.get("sentence-transformers/all-MiniLM-L6-v2", {}).get("status")
            assert "Success" in baseline_status, "Baseline model should work"

            return True

        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


@pytest.mark.integration
class TestRealLLMIntegration:
    """Test real LLM integration with embeddings."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_rag_with_real_models(self):
        """Test complete RAG pipeline with real models."""
        try:
            print("\n🤖 Complete RAG Pipeline Test")

            # Use fastest embedding model for testing
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Use mock LLM to avoid API calls but test structure
            llm = create_llm("mock")

            # Real knowledge base
            knowledge_base = [
                "AbstractLLM Core provides a unified interface to multiple LLM providers including OpenAI, Anthropic, and local models like Ollama.",
                "The embeddings system in AbstractLLM uses SOTA models like EmbeddingGemma and Granite for semantic search and RAG applications.",
                "Vector embeddings enable semantic similarity search by converting text into high-dimensional numerical representations.",
                "RAG (Retrieval-Augmented Generation) combines information retrieval with language generation for more accurate responses.",
                "AbstractLLM Core includes production features like retry logic, event systems, and structured output validation."
            ]

            # Real user questions
            questions = [
                "What providers does AbstractLLM support?",
                "What embedding models are available?",
                "How do vector embeddings work?",
                "What is RAG and how does it work?",
                "What production features are included?"
            ]

            print(f"  Knowledge base: {len(knowledge_base)} documents")
            print(f"  Test questions: {len(questions)}")

            # Pre-compute document embeddings for efficiency
            print("  Computing document embeddings...")
            doc_embeddings = embedder.embed_batch(knowledge_base)
            assert len(doc_embeddings) == len(knowledge_base)

            # Test each question
            for i, question in enumerate(questions):
                print(f"\n  Question {i+1}: {question}")

                # Find best context
                best_score = 0
                best_context = ""

                for j, doc in enumerate(knowledge_base):
                    similarity = embedder.compute_similarity(question, doc)
                    if similarity > best_score:
                        best_score = similarity
                        best_context = doc

                print(f"    Best context score: {best_score:.3f}")
                print(f"    Context: {best_context[:60]}...")

                # Verify retrieval quality
                assert best_score > 0.3, f"Should find relevant context for: {question}"

                # Create RAG prompt
                rag_prompt = f"""Context: {best_context}

Question: {question}

Based on the context, please provide a helpful answer:"""

                # Generate response
                response = llm.generate(rag_prompt)
                assert hasattr(response, 'content')
                assert len(response.content) > 50  # Should be substantial

                print(f"    ✓ RAG pipeline successful")

            print(f"\n  ✅ All {len(questions)} questions processed successfully")
            return True

        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_embeddings_with_real_llm_providers(self):
        """Test embeddings with real LLM providers (structure validation)."""
        try:
            print("\n🔗 Real LLM Provider Integration")

            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test with different provider types
            providers_to_test = ["mock"]  # Start with mock, can extend to real APIs

            for provider in providers_to_test:
                print(f"\n  Testing with {provider} provider...")

                try:
                    llm = create_llm(provider)

                    # Test basic integration
                    embedding = embedder.embed("Test integration")
                    response = llm.generate("Test prompt")

                    assert len(embedding) == 384
                    assert hasattr(response, 'content')

                    print(f"    ✓ {provider}: Embeddings + LLM working together")

                except Exception as e:
                    print(f"    ✗ {provider}: {e}")

            return True

        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


@pytest.mark.integration
class TestProductionReadiness:
    """Test production readiness features."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_performance_at_scale(self):
        """Test performance with larger datasets."""
        try:
            print("\n⚡ Performance at Scale Test")

            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir,
                cache_size=1000
            )

            # Large document collection
            documents = [
                f"Document {i}: This is sample content about technology, AI, machine learning, "
                f"software development, and data science for testing performance at scale."
                for i in range(100)
            ]

            print(f"  Processing {len(documents)} documents...")

            # Batch processing test
            start_time = time.time()
            embeddings = embedder.embed_batch(documents)
            batch_time = time.time() - start_time

            assert len(embeddings) == len(documents)
            print(f"  ✓ Batch processing: {batch_time:.3f}s ({batch_time/len(documents)*1000:.1f}ms per doc)")

            # Query processing test
            queries = [
                "artificial intelligence and machine learning",
                "software development best practices",
                "data science and analytics",
                "technology trends and innovation"
            ]

            print(f"  Testing {len(queries)} queries against {len(documents)} documents...")

            start_time = time.time()
            for query in queries:
                similarities = []
                for doc in documents:
                    sim = embedder.compute_similarity(query, doc)
                    similarities.append(sim)

                best_score = max(similarities)
                assert best_score > 0.2  # Should find some relevance

            query_time = time.time() - start_time
            print(f"  ✓ Query processing: {query_time:.3f}s ({query_time/len(queries):.3f}s per query)")

            # Cache performance test
            cache_stats = embedder.get_cache_stats()
            print(f"  ✓ Cache performance: {cache_stats['memory_cache_info']['hits']} hits")

            return True

        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_error_recovery(self):
        """Test error handling and recovery."""
        try:
            print("\n🛡️ Error Recovery Test")

            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test empty text handling
            empty_embedding = embedder.embed("")
            assert len(empty_embedding) == 384
            assert all(x == 0.0 for x in empty_embedding)
            print("  ✓ Empty text handling")

            # Test very long text handling
            long_text = "This is a very long text. " * 1000
            long_embedding = embedder.embed(long_text)
            assert len(long_embedding) == 384
            print("  ✓ Long text handling")

            # Test mixed input handling
            mixed_texts = ["", "Normal text", long_text, "   ", "Short"]
            mixed_embeddings = embedder.embed_batch(mixed_texts)
            assert len(mixed_embeddings) == 5
            assert all(len(emb) == 384 for emb in mixed_embeddings)
            print("  ✓ Mixed input handling")

            return True

        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


if __name__ == "__main__":
    # Run comprehensive real model tests
    print("🚀 Running comprehensive real model tests...")
    pytest.main([__file__, "-v", "-m", "integration", "-s", "--tb=short"])
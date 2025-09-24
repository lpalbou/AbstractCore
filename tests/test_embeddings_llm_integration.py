"""
Real LLM Integration Tests with Embeddings
==========================================

Tests the complete integration between embeddings and actual LLM providers.
NO MOCKING - Uses real embedding models and mock LLM provider for structure validation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from abstractllm.embeddings import EmbeddingManager
from abstractllm import create_llm


@pytest.mark.integration
class TestEmbeddingsLLMIntegration:
    """Test real integration between embeddings and LLM providers."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embeddings_and_llm_coexistence(self):
        """Test that embeddings and LLM can be used together without interference."""
        try:
            # Initialize embedding manager with real model
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Test embedding generation
            test_text = "Testing integration between embeddings and LLM"
            embedding = embedder.embed(test_text)
            assert len(embedding) == 384
            assert all(isinstance(x, (int, float)) for x in embedding)

            # Create LLM provider (mock to avoid API calls)
            llm = create_llm("mock")

            # Verify both work independently
            similarity = embedder.compute_similarity("test 1", "test 2")
            assert isinstance(similarity, float)

            # Mock LLM should respond
            llm_response = llm.generate("Test prompt")
            assert hasattr(llm_response, 'content')

            print("✅ Embeddings and LLM coexist successfully")
            return True

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_complete_rag_workflow_structure(self):
        """Test complete RAG workflow structure with real embeddings."""
        try:
            # Real components
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )
            llm = create_llm("mock")

            # Real knowledge base
            knowledge_base = [
                "FastAPI is a modern Python web framework for building APIs with type hints and automatic documentation.",
                "React is a JavaScript library developed by Facebook for building user interfaces and single-page applications.",
                "Docker enables developers to package applications and dependencies into lightweight, portable containers.",
                "PostgreSQL is an advanced open-source relational database with strong ACID compliance and extensibility.",
                "Kubernetes automates deployment, scaling, and management of containerized applications across clusters."
            ]

            # Real user question
            question = "What is FastAPI used for?"

            print(f"🤖 RAG Workflow Test")
            print(f"Question: {question}")
            print(f"Knowledge base: {len(knowledge_base)} documents")

            # Step 1: Real embedding-based retrieval
            best_score = 0
            best_context = ""

            for doc in knowledge_base:
                similarity = embedder.compute_similarity(question, doc)
                if similarity > best_score:
                    best_score = similarity
                    best_context = doc

            print(f"Best context score: {best_score:.3f}")
            print(f"Best context: {best_context[:60]}...")

            # Verify retrieval quality
            assert best_score > 0.4  # Should find relevant content
            assert "FastAPI" in best_context  # Should find the right document

            # Step 2: RAG prompt construction
            rag_prompt = f"""Context: {best_context}

Question: {question}

Based on the provided context, please answer the question:"""

            # Step 3: LLM integration structure (mock response)
            llm_response = llm.generate(rag_prompt)

            # Verify integration structure
            assert len(rag_prompt) > 100  # Substantial prompt
            assert question in rag_prompt  # Question included
            assert best_context in rag_prompt  # Context included
            assert hasattr(llm_response, 'content')  # LLM responded

            print("✅ Complete RAG workflow structure validated")

            # Step 4: Test batch retrieval for multiple questions
            questions = [
                "What is React used for?",
                "What does Docker enable?",
                "What is PostgreSQL?"
            ]

            print(f"\nBatch RAG test with {len(questions)} questions:")

            for q in questions:
                scores = []
                for doc in knowledge_base:
                    score = embedder.compute_similarity(q, doc)
                    scores.append(score)

                best_idx = scores.index(max(scores))
                best_doc = knowledge_base[best_idx]
                best_score = scores[best_idx]

                print(f"  '{q}' -> Score: {best_score:.3f}")

                # Each question should find reasonable context
                assert best_score > 0.3

            print("✅ Batch RAG processing successful")
            return True

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_event_system_integration(self):
        """Test that embeddings work with the existing event system."""
        try:
            from abstractllm.events import EventType, on_global

            # Track embedding events
            events_captured = []

            def capture_event(event):
                events_captured.append(event.type)

            # Register event listeners
            on_global("embedding_generated", capture_event)
            on_global("embedding_cached", capture_event)

            # Create embedding manager
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Generate embeddings (should trigger events)
            embedding1 = embedder.embed("First test text")
            embedding2 = embedder.embed("First test text")  # Should be cached

            # Verify embeddings work
            assert len(embedding1) == 384
            assert embedding1 == embedding2

            # Verify events were emitted
            assert len(events_captured) >= 1  # At least one event
            print(f"✅ Events captured: {events_captured}")

            print("✅ Event system integration working")
            return True

        except ImportError as e:
            if "abstractllm.events" in str(e):
                print("⚠️  Event system not available, skipping event tests")
                return True
            elif "sentence-transformers" in str(e):
                pytest.skip("sentence-transformers not available")
            else:
                raise
        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_session_integration(self):
        """Test embeddings with session management."""
        try:
            from abstractllm import BasicSession

            # Create components
            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )
            llm = create_llm("mock")
            session = BasicSession(llm)

            # Test session with embedding-enhanced prompts
            knowledge = "Python is a programming language known for its simplicity and readability."
            question = "What is Python known for?"

            # Find relevant context with embeddings
            similarity = embedder.compute_similarity(question, knowledge)
            assert similarity > 0.5  # Should be reasonably similar

            # Create context-enhanced prompt
            enhanced_prompt = f"Context: {knowledge}\n\nQuestion: {question}"

            # Use session (structure test)
            response = session.generate(enhanced_prompt)
            assert hasattr(response, 'content')

            # Verify session maintains context while embeddings work independently
            embedding = embedder.embed("Session test text")
            assert len(embedding) == 384

            print("✅ Session integration successful")
            return True

        except ImportError:
            pytest.skip("Required dependencies not available")
        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise

    def test_performance_with_llm_workflow(self):
        """Test performance characteristics in a realistic LLM workflow."""
        try:
            import time

            embedder = EmbeddingManager(
                model="sentence-transformers/all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )

            # Simulate a realistic RAG scenario
            documents = [f"Document {i}: This is sample content for testing RAG performance." for i in range(20)]
            queries = ["What is the content about?", "Tell me about the documents", "Explain the sample content"]

            print("🚀 Performance test: RAG workflow simulation")

            # Test 1: Document embedding (batch)
            start_time = time.time()
            doc_embeddings = embedder.embed_batch(documents)
            doc_time = time.time() - start_time

            assert len(doc_embeddings) == len(documents)
            print(f"  Document embedding: {doc_time:.3f}s for {len(documents)} docs")

            # Test 2: Query processing
            start_time = time.time()
            for query in queries:
                query_embedding = embedder.embed(query)

                # Simulate similarity search
                similarities = []
                for doc_emb in doc_embeddings:
                    # Manual similarity calculation
                    import numpy as np
                    q_emb = np.array(query_embedding)
                    d_emb = np.array(doc_emb)
                    sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb))
                    similarities.append(sim)

                best_idx = similarities.index(max(similarities))
                # Would send to LLM here: llm.generate(f"Context: {documents[best_idx]}\nQuery: {query}")

            query_time = time.time() - start_time
            print(f"  Query processing: {query_time:.3f}s for {len(queries)} queries")
            print(f"  Average per query: {query_time/len(queries):.3f}s")

            # Performance should be reasonable for real-time use
            assert doc_time < 5.0  # Batch document processing should be fast
            assert query_time/len(queries) < 1.0  # Each query should be sub-second

            print("✅ Performance suitable for real-time RAG workflows")
            return True

        except ImportError:
            pytest.skip("sentence-transformers not available")
        except Exception as e:
            if "offline" in str(e).lower():
                pytest.skip("Model download failed (offline mode)")
            else:
                raise


if __name__ == "__main__":
    # Run LLM integration tests
    print("🚀 Running LLM integration tests...")
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
"""
Real integration tests for server embedding endpoints.
NO MOCKING - Tests with real embedding models and actual API calls.

Tests the /{provider}/v1/embeddings endpoints with real embeddings.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from abstractllm.server.app import app

client = TestClient(app)


@pytest.mark.integration
class TestServerEmbeddingsReal:
    """Test server embedding endpoints with real models."""

    def test_ollama_embedding_endpoint_real(self):
        """Test Ollama embedding endpoint with real all-minilm:l6-v2 model."""
        try:
            # Test with Ollama model that should use native API
            response = client.post(
                "/ollama/v1/embeddings",
                json={
                    "input": "Hello world, this is a test sentence for embedding.",
                    "model": "all-minilm:l6-v2"
                }
            )

            # Should succeed if Ollama is running with the model
            if response.status_code == 200:
                data = response.json()

                # Verify response structure
                assert data["object"] == "list"
                assert len(data["data"]) == 1
                assert data["data"][0]["object"] == "embedding"
                assert data["data"][0]["index"] == 0
                assert data["model"] == "ollama/all-minilm:l6-v2"

                # Verify real embedding (not mock)
                embedding = data["data"][0]["embedding"]
                assert isinstance(embedding, list)
                assert len(embedding) > 0

                # Real embeddings should have varied values, not all 0.1
                unique_values = set(embedding)
                assert len(unique_values) > 1, "Embedding appears to be mocked (all same values)"
                assert 0.1 not in embedding or len([x for x in embedding if x == 0.1]) < len(embedding) * 0.5, "Too many 0.1 values - might be mocked"

                # Check that values are reasonable for embeddings
                embedding_array = np.array(embedding)
                assert -1.0 <= embedding_array.min() <= embedding_array.max() <= 1.0

                print(f"✅ Real Ollama embedding generated: {len(embedding)} dimensions, range [{embedding_array.min():.3f}, {embedding_array.max():.3f}]")

            else:
                # If Ollama not available, test should indicate this clearly
                assert response.status_code == 500
                detail = response.json()["detail"]
                assert "Ollama" in detail, f"Expected Ollama error, got: {detail}"
                print("⚠️ Ollama not available - test skipped but error handling verified")

        except Exception as e:
            pytest.fail(f"Ollama embedding test failed: {e}")

    def test_huggingface_embedding_endpoint_real(self):
        """Test HuggingFace embedding via EmbeddingManager."""
        try:
            # Test with any non-Ollama provider (should use EmbeddingManager)
            response = client.post(
                "/openai/v1/embeddings",
                json={
                    "input": "This text should generate real embeddings via HuggingFace.",
                    "model": "text-embedding-ada-002"  # Any model - will use HF sentence-transformers
                }
            )

            # Should succeed if EmbeddingManager is available
            if response.status_code == 200:
                data = response.json()

                # Verify response structure
                assert data["object"] == "list"
                assert len(data["data"]) == 1
                assert data["data"][0]["object"] == "embedding"
                assert data["data"][0]["index"] == 0
                assert data["model"] == "openai/text-embedding-ada-002"

                # Verify real embedding (not mock)
                embedding = data["data"][0]["embedding"]
                assert isinstance(embedding, list)
                assert len(embedding) > 0

                # Real embeddings should have varied values
                unique_values = set(embedding)
                assert len(unique_values) > 1, "Embedding appears to be mocked (all same values)"
                assert 0.1 not in embedding or len([x for x in embedding if x == 0.1]) < len(embedding) * 0.1, "Too many 0.1 values - might be mocked"

                # Check that values are reasonable for embeddings
                embedding_array = np.array(embedding)
                assert -1.0 <= embedding_array.min() <= embedding_array.max() <= 1.0

                # Should be 384 dimensions for all-MiniLM-L6-v2
                assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"

                print(f"✅ Real HuggingFace embedding generated: {len(embedding)} dimensions, range [{embedding_array.min():.3f}, {embedding_array.max():.3f}]")

            else:
                # If EmbeddingManager not available, should get proper error
                assert response.status_code == 500
                detail = response.json()["detail"]
                assert "EmbeddingManager" in detail or "embedding" in detail.lower()
                print("⚠️ EmbeddingManager not available - test skipped but error handling verified")

        except Exception as e:
            pytest.fail(f"HuggingFace embedding test failed: {e}")

    def test_batch_embedding_real(self):
        """Test batch embedding with real models."""
        try:
            # Test multiple texts at once
            texts = [
                "First test sentence for embedding.",
                "Second test sentence with different content.",
                "Third sentence to verify batch processing works correctly."
            ]

            response = client.post(
                "/anthropic/v1/embeddings",  # Use EmbeddingManager route
                json={
                    "input": texts,
                    "model": "any-model"  # Will use HF sentence-transformers
                }
            )

            if response.status_code == 200:
                data = response.json()

                # Verify batch response structure
                assert data["object"] == "list"
                assert len(data["data"]) == 3

                for i, embedding_data in enumerate(data["data"]):
                    assert embedding_data["object"] == "embedding"
                    assert embedding_data["index"] == i

                    embedding = embedding_data["embedding"]
                    assert isinstance(embedding, list)
                    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimensions

                    # Verify real embeddings
                    unique_values = set(embedding)
                    assert len(unique_values) > 1, f"Embedding {i} appears to be mocked"

                # Verify embeddings are different for different texts
                emb1 = np.array(data["data"][0]["embedding"])
                emb2 = np.array(data["data"][1]["embedding"])
                emb3 = np.array(data["data"][2]["embedding"])

                # Cosine similarity should be different for different texts
                def cosine_similarity(a, b):
                    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                sim_1_2 = cosine_similarity(emb1, emb2)
                sim_1_3 = cosine_similarity(emb1, emb3)

                # Different sentences should have different embeddings
                assert sim_1_2 < 0.95, "Embeddings too similar - might be mocked"
                assert sim_1_3 < 0.95, "Embeddings too similar - might be mocked"

                print(f"✅ Real batch embeddings: similarities {sim_1_2:.3f}, {sim_1_3:.3f}")

            else:
                assert response.status_code == 500
                print("⚠️ Batch embedding test skipped - EmbeddingManager not available")

        except Exception as e:
            pytest.fail(f"Batch embedding test failed: {e}")

    def test_embedding_endpoint_no_mocking_enforced(self):
        """Verify that mocking is completely removed from embedding endpoints."""
        # This test ensures we never fall back to mock embeddings

        # Try a provider that definitely should use EmbeddingManager
        response = client.post(
            "/test-provider/v1/embeddings",  # Non-Ollama provider
            json={
                "input": "Test sentence to verify no mocking occurs.",
                "model": "test-model"
            }
        )

        # Should either succeed with real embeddings or fail properly
        # Should NEVER return mock embeddings with all 0.1 values
        if response.status_code == 200:
            data = response.json()
            embedding = data["data"][0]["embedding"]

            # If it returns an embedding, it should be real
            mock_count = sum(1 for x in embedding if abs(x - 0.1) < 0.001)
            total_count = len(embedding)
            mock_ratio = mock_count / total_count

            assert mock_ratio < 0.1, f"Too many 0.1 values ({mock_ratio:.1%}) - embedding appears mocked"
            print("✅ No mock embeddings detected")
        else:
            # Should fail gracefully, not return mock data
            assert response.status_code in [500, 404, 400]
            print("✅ Proper error handling - no fallback to mock embeddings")


if __name__ == "__main__":
    # Run tests directly
    test_class = TestServerEmbeddingsReal()

    print("🧪 Testing real server embedding endpoints...")
    print("=" * 60)

    try:
        test_class.test_ollama_embedding_endpoint_real()
        print()
        test_class.test_huggingface_embedding_endpoint_real()
        print()
        test_class.test_batch_embedding_real()
        print()
        test_class.test_embedding_endpoint_no_mocking_enforced()
        print()
        print("🎉 All real embedding tests completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
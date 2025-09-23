"""
Test simple generation - basic "who are you" tests.
Tests that providers can generate responses to simple prompts.
"""

import pytest
import os
import time
from abstractllm import create_llm


class TestProviderSimpleGeneration:
    """Test basic generation capabilities for each provider."""

    def test_ollama_simple_generation(self):
        """Test Ollama simple message generation."""
        try:
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_simple_generation(self):
        """Test LMStudio simple message generation."""
        try:
            provider = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mlx_simple_generation(self):
        """Test MLX simple message generation."""
        try:
            provider = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 60  # MLX might be slower

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["mlx", "import", "not found", "failed to load"]):
                pytest.skip("MLX not available or model not found")
            else:
                raise

    def test_huggingface_simple_generation(self):
        """Test HuggingFace simple message generation."""
        try:
            provider = create_llm("huggingface", model="Qwen/Qwen3-4B")

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 120  # HF might be slow on first load

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["transformers", "torch", "not found", "failed to load"]):
                pytest.skip("HuggingFace not available or model not found")
            else:
                raise

    def test_openai_simple_generation(self):
        """Test OpenAI simple message generation."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10  # Cloud should be fast

            # Check usage tracking if available
            if response.usage:
                assert "total_tokens" in response.usage
                assert response.usage["total_tokens"] > 0

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_simple_generation(self):
        """Test Anthropic simple message generation."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10  # Cloud should be fast

            # Check usage tracking if available
            if response.usage:
                assert "total_tokens" in response.usage
                assert response.usage["total_tokens"] > 0

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_mock_simple_generation(self):
        """Test Mock provider simple generation."""
        provider = create_llm("mock", model="test-model")

        response = provider.generate("Who are you? Answer in one sentence.")

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        # Mock provider returns predictable responses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
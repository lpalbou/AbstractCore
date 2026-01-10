"""
Test simple generation - basic "who are you" tests.
Tests that providers can generate responses to simple prompts.
"""

import pytest
import os
import time
from abstractcore import create_llm


class TestProviderSimpleGeneration:
    """Test basic generation capabilities for each provider."""

    @staticmethod
    def _looks_like_connectivity_issue(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            keyword in msg
            for keyword in [
                "connection",
                "refused",
                "timeout",
                "operation not permitted",
            ]
        )

    def test_ollama_simple_generation(self):
        """Test Ollama simple message generation."""
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            provider = create_llm(
                "ollama",
                model="qwen3:4b-instruct",
                base_url="http://localhost:11434",
                timeout=10.0,
            )

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            if self._looks_like_connectivity_issue(e):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_simple_generation(self):
        """Test LMStudio simple message generation."""
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            provider = create_llm(
                "lmstudio",
                model="qwen/qwen3-4b-2507",
                base_url="http://localhost:1234/v1",
                timeout=10.0,
            )

            start = time.time()
            response = provider.generate("Who are you? Answer in one sentence.")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            if self._looks_like_connectivity_issue(e):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mlx_simple_generation(self):
        """Test MLX simple message generation."""
        if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS") != "1":
            pytest.skip("MLX generation test is heavy; set ABSTRACTCORE_RUN_MLX_TESTS=1 to run")
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
        if os.getenv("ABSTRACTCORE_RUN_HUGGINGFACE_TESTS") != "1":
            pytest.skip("HuggingFace generation test is heavy; set ABSTRACTCORE_RUN_HUGGINGFACE_TESTS=1 to run")
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
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-5-mini", timeout=30.0)

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
            if self._looks_like_connectivity_issue(e):
                pytest.skip("OpenAI not reachable")
            else:
                raise

    def test_anthropic_simple_generation(self):
        """Test Anthropic simple message generation."""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-haiku-4-5", timeout=30.0)

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
            if self._looks_like_connectivity_issue(e):
                pytest.skip("Anthropic not reachable")
            else:
                raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

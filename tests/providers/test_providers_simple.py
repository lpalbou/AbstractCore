"""
Simple test of all local providers using correct model names.
"""

import pytest
import os
import time
from abstractcore import create_llm, BasicSession


class TestProvidersSimple:
    """Test providers with real models using correct names."""

    def test_ollama_provider(self):
        """Test Ollama provider with qwen3:4b-instruct."""
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct", base_url="http://localhost:11434", timeout=10.0)

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            # This is an optional local integration test. Skip when:
            # - Ollama isn't running/reachable
            # - The expected model isn't installed locally (varies across dev machines)
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout", "operation not permitted", "model", "not found", "404"]):
                pytest.skip("Ollama not running or model not available")
            else:
                raise

    def test_lmstudio_provider(self):
        """Test LMStudio provider with qwen/qwen3-4b-2507."""
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            llm = create_llm("lmstudio", model="qwen/qwen3-4b-2507", base_url="http://localhost:1234/v1", timeout=10.0)

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30  # Should respond within 30 seconds

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout", "operation not permitted"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mlx_provider(self):
        """Test MLX provider with mlx-community/Qwen3-4B-4bit."""
        if os.getenv("ABSTRACTCORE_RUN_MLX_TESTS") != "1":
            pytest.skip("MLX provider smoke test is heavy; set ABSTRACTCORE_RUN_MLX_TESTS=1 to run")
        try:
            llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit", timeout=5.0)

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
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

    def test_huggingface_provider(self):
        """Test HuggingFace provider with Qwen/Qwen3-4B."""
        if os.getenv("ABSTRACTCORE_RUN_HUGGINGFACE_TESTS") != "1":
            pytest.skip("HuggingFace provider smoke test is heavy; set ABSTRACTCORE_RUN_HUGGINGFACE_TESTS=1 to run")
        try:
            llm = create_llm("huggingface", model="Qwen/Qwen3-4B")

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
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

    def test_openai_provider(self):
        """Test OpenAI provider with gpt-4o-mini."""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API test; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            # Keep this test bounded even if network is flaky.
            llm = create_llm("openai", model="gpt-5-mini", timeout=30)

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            # Do not assert strict latency here; this is an integration smoke test
            # and can vary due to network/provider conditions.

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["authentication", "api_key"]):
                pytest.skip("OpenAI authentication failed")
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout", "network"]):
                pytest.skip("OpenAI not reachable")
            else:
                raise

    def test_anthropic_provider(self):
        """Test Anthropic provider with claude-3-5-haiku-20241022."""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API test; set ABSTRACTCORE_RUN_LIVE_API_TESTS=1 to run")
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            # Keep this test bounded even if network is flaky.
            llm = create_llm("anthropic", model="claude-haiku-4-5", timeout=30)

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            # Do not assert strict latency here; this is an integration smoke test
            # and can vary due to network/provider conditions.

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["authentication", "api_key"]):
                pytest.skip("Anthropic authentication failed")
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout", "network"]):
                pytest.skip("Anthropic not reachable")
            else:
                raise

    def test_basic_session_with_ollama(self):
        """Test BasicSession maintains context with Ollama."""
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct", base_url="http://localhost:11434", timeout=10.0)
            session = BasicSession(provider=llm, system_prompt="You are a helpful AI assistant.")

            # Test conversation
            resp1 = session.generate("What is 2+2?")
            assert resp1 is not None
            assert resp1.content is not None

            resp2 = session.generate("What was my previous question?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained (should mention 2+2 or math)
            context_maintained = any(term in resp2.content.lower() for term in ["2+2", "math", "addition", "previous"])
            assert context_maintained, "Session should maintain context about previous question"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout", "operation not permitted", "model", "not found", "404"]):
                pytest.skip("Ollama not running or model not available")
            else:
                raise


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])

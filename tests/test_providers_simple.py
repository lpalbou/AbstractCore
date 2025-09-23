"""
Simple test of all local providers using correct model names.
"""

import pytest
import os
import time
from abstractllm import create_llm, BasicSession


class TestProvidersSimple:
    """Test providers with real models using correct names."""

    def test_ollama_provider(self):
        """Test Ollama provider with qwen3-coder:30b."""
        try:
            llm = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
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

    def test_lmstudio_provider(self):
        """Test LMStudio provider with qwen/qwen3-coder-30b."""
        try:
            llm = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
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

    def test_mlx_provider(self):
        """Test MLX provider with mlx-community/Qwen3-4B-4bit."""
        try:
            llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")

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
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            llm = create_llm("openai", model="gpt-4o-mini")

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10  # Cloud should be fast

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_provider(self):
        """Test Anthropic provider with claude-3-5-haiku-20241022."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            llm = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            start = time.time()
            response = llm.generate("Who are you in one sentence?")
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10  # Cloud should be fast

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_basic_session_with_ollama(self):
        """Test BasicSession maintains context with Ollama."""
        try:
            llm = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")
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
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
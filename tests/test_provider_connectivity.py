"""
Test provider connectivity - just connection and basic instantiation.
Fast smoke tests to verify providers can be created.
"""

import pytest
import os
from abstractllm import create_llm


class TestProviderConnectivity:
    """Test that each provider can be instantiated and connected."""

    def test_ollama_connectivity(self):
        """Test Ollama provider can be created."""
        try:
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")
            assert provider is not None
            assert provider.model == "qwen3-coder:30b"
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_connectivity(self):
        """Test LMStudio provider can be created."""
        try:
            provider = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")
            assert provider is not None
            assert provider.model == "qwen/qwen3-coder-30b"
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mlx_connectivity(self):
        """Test MLX provider can be created."""
        try:
            # Use the model from user's test specifications
            provider = create_llm("mlx", model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
            assert provider is not None
            assert provider.model == "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
        except Exception as e:
            # MLX should be available in this environment - don't skip, let it fail if there's an issue
            raise RuntimeError(f"MLX provider should be available in this environment but failed: {e}")

    def test_huggingface_connectivity(self):
        """Test HuggingFace provider can be created."""
        try:
            provider = create_llm("huggingface", model="Qwen/Qwen3-4B")
            assert provider is not None
            assert provider.model == "Qwen/Qwen3-4B"
        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["transformers", "torch", "not found", "failed to load"]):
                pytest.skip("HuggingFace not available or model not found")
            else:
                raise

    def test_openai_connectivity(self):
        """Test OpenAI provider can be created."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")
            assert provider is not None
            assert provider.model == "gpt-4o-mini"
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_connectivity(self):
        """Test Anthropic provider can be created."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")
            assert provider is not None
            assert provider.model == "claude-3-5-haiku-20241022"
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_mock_connectivity(self):
        """Test Mock provider can be created."""
        provider = create_llm("mock", model="test-model")
        assert provider is not None
        assert provider.model == "test-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
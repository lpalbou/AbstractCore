"""
Test provider connectivity - just connection and basic instantiation.
Fast smoke tests to verify providers can be created.
"""

import pytest
import os
from abstractcore import create_llm


class TestProviderConnectivity:
    """Test that each provider can be instantiated and connected."""

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

    def test_ollama_connectivity(self):
        """Test Ollama provider can be created."""
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            provider = create_llm("ollama", model="qwen3:4b-instruct", base_url="http://127.0.0.1:11434", timeout=5.0)
            assert provider is not None
            assert provider.model == "qwen3:4b-instruct"
        except Exception as e:
            if self._looks_like_connectivity_issue(e):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_connectivity(self):
        """Test LMStudio provider can be created."""
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            provider = create_llm("lmstudio", model="qwen/qwen3-4b-2507", base_url="http://127.0.0.1:1234/v1", timeout=5.0)
            assert provider is not None
            assert provider.model == "qwen/qwen3-4b-2507"
        except Exception as e:
            if self._looks_like_connectivity_issue(e):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mlx_importable(self):
        """MLX provider is heavy (loads models at init); smoke-test import only."""
        try:
            from abstractcore.providers.mlx_provider import MLXProvider  # noqa: F401
        except ImportError:
            pytest.skip("MLX provider not available")

    def test_huggingface_importable(self):
        """HuggingFace provider is heavy (may load large models); smoke-test import only."""
        try:
            from abstractcore.providers.huggingface_provider import HuggingFaceProvider  # noqa: F401
        except ImportError:
            pytest.skip("HuggingFace provider not available")

    def test_openai_connectivity(self):
        """Test OpenAI provider can be created."""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-5-mini", timeout=5.0)
            assert provider is not None
            assert provider.model == "gpt-5-mini"
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            if self._looks_like_connectivity_issue(e):
                pytest.skip("OpenAI not reachable")
            else:
                raise

    def test_anthropic_connectivity(self):
        """Test Anthropic provider can be created."""
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-haiku-4-5", timeout=5.0)
            assert provider is not None
            assert provider.model == "claude-haiku-4-5"
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            if self._looks_like_connectivity_issue(e):
                pytest.skip("Anthropic not reachable")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test unified token parameter translation across all providers - NO MOCKING
This ensures our unified parameters (max_tokens, max_input_tokens, max_output_tokens)
are correctly translated to each provider's native API parameters.
"""
import pytest
import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractllm import create_llm
from abstractllm.providers.huggingface_provider import HuggingFaceProvider
from abstractllm.providers.openai_provider import OpenAIProvider
from abstractllm.providers.anthropic_provider import AnthropicProvider
from abstractllm.providers.ollama_provider import OllamaProvider
from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.providers.lmstudio_provider import LMStudioProvider


class TestProviderTokenTranslation:
    """Test that unified token parameters are correctly translated to provider-specific APIs"""

    def test_huggingface_gguf_token_translation(self):
        """Test HuggingFace GGUF provider translates tokens correctly - REAL IMPLEMENTATION"""
        if not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"):
            pytest.skip("Test GGUF model not found in cache")

        llm = create_llm("huggingface",
                        model="unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF",
                        max_tokens=4096,
                        max_output_tokens=200,
                        debug=False)

        # Test unified parameter mapping
        kwargs = llm._prepare_generation_kwargs(max_output_tokens=150)
        provider_max_tokens = llm._get_provider_max_tokens_param(kwargs)

        # For GGUF models, max_output_tokens should map directly
        assert provider_max_tokens == 150

        # Test in actual generation call
        response = llm.generate("Hello", max_output_tokens=100)

        # Verify the response respects the token limit
        assert response.content is not None

        # If usage info is provided, check it
        if response.usage and "completion_tokens" in response.usage:
            completion_tokens = response.usage["completion_tokens"]
            # Should be within the limit (allowing some tolerance)
            assert completion_tokens <= 120  # 100 + 20% tolerance

        # Check that llama-cpp-python n_ctx parameter was set to our max_tokens
        assert llm.llm.n_ctx() == 4096

        del llm

    def test_huggingface_transformers_token_translation(self):
        """Test HuggingFace transformers provider translates tokens correctly"""
        # Skip if transformers not available or no cache models
        if not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub"):
            pytest.skip("No HuggingFace cache available")

        try:
            llm = create_llm("huggingface",
                            model="microsoft/DialoGPT-medium",
                            max_tokens=2048,
                            max_output_tokens=100,
                            debug=False)

            # Test parameter mapping
            kwargs = llm._prepare_generation_kwargs(max_output_tokens=80)
            provider_max_tokens = llm._get_provider_max_tokens_param(kwargs)

            # For transformers, this should be max_new_tokens
            assert provider_max_tokens == 80

            del llm

        except Exception as e:
            if "not found" in str(e).lower():
                pytest.skip(f"Model not available: {e}")
            raise

    def test_provider_specific_parameter_mapping(self):
        """Test each provider's parameter mapping logic without actual API calls"""

        # Test OpenAI provider mapping
        openai_provider = OpenAIProvider.__new__(OpenAIProvider)
        openai_provider.max_output_tokens = 1000
        kwargs = {"max_output_tokens": 500}
        assert openai_provider._get_provider_max_tokens_param(kwargs) == 500

        # Test Anthropic provider mapping
        anthropic_provider = AnthropicProvider.__new__(AnthropicProvider)
        anthropic_provider.max_output_tokens = 1000
        kwargs = {"max_output_tokens": 300}
        assert anthropic_provider._get_provider_max_tokens_param(kwargs) == 300

        # Test Ollama provider mapping
        ollama_provider = OllamaProvider.__new__(OllamaProvider)
        ollama_provider.max_output_tokens = 1000
        kwargs = {"max_output_tokens": 400}
        assert ollama_provider._get_provider_max_tokens_param(kwargs) == 400

        # Test MLX provider mapping
        mlx_provider = MLXProvider.__new__(MLXProvider)
        mlx_provider.max_output_tokens = 1000
        kwargs = {"max_output_tokens": 250}
        assert mlx_provider._get_provider_max_tokens_param(kwargs) == 250

        # Test LMStudio provider mapping
        lmstudio_provider = LMStudioProvider.__new__(LMStudioProvider)
        lmstudio_provider.max_output_tokens = 1000
        kwargs = {"max_output_tokens": 350}
        assert lmstudio_provider._get_provider_max_tokens_param(kwargs) == 350


class TestTokenLimitValidation:
    """Test that token limits are enforced and validated correctly"""

    def test_huggingface_gguf_token_validation(self):
        """Test GGUF model enforces token limits - REAL IMPLEMENTATION"""
        if not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"):
            pytest.skip("Test GGUF model not found in cache")

        llm = create_llm("huggingface",
                        model="unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF",
                        max_tokens=1024,  # Small total context
                        max_output_tokens=200,
                        debug=False)

        # Test validation logic
        effective_max_tokens, effective_max_output, effective_max_input = llm._calculate_effective_token_limits()

        assert effective_max_tokens == 1024
        assert effective_max_output == 200
        assert effective_max_input == 824  # 1024 - 200

        # Test that validation catches overages
        with pytest.raises(ValueError) as exc_info:
            llm.validate_token_usage(input_tokens=900, requested_output_tokens=200)

        error_msg = str(exc_info.value)
        assert "exceed" in error_msg.lower()

        # Test valid usage passes
        assert llm.validate_token_usage(input_tokens=500, requested_output_tokens=150) == True

        del llm

    def test_effective_token_calculation(self):
        """Test effective token limit calculations across different scenarios"""

        # Test with all parameters set
        if os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"):
            llm1 = create_llm("huggingface",
                             model="unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF",
                             max_tokens=2048,
                             max_input_tokens=1500,
                             max_output_tokens=500,
                             debug=False)

            max_total, max_out, max_in = llm1._calculate_effective_token_limits()
            assert max_total == 2048
            assert max_out == 500
            assert max_in == 1500

            del llm1

        # Test with calculated input tokens
        if os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"):
            llm2 = create_llm("huggingface",
                             model="unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF",
                             max_tokens=3000,
                             max_output_tokens=800,
                             debug=False)

            max_total, max_out, max_in = llm2._calculate_effective_token_limits()
            assert max_total == 3000
            assert max_out == 800
            assert max_in == 2200  # 3000 - 800 (calculated)

            del llm2


class TestActualTokenUsage:
    """Test actual token usage matches expected limits in real generation"""

    def test_gguf_generation_respects_limits(self):
        """Test GGUF generation actually respects token limits - REAL IMPLEMENTATION"""
        if not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"):
            pytest.skip("Test GGUF model not found in cache")

        llm = create_llm("huggingface",
                        model="unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF",
                        max_tokens=2048,
                        max_output_tokens=50,  # Very tight limit
                        debug=False)

        # Generate with tight limit
        response = llm.generate(
            "Write a detailed explanation of quantum computing including history, principles, and applications.",
            max_output_tokens=30  # Even tighter override
        )

        assert response.content is not None

        # Check that output was actually limited
        word_count = len(response.content.split())
        # With only 30 output tokens, should be quite short
        assert word_count < 80, f"Expected short response, got {word_count} words: {response.content}"

        # Check usage if provided
        if response.usage and "completion_tokens" in response.usage:
            completion_tokens = response.usage["completion_tokens"]
            # Should be at or near the limit
            assert completion_tokens <= 40, f"Expected ≤40 tokens, got {completion_tokens}"

        del llm

    def test_streaming_respects_limits(self):
        """Test streaming generation respects token limits - REAL IMPLEMENTATION"""
        if not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"):
            pytest.skip("Test GGUF model not found in cache")

        llm = create_llm("huggingface",
                        model="unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF",
                        max_tokens=2048,
                        max_output_tokens=40,
                        debug=False)

        # Test streaming with limits
        stream = llm.generate(
            "Explain artificial intelligence in detail.",
            stream=True,
            max_output_tokens=25  # Very tight
        )

        all_content = ""
        chunk_count = 0

        for chunk in stream:
            if chunk.content:
                all_content += chunk.content
            chunk_count += 1

        # Should have been limited
        word_count = len(all_content.split())
        assert word_count < 60, f"Expected short streaming response, got {word_count} words"

        # Should have gotten multiple chunks
        assert chunk_count > 1, "Expected multiple chunks in streaming"

        del llm

    def test_parameter_override_in_generation(self):
        """Test that generation-time parameter overrides work correctly"""
        if not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF"):
            pytest.skip("Test GGUF model not found in cache")

        llm = create_llm("huggingface",
                        model="unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF",
                        max_tokens=2048,
                        max_output_tokens=200,  # Default
                        debug=False)

        # Generate with override
        response = llm.generate(
            "Hello world",
            max_output_tokens=20  # Override to smaller value
        )

        assert response.content is not None

        # Should be quite short due to override
        word_count = len(response.content.split())
        assert word_count < 40, f"Expected short response due to override, got {word_count} words"

        del llm


if __name__ == "__main__":
    pytest.main([__file__])
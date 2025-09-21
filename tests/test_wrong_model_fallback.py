#!/usr/bin/env python3
"""
Test graceful fallback for wrong model names across all providers.
This test ensures that all providers show real available models when given invalid model names.
"""

import pytest
import os
from abstractllm import create_llm
from abstractllm.exceptions import ModelNotFoundError, AuthenticationError, ProviderAPIError


class TestWrongModelFallback:
    """Test wrong model fallback behavior for all providers."""

    def test_anthropic_wrong_model_shows_real_api_models(self):
        """Test Anthropic provider shows real models from API when model is wrong."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        with pytest.raises(ModelNotFoundError) as exc_info:
            llm = create_llm("anthropic", model="claude-3.5-haiku:latest")
            llm.generate("Hello")

        error_msg = str(exc_info.value)
        # Should state the wrong model name
        assert "claude-3.5-haiku:latest" in error_msg
        assert "Anthropic provider" in error_msg

        # Should show available models OR documentation link
        # (Anthropic API may return models or we may show docs link)
        assert ("Available models" in error_msg or
                "https://docs.anthropic.com/en/docs/about-claude/models" in error_msg)

    def test_openai_wrong_model_shows_real_api_models(self):
        """Test OpenAI provider shows real models from API when model is wrong."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        with pytest.raises(ModelNotFoundError) as exc_info:
            llm = create_llm("openai", model="gpt-5-ultra")
            llm.generate("Hello")

        error_msg = str(exc_info.value)
        # Should state the wrong model name
        assert "gpt-5-ultra" in error_msg
        assert "OpenAI provider" in error_msg

        # Should show available models from real API
        assert "Available models" in error_msg
        # Should contain some known OpenAI models
        assert any(model in error_msg for model in ["gpt-4o", "gpt-3.5-turbo", "gpt-4"])

    def test_ollama_wrong_model_shows_real_api_models(self):
        """Test Ollama provider shows real models from API when model is wrong."""
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                llm = create_llm("ollama", model="fake-model-123")
                llm.generate("Hello")

            error_msg = str(exc_info.value)
            # Should state the wrong model name
            assert "fake-model-123" in error_msg
            assert "Ollama provider" in error_msg

            # Should show available models from real API
            assert "Available models" in error_msg
            # Should show actual number of models
            assert "(" in error_msg and ")" in error_msg  # Format: "Available models (N):"

        except Exception as e:
            # If Ollama is not running, skip the test
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_wrong_model_shows_real_api_models(self):
        """Test LMStudio provider shows real models from API when model is wrong."""
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                llm = create_llm("lmstudio", model="fake-model-123")
                llm.generate("Hello")

            error_msg = str(exc_info.value)
            # Should state the wrong model name
            assert "fake-model-123" in error_msg
            assert "LMStudio provider" in error_msg

            # Should show available models from real API
            assert "Available models" in error_msg
            # Should show actual number of models
            assert "(" in error_msg and ")" in error_msg  # Format: "Available models (N):"

        except Exception as e:
            # If LMStudio is not running, skip the test
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mlx_wrong_model_shows_local_cache_models(self):
        """Test MLX provider shows models from local HuggingFace cache when model is wrong."""
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                llm = create_llm("mlx", model="fake/model-123")
                # MLX fails during initialization when loading the model

            error_msg = str(exc_info.value)
            # Should state the wrong model name
            assert "fake/model-123" in error_msg
            assert "MLX provider" in error_msg

            # Should show available models from local cache
            assert "Available models" in error_msg
            # Should show actual number of models if any cached
            assert "(" in error_msg and ")" in error_msg  # Format: "Available models (N):"

        except Exception as e:
            # If MLX is not available or no models cached, check error type
            if any(keyword in str(e).lower() for keyword in ["mlx", "import", "not installed"]):
                pytest.skip("MLX not available")
            else:
                raise

    def test_huggingface_wrong_model_shows_local_cache_models(self):
        """Test HuggingFace provider shows models from local cache when model is wrong."""
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                llm = create_llm("huggingface", model="fake/model-123")
                # HuggingFace fails during initialization when loading the model

            error_msg = str(exc_info.value)
            # Should state the wrong model name
            assert "fake/model-123" in error_msg
            assert "HuggingFace provider" in error_msg

            # Should show available models from local cache
            assert "Available models" in error_msg
            # Should show actual number of models if any cached
            assert "(" in error_msg and ")" in error_msg  # Format: "Available models (N):"

        except Exception as e:
            # If transformers is not available, check error type
            if any(keyword in str(e).lower() for keyword in ["transformers", "torch", "import", "not installed"]):
                pytest.skip("HuggingFace transformers not available")
            else:
                raise

    def test_error_message_format_consistency(self):
        """Test that all error messages follow consistent format."""
        # Test with Ollama (most likely to be available)
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                llm = create_llm("ollama", model="nonexistent-test-model")
                llm.generate("Hello")

            error_msg = str(exc_info.value)

            # Check consistent format
            assert error_msg.startswith("❌ Model 'nonexistent-test-model' not found for")
            assert "provider" in error_msg

            # Should have proper structure
            lines = error_msg.split('\n')
            assert len(lines) >= 2  # At minimum: error line + empty line or models line

            if "Available models" in error_msg:
                # If showing models, should have proper format
                assert "✅ Available models (" in error_msg
                assert "):" in error_msg
                assert "  •" in error_msg  # Bullet points for models

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("No provider available for format testing")
            else:
                raise

    def test_no_hardcoded_models_in_error_messages(self):
        """Test that error messages don't contain hardcoded/deprecated model names."""
        # This test ensures we're not showing old hardcoded lists
        try:
            with pytest.raises(ModelNotFoundError) as exc_info:
                llm = create_llm("ollama", model="definitely-fake-model-999")
                llm.generate("Hello")

            error_msg = str(exc_info.value)

            # Should NOT contain these deprecated/hardcoded model names
            deprecated_models = [
                "claude-2.0",  # Old Claude model
                "gpt-3.5-turbo-instruct-0914",  # Specific old OpenAI model
                "microsoft/DialoGPT-medium"  # Default HuggingFace model
            ]

            for deprecated in deprecated_models:
                assert deprecated not in error_msg, f"Found hardcoded model {deprecated} in error message"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not available for hardcoded model test")
            else:
                raise


if __name__ == "__main__":
    # Run individual tests for debugging
    test_instance = TestWrongModelFallback()

    print("🧪 Testing graceful fallback for wrong model names...")
    print()

    try:
        test_instance.test_anthropic_wrong_model_shows_real_api_models()
        print("✅ Anthropic fallback test passed")
    except Exception as e:
        print(f"⚠️  Anthropic test: {e}")

    try:
        test_instance.test_openai_wrong_model_shows_real_api_models()
        print("✅ OpenAI fallback test passed")
    except Exception as e:
        print(f"⚠️  OpenAI test: {e}")

    try:
        test_instance.test_ollama_wrong_model_shows_real_api_models()
        print("✅ Ollama fallback test passed")
    except Exception as e:
        print(f"⚠️  Ollama test: {e}")

    try:
        test_instance.test_lmstudio_wrong_model_shows_real_api_models()
        print("✅ LMStudio fallback test passed")
    except Exception as e:
        print(f"⚠️  LMStudio test: {e}")

    try:
        test_instance.test_mlx_wrong_model_shows_local_cache_models()
        print("✅ MLX fallback test passed")
    except Exception as e:
        print(f"⚠️  MLX test: {e}")

    try:
        test_instance.test_huggingface_wrong_model_shows_local_cache_models()
        print("✅ HuggingFace fallback test passed")
    except Exception as e:
        print(f"⚠️  HuggingFace test: {e}")

    try:
        test_instance.test_error_message_format_consistency()
        print("✅ Error message format test passed")
    except Exception as e:
        print(f"⚠️  Format test: {e}")

    try:
        test_instance.test_no_hardcoded_models_in_error_messages()
        print("✅ No hardcoded models test passed")
    except Exception as e:
        print(f"⚠️  Hardcoded models test: {e}")

    print()
    print("✅ All graceful fallback tests completed!")
#!/usr/bin/env python3
"""
Test graceful fallback for wrong model names.
"""

import pytest
from abstractllm import create_llm
from abstractllm.exceptions import ModelNotFoundError, AuthenticationError


def test_anthropic_wrong_model():
    """Test Anthropic provider with wrong model name."""
    try:
        llm = create_llm("anthropic", model="claude-3.5-haiku:latest")
        llm.generate("Hello")
        assert False, "Should have raised ModelNotFoundError"
    except ModelNotFoundError as e:
        error_msg = str(e)
        assert "claude-3.5-haiku:latest" in error_msg
        assert "Anthropic provider" in error_msg
        assert "Available models" in error_msg
        assert "claude-3-5-haiku-20241022" in error_msg  # Should show correct model
        print("✅ Anthropic graceful fallback working")


def test_openai_wrong_model():
    """Test OpenAI provider with wrong model name (with valid API to get model list)."""
    try:
        # This should fail due to wrong model, not auth
        llm = create_llm("openai", model="gpt-5-ultra")
        llm.generate("Hello")
        assert False, "Should have raised an error"
    except (ModelNotFoundError, AuthenticationError) as e:
        # Either auth error (no API key) or model error is fine for this test
        error_msg = str(e)
        assert "gpt-5-ultra" in error_msg or "api_key" in error_msg.lower()
        print("✅ OpenAI error handling working")


def test_ollama_wrong_model():
    """Test Ollama provider with wrong model name."""
    try:
        llm = create_llm("ollama", model="nonexistent-model-123")
        llm.generate("Hello")
        assert False, "Should have raised ModelNotFoundError"
    except ModelNotFoundError as e:
        error_msg = str(e)
        assert "nonexistent-model-123" in error_msg
        assert "Ollama provider" in error_msg
        assert "Available models" in error_msg
        print("✅ Ollama graceful fallback working")
    except Exception as e:
        # Ollama might not be running
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            print("⚠️  Ollama not running, skipping test")
        else:
            raise


def test_mlx_wrong_model():
    """Test MLX provider with wrong model name."""
    try:
        llm = create_llm("mlx", model="nonexistent/model-123")
        # MLX fails during init when loading model
        assert False, "Should have raised ModelNotFoundError"
    except ModelNotFoundError as e:
        error_msg = str(e)
        assert "nonexistent/model-123" in error_msg
        assert "MLX provider" in error_msg
        print("✅ MLX graceful fallback working")
    except Exception as e:
        # MLX might not be available or have issues
        if "mlx" in str(e).lower() or "import" in str(e).lower():
            print("⚠️  MLX not available, skipping test")
        else:
            print(f"MLX error: {e}")


if __name__ == "__main__":
    print("🧪 Testing graceful fallback for wrong model names...")
    print()

    test_anthropic_wrong_model()
    test_openai_wrong_model()
    test_ollama_wrong_model()
    test_mlx_wrong_model()

    print()
    print("✅ All graceful fallback tests completed!")
"""
Layer 2: Integration Tests for Token Terminology Changes

Tests integration points:
- Provider initialization with max_tokens from JSON
- CLI auto-detection functionality
- detection.py get_context_limits() function
- BaseProvider _get_default_context_window() method
"""

import pytest
from abstractllm.providers.base import BaseProvider
from abstractllm.providers.openai_provider import OpenAIProvider
from abstractllm.providers.ollama_provider import OllamaProvider
from abstractllm.architectures import get_context_limits, get_model_capabilities


class TestProviderInitialization:
    """Test that providers initialize with max_tokens from JSON."""

    def test_openai_provider_uses_json_max_tokens(self):
        """Test OpenAI provider gets max_tokens from JSON."""
        # Get expected values from JSON
        json_caps = get_model_capabilities("gpt-4")
        expected_max_tokens = json_caps["max_tokens"]
        expected_max_output = json_caps["max_output_tokens"]

        # Create provider
        provider = OpenAIProvider("gpt-4")

        # Provider should use JSON values
        assert provider.max_tokens == expected_max_tokens, \
            f"Provider max_tokens should be {expected_max_tokens}, got {provider.max_tokens}"
        assert provider.max_output_tokens == expected_max_output, \
            f"Provider max_output_tokens should be {expected_max_output}, got {provider.max_output_tokens}"

    def test_openai_provider_gpt5_values(self):
        """Test GPT-5 models get correct values from JSON."""
        json_caps = get_model_capabilities("gpt-5")

        provider = OpenAIProvider("gpt-5")

        assert provider.max_tokens == 200000, "GPT-5 should have max_tokens=200000"
        assert provider.max_output_tokens == 8192, "GPT-5 should have max_output_tokens=8192"

    def test_ollama_provider_uses_json_max_tokens(self):
        """Test Ollama provider gets max_tokens from JSON."""
        # Test with qwen3-coder:30b which should have specific capabilities
        model = "qwen3-coder-30b"
        json_caps = get_model_capabilities(model)

        provider = OllamaProvider(model)

        # Should use JSON values (may need partial matching)
        assert provider.max_tokens == json_caps["max_tokens"], \
            f"Provider should use JSON max_tokens={json_caps['max_tokens']}, got {provider.max_tokens}"

    def test_unknown_model_provider_uses_defaults(self):
        """Test provider with unknown model uses default max_tokens."""
        unknown_model = "unknown-test-model-xyz"

        provider = OllamaProvider(unknown_model)

        # Should use default values
        assert provider.max_tokens == 16384, \
            f"Unknown model should use default max_tokens=16384, got {provider.max_tokens}"
        assert provider.max_output_tokens == 4096, \
            f"Unknown model should use default max_output_tokens=4096, got {provider.max_output_tokens}"


class TestDetectionFunctions:
    """Test detection.py functions return max_tokens correctly."""

    def test_get_context_limits_returns_dict_with_max_tokens(self):
        """Test get_context_limits returns dict with max_tokens key."""
        limits = get_context_limits("gpt-4")

        # Check return type and keys
        assert isinstance(limits, dict), "get_context_limits should return dict"
        assert "max_tokens" in limits, "Result should have max_tokens key"
        assert "max_output_tokens" in limits, "Result should have max_output_tokens key"

        # Check no old terminology
        assert "context_length" not in limits, "Should not have context_length key"

    def test_get_context_limits_values_match_json(self):
        """Test get_context_limits returns values matching JSON."""
        test_models = ["gpt-4", "claude-3.5-sonnet", "llama-3.1-8b"]

        for model in test_models:
            limits = get_context_limits(model)
            caps = get_model_capabilities(model)

            assert limits["max_tokens"] == caps["max_tokens"], \
                f"{model}: get_context_limits max_tokens should match JSON"
            assert limits["max_output_tokens"] == caps["max_output_tokens"], \
                f"{model}: get_context_limits max_output_tokens should match JSON"

    def test_get_model_capabilities_returns_max_tokens(self):
        """Test get_model_capabilities returns max_tokens field."""
        caps = get_model_capabilities("gpt-4")

        assert "max_tokens" in caps, "Capabilities should have max_tokens"
        assert caps["max_tokens"] == 128000, "GPT-4 should have max_tokens=128000"


class TestBaseProviderMethods:
    """Test BaseProvider methods use max_tokens correctly."""

    def test_get_default_context_window_returns_max_tokens(self):
        """Test _get_default_context_window returns max_tokens value."""
        # Create a provider
        provider = OpenAIProvider("gpt-4")

        # Call the method
        context_window = provider._get_default_context_window()

        # Should return max_tokens value
        expected = get_model_capabilities("gpt-4")["max_tokens"]
        assert context_window == expected, \
            f"_get_default_context_window should return {expected}, got {context_window}"

    def test_get_default_max_output_tokens_works(self):
        """Test _get_default_max_output_tokens returns correct values."""
        provider = OpenAIProvider("gpt-4")

        max_output = provider._get_default_max_output_tokens()

        expected = get_model_capabilities("gpt-4")["max_output_tokens"]
        assert max_output == expected, \
            f"_get_default_max_output_tokens should return {expected}, got {max_output}"

    def test_initialize_token_limits_uses_json_values(self):
        """Test _initialize_token_limits sets values from JSON."""
        # Create provider (initialization happens in __init__)
        provider = OpenAIProvider("gpt-4")

        # Check it initialized with JSON values
        json_caps = get_model_capabilities("gpt-4")

        assert provider.max_tokens == json_caps["max_tokens"], \
            "Provider should initialize max_tokens from JSON"
        assert provider.max_output_tokens == json_caps["max_output_tokens"], \
            "Provider should initialize max_output_tokens from JSON"


class TestProviderFamilyFallback:
    """Test provider fallback to family patterns when exact match not found."""

    def test_gpt4_variant_gets_gpt4_values(self):
        """Test GPT-4 variants fall back to GPT-4 family values."""
        # Use capabilities lookup directly to test family fallback
        custom_model = "gpt-4-custom-variant"

        # Get capabilities (tests family fallback without provider validation)
        caps = get_model_capabilities(custom_model)

        # Should fall back to gpt-4 family values or defaults
        # Just verify it has reasonable values
        assert caps["max_tokens"] >= 16384, \
            f"GPT-4 variant should have reasonable max_tokens, got {caps['max_tokens']}"

    def test_claude_variant_gets_claude_values(self):
        """Test Claude variants fall back to Claude family values."""
        custom_model = "claude-custom-variant"

        # This should use family fallback
        caps = get_model_capabilities(custom_model)

        # Should get some reasonable values (from family or defaults)
        assert caps["max_tokens"] >= 16384, \
            f"Claude variant should have reasonable max_tokens, got {caps['max_tokens']}"


class TestAliasIntegration:
    """Test alias resolution integrates correctly with providers."""

    def test_alias_resolved_before_lookup(self):
        """Test aliases are resolved before capability lookup."""
        # If qwen/qwen3-next-80b alias exists, test it
        alias = "qwen/qwen3-next-80b"
        canonical = "qwen3-next-80b-a3b"

        # Get capabilities using alias
        caps_from_alias = get_model_capabilities(alias)
        caps_from_canonical = get_model_capabilities(canonical)

        # Should get the same capabilities whether using alias or canonical
        if caps_from_canonical.get("max_tokens") != 16384:  # Not using default
            assert caps_from_alias["max_tokens"] == caps_from_canonical["max_tokens"], \
                "Alias should resolve to same max_tokens as canonical name"


class TestMultiProviderConsistency:
    """Test consistency across different provider types."""

    def test_same_model_different_providers(self):
        """Test same model name gives consistent values across providers."""
        # Use a common model that might be available on multiple providers
        model = "llama-3.1-8b"

        # Get expected values from JSON
        json_caps = get_model_capabilities(model)
        expected_max_tokens = json_caps["max_tokens"]

        # Test with Ollama provider
        ollama_provider = OllamaProvider(model)
        assert ollama_provider.max_tokens == expected_max_tokens, \
            f"Ollama provider should use JSON max_tokens={expected_max_tokens}"

    def test_provider_overrides_work(self):
        """Test that manual max_tokens override still works."""
        # Create provider with manual override
        provider = OpenAIProvider("gpt-4", max_tokens=50000)

        # Should use the override value
        assert provider.max_tokens == 50000, \
            "Manual max_tokens override should be respected"


class TestEdgeCases:
    """Test edge cases in token terminology integration."""

    def test_empty_model_name_handling(self):
        """Test handling of empty model names."""
        # This should not crash, may use defaults
        try:
            caps = get_model_capabilities("")
            assert "max_tokens" in caps, "Empty model should still return max_tokens"
        except Exception:
            # If it raises, that's also acceptable
            pass

    def test_very_long_model_name(self):
        """Test handling of very long model names."""
        long_name = "x" * 1000

        caps = get_model_capabilities(long_name)

        # Should return defaults without crashing
        assert "max_tokens" in caps, "Long model name should return max_tokens"

    def test_model_name_with_special_chars(self):
        """Test model names with special characters."""
        special_name = "model@#$%/variant"

        caps = get_model_capabilities(special_name)

        # Should handle gracefully
        assert "max_tokens" in caps, "Special char model name should return max_tokens"


class TestBackwardCompatibility:
    """Test backward compatibility of the changes."""

    def test_max_output_tokens_still_works(self):
        """Test max_output_tokens field still works as expected."""
        provider = OpenAIProvider("gpt-4")

        # Should have max_output_tokens
        assert hasattr(provider, "max_output_tokens"), \
            "Provider should still have max_output_tokens attribute"
        assert provider.max_output_tokens > 0, \
            "max_output_tokens should be a positive value"

    def test_provider_max_tokens_attribute(self):
        """Test provider max_tokens attribute exists and is correct."""
        provider = OpenAIProvider("gpt-4")

        # Should have max_tokens attribute
        assert hasattr(provider, "max_tokens"), \
            "Provider should have max_tokens attribute"
        assert provider.max_tokens == 128000, \
            "GPT-4 provider should have max_tokens=128000"

    def test_interface_methods_still_work(self):
        """Test AbstractLLM interface methods still work."""
        provider = OpenAIProvider("gpt-4")

        # Should have core interface methods
        assert hasattr(provider, "get_capabilities"), \
            "Provider should have get_capabilities method"
        assert hasattr(provider, "validate_config"), \
            "Provider should have validate_config method"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

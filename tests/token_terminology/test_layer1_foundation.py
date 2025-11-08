"""
Layer 1: Foundation Tests for Token Terminology Changes

Tests the fundamental changes to token terminology:
- max_tokens field in model_capabilities.json
- Alias resolution for model names
- Default fallback behavior
- Validation of no context_length references
"""

import pytest
import json
from pathlib import Path
from abstractcore.architectures import (
    get_model_capabilities,
    get_context_limits
)
from abstractcore.architectures.detection import resolve_model_alias, _load_json_assets


class TestModelCapabilitiesJSON:
    """Test that model_capabilities.json uses max_tokens consistently."""

    def test_json_uses_max_tokens_not_context_length(self):
        """Verify all models in JSON use max_tokens instead of context_length."""
        # Load the JSON file directly
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        # Check every model has max_tokens and NOT context_length
        for model_name, model_data in models.items():
            assert "max_tokens" in model_data, f"Model {model_name} missing max_tokens field"
            assert "context_length" not in model_data, f"Model {model_name} still has context_length field"

            # Verify max_tokens is a non-negative integer
            max_tokens = model_data["max_tokens"]
            assert isinstance(max_tokens, int), f"Model {model_name} max_tokens is not an int: {type(max_tokens)}"
            
            # Embedding models can have max_tokens = 0 since they don't generate text
            model_type = model_data.get("model_type", "generative")
            if model_type == "embedding":
                assert max_tokens >= 0, f"Embedding model {model_name} max_tokens must be non-negative: {max_tokens}"
            else:
                assert max_tokens > 0, f"Generative model {model_name} max_tokens must be positive: {max_tokens}"

    def test_default_capabilities_use_max_tokens(self):
        """Verify default_capabilities in JSON uses max_tokens."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        defaults = data.get("default_capabilities", {})

        assert "max_tokens" in defaults, "default_capabilities missing max_tokens"
        assert "context_length" not in defaults, "default_capabilities still has context_length"
        assert defaults["max_tokens"] == 16384, "Default max_tokens should be 16384"

    def test_sample_models_max_tokens_values(self):
        """Test specific models have correct max_tokens values."""
        test_cases = [
            ("gpt-4", 128000),
            ("gpt-4o-mini", 128000),
            ("claude-3.5-sonnet", 200000),
            ("llama-3.1-8b", 128000),
            ("qwen2.5-7b", 131072),
            ("phi-4", 16000),
        ]

        for model_name, expected_max_tokens in test_cases:
            caps = get_model_capabilities(model_name)
            assert caps["max_tokens"] == expected_max_tokens, \
                f"{model_name} should have max_tokens={expected_max_tokens}, got {caps['max_tokens']}"


class TestAliasResolution:
    """Test model alias resolution works correctly."""

    def test_canonical_name_resolution(self):
        """Test models resolve to their canonical names."""
        # Force reload of JSON assets
        _load_json_assets()

        # Load the JSON to get models dict
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)
        models = data.get("models", {})

        # Test: canonical name resolves to itself
        canonical = resolve_model_alias("gpt-4", models)
        assert canonical == "gpt-4", f"Canonical name should resolve to itself: {canonical}"

    def test_alias_resolves_to_canonical(self):
        """Test that aliases resolve to their canonical names."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)
        models = data.get("models", {})

        # Find a model with aliases
        model_with_alias = None
        alias_to_test = None
        canonical_name = None

        for model_name, model_data in models.items():
            aliases = model_data.get("aliases", [])
            if aliases:
                model_with_alias = model_name
                alias_to_test = aliases[0]
                canonical_name = model_data.get("canonical_name", model_name)
                break

        if model_with_alias:
            # Test alias resolution
            resolved = resolve_model_alias(alias_to_test, models)
            assert resolved == model_with_alias, \
                f"Alias {alias_to_test} should resolve to {model_with_alias}, got {resolved}"

    def test_qwen_alias_resolution(self):
        """Test specific qwen3-next-80b alias resolution."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)
        models = data.get("models", {})

        # qwen/qwen3-next-80b should resolve to qwen3-next-80b-a3b
        alias = "qwen/qwen3-next-80b"
        resolved = resolve_model_alias(alias, models)

        # Check if the alias exists in the JSON
        if alias in str(models):
            assert resolved == "qwen3-next-80b-a3b", \
                f"Alias {alias} should resolve to qwen3-next-80b-a3b, got {resolved}"

    def test_unknown_model_returns_original_name(self):
        """Test that unknown models return their original name."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)
        models = data.get("models", {})

        unknown = "totally-unknown-model-xyz"
        resolved = resolve_model_alias(unknown, models)
        assert resolved == unknown, \
            f"Unknown model should return original name, got {resolved}"


class TestContextLimitsFunction:
    """Test get_context_limits() function returns max_tokens."""

    def test_returns_max_tokens_key(self):
        """Verify get_context_limits returns max_tokens (not context_length)."""
        limits = get_context_limits("gpt-4")

        # Should return max_tokens, not context_length
        assert "max_tokens" in limits, "get_context_limits should return max_tokens"
        assert "context_length" not in limits, "get_context_limits should not return context_length"
        assert "max_output_tokens" in limits, "get_context_limits should return max_output_tokens"

    def test_max_tokens_values_correct(self):
        """Test specific models return correct max_tokens values."""
        test_cases = [
            ("gpt-4", 128000),
            ("claude-3.5-sonnet", 200000),
            ("llama-3.1-8b", 128000),
        ]

        for model_name, expected_max_tokens in test_cases:
            limits = get_context_limits(model_name)
            assert limits["max_tokens"] == expected_max_tokens, \
                f"{model_name} should have max_tokens={expected_max_tokens}, got {limits['max_tokens']}"

    def test_unknown_model_default_fallback(self):
        """Test unknown models get default max_tokens value."""
        unknown_model = "unknown-test-model-xyz"
        limits = get_context_limits(unknown_model)

        # Should get default value from model_capabilities.json
        assert limits["max_tokens"] == 16384, \
            f"Unknown model should get default max_tokens=16384, got {limits['max_tokens']}"
        assert limits["max_output_tokens"] == 4096, \
            f"Unknown model should get default max_output_tokens=4096, got {limits['max_output_tokens']}"


class TestDefaultFallbackBehavior:
    """Test default fallback behavior for unknown models."""

    def test_unknown_model_gets_defaults(self):
        """Test that completely unknown models get default capabilities."""
        unknown_model = "completely-unknown-model-12345"
        caps = get_model_capabilities(unknown_model)

        # Should get default capabilities
        assert caps["max_tokens"] == 16384, "Should use default max_tokens"
        assert caps["max_output_tokens"] == 4096, "Should use default max_output_tokens"
        assert caps["architecture"] == "generic", "Should use generic architecture"

    def test_partial_match_fallback(self):
        """Test models with partial name matches get appropriate capabilities."""
        # Model name contains 'gpt' but doesn't match exactly
        partial_match = "custom-gpt-variant"
        caps = get_model_capabilities(partial_match)

        # Should get some capabilities (either from partial match or defaults)
        assert "max_tokens" in caps
        assert "max_output_tokens" in caps
        assert caps["max_tokens"] > 0


class TestNoContextLengthReferences:
    """Verify no context_length references remain in the codebase."""

    def test_json_has_no_context_length(self):
        """Verify model_capabilities.json has no context_length references."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            json_content = f.read()

        # Should not contain "context_length" anywhere
        assert "context_length" not in json_content, \
            "model_capabilities.json should not contain 'context_length'"

    def test_detection_py_uses_max_tokens(self):
        """Verify detection.py get_context_limits returns max_tokens."""
        # Import and check the function
        from abstractcore.architectures.detection import get_context_limits

        # Check function source doesn't reference context_length
        import inspect
        source = inspect.getsource(get_context_limits)

        assert "max_tokens" in source, "get_context_limits should use max_tokens"
        assert "context_length" not in source, "get_context_limits should not use context_length"

    def test_base_provider_uses_max_tokens(self):
        """Verify BaseProvider uses max_tokens terminology."""
        from abstractcore.providers.base import BaseProvider
        import inspect

        # Check _get_default_context_window method
        source = inspect.getsource(BaseProvider._get_default_context_window)

        assert "max_tokens" in source, "_get_default_context_window should use max_tokens"
        # Note: "context" in the method name is still valid (context window is a concept)


class TestAllModelsHaveMaxTokens:
    """Verify all 85 models in JSON have valid max_tokens values."""

    def test_all_models_count(self):
        """Verify we have approximately 85 models in the JSON."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})
        model_count = len(models)

        # Should have around 85 models (allow some variance for additions/removals)
        assert 80 <= model_count <= 90, \
            f"Expected around 85 models, found {model_count}"

    def test_all_models_have_valid_max_tokens(self):
        """Verify every model has a valid max_tokens value."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        for model_name, model_data in models.items():
            # Must have max_tokens
            assert "max_tokens" in model_data, \
                f"Model {model_name} missing max_tokens"

            # max_tokens must be a positive integer
            max_tokens = model_data["max_tokens"]
            assert isinstance(max_tokens, int), \
                f"Model {model_name} max_tokens must be int, got {type(max_tokens)}"
            assert max_tokens > 0, \
                f"Model {model_name} max_tokens must be positive, got {max_tokens}"

            # max_tokens should be reasonable (between 1K and 10M tokens)
            assert 1000 <= max_tokens <= 10000000, \
                f"Model {model_name} max_tokens seems unreasonable: {max_tokens}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

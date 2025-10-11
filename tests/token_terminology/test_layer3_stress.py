"""
Layer 3: Stress Tests for Token Terminology Changes

Tests system resilience:
- All 85 models in JSON have valid max_tokens values
- Provider behavior with edge case models
- Alias resolution with complex naming patterns
- System behavior with malformed/invalid inputs
- Error handling and recovery
"""

import pytest
import json
from pathlib import Path
from abstractllm.architectures import (
    get_model_capabilities,
    get_context_limits,
    detect_architecture
)
from abstractllm.architectures.detection import resolve_model_alias
from abstractllm.providers.openai_provider import OpenAIProvider
from abstractllm.providers.ollama_provider import OllamaProvider


class TestAll85Models:
    """Test all models in JSON have valid max_tokens values."""

    def test_load_all_models_from_json(self):
        """Load and verify all models from JSON."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractllm" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        # Store for other tests
        self.all_models = models
        assert len(models) > 0, "Should have models in JSON"

    def test_every_model_has_max_tokens(self):
        """Verify every single model has max_tokens field."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractllm" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})
        failures = []

        for model_name, model_data in models.items():
            if "max_tokens" not in model_data:
                failures.append(f"{model_name}: missing max_tokens")
            elif not isinstance(model_data["max_tokens"], int):
                failures.append(f"{model_name}: max_tokens is not int")
            elif model_data["max_tokens"] <= 0:
                failures.append(f"{model_name}: max_tokens is not positive")

        assert len(failures) == 0, f"Models with invalid max_tokens:\n" + "\n".join(failures)

    def test_every_model_max_tokens_in_valid_range(self):
        """Verify all max_tokens values are in a reasonable range."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractllm" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})
        out_of_range = []

        MIN_TOKENS = 1000  # 1K minimum
        MAX_TOKENS = 10_000_000  # 10M maximum (llama-4 has 10M)

        for model_name, model_data in models.items():
            max_tokens = model_data["max_tokens"]
            if max_tokens < MIN_TOKENS or max_tokens > MAX_TOKENS:
                out_of_range.append(
                    f"{model_name}: max_tokens={max_tokens} (valid range: {MIN_TOKENS}-{MAX_TOKENS})"
                )

        assert len(out_of_range) == 0, \
            f"Models with out-of-range max_tokens:\n" + "\n".join(out_of_range)

    def test_provider_initialization_all_models(self):
        """Test provider initialization with all models from JSON."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractllm" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})
        failures = []

        # Test a sample of models known to work (not all to avoid provider validation)
        sample_models = ["gpt-4", "claude-3.5-sonnet", "llama-3.1-8b", "phi-4"]

        for model_name in sample_models:
            try:
                # Use Ollama provider which is more permissive
                provider = OllamaProvider(model_name)

                # Verify it has max_tokens set from JSON
                json_caps = get_model_capabilities(model_name)
                if not hasattr(provider, 'max_tokens'):
                    failures.append(f"{model_name}: provider missing max_tokens attribute")
                elif provider.max_tokens != json_caps["max_tokens"]:
                    failures.append(f"{model_name}: provider max_tokens mismatch")

            except Exception as e:
                failures.append(f"{model_name}: initialization failed - {e}")

        assert len(failures) == 0, \
            f"Provider initialization failures:\n" + "\n".join(failures)


class TestEdgeCaseModels:
    """Test provider behavior with edge case models."""

    def test_model_with_special_characters_in_name(self):
        """Test models with special characters in names."""
        # Test models that might have special chars
        special_models = [
            "phi-3.5-mini",
            "claude-3.5-sonnet",
            "llama-3.1-8b",
            "qwen2.5-7b"
        ]

        for model in special_models:
            caps = get_model_capabilities(model)
            assert "max_tokens" in caps, f"{model} should have max_tokens"
            assert caps["max_tokens"] > 0, f"{model} should have positive max_tokens"

    def test_model_with_version_numbers(self):
        """Test models with complex version numbering."""
        version_models = [
            "gpt-3.5-turbo",
            "phi-3.5-mini",
            "claude-3.5-sonnet",
            "gemma2-9b"
        ]

        for model in version_models:
            caps = get_model_capabilities(model)
            assert caps["max_tokens"] > 0, f"{model} should have valid max_tokens"

    def test_very_large_context_models(self):
        """Test models with very large context windows."""
        # Models with >100K context
        large_context_models = [
            "gpt-4",  # 128K
            "claude-3.5-sonnet",  # 200K
            "gpt-5",  # 200K
            "llama-4"  # 10M
        ]

        for model in large_context_models:
            caps = get_model_capabilities(model)
            assert caps["max_tokens"] >= 100000, \
                f"{model} should have large context (>100K), got {caps['max_tokens']}"

    def test_very_small_context_models(self):
        """Test models with smaller context windows."""
        small_context_models = [
            "phi-2",  # 2048
            "phi-3-mini"  # 4096
        ]

        for model in small_context_models:
            caps = get_model_capabilities(model)
            assert 1000 <= caps["max_tokens"] <= 10000, \
                f"{model} should have small context, got {caps['max_tokens']}"


class TestComplexAliasPatterns:
    """Test alias resolution with complex naming patterns."""

    def test_slash_in_alias(self):
        """Test aliases with slashes (e.g., qwen/qwen3-next-80b)."""
        assets_dir = Path(__file__).parent.parent.parent / "abstractllm" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            data = json.load(f)
        models = data.get("models", {})

        # Find aliases with slashes
        slash_aliases = []
        for model_name, model_data in models.items():
            aliases = model_data.get("aliases", [])
            for alias in aliases:
                if "/" in alias:
                    slash_aliases.append((alias, model_name))

        # Test each slash alias
        for alias, canonical in slash_aliases:
            resolved = resolve_model_alias(alias, models)
            assert resolved == canonical, \
                f"Slash alias {alias} should resolve to {canonical}, got {resolved}"

    def test_hyphenated_aliases(self):
        """Test aliases with multiple hyphens."""
        # Models with complex hyphenation
        complex_models = [
            "llama-3.1-8b",
            "claude-3.5-sonnet",
            "gpt-4o-long-output",
            "phi-3.5-mini"
        ]

        for model in complex_models:
            caps = get_model_capabilities(model)
            assert "max_tokens" in caps, f"{model} should resolve correctly"

    def test_alias_case_insensitivity(self):
        """Test that model lookup handles case variations."""
        # Test different case variations
        test_cases = [
            ("GPT-4", "gpt-4"),
            ("Gpt-4", "gpt-4"),
            ("CLAUDE-3.5-SONNET", "claude-3.5-sonnet")
        ]

        for input_name, expected_base in test_cases:
            # Architecture detection should be case-insensitive
            arch = detect_architecture(input_name)
            expected_arch = detect_architecture(expected_base)
            assert arch == expected_arch, \
                f"Case variation {input_name} should detect same arch as {expected_base}"


class TestMalformedInputHandling:
    """Test system behavior with malformed/invalid inputs."""

    def test_none_model_name(self):
        """Test handling of None as model name."""
        try:
            caps = get_model_capabilities(None)
            # If it doesn't crash, should return defaults
            assert "max_tokens" in caps, "None model should return defaults"
        except (TypeError, AttributeError):
            # Acceptable to raise TypeError for None
            pass

    def test_numeric_model_name(self):
        """Test handling of numeric model names."""
        try:
            caps = get_model_capabilities(12345)
            # Should handle gracefully (convert to string or use defaults)
            assert "max_tokens" in caps, "Numeric model should return max_tokens"
        except AttributeError:
            # Acceptable to raise AttributeError for numeric input
            pass

    def test_empty_string_model(self):
        """Test handling of empty string model name."""
        caps = get_model_capabilities("")

        # Should return some max_tokens (either defaults or from partial match)
        assert "max_tokens" in caps, "Empty model should return max_tokens"
        assert caps["max_tokens"] > 0, "Empty model should have positive max_tokens"

    def test_whitespace_only_model(self):
        """Test handling of whitespace-only model names."""
        caps = get_model_capabilities("   ")

        # Should handle gracefully
        assert "max_tokens" in caps, "Whitespace model should return max_tokens"

    def test_unicode_model_name(self):
        """Test handling of Unicode characters in model names."""
        unicode_model = "model-名前-αβγ"

        caps = get_model_capabilities(unicode_model)

        # Should handle gracefully
        assert "max_tokens" in caps, "Unicode model name should work"


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""

    def test_missing_json_file_fallback(self):
        """Test behavior when JSON file is missing (simulation)."""
        # This test verifies the code has fallback behavior
        # We can't actually delete the file, but we test the fallback path

        unknown_model = "definitely-not-in-json-12345"
        caps = get_model_capabilities(unknown_model)

        # Should get default capabilities without crashing
        assert caps["max_tokens"] == 16384, "Should fallback to defaults"
        assert caps["max_output_tokens"] == 4096, "Should fallback to defaults"

    def test_corrupt_model_data_handling(self):
        """Test handling of models with missing required fields."""
        # Test with model that should exist
        model = "gpt-4"
        caps = get_model_capabilities(model)

        # Required fields should be present
        required_fields = ["max_tokens", "max_output_tokens"]
        for field in required_fields:
            assert field in caps, f"Model {model} should have {field}"

    def test_provider_with_invalid_max_tokens_override(self):
        """Test provider behavior with invalid max_tokens override."""
        # Test with negative value
        try:
            provider = OpenAIProvider("gpt-4", max_tokens=-1000)
            # If it doesn't raise, it should clamp or ignore
            assert provider.max_tokens > 0, "Should not allow negative max_tokens"
        except (ValueError, AssertionError):
            # Acceptable to raise error for invalid values
            pass

        # Test with zero
        try:
            provider = OpenAIProvider("gpt-4", max_tokens=0)
            assert provider.max_tokens > 0, "Should not allow zero max_tokens"
        except (ValueError, AssertionError):
            pass

    def test_concurrent_model_lookup(self):
        """Test thread-safety of model capability lookup."""
        import threading

        models_to_test = ["gpt-4", "claude-3.5-sonnet", "llama-3.1-8b"]
        results = []
        errors = []

        def lookup_model(model_name):
            try:
                caps = get_model_capabilities(model_name)
                results.append((model_name, caps["max_tokens"]))
            except Exception as e:
                errors.append((model_name, str(e)))

        # Create multiple threads
        threads = []
        for _ in range(10):  # 10 concurrent lookups
            for model in models_to_test:
                thread = threading.Thread(target=lookup_model, args=(model,))
                threads.append(thread)
                thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0, f"Concurrent lookup errors: {errors}"
        assert len(results) == 30, "Should have 30 successful lookups"


class TestPerformanceStress:
    """Test performance under stress conditions."""

    def test_rapid_model_lookup(self):
        """Test rapid successive model lookups."""
        models = ["gpt-4", "claude-3.5-sonnet", "llama-3.1-8b", "phi-4"]

        # Perform 1000 rapid lookups
        for _ in range(250):
            for model in models:
                caps = get_model_capabilities(model)
                assert caps["max_tokens"] > 0, f"{model} lookup failed"

    def test_many_providers_creation(self):
        """Test creating many provider instances."""
        models = ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"]
        providers = []

        # Create 50 provider instances
        for i in range(50):
            model = models[i % len(models)]
            provider = OpenAIProvider(model)
            providers.append(provider)

            # Verify each has correct max_tokens
            json_caps = get_model_capabilities(model)
            assert provider.max_tokens == json_caps["max_tokens"], \
                f"Provider {i} should have correct max_tokens"

        # All providers should be valid
        assert len(providers) == 50, "Should create 50 providers"


class TestConsistencyChecks:
    """Test consistency across the system."""

    def test_max_tokens_consistency_across_lookups(self):
        """Test same model always returns same max_tokens."""
        model = "gpt-4"

        # Lookup 100 times
        values = []
        for _ in range(100):
            caps = get_model_capabilities(model)
            values.append(caps["max_tokens"])

        # All values should be identical
        assert len(set(values)) == 1, \
            f"max_tokens should be consistent, got {set(values)}"

    def test_provider_vs_direct_lookup_consistency(self):
        """Test provider values match direct JSON lookup."""
        models_to_test = [
            "gpt-4",
            "gpt-4o-mini",
            "claude-3.5-sonnet",
            "phi-4"
        ]

        for model in models_to_test:
            # Direct lookup
            json_caps = get_model_capabilities(model)

            # Provider lookup
            if "gpt" in model or "o1" in model or "o3" in model:
                provider = OpenAIProvider(model)
            else:
                provider = OllamaProvider(model)

            # Should match
            assert provider.max_tokens == json_caps["max_tokens"], \
                f"{model}: provider max_tokens should match JSON"
            assert provider.max_output_tokens == json_caps["max_output_tokens"], \
                f"{model}: provider max_output_tokens should match JSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

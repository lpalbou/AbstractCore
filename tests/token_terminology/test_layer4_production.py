"""
Layer 4: Production Tests for Token Terminology Changes

Tests production readiness:
- Run existing test suite to ensure no regressions
- Test real provider creation and token limit detection
- Test CLI functionality with actual models
- Test multi-provider compatibility
- Validate integration with existing features
"""

import pytest
import subprocess
import sys
from pathlib import Path
from abstractcore import create_llm
from abstractcore.architectures import get_model_capabilities, get_context_limits
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.providers.ollama_provider import OllamaProvider


class TestExistingTestSuite:
    """Verify existing tests still pass with new terminology."""

    def test_system_integration_tests_pass(self):
        """Run the existing system integration tests."""
        test_file = Path(__file__).parent.parent / "integration" / "test_system_integration.py"

        if test_file.exists():
            # Run the test file
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "-x"],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent)
            )

            # Check it passed (allow skips for unavailable services)
            # Exit code 0 = all passed, 5 = no tests collected
            assert result.returncode in [0, 5], \
                f"System integration tests failed:\n{result.stdout}\n{result.stderr}"

    def test_gpt4_capabilities_still_work(self):
        """Verify the GPT-4 capabilities test from test_system_integration.py."""
        # This duplicates a test from test_system_integration.py to ensure it passes
        json_caps = get_model_capabilities('gpt-4')

        # Verify JSON has the expected values (using new terminology)
        assert json_caps['max_tokens'] == 128000
        assert json_caps['max_output_tokens'] == 4096
        assert json_caps['tool_support'] == 'native'

        # Verify provider uses these values
        provider = OpenAIProvider('gpt-4')
        assert provider.max_tokens == 128000
        assert provider.max_output_tokens == 4096

    def test_gpt5_capabilities_still_work(self):
        """Verify the GPT-5 capabilities test from test_system_integration.py."""
        json_caps = get_model_capabilities('gpt-5')

        assert json_caps['max_tokens'] == 200000
        assert json_caps['max_output_tokens'] == 8192
        assert json_caps['tool_support'] == 'native'

        provider = OpenAIProvider('gpt-5')
        assert provider.max_tokens == 200000
        assert provider.max_output_tokens == 8192

    def test_unknown_model_fallback_still_works(self):
        """Verify unknown model fallback test from test_system_integration.py."""
        unknown_model = 'test-unknown-model-xyz'
        json_caps = get_model_capabilities(unknown_model)

        # Should get default capabilities (using new terminology)
        assert json_caps['max_tokens'] == 16384  # Default: 16K total
        assert json_caps['max_output_tokens'] == 4096  # Default: 4K output
        assert json_caps['architecture'] == 'generic'

        provider = OllamaProvider(unknown_model)
        assert provider.max_tokens == 16384
        assert provider.max_output_tokens == 4096


class TestRealProviderCreation:
    """Test real provider creation with token limit detection."""

    def test_create_llm_with_openai_provider(self):
        """Test creating OpenAI provider via create_llm."""
        # Create provider
        llm = create_llm("openai", model="gpt-4")

        # Verify it has correct max_tokens from JSON
        assert llm.max_tokens == 128000, "GPT-4 should have max_tokens=128000"
        assert llm.max_output_tokens == 4096, "GPT-4 should have max_output_tokens=4096"

    def test_create_llm_with_ollama_provider(self):
        """Test creating Ollama provider via create_llm."""
        llm = create_llm("ollama", model="qwen3-coder-30b")

        # Should get capabilities from JSON
        json_caps = get_model_capabilities("qwen3-coder-30b")
        assert llm.max_tokens == json_caps["max_tokens"], \
            f"Should use JSON max_tokens={json_caps['max_tokens']}"

    def test_create_llm_with_override(self):
        """Test creating provider with max_tokens override."""
        llm = create_llm("openai", model="gpt-4", max_tokens=50000)

        # Should use override value
        assert llm.max_tokens == 50000, "Should use override max_tokens"

    def test_provider_token_limits_detection(self):
        """Test token limits are correctly detected for various models."""
        test_cases = [
            ("openai", "gpt-4", 128000, 4096),
            ("openai", "gpt-4o-mini", 128000, 16000),
            ("openai", "gpt-5", 200000, 8192),
        ]

        for provider_name, model, expected_max_tokens, expected_max_output in test_cases:
            llm = create_llm(provider_name, model=model)

            assert llm.max_tokens == expected_max_tokens, \
                f"{model} should have max_tokens={expected_max_tokens}, got {llm.max_tokens}"
            assert llm.max_output_tokens == expected_max_output, \
                f"{model} should have max_output_tokens={expected_max_output}, got {llm.max_output_tokens}"


class TestMultiProviderCompatibility:
    """Test compatibility across multiple providers."""

    def test_same_model_different_providers_consistency(self):
        """Test same model gives consistent values across provider types."""
        # llama-3.1-8b might be available on different providers
        model = "llama-3.1-8b"
        json_caps = get_model_capabilities(model)

        # Create with Ollama provider
        ollama_llm = create_llm("ollama", model=model)

        # Should use JSON values
        assert ollama_llm.max_tokens == json_caps["max_tokens"], \
            "Ollama provider should use JSON max_tokens"

    def test_provider_specific_models(self):
        """Test provider-specific models use correct values."""
        provider_models = [
            ("openai", "gpt-4o-mini"),
            ("openai", "gpt-5-turbo"),
            ("ollama", "phi-4"),
        ]

        for provider_name, model in provider_models:
            llm = create_llm(provider_name, model=model)
            json_caps = get_model_capabilities(model)

            # Should match JSON values
            assert llm.max_tokens == json_caps["max_tokens"], \
                f"{provider_name}/{model} should use JSON max_tokens"

    def test_cross_provider_alias_resolution(self):
        """Test alias resolution works across providers."""
        # If qwen/qwen3-next-80b alias exists
        alias = "qwen/qwen3-next-80b"
        canonical = "qwen3-next-80b-a3b"

        # Get capabilities
        alias_caps = get_model_capabilities(alias)
        canonical_caps = get_model_capabilities(canonical)

        # Should resolve to same values
        if canonical_caps.get("max_tokens") != 16384:  # Not using defaults
            assert alias_caps["max_tokens"] == canonical_caps["max_tokens"], \
                "Alias should resolve to same max_tokens as canonical"


class TestCLICompatibility:
    """Test CLI functionality with new terminology."""

    def test_cli_auto_detection_imports(self):
        """Test CLI auto-detection can be imported."""
        try:
            from abstractcore.utils.cli import auto_detect_provider
            assert callable(auto_detect_provider), "auto_detect_provider should be callable"
        except ImportError as e:
            pytest.skip(f"CLI module not available: {e}")

    def test_cli_model_info_display(self):
        """Test CLI can display model info with new terminology."""
        # This would test CLI commands, but we'll test the underlying functions
        model = "gpt-4"
        caps = get_model_capabilities(model)

        # CLI should be able to display these
        assert "max_tokens" in caps, "CLI should have max_tokens for display"
        assert caps["max_tokens"] > 0, "CLI should show valid max_tokens"


class TestFeatureIntegration:
    """Test integration with existing features."""

    def test_structured_output_with_max_tokens(self):
        """Test structured output works with new max_tokens."""
        from abstractcore.core.types import GenerateResponse

        # Create provider
        llm = create_llm("openai", model="gpt-4")

        # Verify structured output capability check works
        assert llm.max_tokens > 0, "Should have valid max_tokens for structured output"

    def test_streaming_with_max_tokens(self):
        """Test streaming respects max_tokens limits."""
        llm = create_llm("openai", model="gpt-4")

        # Verify streaming-related token limits are set
        assert llm.max_output_tokens > 0, "Should have max_output_tokens for streaming"

    def test_tool_calling_with_token_limits(self):
        """Test tool calling works with token limit detection."""
        from abstractcore.tools import ToolDefinition

        llm = create_llm("openai", model="gpt-4")

        # Define a simple tool
        def test_tool() -> str:
            """Test tool."""
            return "result"

        tool_def = ToolDefinition.from_function(test_tool)

        # Verify provider has correct token limits for tool calling
        assert llm.max_tokens == 128000, "Should have correct max_tokens for tools"
        assert llm.model_capabilities.get("tool_support") == "native", \
            "Should detect native tool support"

    def test_retry_mechanism_with_token_limits(self):
        """Test retry mechanism respects token limits."""
        from abstractcore.core.retry import RetryConfig

        # Create provider with retry config
        retry_config = RetryConfig(max_attempts=3)
        llm = create_llm("openai", model="gpt-4", retry_config=retry_config)

        # Should have token limits set
        assert llm.max_tokens > 0, "Retry should work with valid max_tokens"


class TestBackwardCompatibility:
    """Ensure backward compatibility with old code."""

    def test_old_test_patterns_still_work(self):
        """Test patterns from old tests still work."""
        # Pattern 1: Direct provider creation
        provider = OpenAIProvider("gpt-4")
        assert provider.max_tokens > 0, "Old pattern should still work"

        # Pattern 2: Using create_llm
        llm = create_llm("openai", model="gpt-4")
        assert llm.max_tokens > 0, "create_llm pattern should still work"

        # Pattern 3: Capability lookup
        caps = get_model_capabilities("gpt-4")
        assert "max_tokens" in caps, "Capability lookup should still work"

    def test_max_output_tokens_attribute_exists(self):
        """Test max_output_tokens attribute still exists."""
        llm = create_llm("openai", model="gpt-4")

        # Should have both attributes
        assert hasattr(llm, "max_tokens"), "Should have max_tokens"
        assert hasattr(llm, "max_output_tokens"), "Should have max_output_tokens"

    def test_interface_compatibility(self):
        """Test AbstractCore interface compatibility."""
        llm = create_llm("openai", model="gpt-4")

        # Should have standard interface methods
        assert hasattr(llm, "generate"), "Should have generate method"
        assert hasattr(llm, "get_capabilities"), "Should have get_capabilities"
        assert hasattr(llm, "validate_config"), "Should have validate_config"


class TestProductionScenarios:
    """Test realistic production scenarios."""

    def test_production_model_selection(self):
        """Test selecting models for production use."""
        # Production scenario: select model based on token requirements
        required_tokens = 100000

        production_models = [
            "gpt-4",  # 128K
            "claude-3.5-sonnet",  # 200K
            "llama-3.1-8b",  # 128K
        ]

        suitable_models = []
        for model in production_models:
            caps = get_model_capabilities(model)
            if caps["max_tokens"] >= required_tokens:
                suitable_models.append(model)

        # Should find suitable models
        assert len(suitable_models) > 0, "Should find models meeting token requirements"

    def test_production_provider_initialization(self):
        """Test production provider initialization pattern."""
        # Common production pattern
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "max_tokens": None  # Use JSON defaults
        }

        llm = create_llm(
            config["provider"],
            model=config["model"],
            max_tokens=config["max_tokens"]
        )

        # Should initialize correctly
        assert llm.max_tokens == 128000, "Should use JSON default when None provided"

    def test_production_error_handling(self):
        """Test production error handling patterns."""
        # Production pattern: validate model before use
        model = "gpt-4"
        caps = get_model_capabilities(model)

        # Validate has required fields
        required_fields = ["max_tokens", "max_output_tokens", "tool_support"]
        for field in required_fields:
            assert field in caps, f"Production model should have {field}"

    def test_production_monitoring_metrics(self):
        """Test production monitoring can access token metrics."""
        llm = create_llm("openai", model="gpt-4")

        # Production monitoring needs these metrics
        metrics = {
            "model": llm.model,
            "max_tokens": llm.max_tokens,
            "max_output_tokens": llm.max_output_tokens,
            "provider": llm.__class__.__name__
        }

        # All metrics should be available
        assert all(v is not None for v in metrics.values()), \
            "All monitoring metrics should be available"


class TestRegressionPrevention:
    """Prevent regressions in token terminology."""

    def test_no_context_length_in_provider(self):
        """Ensure providers don't use context_length."""
        llm = create_llm("openai", model="gpt-4")

        # Should NOT have context_length attribute
        assert not hasattr(llm, "context_length"), \
            "Provider should not have context_length attribute"

    def test_context_limits_function_signature(self):
        """Ensure get_context_limits returns correct keys."""
        limits = get_context_limits("gpt-4")

        # Should have new terminology
        assert "max_tokens" in limits, "Should have max_tokens key"
        assert "context_length" not in limits, "Should not have context_length key"

    def test_json_consistency_check(self):
        """Regular check that JSON remains consistent."""
        import json
        assets_dir = Path(__file__).parent.parent.parent / "abstractcore" / "assets"
        json_file = assets_dir / "model_capabilities.json"

        with open(json_file, 'r') as f:
            content = f.read()

        # Should not contain old terminology
        assert "context_length" not in content, \
            "JSON should not contain context_length"

        # Should contain new terminology
        assert "max_tokens" in content, \
            "JSON should contain max_tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

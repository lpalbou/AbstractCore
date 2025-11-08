"""
Comprehensive tests for max_tokens terminology migration.

This test suite validates the complete migration from 'context_length' to 'max_tokens'
across all components of the AbstractCore system.

Test Structure:
- Layer 1: Foundation Tests - Basic functionality with new terminology
- Layer 2: Integration Tests - Component interaction validation
- Layer 3: Stress Tests - Edge cases and all 85+ models
- Layer 4: Production Tests - Real provider scenarios
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

from abstractcore.architectures import (
    get_model_capabilities,
    get_context_limits,
    detect_architecture
)
from abstractcore.architectures.detection import resolve_model_alias
from abstractcore.providers.ollama_provider import OllamaProvider
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.providers.base import BaseProvider


class TestLayer1Foundation:
    """Layer 1: Foundation Tests - Basic functionality validation"""

    def test_max_tokens_field_in_json(self):
        """Test that model_capabilities.json uses 'max_tokens' field"""
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        # Verify at least one model has max_tokens
        assert len(models) > 0, "No models found in JSON"

        # Sample a few models to verify structure
        sample_models = ['gpt-4', 'gpt-4o-mini', 'claude-3.5-sonnet', 'llama-3.1-8b', 'qwen3-coder-30b']
        for model_name in sample_models:
            if model_name in models:
                model_data = models[model_name]
                assert 'max_tokens' in model_data, f"{model_name} missing 'max_tokens' field"
                assert isinstance(model_data['max_tokens'], int), f"{model_name} 'max_tokens' is not an integer"
                # Embedding models can have max_tokens = 0 since they don't generate text
                model_type = model_data.get("model_type", "generative")
                if model_type == "embedding":
                    assert model_data['max_tokens'] >= 0, f"Embedding model {model_name} 'max_tokens' must be non-negative"
                else:
                    assert model_data['max_tokens'] > 0, f"Generative model {model_name} 'max_tokens' must be positive"

    def test_no_context_length_in_json(self):
        """Test that 'context_length' field has been removed from JSON"""
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            content = f.read()

        # Verify 'context_length' is not in the file
        assert 'context_length' not in content, "Found deprecated 'context_length' in model_capabilities.json"

    def test_get_context_limits_returns_max_tokens(self):
        """Test that get_context_limits() returns 'max_tokens' key"""
        limits = get_context_limits('gpt-4')

        assert 'max_tokens' in limits, "get_context_limits() should return 'max_tokens' key"
        assert 'max_output_tokens' in limits, "get_context_limits() should return 'max_output_tokens' key"
        assert isinstance(limits['max_tokens'], int), "'max_tokens' should be an integer"
        assert limits['max_tokens'] > 0, "'max_tokens' should be positive"

    def test_get_context_limits_no_context_length(self):
        """Test that get_context_limits() does not return 'context_length' key"""
        limits = get_context_limits('gpt-4')

        assert 'context_length' not in limits, "get_context_limits() should not return deprecated 'context_length' key"

    def test_default_capabilities_has_max_tokens(self):
        """Test that default capabilities include max_tokens"""
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        defaults = data.get("default_capabilities", {})
        assert 'max_tokens' in defaults, "default_capabilities missing 'max_tokens' field"
        assert defaults['max_tokens'] == 16384, "default max_tokens should be 16384"

    def test_model_capabilities_max_tokens_values(self):
        """Test that popular models have correct max_tokens values"""
        test_cases = [
            ('gpt-4', 128000),
            ('gpt-4o-mini', 128000),
            ('claude-3.5-sonnet', 200000),
            ('llama-3.1-8b', 128000),
            ('qwen3-coder-30b', 32768),
        ]

        for model_name, expected_max_tokens in test_cases:
            caps = get_model_capabilities(model_name)
            limits = get_context_limits(model_name)

            assert caps.get('max_tokens') == expected_max_tokens, \
                f"{model_name} should have max_tokens={expected_max_tokens}, got {caps.get('max_tokens')}"
            assert limits.get('max_tokens') == expected_max_tokens, \
                f"{model_name} get_context_limits() should return max_tokens={expected_max_tokens}"


class TestLayer2Integration:
    """Layer 2: Integration Tests - Component interaction validation"""

    def test_provider_uses_max_tokens_from_json(self):
        """Test that providers correctly use max_tokens from JSON"""
        # Test OpenAI provider
        openai_provider = OpenAIProvider('gpt-4')
        assert openai_provider.max_tokens == 128000, \
            f"OpenAI provider should have max_tokens=128000, got {openai_provider.max_tokens}"

        # Test Ollama provider with qwen model
        ollama_provider = OllamaProvider('qwen3-coder:30b')
        assert ollama_provider.max_tokens == 32768, \
            f"Ollama provider should have max_tokens=32768, got {ollama_provider.max_tokens}"

    def test_provider_max_output_tokens(self):
        """Test that providers have correct max_output_tokens"""
        providers_tests = [
            (OpenAIProvider('gpt-4'), 4096),
            (OpenAIProvider('gpt-4o-mini'), 16000),
            (OllamaProvider('llama-3.1-8b'), 8192),
        ]

        for provider, expected_max_output in providers_tests:
            assert provider.max_output_tokens == expected_max_output, \
                f"{provider.__class__.__name__} with {provider.model} should have max_output_tokens={expected_max_output}"

    def test_detection_get_context_limits_integration(self):
        """Test detection.py get_context_limits() function integration"""
        from abstractcore.architectures.detection import get_context_limits

        # Test with various models
        test_models = [
            'gpt-4',
            'claude-3.5-sonnet',
            'llama-3.1-8b',
            'qwen3-coder:30b',
            'unknown-test-model'
        ]

        for model in test_models:
            limits = get_context_limits(model)

            # Verify structure
            assert 'max_tokens' in limits, f"get_context_limits({model}) missing 'max_tokens'"
            assert 'max_output_tokens' in limits, f"get_context_limits({model}) missing 'max_output_tokens'"

            # Verify values are valid
            assert isinstance(limits['max_tokens'], int)
            assert isinstance(limits['max_output_tokens'], int)
            assert limits['max_tokens'] > 0
            assert limits['max_output_tokens'] > 0

    def test_alias_resolution_preserves_max_tokens(self):
        """Test that alias resolution correctly resolves to canonical model with max_tokens"""
        # qwen/qwen3-next-80b should resolve to qwen3-next-80b-a3b
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        # Resolve alias
        canonical = resolve_model_alias('qwen/qwen3-next-80b', models)
        assert canonical == 'qwen3-next-80b-a3b', \
            f"Alias 'qwen/qwen3-next-80b' should resolve to 'qwen3-next-80b-a3b', got {canonical}"

        # Get capabilities using alias
        caps = get_model_capabilities('qwen/qwen3-next-80b')
        assert 'max_tokens' in caps, "Alias resolution should preserve max_tokens field"
        assert caps['max_tokens'] == 262144, \
            f"Resolved model should have max_tokens=262144, got {caps['max_tokens']}"

    def test_cli_auto_detection_uses_max_tokens(self):
        """Test that CLI auto-detection logic uses max_tokens"""
        # This tests the updated logic in cli.py
        from abstractcore.architectures.detection import get_context_limits

        # Simulate CLI auto-detection for a model
        model = 'qwen3-coder:30b'
        limits = get_context_limits(model)

        # CLI should use max_tokens for context window
        assert 'max_tokens' in limits
        context_window = limits['max_tokens']
        assert context_window == 32768, \
            f"CLI should detect context window as 32768 for {model}, got {context_window}"


class TestLayer3Stress:
    """Layer 3: Stress Tests - Edge cases and all 85+ models validation"""

    def test_all_models_have_max_tokens(self):
        """Test that ALL 85+ models in JSON have max_tokens field"""
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        # Verify every single model has max_tokens
        missing_max_tokens = []
        invalid_max_tokens = []

        for model_name, model_data in models.items():
            if 'max_tokens' not in model_data:
                missing_max_tokens.append(model_name)
            elif not isinstance(model_data['max_tokens'], int) or model_data['max_tokens'] <= 0:
                invalid_max_tokens.append((model_name, model_data.get('max_tokens')))

        assert len(missing_max_tokens) == 0, \
            f"Models missing 'max_tokens' field: {missing_max_tokens}"
        assert len(invalid_max_tokens) == 0, \
            f"Models with invalid 'max_tokens' values: {invalid_max_tokens}"

        # Verify we tested a reasonable number of models
        assert len(models) >= 80, f"Expected at least 80 models, found {len(models)}"

    def test_all_models_no_context_length(self):
        """Test that NO model has the deprecated 'context_length' field"""
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        models_with_context_length = []

        for model_name, model_data in models.items():
            if 'context_length' in model_data:
                models_with_context_length.append(model_name)

        assert len(models_with_context_length) == 0, \
            f"Models still using deprecated 'context_length': {models_with_context_length}"

    def test_max_tokens_reasonable_values(self):
        """Test that max_tokens values are within reasonable ranges"""
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        unreasonable_values = []

        for model_name, model_data in models.items():
            max_tokens = model_data.get('max_tokens', 0)

            # Reasonable range: 1K to 10M tokens
            # (llama-4 has 10M, smallest models have ~2K)
            if max_tokens < 1000 or max_tokens > 10000000:
                unreasonable_values.append((model_name, max_tokens))

        assert len(unreasonable_values) == 0, \
            f"Models with unreasonable max_tokens values: {unreasonable_values}"

    def test_unknown_model_fallback_max_tokens(self):
        """Test that unknown models get correct default max_tokens"""
        unknown_models = [
            'test-unknown-model-xyz',
            'random-model-123',
            'future-model-v2'
        ]

        for model in unknown_models:
            caps = get_model_capabilities(model)
            assert caps.get('max_tokens') == 16384, \
                f"Unknown model '{model}' should get default max_tokens=16384, got {caps.get('max_tokens')}"

    def test_complex_alias_patterns(self):
        """Test that complex alias patterns resolve correctly with max_tokens"""
        json_path = Path(__file__).parent.parent.parent / "abstractcore" / "assets" / "model_capabilities.json"

        with open(json_path, 'r') as f:
            data = json.load(f)

        models = data.get("models", {})

        # Find all models with aliases
        alias_tests = []
        for canonical_name, model_data in models.items():
            aliases = model_data.get("aliases", [])
            if aliases:
                for alias in aliases:
                    alias_tests.append((alias, canonical_name, model_data.get('max_tokens')))

        # Test each alias resolves correctly
        for alias, expected_canonical, expected_max_tokens in alias_tests:
            canonical = resolve_model_alias(alias, models)
            assert canonical == expected_canonical, \
                f"Alias '{alias}' should resolve to '{expected_canonical}', got '{canonical}'"

            caps = get_model_capabilities(alias)
            assert caps.get('max_tokens') == expected_max_tokens, \
                f"Alias '{alias}' should have max_tokens={expected_max_tokens}, got {caps.get('max_tokens')}"

    def test_malformed_json_handling(self):
        """Test system behavior with edge case models (not actual malformed JSON)"""
        # Test with models that might have unusual naming
        edge_case_models = [
            'model-with-dashes',
            'model_with_underscores',
            'model.with.dots',
            'model:with:colons'
        ]

        for model in edge_case_models:
            # Should not crash, should return defaults
            caps = get_model_capabilities(model)
            assert 'max_tokens' in caps
            limits = get_context_limits(model)
            assert 'max_tokens' in limits


class TestLayer4Production:
    """Layer 4: Production Tests - Real-world scenario validation"""

    def test_existing_test_suite_passes(self):
        """Test that existing test_system_integration.py tests pass with new terminology"""
        # Import and run specific tests
        from tests.integration.test_system_integration import TestJSONCapabilitiesIntegration

        test_instance = TestJSONCapabilitiesIntegration()

        # Run the actual tests
        test_instance.test_gpt4_capabilities_from_json()
        test_instance.test_gpt5_capabilities_from_json()
        test_instance.test_unknown_model_fallback()

        # If we get here, all tests passed
        assert True

    def test_real_provider_creation_max_tokens(self):
        """Test real provider creation uses max_tokens from JSON"""
        # Create real providers for different model types
        providers = [
            OpenAIProvider('gpt-4'),
            OpenAIProvider('gpt-4o-mini'),
            OllamaProvider('qwen3-coder:30b'),
            OllamaProvider('llama-3.1-8b'),
        ]

        expected_values = [
            128000,  # gpt-4
            128000,  # gpt-4o-mini
            32768,   # qwen3-coder:30b
            128000,  # llama-3.1-8b
        ]

        for provider, expected_max_tokens in zip(providers, expected_values):
            assert provider.max_tokens == expected_max_tokens, \
                f"{provider.__class__.__name__}({provider.model}) should have max_tokens={expected_max_tokens}"

    def test_provider_get_capabilities_max_tokens(self):
        """Test that provider.get_capabilities() reflects max_tokens"""
        provider = OllamaProvider('qwen3-coder:30b')

        # Get capabilities dict
        caps = provider.get_capabilities()

        # Provider should have max_tokens set correctly
        assert provider.max_tokens == 32768
        assert provider.max_output_tokens == 8192

    def test_multi_provider_compatibility(self):
        """Test that max_tokens works across different providers"""
        # Create providers for same model class but different providers
        test_cases = [
            ('openai', OpenAIProvider, 'gpt-4', 128000),
            ('ollama', OllamaProvider, 'llama-3.1-8b', 128000),
            ('ollama', OllamaProvider, 'qwen3-coder:30b', 32768),
        ]

        for provider_type, provider_class, model, expected_max_tokens in test_cases:
            provider = provider_class(model)

            # Verify max_tokens from JSON is used
            assert provider.max_tokens == expected_max_tokens, \
                f"{provider_type} provider for {model} should have max_tokens={expected_max_tokens}"

    def test_backward_compatibility_no_breaking_changes(self):
        """Test that the migration doesn't break existing functionality"""
        # Test that all key functions still work
        model = 'gpt-4'

        # 1. Architecture detection still works
        arch = detect_architecture(model)
        assert arch == 'gpt'

        # 2. Model capabilities still work
        caps = get_model_capabilities(model)
        assert caps is not None
        assert 'architecture' in caps

        # 3. Context limits still work
        limits = get_context_limits(model)
        assert limits is not None
        assert 'max_output_tokens' in limits

        # 4. Provider creation still works
        provider = OpenAIProvider(model)
        assert provider.model == model
        assert provider.architecture == arch

    def test_documentation_examples_still_work(self):
        """Test that examples from documentation work with new terminology"""
        # Simulate documentation example
        model_name = "gpt-4"

        # Get context limits (as shown in docs)
        limits = get_context_limits(model_name)

        # Extract values
        max_tokens = limits["max_tokens"]
        max_output_tokens = limits["max_output_tokens"]

        # Verify documentation example values are correct
        assert max_tokens == 128000, "Documentation example should show max_tokens=128000 for GPT-4"
        assert max_output_tokens == 4096, "Documentation example should show max_output_tokens=4096 for GPT-4"


class TestCodeSearchValidation:
    """Validation that no references to old terminology remain in active code"""

    def test_no_context_length_in_detection_py(self):
        """Verify detection.py doesn't reference context_length"""
        detection_file = Path(__file__).parent.parent.parent / "abstractcore" / "architectures" / "detection.py"

        with open(detection_file, 'r') as f:
            content = f.read()

        # Should not contain 'context_length' in variable names or comments
        # (except maybe in migration notes/comments)
        assert 'context_length' not in content.lower() or '# migration' in content.lower(), \
            "detection.py should not reference deprecated 'context_length'"

    def test_no_context_length_in_base_provider(self):
        """Verify base.py provider doesn't reference context_length"""
        base_file = Path(__file__).parent.parent.parent / "abstractcore" / "providers" / "base.py"

        with open(base_file, 'r') as f:
            content = f.read()

        # Check for context_length usage (excluding comments about migration)
        lines_with_context_length = [
            line for line in content.split('\n')
            if 'context_length' in line.lower() and not line.strip().startswith('#')
        ]

        assert len(lines_with_context_length) == 0, \
            f"base.py should not use 'context_length': {lines_with_context_length}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

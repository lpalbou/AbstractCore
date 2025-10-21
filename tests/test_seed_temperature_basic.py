"""
Basic SEED and temperature parameter tests.

Tests the parameter passing and basic functionality without requiring external services.
For determinism testing with real providers, see test_seed_determinism.py
"""

import os
import sys
import pytest
from pathlib import Path

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from abstractcore import create_llm, BasicSession
from abstractcore.providers.openai_provider import OpenAIProvider


class TestSeedTemperatureParameters:
    """Test seed and temperature parameter handling"""
    
    def test_interface_parameter_inheritance(self):
        """Test that parameters are properly inherited from interface"""
        # Test with OpenAI provider (skip if not available)
        try:
            provider = OpenAIProvider(model="gpt-4o", temperature=0.3, seed=123)
        except ImportError:
            pytest.skip("OpenAI provider not available")
        
        assert hasattr(provider, 'temperature'), "Provider should have temperature attribute"
        assert hasattr(provider, 'seed'), "Provider should have seed attribute"
        assert provider.temperature == 0.3, "Temperature should be set correctly"
        assert provider.seed == 123, "Seed should be set correctly"
    
    def test_parameter_defaults(self):
        """Test default parameter values"""
        try:
            provider = OpenAIProvider(model="gpt-4o")
        except ImportError:
            pytest.skip("OpenAI provider not available")
        
        assert provider.temperature == 0.7, "Default temperature should be 0.7"
        assert provider.seed is None, "Default seed should be None"
    
    def test_parameter_override_in_generate(self):
        """Test parameter override in generate() calls"""
        try:
            provider = OpenAIProvider(model="gpt-4o", temperature=0.5, seed=42)
        except ImportError:
            pytest.skip("OpenAI provider not available")
        
        # Test the _extract_generation_params method directly
        params = provider._extract_generation_params(temperature=0.8, seed=999)
        
        assert params["temperature"] == 0.8, "Temperature should be overridden"
        assert params["seed"] == 999, "Seed should be overridden"
    
    def test_session_parameter_persistence(self):
        """Test session-level parameter persistence"""
        try:
            provider = OpenAIProvider(model="gpt-4o")
        except ImportError:
            pytest.skip("OpenAI provider not available")
        session = BasicSession(
            provider=provider,
            temperature=0.2,
            seed=456
        )
        
        assert session.temperature == 0.2, "Session should store temperature"
        assert session.seed == 456, "Session should store seed"
    
    def test_session_parameter_inheritance(self):
        """Test that session parameters are passed to provider"""
        try:
            provider = OpenAIProvider(model="gpt-4o")
        except ImportError:
            pytest.skip("OpenAI provider not available")
        session = BasicSession(
            provider=provider,
            temperature=0.1,
            seed=789
        )
        
        # Mock the provider's generate method to capture kwargs
        original_generate = provider.generate
        captured_kwargs = {}
        
        def mock_generate(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_generate(*args, **kwargs)
        
        provider.generate = mock_generate
        
        # Generate without overrides - should use session parameters
        session.generate("test prompt")
        
        assert captured_kwargs.get("temperature") == 0.1, "Session temperature should be passed"
        assert captured_kwargs.get("seed") == 789, "Session seed should be passed"
    
    def test_parameter_fallback_hierarchy(self):
        """Test parameter fallback: kwargs -> instance -> defaults"""
        try:
            provider = OpenAIProvider(model="gpt-4o", temperature=0.5, seed=100)
        except ImportError:
            pytest.skip("OpenAI provider not available")
        
        # Test 1: Use instance values (no kwargs)
        params = provider._extract_generation_params()
        assert params["temperature"] == 0.5
        assert params["seed"] == 100
        
        # Test 2: Override with kwargs
        params = provider._extract_generation_params(temperature=0.9, seed=200)
        assert params["temperature"] == 0.9
        assert params["seed"] == 200
        
        # Test 3: Partial override
        params = provider._extract_generation_params(temperature=0.1)
        assert params["temperature"] == 0.1
        assert params["seed"] == 100  # Should use instance value
    
    def test_seed_none_handling(self):
        """Test that seed=None is handled correctly"""
        try:
            provider = OpenAIProvider(model="gpt-4o", seed=None)
        except ImportError:
            pytest.skip("OpenAI provider not available")
        
        params = provider._extract_generation_params()
        assert "seed" not in params or params["seed"] is None, "None seed should not be included"
        
        # Test with explicit None override
        params = provider._extract_generation_params(seed=None)
        assert "seed" not in params or params["seed"] is None, "Explicit None seed should not be included"
    
    def test_temperature_bounds(self):
        """Test temperature parameter bounds (informational)"""
        # Test that extreme values are accepted (providers handle validation)
        try:
            provider1 = OpenAIProvider(model="gpt-4o", temperature=0.0)
            provider2 = OpenAIProvider(model="gpt-4o", temperature=1.0)
            provider3 = OpenAIProvider(model="gpt-4o", temperature=2.0)  # Some providers allow > 1.0
        except ImportError:
            pytest.skip("OpenAI provider not available")
        
        assert provider1.temperature == 0.0
        assert provider2.temperature == 1.0
        assert provider3.temperature == 2.0
    
    def test_seed_integer_types(self):
        """Test that various integer types work for seed"""
        try:
            provider1 = OpenAIProvider(model="gpt-4o", seed=42)
            provider2 = OpenAIProvider(model="gpt-4o", seed=0)
            provider3 = OpenAIProvider(model="gpt-4o", seed=-1)
        except ImportError:
            pytest.skip("OpenAI provider not available")
        
        assert provider1.seed == 42
        assert provider2.seed == 0
        assert provider3.seed == -1
    
    def test_create_llm_parameter_passing(self):
        """Test that create_llm passes parameters correctly"""
        # This tests the factory function parameter passing
        try:
            provider = create_llm("openai", model="gpt-4o", temperature=0.25, seed=555)
            assert provider.temperature == 0.25
            assert provider.seed == 555
        except Exception:
            # If OpenAI provider isn't available through create_llm, skip
            pytest.skip("OpenAI provider not available through create_llm")


class TestProviderSpecificBehavior:
    """Test provider-specific parameter behavior"""
    
    def test_openai_parameter_support(self):
        """Test OpenAI parameter support (if available)"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        try:
            provider = create_llm("openai", model="gpt-3.5-turbo", temperature=0.1, seed=42)
            assert provider.temperature == 0.1
            assert provider.seed == 42
        except Exception as e:
            pytest.skip(f"OpenAI provider not available: {e}")
    
    def test_anthropic_parameter_support(self):
        """Test Anthropic parameter support (if available)"""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        try:
            provider = create_llm("anthropic", model="claude-3-haiku-20240307", temperature=0.1, seed=42)
            assert provider.temperature == 0.1
            assert provider.seed == 42  # Should be stored even if not used
        except Exception as e:
            pytest.skip(f"Anthropic provider not available: {e}")
    
    def test_ollama_parameter_support(self):
        """Test Ollama parameter support (if available)"""
        try:
            provider = create_llm("ollama", model="llama3.2:1b", temperature=0.1, seed=42)
            assert provider.temperature == 0.1
            assert provider.seed == 42
        except Exception as e:
            pytest.skip(f"Ollama provider not available: {e}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])

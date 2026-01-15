"""
Basic SEED and temperature parameter tests.

Tests the parameter passing and basic functionality without requiring external services.
For determinism testing with real providers, see test_seed_determinism.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from abstractcore import BasicSession
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class DummyProvider(BaseProvider):
    """Dependency-light provider for contract tests (no network)."""

    def __init__(self, model: str = "dummy", **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "dummy"
        self.last_generate_kwargs: Dict[str, Any] = {}

    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        self.last_generate_kwargs = dict(kwargs)
        params = self._extract_generation_params(**kwargs)
        return GenerateResponse(
            content=f"ok:{prompt}",
            model=self.model,
            finish_reason="stop",
            metadata={"generation_params": params},
        )

    def get_capabilities(self) -> List[str]:
        return []

    def list_available_models(self, **kwargs) -> List[str]:
        return [self.model]

    def unload_model(self, model_name: str) -> None:
        return None


class TestSeedTemperatureParameters:
    """Test seed and temperature parameter handling"""
    
    def test_interface_parameter_inheritance(self):
        """Test that parameters are properly inherited from interface"""
        provider = DummyProvider(model="dummy", temperature=0.3, seed=123)
        
        assert hasattr(provider, 'temperature'), "Provider should have temperature attribute"
        assert hasattr(provider, 'seed'), "Provider should have seed attribute"
        assert provider.temperature == 0.3, "Temperature should be set correctly"
        assert provider.seed == 123, "Seed should be set correctly"
    
    def test_parameter_defaults(self):
        """Test default parameter values"""
        provider = DummyProvider(model="dummy")
        
        assert provider.temperature == 0.7, "Default temperature should be 0.7"
        assert provider.seed == -1, "Default seed should be -1 (random/unset)"
        params = provider._extract_generation_params()
        assert "seed" not in params, "Unset seed should not be extracted/forwarded"

        prepared = provider._prepare_generation_kwargs()
        assert "seed" not in prepared, "Unset seed should not be forwarded in generation kwargs"

    def test_seed_normalization_prepare_kwargs(self):
        provider = DummyProvider(model="dummy", seed=-1)
        assert "seed" not in provider._prepare_generation_kwargs(), "Negative seed should be treated as unset"

        provider = DummyProvider(model="dummy", seed=123)
        prepared = provider._prepare_generation_kwargs()
        assert prepared.get("seed") == 123

        prepared = provider._prepare_generation_kwargs(seed=-1)
        assert "seed" not in prepared, "Per-call negative seed should be treated as unset"
    
    def test_parameter_override_in_generate(self):
        """Test parameter override in generate() calls"""
        provider = DummyProvider(model="dummy", temperature=0.5, seed=42)
        
        # Test the _extract_generation_params method directly
        params = provider._extract_generation_params(temperature=0.8, seed=999)
        
        assert params["temperature"] == 0.8, "Temperature should be overridden"
        assert params["seed"] == 999, "Seed should be overridden"
    
    def test_session_parameter_persistence(self):
        """Test session-level parameter persistence"""
        provider = DummyProvider(model="dummy")
        session = BasicSession(
            provider=provider,
            temperature=0.2,
            seed=456
        )
        
        assert session.temperature == 0.2, "Session should store temperature"
        assert session.seed == 456, "Session should store seed"
    
    def test_session_parameter_inheritance(self):
        """Test that session parameters are passed to provider"""
        provider = DummyProvider(model="dummy")
        session = BasicSession(
            provider=provider,
            temperature=0.1,
            seed=789
        )

        # Generate without overrides - should use session parameters
        session.generate("test prompt")

        assert provider.last_generate_kwargs.get("temperature") == 0.1, "Session temperature should be passed"
        assert provider.last_generate_kwargs.get("seed") == 789, "Session seed should be passed"

    def test_session_does_not_forward_unset_seed(self):
        provider = DummyProvider(model="dummy")
        session = BasicSession(provider=provider, temperature=0.1, seed=-1)
        session.generate("test prompt")
        assert "seed" not in provider.last_generate_kwargs, "seed=-1 should not be forwarded"
    
    def test_parameter_fallback_hierarchy(self):
        """Test parameter fallback: kwargs -> instance -> defaults"""
        provider = DummyProvider(model="dummy", temperature=0.5, seed=100)
        
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
        provider = DummyProvider(model="dummy", seed=None)
        
        params = provider._extract_generation_params()
        assert "seed" not in params, "None seed should not be included"
        
        # Test with explicit None override
        params = provider._extract_generation_params(seed=None)
        assert "seed" not in params, "Explicit None seed should not be included"
    
    def test_temperature_bounds(self):
        """Test temperature parameter bounds (informational)"""
        # Test that extreme values are accepted (providers handle validation)
        provider1 = DummyProvider(model="dummy", temperature=0.0)
        provider2 = DummyProvider(model="dummy", temperature=1.0)
        provider3 = DummyProvider(model="dummy", temperature=2.0)  # Some providers allow > 1.0
        
        assert provider1.temperature == 0.0
        assert provider2.temperature == 1.0
        assert provider3.temperature == 2.0
    
    def test_seed_integer_types(self):
        """Test that various integer types work for seed"""
        provider1 = DummyProvider(model="dummy", seed=42)
        provider2 = DummyProvider(model="dummy", seed=0)
        provider3 = DummyProvider(model="dummy", seed=-1)
        
        assert provider1.seed == 42
        assert provider2.seed == 0
        assert provider3.seed == -1


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])

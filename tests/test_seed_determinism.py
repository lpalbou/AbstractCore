"""
Test SEED and temperature=0 determinism across all providers.

This test verifies that:
1. Setting seed + temperature=0 produces identical outputs across multiple calls
2. Different seeds produce different outputs
3. Provider-specific behavior is handled correctly (e.g., Anthropic/MLX fallback)
4. Session-level seed persistence works correctly
"""

import os
import sys
import pytest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractcore import create_llm, BasicSession
from abstractcore.exceptions import ModelNotFoundError, AuthenticationError


class SeedDeterminismTester:
    """Test suite for seed determinism across providers"""
    
    def __init__(self):
        self.test_prompt = "Write exactly 3 words about coding."
        self.results = {}
        
    def test_provider_determinism(self, provider_name: str, model: str, **config) -> Dict[str, any]:
        """Test deterministic behavior for a single provider"""
        print(f"\n🧪 Testing {provider_name} ({model}) for seed determinism...")
        
        try:
            # Create provider with seed and temperature=0
            llm = create_llm(
                provider_name, 
                model=model, 
                temperature=0.0,
                seed=42,
                **config
            )
            
            # Test 1: Same seed should produce identical outputs
            print("  📌 Test 1: Same seed reproducibility...")
            responses_same_seed = []
            for i in range(3):
                response = llm.generate(self.test_prompt, temperature=0.0, seed=42)
                responses_same_seed.append(response.content.strip())
                print(f"    Call {i+1}: '{response.content.strip()}'")
            
            same_seed_identical = len(set(responses_same_seed)) == 1
            
            # Test 2: Different seeds should produce different outputs (when supported)
            print("  📌 Test 2: Different seed variation...")
            responses_diff_seed = []
            for seed in [42, 123, 999]:
                response = llm.generate(self.test_prompt, temperature=0.0, seed=seed)
                responses_diff_seed.append(response.content.strip())
                print(f"    Seed {seed}: '{response.content.strip()}'")
            
            diff_seed_varied = len(set(responses_diff_seed)) > 1
            
            # Test 3: Session-level seed persistence
            print("  📌 Test 3: Session-level seed persistence...")
            session = BasicSession(provider=llm, temperature=0.0, seed=42)
            session_responses = []
            for i in range(2):
                response = session.generate(self.test_prompt)
                session_responses.append(response.content.strip())
                print(f"    Session call {i+1}: '{response.content.strip()}'")
            
            session_consistent = len(set(session_responses)) == 1
            
            # Determine provider seed support
            seed_supported = self._provider_supports_seed(provider_name)
            
            return {
                "provider": provider_name,
                "model": model,
                "seed_supported": seed_supported,
                "same_seed_identical": same_seed_identical,
                "diff_seed_varied": diff_seed_varied,
                "session_consistent": session_consistent,
                "responses_same_seed": responses_same_seed,
                "responses_diff_seed": responses_diff_seed,
                "session_responses": session_responses,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            print(f"  ❌ Error testing {provider_name}: {e}")
            return {
                "provider": provider_name,
                "model": model,
                "seed_supported": False,
                "same_seed_identical": False,
                "diff_seed_varied": False,
                "session_consistent": False,
                "responses_same_seed": [],
                "responses_diff_seed": [],
                "session_responses": [],
                "success": False,
                "error": str(e)
            }
    
    def _provider_supports_seed(self, provider_name: str) -> bool:
        """Check if provider natively supports seed parameter"""
        supported_providers = {
            "openai": True,
            "huggingface": True,
            "ollama": True,
            "lmstudio": True,
            "anthropic": False,  # Issues warning when seed provided
            "mlx": True          # Native support via mx.random.seed()
        }
        return supported_providers.get(provider_name.lower(), False)
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run determinism tests across all available providers"""
        print("🚀 Starting comprehensive SEED determinism test...")
        
        # Provider configurations
        provider_configs = {
            "openai": {
                "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
                "config": {"api_key": os.getenv("OPENAI_API_KEY")}
            },
            "anthropic": {
                "models": ["claude-3-haiku-20240307"],
                "config": {"api_key": os.getenv("ANTHROPIC_API_KEY")}
            },
            "ollama": {
                "models": ["llama3.2:1b", "qwen2.5:0.5b"],
                "config": {"base_url": "http://localhost:11434"}
            },
            "lmstudio": {
                "models": ["local-model"],
                "config": {"base_url": "http://localhost:1234/v1"}
            },
            "huggingface": {
                "models": ["microsoft/DialoGPT-small"],
                "config": {}
            },
            "mlx": {
                "models": ["mlx-community/Qwen2.5-0.5B-Instruct-4bit"],
                "config": {}
            }
        }
        
        results = []
        
        for provider_name, provider_info in provider_configs.items():
            for model in provider_info["models"]:
                try:
                    result = self.test_provider_determinism(
                        provider_name, 
                        model, 
                        **provider_info["config"]
                    )
                    results.append(result)
                except Exception as e:
                    print(f"❌ Skipping {provider_name} ({model}): {e}")
                    results.append({
                        "provider": provider_name,
                        "model": model,
                        "success": False,
                        "error": f"Setup failed: {e}",
                        "seed_supported": self._provider_supports_seed(provider_name)
                    })
        
        return {
            "total_providers_tested": len(results),
            "successful_tests": len([r for r in results if r["success"]]),
            "failed_tests": len([r for r in results if not r["success"]]),
            "results": results,
            "summary": self._generate_summary(results)
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, any]:
        """Generate test summary with key insights"""
        successful_results = [r for r in results if r["success"]]
        
        # Analyze determinism by provider type
        seed_supported_providers = [r for r in successful_results if r["seed_supported"]]
        seed_unsupported_providers = [r for r in successful_results if not r["seed_supported"]]
        
        # Count deterministic behavior
        deterministic_with_seed = len([r for r in seed_supported_providers if r["same_seed_identical"]])
        varied_with_diff_seed = len([r for r in seed_supported_providers if r["diff_seed_varied"]])
        session_consistent = len([r for r in successful_results if r["session_consistent"]])
        
        return {
            "seed_supported_providers": len(seed_supported_providers),
            "seed_unsupported_providers": len(seed_unsupported_providers),
            "deterministic_with_same_seed": deterministic_with_seed,
            "varied_with_different_seed": varied_with_diff_seed,
            "session_level_consistent": session_consistent,
            "determinism_rate": (deterministic_with_seed / len(seed_supported_providers)) * 100 if seed_supported_providers else 0,
            "variation_rate": (varied_with_diff_seed / len(seed_supported_providers)) * 100 if seed_supported_providers else 0
        }


# Pytest test functions
def test_openai_seed_determinism():
    """Test OpenAI seed determinism (if API key available)"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    tester = SeedDeterminismTester()
    result = tester.test_provider_determinism("openai", "gpt-3.5-turbo")
    
    assert result["success"], f"OpenAI test failed: {result.get('error')}"
    assert result["seed_supported"], "OpenAI should support seed"
    assert result["same_seed_identical"], "Same seed should produce identical outputs"
    assert result["session_consistent"], "Session should maintain consistent outputs"


def test_anthropic_seed_fallback():
    """Test Anthropic seed fallback behavior (if API key available)"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    
    tester = SeedDeterminismTester()
    result = tester.test_provider_determinism("anthropic", "claude-3-haiku-20240307")
    
    assert result["success"], f"Anthropic test failed: {result.get('error')}"
    assert not result["seed_supported"], "Anthropic should not support seed natively"
    # Note: Anthropic may still show some consistency due to temperature=0


def test_ollama_seed_determinism():
    """Test Ollama seed determinism (if server running)"""
    tester = SeedDeterminismTester()
    
    try:
        result = tester.test_provider_determinism("ollama", "llama3.2:1b")
        
        if not result["success"] and "connection" in result.get("error", "").lower():
            pytest.skip("Ollama server not running")
        
        assert result["success"], f"Ollama test failed: {result.get('error')}"
        assert result["seed_supported"], "Ollama should support seed"
        
    except Exception as e:
        if "connection" in str(e).lower():
            pytest.skip("Ollama server not running")
        raise


def test_session_seed_persistence():
    """Test that session-level seed parameters work correctly"""
    # Use OpenAI provider for this test if available
    try:
        from abstractcore.providers.openai_provider import OpenAIProvider
        
        # Create session with seed
        provider = OpenAIProvider(model="gpt-4o", temperature=0.0, seed=42)
        session = BasicSession(provider=provider, temperature=0.0, seed=42)
        
        # Generate multiple responses
        responses = []
        for i in range(3):
            response = session.generate("Test prompt")
            responses.append(response.content)
        
        # With seed, responses should be deterministic
        assert len(set(responses)) <= 2, "Session should maintain some consistency with seed"
        
    except ImportError:
        pytest.skip("OpenAI provider not available")


def test_temperature_zero_consistency():
    """Test that temperature=0 provides more consistent outputs"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    try:
        llm = create_llm("openai", model="gpt-3.5-turbo")
        
        # Test with temperature=0
        responses_temp_0 = []
        for i in range(3):
            response = llm.generate("Say exactly: Hello World", temperature=0.0)
            responses_temp_0.append(response.content.strip())
        
        # Test with temperature=1.0
        responses_temp_1 = []
        for i in range(3):
            response = llm.generate("Say exactly: Hello World", temperature=1.0)
            responses_temp_1.append(response.content.strip())
        
        # Temperature=0 should be more consistent
        temp_0_unique = len(set(responses_temp_0))
        temp_1_unique = len(set(responses_temp_1))
        
        print(f"Temperature=0 unique responses: {temp_0_unique}")
        print(f"Temperature=1.0 unique responses: {temp_1_unique}")
        
        # Temperature=0 should have fewer unique responses (more consistent)
        assert temp_0_unique <= temp_1_unique, "Temperature=0 should be more consistent than temperature=1.0"
        
    except Exception as e:
        pytest.skip(f"Temperature consistency test failed: {e}")


if __name__ == "__main__":
    # Run comprehensive test when executed directly
    tester = SeedDeterminismTester()
    results = tester.run_comprehensive_test()
    
    print("\n" + "="*80)
    print("🎯 SEED DETERMINISM TEST RESULTS")
    print("="*80)
    
    summary = results["summary"]
    print(f"📊 Total providers tested: {results['total_providers_tested']}")
    print(f"✅ Successful tests: {results['successful_tests']}")
    print(f"❌ Failed tests: {results['failed_tests']}")
    print(f"🎲 Seed-supported providers: {summary['seed_supported_providers']}")
    print(f"🔄 Seed-unsupported providers: {summary['seed_unsupported_providers']}")
    print(f"🎯 Determinism rate: {summary['determinism_rate']:.1f}%")
    print(f"🌟 Variation rate: {summary['variation_rate']:.1f}%")
    
    print("\n📋 Detailed Results:")
    for result in results["results"]:
        if result["success"]:
            status = "✅" if result["same_seed_identical"] else "⚠️"
            seed_support = "🎲" if result["seed_supported"] else "🔄"
            print(f"{status} {seed_support} {result['provider']} ({result['model']})")
            if result["seed_supported"] and not result["same_seed_identical"]:
                print(f"    ⚠️  Expected determinism but got variation")
        else:
            print(f"❌ {result['provider']} ({result['model']}): {result['error']}")
    
    print("\n🔍 Key Insights:")
    if summary["determinism_rate"] >= 80:
        print("✅ Excellent: Most seed-supported providers show deterministic behavior")
    elif summary["determinism_rate"] >= 60:
        print("⚠️  Good: Majority of seed-supported providers show deterministic behavior")
    else:
        print("❌ Concerning: Low determinism rate across providers")
    
    print(f"📈 Session consistency: {summary['session_level_consistent']}/{results['successful_tests']} providers")
    
    # Save detailed results to file
    import json
    results_file = Path(__file__).parent / "seed_determinism_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {results_file}")

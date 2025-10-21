#!/usr/bin/env python3
"""
Manual SEED and temperature determinism verification script.

This script can be run manually to test actual determinism with available providers.
It's designed to be simple and provide clear visual feedback.

Usage:
    python tests/manual_seed_verification.py
    
    # Test specific provider:
    python tests/manual_seed_verification.py --provider openai
    
    # Test with custom prompt:
    python tests/manual_seed_verification.py --prompt "Count to 5"
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractcore import create_llm, BasicSession


def test_provider_determinism(provider_name: str, model: str, test_prompt: str, **config):
    """Test determinism for a single provider with visual output"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {provider_name.upper()} ({model})")
    print(f"{'='*60}")
    
    try:
        # Create provider with deterministic settings
        print(f"📝 Prompt: '{test_prompt}'")
        print(f"⚙️  Settings: temperature=0.0, seed=42")
        
        llm = create_llm(
            provider_name,
            model=model,
            temperature=0.0,
            seed=42,
            **config
        )
        
        print(f"\n🔄 Testing same seed reproducibility (3 calls)...")
        responses = []
        for i in range(3):
            try:
                response = llm.generate(test_prompt, temperature=0.0, seed=42)
                content = response.content.strip()
                responses.append(content)
                print(f"  Call {i+1}: '{content}'")
            except Exception as e:
                print(f"  Call {i+1}: ❌ Error: {e}")
                return False
        
        # Check determinism
        unique_responses = set(responses)
        is_deterministic = len(unique_responses) == 1
        
        print(f"\n📊 Results:")
        print(f"  Unique responses: {len(unique_responses)}")
        print(f"  Deterministic: {'✅ YES' if is_deterministic else '❌ NO'}")
        
        if not is_deterministic:
            print(f"  ⚠️  Expected identical responses but got {len(unique_responses)} different ones")
            for i, resp in enumerate(unique_responses, 1):
                print(f"    Variant {i}: '{resp}'")
        
        # Test different seeds
        print(f"\n🎲 Testing different seeds (should vary)...")
        seed_responses = []
        for seed in [42, 123, 999]:
            try:
                response = llm.generate(test_prompt, temperature=0.0, seed=seed)
                content = response.content.strip()
                seed_responses.append(content)
                print(f"  Seed {seed}: '{content}'")
            except Exception as e:
                print(f"  Seed {seed}: ❌ Error: {e}")
        
        unique_seed_responses = set(seed_responses)
        has_variation = len(unique_seed_responses) > 1
        
        print(f"\n📊 Seed variation:")
        print(f"  Unique responses: {len(unique_seed_responses)}")
        print(f"  Has variation: {'✅ YES' if has_variation else '⚠️  NO (may not support seed)'}")
        
        # Test session persistence
        print(f"\n🔗 Testing session-level seed persistence...")
        session = BasicSession(provider=llm, temperature=0.0, seed=42)
        session_responses = []
        for i in range(2):
            try:
                response = session.generate(test_prompt)
                content = response.content.strip()
                session_responses.append(content)
                print(f"  Session call {i+1}: '{content}'")
            except Exception as e:
                print(f"  Session call {i+1}: ❌ Error: {e}")
        
        session_consistent = len(set(session_responses)) == 1
        print(f"  Session consistent: {'✅ YES' if session_consistent else '❌ NO'}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if is_deterministic and session_consistent:
            print(f"  ✅ EXCELLENT: Full determinism support")
        elif is_deterministic or session_consistent:
            print(f"  ⚠️  PARTIAL: Some determinism features work")
        else:
            print(f"  ❌ LIMITED: No clear determinism (may not support seed)")
        
        return is_deterministic
        
    except Exception as e:
        print(f"❌ Failed to test {provider_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SEED determinism across providers")
    parser.add_argument("--provider", help="Test specific provider (openai, anthropic, ollama, etc.)")
    parser.add_argument("--prompt", default="Write exactly 3 words about coding.", help="Test prompt")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🚀 AbstractCore SEED Determinism Verification")
    print("=" * 50)
    
    # Provider configurations
    providers = {
        "openai": {
            "models": ["gpt-3.5-turbo"],
            "config": {"api_key": os.getenv("OPENAI_API_KEY")},
            "required_env": "OPENAI_API_KEY"
        },
        "anthropic": {
            "models": ["claude-3-haiku-20240307"],
            "config": {"api_key": os.getenv("ANTHROPIC_API_KEY")},
            "required_env": "ANTHROPIC_API_KEY"
        },
        "ollama": {
            "models": ["llama3.2:1b"],
            "config": {"base_url": "http://localhost:11434"},
            "required_env": None
        },
        "lmstudio": {
            "models": ["local-model"],
            "config": {"base_url": "http://localhost:1234/v1"},
            "required_env": None
        }
    }
    
    # Filter providers if specific one requested
    if args.provider:
        if args.provider.lower() in providers:
            providers = {args.provider.lower(): providers[args.provider.lower()]}
        else:
            print(f"❌ Unknown provider: {args.provider}")
            print(f"Available providers: {', '.join(providers.keys())}")
            return 1
    
    results = {}
    tested_count = 0
    successful_count = 0
    
    for provider_name, provider_info in providers.items():
        # Check if required environment variable is set
        if provider_info["required_env"] and not os.getenv(provider_info["required_env"]):
            print(f"\n⏭️  Skipping {provider_name}: {provider_info['required_env']} not set")
            continue
        
        for model in provider_info["models"]:
            tested_count += 1
            try:
                success = test_provider_determinism(
                    provider_name,
                    model,
                    args.prompt,
                    **provider_info["config"]
                )
                results[f"{provider_name}:{model}"] = success
                if success:
                    successful_count += 1
            except KeyboardInterrupt:
                print("\n⏹️  Test interrupted by user")
                return 1
            except Exception as e:
                print(f"\n❌ Unexpected error testing {provider_name}: {e}")
                results[f"{provider_name}:{model}"] = False
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY")
    print(f"{'='*60}")
    print(f"Total providers tested: {tested_count}")
    print(f"Deterministic providers: {successful_count}")
    print(f"Success rate: {(successful_count/tested_count*100) if tested_count > 0 else 0:.1f}%")
    
    print(f"\n📊 Detailed Results:")
    for provider_model, success in results.items():
        status = "✅ Deterministic" if success else "❌ Non-deterministic"
        print(f"  {provider_model}: {status}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if successful_count == 0:
        print("  • No providers showed deterministic behavior")
        print("  • Check if providers support seed parameter")
        print("  • Verify API keys and connectivity")
    elif successful_count < tested_count:
        print("  • Some providers don't support seed natively (e.g., Anthropic)")
        print("  • Use deterministic providers for reproducible results")
        print("  • Consider temperature=0 for more consistent outputs")
    else:
        print("  • All tested providers show deterministic behavior! 🎉")
        print("  • You can rely on seed+temperature=0 for reproducible results")
    
    print(f"\n🔍 Provider-specific notes:")
    print(f"  • OpenAI: Native seed support (except o1 models)")
    print(f"  • Anthropic: No seed support (issues warning when provided)")
    print(f"  • Ollama: Native seed support")
    print(f"  • LMStudio: OpenAI-compatible seed support")
    print(f"  • HuggingFace: Full seed support (transformers + GGUF)")
    print(f"  • MLX: Native seed support via mx.random.seed()")
    
    return 0 if successful_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Example 1: Basic Generation - Your First AbstractLLM Experience
================================================================

This example demonstrates the fundamental concepts of AbstractLLM Core:
- Creating an LLM instance with any provider
- Basic text generation
- Understanding the unified interface
- Exploring response objects

Technical Architecture Highlights:
- Factory pattern for provider abstraction
- Unified token parameter system
- Consistent response structure across providers

Required: pip install abstractllm
Optional: pip install abstractllm[ollama] for local models
"""

import os
import sys
from typing import Optional
import logging

# Add project root to path for development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from abstractllm import create_llm, GenerateResponse
from abstractllm.exceptions import ModelNotFoundError, ProviderAPIError

# Configure logging to see what's happening under the hood
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def basic_generation_example():
    """
    Demonstrates the simplest AbstractLLM usage pattern.

    Architecture Notes:
    - create_llm() is the universal factory function
    - Provider selection happens at runtime, not compile time
    - Same code works with OpenAI, Anthropic, Ollama, etc.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Text Generation")
    print("=" * 70)

    # Create an LLM instance - defaults to mock provider for testing
    # In production, you'd use: create_llm("openai", "gpt-4o") or similar
    llm = create_llm(
        provider="mock",
        model="mock-model",
        max_tokens=2048,        # Total context window budget
        max_output_tokens=500   # Reserve 500 tokens for response
    )

    # Generate a response - simple and clean
    print("\n📝 Generating response...")
    response = llm.generate("Explain quantum computing in one paragraph.")

    # The response is a GenerateResponse object with rich metadata
    print(f"\n🤖 Response: {response.content}")
    print(f"\n📊 Metadata:")
    print(f"   • Model: {response.model}")
    print(f"   • Finish reason: {response.finish_reason}")
    if response.usage:
        print(f"   • Tokens used: {response.usage.get('total_tokens', 'N/A')}")


def explore_response_object():
    """
    Deep dive into the GenerateResponse object structure.

    Architecture Notes:
    - GenerateResponse provides consistent interface across all providers
    - Includes token usage, finish reasons, and raw provider data
    - Designed for both simple use and advanced monitoring
    """
    print("\n" + "=" * 70)
    print("Understanding the Response Object")
    print("=" * 70)

    llm = create_llm("mock", "mock-model", max_tokens=1000)

    # Generate with more complex prompt
    prompt = """
    Create a haiku about artificial intelligence.
    Make it thoughtful and contemplative.
    """

    response = llm.generate(prompt)

    # Explore all response attributes
    print("\n📦 Full Response Structure:")
    print(f"   • content: {response.content[:50]}..." if len(response.content) > 50 else f"   • content: {response.content}")
    print(f"   • model: {response.model}")
    print(f"   • finish_reason: {response.finish_reason}")
    print(f"   • usage: {response.usage}")
    print(f"   • raw_response available: {response.raw_response is not None}")

    # Performance tip: Token usage tracking
    if response.usage:
        prompt_tokens = response.usage.get('prompt_tokens', 0)
        completion_tokens = response.usage.get('completion_tokens', 0)
        total_tokens = response.usage.get('total_tokens', 0)

        print(f"\n💰 Token Economics:")
        print(f"   • Prompt: {prompt_tokens} tokens")
        print(f"   • Completion: {completion_tokens} tokens")
        print(f"   • Total: {total_tokens} tokens")
        print(f"   • Efficiency: {completion_tokens/total_tokens*100:.1f}% output vs input")


def provider_switching_demo():
    """
    Demonstrates seamless provider switching - same code, different backends.

    Architecture Notes:
    - Provider abstraction is core to AbstractLLM's design
    - Switch providers without changing application code
    - Enables A/B testing, fallbacks, and multi-provider strategies
    """
    print("\n" + "=" * 70)
    print("Provider Flexibility - Write Once, Run Anywhere")
    print("=" * 70)

    # List of providers to try (mock for demo, but could be real providers)
    providers = [
        ("mock", "mock-gpt-4"),
        # Uncomment these to test with real providers:
        # ("openai", "gpt-4o-mini"),
        # ("anthropic", "claude-3-5-haiku-latest"),
        # ("ollama", "qwen3-coder:30b"),
    ]

    prompt = "What is 2+2? Answer in exactly one word."

    for provider_name, model_name in providers:
        try:
            print(f"\n🔄 Testing {provider_name} with {model_name}...")

            # Same create_llm pattern, different provider
            llm = create_llm(
                provider=provider_name,
                model=model_name,
                max_tokens=100  # Small context for simple query
            )

            # Same generate() call
            response = llm.generate(prompt)

            print(f"   ✅ Response: {response.content}")
            print(f"   📊 Tokens: {response.usage.get('total_tokens', 'N/A') if response.usage else 'N/A'}")

        except (ModelNotFoundError, ProviderAPIError) as e:
            print(f"   ❌ Error: {e}")
        except ImportError as e:
            print(f"   ⚠️ Provider not installed: {e}")


def error_handling_patterns():
    """
    Demonstrates robust error handling patterns.

    Architecture Notes:
    - Consistent exception hierarchy across providers
    - Graceful degradation strategies
    - Production-ready error recovery
    """
    print("\n" + "=" * 70)
    print("Error Handling Best Practices")
    print("=" * 70)

    # Pattern 1: Handle specific exceptions
    try:
        llm = create_llm("openai", "gpt-99-ultra")  # Non-existent model
        response = llm.generate("Hello")
    except ModelNotFoundError as e:
        print(f"\n❌ Model not found: {e}")
        print("   💡 Tip: Check available models with your provider")
    except ProviderAPIError as e:
        print(f"\n❌ API error: {e}")
        print("   💡 Tip: Check API keys and network connectivity")
    except ImportError:
        print("\n⚠️ OpenAI not installed - using mock provider instead")
        llm = create_llm("mock", "mock-model")
        response = llm.generate("Hello")
        print(f"   ✅ Fallback response: {response.content}")

    # Pattern 2: Fallback chain
    print("\n🔄 Implementing provider fallback chain...")

    fallback_providers = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-latest"),
        ("ollama", "qwen3-coder:30b"),
        ("mock", "mock-model"),  # Always available fallback
    ]

    for provider, model in fallback_providers:
        try:
            print(f"   Trying {provider}...")
            llm = create_llm(provider, model)
            response = llm.generate("Say 'Hello, AbstractLLM!'")
            print(f"   ✅ Success with {provider}: {response.content}")
            break
        except Exception as e:
            print(f"   ❌ {provider} failed: {type(e).__name__}")
            continue


def token_management_insights():
    """
    Deep dive into AbstractLLM's unified token management system.

    Architecture Notes:
    - Unified vocabulary: max_tokens, max_output_tokens, max_input_tokens
    - Automatic validation and constraint checking
    - Provider-agnostic token budgeting
    """
    print("\n" + "=" * 70)
    print("Token Management Architecture")
    print("=" * 70)

    # Strategy 1: Budget + Output Reserve (Recommended)
    print("\n📊 Strategy 1: Total Budget with Output Reserve")
    llm = create_llm(
        "mock",
        "mock-model",
        max_tokens=8000,        # Total budget for input + output
        max_output_tokens=2000  # Reserve 2000 for generation
    )

    # The LLM automatically calculates max_input_tokens = 6000
    print(f"   • Total budget: 8000 tokens")
    print(f"   • Output reserve: 2000 tokens")
    print(f"   • Input capacity: 6000 tokens (auto-calculated)")

    # Strategy 2: Explicit Input/Output (Advanced)
    print("\n📊 Strategy 2: Explicit Input and Output Limits")
    llm = create_llm(
        "mock",
        "mock-model",
        max_input_tokens=6000,   # Explicit input limit
        max_output_tokens=2000   # Explicit output limit
    )

    print(f"   • Input limit: 6000 tokens")
    print(f"   • Output limit: 2000 tokens")
    print(f"   • Total usage: up to 8000 tokens")

    # Performance measurement
    print("\n⚡ Performance Implications:")
    import time

    start = time.perf_counter()
    response = llm.generate("Write a short poem about AI.")
    elapsed = time.perf_counter() - start

    print(f"   • Generation time: {elapsed*1000:.2f}ms")
    print(f"   • Throughput: ~{len(response.content)/elapsed:.0f} chars/sec")

    if response.usage:
        tokens = response.usage.get('completion_tokens', 0)
        if tokens > 0:
            print(f"   • Token generation rate: ~{tokens/elapsed:.0f} tokens/sec")


def main():
    """
    Main entry point - runs all examples in sequence.

    This demonstrates the progressive complexity of AbstractLLM,
    from simple generation to advanced architectural patterns.
    """
    print("\n" + "🚀 " * 20)
    print(" AbstractLLM Core - Example 1: Basic Generation")
    print("🚀 " * 20)

    # Run examples in order of increasing complexity
    basic_generation_example()
    explore_response_object()
    provider_switching_demo()
    error_handling_patterns()
    token_management_insights()

    print("\n" + "=" * 70)
    print("✅ Example 1 Complete!")
    print("\nKey Takeaways:")
    print("• AbstractLLM provides a unified interface to all LLM providers")
    print("• Same code works with OpenAI, Anthropic, Ollama, and more")
    print("• Consistent token management and response structure")
    print("• Production-ready error handling and fallback patterns")
    print("\nNext: Run example_2_provider_configuration.py for advanced provider setup")
    print("=" * 70)


if __name__ == "__main__":
    main()
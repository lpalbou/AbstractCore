"""
Simple test of all local providers.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractllm import create_llm, BasicSession


def test_provider(provider_name, model, config=None):
    """Test a single provider"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} with model {model}")
    print('='*60)

    try:
        # Create provider
        llm = create_llm(provider_name, model=model, **(config or {}))

        # Test simple generation
        start = time.time()
        response = llm.generate("Who are you in one sentence?")
        elapsed = time.time() - start

        if response and response.content:
            print(f"✅ Response received in {elapsed:.2f}s:")
            print(f"   {response.content[:200]}...")
            return True
        else:
            print(f"❌ No response received")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Test main providers"""

    # Test configurations
    tests = [
        # Ollama with Qwen3-Coder
        ("ollama", "qwen3-coder:30b", {"base_url": "http://localhost:11434"}),

        # MLX with Qwen3-Coder
        ("mlx", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", {}),

        # LMStudio (if running)
        ("lmstudio", "local-model", {"base_url": "http://localhost:1234"})
    ]

    results = []
    for provider, model, config in tests:
        success = test_provider(provider, model, config)
        results.append((provider, success))

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for provider, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {provider.upper()}")

    # Test session with best working provider
    print(f"\n{'='*60}")
    print("Testing BasicSession with Ollama")
    print('='*60)

    try:
        llm = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")
        session = BasicSession(provider=llm, system_prompt="You are a helpful AI assistant.")

        # Test conversation
        resp1 = session.generate("What is 2+2?")
        print(f"Q: What is 2+2?")
        print(f"A: {resp1.content}")

        resp2 = session.generate("What was my previous question?")
        print(f"Q: What was my previous question?")
        print(f"A: {resp2.content}")

        if "2+2" in resp2.content.lower() or "math" in resp2.content.lower():
            print("✅ Session maintains context")
        else:
            print("⚠️  Session may not be maintaining context")

    except Exception as e:
        print(f"❌ Session test error: {e}")


if __name__ == "__main__":
    main()
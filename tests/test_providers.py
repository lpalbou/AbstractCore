"""
Test all providers with real implementations.
No mocking - test actual provider connections.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractllm import create_llm, BasicSession
from abstractllm.core.types import GenerateResponse


def test_simple_message(provider_name: str, model: str, config: Dict[str, Any] = None) -> bool:
    """Test simple message generation with a provider"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} with model: {model}")
    print('='*60)

    try:
        # Create provider
        provider_config = config or {}
        provider = create_llm(provider_name, model=model, **provider_config)

        # Test simple generation
        prompt = "Who are you? Please respond in one sentence."
        print(f"Prompt: {prompt}")

        start_time = time.time()
        response = provider.generate(prompt)
        elapsed = time.time() - start_time

        if response and response.content:
            print(f"✅ Response received in {elapsed:.2f}s:")
            print(f"   {response.content[:200]}...")
            if response.usage:
                print(f"   Tokens: {response.usage}")
            return True
        else:
            print(f"❌ No response received")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_tool_call(provider_name: str, model: str, config: Dict[str, Any] = None) -> bool:
    """Test tool calling with a provider"""
    print(f"\n{'='*60}")
    print(f"Testing tool calls for {provider_name} with model: {model}")
    print('='*60)

    try:
        # Create provider
        provider_config = config or {}
        provider = create_llm(provider_name, model=model, **provider_config)

        # Define a simple tool
        tools = [{
            "name": "list_files",
            "description": "List files in the current directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list files from"
                    }
                },
                "required": ["path"]
            }
        }]

        # Test tool generation
        prompt = "Please list the files in the current directory"
        print(f"Prompt: {prompt}")
        print(f"Available tools: {[t['name'] for t in tools]}")

        start_time = time.time()
        response = provider.generate(prompt, tools=tools)
        elapsed = time.time() - start_time

        if response:
            if response.has_tool_calls():
                print(f"✅ Tool call response received in {elapsed:.2f}s:")
                for tool_call in response.tool_calls:
                    print(f"   Tool: {tool_call.get('name')}")
                    print(f"   Args: {tool_call.get('arguments')}")
                return True
            elif response.content:
                print(f"⚠️  Response received but no tool calls (provider may not support tools):")
                print(f"   {response.content[:200]}...")
                return False
        else:
            print(f"❌ No response received")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_session_with_provider(provider_name: str, model: str, config: Dict[str, Any] = None) -> bool:
    """Test BasicSession with a provider"""
    print(f"\n{'='*60}")
    print(f"Testing BasicSession with {provider_name}")
    print('='*60)

    try:
        # Create provider
        provider_config = config or {}
        provider = create_llm(provider_name, model=model, **provider_config)

        # Create session
        session = BasicSession(
            provider=provider,
            system_prompt="You are a helpful assistant."
        )

        # Test conversation
        response1 = session.generate("What is 2+2?")
        print(f"Q: What is 2+2?")
        print(f"A: {response1.content}")

        response2 = session.generate("What was my previous question?")
        print(f"Q: What was my previous question?")
        print(f"A: {response2.content}")

        # Check if context is maintained
        if "2+2" in response2.content.lower() or "math" in response2.content.lower():
            print("✅ Session maintains context correctly")
            return True
        else:
            print("⚠️  Session may not be maintaining context")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run all provider tests"""

    # Test configurations for each provider
    test_configs = [
        # Ollama - test with local model
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:3b",  # Using smaller model for testing
            "config": {"base_url": "http://localhost:11434"}
        },
        # LMStudio - test with local model
        {
            "provider": "lmstudio",
            "model": "qwen/qwen2.5-coder-3b",
            "config": {"base_url": "http://localhost:1234"}
        },
        # MLX - test with local model
        {
            "provider": "mlx",
            "model": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
            "config": {}
        }
    ]

    # Also test OpenAI and Anthropic if API keys are available
    if os.getenv("OPENAI_API_KEY"):
        test_configs.append({
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "config": {}
        })
    else:
        print("⚠️  Skipping OpenAI tests - OPENAI_API_KEY not set")

    if os.getenv("ANTHROPIC_API_KEY"):
        test_configs.append({
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307",
            "config": {}
        })
    else:
        print("⚠️  Skipping Anthropic tests - ANTHROPIC_API_KEY not set")

    # Run tests for each provider
    results = {}

    for config in test_configs:
        provider = config["provider"]
        model = config["model"]
        provider_config = config["config"]

        print(f"\n{'#'*60}")
        print(f"# Testing {provider.upper()} Provider")
        print('#'*60)

        # Check if provider is available
        if provider in ["ollama", "lmstudio"]:
            # Check if server is running
            import httpx
            try:
                base_url = provider_config.get("base_url", "http://localhost:11434" if provider == "ollama" else "http://localhost:1234")
                client = httpx.Client(timeout=5.0)
                response = client.get(f"{base_url}/api/tags" if provider == "ollama" else f"{base_url}/v1/models")
                if response.status_code != 200:
                    print(f"⚠️  {provider} server not responding at {base_url}")
                    continue
            except Exception as e:
                print(f"⚠️  {provider} server not available: {e}")
                continue

        # Run tests
        test_results = {
            "simple_message": test_simple_message(provider, model, provider_config),
            "tool_call": test_tool_call(provider, model, provider_config),
            "session": test_session_with_provider(provider, model, provider_config)
        }

        results[provider] = test_results

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    for provider, test_results in results.items():
        passed = sum(1 for v in test_results.values() if v)
        total = len(test_results)
        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        print(f"{status} {provider.upper()}: {passed}/{total} tests passed")
        for test_name, passed in test_results.items():
            print(f"   {'✅' if passed else '❌'} {test_name}")

    # Return overall success
    all_passed = all(all(tr.values()) for tr in results.values())
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
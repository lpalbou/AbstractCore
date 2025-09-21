"""
Test tool calling functionality across providers.
"""

import sys
import os
import json
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractllm import create_llm, BasicSession
from abstractllm.tools.core import ToolDefinition


def list_files(directory: str = ".") -> str:
    """List files in the specified directory"""
    import os
    try:
        files = os.listdir(directory)
        return f"Files in {directory}: {', '.join(files[:10])}"
    except Exception as e:
        return f"Error listing files: {str(e)}"


def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        # Safe evaluation for simple math
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def get_weather(city: str) -> str:
    """Get weather for a city (mock implementation)"""
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 59°F",
        "Tokyo": "Rainy, 68°F",
        "Paris": "Partly cloudy, 64°F"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def test_tool_calling(provider_name: str, model: str, config: Dict[str, Any] = None):
    """Test tool calling with a provider"""
    print(f"\n{'='*60}")
    print(f"Testing tool calling for {provider_name}")
    print('='*60)

    try:
        # Create provider
        llm = create_llm(provider_name, model=model, **(config or {}))

        # Define tools in OpenAI format
        tools = [
            {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path"
                        }
                    },
                    "required": ["directory"]
                }
            },
            {
                "name": "calculate",
                "description": "Calculate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["city"]
                }
            }
        ]

        # Test prompts that should trigger tool use
        test_cases = [
            ("What files are in the current directory?", "list_files"),
            ("What is 42 * 17?", "calculate"),
            ("What's the weather in New York?", "get_weather")
        ]

        results = []

        for prompt, expected_tool in test_cases:
            print(f"\nPrompt: {prompt}")
            print(f"Expected tool: {expected_tool}")

            start = time.time()
            response = llm.generate(prompt, tools=tools)
            elapsed = time.time() - start

            if response:
                if response.has_tool_calls():
                    print(f"✅ Tool calls detected in {elapsed:.2f}s:")
                    for tc in response.tool_calls:
                        print(f"   Tool: {tc.get('name')}")
                        print(f"   Args: {tc.get('arguments')}")

                        # Execute the tool
                        tool_name = tc.get('name')
                        if tool_name == 'list_files':
                            args = json.loads(tc.get('arguments')) if isinstance(tc.get('arguments'), str) else tc.get('arguments', {})
                            result = list_files(args.get('directory', '.'))
                        elif tool_name == 'calculate':
                            args = json.loads(tc.get('arguments')) if isinstance(tc.get('arguments'), str) else tc.get('arguments', {})
                            result = calculate(args.get('expression', ''))
                        elif tool_name == 'get_weather':
                            args = json.loads(tc.get('arguments')) if isinstance(tc.get('arguments'), str) else tc.get('arguments', {})
                            result = get_weather(args.get('city', ''))
                        else:
                            result = "Unknown tool"

                        print(f"   Result: {result}")

                    results.append(True)
                elif response.content:
                    print(f"⚠️  No tool calls, got text response:")
                    print(f"   {response.content[:200]}...")
                    results.append(False)
                else:
                    print(f"❌ No response")
                    results.append(False)
            else:
                print(f"❌ Error: No response received")
                results.append(False)

        success_rate = sum(results) / len(results) if results else 0
        print(f"\nSuccess rate: {success_rate:.1%} ({sum(results)}/{len(results)})")
        return success_rate > 0.5

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_ollama_tool_format():
    """Test Ollama with architecture-specific tool format"""
    print(f"\n{'='*60}")
    print("Testing Ollama with Qwen3 tool format")
    print('='*60)

    try:
        llm = create_llm("ollama", model="qwen3-coder:30b")

        # Qwen3 uses special tool format
        prompt = """<|tool_call|>
list_files
{"directory": "/tmp"}
<|tool_call_end|>

Please list the files in /tmp directory."""

        response = llm.generate(prompt)
        if response and response.content:
            print(f"✅ Response: {response.content[:200]}...")
            return True
        else:
            print("❌ No response")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run tool calling tests"""

    # Test configurations
    tests = [
        # Ollama - may need architecture-specific handling
        ("ollama", "qwen3-coder:30b", {"base_url": "http://localhost:11434"}),
    ]

    # Add OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        tests.append(("openai", "gpt-3.5-turbo", {}))

    # Add Anthropic if available
    if os.getenv("ANTHROPIC_API_KEY"):
        tests.append(("anthropic", "claude-3-haiku-20240307", {}))

    results = {}

    for provider, model, config in tests:
        success = test_tool_calling(provider, model, config)
        results[provider] = success

    # Also test Ollama-specific format
    ollama_special = test_ollama_tool_format()

    # Print summary
    print(f"\n{'='*60}")
    print("TOOL CALLING TEST SUMMARY")
    print('='*60)

    for provider, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {provider.upper()}: {'Supported' if success else 'Not supported/Failed'}")

    print(f"{'✅' if ollama_special else '❌'} Ollama architecture-specific: {'Working' if ollama_special else 'Failed'}")


if __name__ == "__main__":
    main()
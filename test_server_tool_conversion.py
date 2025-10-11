#!/usr/bin/env python3
"""
Test script to verify universal tool call conversion in the AbstractCore server.

This tests that tool calls in various formats (<|tool_call|>, <function_call>, etc.)
are properly converted to OpenAI's JSON format for Codex/Cline compatibility.
"""

import json
import requests
import time
from typing import Dict, List

# Server configuration
SERVER_URL = "http://localhost:9090"
API_ENDPOINT = f"{SERVER_URL}/v1/chat/completions"


def test_tool_call_conversion(model: str, stream: bool = False):
    """Test tool call conversion for a specific model."""

    print(f"\n{'='*60}")
    print(f"Testing: {model} (stream={stream})")
    print('='*60)

    # Prepare the request with tools
    request_data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to tools."
            },
            {
                "role": "user",
                "content": "List the files in the current directory using the shell command 'ls -la'"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Command and arguments as array"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ],
        "stream": stream
    }

    try:
        # Make the request
        response = requests.post(API_ENDPOINT, json=request_data)

        if stream:
            print("\nStreaming Response:")
            print("-" * 40)

            # Process SSE stream
            tool_calls_detected = False
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})

                                # Check for content
                                if "content" in delta and delta["content"]:
                                    print(f"Content: {delta['content']}", end='')

                                # Check for tool calls
                                if "tool_calls" in delta:
                                    tool_calls_detected = True
                                    print(f"\n✅ Tool call detected in stream!")
                                    for tool_call in delta["tool_calls"]:
                                        print(f"   - Function: {tool_call.get('function', {}).get('name')}")
                                        print(f"   - Arguments: {tool_call.get('function', {}).get('arguments')}")
                                        print(f"   - ID: {tool_call.get('id')}")
                                        print(f"   - Type: {tool_call.get('type')}")

                        except json.JSONDecodeError as e:
                            print(f"Failed to parse chunk: {e}")

            if not tool_calls_detected:
                print("\n⚠️ No tool calls detected in streaming response!")
                print("   The model may have output raw tool call tags instead of converted format.")

        else:
            print("\nNon-Streaming Response:")
            print("-" * 40)

            if response.status_code == 200:
                data = response.json()

                # Check for tool calls in the response
                if "choices" in data and data["choices"]:
                    message = data["choices"][0].get("message", {})

                    # Display content if present
                    if message.get("content"):
                        print(f"Content: {message['content']}")

                    # Check for tool calls
                    if "tool_calls" in message:
                        print("\n✅ Tool calls detected in response!")
                        for tool_call in message["tool_calls"]:
                            print(f"   - ID: {tool_call.get('id')}")
                            print(f"   - Type: {tool_call.get('type')}")
                            print(f"   - Function: {tool_call.get('function', {}).get('name')}")
                            print(f"   - Arguments: {tool_call.get('function', {}).get('arguments')}")
                    else:
                        print("\n⚠️ No tool calls detected in response!")
                        print("   Check if model output contains raw tool call tags.")

                    # Show finish reason
                    finish_reason = data["choices"][0].get("finish_reason")
                    print(f"\nFinish reason: {finish_reason}")

                    if finish_reason != "tool_calls" and message.get("content"):
                        # Check for raw tool call tags in content
                        content = message["content"]
                        if any(tag in content for tag in ['<|tool_call|>', '<function_call>', '<tool_call>']):
                            print("\n❌ ERROR: Raw tool call tags found in content!")
                            print("   Tool calls were not properly converted to OpenAI format.")
                            print(f"   Raw content: {content[:200]}...")

            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text}")

    except Exception as e:
        print(f"❌ Error testing {model}: {e}")


def main():
    """Run tests for multiple models."""

    print("Universal Tool Call Conversion Test")
    print("====================================")
    print(f"Server: {SERVER_URL}")
    print(f"Endpoint: {API_ENDPOINT}")

    # Test models that might output different tool call formats
    test_models = [
        "ollama/qwen3-coder:30b",       # Should output <|tool_call|> format
        "ollama/qwen3-next-80b",        # Should output <function_call> format
        "lmstudio/qwen/qwen3-coder",    # LMStudio variant
        "lmstudio/qwen/qwen3-next-80b", # LMStudio variant
    ]

    # Check server availability
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code != 200:
            print(f"\n❌ Server not responding at {SERVER_URL}")
            print("Please start the AbstractCore server first:")
            print("  python -m abstractllm.server")
            return
    except:
        print(f"\n❌ Cannot connect to server at {SERVER_URL}")
        print("Please start the AbstractCore server first:")
        print("  python -m abstractllm.server")
        return

    # Test each model with both streaming and non-streaming
    for model in test_models:
        # Non-streaming test
        test_tool_call_conversion(model, stream=False)
        time.sleep(1)  # Small delay between tests

        # Streaming test
        test_tool_call_conversion(model, stream=True)
        time.sleep(1)  # Small delay between tests

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\n✅ If tool calls were detected and properly formatted,")
    print("   the universal conversion is working correctly.")
    print("\n❌ If raw tool call tags appear in content,")
    print("   the conversion may need adjustment.")


if __name__ == "__main__":
    main()
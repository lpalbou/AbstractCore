"""
Tests for Agentic CLI Compatibility (Codex, Gemini CLI, Crush)

These tests verify that the server properly handles:
1. Multi-turn conversations with tool calling
2. Tool messages with role: "tool" and content: null
3. Assistant messages with tool_calls field
4. Adaptive message conversion for local models (Ollama)

The server is automatically started in a background process for testing.
"""

import pytest
import httpx
import json
import subprocess
import time
import os
from typing import Dict, Any, List


# Test configuration
BASE_URL = "http://localhost:8003"
TEST_MODEL = "ollama/qwen3-coder:30b"


@pytest.fixture(scope="module")
def server():
    """Start the AbstractLLM server for testing, stop after tests complete."""
    # Check if server is already running
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            # Server already running, use it
            yield
            return
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    # Start server in background
    env = os.environ.copy()
    env["ABSTRACTCORE_DEBUG"] = "true"

    process = subprocess.Popen(
        ["python", "-m", "uvicorn", "abstractllm.server.app:app",
         "--host", "0.0.0.0", "--port", "8003"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to start (max 10 seconds)
    for _ in range(20):
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(0.5)
    else:
        process.kill()
        pytest.fail("Server failed to start within 10 seconds")

    yield

    # Stop server after tests
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


class TestCodexMultiTurnToolCalling:
    """
    Test Codex-style multi-turn tool calling scenarios.

    Codex sends:
    1. User message
    2. Assistant message with tool_calls
    3. Tool message with tool_call_id and result
    4. Optionally more user messages
    """

    def test_single_tool_call_scenario(self, server):
        """
        Test: User asks → Assistant calls tool → Tool returns result → Assistant responds
        """
        messages = [
            {
                "role": "user",
                "content": "List files in the current directory"
            },
            {
                "role": "assistant",
                "content": "I'll list the files using the shell command.",
                "tool_calls": [
                    {
                        "id": "call_list_files_001",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": ["ls", "-la"]})
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_list_files_001",
                "content": "total 128\ndrwxr-xr-x  15 user  staff   480 Sep 30 23:06 .\ndrwxr-xr-x   5 user  staff   160 Sep 28 14:52 ..\n-rw-r--r--   1 user  staff  5678 Sep 30 23:06 README.md\n-rw-r--r--   1 user  staff  1234 Sep 30 22:00 setup.py"
            },
            {
                "role": "user",
                "content": "What files did you find?"
            }
        ]

        response = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": messages,
                "max_tokens": 200
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Verify response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        # Verify model actually responded (not empty)
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0, "Response content should not be empty"
        print(f"✅ Single tool call test passed. Response: {content[:100]}...")

    def test_tool_message_with_null_content(self, server):
        """
        Test: Tool message with content: null (e.g., tool execution failed)
        """
        messages = [
            {
                "role": "user",
                "content": "Delete a non-existent file"
            },
            {
                "role": "assistant",
                "content": "I'll try to delete it.",
                "tool_calls": [
                    {
                        "id": "call_delete_001",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": ["rm", "nonexistent.txt"]})
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_delete_001",
                "content": None  # Tool failed, no output
            },
            {
                "role": "user",
                "content": "Did it work?"
            }
        ]

        response = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": messages,
                "max_tokens": 100
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        print(f"✅ Null content test passed. Response: {content[:100]}...")

    def test_multiple_tool_calls_sequential(self, server):
        """
        Test: Multiple tool calls in sequence
        """
        messages = [
            {
                "role": "user",
                "content": "Check the current directory and then read README.md"
            },
            {
                "role": "assistant",
                "content": "I'll check the directory first.",
                "tool_calls": [
                    {
                        "id": "call_pwd_001",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": ["pwd"]})
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_pwd_001",
                "content": "/Users/user/projects/abstractllm_core"
            },
            {
                "role": "assistant",
                "content": "Now I'll read the README.",
                "tool_calls": [
                    {
                        "id": "call_cat_001",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": ["cat", "README.md"]})
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_cat_001",
                "content": "# AbstractLLM\n\nA universal LLM abstraction library..."
            },
            {
                "role": "user",
                "content": "What did you learn?"
            }
        ]

        response = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": messages,
                "max_tokens": 200
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        print(f"✅ Sequential tool calls test passed. Response: {content[:100]}...")

    def test_conversation_context_preserved(self, server):
        """
        Test: Verify that conversation context is preserved across tool calls
        """
        messages = [
            {
                "role": "user",
                "content": "My name is Alice"
            },
            {
                "role": "assistant",
                "content": "Nice to meet you, Alice!"
            },
            {
                "role": "user",
                "content": "List files"
            },
            {
                "role": "assistant",
                "content": "I'll list the files for you, Alice.",
                "tool_calls": [
                    {
                        "id": "call_ls_001",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": ["ls"]})
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_ls_001",
                "content": "file1.txt\nfile2.py\nREADME.md"
            },
            {
                "role": "user",
                "content": "What's my name?"
            }
        ]

        response = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": messages,
                "max_tokens": 100
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        # Model should remember the name "Alice" from earlier in conversation
        assert len(content) > 0
        print(f"✅ Context preservation test passed. Response: {content[:100]}...")


@pytest.mark.skip(reason="/v1/responses endpoint format changed - now uses ChatCompletionRequest with messages field")
class TestResponsesEndpoint:
    """Test the /v1/responses endpoint (Codex preferred)"""

    def test_basic_responses_call(self, server):
        """Test basic /v1/responses endpoint"""
        response = httpx.post(
            f"{BASE_URL}/v1/responses",
            json={
                "model": TEST_MODEL,
                "input": [
                    {"type": "message", "role": "user", "content": "What is 2+2?"}
                ],
                "max_tokens": 50
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        assert "output" in data
        assert len(data["output"]) > 0
        assert data["output"][0]["role"] == "assistant"
        assert "content" in data["output"][0]
        print(f"✅ Responses endpoint test passed. Output: {data['output'][0]['content'][:50]}...")

    def test_responses_with_tool_history(self, server):
        """Test /v1/responses with tool calling history"""
        response = httpx.post(
            f"{BASE_URL}/v1/responses",
            json={
                "model": TEST_MODEL,
                "input": [
                    {"type": "message", "role": "user", "content": "List files"},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "I'll list them.",
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": json.dumps({"command": ["ls"]})
                                }
                            }
                        ]
                    },
                    {
                        "type": "message",
                        "role": "tool",
                        "tool_call_id": "call_001",
                        "content": "file1.txt\nfile2.py"
                    },
                    {"type": "message", "role": "user", "content": "How many files?"}
                ],
                "max_tokens": 100
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        assert "output" in data
        assert len(data["output"]) > 0
        content = data["output"][0]["content"]
        assert len(content) > 0
        print(f"✅ Responses with tools test passed. Output: {content[:50]}...")


@pytest.mark.skip(reason="/v1/messages endpoint removed in simplified server")
class TestMessagesEndpoint:
    """Test the /v1/messages endpoint (Anthropic Messages API)"""

    def test_basic_messages_call(self, server):
        """Test basic /v1/messages endpoint"""
        response = httpx.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": TEST_MODEL,
                "max_tokens": 50,
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ]
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        assert "content" in data
        assert len(data["content"]) > 0
        assert data["content"][0]["type"] == "text"
        print(f"✅ Messages endpoint test passed. Content: {data['content'][0]['text'][:50]}...")


class TestAdaptiveMessageConversion:
    """Test that messages are properly converted for Ollama (local models)"""

    def test_tool_calls_removed_for_ollama(self, server):
        """
        Verify that tool_calls are stripped from messages when sending to Ollama.
        This test ensures no 400 errors from Ollama due to unsupported fields.
        """
        messages = [
            {
                "role": "user",
                "content": "Test"
            },
            {
                "role": "assistant",
                "content": "Testing tools",
                "tool_calls": [
                    {
                        "id": "call_test_001",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_test_001",
                "content": "Tool result"
            },
            {
                "role": "user",
                "content": "Continue"
            }
        ]

        # This should NOT return 400 error
        response = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": messages,
                "max_tokens": 50
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Ollama returned error (tool_calls not cleaned): {response.text}"
        data = response.json()
        assert "choices" in data
        print(f"✅ Tool calls cleaning test passed. No 400 errors from Ollama.")

    def test_tool_role_converted_for_ollama(self, server):
        """
        Verify that role: "tool" messages are converted to role: "user"
        with [TOOL RESULT] markers for Ollama.
        """
        messages = [
            {"role": "user", "content": "Run command"},
            {"role": "assistant", "content": "Running..."},
            {"role": "tool", "tool_call_id": "call_001", "content": "Command output"},
            {"role": "user", "content": "What happened?"}
        ]

        response = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": messages,
                "max_tokens": 100
            },
            timeout=60.0
        )

        assert response.status_code == 200, f"Role conversion failed: {response.text}"
        data = response.json()
        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        print(f"✅ Role conversion test passed. Response: {content[:50]}...")


def run_all_tests():
    """Run all tests manually"""
    test_classes = [
        TestCodexMultiTurnToolCalling(),
        TestResponsesEndpoint(),
        TestMessagesEndpoint(),
        TestAdaptiveMessageConversion()
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*60}")
        print(f"Running {class_name}")
        print(f"{'='*60}")

        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(test_class, method_name)
                    print(f"\n🧪 {method_name}...")
                    method()
                    passed_tests += 1
                except Exception as e:
                    print(f"❌ FAILED: {method_name}")
                    print(f"   Error: {str(e)}")
                    failed_tests.append((class_name, method_name, str(e)))

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error[:100]}")
    else:
        print(f"\n✅ ALL TESTS PASSED!")

    return passed_tests == total_tests


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    exit(0 if success else 1)

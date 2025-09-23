"""
Test tool calling detection - verify providers can detect and format tool calls.
Focus on tool call detection, not execution (that's for agents).
"""

import pytest
import os
import json
from abstractllm import create_llm
from abstractllm.tools.common_tools import list_files, search_files, read_file, write_file, web_search


class TestProviderToolDetection:
    """Test tool call detection capabilities for each provider."""

    def test_openai_tool_detection(self):
        """Test OpenAI can detect and format tool calls."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            # Use enhanced list_files tool
            tools = [list_files]

            response = provider.generate(
                "Please list the files in the current directory",
                tools=tools
            )

            assert response is not None

            if response.has_tool_calls():
                # Tool calling worked - verify format
                assert len(response.tool_calls) > 0
                tool_call = response.tool_calls[0]

                assert tool_call.get('name') == 'list_files'
                assert 'arguments' in tool_call

                # Verify arguments are properly formatted
                args = tool_call.get('arguments', {})
                if isinstance(args, str):
                    args = json.loads(args)
                assert isinstance(args, dict)

            else:
                # OpenAI should support tools, but prompt might not trigger it
                pytest.skip("OpenAI didn't use tools (prompt might need adjustment)")

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_tool_detection(self):
        """Test Anthropic can detect and format tool calls."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            # Use enhanced list_files tool
            tools = [list_files]

            response = provider.generate(
                "Please list the files in the current directory",
                tools=tools
            )

            assert response is not None

            # Check for tool usage - Anthropic may use prompted format
            tool_used = False

            if response.has_tool_calls():
                # Native tool calling format
                assert len(response.tool_calls) > 0
                tool_call = response.tool_calls[0]
                assert tool_call.get('name') == 'list_files'
                assert 'arguments' in tool_call
                tool_used = True

            elif "<tool_call>" in response.content and "list_files" in response.content:
                # Prompted tool calling format (Anthropic style)
                # Tool was executed and results included in content
                assert "Tool Results:" in response.content or "files in" in response.content.lower()
                tool_used = True

            assert tool_used, f"Tool should have been used. Response: {response.content[:200]}..."

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_ollama_tool_detection(self):
        """Test Ollama tool call detection (limited by model capabilities)."""
        try:
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

            # Get a simple tool
            tools = [{
                "name": "get_time",
                "description": "Get the current time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }]

            response = provider.generate(
                "What time is it? Please use the get_time function.",
                tools=tools
            )

            assert response is not None

            # Ollama may or may not support tool calling depending on the model
            # This test mainly verifies the provider handles tool requests gracefully
            if response.has_tool_calls():
                # If tools work, verify format
                assert len(response.tool_calls) > 0
                tool_call = response.tool_calls[0]

                # Handle both dict and ToolCall object formats
                if hasattr(tool_call, 'name'):
                    assert tool_call.name is not None
                elif isinstance(tool_call, dict):
                    assert 'name' in tool_call
                else:
                    assert False, f"Unexpected tool_call format: {type(tool_call)}"
            else:
                # Many Ollama models don't support structured tool calling
                # This is expected and not a failure
                assert response.content is not None

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_tool_call_format_validation(self):
        """Test that tool call formats are valid across providers."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            # Tool with specific parameter requirements
            calculator_tool = {
                "name": "calculate",
                "description": "Perform basic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"]
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }

            response = provider.generate(
                "Calculate 15 + 27 using the calculate function",
                tools=[calculator_tool]
            )

            if response and response.has_tool_calls():
                tool_call = response.tool_calls[0]

                # Verify required fields
                assert tool_call.get('name') == 'calculate'
                assert 'arguments' in tool_call

                # Parse and validate arguments
                args = tool_call.get('arguments', {})
                if isinstance(args, str):
                    args = json.loads(args)

                # Check required parameters are present
                assert 'operation' in args
                assert 'a' in args
                assert 'b' in args

                # Validate types and values
                assert args['operation'] in ['add', 'subtract', 'multiply', 'divide']
                assert isinstance(args['a'], (int, float))
                assert isinstance(args['b'], (int, float))

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_multiple_tool_detection(self):
        """Test detection of multiple tools in one response."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            tools = [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                },
                {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ]

            response = provider.generate(
                "What's the weather in Paris and what time is it?",
                tools=tools
            )

            if response and response.has_tool_calls():
                # Should detect multiple tool calls
                tool_calls = response.tool_calls
                tool_names = [call.get('name') for call in tool_calls]

                # Verify we got tool calls (might be 1 or 2 depending on model behavior)
                assert len(tool_calls) > 0
                assert all(name in ['get_weather', 'get_time'] for name in tool_names)

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
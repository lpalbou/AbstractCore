"""
Test tool calling functionality across providers.
"""

import pytest
import os
import json
import time
from typing import Dict, Any, List
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


class TestToolCalling:
    """Test tool calling functionality across providers."""

    @pytest.fixture
    def tool_definitions(self):
        """Define tools in OpenAI format"""
        return [
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

    @pytest.fixture
    def test_cases(self):
        """Define test cases for tool calling"""
        return [
            ("What files are in the current directory?", "list_files"),
            ("What is 42 * 17?", "calculate"),
            ("What's the weather in New York?", "get_weather")
        ]

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool with given arguments"""
        if tool_name == 'list_files':
            return list_files(arguments.get('directory', '.'))
        elif tool_name == 'calculate':
            return calculate(arguments.get('expression', ''))
        elif tool_name == 'get_weather':
            return get_weather(arguments.get('city', ''))
        else:
            return "Unknown tool"

    def test_openai_tool_calling(self, tool_definitions, test_cases):
        """Test OpenAI provider tool calling with gpt-4o-mini."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            llm = create_llm("openai", model="gpt-4o-mini")

            results = []
            for prompt, expected_tool in test_cases:
                response = llm.generate(prompt, tools=tool_definitions)

                assert response is not None

                if response.has_tool_calls():
                    # Tool calling worked
                    results.append(True)

                    # Verify tool execution
                    for tc in response.tool_calls:
                        tool_name = tc.get('name')
                        args = json.loads(tc.get('arguments')) if isinstance(tc.get('arguments'), str) else tc.get('arguments', {})
                        result = self.execute_tool(tool_name, args)
                        assert len(result) > 0  # Should get some result
                else:
                    # No tool calls, but might still be valid response
                    results.append(False)

            # OpenAI should have good tool calling success rate
            success_rate = sum(results) / len(results) if results else 0
            assert success_rate > 0.5, f"OpenAI tool calling success rate too low: {success_rate:.1%}"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_tool_calling(self, tool_definitions, test_cases):
        """Test Anthropic provider tool calling with claude-3-5-haiku-20241022."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            llm = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            results = []
            for prompt, expected_tool in test_cases:
                response = llm.generate(prompt, tools=tool_definitions)

                assert response is not None

                if response.has_tool_calls():
                    # Tool calling worked
                    results.append(True)

                    # Verify tool execution
                    for tc in response.tool_calls:
                        tool_name = tc.get('name')
                        args = json.loads(tc.get('arguments')) if isinstance(tc.get('arguments'), str) else tc.get('arguments', {})
                        result = self.execute_tool(tool_name, args)
                        assert len(result) > 0  # Should get some result
                else:
                    # No tool calls, but might still be valid response
                    results.append(False)

            # Anthropic should have good tool calling success rate
            success_rate = sum(results) / len(results) if results else 0
            assert success_rate > 0.5, f"Anthropic tool calling success rate too low: {success_rate:.1%}"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_ollama_basic_generation(self):
        """Test Ollama basic generation (tool calling may not work with qwen3:4b)."""
        try:
            llm = create_llm("ollama", model="qwen3:4b", base_url="http://localhost:11434")

            # Test basic generation (without tools first)
            response = llm.generate("What is 2+2?")
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_ollama_tool_format(self):
        """Test Ollama with architecture-specific tool format"""
        try:
            llm = create_llm("ollama", model="qwen3:4b", base_url="http://localhost:11434")

            # Qwen3 uses special tool format
            prompt = """<|tool_call|>
list_files
{"directory": "/tmp"}
<|tool_call_end|>

Please list the files in /tmp directory."""

            response = llm.generate(prompt)
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_lmstudio_basic_generation(self):
        """Test LMStudio basic generation (tool calling may vary)."""
        try:
            llm = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")

            # Test basic generation
            response = llm.generate("What is the capital of France?")
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_tool_definition_creation(self):
        """Test that ToolDefinition can be created from functions"""
        # Test creating tool definition from function
        tool_def = ToolDefinition.from_function(list_files)

        assert tool_def.name == "list_files"
        assert "List files" in tool_def.description
        assert "parameters" in tool_def.to_dict()

    def test_tool_execution_safety(self):
        """Test that tool execution handles errors gracefully"""
        # Test with invalid directory
        result = list_files("/nonexistent/directory/path")
        assert "Error" in result

        # Test with invalid math expression
        result = calculate("invalid_expression")
        assert "Error" in result

        # Test with unknown city
        result = get_weather("UnknownCityXYZ")
        assert "not available" in result


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
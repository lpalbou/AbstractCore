"""
Test for OpenAI format conversion bug - arguments must be JSON string, not JSON object.

This test validates that the _convert_to_openai_format() method properly encodes
the arguments field as a JSON string (with escaped quotes) rather than a JSON object.
"""

import json
import pytest
from abstractllm.providers.streaming import UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse


def test_openai_format_arguments_as_json_string():
    """
    Test that OpenAI format conversion produces arguments as a properly escaped JSON string.

    Per Codex documentation:
    - Arguments MUST be a JSON string: '{"command": ["ls", "-la"]}'
    - NOT a JSON object: {"command": ["ls", "-la"]}

    The difference is critical:
    - CORRECT: "arguments": "{\"command\": [\"ls\", \"-la\"]}"
    - WRONG: "arguments": "{"command": ["ls", "-la"]}"  (malformed, missing escapes)
    """
    # Create processor with OpenAI format target
    processor = UnifiedStreamProcessor(
        model_name="qwen3-coder:30b",
        execute_tools=False,
        tool_call_tags="openai"  # This should trigger OpenAI JSON conversion
    )

    # Simulate a tool call in Qwen3 format (text-based)
    qwen_tool_call = '<|tool_call|>{"name": "shell", "arguments": {"command": ["ls", "-la"], "workdir": "/tmp"}}</|tool_call|>'

    # Create a mock stream with the tool call
    def mock_stream():
        yield GenerateResponse(
            content=qwen_tool_call,
            model="qwen3-coder:30b",
            finish_reason=None
        )

    # Process the stream
    results = list(processor.process_stream(mock_stream()))

    # Should have at least one result
    assert len(results) > 0, "Expected at least one result from stream processing"

    # Get the converted content
    converted_content = results[0].content
    print(f"\nConverted content:\n{converted_content}\n")

    # Parse the result - should be valid JSON
    try:
        openai_tool_call = json.loads(converted_content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Converted content is not valid JSON: {e}\nContent: {converted_content}")

    # Validate structure
    assert "id" in openai_tool_call, "Missing 'id' field"
    assert "type" in openai_tool_call, "Missing 'type' field"
    assert openai_tool_call["type"] == "function", f"Expected type='function', got {openai_tool_call['type']}"
    assert "function" in openai_tool_call, "Missing 'function' field"

    function_obj = openai_tool_call["function"]
    assert "name" in function_obj, "Missing 'name' in function object"
    assert function_obj["name"] == "shell", f"Expected name='shell', got {function_obj['name']}"
    assert "arguments" in function_obj, "Missing 'arguments' in function object"

    # CRITICAL TEST: Arguments must be a STRING, not an object
    arguments_value = function_obj["arguments"]
    assert isinstance(arguments_value, str), (
        f"Arguments must be a JSON STRING, not {type(arguments_value).__name__}. "
        f"Got: {repr(arguments_value)}"
    )

    # The string should be properly escaped JSON
    try:
        parsed_arguments = json.loads(arguments_value)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"Arguments field is not a valid JSON string: {e}\n"
            f"Arguments value: {repr(arguments_value)}"
        )

    # Validate the parsed arguments content
    assert "command" in parsed_arguments, "Missing 'command' in parsed arguments"
    assert parsed_arguments["command"] == ["ls", "-la"], f"Wrong command value: {parsed_arguments['command']}"
    assert "workdir" in parsed_arguments, "Missing 'workdir' in parsed arguments"
    assert parsed_arguments["workdir"] == "/tmp", f"Wrong workdir value: {parsed_arguments['workdir']}"

    # Validate the raw JSON format (with proper escaping)
    # The arguments field should look like: "arguments": "{\"command\":[\"ls\",\"-la\"],\"workdir\":\"/tmp\"}"
    # NOT like: "arguments": "{"command":["ls","-la"],"workdir":"/tmp"}"
    assert '\\"' in converted_content or '\\\\"' in converted_content, (
        "Arguments field does not contain escaped quotes - it's likely malformed. "
        f"Content: {converted_content}"
    )

    print("✅ OpenAI format validation PASSED!")
    print(f"   - Arguments field is a properly escaped JSON string")
    print(f"   - Raw value: {repr(arguments_value)}")
    print(f"   - Parsed successfully: {parsed_arguments}")


def test_openai_format_llama_input():
    """Test OpenAI format conversion from LLaMA format."""
    processor = UnifiedStreamProcessor(
        model_name="llama3:8b",
        execute_tools=False,
        tool_call_tags="openai"
    )

    llama_tool_call = '<function_call>{"name": "get_weather", "arguments": {"location": "San Francisco", "units": "celsius"}}</function_call>'

    def mock_stream():
        yield GenerateResponse(content=llama_tool_call, model="llama3:8b", finish_reason=None)

    results = list(processor.process_stream(mock_stream()))
    assert len(results) > 0

    converted_content = results[0].content
    openai_tool_call = json.loads(converted_content)

    # Validate arguments is a JSON string
    arguments_value = openai_tool_call["function"]["arguments"]
    assert isinstance(arguments_value, str), f"Arguments must be a string, got {type(arguments_value)}"

    # Parse and validate
    parsed_args = json.loads(arguments_value)
    assert parsed_args["location"] == "San Francisco"
    assert parsed_args["units"] == "celsius"

    print("✅ LLaMA format conversion PASSED!")


def test_openai_format_xml_input():
    """Test OpenAI format conversion from XML format."""
    processor = UnifiedStreamProcessor(
        model_name="gemini-pro",
        execute_tools=False,
        tool_call_tags="openai"
    )

    xml_tool_call = '<tool_call>{"name": "search", "arguments": {"query": "Python tutorials", "max_results": 5}}</tool_call>'

    def mock_stream():
        yield GenerateResponse(content=xml_tool_call, model="gemini-pro", finish_reason=None)

    results = list(processor.process_stream(mock_stream()))
    assert len(results) > 0

    converted_content = results[0].content
    openai_tool_call = json.loads(converted_content)

    # Validate arguments is a JSON string
    arguments_value = openai_tool_call["function"]["arguments"]
    assert isinstance(arguments_value, str), f"Arguments must be a string, got {type(arguments_value)}"

    # Parse and validate
    parsed_args = json.loads(arguments_value)
    assert parsed_args["query"] == "Python tutorials"
    assert parsed_args["max_results"] == 5

    print("✅ XML format conversion PASSED!")


def test_openai_format_with_complex_arguments():
    """Test OpenAI format with complex nested arguments."""
    processor = UnifiedStreamProcessor(
        model_name="qwen3-coder:30b",
        execute_tools=False,
        tool_call_tags="openai"
    )

    # Complex nested structure
    complex_tool_call = '''<|tool_call|>{
        "name": "execute_code",
        "arguments": {
            "language": "python",
            "code": "print('hello')",
            "options": {
                "timeout": 30,
                "env": {"PATH": "/usr/bin", "HOME": "/home/user"}
            },
            "files": ["main.py", "test.py"]
        }
    }</|tool_call|>'''

    def mock_stream():
        yield GenerateResponse(content=complex_tool_call, model="qwen3-coder:30b", finish_reason=None)

    results = list(processor.process_stream(mock_stream()))
    assert len(results) > 0

    converted_content = results[0].content
    openai_tool_call = json.loads(converted_content)

    # Validate arguments is a JSON string
    arguments_value = openai_tool_call["function"]["arguments"]
    assert isinstance(arguments_value, str), f"Arguments must be a string, got {type(arguments_value)}"

    # Parse complex structure
    parsed_args = json.loads(arguments_value)
    assert parsed_args["language"] == "python"
    assert parsed_args["code"] == "print('hello')"
    assert parsed_args["options"]["timeout"] == 30
    assert parsed_args["options"]["env"]["PATH"] == "/usr/bin"
    assert parsed_args["files"] == ["main.py", "test.py"]

    print("✅ Complex arguments conversion PASSED!")


def test_openai_format_empty_arguments():
    """Test OpenAI format with empty arguments."""
    processor = UnifiedStreamProcessor(
        model_name="qwen3-coder:30b",
        execute_tools=False,
        tool_call_tags="openai"
    )

    empty_args_tool_call = '<|tool_call|>{"name": "ping", "arguments": {}}</|tool_call|>'

    def mock_stream():
        yield GenerateResponse(content=empty_args_tool_call, model="qwen3-coder:30b", finish_reason=None)

    results = list(processor.process_stream(mock_stream()))
    assert len(results) > 0

    converted_content = results[0].content
    openai_tool_call = json.loads(converted_content)

    # Validate arguments is a JSON string (even if empty)
    arguments_value = openai_tool_call["function"]["arguments"]
    assert isinstance(arguments_value, str), f"Arguments must be a string, got {type(arguments_value)}"

    # Should be "{}" as a string
    assert arguments_value == "{}" or arguments_value == ""

    print("✅ Empty arguments conversion PASSED!")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing OpenAI Format Arguments Encoding Bug")
    print("=" * 80)

    test_openai_format_arguments_as_json_string()
    test_openai_format_llama_input()
    test_openai_format_xml_input()
    test_openai_format_with_complex_arguments()
    test_openai_format_empty_arguments()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - OpenAI format is correct!")
    print("=" * 80)

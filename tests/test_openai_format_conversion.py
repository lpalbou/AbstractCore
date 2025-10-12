"""
Comprehensive tests for OpenAI format conversion.

This test suite validates that when `tool_call_tags="openai"` is set,
the system correctly converts text-based tool calls (Qwen3, LLaMA, XML, Gemma)
to OpenAI's structured JSON format.
"""

import pytest
import json
from abstractllm.providers.streaming import UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse


class TestOpenAIFormatConversion:
    """Test OpenAI format conversion from text-based tool calls."""

    def test_openai_format_initialization(self):
        """Test that openai format sets convert_to_openai_json flag."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Should enable OpenAI JSON conversion mode
        assert processor.convert_to_openai_json is True
        # Should NOT have text tag rewriter
        assert processor.tag_rewriter is None

    def test_qwen3_to_openai_conversion(self):
        """Test conversion from Qwen3 format to OpenAI JSON."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Qwen3 format
        qwen_input = '<|tool_call|>{"name": "shell", "arguments": {"command": ["ls", "-la"]}}</|tool_call|>'

        # Convert
        result = processor._convert_to_openai_format(qwen_input)

        # Parse result as JSON
        result_json = json.loads(result)

        # Validate OpenAI format
        assert result_json["type"] == "function"
        assert result_json["function"]["name"] == "shell"
        assert "id" in result_json
        assert result_json["id"].startswith("call_")
        assert isinstance(result_json["function"]["arguments"], str)

        # Validate arguments
        args = json.loads(result_json["function"]["arguments"])
        assert args["command"] == ["ls", "-la"]

    def test_llama_to_openai_conversion(self):
        """Test conversion from LLaMA format to OpenAI JSON."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: LLaMA format
        llama_input = '<function_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</function_call>'

        # Convert
        result = processor._convert_to_openai_format(llama_input)

        # Parse result as JSON
        result_json = json.loads(result)

        # Validate OpenAI format
        assert result_json["type"] == "function"
        assert result_json["function"]["name"] == "get_weather"
        assert "id" in result_json

        # Validate arguments
        args = json.loads(result_json["function"]["arguments"])
        assert args["location"] == "San Francisco"

    def test_xml_to_openai_conversion(self):
        """Test conversion from XML format to OpenAI JSON."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: XML format
        xml_input = '<tool_call>{"name": "read_file", "arguments": {"path": "/etc/hosts"}}</tool_call>'

        # Convert
        result = processor._convert_to_openai_format(xml_input)

        # Parse result as JSON
        result_json = json.loads(result)

        # Validate OpenAI format
        assert result_json["type"] == "function"
        assert result_json["function"]["name"] == "read_file"

        # Validate arguments
        args = json.loads(result_json["function"]["arguments"])
        assert args["path"] == "/etc/hosts"

    def test_gemma_to_openai_conversion(self):
        """Test conversion from Gemma format to OpenAI JSON."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Gemma format
        gemma_input = '```tool_code\n{"name": "calculate", "arguments": {"expression": "2+2"}}\n```'

        # Convert
        result = processor._convert_to_openai_format(gemma_input)

        # Parse result as JSON
        result_json = json.loads(result)

        # Validate OpenAI format
        assert result_json["type"] == "function"
        assert result_json["function"]["name"] == "calculate"

        # Validate arguments
        args = json.loads(result_json["function"]["arguments"])
        assert args["expression"] == "2+2"

    def test_multiple_tool_calls_conversion(self):
        """Test conversion of multiple tool calls in sequence."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Multiple Qwen3 tool calls
        input_text = (
            'First, I will list files: <|tool_call|>{"name": "shell", "arguments": {"command": ["ls"]}}</|tool_call|> '
            'Then check date: <|tool_call|>{"name": "shell", "arguments": {"command": ["date"]}}</|tool_call|>'
        )

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Should contain two OpenAI-formatted tool calls
        assert result.count('"type": "function"') == 2
        assert result.count('"name": "shell"') == 2
        assert 'call_' in result  # Has generated IDs

    def test_mixed_content_with_tool_calls(self):
        """Test conversion with mixed text and tool calls."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Text mixed with tool calls
        input_text = (
            'I will help you with that. '
            '<|tool_call|>{"name": "shell", "arguments": {"command": ["pwd"]}}</|tool_call|> '
            'Let me check the current directory.'
        )

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Should preserve surrounding text
        assert 'I will help you with that.' in result
        assert 'Let me check the current directory.' in result

        # Should convert tool call
        assert '"type": "function"' in result
        assert '"name": "shell"' in result

    def test_no_tool_calls_passthrough(self):
        """Test that content without tool calls passes through unchanged."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Plain text with no tool calls
        input_text = 'This is just regular text with no tool calls.'

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Should be unchanged
        assert result == input_text

    def test_malformed_tool_call_handling(self):
        """Test graceful handling of malformed tool calls."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Malformed JSON in tool call
        input_text = '<|tool_call|>{"name": "shell", invalid json}</|tool_call|>'

        # Convert - should handle error gracefully
        result = processor._convert_to_openai_format(input_text)

        # Original content should be preserved (no crash)
        assert '<|tool_call|>' in result

    def test_tool_call_without_arguments(self):
        """Test tool call with no arguments field."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Tool call without arguments
        input_text = '<|tool_call|>{"name": "get_current_time"}</|tool_call|>'

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Parse result
        result_json = json.loads(result)

        # Should have empty arguments
        assert result_json["function"]["name"] == "get_current_time"
        args = json.loads(result_json["function"]["arguments"])
        assert args == {}

    def test_complex_nested_arguments(self):
        """Test tool call with complex nested arguments."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Complex nested structure
        input_text = '''<|tool_call|>{
            "name": "database_query",
            "arguments": {
                "query": "SELECT * FROM users",
                "options": {
                    "limit": 10,
                    "offset": 0,
                    "order_by": ["created_at", "desc"]
                }
            }
        }</|tool_call|>'''

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Parse result
        result_json = json.loads(result)

        # Validate structure preservation
        args = json.loads(result_json["function"]["arguments"])
        assert args["query"] == "SELECT * FROM users"
        assert args["options"]["limit"] == 10
        assert args["options"]["order_by"] == ["created_at", "desc"]

    def test_streaming_integration(self):
        """Test OpenAI conversion in streaming context."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Simulate streaming chunks
        chunks = [
            GenerateResponse(content="I will check files for you. ", model="test-model"),
            GenerateResponse(
                content='<|tool_call|>{"name": "shell", "arguments": {"command": ["ls", "-la"]}}</|tool_call|>',
                model="test-model"
            ),
            GenerateResponse(content=" Done!", model="test-model")
        ]

        # Process stream
        results = list(processor.process_stream(iter(chunks)))

        # Collect all content
        full_content = "".join([r.content for r in results if r.content])

        # Should contain converted OpenAI format
        assert '"type": "function"' in full_content
        assert '"name": "shell"' in full_content
        assert 'call_' in full_content  # Has generated ID

    def test_openai_format_vs_text_rewriting(self):
        """Test that openai format does JSON conversion, not text rewriting."""
        openai_processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        llama_processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="llama3"
        )

        # Input: Qwen3 format
        input_text = '<|tool_call|>{"name": "shell", "arguments": {"command": ["ls"]}}</|tool_call|>'

        # OpenAI conversion
        openai_result = openai_processor._convert_to_openai_format(input_text)

        # LLaMA text rewriting
        llama_result = llama_processor._apply_tag_rewriting_direct(input_text)

        # OpenAI should produce JSON structure
        assert '"type": "function"' in openai_result
        assert '"id":' in openai_result

        # LLaMA should produce text rewriting
        assert '<function_call>' in llama_result
        assert '</function_call>' in llama_result
        assert '"type": "function"' not in llama_result

    def test_unique_call_ids(self):
        """Test that each tool call gets a unique ID."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Multiple identical tool calls
        input_text = (
            '<|tool_call|>{"name": "shell", "arguments": {"command": ["ls"]}}</|tool_call|> '
            '<|tool_call|>{"name": "shell", "arguments": {"command": ["ls"]}}</|tool_call|>'
        )

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Extract all IDs
        import re
        id_pattern = r'"id":\s*"(call_[a-f0-9]+)"'
        ids = re.findall(id_pattern, result)

        # Should have 2 unique IDs
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_whitespace_handling(self):
        """Test that whitespace in tool calls is handled correctly."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Tool call with extra whitespace
        input_text = '''<|tool_call|>
        {
            "name": "shell",
            "arguments": {
                "command": ["ls"]
            }
        }
        </|tool_call|>'''

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Should parse successfully
        result_json = json.loads(result)
        assert result_json["function"]["name"] == "shell"


class TestOpenAIFormatEdgeCases:
    """Test edge cases and error scenarios for OpenAI format conversion."""

    def test_empty_content(self):
        """Test handling of empty content."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        result = processor._convert_to_openai_format("")
        assert result == ""

        result = processor._convert_to_openai_format(None)
        assert result is None

    def test_tool_call_without_name(self):
        """Test tool call missing required 'name' field."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Tool call without name
        input_text = '<|tool_call|>{"arguments": {"command": ["ls"]}}</|tool_call|>'

        # Should handle gracefully
        result = processor._convert_to_openai_format(input_text)

        # Original content should be preserved
        assert '<|tool_call|>' in result

    def test_non_json_content_in_tags(self):
        """Test non-JSON content inside tool call tags."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Non-JSON content
        input_text = '<|tool_call|>This is not JSON!</|tool_call|>'

        # Should handle gracefully
        result = processor._convert_to_openai_format(input_text)

        # Original content should be preserved
        assert 'This is not JSON!' in result

    def test_partial_tool_call_tags(self):
        """Test incomplete tool call tags."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Incomplete tags
        input_text = '<|tool_call|>{"name": "shell"}'  # Missing closing tag

        # Should handle gracefully (no match, no conversion)
        result = processor._convert_to_openai_format(input_text)
        assert result == input_text

    def test_unicode_in_tool_calls(self):
        """Test Unicode characters in tool call arguments."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Input: Unicode characters
        input_text = '<|tool_call|>{"name": "search", "arguments": {"query": "こんにちは 世界"}}</|tool_call|>'

        # Convert
        result = processor._convert_to_openai_format(input_text)

        # Parse result
        result_json = json.loads(result)
        args = json.loads(result_json["function"]["arguments"])

        # Unicode should be preserved
        assert args["query"] == "こんにちは 世界"


class TestOpenAIFormatBackwardCompatibility:
    """Test backward compatibility with existing tests."""

    def test_environment_variable_compatibility(self):
        """Test compatibility with environment variable test expectations."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # From test_environment_variable_tool_call_tags.py
        # Should NOT have tag rewriter
        assert processor.tag_rewriter is None

        # Should have OpenAI conversion enabled
        assert processor.convert_to_openai_json is True

    def test_non_openai_formats_unchanged(self):
        """Test that non-openai formats still use text rewriting."""
        for format_name in ["qwen3", "llama3", "xml", "gemma"]:
            processor = UnifiedStreamProcessor(
                model_name="test-model",
                tool_call_tags=format_name
            )

            # Should have tag rewriter
            assert processor.tag_rewriter is not None

            # Should NOT have OpenAI conversion
            assert processor.convert_to_openai_json is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

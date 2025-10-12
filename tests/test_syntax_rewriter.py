"""
Comprehensive Test Suite for Tool Call Syntax Rewriter

Tests all format conversions, auto-detection, and edge cases for the enhanced
syntax rewriter that supports multiple agent formats.
"""

import pytest
import json
import uuid
from unittest.mock import patch

from abstractllm.tools.syntax_rewriter import (
    ToolCallSyntaxRewriter,
    SyntaxFormat,
    CustomFormatConfig,
    auto_detect_format,
    create_openai_rewriter,
    create_codex_rewriter,
    create_passthrough_rewriter,
    create_custom_rewriter
)
from abstractllm.tools.core import ToolCall


class TestSyntaxRewriter:
    """Test cases for the ToolCallSyntaxRewriter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_tool_call = ToolCall(
            name="list_files",
            arguments={"directory": "/tmp", "include_hidden": True},
            call_id="call_123"
        )

        self.sample_content_qwen = """
I'll help you list the files.

<|tool_call|>
{"name": "list_files", "arguments": {"directory": "/tmp", "include_hidden": true}}
</|tool_call|>

Let me check that directory for you.
"""

        self.sample_content_llama = """
I'll list the files for you.

<function_call>
{"name": "list_files", "arguments": {"directory": "/tmp", "include_hidden": true}}
</function_call>

Here are the results.
"""

        self.sample_content_xml = """
Let me list those files.

<tool_call>
{"name": "list_files", "arguments": {"directory": "/tmp", "include_hidden": true}}
</tool_call>

Done!
"""

    def test_passthrough_rewriter(self):
        """Test passthrough mode returns content unchanged."""
        rewriter = create_passthrough_rewriter()

        result = rewriter.rewrite_content(self.sample_content_qwen)
        assert result == self.sample_content_qwen

        result = rewriter.rewrite_content(self.sample_content_llama)
        assert result == self.sample_content_llama

    def test_openai_format_conversion(self):
        """Test conversion to OpenAI format."""
        rewriter = create_openai_rewriter("test-model")

        # Test OpenAI tool call format conversion
        openai_tools = rewriter.convert_to_openai_format([self.sample_tool_call])

        assert len(openai_tools) == 1
        tool = openai_tools[0]

        assert tool["id"] == "call_123"
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "list_files"

        # Arguments should be JSON string
        args = json.loads(tool["function"]["arguments"])
        assert args["directory"] == "/tmp"
        assert args["include_hidden"] is True

    def test_openai_format_auto_id_generation(self):
        """Test automatic ID generation for OpenAI format."""
        tool_call_no_id = ToolCall(
            name="test_tool",
            arguments={"param": "value"}
        )

        rewriter = create_openai_rewriter("test-model")
        openai_tools = rewriter.convert_to_openai_format([tool_call_no_id])

        assert len(openai_tools) == 1
        assert openai_tools[0]["id"].startswith("call_")
        assert len(openai_tools[0]["id"]) == 13  # "call_" + 8 chars

    def test_qwen3_format_conversion(self):
        """Test conversion to Qwen3 format."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.QWEN3, model_name="qwen3-coder")

        # Mock parse_tool_calls to return our sample tool call
        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [self.sample_tool_call]

            result = rewriter.rewrite_content(self.sample_content_llama)

            # Should contain Qwen3 format
            assert "<|tool_call|>" in result
            assert "</|tool_call|>" in result
            assert '"name": "list_files"' in result

            # Should not contain original LLaMA format
            assert "<function_call>" not in result
            assert "</function_call>" not in result

    def test_llama3_format_conversion(self):
        """Test conversion to LLaMA3 format."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.LLAMA3, model_name="llama3")

        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [self.sample_tool_call]

            result = rewriter.rewrite_content(self.sample_content_qwen)

            # Should contain LLaMA3 format
            assert "<function_call>" in result
            assert "</function_call>" in result
            assert '"name": "list_files"' in result

            # Should not contain original Qwen format
            assert "<|tool_call|>" not in result
            assert "</|tool_call|>" not in result

    def test_gemma_format_conversion(self):
        """Test conversion to Gemma format."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.GEMMA, model_name="gemma")

        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [self.sample_tool_call]

            result = rewriter.rewrite_content(self.sample_content_qwen)

            # Should contain Gemma format
            assert "```tool_code" in result
            assert "```" in result.split("```tool_code")[1]
            assert '"name": "list_files"' in result

    def test_xml_format_conversion(self):
        """Test conversion to XML format."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.XML, model_name="claude")

        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [self.sample_tool_call]

            result = rewriter.rewrite_content(self.sample_content_qwen)

            # Should contain XML format
            assert "<tool_call>" in result
            assert "</tool_call>" in result
            assert '"name": "list_files"' in result

    def test_custom_format_basic(self):
        """Test basic custom format conversion."""
        config = CustomFormatConfig(
            start_tag="[TOOL]",
            end_tag="[/TOOL]",
            json_wrapper=True
        )
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.CUSTOM, custom_config=config)

        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [self.sample_tool_call]

            result = rewriter.rewrite_content(self.sample_content_qwen)

            # Should contain custom format
            assert "[TOOL]" in result
            assert "[/TOOL]" in result
            assert '"name": "list_files"' in result

    def test_custom_format_with_template(self):
        """Test custom format with template."""
        config = CustomFormatConfig(
            start_tag="",
            end_tag="",
            format_template="CALL: {name} WITH {arguments} ID: {call_id}"
        )
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.CUSTOM, custom_config=config)

        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [self.sample_tool_call]

            result = rewriter.rewrite_content(self.sample_content_qwen)

            # Should contain custom template format
            assert "CALL: list_files WITH" in result
            assert "ID: call_123" in result

    def test_custom_format_requires_config(self):
        """Test that custom format requires configuration."""
        with pytest.raises(ValueError, match="Custom format requires CustomFormatConfig"):
            ToolCallSyntaxRewriter(SyntaxFormat.CUSTOM)

    def test_multiple_tool_calls(self):
        """Test handling multiple tool calls in content."""
        tool_call_1 = ToolCall(name="tool1", arguments={"param": "value1"}, call_id="call_1")
        tool_call_2 = ToolCall(name="tool2", arguments={"param": "value2"}, call_id="call_2")

        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.QWEN3)

        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [tool_call_1, tool_call_2]

            result = rewriter.rewrite_content("Some content with tools")

            # Should contain both tool calls
            assert '"name": "tool1"' in result
            assert '"name": "tool2"' in result
            assert result.count("<|tool_call|>") == 2
            assert result.count("</|tool_call|>") == 2

    def test_empty_content_handling(self):
        """Test handling of empty or None content."""
        rewriter = create_openai_rewriter()

        assert rewriter.rewrite_content("") == ""
        assert rewriter.rewrite_content(None) == None
        assert rewriter.rewrite_content("   ") == "   "

    def test_content_without_tool_calls(self):
        """Test content without any tool calls."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.QWEN3)
        content = "This is just regular text without any tool calls."

        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = []

            result = rewriter.rewrite_content(content)
            assert result == content

    def test_arguments_as_string_handling(self):
        """Test handling of arguments that are already strings."""
        tool_call_str_args = ToolCall(
            name="test_tool",
            arguments='{"already": "a_string"}',
            call_id="call_456"
        )

        rewriter = create_openai_rewriter()
        openai_tools = rewriter.convert_to_openai_format([tool_call_str_args])

        # Should validate and use the string as-is if it's valid JSON
        args_str = openai_tools[0]["function"]["arguments"]
        assert args_str == '{"already": "a_string"}'

        # Should be parseable as JSON
        parsed = json.loads(args_str)
        assert parsed["already"] == "a_string"

    def test_invalid_json_arguments_handling(self):
        """Test handling of invalid JSON in arguments."""
        tool_call_invalid = ToolCall(
            name="test_tool",
            arguments="not valid json",
            call_id="call_789"
        )

        rewriter = create_openai_rewriter()
        openai_tools = rewriter.convert_to_openai_format([tool_call_invalid])

        # Should wrap invalid JSON in a container
        args_str = openai_tools[0]["function"]["arguments"]
        parsed = json.loads(args_str)
        assert parsed["value"] == "not valid json"


class TestAutoDetection:
    """Test cases for auto-detection functionality."""

    def test_auto_detect_from_user_agent(self):
        """Test auto-detection from User-Agent header."""
        # Codex detection
        format = auto_detect_format("some-model", user_agent="Codex CLI v1.0")
        assert format == SyntaxFormat.CODEX

        # Default case
        format = auto_detect_format("some-model", user_agent="Regular Browser")
        assert format == SyntaxFormat.OPENAI

    def test_auto_detect_from_model_name(self):
        """Test auto-detection from model name patterns."""
        # OpenAI models should use passthrough
        assert auto_detect_format("openai/gpt-4o-mini") == SyntaxFormat.PASSTHROUGH
        assert auto_detect_format("gpt-3.5-turbo") == SyntaxFormat.PASSTHROUGH

        # Qwen models
        assert auto_detect_format("qwen3-coder:30b") == SyntaxFormat.QWEN3
        assert auto_detect_format("ollama/qwen3-next-80b") == SyntaxFormat.QWEN3

        # LLaMA models
        assert auto_detect_format("llama3:8b") == SyntaxFormat.LLAMA3
        assert auto_detect_format("llama-2-chat") == SyntaxFormat.LLAMA3

        # Gemma models
        assert auto_detect_format("gemma:7b") == SyntaxFormat.GEMMA

        # Claude models
        assert auto_detect_format("claude-3-5-haiku-latest") == SyntaxFormat.XML

        # Unknown model defaults to OpenAI
        assert auto_detect_format("unknown-model") == SyntaxFormat.OPENAI

    def test_auto_detect_from_custom_headers(self):
        """Test auto-detection from custom headers."""
        headers = {"X-Agent-Type": "codex"}
        format = auto_detect_format("some-model", custom_headers=headers)
        assert format == SyntaxFormat.CODEX

        headers = {"X-Agent-Type": "qwen3"}
        format = auto_detect_format("some-model", custom_headers=headers)
        assert format == SyntaxFormat.QWEN3

    def test_auto_detect_priority(self):
        """Test that custom headers have priority over other detection."""
        # Custom header should override model name
        headers = {"X-Agent-Type": "codex"}
        format = auto_detect_format("qwen3-coder:30b", custom_headers=headers)
        assert format == SyntaxFormat.CODEX  # Not QWEN3


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_create_openai_rewriter(self):
        """Test OpenAI rewriter creation."""
        rewriter = create_openai_rewriter("gpt-4")
        assert rewriter.target_format == SyntaxFormat.OPENAI
        assert rewriter.model_name == "gpt-4"

    def test_create_codex_rewriter(self):
        """Test Codex rewriter creation."""
        rewriter = create_codex_rewriter("gpt-4")
        assert rewriter.target_format == SyntaxFormat.CODEX
        assert rewriter.model_name == "gpt-4"

    def test_create_passthrough_rewriter(self):
        """Test passthrough rewriter creation."""
        rewriter = create_passthrough_rewriter()
        assert rewriter.target_format == SyntaxFormat.PASSTHROUGH

    def test_create_custom_rewriter(self):
        """Test custom rewriter creation."""
        rewriter = create_custom_rewriter(
            start_tag="<custom>",
            end_tag="</custom>",
            json_wrapper=True,
            model_name="test-model"
        )
        assert rewriter.target_format == SyntaxFormat.CUSTOM
        assert rewriter.custom_config.start_tag == "<custom>"
        assert rewriter.custom_config.end_tag == "</custom>"
        assert rewriter.custom_config.json_wrapper is True


class TestErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_invalid_format_string(self):
        """Test handling of invalid format strings."""
        with pytest.raises(ValueError, match="Unsupported format"):
            ToolCallSyntaxRewriter("invalid_format")

    def test_valid_format_enum(self):
        """Test using SyntaxFormat enum directly."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.QWEN3)
        assert rewriter.target_format == SyntaxFormat.QWEN3

    def test_openai_format_with_complex_arguments(self):
        """Test OpenAI format with complex nested arguments."""
        complex_tool_call = ToolCall(
            name="complex_tool",
            arguments={
                "nested": {
                    "array": [1, 2, 3],
                    "object": {"key": "value"}
                },
                "simple": "string"
            },
            call_id="call_complex"
        )

        rewriter = create_openai_rewriter()
        openai_tools = rewriter.convert_to_openai_format([complex_tool_call])

        args_str = openai_tools[0]["function"]["arguments"]
        parsed = json.loads(args_str)

        assert parsed["nested"]["array"] == [1, 2, 3]
        assert parsed["nested"]["object"]["key"] == "value"
        assert parsed["simple"] == "string"


class TestPatternRemoval:
    """Test cases for tool call pattern removal."""

    def test_pattern_removal_qwen(self):
        """Test removal of Qwen3 patterns when converting to OpenAI."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.OPENAI)

        content = """
        Text before
        <|tool_call|>
        {"name": "test", "arguments": {}}
        </|tool_call|>
        Text after
        """

        # Mock to return a tool call that will be processed
        test_tool_call = ToolCall(name="test", arguments={})
        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [test_tool_call]

            result = rewriter.rewrite_content(content)

            # Tool call syntax should be removed (converted to OpenAI format removes tags)
            assert "<|tool_call|>" not in result
            assert "</|tool_call|>" not in result
            assert "Text before" in result
            assert "Text after" in result

    def test_pattern_removal_llama(self):
        """Test removal of LLaMA3 patterns when converting to OpenAI."""
        rewriter = ToolCallSyntaxRewriter(SyntaxFormat.OPENAI)

        content = """
        Text before
        <function_call>
        {"name": "test", "arguments": {}}
        </function_call>
        Text after
        """

        # Mock to return a tool call that will be processed
        test_tool_call = ToolCall(name="test", arguments={})
        with patch('abstractllm.tools.syntax_rewriter.parse_tool_calls') as mock_parse:
            mock_parse.return_value = [test_tool_call]

            result = rewriter.rewrite_content(content)

            # Tool call syntax should be removed (converted to OpenAI format removes tags)
            assert "<function_call>" not in result
            assert "</function_call>" not in result
            assert "Text before" in result
            assert "Text after" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Comprehensive tests for ABSTRACTCORE_DEFAULT_TOOL_CALL_TAGS environment variable support.

Tests all predefined formats (qwen3, openai, llama3, xml, gemma), custom tags,
and edge cases to ensure the environment variable works correctly.
"""

import pytest
from unittest.mock import Mock
from abstractllm.providers.streaming import UnifiedStreamProcessor


class TestEnvironmentVariableToolCallTags:
    """Test environment variable support for tool call tag formats"""

    def test_predefined_format_qwen3(self):
        """Test qwen3 predefined format initialization"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="qwen3"
        )

        assert processor.tag_rewriter is not None
        # Check the target tags are correctly set
        assert processor.tag_rewriter.target_tags.start_tag == "<|tool_call|>"
        assert processor.tag_rewriter.target_tags.end_tag == "</|tool_call|>"
        assert processor.tag_rewriter.target_tags.auto_format == False

    def test_predefined_format_openai_no_rewriting(self):
        """Test openai predefined format enables JSON conversion instead of tag rewriting"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # Should have NO tag rewriter (OpenAI uses JSON conversion instead)
        assert processor.tag_rewriter is None

        # Should have OpenAI JSON conversion enabled
        assert processor.convert_to_openai_json is True

    def test_predefined_format_llama3(self):
        """Test llama3 predefined format initialization"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="llama3"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "<function_call>"
        assert processor.tag_rewriter.target_tags.end_tag == "</function_call>"
        assert processor.tag_rewriter.target_tags.auto_format == False

    def test_predefined_format_xml(self):
        """Test xml predefined format initialization"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="xml"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "<tool_call>"
        assert processor.tag_rewriter.target_tags.end_tag == "</tool_call>"
        assert processor.tag_rewriter.target_tags.auto_format == False

    def test_predefined_format_gemma(self):
        """Test gemma predefined format initialization"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="gemma"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "```tool_code\n"
        assert processor.tag_rewriter.target_tags.end_tag == "\n```"
        assert processor.tag_rewriter.target_tags.auto_format == False

    def test_custom_comma_separated_tags(self):
        """Test custom comma-separated tags"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="START,END"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "START"
        assert processor.tag_rewriter.target_tags.end_tag == "END"
        assert processor.tag_rewriter.target_tags.auto_format == False

    def test_custom_complex_comma_separated_tags(self):
        """Test custom complex comma-separated tags"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="[TOOL],[/TOOL]"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "[TOOL]"
        assert processor.tag_rewriter.target_tags.end_tag == "[/TOOL]"
        assert processor.tag_rewriter.target_tags.auto_format == False

    def test_custom_single_tag_auto_format(self):
        """Test custom single tag with auto-formatting"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="MYTOOL"
        )

        assert processor.tag_rewriter is not None
        # When auto_format=True, tags get wrapped with angle brackets
        assert processor.tag_rewriter.target_tags.start_tag == "<MYTOOL>"
        assert processor.tag_rewriter.target_tags.end_tag == "</MYTOOL>"
        assert processor.tag_rewriter.target_tags.auto_format == True

    def test_fallback_to_default_when_no_tool_call_tags(self):
        """Test fallback to default format when no tool_call_tags provided"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags=None,
            default_target_format="llama3"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "<function_call>"
        assert processor.tag_rewriter.target_tags.end_tag == "</function_call>"

    def test_invalid_comma_separated_format(self):
        """Test handling of invalid comma-separated format"""
        # Too many parts
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="START,MIDDLE,END"
        )

        # Should still create a processor but may not have a valid rewriter
        assert processor is not None

    def test_empty_tool_call_tags(self):
        """Test handling of empty tool_call_tags"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="",
            default_target_format="qwen3"
        )

        assert processor.tag_rewriter is not None
        # Should fall back to default format
        assert processor.tag_rewriter.target_tags.start_tag == "<|tool_call|>"
        assert processor.tag_rewriter.target_tags.end_tag == "</|tool_call|>"


class TestEnvironmentVariableTagRewriting:
    """Test actual tag rewriting functionality with environment variable formats"""

    def test_qwen3_format_rewriting(self):
        """Test tool call rewriting with qwen3 format"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="qwen3"
        )

        # LLaMA3 format input should be rewritten to qwen3 format
        input_text = '<function_call>\n{"name": "test_tool", "arguments": {"param": "value"}}\n</function_call>'
        expected_output = '<|tool_call|>{"name": "test_tool", "arguments": {"param": "value"}}</|tool_call|>'

        result = processor._apply_tag_rewriting_direct(input_text)
        assert expected_output in result

    def test_openai_format_no_rewriting(self):
        """Test tool call rewriting with openai format does JSON conversion instead"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="openai"
        )

        # OpenAI should convert text-based tool calls to JSON format
        # Regular text without tool calls should be unchanged
        input_text = 'This is regular text with no tool calls to rewrite.'

        result = processor._apply_tag_rewriting_direct(input_text)
        # Should return unchanged (no tag rewriter when format is openai)
        assert result == input_text

        # But when there ARE tool calls, they should be converted to OpenAI JSON
        tool_call_text = '<|tool_call|>{"name": "test", "arguments": {}}</|tool_call|>'
        result = processor._convert_to_openai_format(tool_call_text)
        # Should contain OpenAI JSON structure
        assert '"type": "function"' in result
        assert '"name": "test"' in result

    def test_llama3_format_rewriting(self):
        """Test tool call rewriting with llama3 format"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="llama3"
        )

        # Qwen3 format input should be rewritten to llama3 format
        input_text = '<|tool_call|>\n{"name": "test_tool", "arguments": {"param": "value"}}\n</|tool_call|>'
        expected_output = '<function_call>{"name": "test_tool", "arguments": {"param": "value"}}</function_call>'

        result = processor._apply_tag_rewriting_direct(input_text)
        assert expected_output in result

    def test_xml_format_rewriting(self):
        """Test tool call rewriting with xml format"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="xml"
        )

        # Qwen3 format input should be rewritten to xml format
        input_text = '<|tool_call|>\n{"name": "test_tool", "arguments": {"param": "value"}}\n</|tool_call|>'
        expected_output = '<tool_call>{"name": "test_tool", "arguments": {"param": "value"}}</tool_call>'

        result = processor._apply_tag_rewriting_direct(input_text)
        assert expected_output in result

    def test_gemma_format_rewriting(self):
        """Test tool call rewriting with gemma format"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="gemma"
        )

        # Qwen3 format input should be rewritten to gemma format
        input_text = '<|tool_call|>\n{"name": "test_tool", "arguments": {"param": "value"}}\n</|tool_call|>'
        expected_output = '```tool_code\n{"name": "test_tool", "arguments": {"param": "value"}}\n```'

        result = processor._apply_tag_rewriting_direct(input_text)
        assert expected_output in result

    def test_custom_comma_separated_rewriting(self):
        """Test tool call rewriting with custom comma-separated tags"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="[TOOL],[/TOOL]"
        )

        # Qwen3 format input should be rewritten to custom format
        input_text = '<|tool_call|>\n{"name": "test_tool", "arguments": {"param": "value"}}\n</|tool_call|>'
        expected_output = '[TOOL]{"name": "test_tool", "arguments": {"param": "value"}}[/TOOL]'

        result = processor._apply_tag_rewriting_direct(input_text)
        assert expected_output in result

    def test_mixed_content_rewriting(self):
        """Test rewriting tool calls within mixed content"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="llama3"
        )

        input_text = 'I need to call a tool: <|tool_call|>{"name": "calc", "arguments": {"expr": "2+2"}}</|tool_call|> to calculate.'
        expected_output = '<function_call>{"name": "calc", "arguments": {"expr": "2+2"}}</function_call>'

        result = processor._apply_tag_rewriting_direct(input_text)
        assert expected_output in result
        assert "I need to call a tool:" in result
        assert "to calculate." in result

    def test_no_tool_calls_in_content(self):
        """Test rewriting when content has no tool calls"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="llama3"
        )

        input_text = "This is just regular text with no tool calls."
        result = processor._apply_tag_rewriting_direct(input_text)

        # Should return unchanged
        assert result == input_text


class TestEnvironmentVariableEdgeCases:
    """Test edge cases and error handling for environment variable tool call tags"""

    def test_whitespace_in_comma_separated_tags(self):
        """Test handling of whitespace in comma-separated tags"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags=" START , END "
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "START"
        assert processor.tag_rewriter.target_tags.end_tag == "END"

    def test_case_sensitivity_predefined_formats(self):
        """Test case sensitivity of predefined format names"""
        # Should work with lowercase
        processor_lower = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="llama3"
        )
        assert processor_lower.tag_rewriter.target_tags.start_tag == "<function_call>"

        # Should not work with uppercase (falls back to custom single tag)
        processor_upper = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="LLAMA3"
        )
        # Should be treated as custom single tag with auto-formatting
        assert processor_upper.tag_rewriter.target_tags.auto_format == True

    def test_unknown_predefined_format_fallback(self):
        """Test fallback behavior for unknown format names"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="unknown_format"
        )

        # Should be treated as custom single tag with auto-formatting
        assert processor.tag_rewriter is not None
        # Auto-formatting wraps with angle brackets
        assert processor.tag_rewriter.target_tags.start_tag == "<unknown_format>"
        assert processor.tag_rewriter.target_tags.end_tag == "</unknown_format>"
        assert processor.tag_rewriter.target_tags.auto_format == True

    def test_none_tool_call_tags_with_custom_default(self):
        """Test None tool_call_tags with custom default format"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags=None,
            default_target_format="xml"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "<tool_call>"
        assert processor.tag_rewriter.target_tags.end_tag == "</tool_call>"

    def test_empty_string_tool_call_tags(self):
        """Test empty string tool_call_tags"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="",
            default_target_format="llama3"
        )

        # Should fall back to default format
        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "<function_call>"
        assert processor.tag_rewriter.target_tags.end_tag == "</function_call>"


class TestEnvironmentVariableStreamingIntegration:
    """Test environment variable formats work correctly with streaming functionality"""

    def create_mock_response_stream(self, content_chunks):
        """Create a mock response stream for testing"""
        from abstractllm.core.types import GenerateResponse
        for chunk in content_chunks:
            yield GenerateResponse(content=chunk, model="test-model")

    def test_streaming_with_environment_variable_format(self):
        """Test streaming with environment variable format works correctly"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="llama3"
        )

        # Simulate streaming chunks that form a tool call
        chunks = [
            "I need to calculate: ",
            "<|tool_call|>",
            '{"name": "calc", "arguments": {"expr": "5*5"}}',
            "</|tool_call|>",
            " The result is 25."
        ]

        response_stream = self.create_mock_response_stream(chunks)
        processed_chunks = list(processor.process_stream(response_stream))

        # Collect all content
        full_content = "".join(chunk.content for chunk in processed_chunks if chunk.content)

        # Should contain rewritten tool call in llama3 format
        assert '<function_call>{"name": "calc", "arguments": {"expr": "5*5"}}</function_call>' in full_content
        assert "I need to calculate:" in full_content
        assert "The result is 25." in full_content

    def test_streaming_preserves_non_tool_content(self):
        """Test streaming preserves non-tool content correctly"""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            tool_call_tags="xml"
        )

        chunks = [
            "This is regular text ",
            "without any tool calls ",
            "in the content."
        ]

        response_stream = self.create_mock_response_stream(chunks)
        processed_chunks = list(processor.process_stream(response_stream))

        # Should preserve all content unchanged
        full_content = "".join(chunk.content for chunk in processed_chunks if chunk.content)
        assert full_content == "This is regular text without any tool calls in the content."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
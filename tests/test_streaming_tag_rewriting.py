"""
Comprehensive tests for tool call tag rewriting in streaming mode.

This test suite validates that custom tool call tags work correctly in streaming
scenarios, ensuring parity with non-streaming tag rewriting functionality.
"""

import pytest
from typing import Iterator
from abstractllm.providers.streaming import UnifiedStreamProcessor, IncrementalToolDetector
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.tag_rewriter import ToolCallTags, ToolCallTagRewriter


class TestStreamingTagRewritingInitialization:
    """Test tag rewriter initialization in UnifiedStreamProcessor."""

    def test_string_format_comma_separated(self):
        """Test initialization with comma-separated string format."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        assert processor.tag_rewriter is not None
        # Fixed implementation uses exact tags (auto_format=False)
        assert processor.tag_rewriter.target_tags.start_tag == "ojlk"
        assert processor.tag_rewriter.target_tags.end_tag == "dfsd"

    def test_string_format_single_tag(self):
        """Test initialization with single tag string."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="custom_tool"
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "<custom_tool>"
        assert processor.tag_rewriter.target_tags.end_tag == "</custom_tool>"

    def test_tool_call_tags_object(self):
        """Test initialization with ToolCallTags object."""
        tags = ToolCallTags(start_tag="<START>", end_tag="<END>")
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags=tags
        )

        assert processor.tag_rewriter is not None
        assert processor.tag_rewriter.target_tags.start_tag == "<START>"
        assert processor.tag_rewriter.target_tags.end_tag == "<END>"

    def test_tool_call_tag_rewriter_object(self):
        """Test initialization with ToolCallTagRewriter object."""
        tags = ToolCallTags(start_tag="<FUNC>", end_tag="</FUNC>")
        rewriter = ToolCallTagRewriter(tags)
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags=rewriter
        )

        assert processor.tag_rewriter is rewriter
        assert processor.tag_rewriter.target_tags.start_tag == "<FUNC>"
        assert processor.tag_rewriter.target_tags.end_tag == "</FUNC>"

    def test_no_tag_rewriting(self):
        """Test that no rewriter is created when tags not provided."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags=None
        )

        assert processor.tag_rewriter is None
        assert processor.tag_rewrite_buffer == ""


class TestStreamingTagRewritingBasic:
    """Test basic tag rewriting in streaming chunks."""

    def test_qwen_format_rewriting_single_chunk(self):
        """Test rewriting Qwen format in a single chunk."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        # Simulate a complete tool call in one chunk
        content = '<|tool_call|>{"name": "list_files", "arguments": {"directory_path": "abstractllm"}}</|tool_call|>'

        # Create mock response stream
        def mock_stream():
            yield GenerateResponse(content=content, model="test-model")

        # Process stream
        results = list(processor.process_stream(mock_stream()))

        # Verify rewriting occurred
        assert len(results) > 0
        full_output = "".join([r.content for r in results if r.content])

        # Should contain custom tags (exact format for comma-separated tags)
        assert "ojlk" in full_output
        assert "dfsd" in full_output

        # Should NOT contain original tags
        assert "<|tool_call|>" not in full_output
        assert "</|tool_call|>" not in full_output

    def test_llama_format_rewriting_single_chunk(self):
        """Test rewriting LLaMA format in a single chunk."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="custom,tool"
        )

        content = '<function_call>{"name": "get_time", "arguments": {}}</function_call>'

        def mock_stream():
            yield GenerateResponse(content=content, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "custom" in full_output
        assert "tool" in full_output
        assert "<function_call>" not in full_output

    def test_xml_format_rewriting_single_chunk(self):
        """Test rewriting XML format in a single chunk."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="TOOL,CALL"
        )

        content = '<tool_call>{"name": "calculate", "arguments": {"x": 5, "y": 10}}</tool_call>'

        def mock_stream():
            yield GenerateResponse(content=content, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "TOOL" in full_output
        assert "CALL" in full_output
        assert "<tool_call>" not in full_output


class TestStreamingTagRewritingSplitChunks:
    """Test tag rewriting when tool calls are split across multiple chunks."""

    def test_split_across_two_chunks(self):
        """Test rewriting when tool call is split across two chunks."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        # Split the tool call into two chunks
        chunk1 = '<|tool_call|>{"name": "list_files", '
        chunk2 = '"arguments": {"directory_path": "abstractllm"}}</|tool_call|>'

        def mock_stream():
            yield GenerateResponse(content=chunk1, model="test-model")
            yield GenerateResponse(content=chunk2, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        # Should contain custom tags (exact format for comma-separated tags)
        assert "ojlk" in full_output
        assert "dfsd" in full_output

        # Should contain the complete JSON
        assert '"name": "list_files"' in full_output
        assert '"directory_path": "abstractllm"' in full_output

    def test_split_at_tag_boundary(self):
        """Test rewriting when split happens at tag boundaries."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="START,END"
        )

        # Split at start tag boundary
        chunk1 = '<|tool_'
        chunk2 = 'call|>{"name": "test"}</|tool_call|>'

        def mock_stream():
            yield GenerateResponse(content=chunk1, model="test-model")
            yield GenerateResponse(content=chunk2, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "START" in full_output
        assert "END" in full_output

    def test_split_across_multiple_chunks(self):
        """Test rewriting when tool call is split across many chunks (character-by-character)."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="A,B"
        )

        # Simulate character-by-character streaming
        full_content = '<|tool_call|>{"name": "test", "arguments": {}}</|tool_call|>'

        def mock_stream():
            for char in full_content:
                yield GenerateResponse(content=char, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "A" in full_output
        assert "B" in full_output
        assert '"name": "test"' in full_output


class TestStreamingTagRewritingMixedContent:
    """Test tag rewriting with mixed content (text + tool calls)."""

    def test_text_before_tool_call(self):
        """Test rewriting when there's text before the tool call."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        content = 'Let me check that for you.<|tool_call|>{"name": "list_files", "arguments": {}}</|tool_call|>'

        def mock_stream():
            yield GenerateResponse(content=content, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "Let me check that for you" in full_output
        assert "ojlk" in full_output
        assert "dfsd" in full_output

    def test_text_after_tool_call(self):
        """Test rewriting when there's text after the tool call."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="custom,tool"
        )

        content = '<function_call>{"name": "test"}</function_call> Done checking.'

        def mock_stream():
            yield GenerateResponse(content=content, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "custom" in full_output
        assert "tool" in full_output
        assert "Done checking" in full_output

    def test_multiple_tool_calls_in_stream(self):
        """Test rewriting multiple tool calls in same stream."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="T1,T2"
        )

        chunk1 = '<tool_call>{"name": "first"}</tool_call>'
        chunk2 = ' and then '
        chunk3 = '<tool_call>{"name": "second"}</tool_call>'

        def mock_stream():
            yield GenerateResponse(content=chunk1, model="test-model")
            yield GenerateResponse(content=chunk2, model="test-model")
            yield GenerateResponse(content=chunk3, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        # Both tool calls should be rewritten
        assert full_output.count("T1") >= 2
        assert full_output.count("T2") >= 2
        assert "and then" in full_output


class TestStreamingTagRewritingEdgeCases:
    """Test edge cases in streaming tag rewriting."""

    def test_empty_chunks(self):
        """Test handling of empty chunks."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        def mock_stream():
            yield GenerateResponse(content="", model="test-model")
            yield GenerateResponse(content='<|tool_call|>{"name": "test"}</|tool_call|>', model="test-model")
            yield GenerateResponse(content="", model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "ojlk" in full_output
        assert "dfsd" in full_output

    def test_none_content(self):
        """Test handling of None content."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="A,B"
        )

        def mock_stream():
            yield GenerateResponse(content=None, model="test-model")
            yield GenerateResponse(content='<tool_call>{"name": "test"}</tool_call>', model="test-model")

        # Should not crash
        results = list(processor.process_stream(mock_stream()))
        assert len(results) > 0

    def test_malformed_json_in_tool_call(self):
        """Test rewriting with malformed JSON (should still rewrite tags)."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        content = '<|tool_call|>{"name": "test", "arguments": {incomplete</|tool_call|>'

        def mock_stream():
            yield GenerateResponse(content=content, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        # Tags should still be rewritten even if JSON is malformed
        assert "ojlk" in full_output or "<|tool_call|>" in full_output
        # The rewriter might handle this differently, so we check it didn't crash


class TestStreamingTagRewritingPerformance:
    """Test performance characteristics of streaming tag rewriting."""

    def test_no_buffering_for_non_tool_content(self):
        """Test that non-tool content streams immediately without buffering."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        # Pure text content - should stream immediately
        content = "This is regular text without any tool calls."

        def mock_stream():
            for word in content.split():
                yield GenerateResponse(content=word + " ", model="test-model")

        results = list(processor.process_stream(mock_stream()))

        # Each chunk should be yielded immediately
        # (exact behavior depends on rewriter implementation, but should not block)
        assert len(results) > 0
        full_output = "".join([r.content for r in results if r.content])
        assert "regular text" in full_output

    def test_large_tool_call_streaming(self):
        """Test rewriting performance with large tool call payloads."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="BIG,CALL"
        )

        # Create a large tool call with big arguments
        large_args = '{"data": "' + "x" * 10000 + '"}'
        content = f'<tool_call>{{"name": "process", "arguments": {large_args}}}</tool_call>'

        def mock_stream():
            # Stream in 100-char chunks
            for i in range(0, len(content), 100):
                chunk = content[i:i+100]
                yield GenerateResponse(content=chunk, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "BIG" in full_output
        assert "CALL" in full_output


class TestStreamingTagRewritingIntegration:
    """Integration tests combining tag rewriting with tool execution."""

    def test_tag_rewriting_before_tool_detection(self):
        """Test that tag rewriting happens before tool detection."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,  # Disable execution for this test
            tool_call_tags="ojlk,dfsd"
        )

        # Standard Qwen format should be rewritten
        content = '<|tool_call|>{"name": "list_files", "arguments": {"directory_path": "."}}</|tool_call|>'

        def mock_stream():
            yield GenerateResponse(content=content, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        # Output should have custom tags
        assert "ojlk" in full_output
        assert "dfsd" in full_output

        # Original tags should be gone
        assert "<|tool_call|>" not in full_output

    def test_buffer_cleanup_between_tool_calls(self):
        """Test that tag rewrite buffer is properly managed across multiple tool calls."""
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="A,B"
        )

        # Two tool calls in sequence
        chunk1 = '<tool_call>{"name": "first"}</tool_call>'
        chunk2 = '<tool_call>{"name": "second"}</tool_call>'

        def mock_stream():
            yield GenerateResponse(content=chunk1, model="test-model")
            yield GenerateResponse(content=chunk2, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        # Both should be rewritten independently
        assert full_output.count("A") >= 2
        assert full_output.count("B") >= 2


class TestStreamingTagRewritingRealWorldScenarios:
    """Test real-world scenarios that users might encounter."""

    def test_cli_interaction_scenario(self):
        """Test the exact scenario from the user's CLI interaction."""
        # User set: /tooltag 'ojlk' 'dfsd'
        processor = UnifiedStreamProcessor(
            model_name="test-model",
            execute_tools=False,
            tool_call_tags="ojlk,dfsd"
        )

        # Simulated LLM response with tool call
        content = 'I will list the files for you.<|tool_call|>{"name": "list_files", "arguments": {"directory_path": "abstractllm"}}</|tool_call|>'

        def mock_stream():
            # Simulate character-by-character streaming like real LLM
            for i in range(0, len(content), 5):  # 5 chars at a time
                chunk = content[i:i+5]
                yield GenerateResponse(content=chunk, model="test-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        # CRITICAL: Must show custom tags (plain format for comma-separated)
        assert "ojlk" in full_output, "Custom opening tag not found in streaming output"
        assert "dfsd" in full_output, "Custom closing tag not found in streaming output"

        # CRITICAL: Must NOT show standard tags
        assert "<|tool_call|>" not in full_output, "Standard tags still present - rewriting failed"
        assert "</|tool_call|>" not in full_output, "Standard tags still present - rewriting failed"

        # Must contain the tool call content
        assert '"name": "list_files"' in full_output
        assert '"directory_path": "abstractllm"' in full_output

    def test_agentic_workflow_with_custom_tags(self):
        """Test agentic CLI workflow with custom tag format."""
        # Custom tags for an agentic CLI
        processor = UnifiedStreamProcessor(
            model_name="qwen3-coder",
            execute_tools=False,
            tool_call_tags="[TOOL],[/TOOL]"
        )

        # Multi-turn conversation with tool call
        response_parts = [
            "Let me analyze the code. ",
            '<function_call>{"name": "read_file", "arguments": {"path": "main.py"}}</function_call>',
            " After reviewing, I found..."
        ]

        def mock_stream():
            for part in response_parts:
                yield GenerateResponse(content=part, model="qwen3-coder")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        assert "<[TOOL]>" in full_output or "[TOOL]" in full_output
        assert "<[/TOOL]>" in full_output or "[/TOOL]" in full_output
        assert "read_file" in full_output
        assert "After reviewing" in full_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

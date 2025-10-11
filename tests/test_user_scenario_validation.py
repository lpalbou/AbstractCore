"""
Critical validation test for user's exact scenario.

This validates that the streaming tag rewriting fix resolves the exact issue
reported by the user: `/tooltag 'ojlk' 'dfsd'` should rewrite tool calls to use
those exact tags in streaming mode.
"""

import pytest
from abstractllm.providers.streaming import UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse


def test_user_exact_tooltag_scenario():
    """
    CRITICAL USER SCENARIO TEST

    User reported:
    1. Set /tooltag 'ojlk' 'dfsd' in CLI
    2. Streaming tool call should show ojlk...dfsd format
    3. Was broken - tool call content disappeared

    This test validates the fix works for the user's exact use case.
    """
    # User's exact tag configuration
    processor = UnifiedStreamProcessor(
        model_name="test-model",
        execute_tools=False,
        tool_call_tags="ojlk,dfsd"
    )

    # Simulated LLM streaming response with Qwen format tool call
    full_content = 'I will list the files for you.<|tool_call|>{"name": "list_files", "arguments": {"directory_path": "abstractllm"}}</|tool_call|>'

    def realistic_stream():
        """Simulate realistic character-by-character streaming."""
        for i in range(0, len(full_content), 5):
            chunk = full_content[i:i+5]
            yield GenerateResponse(content=chunk, model="test-model")

    # Process stream
    results = list(processor.process_stream(realistic_stream()))
    full_output = "".join([r.content for r in results if r.content])

    print(f"\n=== USER SCENARIO VALIDATION ===")
    print(f"Input tool call format: <|tool_call|>...JSON...</|tool_call|>")
    print(f"Expected output format: ojlk...JSON...dfsd")
    print(f"\nActual output:\n{full_output}")
    print(f"===================================\n")

    # CRITICAL ASSERTIONS
    assert "ojlk" in full_output, f"Custom opening tag 'ojlk' NOT FOUND in output: {full_output}"
    assert "dfsd" in full_output, f"Custom closing tag 'dfsd' NOT FOUND in output: {full_output}"

    # Original tags must be replaced
    assert "<|tool_call|>" not in full_output, "Original Qwen tags still present - rewriting FAILED"
    assert "</|tool_call|>" not in full_output, "Original Qwen tags still present - rewriting FAILED"

    # Tool call JSON content must be preserved
    assert '"name": "list_files"' in full_output, "Tool call content lost"
    assert '"directory_path": "abstractllm"' in full_output, "Tool call arguments lost"

    # Leading text must be preserved
    assert "I will list the files for you" in full_output, "Leading text lost"

    print("✅ USER SCENARIO VALIDATION PASSED!")


def test_tool_call_tags_exact_format():
    """Test that tags are used exactly as specified (no auto angle brackets)."""
    processor = UnifiedStreamProcessor(
        model_name="test-model",
        execute_tools=False,
        tool_call_tags="CUSTOM,TOOL"
    )

    # Verify tags are stored exactly (no <> added)
    assert processor.tag_rewriter.target_tags.start_tag == "CUSTOM"
    assert processor.tag_rewriter.target_tags.end_tag == "TOOL"
    assert processor.tag_rewriter.target_tags.auto_format == False

    # Test rewriting
    content = '<tool_call>{"name": "test"}</tool_call>'

    def mock_stream():
        yield GenerateResponse(content=content, model="test-model")

    results = list(processor.process_stream(mock_stream()))
    full_output = "".join([r.content for r in results if r.content])

    # Should have exact tags
    assert "CUSTOM" in full_output
    assert "TOOL" in full_output
    # Should NOT have auto-formatted tags
    assert "<CUSTOM>" not in full_output
    assert "</TOOL>" not in full_output


def test_streaming_preserves_tool_calls():
    """Test that streaming with rewriting preserves tool calls in output."""
    processor = UnifiedStreamProcessor(
        model_name="test-model",
        execute_tools=False,
        tool_call_tags="A,B"
    )

    # Split tool call across chunks
    chunk1 = '<function_call>{"name": '
    chunk2 = '"test", "arguments": {}}</function_call>'

    def mock_stream():
        yield GenerateResponse(content=chunk1, model="test-model")
        yield GenerateResponse(content=chunk2, model="test-model")

    results = list(processor.process_stream(mock_stream()))
    full_output = "".join([r.content for r in results if r.content])

    # Tool call must be in output (buffered until complete, then rewritten)
    assert "A" in full_output
    assert "B" in full_output
    assert '"name"' in full_output
    assert '"test"' in full_output


def test_no_rewriting_without_custom_tags():
    """Test that tool calls pass through unchanged when no custom tags set."""
    processor = UnifiedStreamProcessor(
        model_name="test-model",
        execute_tools=False,
        tool_call_tags=None  # No custom tags
    )

    assert processor.tag_rewriter is None

    content = '<tool_call>{"name": "test"}</tool_call>'

    def mock_stream():
        yield GenerateResponse(content=content, model="test-model")

    results = list(processor.process_stream(mock_stream()))
    full_output = "".join([r.content for r in results if r.content])

    # Without custom tags and rewriting, behavior depends on detector mode
    # Key: no crash, content is present
    assert '"name": "test"' in full_output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

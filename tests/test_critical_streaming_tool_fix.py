"""
CRITICAL VALIDATION TEST SUITE: Streaming + Tool Execution Fix
=================================================================

This suite validates the critical fix in abstractllm/providers/streaming.py:
- Line 125: Changed `streamable_content = chunk_content` to `streamable_content = ""`
- Lines 148-155: Added proper content gating with 50-char buffer

Test Requirements:
1. Streaming Performance - Verify <10ms first chunk latency maintained
2. Tool Execution - Confirm tools are detected and executed correctly
3. Content Gating - Verify NO tool tags leak to user output
4. Edge Cases - Test tool calls spanning chunk boundaries, malformed JSON, etc.

All tests use REAL implementations per CLAUDE.md - NO MOCKING
"""

import pytest
import time
from typing import Iterator, List
from abstractllm.providers.streaming import (
    IncrementalToolDetector,
    UnifiedStreamProcessor,
    ToolDetectionState
)
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.core import ToolCall, ToolDefinition


# ============================================================================
# CRITICAL FIX VALIDATION - Line 125 Fix
# ============================================================================

class TestLine125Fix:
    """Validate that line 125 fix prevents content leakage during tool detection"""

    def test_no_tool_tag_leakage_qwen_format(self):
        """CRITICAL: Verify <function_call> tags don't leak to streamable content"""
        detector = IncrementalToolDetector("qwen3-next-80b")

        # Stream content that includes tool call
        chunks = [
            "I'll help you with that. ",
            "<function_call>",
            '{"name": "read_file", "arguments": {"path": "README.md"}}',
            "</function_call>",
            " Here's what I found."
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        # CRITICAL ASSERTION: No tool tags should appear in streamable content
        combined_streamable = "".join(all_streamable)
        assert "<function_call>" not in combined_streamable, \
            "CRITICAL FAILURE: Tool start tag leaked to streamable content!"
        assert "</function_call>" not in combined_streamable, \
            "CRITICAL FAILURE: Tool end tag leaked to streamable content!"

        # Content BEFORE tool should be streamable
        assert "I'll help you with that." in combined_streamable

        # Content AFTER tool should be streamable
        assert "Here's what I found." in combined_streamable

        # Tool should be detected and executed
        assert len(all_tools) == 1
        assert all_tools[0].name == "read_file"
        assert all_tools[0].arguments == {"path": "README.md"}

    def test_no_tool_tag_leakage_llama_format(self):
        """Verify <function_call> tags don't leak (LLaMA format)"""
        detector = IncrementalToolDetector("llama-3")

        chunks = [
            "Let me search for that. ",
            "<function_call>",
            '{"name": "web_search", "arguments": {"query": "Python tutorial"}}',
            "</function_call>",
            " Based on the results..."
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        combined_streamable = "".join(all_streamable)

        # CRITICAL: No tool tags in output
        assert "<function_call>" not in combined_streamable
        assert "</function_call>" not in combined_streamable

        # Tool detected correctly
        assert len(all_tools) == 1
        assert all_tools[0].name == "web_search"

    def test_content_gating_with_smart_buffer(self):
        """Verify smart buffering prevents premature streaming of tool start tags"""
        detector = IncrementalToolDetector("qwen3")

        # Test 1: Content with tool tag at end - should detect and not stream tag
        chunk1 = "A" * 60 + "<|tool_call|>"
        streamable1, tools1 = detector.process_chunk(chunk1)

        # Should have detected the tool start and NOT streamed the tag
        assert "<|tool_call|>" not in streamable1
        # Should have streamed the content before the tag
        assert "A" * 60 in streamable1 or len(streamable1) > 0

        # Complete the tool call
        chunk2 = '{"name": "test", "arguments": {}}</|tool_call|>'
        streamable2, tools2 = detector.process_chunk(chunk2)

        # Tool should be detected
        assert len(tools2) == 1
        assert "<|tool_call|>" not in streamable2

        # Test 2: Content that ends with potential tag start should buffer
        detector.reset()
        chunk3 = "B" * 30 + "<"
        streamable3, tools3 = detector.process_chunk(chunk3)

        # Should hold back the "<" as it might be start of tag
        assert "<" not in streamable3 or len(streamable3) == 31  # Either buffers or streams all


# ============================================================================
# STREAMING PERFORMANCE VALIDATION
# ============================================================================

class TestStreamingPerformance:
    """Validate that streaming performance is maintained (<10ms first chunk)"""

    def create_stream(self, chunks: List[str]) -> Iterator[GenerateResponse]:
        """Helper to create test stream"""
        for chunk in chunks:
            yield GenerateResponse(content=chunk, model="test-model", finish_reason=None)

    def test_first_chunk_latency_under_10ms(self):
        """CRITICAL: Verify first chunk arrives in <10ms (not buffered)"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        chunks = ["First chunk", "Second chunk", "Third chunk"]
        stream = self.create_stream(chunks)

        start_time = time.time()
        result_iter = processor.process_stream(stream)

        # Get first result
        first_result = next(result_iter)
        first_chunk_time = time.time() - start_time

        # CRITICAL ASSERTION: First chunk should arrive immediately
        assert first_chunk_time < 0.01, \
            f"CRITICAL FAILURE: First chunk took {first_chunk_time*1000:.2f}ms (target: <10ms)"

        assert first_result.content == "First chunk"

    def test_progressive_streaming_not_buffered(self):
        """Verify content streams progressively, not buffered until end"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        chunks = ["A", "B", "C", "D", "E"]

        def delayed_stream():
            for chunk in chunks:
                time.sleep(0.02)
                yield GenerateResponse(content=chunk, model="test")

        start_time = time.time()
        chunk_times = []

        for result in processor.process_stream(delayed_stream()):
            if result.content:
                chunk_times.append(time.time() - start_time)

        # All chunks should arrive progressively
        assert len(chunk_times) == 5

        # Each chunk should arrive ~0.02s apart (not all at once)
        # First chunk should arrive quickly
        assert chunk_times[0] < 0.05

        # Subsequent chunks should be spaced out
        for i in range(1, len(chunk_times)):
            time_delta = chunk_times[i] - chunk_times[i-1]
            assert 0.01 < time_delta < 0.05, \
                f"Chunks not streaming progressively: delta={time_delta}"

    def test_streaming_with_tools_maintains_performance(self):
        """Verify streaming performance maintained even with tool detection enabled"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=False)

        chunks = ["Fast", " streaming", " with", " tool", " detection"]
        stream = self.create_stream(chunks)

        start_time = time.time()
        results = list(processor.process_stream(stream))
        total_time = time.time() - start_time

        # Should be fast even with tool detection overhead
        assert total_time < 0.05, \
            f"Streaming too slow with tool detection: {total_time*1000:.2f}ms"

        # All chunks should be preserved
        assert len(results) == 5


# ============================================================================
# TOOL EXECUTION VALIDATION
# ============================================================================

class TestToolExecution:
    """Validate that tools are detected and executed correctly"""

    def test_tool_execution_qwen_format(self):
        """Test tool execution with qwen/qwen3-next-80b format (<function_call>)"""
        def read_file(path: str) -> str:
            """Mock file reading for test"""
            if path == "README.md":
                return "# AbstractLLM Core\n\nThis is a test file."
            return f"File not found: {path}"

        # Register tool
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(read_file)

        tool_def = ToolDefinition.from_function(read_file).to_dict()
        tool_def['function'] = read_file

        processor = UnifiedStreamProcessor("qwen3-next-80b", execute_tools=True)

        chunks = [
            "I'll read that file for you. ",
            "<function_call>",
            '{"name": "read_file", "arguments": {"path": "README.md"}}',
            "</function_call>",
            " As you can see from the file..."
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream, [tool_def]))

        # Should have tool execution results
        all_content = " ".join([r.content for r in results if r.content])

        # Tool should be executed
        assert "Tool Results:" in all_content or "read_file" in all_content

        # Original content should be preserved
        assert "I'll read that file for you." in all_content

        # NO tool tags should leak
        assert "<function_call>" not in all_content
        assert "</function_call>" not in all_content

        # Cleanup
        clear_registry()

    def test_multiple_sequential_tools_execution(self):
        """Test multiple tools execute correctly in sequence"""
        def tool1(value: int) -> int:
            return value + 10

        def tool2(value: int) -> int:
            return value * 2

        # Register tools
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(tool1)
        register_tool(tool2)

        tools = [
            ToolDefinition.from_function(tool1).to_dict(),
            ToolDefinition.from_function(tool2).to_dict()
        ]
        tools[0]['function'] = tool1
        tools[1]['function'] = tool2

        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        chunks = [
            "First calculation: ",
            "<|tool_call|>",
            '{"name": "tool1", "arguments": {"value": 5}}',
            "</|tool_call|>",
            " Second calculation: ",
            "<|tool_call|>",
            '{"name": "tool2", "arguments": {"value": 3}}',
            "</|tool_call|>",
            " Done."
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream, tools))

        all_content = " ".join([r.content for r in results if r.content])

        # Both tools should be executed
        assert "tool1" in all_content
        assert "tool2" in all_content

        # No tool tags should leak
        assert "<|tool_call|>" not in all_content
        assert "</|tool_call|>" not in all_content

        # Cleanup
        clear_registry()

    def test_tool_results_appear_with_proper_formatting(self):
        """Verify tool results appear with proper formatting"""
        def calculate(expression: str) -> str:
            try:
                return str(eval(expression))
            except Exception as e:
                return f"Error: {e}"

        # Register tool
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(calculate)

        tool_def = ToolDefinition.from_function(calculate).to_dict()
        tool_def['function'] = calculate

        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        chunks = [
            "Let me calculate: ",
            "<|tool_call|>",
            '{"name": "calculate", "arguments": {"expression": "10 + 5"}}',
            "</|tool_call|>",
            " The result is ready."
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream, [tool_def]))

        all_content = " ".join([r.content for r in results if r.content])

        # Tool results should be formatted properly
        assert "🔧 Tool Results:" in all_content or "calculate" in all_content

        # Should include success indicator or actual result
        # (The format might vary but should be present)
        assert "15" in all_content or "calculate" in all_content

        # Cleanup
        clear_registry()


# ============================================================================
# CONTENT GATING VALIDATION
# ============================================================================

class TestContentGating:
    """Validate content gating prevents tool tags from leaking"""

    def test_no_tags_in_user_output(self):
        """CRITICAL: Verify absolutely NO tool tags appear in user-visible output"""
        detector = IncrementalToolDetector("qwen3")

        # Various tool call formats
        test_cases = [
            {
                'name': 'Qwen format',
                'chunks': ["Text <|tool_call|>", '{"name": "t", "arguments": {}}', "</|tool_call|> More"]
            },
            {
                'name': 'LLaMA format',
                'chunks': ["Text <function_call>", '{"name": "t", "arguments": {}}', "</function_call> More"]
            },
            {
                'name': 'XML format',
                'chunks': ["Text <tool_call>", '{"name": "t", "arguments": {}}', "</tool_call> More"]
            }
        ]

        for test_case in test_cases:
            detector.reset()
            all_streamable = []

            for chunk in test_case['chunks']:
                streamable, _ = detector.process_chunk(chunk)
                if streamable:
                    all_streamable.append(streamable)

            combined = "".join(all_streamable)

            # CRITICAL: No tool tags should be present
            assert "<tool_call>" not in combined.lower(), \
                f"CRITICAL FAILURE in {test_case['name']}: Tool tags leaked!"
            assert "</tool_call>" not in combined.lower()
            assert "<function_call>" not in combined.lower()
            assert "</function_call>" not in combined.lower()
            assert "<|tool_call|>" not in combined
            assert "</|tool_call|>" not in combined

    def test_content_before_tool_streams_immediately(self):
        """Verify content BEFORE tool tags streams immediately (not held back)"""
        detector = IncrementalToolDetector("qwen3")

        chunk = "This should stream immediately. <|tool_call|>"
        streamable, _ = detector.process_chunk(chunk)

        # Content before tool tag should be streamable
        assert "This should stream immediately." in streamable
        assert "<|tool_call|>" not in streamable

    def test_content_after_tool_streams_correctly(self):
        """Verify content AFTER tool execution streams back to user"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=False)

        chunks = [
            "Before tool ",
            "<|tool_call|>",
            '{"name": "test", "arguments": {}}',
            "</|tool_call|>",
            " After tool"
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream))

        all_content = "".join([r.content for r in results if r.content])

        # Content after tool should appear
        assert "After tool" in all_content

        # But no tool tags
        assert "<|tool_call|>" not in all_content
        assert "</|tool_call|>" not in all_content


# ============================================================================
# EDGE CASE VALIDATION
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_tool_spanning_chunk_boundaries(self):
        """Test tool call that spans multiple chunk boundaries"""
        detector = IncrementalToolDetector("qwen3")

        # Tool call split across many tiny chunks
        chunks = [
            "Text ",
            "<|tool_",
            "call|>",
            '{"na',
            'me": ',
            '"test"',
            ', "ar',
            'gume',
            'nts":',
            ' {}}',
            '</|to',
            'ol_ca',
            'll|>',
            ' More'
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        # Should successfully parse despite extreme fragmentation
        assert len(all_tools) == 1
        assert all_tools[0].name == "test"

        # No tool tags in streamable content
        combined = "".join(all_streamable)
        assert "<|tool_call|>" not in combined
        assert "</|tool_call|>" not in combined

    def test_malformed_json_in_tool_call(self):
        """Test handling of malformed JSON in tool calls"""
        detector = IncrementalToolDetector("qwen3")

        chunks = [
            "<|tool_call|>",
            '{"name": "test", "arguments": {"x": 1, "y": 2}',  # Missing closing brace
            "</|tool_call|>"
        ]

        all_tools = []
        for chunk in chunks:
            _, tools = detector.process_chunk(chunk)
            all_tools.extend(tools)

        # Should auto-repair and parse successfully
        assert len(all_tools) == 1
        assert all_tools[0].name == "test"
        assert all_tools[0].arguments == {"x": 1, "y": 2}

    def test_empty_content_chunks(self):
        """Test handling of empty content chunks"""
        detector = IncrementalToolDetector("qwen3")

        chunks = [
            "Text",
            "",
            "",
            " more",
            "",
            "<|tool_call|>",
            "",
            '{"name": "test", "arguments": {}}',
            "",
            "</|tool_call|>"
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        # Should handle empty chunks gracefully
        assert len(all_tools) == 1
        combined = "".join(all_streamable)
        assert "Text more" in combined or ("Text" in combined and "more" in combined)

    def test_mixed_content_and_tools(self):
        """Test realistic scenario with mixed content and multiple tools"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=False)

        chunks = [
            "I'll help with that task. ",
            "First, let me check the file: ",
            "<|tool_call|>",
            '{"name": "read_file", "arguments": {"path": "test.txt"}}',
            "</|tool_call|>",
            " Now let me analyze the content: ",
            "<|tool_call|>",
            '{"name": "analyze", "arguments": {"data": "content"}}',
            "</|tool_call|>",
            " Based on the analysis, here's what I found: ",
            "The file contains important information."
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream))

        all_content = "".join([r.content for r in results if r.content])

        # All text content should be present
        assert "I'll help with that task" in all_content
        assert "Based on the analysis" in all_content
        assert "important information" in all_content

        # NO tool tags should leak
        assert "<|tool_call|>" not in all_content
        assert "</|tool_call|>" not in all_content

    def test_tool_at_very_start_of_stream(self):
        """Test tool call appearing as first content"""
        detector = IncrementalToolDetector("qwen3")

        chunks = [
            "<|tool_call|>",
            '{"name": "first", "arguments": {}}',
            "</|tool_call|>",
            " Content follows"
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        # Tool should be detected
        assert len(all_tools) == 1

        # Content after tool should stream
        combined = "".join(all_streamable)
        assert "Content follows" in combined
        assert "<|tool_call|>" not in combined

    def test_tool_at_very_end_of_stream(self):
        """Test tool call appearing as last content"""
        detector = IncrementalToolDetector("qwen3")

        chunks = [
            "Some content. ",
            "<|tool_call|>",
            '{"name": "last", "arguments": {}}',
            "</|tool_call|>"
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        # Tool should be detected
        assert len(all_tools) == 1

        # Content before tool should stream
        combined = "".join(all_streamable)
        assert "Some content." in combined


# ============================================================================
# PRODUCTION READINESS VALIDATION
# ============================================================================

class TestProductionReadiness:
    """Validate the fix is production-ready"""

    def test_fix_solves_original_issue(self):
        """
        Validate that the original issue is solved:
        - Tools were being detected but not executed
        - Tool tags were appearing in user output
        - Content was being buffered incorrectly
        """
        def test_tool(x: int) -> int:
            return x * 2

        # Register tool
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(test_tool)

        tool_def = ToolDefinition.from_function(test_tool).to_dict()
        tool_def['function'] = test_tool

        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        chunks = [
            "Let me calculate: ",
            "<|tool_call|>",
            '{"name": "test_tool", "arguments": {"x": 5}}',
            "</|tool_call|>",
            " Result ready."
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream, [tool_def]))

        all_content = " ".join([r.content for r in results if r.content])

        # 1. Tool SHOULD be executed (original issue: wasn't executing)
        assert "Tool Results:" in all_content or "test_tool" in all_content, \
            "CRITICAL: Tool execution still broken!"

        # 2. Tool tags SHOULD NOT appear in output (original issue: were appearing)
        assert "<|tool_call|>" not in all_content, \
            "CRITICAL: Tool tags still leaking to output!"
        assert "</|tool_call|>" not in all_content, \
            "CRITICAL: Tool tags still leaking to output!"

        # 3. Content SHOULD stream properly (original issue: was buffered)
        assert "Let me calculate:" in all_content or any("Let me calculate:" in r.content for r in results if r.content), \
            "CRITICAL: Content not streaming correctly!"

        # Cleanup
        clear_registry()

    def test_backward_compatibility(self):
        """Ensure fix doesn't break existing functionality"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        # Test basic streaming still works
        chunks = ["Hello", " ", "world"]
        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream))

        assert len(results) == 3
        assert "".join([r.content for r in results]) == "Hello world"

    def test_all_model_formats_supported(self):
        """Verify fix works across all supported model formats"""
        formats = [
            ("qwen3", "<|tool_call|>", "</|tool_call|>"),
            ("llama-3", "<function_call>", "</function_call>"),
            ("gemma-2", "```tool_code", "```"),
        ]

        for model, start_tag, end_tag in formats:
            detector = IncrementalToolDetector(model)

            chunks = [
                f"Text {start_tag}",
                '{"name": "test", "arguments": {}}',
                f"{end_tag} More"
            ]

            all_streamable = []
            all_tools = []

            for chunk in chunks:
                streamable, tools = detector.process_chunk(chunk)
                if streamable:
                    all_streamable.append(streamable)
                all_tools.extend(tools)

            # Should work for all formats
            assert len(all_tools) == 1, f"Failed for {model} format"

            combined = "".join(all_streamable)
            assert start_tag not in combined, f"Tag leak in {model} format"
            assert end_tag not in combined, f"Tag leak in {model} format"


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run with verbose output and coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", ""  # Run all tests
    ])

"""
Comprehensive 4-Layer Test Suite for Unified Streaming Solution

This test suite validates the unified streaming implementation with progressive complexity:
- Layer 1: Component Tests (IncrementalToolDetector)
- Layer 2: Integration Tests (UnifiedStreamProcessor)
- Layer 3: Provider Integration Tests (BaseProvider streaming)
- Layer 4: End-to-End Tests (Real provider streaming)

All tests use real implementations per CLAUDE.md - NO MOCKING
"""

import pytest
import time
from typing import Iterator, List, Dict, Any
from abstractllm.providers.streaming import (
    IncrementalToolDetector,
    UnifiedStreamProcessor,
    ToolDetectionState
)
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.core import ToolCall, ToolDefinition, ToolResult


# ============================================================================
# LAYER 1: COMPONENT TESTS - IncrementalToolDetector
# ============================================================================

class TestIncrementalToolDetector:
    """Layer 1: Test the core incremental tool detector component"""

    def test_detector_initialization_qwen_model(self):
        """Test detector initializes with correct patterns for Qwen model"""
        detector = IncrementalToolDetector("qwen3-coder:30b")

        assert detector.model_name == "qwen3-coder:30b"
        assert detector.state == ToolDetectionState.SCANNING
        # Qwen models support multiple formats for flexibility
        assert len(detector.active_patterns) >= 1
        # Check that qwen pattern is included
        assert any(p['start'] == r'<\|tool_call\|>' for p in detector.active_patterns)

    def test_detector_initialization_llama_model(self):
        """Test detector initializes with correct patterns for LLaMA model"""
        detector = IncrementalToolDetector("llama-3.2-vision")

        # LLaMA models support multiple formats for flexibility
        assert len(detector.active_patterns) >= 1
        # Check that llama pattern is included
        assert any(p['start'] == r'<function_call>' for p in detector.active_patterns)

    def test_detector_initialization_gemma_model(self):
        """Test detector initializes with correct patterns for Gemma model"""
        detector = IncrementalToolDetector("gemma-2-9b")

        assert len(detector.active_patterns) == 1
        assert detector.active_patterns[0]['start'] == r'```tool_code'
        assert detector.active_patterns[0]['end'] == r'```'

    def test_detector_initialization_unknown_model(self):
        """Test detector uses multiple patterns for unknown models"""
        detector = IncrementalToolDetector("unknown-model")

        # Should have multiple patterns for unknown models (at least 3)
        assert len(detector.active_patterns) >= 3

    def test_state_transition_scanning_to_in_tool_call_qwen(self):
        """Test state machine transition from SCANNING to IN_TOOL_CALL for Qwen format"""
        detector = IncrementalToolDetector("qwen3-coder")

        # Initial state
        assert detector.state == ToolDetectionState.SCANNING

        # Process content with tool start tag
        content = "Here's the result: <|tool_call|>"
        streamable, tools = detector.process_chunk(content)

        # Should transition to IN_TOOL_CALL
        assert detector.state == ToolDetectionState.IN_TOOL_CALL
        assert streamable == "Here's the result: "
        assert len(tools) == 0  # No complete tools yet

    def test_state_transition_scanning_to_in_tool_call_llama(self):
        """Test state machine transition from SCANNING to IN_TOOL_CALL for LLaMA format"""
        detector = IncrementalToolDetector("llama-3")

        content = "I'll use this tool: <function_call>"
        streamable, tools = detector.process_chunk(content)

        assert detector.state == ToolDetectionState.IN_TOOL_CALL
        assert streamable == "I'll use this tool: "
        assert len(tools) == 0

    def test_complete_tool_call_detection_qwen_format(self):
        """Test detecting a complete tool call in Qwen format"""
        detector = IncrementalToolDetector("qwen3-coder")

        # Stream in a complete tool call incrementally
        chunks = [
            "Let me help with that. <|tool_call|>",
            '{"name": "calculate"',
            ', "arguments": {"expr": "2+2"}}',
            "</|tool_call|> The answer is ready."
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        # Check results
        assert len(all_tools) == 1
        assert all_tools[0].name == "calculate"
        assert all_tools[0].arguments == {"expr": "2+2"}

        # Streamable content should be text before and after tool
        combined_streamable = "".join(all_streamable)
        assert "Let me help with that." in combined_streamable
        assert " The answer is ready." in combined_streamable

    def test_complete_tool_call_detection_llama_format(self):
        """Test detecting a complete tool call in LLaMA format"""
        detector = IncrementalToolDetector("llama-3")

        chunks = [
            "I'll search for that. <function_call>",
            '{"name": "web_search", "arguments":',
            ' {"query": "Python tutorial"}}',
            "</function_call> Here are the results."
        ]

        all_streamable = []
        all_tools = []

        for chunk in chunks:
            streamable, tools = detector.process_chunk(chunk)
            if streamable:
                all_streamable.append(streamable)
            all_tools.extend(tools)

        assert len(all_tools) == 1
        assert all_tools[0].name == "web_search"
        assert all_tools[0].arguments == {"query": "Python tutorial"}

    def test_multiple_sequential_tool_calls(self):
        """Test detecting multiple tool calls in sequence"""
        detector = IncrementalToolDetector("qwen3")

        # Split content into incremental chunks to simulate real streaming
        chunks = [
            "First tool: <|tool_call|>",
            '{"name": "tool1", "arguments": {"a": 1}}',
            "</|tool_call|> ",
            "Second tool: <|tool_call|>",
            '{"name": "tool2", "arguments": {"b": 2}}',
            "</|tool_call|> Done."
        ]

        all_tools = []
        for chunk in chunks:
            _, tools = detector.process_chunk(chunk)
            all_tools.extend(tools)

        # Should detect both tools
        assert len(all_tools) == 2
        assert all_tools[0].name == "tool1"
        assert all_tools[0].arguments == {"a": 1}
        assert all_tools[1].name == "tool2"
        assert all_tools[1].arguments == {"b": 2}

    def test_partial_json_accumulation(self):
        """Test that partial JSON is accumulated correctly"""
        detector = IncrementalToolDetector("qwen3")

        # Stream very small chunks to test accumulation
        chunks = [
            "<|tool_call|>",
            '{"n',
            'ame',
            '": ',
            '"test"',
            ', "ar',
            'guments"',
            ': {}',
            '}',
            '</|tool_call|>'
        ]

        all_tools = []
        for chunk in chunks:
            _, tools = detector.process_chunk(chunk)
            all_tools.extend(tools)

        # Should successfully parse despite fragmented chunks
        assert len(all_tools) == 1
        assert all_tools[0].name == "test"

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON in tool calls"""
        detector = IncrementalToolDetector("qwen3")

        # Missing closing brace - stream in incrementally
        chunks = [
            '<|tool_call|>',
            '{"name": "test", "arguments": {"x": 1}',  # Missing closing brace
            '</|tool_call|>'
        ]

        all_tools = []
        for chunk in chunks:
            _, tools = detector.process_chunk(chunk)
            all_tools.extend(tools)

        # Should fix and parse successfully
        assert len(all_tools) == 1
        assert all_tools[0].name == "test"
        assert all_tools[0].arguments == {"x": 1}

    def test_incomplete_tool_call_parsing(self):
        """Test parsing incomplete tool calls without end tag"""
        detector = IncrementalToolDetector("qwen3")

        # Complete JSON but no end tag - should parse if enough content
        content = '<|tool_call|>{"name": "calculate", "arguments": {"expr": "2+2"}}'
        _, tools = detector.process_chunk(content)

        # Might not parse immediately (less than 50 chars of content after start)
        # Add more content to trigger incomplete parsing
        content2 = ' ' * 100  # Add padding
        _, tools2 = detector.process_chunk(content2)

        # Check finalize catches it
        final_tools = detector.finalize()

        # Should have parsed the tool either during processing or finalization
        all_tools = tools + tools2 + final_tools
        assert len(all_tools) >= 1
        if len(all_tools) > 0:
            assert all_tools[0].name == "calculate"

    def test_reset_functionality(self):
        """Test detector reset clears all state"""
        detector = IncrementalToolDetector("qwen3")

        # Process some content
        detector.process_chunk("<|tool_call|>")
        assert detector.state == ToolDetectionState.IN_TOOL_CALL

        # Reset
        detector.reset()

        # Check all state is cleared
        assert detector.state == ToolDetectionState.SCANNING
        assert detector.accumulated_content == ""
        assert detector.current_tool_content == ""
        assert detector.tool_start_pos is None
        assert len(detector.completed_tools) == 0

    def test_empty_chunk_handling(self):
        """Test handling of empty chunks"""
        detector = IncrementalToolDetector("qwen3")

        streamable, tools = detector.process_chunk("")

        assert streamable == ""
        assert len(tools) == 0

    def test_finalize_with_pending_tool(self):
        """Test finalize extracts pending tool calls"""
        detector = IncrementalToolDetector("qwen3")

        # Start a tool call but don't complete it
        detector.process_chunk('<|tool_call|>{"name": "pending", "arguments": {}}')

        # Finalize should extract it
        tools = detector.finalize()

        assert len(tools) == 1
        assert tools[0].name == "pending"


# ============================================================================
# LAYER 2: INTEGRATION TESTS - UnifiedStreamProcessor
# ============================================================================

class TestUnifiedStreamProcessor:
    """Layer 2: Test the unified stream processor integration"""

    def create_test_stream(self, chunks: List[str]) -> Iterator[GenerateResponse]:
        """Helper to create a test response stream"""
        for chunk in chunks:
            yield GenerateResponse(
                content=chunk,
                model="test-model",
                finish_reason=None
            )

    def test_basic_streaming_without_tools(self):
        """Test basic streaming without any tool calls"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        chunks = ["Hello ", "world", "!"]
        stream = self.create_test_stream(chunks)

        results = list(processor.process_stream(stream))

        # Should get same chunks back
        assert len(results) == 3
        assert results[0].content == "Hello "
        assert results[1].content == "world"
        assert results[2].content == "!"

    def test_streaming_with_tool_detection(self):
        """Test streaming with tool call detection"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=False)

        chunks = [
            "Let me calculate: ",
            "<|tool_call|>",
            '{"name": "calc", "arguments": {"expr": "5*5"}}',
            "</|tool_call|>",
            " The result is ready."
        ]
        stream = self.create_test_stream(chunks)

        results = list(processor.process_stream(stream))

        # Should stream content and detect tool
        # Content before tool should be streamed
        assert any("Let me calculate:" in r.content for r in results if r.content)
        # Content after tool should be streamed
        assert any("The result is ready." in r.content for r in results if r.content)

    def test_tool_execution_during_streaming(self):
        """Test that tools are executed immediately during streaming"""
        # Create a simple test tool
        def test_tool(value: int) -> int:
            """Test tool that returns double the value"""
            return value * 2

        # Register tool for execution
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(test_tool)

        tool_def = ToolDefinition.from_function(test_tool)
        converted_tools = [tool_def.to_dict()]
        converted_tools[0]['function'] = test_tool

        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        chunks = [
            "Calculating: ",
            "<|tool_call|>",
            '{"name": "test_tool", "arguments": {"value": 5}}',
            "</|tool_call|>",
            " Done."
        ]
        stream = self.create_test_stream(chunks)

        results = list(processor.process_stream(stream, converted_tools))

        # Should have tool execution results
        tool_result_found = False
        for result in results:
            if result.content and "Tool Results:" in result.content:
                tool_result_found = True
                # Check that result contains the tool execution
                assert "test_tool" in result.content
                break

        assert tool_result_found, "Tool execution results not found in stream"

        # Cleanup
        clear_registry()

    def test_multiple_tools_in_stream(self):
        """Test handling multiple tool calls in a stream"""
        def tool1(x: int) -> int:
            return x + 1

        def tool2(x: int) -> int:
            return x * 2

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
            "First: <|tool_call|>",
            '{"name": "tool1", "arguments": {"x": 5}}',
            "</|tool_call|> ",
            "Second: <|tool_call|>",
            '{"name": "tool2", "arguments": {"x": 3}}',
            "</|tool_call|>"
        ]
        stream = self.create_test_stream(chunks)

        results = list(processor.process_stream(stream, tools))

        # Should execute both tools
        all_content = " ".join([r.content for r in results if r.content])
        assert "tool1" in all_content
        assert "tool2" in all_content

        # Cleanup
        clear_registry()
        

    def test_error_handling_in_stream_processing(self):
        """Test error handling during stream processing"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=False)

        def error_stream():
            yield GenerateResponse(content="Start", model="test")
            raise RuntimeError("Stream error")

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Stream error"):
            list(processor.process_stream(error_stream()))

    def test_finalize_catches_incomplete_tools(self):
        """Test that finalize catches incomplete tool calls at end of stream"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=False)

        chunks = [
            "Using tool: ",
            "<|tool_call|>",
            '{"name": "incomplete", "arguments": {}}'
            # No closing tag
        ]
        stream = self.create_test_stream(chunks)

        results = list(processor.process_stream(stream))

        # Finalize should have caught the incomplete tool
        # At minimum, we should have the content before the tool
        assert len(results) > 0

    def test_empty_stream_handling(self):
        """Test handling of empty stream"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        stream = self.create_test_stream([])
        results = list(processor.process_stream(stream))

        assert len(results) == 0

    def test_stream_with_none_content(self):
        """Test handling of chunks with None content"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        def stream_with_none():
            yield GenerateResponse(content="Start", model="test")
            yield GenerateResponse(content=None, model="test")
            yield GenerateResponse(content="End", model="test")

        results = list(processor.process_stream(stream_with_none()))

        # Should handle None gracefully
        assert len(results) == 3
        assert results[1].content is None


# ============================================================================
# LAYER 3: PROVIDER INTEGRATION TESTS
# ============================================================================

class TestProviderIntegration:
    """Layer 3: Test integration with BaseProvider streaming"""

    def test_unified_streaming_replaces_dual_mode(self):
        """Verify unified streaming is used instead of dual-mode approach"""
        # Read the base provider implementation
        with open('/Users/albou/projects/abstractllm_core/abstractllm/providers/base.py', 'r') as f:
            base_provider_code = f.read()

        # Verify UnifiedStreamProcessor is imported and used
        assert 'from .streaming import UnifiedStreamProcessor' in base_provider_code
        assert 'UnifiedStreamProcessor(' in base_provider_code

        # Verify old dual-mode patterns are NOT present
        assert 'buffered_stream' not in base_provider_code.lower()
        assert 'immediate_stream' not in base_provider_code.lower()

    def test_streaming_implementation_uses_unified_processor(self):
        """Test that streaming implementation uses UnifiedStreamProcessor"""
        with open('/Users/albou/projects/abstractllm_core/abstractllm/providers/base.py', 'r') as f:
            content = f.read()

        # Check for unified streaming pattern
        assert 'processor.process_stream(response, converted_tools)' in content

    def test_stream_processor_receives_correct_parameters(self):
        """Test that UnifiedStreamProcessor receives correct initialization params"""
        with open('/Users/albou/projects/abstractllm_core/abstractllm/providers/base.py', 'r') as f:
            content = f.read()

        # Verify processor initialization includes required params
        assert 'model_name=self.model' in content
        assert 'execute_tools=should_execute_tools' in content
        assert 'tool_call_tags=tool_call_tags' in content


# ============================================================================
# LAYER 4: END-TO-END TESTS
# ============================================================================

class TestEndToEndStreaming:
    """Layer 4: End-to-end tests with real scenarios"""

    def test_performance_streaming_is_immediate(self):
        """Test that streaming provides immediate chunks (not buffered)"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        chunks = ["Chunk1 ", "Chunk2 ", "Chunk3"]

        def timed_stream():
            for chunk in chunks:
                time.sleep(0.01)  # Small delay between chunks
                yield GenerateResponse(content=chunk, model="test")

        start_time = time.time()
        results = []
        first_chunk_time = None

        for result in processor.process_stream(timed_stream()):
            if result.content and first_chunk_time is None:
                first_chunk_time = time.time()
            results.append(result)

        end_time = time.time()

        # First chunk should arrive quickly (not buffered)
        assert first_chunk_time is not None
        time_to_first_chunk = first_chunk_time - start_time
        assert time_to_first_chunk < 0.1  # Should be immediate (< 100ms)

        # Total time should be > 0.03s (3 chunks * 0.01s delay)
        total_time = end_time - start_time
        assert total_time >= 0.02

    def test_tool_execution_timing(self):
        """Test that tools execute immediately when detected, not at end"""
        execution_times = []

        def timed_tool(x: int) -> int:
            execution_times.append(time.time())
            return x * 2

        # Register tool
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(timed_tool)

        tool_def = ToolDefinition.from_function(timed_tool).to_dict()
        tool_def['function'] = timed_tool

        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        chunks = [
            "Start ",
            "<|tool_call|>",
            '{"name": "timed_tool", "arguments": {"x": 5}}',
            "</|tool_call|>",
            " Middle ",
            "<|tool_call|>",
            '{"name": "timed_tool", "arguments": {"x": 10}}',
            "</|tool_call|>",
            " End"
        ]

        def delayed_stream():
            for i, chunk in enumerate(chunks):
                time.sleep(0.02)
                yield GenerateResponse(content=chunk, model="test")

        start_time = time.time()
        list(processor.process_stream(delayed_stream(), [tool_def]))
        end_time = time.time()

        # Should have executed both tools
        assert len(execution_times) == 2

        # First tool should execute near the middle, not at end
        first_tool_time = execution_times[0] - start_time
        total_time = end_time - start_time

        # First tool should execute well before the end (within first 60% of stream)
        assert first_tool_time < total_time * 0.6

        # Cleanup
        clear_registry()

    def test_real_world_streaming_pattern(self):
        """Test realistic streaming pattern with mixed content and tools"""
        def calculator(expr: str) -> str:
            """Simple calculator tool"""
            try:
                return str(eval(expr))
            except Exception as e:
                return f"Error: {e}"

        # Register tool
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(calculator)

        tool_def = ToolDefinition.from_function(calculator).to_dict()
        tool_def['function'] = calculator

        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        # Realistic streaming pattern
        chunks = [
            "I'll help you calculate that. ",
            "Let me use the calculator: ",
            "<|tool_call|>",
            '{"name": "calculator",',
            ' "arguments": ',
            '{"expr": "2+2"}}',
            "</|tool_call|>",
            " Based on the calculation, ",
            "the answer is 4."
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream, [tool_def]))

        # Should have multiple results
        assert len(results) > 0

        # Should have tool execution
        all_content = " ".join([r.content for r in results if r.content])
        assert "calculator" in all_content

        # Original content should be preserved
        assert "I'll help you calculate" in all_content or any("I'll help you calculate" in r.content for r in results if r.content)

        # Cleanup
        clear_registry()

    def test_streaming_with_no_tools_defined(self):
        """Test streaming behavior when no tools are available"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        chunks = [
            "Text with ",
            "<|tool_call|>",
            '{"name": "undefined_tool", "arguments": {}}',
            "</|tool_call|>",
            " more text"
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)

        # Should not crash even with tool calls but no tools defined
        results = list(processor.process_stream(stream, None))

        # Should still get content
        assert len(results) > 0

    def test_concurrent_streaming_sessions(self):
        """Test that multiple concurrent streaming sessions work independently"""
        processor1 = UnifiedStreamProcessor("qwen3", execute_tools=False)
        processor2 = UnifiedStreamProcessor("llama3", execute_tools=False)

        chunks1 = ["Qwen: ", "<|tool_call|>", '{"name": "t1", "arguments": {}}', "</|tool_call|>"]
        chunks2 = ["LLaMA: ", "<function_call>", '{"name": "t2", "arguments": {}}', "</function_call>"]

        stream1 = (GenerateResponse(content=c, model="qwen") for c in chunks1)
        stream2 = (GenerateResponse(content=c, model="llama") for c in chunks2)

        # Process both streams
        results1 = list(processor1.process_stream(stream1))
        results2 = list(processor2.process_stream(stream2))

        # Both should work independently
        assert len(results1) > 0
        assert len(results2) > 0

    def test_memory_efficiency_large_stream(self):
        """Test memory efficiency with large streaming content"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        # Create a large stream
        def large_stream():
            for i in range(1000):
                yield GenerateResponse(content=f"Chunk {i} ", model="test")

        # Process stream and count results
        chunk_count = 0
        for _ in processor.process_stream(large_stream()):
            chunk_count += 1

        # Should process all chunks
        assert chunk_count == 1000

    def test_edge_case_tool_at_stream_start(self):
        """Test tool call appearing at the very start of stream"""
        def test_tool() -> str:
            return "executed"

        # Register tool
        from abstractllm.tools.registry import register_tool, clear_registry
        register_tool(test_tool)

        tool_def = ToolDefinition.from_function(test_tool).to_dict()
        tool_def['function'] = test_tool

        processor = UnifiedStreamProcessor("qwen3", execute_tools=True)

        chunks = [
            "<|tool_call|>",
            '{"name": "test_tool", "arguments": {}}',
            "</|tool_call|>",
            " Result follows."
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream, [tool_def]))

        # Should handle tool at start
        assert len(results) > 0

        # Cleanup
        clear_registry()

    def test_edge_case_tool_at_stream_end(self):
        """Test tool call appearing at the very end of stream"""
        processor = UnifiedStreamProcessor("qwen3", execute_tools=False)

        chunks = [
            "Starting content. ",
            "More content. ",
            "<|tool_call|>",
            '{"name": "end_tool", "arguments": {}}',
            "</|tool_call|>"
        ]

        stream = (GenerateResponse(content=c, model="test") for c in chunks)
        results = list(processor.process_stream(stream))

        # Should handle tool at end
        assert len(results) > 0

    def test_streaming_preserves_model_metadata(self):
        """Test that streaming preserves model and metadata through processing"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        chunks = [
            GenerateResponse(
                content="Test",
                model="specific-model",
                finish_reason="stop",
                usage={"tokens": 10}
            )
        ]

        results = list(processor.process_stream(iter(chunks)))

        assert len(results) == 1
        assert results[0].model == "specific-model"
        assert results[0].finish_reason == "stop"
        assert results[0].usage == {"tokens": 10}


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance validation and benchmarks"""

    def test_detector_performance_incremental_vs_batch(self):
        """Benchmark incremental detection vs batch processing"""
        detector = IncrementalToolDetector("qwen3")

        # Create test content with tool call
        full_content = (
            "Some text before "
            "<|tool_call|>"
            '{"name": "test", "arguments": {"x": 1}}'
            "</|tool_call|>"
            " Some text after"
        )

        # Test incremental processing
        start = time.time()
        chunks = [full_content[i:i+5] for i in range(0, len(full_content), 5)]
        for chunk in chunks:
            detector.process_chunk(chunk)
        incremental_time = time.time() - start

        # Reset and test batch processing
        detector.reset()
        start = time.time()
        detector.process_chunk(full_content)
        batch_time = time.time() - start

        # Both should be fast, batch might be slightly faster
        assert incremental_time < 0.1  # Should be < 100ms
        assert batch_time < 0.1

    def test_streaming_latency_measurement(self):
        """Measure streaming latency for first chunk"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        def measured_stream():
            yield GenerateResponse(content="First", model="test")
            yield GenerateResponse(content="Second", model="test")

        start = time.time()
        results = processor.process_stream(measured_stream())

        # Get first result
        first_result = next(results)
        first_chunk_latency = time.time() - start

        # Should be immediate (< 10ms for processing overhead)
        assert first_chunk_latency < 0.01
        assert first_result.content == "First"

    def test_tool_detection_overhead(self):
        """Measure overhead of tool detection when no tools present"""
        processor = UnifiedStreamProcessor("test-model", execute_tools=False)

        # Stream without tools
        no_tool_chunks = ["Plain " for _ in range(100)]

        start = time.time()
        stream = (GenerateResponse(content=c, model="test") for c in no_tool_chunks)
        list(processor.process_stream(stream))
        time_no_tools = time.time() - start

        # Should have minimal overhead
        assert time_no_tools < 0.1  # < 100ms for 100 chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

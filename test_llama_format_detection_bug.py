#!/usr/bin/env python3
"""
Test to reproduce the critical bug where <function_call> tags appear in output
when they should be detected and converted.

Issue: User reports that tool calls work initially but later in the same session,
<function_call> tags appear in the output without being converted/executed.
"""

import sys
sys.path.insert(0, '/Users/albou/projects/abstractllm_core')

from abstractllm.providers.streaming import IncrementalToolDetector, UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.core import ToolCall

def test_llama_format_detection():
    """Test that <function_call> format is consistently detected"""

    print("=" * 80)
    print("TESTING LLAMA FORMAT DETECTION CONSISTENCY")
    print("=" * 80)

    # Test 1: Single tool call in one chunk (usually works)
    print("\nTest 1: Single chunk with complete tool call")
    detector = IncrementalToolDetector("llama-3")

    chunk1 = 'Let me help you. <function_call>{"name": "list_files", "arguments": {"path": "/tmp"}}</function_call> Done.'
    streamable, tools = detector.process_chunk(chunk1)

    print(f"Input: {chunk1}")
    print(f"Streamable content: {streamable}")
    print(f"Tools detected: {tools}")
    print(f"✓ Tool detected: {len(tools) > 0}")
    print(f"✓ No tags in output: {'<function_call>' not in streamable}")

    # Test 2: Tool call split across multiple chunks (might fail)
    print("\n" + "-" * 40)
    print("\nTest 2: Tool call split across chunks")
    detector.reset()

    chunks = [
        'I will execute that. ',
        '<function_call>',
        '{"name": "shell", ',
        '"arguments": {"command": "ls -la"}}',
        '</function_call>',
        ' The command has been executed.'
    ]

    all_streamable = []
    all_tools = []

    print("Processing chunks:")
    for i, chunk in enumerate(chunks):
        streamable, tools = detector.process_chunk(chunk)
        print(f"  Chunk {i}: '{chunk}' -> streamable: '{streamable}', tools: {tools}")
        if streamable:
            all_streamable.append(streamable)
        all_tools.extend(tools)

    combined_output = "".join(all_streamable)
    print(f"\nCombined output: {combined_output}")
    print(f"Tools detected: {all_tools}")
    print(f"✓ Tool detected: {len(all_tools) > 0}")
    print(f"✓ No tags in output: {'<function_call>' not in combined_output}")

    # Test 3: Multiple tool calls in sequence (inconsistent behavior)
    print("\n" + "-" * 40)
    print("\nTest 3: Multiple tool calls in sequence")
    detector.reset()

    content = '''First, let me check the files.
<function_call>{"name": "list_files", "arguments": {"path": "."}}</function_call>
Now I'll read one of them.
<function_call>{"name": "read_file", "arguments": {"file": "README.md"}}</function_call>
Finally, I'll analyze the content.
<function_call>{"name": "analyze", "arguments": {"text": "content"}}</function_call>
Done with all operations.'''

    # Process in realistic chunks (simulating streaming)
    chunk_size = 50
    all_streamable = []
    all_tools = []

    print(f"Processing content in {chunk_size}-char chunks:")
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        streamable, tools = detector.process_chunk(chunk)

        if streamable:
            all_streamable.append(streamable)
        all_tools.extend(tools)

        if tools:
            print(f"  Tool detected at position {i}: {tools[0].name}")

    combined_output = "".join(all_streamable)
    print(f"\nCombined output: {combined_output}")
    print(f"Tools detected: {[t.name for t in all_tools]}")
    print(f"✓ All 3 tools detected: {len(all_tools) == 3}")
    print(f"✓ No tags in output: {'<function_call>' not in combined_output}")

    # Test 4: Character-by-character streaming (stress test)
    print("\n" + "-" * 40)
    print("\nTest 4: Character-by-character streaming (worst case)")
    detector.reset()

    content = 'Let me help. <function_call>{"name": "test", "arguments": {}}</function_call> Done.'

    all_streamable = []
    all_tools = []

    print("Processing character by character:")
    for char in content:
        streamable, tools = detector.process_chunk(char)
        if streamable:
            all_streamable.append(streamable)
        all_tools.extend(tools)

    combined_output = "".join(all_streamable)
    print(f"Combined output: {combined_output}")
    print(f"Tools detected: {all_tools}")
    print(f"✓ Tool detected: {len(all_tools) > 0}")
    print(f"✓ No tags in output: {'<function_call>' not in combined_output}")

    # Test 5: With tag rewriting enabled
    print("\n" + "-" * 40)
    print("\nTest 5: With tag rewriting to custom format")

    processor = UnifiedStreamProcessor(
        model_name="llama-3",
        execute_tools=False,
        tool_call_tags="CUSTOM_START,CUSTOM_END"
    )

    chunks = [
        'Processing your request. ',
        '<function_call>',
        '{"name": "execute", "arguments": {"cmd": "test"}}',
        '</function_call>',
        ' Task completed.'
    ]

    def mock_stream():
        for chunk in chunks:
            yield GenerateResponse(content=chunk, model="llama-3")

    results = list(processor.process_stream(mock_stream()))
    full_output = "".join([r.content for r in results if r.content])

    print(f"Input chunks: {chunks}")
    print(f"Output: {full_output}")
    print(f"✓ Custom tags applied: {'CUSTOM_START' in full_output}")
    print(f"✓ Original tags removed: {'<function_call>' not in full_output}")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    issues_found = []

    # Check for potential issues
    if '<function_call>' in combined_output:
        issues_found.append("❌ Tool tags leaked to output in some tests")

    if len(all_tools) < 3:
        issues_found.append("❌ Not all tools were detected in multi-tool test")

    if not issues_found:
        print("✅ All tests passed - tool detection working correctly")
    else:
        print("Issues found:")
        for issue in issues_found:
            print(f"  {issue}")

    return len(issues_found) == 0

if __name__ == "__main__":
    success = test_llama_format_detection()
    sys.exit(0 if success else 1)
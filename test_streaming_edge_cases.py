#!/usr/bin/env python3
"""
Test edge cases for the new streaming implementation.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abstractllm.tools.tag_rewriter import create_tag_rewriter

def test_streaming_edge_cases():
    """Test edge cases for streaming."""
    print("🔧 Testing Streaming Edge Cases")
    print("=" * 50)
    
    # Test 1: Multiple tool calls in sequence
    print("Test 1: Multiple tool calls in sequence")
    print("-" * 40)
    
    rewriter = create_tag_rewriter("llama3")
    buffer = ""
    
    chunks = [
        "I'll help you with two tasks.\n\n<function_call>\n",
        '{"name": "get_weather", "arguments": {"location": "Paris"}}\n',
        "</function_call>\n\nNow let me calculate something.\n\n<function_call>\n",
        '{"name": "calculate", "arguments": {"expression": "2+2"}}\n',
        "</function_call>\n\nDone!"
    ]
    
    full_output = ""
    for i, chunk in enumerate(chunks):
        rewritten_chunk, buffer = rewriter.rewrite_streaming_chunk(chunk, buffer)
        full_output += rewritten_chunk
        print(f"Chunk {i+1}: {repr(chunk)}")
        print(f"Rewritten: {repr(rewritten_chunk)}")
        print(f"Buffer: {repr(buffer)}")
        print()
    
    start_count = full_output.count("<function_call>")
    end_count = full_output.count("</function_call>")
    print(f"Start tags: {start_count}, End tags: {end_count}")
    print(f"Status: {'✅ PASS' if start_count == end_count == 2 else '❌ FAIL'}")
    
    # Test 2: Tool call split across many small chunks
    print("\nTest 2: Tool call split across many small chunks")
    print("-" * 50)
    
    rewriter = create_tag_rewriter("qwen3")
    buffer = ""
    
    # Split a tool call into very small chunks
    tool_call = '<|tool_call|>\n{"name": "test", "arguments": {"param": "value"}}\n</|tool_call|>'
    chunks = [char for char in tool_call]
    
    full_output = ""
    for i, chunk in enumerate(chunks):
        rewritten_chunk, buffer = rewriter.rewrite_streaming_chunk(chunk, buffer)
        full_output += rewritten_chunk
    
    start_count = full_output.count("<|tool_call|>")
    end_count = full_output.count("</|tool_call|>")
    print(f"Start tags: {start_count}, End tags: {end_count}")
    print(f"Status: {'✅ PASS' if start_count == end_count == 1 else '❌ FAIL'}")
    
    # Test 3: Mixed content with tool calls
    print("\nTest 3: Mixed content with tool calls")
    print("-" * 40)
    
    rewriter = create_tag_rewriter("xml")
    buffer = ""
    
    chunks = [
        "Here's some text before the tool call.\n\n<tool_call>\n",
        '{"name": "process", "arguments": {"data": "example"}}\n',
        "</tool_call>\n\nAnd some text after.\n\nMore text here."
    ]
    
    full_output = ""
    for i, chunk in enumerate(chunks):
        rewritten_chunk, buffer = rewriter.rewrite_streaming_chunk(chunk, buffer)
        full_output += rewritten_chunk
    
    start_count = full_output.count("<tool_call>")
    end_count = full_output.count("</tool_call>")
    print(f"Start tags: {start_count}, End tags: {end_count}")
    print(f"Status: {'✅ PASS' if start_count == end_count == 1 else '❌ FAIL'}")
    
    # Test 4: Plain JSON tool call
    print("\nTest 4: Plain JSON tool call")
    print("-" * 30)
    
    rewriter = create_tag_rewriter("llama3")
    buffer = ""
    
    chunks = [
        "I'll use a tool: ",
        '{"name": "simple_tool", "arguments": {}}',
        " and continue."
    ]
    
    full_output = ""
    for i, chunk in enumerate(chunks):
        rewritten_chunk, buffer = rewriter.rewrite_streaming_chunk(chunk, buffer)
        full_output += rewritten_chunk
    
    start_count = full_output.count("<function_call>")
    end_count = full_output.count("</function_call>")
    print(f"Start tags: {start_count}, End tags: {end_count}")
    print(f"Status: {'✅ PASS' if start_count == end_count == 1 else '❌ FAIL'}")

if __name__ == "__main__":
    test_streaming_edge_cases()
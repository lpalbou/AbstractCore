#!/usr/bin/env python3
"""
Test the new SOTA streaming implementation.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abstractllm.tools.tag_rewriter import create_tag_rewriter

def test_new_streaming():
    """Test the new SOTA streaming implementation."""
    print("🔧 Testing New SOTA Streaming Implementation")
    print("=" * 60)
    
    # Test case that previously caused double tags
    rewriter = create_tag_rewriter("llama3")
    buffer = ""
    
    # Simulate streaming chunks
    chunks = [
        "I'll help you get the weather information.\n\n<function_call>\n",
        '{"name": "get_weather", "arguments": {"location": "Paris"}}\n',
        "</function_call>\n\nLet me check the weather for Paris."
    ]
    
    print("Chunk-by-chunk processing:")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {repr(chunk)}")
        rewritten_chunk, buffer = rewriter.rewrite_streaming_chunk(chunk, buffer)
        print(f"Rewritten: {repr(rewritten_chunk)}")
        if buffer:
            print(f"Buffer: {repr(buffer)}")
        print()
    
    print(f"Final buffer: {repr(buffer)}")
    
    # Check for double tags
    full_output = ""
    test_buffer = ""
    for chunk in chunks:
        rewritten_chunk, test_buffer = rewriter.rewrite_streaming_chunk(chunk, test_buffer)
        full_output += rewritten_chunk
    
    print(f"\nFull output: {repr(full_output)}")
    
    # Count occurrences of start and end tags
    start_count = full_output.count("<function_call>")
    end_count = full_output.count("</function_call>")
    
    print(f"\nTag counts:")
    print(f"Start tags: {start_count}")
    print(f"End tags: {end_count}")
    
    if start_count == end_count and start_count == 1:
        print("✅ SUCCESS: No double tags detected!")
        return True
    else:
        print("❌ FAILURE: Double tags detected!")
        return False

def test_complete_tool_call_detection():
    """Test complete tool call detection."""
    print("\n🔍 Testing Complete Tool Call Detection")
    print("=" * 50)
    
    rewriter = create_tag_rewriter("llama3")
    
    test_cases = [
        {
            "name": "Complete Qwen3 tool call",
            "text": '<|tool_call|>\n{"name": "test", "arguments": {}}\n</|tool_call|>',
            "expected": True
        },
        {
            "name": "Incomplete Qwen3 tool call",
            "text": '<|tool_call|>\n{"name": "test", "arguments": {}}',
            "expected": False
        },
        {
            "name": "Complete LLaMA3 tool call",
            "text": '<function_call>\n{"name": "test", "arguments": {}}\n</function_call>',
            "expected": True
        },
        {
            "name": "Incomplete LLaMA3 tool call",
            "text": '<function_call>\n{"name": "test", "arguments": {}}',
            "expected": False
        },
        {
            "name": "Plain JSON tool call",
            "text": '{"name": "test", "arguments": {}}',
            "expected": True
        },
        {
            "name": "Regular text",
            "text": "Just some regular text without tool calls.",
            "expected": False
        }
    ]
    
    for test_case in test_cases:
        result = rewriter._has_complete_tool_call(test_case["text"])
        status = "✅ PASS" if result == test_case["expected"] else "❌ FAIL"
        print(f"{test_case['name']}: {result} (expected {test_case['expected']}) {status}")

if __name__ == "__main__":
    success1 = test_new_streaming()
    test_complete_tool_call_detection()
    print(f"\nOverall test result: {'PASS' if success1 else 'FAIL'}")
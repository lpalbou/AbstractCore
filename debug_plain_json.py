#!/usr/bin/env python3
"""
Debug plain JSON tool call detection.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abstractllm.tools.tag_rewriter import create_tag_rewriter

def debug_plain_json():
    """Debug plain JSON tool call detection."""
    print("🔍 Debugging Plain JSON Tool Call Detection")
    print("=" * 50)
    
    rewriter = create_tag_rewriter("llama3")
    
    # Test the plain JSON detection
    test_text = '{"name": "simple_tool", "arguments": {}}'
    print(f"Test text: {repr(test_text)}")
    
    # Test individual methods
    is_plain_json = rewriter._is_plain_json_tool_call(test_text)
    print(f"_is_plain_json_tool_call: {is_plain_json}")
    
    has_complete = rewriter._has_complete_tool_call(test_text)
    print(f"_has_complete_tool_call: {has_complete}")
    
    # Test streaming with plain JSON
    print("\nTesting streaming with plain JSON:")
    print("-" * 40)
    
    buffer = ""
    chunks = [
        "I'll use a tool: ",
        '{"name": "simple_tool", "arguments": {}}',
        " and continue."
    ]
    
    full_output = ""
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {repr(chunk)}")
        rewritten_chunk, buffer = rewriter.rewrite_streaming_chunk(chunk, buffer)
        print(f"Rewritten: {repr(rewritten_chunk)}")
        print(f"Buffer: {repr(buffer)}")
        full_output += rewritten_chunk
        print()
    
    print(f"Full output: {repr(full_output)}")
    
    start_count = full_output.count("<function_call>")
    end_count = full_output.count("</function_call>")
    print(f"Start tags: {start_count}, End tags: {end_count}")

if __name__ == "__main__":
    debug_plain_json()
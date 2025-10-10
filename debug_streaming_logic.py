#!/usr/bin/env python3
"""
Debug the streaming logic step by step.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abstractllm.tools.tag_rewriter import create_tag_rewriter

def debug_streaming_logic():
    """Debug the streaming logic step by step."""
    print("🔍 Debugging Streaming Logic Step by Step")
    print("=" * 50)
    
    rewriter = create_tag_rewriter("llama3")
    
    # Test with a simple plain JSON tool call
    chunk = '{"name": "simple_tool", "arguments": {}}'
    buffer = ""
    
    print(f"Chunk: {repr(chunk)}")
    print(f"Initial buffer: {repr(buffer)}")
    print()
    
    # Simulate the character-by-character processing
    new_buffer = buffer
    rewritten_chunk = ""
    
    for i, char in enumerate(chunk):
        print(f"Character {i}: {repr(char)}")
        new_buffer += char
        print(f"Buffer after adding char: {repr(new_buffer)}")
        
        # Check conditions
        has_complete = rewriter._has_complete_tool_call(new_buffer)
        has_incomplete = rewriter._has_incomplete_tool_call(new_buffer)
        is_plain_json = rewriter._is_plain_json_tool_call(new_buffer.strip())
        
        print(f"  _has_complete_tool_call: {has_complete}")
        print(f"  _has_incomplete_tool_call: {has_incomplete}")
        print(f"  _is_plain_json_tool_call: {is_plain_json}")
        
        if has_complete:
            print("  -> Complete tool call detected, rewriting...")
            rewritten_tool_call = rewriter._rewrite_complete_tool_call(new_buffer)
            rewritten_chunk += rewritten_tool_call
            new_buffer = ""
            print(f"  -> Rewritten: {repr(rewritten_tool_call)}")
        elif has_incomplete:
            print("  -> Incomplete tool call, buffering...")
        elif is_plain_json:
            print("  -> Plain JSON tool call detected, rewriting...")
            rewritten_tool_call = rewriter._rewrite_complete_tool_call(new_buffer.strip())
            rewritten_chunk += rewritten_tool_call
            new_buffer = ""
            print(f"  -> Rewritten: {repr(rewritten_tool_call)}")
        else:
            # Check if we're in the middle of a potential plain JSON tool call
            if (new_buffer.strip().startswith('{') and 
                not new_buffer.strip().endswith('}') and
                '"name"' in new_buffer):
                print("  -> Potential JSON tool call, buffering...")
            else:
                print("  -> No tool call, outputting character...")
                rewritten_chunk += char
                new_buffer = ""
        
        print(f"  -> Rewritten chunk so far: {repr(rewritten_chunk)}")
        print(f"  -> Buffer after processing: {repr(new_buffer)}")
        print()
    
    print(f"Final result:")
    print(f"  Rewritten chunk: {repr(rewritten_chunk)}")
    print(f"  Final buffer: {repr(new_buffer)}")

if __name__ == "__main__":
    debug_streaming_logic()
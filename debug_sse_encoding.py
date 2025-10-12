#!/usr/bin/env python3
"""
Debug the SSE encoding issue specifically.
"""

import json
import uuid

def test_sse_encoding():
    """Test the exact SSE encoding that the server does."""

    print("="*80)
    print("TESTING SSE ENCODING")
    print("="*80)

    # Simulate the tool call data structure that the server creates
    tool_call_data = {
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "shell",
            "arguments": json.dumps({"command": ["ls", "-la"], "workdir": "/tmp"})
        }
    }

    print("1. TOOL CALL DATA STRUCTURE:")
    print(f"   Arguments field: {repr(tool_call_data['function']['arguments'])}")
    print(f"   Arguments type: {type(tool_call_data['function']['arguments'])}")

    # This is what should be a valid JSON string
    args_json = tool_call_data['function']['arguments']
    try:
        parsed_args = json.loads(args_json)
        print(f"   ✅ Arguments parse correctly: {parsed_args}")
    except json.JSONDecodeError as e:
        print(f"   ❌ Arguments don't parse: {e}")

    # Now simulate the server's SSE chunk creation
    tool_call_chunk = {
        "id": "chatcmpl-12345",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "lmstudio/qwen/qwen3-coder-30b",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    **tool_call_data
                }]
            },
            "finish_reason": None
        }]
    }

    print("\n2. SSE CHUNK STRUCTURE:")
    print(json.dumps(tool_call_chunk, indent=2))

    # Now encode this for SSE (what the server does)
    sse_data = json.dumps(tool_call_chunk)
    print(f"\n3. SSE ENCODED DATA:")
    print(f"   Length: {len(sse_data)}")
    print(f"   Data: {sse_data}")

    # Check if the encoded data contains properly escaped arguments
    if '"arguments":' in sse_data:
        # Find the arguments field in the encoded data
        args_start = sse_data.find('"arguments":')
        args_portion = sse_data[args_start:args_start+100]
        print(f"\n4. ARGUMENTS PORTION IN SSE:")
        print(f"   {args_portion}")

        # Check if it contains proper escaping
        if '\\"command\\"' in sse_data:
            print("   ✅ Arguments are properly escaped")
        else:
            print("   ❌ Arguments are NOT properly escaped")

    # Test parsing the SSE data back
    try:
        parsed_sse = json.loads(sse_data)
        tool_calls = parsed_sse["choices"][0]["delta"]["tool_calls"]
        arguments_field = tool_calls[0]["function"]["arguments"]

        print(f"\n5. PARSED BACK FROM SSE:")
        print(f"   Arguments field: {repr(arguments_field)}")

        # Try to parse the arguments
        try:
            final_args = json.loads(arguments_field)
            print(f"   ✅ Final arguments parse correctly: {final_args}")
        except json.JSONDecodeError as e:
            print(f"   ❌ Final arguments don't parse: {e}")

    except json.JSONDecodeError as e:
        print(f"   ❌ SSE data doesn't parse: {e}")

def test_problematic_case():
    """Test the case that's failing according to user's report."""

    print("\n" + "="*80)
    print("TESTING PROBLEMATIC CASE FROM USER REPORT")
    print("="*80)

    # This is what the user reported as the output
    problematic_output = {
        "id": "call_b07ad72d058a492fbf34e688",
        "type": "function",
        "function": {
            "name": "shell",
            "arguments": '{"command": ["ls", "-la"], "workdir": "/tmp"}'  # This looks correct
        }
    }

    print("1. PROBLEMATIC OUTPUT STRUCTURE:")
    print(json.dumps(problematic_output, indent=2))

    # Test if the arguments parse
    try:
        args = json.loads(problematic_output["function"]["arguments"])
        print(f"   ✅ Arguments parse correctly: {args}")
    except json.JSONDecodeError as e:
        print(f"   ❌ Arguments don't parse: {e}")

    # But the user said they're getting unescaped quotes. Let me test that case.
    malformed_output = {
        "id": "call_b07ad72d058a492fbf34e688",
        "type": "function",
        "function": {
            "name": "shell",
            "arguments": '{"command": ["ls", "-la"], "workdir": "/tmp"}'  # Unescaped quotes
        }
    }

    print("\n2. TESTING MALFORMED CASE (what user might be seeing):")

    # This should fail JSON parsing
    try:
        sse_encoded = json.dumps(malformed_output)
        print(f"   SSE encoding: {sse_encoded}")

        # Try to parse it back
        parsed_back = json.loads(sse_encoded)
        args_field = parsed_back["function"]["arguments"]
        print(f"   Arguments field: {repr(args_field)}")

        try:
            args = json.loads(args_field)
            print(f"   ✅ Arguments parse: {args}")
        except json.JSONDecodeError as e:
            print(f"   ❌ Arguments don't parse: {e}")

    except json.JSONDecodeError as e:
        print(f"   ❌ SSE encoding failed: {e}")

if __name__ == "__main__":
    test_sse_encoding()
    test_problematic_case()
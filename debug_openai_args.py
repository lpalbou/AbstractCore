#!/usr/bin/env python3
"""
Debug script to test OpenAI arguments encoding issue.
"""

import json
import uuid
from abstractllm.providers.streaming import UnifiedStreamProcessor, IncrementalToolDetector
from abstractllm.core.types import GenerateResponse

def test_openai_conversion():
    """Test the exact scenario that's failing."""

    # Input content from LLM (Qwen3 format)
    input_content = '''<|tool_call|>
{"name": "shell", "arguments": {"command": ["ls", "-la"], "workdir": "/tmp"}}
</|tool_call|>'''

    print("="*80)
    print("DEBUGGING OPENAI ARGUMENTS ENCODING")
    print("="*80)

    print("\n1. INPUT CONTENT:")
    print(repr(input_content))

    # Test UnifiedStreamProcessor with OpenAI target
    processor = UnifiedStreamProcessor(
        model_name="qwen/qwen3-coder-30b",
        tool_call_tags="openai"
    )

    print(f"\n2. PROCESSOR CONFIGURATION:")
    print(f"   - convert_to_openai_json: {processor.convert_to_openai_json}")
    print(f"   - tag_rewriter: {processor.tag_rewriter}")

    # Create a response stream
    response_stream = [GenerateResponse(content=input_content)]

    print("\n3. PROCESSING STREAM:")

    # Process the stream
    processed_chunks = list(processor.process_stream(iter(response_stream)))

    for i, chunk in enumerate(processed_chunks):
        print(f"   Chunk {i}: {repr(chunk.content)}")

        # If this contains JSON, try to parse it
        if chunk.content and '{' in chunk.content:
            try:
                parsed = json.loads(chunk.content)
                print(f"   Parsed JSON: {json.dumps(parsed, indent=2)}")

                # Check the arguments field specifically
                if "function" in parsed and "arguments" in parsed["function"]:
                    args_field = parsed["function"]["arguments"]
                    print(f"   Arguments field type: {type(args_field)}")
                    print(f"   Arguments field value: {repr(args_field)}")

                    # Try to parse the arguments as JSON
                    try:
                        args_parsed = json.loads(args_field)
                        print(f"   Arguments parsed successfully: {args_parsed}")
                    except json.JSONDecodeError as e:
                        print(f"   ❌ Arguments JSON parsing failed: {e}")
                        print(f"   ❌ This is the bug! Arguments should be valid JSON string")

            except json.JSONDecodeError as e:
                print(f"   JSON parsing failed: {e}")

    print("\n4. TESTING INCREMENTAL DETECTOR:")

    # Also test the detector directly
    detector = IncrementalToolDetector("qwen/qwen3-coder-30b", rewrite_tags=True)

    streamable, tools = detector.process_chunk(input_content)

    print(f"   Streamable content: {repr(streamable)}")
    print(f"   Detected tools: {len(tools)}")

    for tool in tools:
        print(f"   Tool: {tool.name}")
        print(f"   Arguments type: {type(tool.arguments)}")
        print(f"   Arguments value: {repr(tool.arguments)}")

        # Test manual JSON encoding
        if isinstance(tool.arguments, dict):
            json_encoded = json.dumps(tool.arguments)
            print(f"   Manual json.dumps(): {repr(json_encoded)}")

            # Test parsing it back
            try:
                parsed_back = json.loads(json_encoded)
                print(f"   ✅ Round-trip successful: {parsed_back}")
            except json.JSONDecodeError as e:
                print(f"   ❌ Round-trip failed: {e}")

def test_server_logic():
    """Test the server's argument handling logic."""

    print("\n" + "="*80)
    print("TESTING SERVER ARGUMENT HANDLING")
    print("="*80)

    # Simulate ToolCall object from detector
    from abstractllm.tools.core import ToolCall

    # Test case 1: Dict arguments (normal case)
    tool_call = ToolCall(
        name="shell",
        arguments={"command": ["ls", "-la"], "workdir": "/tmp"},
        call_id="call_123"
    )

    print(f"\n1. TOOLCALL WITH DICT ARGUMENTS:")
    print(f"   Arguments type: {type(tool_call.arguments)}")
    print(f"   Arguments value: {repr(tool_call.arguments)}")

    # Apply server logic
    server_args = tool_call.arguments if isinstance(tool_call.arguments, str) else json.dumps(tool_call.arguments)
    print(f"   Server conversion: {repr(server_args)}")

    # Test if it's valid JSON
    try:
        parsed = json.loads(server_args)
        print(f"   ✅ Valid JSON: {parsed}")
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")

    # Test case 2: String arguments (already converted)
    tool_call2 = ToolCall(
        name="shell",
        arguments='{"command": ["ls", "-la"], "workdir": "/tmp"}',
        call_id="call_124"
    )

    print(f"\n2. TOOLCALL WITH STRING ARGUMENTS:")
    print(f"   Arguments type: {type(tool_call2.arguments)}")
    print(f"   Arguments value: {repr(tool_call2.arguments)}")

    # Apply server logic
    server_args2 = tool_call2.arguments if isinstance(tool_call2.arguments, str) else json.dumps(tool_call2.arguments)
    print(f"   Server conversion: {repr(server_args2)}")

    # Test if it's valid JSON
    try:
        parsed2 = json.loads(server_args2)
        print(f"   ✅ Valid JSON: {parsed2}")
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")

if __name__ == "__main__":
    test_openai_conversion()
    test_server_logic()
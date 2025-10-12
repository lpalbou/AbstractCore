#!/usr/bin/env python3
"""
Test script to reproduce the critical <function_call> detection bug.

This script simulates the exact behavior from the user's screenshot where:
1. Some tool calls work correctly (first 2 calls)
2. Later tool calls fail and <function_call> tags appear in output
3. The detection seems to become inconsistent within the same session

The issue appears to be that the server-side tool call detection in app.py
is not working consistently, particularly in streaming mode.
"""

import json
import sys
from typing import List, Dict, Any

# Add the project root to path
sys.path.insert(0, '/Users/albou/projects/abstractllm_core')

from abstractllm.providers.streaming import IncrementalToolDetector, UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse


def test_scenario_1_working_tool_calls():
    """Test the scenario where tool calls work correctly (first 2 in screenshot)."""
    print("\n" + "="*80)
    print("SCENARIO 1: Working Tool Calls (First 2 in screenshot)")
    print("="*80)

    # Simulate streaming chunks that come from LLM
    chunks = [
        "I'll help you list the files. ",
        "<function_call>",
        '{"name": "list_files", ',
        '"arguments": {"directory_path": "abstractllm"}}',
        "</function_call>",
        "\n\nHere are the results."
    ]

    detector = IncrementalToolDetector("llama-3", rewrite_tags=True)

    print("\nStreaming chunks:")
    accumulated_content = ""
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {repr(chunk)}")
        streamable, tools = detector.process_chunk(chunk)
        accumulated_content += streamable
        if streamable:
            print(f"    → Streamable: {repr(streamable)}")
        if tools:
            print(f"    → Tool detected: {tools[0].name}")

    print(f"\nFinal accumulated content: {repr(accumulated_content)}")
    print(f"Expected: Should contain the <function_call> tags for rewriting")

    # Check if tool call was preserved in content
    if "<function_call>" in accumulated_content:
        print("✅ PASS: Tool call preserved in content for rewriting")
    else:
        print("❌ FAIL: Tool call was removed from content!")


def test_scenario_2_failing_tool_calls():
    """Test the scenario where tool calls fail (later calls in screenshot)."""
    print("\n" + "="*80)
    print("SCENARIO 2: Failing Tool Calls (Later calls in screenshot)")
    print("="*80)

    # Simulate chunks where detection might fail
    # This mimics a situation where the tool call comes in different chunk patterns
    test_cases = [
        {
            "name": "Tool call split across many small chunks",
            "chunks": [
                "Let me check that. ",
                "<",
                "function",
                "_call",
                ">",
                '{"name"',
                ': "exec", ',
                '"arguments": ',
                '{"command": "ls"',
                '}}',
                "</",
                "function_call",
                ">",
                " Done."
            ]
        },
        {
            "name": "Tool call with content before and after",
            "chunks": [
                "I'll execute the command now. <function_",
                "call>{\"name\": \"shell\", \"arguments\":",
                " {\"command\": \"pwd\"}}</function_call> The command",
                " has been executed successfully."
            ]
        },
        {
            "name": "Multiple tool calls in sequence",
            "chunks": [
                "Running multiple commands:\n",
                "<function_call>{\"name\": \"shell\", \"arguments\": {\"cmd\": \"ls\"}}</function_call>\n",
                "<function_call>{\"name\": \"shell\", \"arguments\": {\"cmd\": \"pwd\"}}</function_call>\n",
                "Both commands completed."
            ]
        }
    ]

    for test_case in test_cases:
        print(f"\n📝 Test: {test_case['name']}")
        print("-" * 60)

        detector = IncrementalToolDetector("llama-3", rewrite_tags=True)
        accumulated_content = ""
        tools_found = []

        for i, chunk in enumerate(test_case['chunks']):
            streamable, tools = detector.process_chunk(chunk)
            accumulated_content += streamable
            tools_found.extend(tools)

            if i == 0 or i == len(test_case['chunks']) - 1 or streamable or tools:
                print(f"  Chunk {i:2d}: {repr(chunk)[:30]}")
                if streamable:
                    print(f"           → Stream: {repr(streamable)[:50]}")
                if tools:
                    print(f"           → Tools: {[t.name for t in tools]}")

        print(f"\n  Final content: {accumulated_content[:100]}...")
        print(f"  Tools detected: {len(tools_found)}")

        # Check results
        if "<function_call>" in accumulated_content and tools_found:
            print("  ✅ PASS: Tool calls preserved and detected")
        elif not tools_found:
            print("  ❌ FAIL: No tools detected!")
        elif "<function_call>" not in accumulated_content:
            print("  ❌ FAIL: Tool calls removed from content!")


def test_scenario_3_server_detection():
    """Test the server-side detection logic that's used in app.py."""
    print("\n" + "="*80)
    print("SCENARIO 3: Server-Side Detection (app.py logic)")
    print("="*80)

    # This simulates the exact detection logic from app.py lines 1929-1982

    test_contents = [
        {
            "name": "Complete tool call in single chunk",
            "content": "Let me help. <function_call>{\"name\": \"list_files\", \"arguments\": {}}</function_call> Done.",
            "expected_tools": 1
        },
        {
            "name": "Tool call with newlines",
            "content": "Executing:\n<function_call>\n{\n  \"name\": \"shell\",\n  \"arguments\": {\"cmd\": \"ls\"}\n}\n</function_call>\nComplete.",
            "expected_tools": 1
        },
        {
            "name": "Multiple tool calls",
            "content": "First: <function_call>{\"name\": \"pwd\", \"arguments\": {}}</function_call> Second: <function_call>{\"name\": \"ls\", \"arguments\": {}}</function_call>",
            "expected_tools": 2
        }
    ]

    import re
    tool_call_patterns = [
        (r'<\|tool_call\|>(.*?)</\|tool_call\|>', 'qwen3'),
        (r'<function_call>(.*?)</function_call>', 'llama'),
        (r'<tool_call>(.*?)</tool_call>', 'generic'),
    ]

    for test in test_contents:
        print(f"\n📝 Test: {test['name']}")
        print("-" * 60)
        print(f"  Content: {test['content'][:80]}...")

        tool_calls_found = []
        content = test['content']

        # Try to extract tool calls (server logic)
        for pattern, format_type in tool_call_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    tool_data = json.loads(match.strip())
                    tool_calls_found.append({
                        "name": tool_data.get("name", ""),
                        "arguments": tool_data.get("arguments", {})
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"    ⚠️ Failed to parse: {e}")

        print(f"  Found {len(tool_calls_found)} tool calls (expected {test['expected_tools']})")

        if len(tool_calls_found) == test['expected_tools']:
            print("  ✅ PASS: Correct number of tools detected")
        else:
            print(f"  ❌ FAIL: Expected {test['expected_tools']} but found {len(tool_calls_found)}")


def test_scenario_4_unified_processor():
    """Test the UnifiedStreamProcessor which should handle everything correctly."""
    print("\n" + "="*80)
    print("SCENARIO 4: UnifiedStreamProcessor (Should work correctly)")
    print("="*80)

    # Test with the unified processor which has the fixes
    test_streams = [
        {
            "name": "Simple function call",
            "chunks": [
                GenerateResponse(content="Let me help. "),
                GenerateResponse(content="<function_call>"),
                GenerateResponse(content='{"name": "list_files", '),
                GenerateResponse(content='"arguments": {}}'),
                GenerateResponse(content="</function_call>"),
                GenerateResponse(content=" Done.")
            ]
        },
        {
            "name": "Character-by-character streaming",
            "chunks": [
                GenerateResponse(content=c) for c in
                "Here's the result: <function_call>{\"name\": \"exec\", \"arguments\": {\"cmd\": \"pwd\"}}</function_call> Complete."
            ]
        }
    ]

    for test in test_streams:
        print(f"\n📝 Test: {test['name']}")
        print("-" * 60)

        processor = UnifiedStreamProcessor(
            model_name="llama-3",
            tool_call_tags="llama3"  # This should preserve <function_call> format
        )

        accumulated = ""
        chunk_count = 0

        for response in processor.process_stream(iter(test['chunks'])):
            if response.content:
                accumulated += response.content
                chunk_count += 1
                if chunk_count <= 3 or chunk_count == len(test['chunks']):
                    print(f"  Chunk {chunk_count}: {repr(response.content)[:50]}")

        print(f"\n  Final output: {accumulated[:100]}...")

        # Check that function_call tags are present
        if "<function_call>" in accumulated:
            print("  ✅ PASS: Tool call tags preserved in output")
        else:
            print("  ❌ FAIL: Tool call tags missing from output!")


def main():
    """Run all test scenarios."""
    print("\n" + "="*80)
    print("TESTING <function_call> DETECTION AND CONVERSION")
    print("Reproducing the critical bug from user's screenshot")
    print("="*80)

    # Run all scenarios
    test_scenario_1_working_tool_calls()
    test_scenario_2_failing_tool_calls()
    test_scenario_3_server_detection()
    test_scenario_4_unified_processor()

    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print("""
The issue appears to be in the server-side detection logic (app.py):

1. The IncrementalToolDetector with rewrite_tags=True SHOULD preserve tool calls
2. The server app.py has its own detection logic that might interfere
3. The buffering logic for split chunks may not handle all cases

Key problems identified:
- Character-by-character streaming might break detection
- The server's regex-based detection is separate from the streaming detector
- Tool calls might be getting removed when they shouldn't be

The fix should ensure:
1. Tool calls are always detected, regardless of chunk boundaries
2. When rewriting is enabled, tags are preserved in output
3. The server doesn't duplicate or interfere with detection
""")


if __name__ == "__main__":
    main()
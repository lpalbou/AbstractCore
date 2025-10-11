#!/usr/bin/env python3
"""
Quick test to verify universal tool call conversion works correctly.
This test simulates the server's tool call conversion logic.
"""

import json
import re
import uuid


def test_universal_tool_conversion():
    """Test that all tool call formats are properly converted to OpenAI format"""

    # Test cases with different model formats
    test_cases = [
        {
            "name": "Qwen3 format",
            "input": 'I will help you. <|tool_call|>\n{"name": "shell", "arguments": {"command": ["ls", "-la"]}}\n</|tool_call|>',
            "expected_tool": {"name": "shell", "arguments": {"command": ["ls", "-la"]}}
        },
        {
            "name": "LLaMA format",
            "input": 'Let me execute that. <function_call>\n{"name": "shell", "arguments": {"command": ["ls", "-la"]}}\n</function_call>',
            "expected_tool": {"name": "shell", "arguments": {"command": ["ls", "-la"]}}
        },
        {
            "name": "Generic format",
            "input": 'Here we go: <tool_call>{"name": "shell", "arguments": {"command": ["ls", "-la"]}}</tool_call>',
            "expected_tool": {"name": "shell", "arguments": {"command": ["ls", "-la"]}}
        }
    ]

    # Tool call patterns from server code
    tool_call_patterns = [
        (r'<\|tool_call\|>(.*?)</\|tool_call\|>', 'qwen3'),
        (r'<function_call>(.*?)</function_call>', 'llama'),
        (r'<tool_call>(.*?)</tool_call>', 'generic'),
    ]

    print("🧪 Testing Universal Tool Call Conversion\n")

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: {test_case['input']}")

        content = test_case['input']
        tool_calls_found = []

        # Try to extract tool calls from content (server logic)
        for pattern, format_type in tool_call_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    # Parse the JSON content
                    tool_data = json.loads(match.strip())
                    tool_calls_found.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_data.get("name", ""),
                            "arguments": json.dumps(tool_data.get("arguments", {}))
                        }
                    })
                    # Remove the tool call from content
                    full_match = re.search(pattern, content, re.DOTALL)
                    if full_match:
                        content = content.replace(full_match.group(0), "").strip()
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"❌ Failed to parse: {e}")
                    continue

        # Verify results
        if tool_calls_found:
            tool_call = tool_calls_found[0]
            expected = test_case['expected_tool']

            print(f"✅ Converted to OpenAI format:")
            print(f"   Tool Name: {tool_call['function']['name']}")
            print(f"   Arguments: {tool_call['function']['arguments']}")
            print(f"   Cleaned Content: '{content}'")

            # Verify correctness
            actual_args = json.loads(tool_call['function']['arguments'])
            if (tool_call['function']['name'] == expected['name'] and
                actual_args == expected['arguments']):
                print(f"✅ PASS: Correct conversion\n")
            else:
                print(f"❌ FAIL: Incorrect conversion\n")
        else:
            print(f"❌ FAIL: No tool calls detected\n")


if __name__ == "__main__":
    test_universal_tool_conversion()
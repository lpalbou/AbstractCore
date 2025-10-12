#!/usr/bin/env python3
"""
Simple test to verify Codex format fix.
"""

import json
from abstractllm.tools.core import ToolCall

def test_codex_format_fix():
    """Test that we generate the exact format Codex needs."""

    # Create a ToolCall with dict arguments (normal case)
    tool_call = ToolCall(
        name="shell",
        arguments={"command": ["ls", "-la"]},
        call_id="call_123"
    )

    print("🧪 Testing Codex Format Fix")
    print("="*50)
    print(f"Input arguments: {repr(tool_call.arguments)}")
    print(f"Arguments type: {type(tool_call.arguments)}")

    # Apply the server logic (the fix)
    server_arguments = json.dumps(tool_call.arguments) if isinstance(tool_call.arguments, dict) else str(tool_call.arguments)

    print(f"Server output: {repr(server_arguments)}")

    # This should be exactly what Codex expects
    expected = '{"command": ["ls", "-la"]}'

    if server_arguments == expected:
        print("✅ SUCCESS: Format matches Codex requirements")
    else:
        print("❌ FAIL: Format doesn't match")
        print(f"Expected: {repr(expected)}")
        print(f"Got:      {repr(server_arguments)}")

    # Test that it's valid JSON
    try:
        parsed = json.loads(server_arguments)
        print(f"✅ Valid JSON: {parsed}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")

    return server_arguments == expected

if __name__ == "__main__":
    test_codex_format_fix()
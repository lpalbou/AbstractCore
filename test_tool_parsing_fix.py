#!/usr/bin/env python3
"""
Test and fix tool call parsing issues.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from abstractllm.tools.parser import detect_tool_calls, parse_tool_calls

def test_tool_call_parsing():
    """Test tool call parsing with different formats."""
    print("🔧 Testing Tool Call Parsing")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Qwen3 Format",
            "response": '<|tool_call|>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</|tool_call|>',
            "expected_tools": 1
        },
        {
            "name": "LLaMA3 Format", 
            "response": '<function_call>\n{"name": "calculate", "arguments": {"expression": "2+2"}}\n</function_call>',
            "expected_tools": 1
        },
        {
            "name": "Gemma3 Format",
            "response": '```tool_code\n{"name": "web_search", "arguments": {"query": "AI news"}}\n```',
            "expected_tools": 1
        },
        {
            "name": "Generic JSON Format",
            "response": '{"name": "test_tool", "arguments": {"param": "value"}}',
            "expected_tools": 1
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print("-" * 30)
        print(f"Response: {test_case['response']}")
        
        # Test detection
        detected = detect_tool_calls(test_case['response'])
        print(f"Detected: {'✅ Yes' if detected else '❌ No'}")
        
        if detected:
            tool_calls = parse_tool_calls(test_case['response'])
            print(f"Parsed: {len(tool_calls)} tool calls")
            for i, call in enumerate(tool_calls):
                print(f"  {i+1}. {call.name}({call.arguments})")
        else:
            print("❌ No tool calls detected")

if __name__ == "__main__":
    test_tool_call_parsing()
#!/usr/bin/env python3
"""
Manual test script to validate OpenAI format conversion.

This script demonstrates that the fix works correctly:
- Input: Qwen3 format text-based tool call
- Output: OpenAI JSON structured format
"""

import json
import sys
sys.path.insert(0, '/Users/albou/projects/abstractllm_core')

from abstractllm.providers.streaming import UnifiedStreamProcessor


def test_openai_conversion():
    """Test that OpenAI format conversion works correctly."""
    print("=" * 80)
    print("OpenAI Format Conversion Test")
    print("=" * 80)

    # Create processor with OpenAI target format
    processor = UnifiedStreamProcessor(
        model_name="qwen3-coder:30b",
        tool_call_tags="openai"
    )

    print(f"\nProcessor initialized:")
    print(f"  - tag_rewriter: {processor.tag_rewriter}")
    print(f"  - convert_to_openai_json: {processor.convert_to_openai_json}")

    # Input: Qwen3 format tool call
    qwen_input = '<|tool_call|>{"name": "shell", "arguments": {"command": ["ls", "-la"]}}</|tool_call|>'

    print(f"\nInput (Qwen3 format):")
    print(f"  {qwen_input}")

    # Convert to OpenAI format
    result = processor._convert_to_openai_format(qwen_input)

    print(f"\nOutput (OpenAI format):")
    print(f"  {result}")

    # Parse and validate
    try:
        result_json = json.loads(result)
        print(f"\nParsed JSON structure:")
        print(json.dumps(result_json, indent=2))

        # Validate OpenAI format
        assert result_json["type"] == "function", "Missing 'type' field"
        assert result_json["function"]["name"] == "shell", "Incorrect function name"
        assert "id" in result_json, "Missing 'id' field"
        assert result_json["id"].startswith("call_"), "ID should start with 'call_'"

        # Validate arguments
        args = json.loads(result_json["function"]["arguments"])
        assert args["command"] == ["ls", "-la"], "Incorrect arguments"

        print("\n" + "=" * 80)
        print("✅ TEST PASSED: OpenAI format conversion works correctly!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False


def test_multiple_formats():
    """Test conversion from multiple input formats."""
    print("\n\n" + "=" * 80)
    print("Multiple Format Conversion Test")
    print("=" * 80)

    processor = UnifiedStreamProcessor(
        model_name="test-model",
        tool_call_tags="openai"
    )

    test_cases = [
        ("Qwen3", '<|tool_call|>{"name": "test", "arguments": {}}</|tool_call|>'),
        ("LLaMA", '<function_call>{"name": "test", "arguments": {}}</function_call>'),
        ("XML", '<tool_call>{"name": "test", "arguments": {}}</tool_call>'),
        ("Gemma", '```tool_code\n{"name": "test", "arguments": {}}\n```'),
    ]

    all_passed = True
    for format_name, input_text in test_cases:
        print(f"\nTesting {format_name} format:")
        print(f"  Input: {input_text[:60]}...")

        result = processor._convert_to_openai_format(input_text)

        try:
            result_json = json.loads(result)
            assert result_json["type"] == "function"
            assert result_json["function"]["name"] == "test"
            print(f"  ✅ {format_name} conversion successful")
        except Exception as e:
            print(f"  ❌ {format_name} conversion failed: {e}")
            all_passed = False

    if all_passed:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED: All formats convert correctly!")
        print("=" * 80)
    else:
        print("\n❌ SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    success1 = test_openai_conversion()
    success2 = test_multiple_formats()

    sys.exit(0 if (success1 and success2) else 1)

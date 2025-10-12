#!/usr/bin/env python
"""Direct runner for OpenAI format bug test."""

import sys
sys.path.insert(0, '/Users/albou/projects/abstractllm_core')

# Import and run the tests
from tests.test_openai_format_bug import (
    test_openai_format_arguments_as_json_string,
    test_openai_format_llama_input,
    test_openai_format_xml_input,
    test_openai_format_with_complex_arguments,
    test_openai_format_empty_arguments
)

if __name__ == "__main__":
    print("=" * 80)
    print("Testing OpenAI Format Arguments Encoding Bug")
    print("=" * 80)

    try:
        print("\n[1/5] Testing Qwen3 format conversion...")
        test_openai_format_arguments_as_json_string()

        print("\n[2/5] Testing LLaMA format conversion...")
        test_openai_format_llama_input()

        print("\n[3/5] Testing XML format conversion...")
        test_openai_format_xml_input()

        print("\n[4/5] Testing complex arguments...")
        test_openai_format_with_complex_arguments()

        print("\n[5/5] Testing empty arguments...")
        test_openai_format_empty_arguments()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - OpenAI format is correct!")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Validation script for streaming tag rewriting fix.

This script tests that custom tool call tags work correctly in streaming mode,
matching the exact scenario reported by the user.
"""

from abstractllm.providers.streaming import UnifiedStreamProcessor
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.tag_rewriter import ToolCallTags


def test_user_scenario():
    """Test the exact user scenario: /tooltag 'ojlk' 'dfsd'"""
    print("=" * 70)
    print("TEST: User Scenario - /tooltag 'ojlk' 'dfsd'")
    print("=" * 70)

    # Create processor with custom tags (as set by /tooltag command)
    processor = UnifiedStreamProcessor(
        model_name="test-model",
        execute_tools=False,
        tool_call_tags="ojlk,dfsd"
    )

    # Verify tag rewriter was initialized
    assert processor.tag_rewriter is not None, "Tag rewriter not initialized!"
    print("✅ Tag rewriter initialized")
    print(f"   Opening tag: {processor.tag_rewriter.target_tags.start_tag}")
    print(f"   Closing tag: {processor.tag_rewriter.target_tags.end_tag}")

    # Simulate LLM response with tool call (Qwen format)
    content = 'Let me list those files for you.<|tool_call|>{"name": "list_files", "arguments": {"directory_path": "abstractllm"}}</|tool_call|>'

    print(f"\n📥 Input content:")
    print(f"   {content[:100]}...")

    # Create mock stream
    def mock_stream():
        # Simulate character-by-character streaming (realistic scenario)
        for i in range(0, len(content), 10):  # 10 chars at a time
            chunk = content[i:i+10]
            yield GenerateResponse(content=chunk, model="test-model")

    # Process stream
    print(f"\n🔄 Processing streaming chunks...")
    results = list(processor.process_stream(mock_stream()))
    full_output = "".join([r.content for r in results if r.content])

    print(f"\n📤 Output content:")
    print(f"   {full_output}")

    # CRITICAL VALIDATION
    print(f"\n🔍 Validation:")

    # Must contain custom tags
    has_custom_opening = "<ojlk>" in full_output
    has_custom_closing = "</dfsd>" in full_output
    print(f"   ✅ Custom opening tag '<ojlk>': {'FOUND' if has_custom_opening else '❌ MISSING'}")
    print(f"   ✅ Custom closing tag '</dfsd>': {'FOUND' if has_custom_closing else '❌ MISSING'}")

    # Must NOT contain original tags
    has_original_opening = "<|tool_call|>" in full_output
    has_original_closing = "</|tool_call|>" in full_output
    print(f"   ✅ Original opening tag removed: {'YES' if not has_original_opening else '❌ STILL PRESENT'}")
    print(f"   ✅ Original closing tag removed: {'YES' if not has_original_closing else '❌ STILL PRESENT'}")

    # Must contain the JSON content
    has_tool_name = '"name": "list_files"' in full_output
    has_arguments = '"directory_path": "abstractllm"' in full_output
    print(f"   ✅ Tool call JSON preserved: {'YES' if has_tool_name and has_arguments else '❌ CORRUPTED'}")

    # Final result
    success = has_custom_opening and has_custom_closing and not has_original_opening and not has_original_closing and has_tool_name and has_arguments

    if success:
        print(f"\n✅ TEST PASSED: Streaming tag rewriting works correctly!")
        return True
    else:
        print(f"\n❌ TEST FAILED: Streaming tag rewriting is broken!")
        return False


def test_multiple_formats():
    """Test different tool call formats"""
    print("\n" + "=" * 70)
    print("TEST: Multiple Tool Call Formats")
    print("=" * 70)

    test_cases = [
        ("Qwen format", '<|tool_call|>{"name": "test"}</|tool_call|>', "qwen", "TEST,CALL"),
        ("LLaMA format", '<function_call>{"name": "test"}</function_call>', "llama", "FUNC,END"),
        ("XML format", '<tool_call>{"name": "test"}</tool_call>', "xml", "X,Y"),
    ]

    all_passed = True

    for name, content, model_hint, tags in test_cases:
        print(f"\n🧪 Testing {name}:")

        processor = UnifiedStreamProcessor(
            model_name=f"{model_hint}-model",
            execute_tools=False,
            tool_call_tags=tags
        )

        def mock_stream():
            yield GenerateResponse(content=content, model=f"{model_hint}-model")

        results = list(processor.process_stream(mock_stream()))
        full_output = "".join([r.content for r in results if r.content])

        start_tag, end_tag = tags.split(',')
        has_custom_tags = f"<{start_tag}>" in full_output and f"</{end_tag}>" in full_output

        if has_custom_tags:
            print(f"   ✅ {name} rewritten correctly")
        else:
            print(f"   ❌ {name} rewriting FAILED")
            all_passed = False

    return all_passed


def test_performance():
    """Test that tag rewriting doesn't add significant latency"""
    print("\n" + "=" * 70)
    print("TEST: Performance Impact")
    print("=" * 70)

    import time

    processor = UnifiedStreamProcessor(
        model_name="test-model",
        execute_tools=False,
        tool_call_tags="A,B"
    )

    # Create a stream with 100 chunks
    content = "Hello, this is a test message without tool calls." * 10

    def mock_stream():
        for i in range(0, len(content), 50):
            chunk = content[i:i+50]
            yield GenerateResponse(content=chunk, model="test-model")

    # Measure processing time
    start = time.time()
    results = list(processor.process_stream(mock_stream()))
    duration_ms = (time.time() - start) * 1000

    print(f"   Processed {len(results)} chunks in {duration_ms:.2f}ms")
    print(f"   Average latency per chunk: {duration_ms/len(results):.2f}ms")

    # Should be very fast (< 1ms per chunk for non-tool content)
    if duration_ms / len(results) < 1.0:
        print(f"   ✅ Performance is excellent (<1ms per chunk)")
        return True
    elif duration_ms / len(results) < 5.0:
        print(f"   ✅ Performance is acceptable (<5ms per chunk)")
        return True
    else:
        print(f"   ⚠️ Performance may need optimization (>{duration_ms/len(results):.2f}ms per chunk)")
        return False


def main():
    """Run all validation tests"""
    print("\n🚀 STREAMING TAG REWRITING VALIDATION")
    print("=" * 70)

    results = []

    # Test 1: User scenario
    results.append(("User Scenario", test_user_scenario()))

    # Test 2: Multiple formats
    results.append(("Multiple Formats", test_multiple_formats()))

    # Test 3: Performance
    results.append(("Performance", test_performance()))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20} {status}")

    print("=" * 70)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("✅ ALL TESTS PASSED - Streaming tag rewriting is working correctly!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Streaming tag rewriting has issues!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

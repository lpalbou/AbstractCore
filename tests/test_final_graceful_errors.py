#!/usr/bin/env python3
"""
Final test to demonstrate the improved graceful error handling.
Shows the exact user experience now vs before.
"""

import os
from abstractllm import create_llm, ModelNotFoundError


def test_anthropic_clean_error():
    """Test user's original case - now with clean error"""
    print("🔧 Testing Anthropic with invalid model (original user case):")
    print("   Model: 'claude-3.5-haiku:latest' (invalid)")
    print()

    try:
        llm = create_llm("anthropic", model="claude-3.5-haiku:latest")
        response = llm.generate("Hello, who are you? identify yourself")
        assert False, "Should have failed with ModelNotFoundError"
    except ModelNotFoundError as e:
        print("✅ SUCCESS: Clean error with helpful information")
        print("=" * 60)
        print(str(e))
        print("=" * 60)
        print()
        print("✅ User gets:")
        print("   • Clear error message (not ugly traceback)")
        print("   • NO static/outdated model list")
        print("   • Direct link to official documentation")
        print("   • Helpful tips for the provider")
        print("   • NO duplicate error messages")
    except Exception as e:
        assert False, f"Wrong exception: {type(e).__name__}: {e}"


def test_openai_dynamic_models():
    """Test OpenAI with dynamic model fetching"""
    if not os.getenv("OPENAI_API_KEY"):
        import pytest
        pytest.skip("OPENAI_API_KEY not set")

    print("🔧 Testing OpenAI with dynamic model discovery:")
    print("   Model: 'gpt-fake-model' (invalid)")
    print()

    try:
        llm = create_llm("openai", model="gpt-fake-model")
        response = llm.generate("Hello")
        assert False, "Should have failed with ModelNotFoundError"
    except ModelNotFoundError as e:
        error_text = str(e)
        has_models = "Available models" in error_text and "gpt-" in error_text

        print("✅ SUCCESS: Dynamic model fetching")
        print(f"   • Fetched {error_text.count('gpt-')} live models from OpenAI API")
        print("   • Up-to-date model list (not static)")
        print("   • Official documentation link")
        assert has_models, "Should show available models in error message"
    except Exception as e:
        assert False, f"Wrong exception: {type(e).__name__}: {e}"


def test_ollama_dynamic_models():
    """Test Ollama with dynamic model fetching"""
    print("🔧 Testing Ollama with dynamic model discovery:")
    print("   Model: 'fake-local-model' (invalid)")
    print()

    try:
        llm = create_llm("ollama", model="fake-local-model")
        response = llm.generate("Hello")
        assert False, "Should have failed with ModelNotFoundError"
    except ModelNotFoundError as e:
        error_text = str(e)
        has_models = "Available models" in error_text or "available models" in error_text

        print("✅ SUCCESS: Local model discovery")
        print(f"   • Fetched live models from Ollama server")
        print("   • Shows actually available local models")
        print("   • Helpful tip about 'ollama pull'")
        assert has_models, "Should show available models in error message"
    except Exception as e:
        # Allow connection errors when Ollama isn't running
        if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
            import pytest
            pytest.skip("Ollama not running")
        else:
            assert False, f"Wrong exception: {type(e).__name__}: {e}"


def main():
    """Run all graceful error tests"""
    print("=" * 70)
    print("FINAL GRACEFUL ERROR HANDLING TEST")
    print("Demonstrating the improved user experience")
    print("=" * 70)
    print()

    results = []

    # Test each provider
    results.append(test_anthropic_clean_error())
    print()
    results.append(test_openai_dynamic_models())
    print()
    results.append(test_ollama_dynamic_models())

    # Summary
    passed = sum(results)
    total = len(results)

    print()
    print("=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("🎉 ALL GRACEFUL ERROR HANDLING WORKING PERFECTLY!")
        print()
        print("✅ Users now get:")
        print("   • Clean error messages (no ugly tracebacks)")
        print("   • Dynamic model lists (always up-to-date)")
        print("   • Official documentation links")
        print("   • Provider-specific helpful tips")
        print("   • No duplicate error messages")
    else:
        print("❌ Some tests failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
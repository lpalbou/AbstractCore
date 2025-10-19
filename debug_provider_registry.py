#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_provider_registry():
    """Test that provider registry creates LMStudioProvider correctly"""

    print("🔍 TESTING PROVIDER REGISTRY")
    print("=" * 40)

    try:
        # Test 1: Test factory create_llm
        print("1. Testing factory create_llm...")
        from abstractcore.core.factory import create_llm

        provider_instance = create_llm("lmstudio", model="qwen/qwen3-vl-4b")
        print(f"   Created: {type(provider_instance).__name__}")
        print(f"   Model: {provider_instance.model}")

        # Test 2: Test that it's actually LMStudioProvider
        from abstractcore.providers.lmstudio_provider import LMStudioProvider
        is_lmstudio = isinstance(provider_instance, LMStudioProvider)
        print(f"   Is LMStudioProvider: {'✅ YES' if is_lmstudio else '❌ NO'}")

        if not is_lmstudio:
            print(f"   ❌ ISSUE: Expected LMStudioProvider, got {type(provider_instance).__name__}")
            return False

        # Test 3: Test that generate method exists and is callable
        has_generate = hasattr(provider_instance, 'generate') and callable(getattr(provider_instance, 'generate'))
        print(f"   Has generate method: {'✅ YES' if has_generate else '❌ NO'}")

        # Test 4: Test media processing methods
        has_process_media = hasattr(provider_instance, '_process_media_content')
        has_get_handler = hasattr(provider_instance, '_get_media_handler_for_model')
        print(f"   Has _process_media_content: {'✅ YES' if has_process_media else '❌ NO'}")
        print(f"   Has _get_media_handler_for_model: {'✅ YES' if has_get_handler else '❌ NO'}")

        return True

    except Exception as e:
        print(f"   ❌ Provider registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_create_provider():
    """Test the exact same provider creation that the server does"""

    print("\n2. Testing server provider creation...")

    try:
        # Parse the model string like the server does
        from abstractcore.server.app import parse_model_string

        model_string = "lmstudio/qwen/qwen3-vl-4b"
        provider, model = parse_model_string(model_string)

        print(f"   Parsed model string '{model_string}':")
        print(f"     Provider: '{provider}'")
        print(f"     Model: '{model}'")

        # Create provider like the server does
        from abstractcore.core.factory import create_llm

        llm = create_llm(provider, model=model)

        print(f"   Created: {type(llm).__name__}")
        print(f"   Model: {llm.model}")

        # Check if this is the same as our direct test
        from abstractcore.providers.lmstudio_provider import LMStudioProvider
        is_lmstudio = isinstance(llm, LMStudioProvider)
        print(f"   Is LMStudioProvider: {'✅ YES' if is_lmstudio else '❌ NO'}")

        return is_lmstudio

    except Exception as e:
        print(f"   ❌ Server provider creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registry_direct():
    """Test the provider registry directly"""

    print("\n3. Testing provider registry directly...")

    try:
        from abstractcore.providers.registry import create_provider

        provider_instance = create_provider("lmstudio", "qwen/qwen3-vl-4b")

        print(f"   Created: {type(provider_instance).__name__}")

        from abstractcore.providers.lmstudio_provider import LMStudioProvider
        is_lmstudio = isinstance(provider_instance, LMStudioProvider)
        print(f"   Is LMStudioProvider: {'✅ YES' if is_lmstudio else '❌ NO'}")

        return is_lmstudio

    except Exception as e:
        print(f"   ❌ Registry direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1 = test_provider_registry()
    test2 = test_server_create_provider()
    test3 = test_registry_direct()

    if test1 and test2 and test3:
        print("\n✅ All registry tests pass - LMStudioProvider should be created correctly")
    else:
        print("\n❌ Registry tests failed - provider creation is broken")
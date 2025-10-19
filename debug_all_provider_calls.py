#!/usr/bin/env python3

import sys
import os
import requests
import time

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def monkey_patch_all_provider_systems():
    """Monkey patch everything provider-related to track calls"""

    print("🔍 COMPREHENSIVE PROVIDER CALL TRACKING")
    print("=" * 50)

    call_log = []

    # Patch 1: Factory create_llm
    from abstractcore.core import factory
    original_create_llm = factory.create_llm

    def debug_create_llm(provider, model=None, **kwargs):
        call_info = {
            "system": "factory.create_llm",
            "provider": provider,
            "model": model,
            "timestamp": time.time()
        }
        call_log.append(call_info)
        print(f"📞 FACTORY: create_llm(provider='{provider}', model='{model}')")
        result = original_create_llm(provider, model, **kwargs)
        print(f"   Returned: {type(result).__name__}")
        return result

    factory.create_llm = debug_create_llm

    # Patch 2: Provider registry create_provider
    from abstractcore.providers import registry
    original_create_provider = registry.create_provider

    def debug_create_provider(provider_name, model=None, **kwargs):
        call_info = {
            "system": "registry.create_provider",
            "provider": provider_name,
            "model": model,
            "timestamp": time.time()
        }
        call_log.append(call_info)
        print(f"📞 REGISTRY: create_provider(provider='{provider_name}', model='{model}')")
        result = original_create_provider(provider_name, model, **kwargs)
        print(f"   Returned: {type(result).__name__}")
        return result

    registry.create_provider = debug_create_provider

    # Patch 3: LMStudioProvider constructor
    from abstractcore.providers.lmstudio_provider import LMStudioProvider
    original_lmstudio_init = LMStudioProvider.__init__

    def debug_lmstudio_init(self, model="local-model", **kwargs):
        call_info = {
            "system": "LMStudioProvider.__init__",
            "model": model,
            "timestamp": time.time()
        }
        call_log.append(call_info)
        print(f"📞 LMSTUDIO: __init__(model='{model}')")
        return original_lmstudio_init(self, model, **kwargs)

    LMStudioProvider.__init__ = debug_lmstudio_init

    # Patch 4: LMStudioProvider generate method
    original_lmstudio_generate = LMStudioProvider.generate

    def debug_lmstudio_generate(self, *args, **kwargs):
        call_info = {
            "system": "LMStudioProvider.generate",
            "model": getattr(self, 'model', 'unknown'),
            "args_count": len(args),
            "has_media": 'media' in kwargs and bool(kwargs['media']),
            "timestamp": time.time()
        }
        call_log.append(call_info)
        print(f"📞 LMSTUDIO: generate() called on model '{getattr(self, 'model', 'unknown')}'")
        print(f"   Args: {len(args)}, Has media: {'media' in kwargs and bool(kwargs['media'])}")
        return original_lmstudio_generate(self, *args, **kwargs)

    LMStudioProvider.generate = debug_lmstudio_generate

    return call_log

def make_test_request():
    """Make the failing server request"""

    print(f"\n🌐 MAKING TEST REQUEST...")

    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    payload = {
        "model": "lmstudio/qwen/qwen3-vl-4b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"📨 RESPONSE: {content[:100]}...")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def analyze_call_log(call_log):
    """Analyze the captured calls"""

    print(f"\n📊 CALL LOG ANALYSIS:")
    print(f"   Total calls: {len(call_log)}")

    if not call_log:
        print("   ❌ NO PROVIDER CALLS DETECTED!")
        print("   This means the server is not using the provider system at all.")
        return

    # Group by system
    by_system = {}
    for call in call_log:
        system = call['system']
        if system not in by_system:
            by_system[system] = []
        by_system[system].append(call)

    for system, calls in by_system.items():
        print(f"\n   {system}: {len(calls)} calls")
        for i, call in enumerate(calls):
            provider = call.get('provider', 'N/A')
            model = call.get('model', 'N/A')
            print(f"     Call {i+1}: provider='{provider}', model='{model}'")

    # Check for LMStudio specifically
    lmstudio_calls = [call for call in call_log if
                     'lmstudio' in str(call.get('provider', '')).lower() or
                     'LMStudio' in call.get('system', '')]

    if lmstudio_calls:
        print(f"\n   ✅ LMStudio calls detected: {len(lmstudio_calls)}")
    else:
        print(f"\n   ❌ NO LMSTUDIO CALLS DETECTED!")
        print("   The server is not creating LMStudioProvider instances.")

def main():
    # Set up comprehensive monitoring
    call_log = monkey_patch_all_provider_systems()

    # Make the request
    success = make_test_request()

    # Analyze what happened
    analyze_call_log(call_log)

    # Diagnosis
    if not call_log:
        print(f"\n🚨 CRITICAL ISSUE: Server is not using the provider system at all!")
        print("   Possible causes:")
        print("   - Server is using a cached response")
        print("   - Server is using a different code path")
        print("   - Request is being handled by a different system")
    elif not any('lmstudio' in str(call.get('provider', '')).lower() for call in call_log):
        print(f"\n🚨 ROUTING ISSUE: Server is not routing to LMStudio provider!")
        print("   The request is being handled by a different provider.")
    else:
        print(f"\n✅ Provider system is working - investigate LMStudio-specific issue")

if __name__ == "__main__":
    main()
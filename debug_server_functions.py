#!/usr/bin/env python3

import sys
import os
import requests
import time

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def monkey_patch_server_functions():
    """Monkey patch key server functions to track execution"""

    print("🔍 MONITORING SERVER FUNCTION CALLS")
    print("=" * 50)

    call_log = []

    # Patch 1: Chat completions endpoint
    from abstractcore.server import app
    original_chat_completions = app.chat_completions

    async def debug_chat_completions(request, http_request):
        call_info = {
            "function": "chat_completions",
            "model": request.model,
            "messages": len(request.messages),
            "timestamp": time.time()
        }
        call_log.append(call_info)
        print(f"📞 SERVER: chat_completions(model='{request.model}', messages={len(request.messages)})")
        return await original_chat_completions(request, http_request)

    app.chat_completions = debug_chat_completions

    # Patch 2: Process chat completion
    original_process_chat_completion = app.process_chat_completion

    async def debug_process_chat_completion(provider, model, request, http_request):
        call_info = {
            "function": "process_chat_completion",
            "provider": provider,
            "model": model,
            "timestamp": time.time()
        }
        call_log.append(call_info)
        print(f"📞 SERVER: process_chat_completion(provider='{provider}', model='{model}')")
        return await original_process_chat_completion(provider, model, request, http_request)

    app.process_chat_completion = debug_process_chat_completion

    # Patch 3: Parse model string
    original_parse_model_string = app.parse_model_string

    def debug_parse_model_string(model_string):
        call_info = {
            "function": "parse_model_string",
            "input": model_string,
            "timestamp": time.time()
        }
        call_log.append(call_info)
        result = original_parse_model_string(model_string)
        print(f"📞 SERVER: parse_model_string('{model_string}') -> {result}")
        call_info["output"] = result
        return result

    app.parse_model_string = debug_parse_model_string

    # Patch 4: Factory create_llm (from previous test)
    from abstractcore.core import factory
    original_create_llm = factory.create_llm

    def debug_create_llm(provider, model=None, **kwargs):
        call_info = {
            "function": "create_llm",
            "provider": provider,
            "model": model,
            "timestamp": time.time()
        }
        call_log.append(call_info)
        print(f"📞 FACTORY: create_llm(provider='{provider}', model='{model}')")
        result = original_create_llm(provider, model, **kwargs)
        print(f"   Created: {type(result).__name__}")
        return result

    factory.create_llm = debug_create_llm

    return call_log

def make_test_request():
    """Make the test request"""

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

    print(f"   Target: {url}")
    print(f"   Model: {payload['model']}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   Response: {content[:100]}...")
            return True
        else:
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"   Exception: {e}")
        return False

def analyze_server_calls(call_log):
    """Analyze server function calls"""

    print(f"\n📊 SERVER CALL ANALYSIS:")
    print(f"   Total calls: {len(call_log)}")

    if not call_log:
        print("   ❌ NO SERVER FUNCTIONS CALLED!")
        print("   The request might not be reaching the server endpoints.")
        return

    for i, call in enumerate(call_log):
        func = call['function']
        print(f"   Call {i+1}: {func}")

        if func == "parse_model_string":
            print(f"     Input: '{call['input']}'")
            print(f"     Output: {call.get('output', 'N/A')}")
        elif func in ["process_chat_completion", "create_llm"]:
            provider = call.get('provider', 'N/A')
            model = call.get('model', 'N/A')
            print(f"     Provider: '{provider}', Model: '{model}'")

    # Check execution order
    functions_called = [call['function'] for call in call_log]
    expected_order = ["chat_completions", "parse_model_string", "process_chat_completion", "create_llm"]

    print(f"\n   Function call order: {functions_called}")
    print(f"   Expected order: {expected_order}")

    for expected_func in expected_order:
        if expected_func not in functions_called:
            print(f"   ❌ MISSING: {expected_func} was never called!")
            break
    else:
        print(f"   ✅ All expected functions were called")

def main():
    # Set up monitoring
    call_log = monkey_patch_server_functions()

    # Make the request
    success = make_test_request()

    # Analyze what happened
    analyze_server_calls(call_log)

    # Final diagnosis
    functions_called = [call['function'] for call in call_log] if call_log else []

    if not call_log:
        print(f"\n🚨 CRITICAL: Request not reaching server functions!")
    elif "create_llm" not in functions_called:
        print(f"\n🚨 ISSUE: Server functions called but provider not created!")
        print("   Check for exceptions or early returns in process_chat_completion.")
    else:
        print(f"\n🤔 MYSTERY: All functions called but provider system still bypassed!")

if __name__ == "__main__":
    main()
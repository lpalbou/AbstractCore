#!/usr/bin/env python3

import sys
import os
import json
import threading
import time

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def monkey_patch_provider_for_debugging():
    """Monkey patch the LMStudioProvider to capture what happens during server requests"""

    print("🔍 DEBUGGING SERVER-PROVIDER INTEGRATION")
    print("=" * 50)

    from abstractcore.providers.lmstudio_provider import LMStudioProvider

    # Store original methods
    original_generate_internal = LMStudioProvider._generate_internal
    original_single_generate = LMStudioProvider._single_generate

    # Track calls
    generate_calls = []
    payload_captures = []

    def debug_generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, **kwargs):
        """Debug wrapper for _generate_internal"""

        call_info = {
            "prompt": prompt,
            "messages": messages,
            "media": media,
            "media_count": len(media) if media else 0,
            "timestamp": time.time()
        }
        generate_calls.append(call_info)

        print(f"\n📞 PROVIDER CALL from server:")
        print(f"   Prompt: '{prompt}'")
        print(f"   Messages: {messages}")
        print(f"   Media: {media}")
        print(f"   Media count: {len(media) if media else 0}")

        if media:
            print(f"   Media details:")
            for i, media_item in enumerate(media):
                print(f"     Item {i}: {type(media_item)} - {media_item}")
                if isinstance(media_item, str):
                    if os.path.exists(media_item):
                        size = os.path.getsize(media_item)
                        print(f"       File exists, size: {size} bytes")
                    else:
                        print(f"       File does not exist!")

        # Call original method
        return original_generate_internal(self, prompt, messages, system_prompt, tools, media, **kwargs)

    def debug_single_generate(self, payload):
        """Debug wrapper for _single_generate"""

        payload_info = {
            "model": payload.get("model"),
            "messages_count": len(payload.get("messages", [])),
            "timestamp": time.time()
        }

        print(f"\n📤 PAYLOAD TO LMSTUDIO:")
        print(f"   Model: {payload.get('model')}")
        print(f"   Messages: {len(payload.get('messages', []))}")

        for i, msg in enumerate(payload.get('messages', [])):
            content = msg.get('content')
            print(f"   Message {i}: role={msg.get('role')}")

            if isinstance(content, str):
                print(f"     Content (string): {content[:100]}...")
            elif isinstance(content, list):
                print(f"     Content (array): {len(content)} items")
                has_image = False
                for j, item in enumerate(content):
                    item_type = item.get('type', 'unknown')
                    if item_type == 'image_url':
                        has_image = True
                        url = item.get('image_url', {}).get('url', '')
                        if url.startswith('data:'):
                            print(f"       Item {j}: image_url (data URL, length: {len(url)})")
                        else:
                            print(f"       Item {j}: image_url ({url})")
                    else:
                        print(f"       Item {j}: {item_type}")

                payload_info['has_image'] = has_image
            else:
                print(f"     Content: {type(content)}")

        payload_captures.append(payload_info)

        # Call original method
        return original_single_generate(self, payload)

    # Apply monkey patches
    LMStudioProvider._generate_internal = debug_generate_internal
    LMStudioProvider._single_generate = debug_single_generate

    return generate_calls, payload_captures

def make_server_request():
    """Make the failing server request"""

    import requests

    print(f"\n🌐 MAKING SERVER REQUEST...")

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
        "max_tokens": 100
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"📨 SERVER RESPONSE: {content[:200]}...")
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    # Set up debugging
    generate_calls, payload_captures = monkey_patch_provider_for_debugging()

    # Make the request
    success = make_server_request()

    # Analyze results
    print(f"\n📊 ANALYSIS:")
    print(f"   Provider calls: {len(generate_calls)}")
    print(f"   Payloads captured: {len(payload_captures)}")

    if generate_calls:
        for i, call in enumerate(generate_calls):
            print(f"\n   Call {i}:")
            print(f"     Media items: {call['media_count']}")
            print(f"     Media: {call['media']}")

    if payload_captures:
        for i, capture in enumerate(payload_captures):
            print(f"\n   Payload {i}:")
            print(f"     Has image: {capture.get('has_image', 'unknown')}")
            print(f"     Messages: {capture['messages_count']}")

    # Diagnosis
    if not generate_calls:
        print(f"\n❌ ISSUE: Provider was never called by server!")
    elif generate_calls[0]['media_count'] == 0:
        print(f"\n❌ ISSUE: Provider was called but no media was passed!")
    elif not payload_captures or not payload_captures[0].get('has_image', False):
        print(f"\n❌ ISSUE: Media was passed to provider but not included in LMStudio payload!")
    else:
        print(f"\n✅ Everything looks correct - investigate LMStudio configuration")

if __name__ == "__main__":
    main()
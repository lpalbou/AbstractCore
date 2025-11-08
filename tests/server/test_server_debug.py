#!/usr/bin/env python3

import sys
import os
import requests
import json

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_server_request_debug():
    """Test the server request and capture detailed logs"""

    print("🔍 TESTING SERVER REQUEST WITH DEBUG")
    print("=" * 50)

    # Test the exact same request that was failing
    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

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
        "max_tokens": 300
    }

    print("1. Sending request to server...")
    print(f"   URL: {url}")
    print(f"   Model: {payload['model']}")
    print(f"   Content items: {len(payload['messages'][0]['content'])}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        print(f"\n2. Response received:")
        print(f"   Status code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            print(f"   Response content: {content[:200]}...")

            # Check if model can see the image
            if "can't see" in content.lower() or "don't have access" in content.lower():
                print("   ❌ MODEL STILL CAN'T SEE IMAGES")
                return False
            else:
                print("   ✅ MODEL APPEARS TO SEE IMAGES")
                return True
        else:
            print(f"   ❌ Request failed: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Request error: {e}")
        return False

def test_provider_direct():
    """Test the provider directly to see if media processing works"""

    print("\n3. Testing provider directly...")

    try:
        from abstractcore.providers.lmstudio_provider import LMStudioProvider

        # Create provider
        provider = LMStudioProvider(model="qwen/qwen3-vl-4b", base_url="http://localhost:1234/v1")

        # Test with the same image URL
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        # Create messages as they would come from the server
        messages = [{"role": "user", "content": "What is in this image?"}]
        media_files = [image_url]  # This simulates what the server extracts

        print(f"   Testing with media files: {media_files}")

        # This should trigger the media processing that was failing
        response = provider.generate(
            prompt="",
            messages=messages,
            media=media_files,
            max_tokens=50
        )

        print(f"   Direct provider response: {response.content[:200]}...")

        if "can't see" in response.content.lower():
            print("   ❌ PROVIDER STILL CAN'T PROCESS IMAGES")
            return False
        else:
            print("   ✅ PROVIDER CAN PROCESS IMAGES")
            return True

    except Exception as e:
        print(f"   ❌ Provider test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing server and provider...")

    server_works = test_server_request_debug()
    provider_works = test_provider_direct()

    if server_works and provider_works:
        print("\n✅ SUCCESS: Both server and provider work!")
    elif provider_works and not server_works:
        print("\n⚠️ PARTIAL: Provider works, but server integration has issues")
    elif server_works and not provider_works:
        print("\n⚠️ PARTIAL: Server works, but provider has issues")
    else:
        print("\n❌ FAILURE: Both server and provider have issues")
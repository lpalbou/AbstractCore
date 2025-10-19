#!/usr/bin/env python3

import requests
import json

def test_direct_lmstudio():
    """Test direct connection to LMStudio"""

    print("🔍 TESTING DIRECT LMSTUDIO CONNECTION")
    print("=" * 50)

    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "qwen/qwen3-vl-4b",
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

    print(f"   Direct LMStudio URL: {url}")
    print(f"   Model: {payload['model']}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   LMStudio Response: {content[:150]}...")
            return content
        else:
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"   Exception: {e}")
        return None

def test_abstractcore_server():
    """Test AbstractCore server"""

    print(f"\n🔍 TESTING ABSTRACTCORE SERVER")
    print("=" * 50)

    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

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

    print(f"   AbstractCore URL: {url}")
    print(f"   Model: {payload['model']}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   AbstractCore Response: {content[:150]}...")
            return content
        else:
            print(f"   Error: {response.text}")
            return None

    except Exception as e:
        print(f"   Exception: {e}")
        return None

def compare_responses(lmstudio_response, abstractcore_response):
    """Compare the two responses"""

    print(f"\n📊 RESPONSE COMPARISON")
    print("=" * 50)

    if not lmstudio_response and not abstractcore_response:
        print("   ❌ Both requests failed")
        return

    if not lmstudio_response:
        print("   ❌ LMStudio direct request failed")
        return

    if not abstractcore_response:
        print("   ❌ AbstractCore request failed")
        return

    # Check if responses are identical or very similar
    if lmstudio_response == abstractcore_response:
        print("   ⚠️ IDENTICAL RESPONSES!")
        print("   This suggests AbstractCore is passing requests directly to LMStudio")
        print("   without using the AbstractCore provider system.")
    else:
        print("   ✅ Different responses - AbstractCore is processing the request")

    # Check for vision capability in both
    lmstudio_sees = not any(phrase in lmstudio_response.lower()
                           for phrase in ["can't see", "don't have access", "cannot see"])
    abstractcore_sees = not any(phrase in abstractcore_response.lower()
                               for phrase in ["can't see", "don't have access", "cannot see"])

    print(f"\n   LMStudio direct sees image: {'✅ YES' if lmstudio_sees else '❌ NO'}")
    print(f"   AbstractCore sees image: {'✅ YES' if abstractcore_sees else '❌ NO'}")

    if lmstudio_sees and not abstractcore_sees:
        print("\n   🚨 ISSUE IDENTIFIED:")
        print("   - LMStudio directly can see images")
        print("   - AbstractCore server cannot see images")
        print("   - This confirms the AbstractCore media processing is broken")
    elif not lmstudio_sees and not abstractcore_sees:
        print("\n   🤔 BOTH FAIL:")
        print("   - Neither LMStudio nor AbstractCore can see images")
        print("   - This suggests an LMStudio configuration issue")
    else:
        print("\n   ✅ Both work or AbstractCore works better")

def main():
    lmstudio_response = test_direct_lmstudio()
    abstractcore_response = test_abstractcore_server()
    compare_responses(lmstudio_response, abstractcore_response)

if __name__ == "__main__":
    main()
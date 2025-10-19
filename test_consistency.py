#!/usr/bin/env python3

import requests
import json
import time

def make_test_request(request_num):
    """Make a single test request"""

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

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        else:
            return f"ERROR: {response.status_code}"

    except Exception as e:
        return f"EXCEPTION: {e}"

def test_consistency():
    """Test multiple requests to see if responses are consistent"""

    print("🔍 TESTING RESPONSE CONSISTENCY")
    print("=" * 50)

    responses = []

    for i in range(5):
        print(f"   Request {i+1}...")
        response = make_test_request(i+1)
        responses.append(response)

        # Brief summary
        first_50 = response[:50] if isinstance(response, str) else str(response)[:50]
        print(f"     Response: {first_50}...")

        # Wait between requests
        time.sleep(2)

    print(f"\n📊 CONSISTENCY ANALYSIS:")

    # Categorize responses
    categories = {}
    for i, response in enumerate(responses):
        response_lower = response.lower() if isinstance(response, str) else str(response).lower()

        # Categorize by content
        category = "unknown"
        if "boardwalk" in response_lower or "path" in response_lower or "grass" in response_lower:
            category = "boardwalk"
        elif "firefighter" in response_lower or "orange" in response_lower or "suit" in response_lower:
            category = "firefighter"
        elif "object" in response_lower or "unidentifiable" in response_lower or "blurry" in response_lower:
            category = "blurry_object"
        elif "forest" in response_lower or "trees" in response_lower:
            category = "forest"
        elif "can't see" in response_lower or "don't have access" in response_lower:
            category = "no_vision"
        elif "error" in response_lower or "exception" in response_lower:
            category = "error"

        if category not in categories:
            categories[category] = []
        categories[category].append(i+1)

    print(f"   Response categories:")
    for category, request_nums in categories.items():
        print(f"     {category}: requests {request_nums}")

    # Check consistency
    unique_categories = len(categories)
    if unique_categories == 1:
        print(f"\n   ✅ CONSISTENT: All responses in same category")
    else:
        print(f"\n   ❌ INCONSISTENT: {unique_categories} different response types!")
        print(f"   This indicates non-deterministic behavior in the image processing pipeline.")

    return responses, categories

if __name__ == "__main__":
    responses, categories = test_consistency()

    if len(categories) > 1:
        print(f"\n🚨 ISSUE CONFIRMED: Inconsistent responses indicate unstable image processing!")
        print(f"   Possible causes:")
        print(f"   - Race conditions in media processing")
        print(f"   - Cached/stale image data")
        print(f"   - Concurrent request interference")
        print(f"   - Non-deterministic image download/processing")
    else:
        print(f"\n✅ Responses are consistent - issue may have been resolved")
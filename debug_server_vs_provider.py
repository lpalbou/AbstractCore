#!/usr/bin/env python3

import sys
import os
import requests
import json

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_comprehensive_comparison():
    """Compare server vs provider behavior for media processing"""

    print("🔍 COMPREHENSIVE SERVER VS PROVIDER COMPARISON")
    print("=" * 60)

    # Test 1: Server API
    print("1. TESTING SERVER API")
    print("-" * 30)

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

            print(f"   Server response: {content}")

            # Check for vision capability
            vision_keywords = ["can't see", "don't have access", "can't analyze", "cannot see", "unable to view"]
            server_sees_image = not any(keyword in content.lower() for keyword in vision_keywords)

            print(f"   Server sees image: {'✅ YES' if server_sees_image else '❌ NO'}")
        else:
            print(f"   ❌ Server error: {response.status_code}")
            server_sees_image = False

    except Exception as e:
        print(f"   ❌ Server request failed: {e}")
        server_sees_image = False

    # Test 2: Provider Direct
    print("\n2. TESTING PROVIDER DIRECTLY")
    print("-" * 30)

    try:
        from abstractcore.providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider(model="qwen/qwen3-vl-4b", base_url="http://localhost:1234/v1")

        # Download the image to a local file for the provider test
        import tempfile
        import urllib.request

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        print(f"   Downloading image for provider test...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            try:
                # Add headers to avoid 403 errors
                req = urllib.request.Request(image_url)
                req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                req.add_header('Accept', 'image/webp,image/apng,image/*,*/*;q=0.8')

                with urllib.request.urlopen(req) as response:
                    tmp_file.write(response.read())
                    temp_image_path = tmp_file.name

                print(f"   Image downloaded to: {temp_image_path}")

                # Test provider with local file
                messages = [{"role": "user", "content": "What is in this image?"}]

                provider_response = provider.generate(
                    prompt="",
                    messages=messages,
                    media=[temp_image_path],  # Use local file path
                    max_tokens=100
                )

                print(f"   Provider response: {provider_response.content}")

                # Check for vision capability
                provider_sees_image = not any(keyword in provider_response.content.lower() for keyword in vision_keywords)

                print(f"   Provider sees image: {'✅ YES' if provider_sees_image else '❌ NO'}")

                # Cleanup
                os.unlink(temp_image_path)

            except Exception as download_error:
                print(f"   ❌ Image download failed: {download_error}")
                provider_sees_image = False

    except Exception as e:
        print(f"   ❌ Provider test failed: {e}")
        provider_sees_image = False

    # Test 3: Analysis
    print("\n3. ANALYSIS")
    print("-" * 30)

    if server_sees_image and provider_sees_image:
        print("   ✅ SUCCESS: Both server and provider can process images!")
    elif provider_sees_image and not server_sees_image:
        print("   ⚠️ ISSUE: Provider works, but server integration is broken")
        print("   🔍 The fix worked for the provider, but the server isn't properly passing media")
    elif server_sees_image and not provider_sees_image:
        print("   ⚠️ ISSUE: Server integration works, but provider is broken")
        print("   🔍 This shouldn't happen with our fixes")
    else:
        print("   ❌ FAILURE: Neither server nor provider can process images")
        print("   🔍 More fixes needed")

    return server_sees_image, provider_sees_image

if __name__ == "__main__":
    server_works, provider_works = test_comprehensive_comparison()

    if not server_works:
        print("\n🚨 The original user issue is NOT fixed - server still doesn't pass images correctly!")
    else:
        print("\n✅ The original user issue is fixed!")
#!/usr/bin/env python3

import sys
import os
import json

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def debug_lmstudio_request_payload():
    """Debug what's actually being sent to LMStudio"""

    print("🔍 DEBUGGING LMSTUDIO REQUEST PAYLOAD")
    print("=" * 50)

    try:
        # Patch the LMStudioProvider to log what it sends
        from abstractcore.providers.lmstudio_provider import LMStudioProvider
        import tempfile
        import urllib.request

        print("1. Creating LMStudioProvider...")
        provider = LMStudioProvider(model="qwen/qwen3-vl-4b", base_url="http://localhost:1234/v1")

        print("2. Downloading test image...")
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            req = urllib.request.Request(image_url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            req.add_header('Accept', 'image/webp,image/apng,image/*,*/*;q=0.8')

            with urllib.request.urlopen(req) as response:
                tmp_file.write(response.read())
                temp_image_path = tmp_file.name

        print(f"   Image downloaded: {temp_image_path}")

        print("3. Testing provider's _generate_internal directly...")

        # Simulate the exact call that happens during server request
        messages = [{"role": "user", "content": "What is in this image?"}]
        media = [temp_image_path]

        print(f"   Input messages: {messages}")
        print(f"   Input media: {media}")

        # Monkey patch the _single_generate method to capture the payload
        original_single_generate = provider._single_generate
        captured_payload = None

        def capture_payload(payload):
            nonlocal captured_payload
            captured_payload = payload.copy()
            print(f"\n4. CAPTURED PAYLOAD SENT TO LMSTUDIO:")
            print(f"   Model: {payload.get('model')}")
            print(f"   Messages count: {len(payload.get('messages', []))}")

            for i, msg in enumerate(payload.get('messages', [])):
                print(f"   Message {i}:")
                print(f"     Role: {msg.get('role')}")
                content = msg.get('content')

                if isinstance(content, str):
                    print(f"     Content (string): {content[:100]}...")
                elif isinstance(content, list):
                    print(f"     Content (array): {len(content)} items")
                    for j, item in enumerate(content):
                        item_type = item.get('type', 'unknown')
                        print(f"       Item {j}: type={item_type}")
                        if item_type == 'text':
                            print(f"         Text: '{item.get('text', '')}'")
                        elif item_type == 'image_url':
                            url = item.get('image_url', {}).get('url', '')
                            if url.startswith('data:'):
                                print(f"         Image: data URL (length: {len(url)})")
                            else:
                                print(f"         Image: {url}")
                else:
                    print(f"     Content: {type(content)} - {content}")

            # Call original method
            return original_single_generate(payload)

        provider._single_generate = capture_payload

        try:
            # This should trigger the media processing and payload capture
            response = provider.generate(
                prompt="",
                messages=messages,
                media=media,
                max_tokens=50
            )

            print(f"\n5. RESPONSE ANALYSIS:")
            print(f"   Response: {response.content[:200]}...")

            # Analyze what we captured
            if captured_payload:
                messages_sent = captured_payload.get('messages', [])
                has_multimodal = False

                for msg in messages_sent:
                    content = msg.get('content')
                    if isinstance(content, list):
                        has_image = any(item.get('type') == 'image_url' for item in content if isinstance(item, dict))
                        if has_image:
                            has_multimodal = True
                            break

                print(f"   Contains multimodal content: {'✅ YES' if has_multimodal else '❌ NO'}")

                if not has_multimodal:
                    print("\n❌ ISSUE FOUND: No image data in the request to LMStudio!")
                    print("   This explains why the model can't see images.")
                else:
                    print("\n✅ PAYLOAD LOOKS CORRECT: Image data is included")

            else:
                print("   ❌ No payload was captured")

        finally:
            # Restore original method
            provider._single_generate = original_single_generate

        # Cleanup
        os.unlink(temp_image_path)

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lmstudio_request_payload()
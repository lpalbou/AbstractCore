#!/usr/bin/env python3

import sys
import os
import requests
import json
import time
import threading
import subprocess

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def monitor_server_logs():
    """Monitor server output while making a request"""

    print("🔍 MONITORING SERVER LOGS DURING REQUEST")
    print("=" * 50)

    # The request we want to test
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

    print("Making request and checking for any ImportError or media processing issues...")
    print("Look for these key log messages:")
    print("  - AutoMediaHandler processing")
    print("  - LMStudioProvider ImportError warnings")
    print("  - Media processing success/failure")
    print()

    try:
        # Make the request
        print("Sending request...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            print(f"Response content: {content}")

            # Check if it mentions not being able to see
            if "can't see" in content.lower() or "can't analyze" in content.lower() or "don't have access" in content.lower():
                print("\n❌ ISSUE: Model still can't see the image")
                print("This means either:")
                print("  1. Server isn't downloading the image properly")
                print("  2. Server isn't passing media to provider")
                print("  3. Provider is still hitting ImportError during server requests")
                print("  4. LMStudio isn't receiving proper multimodal format")
                return False
            else:
                print("\n✅ SUCCESS: Model can see the image")
                return True
        else:
            print(f"Request failed: {response.text}")
            return False

    except Exception as e:
        print(f"Request error: {e}")
        return False

def test_server_media_processing():
    """Test what the server does with media processing"""

    print("\n🧪 TESTING SERVER MEDIA PROCESSING COMPONENTS")
    print("=" * 50)

    try:
        # Test server's media processing directly
        from abstractcore.server.app import ChatMessage, process_message_content

        # Simulate the exact message content that comes in
        original_content = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]

        print("1. Testing server's process_message_content...")
        message = ChatMessage(role="user", content=original_content)
        clean_text, media_files = process_message_content(message)

        print(f"   Clean text: '{clean_text}'")
        print(f"   Media files: {media_files}")

        if media_files:
            print(f"   ✅ Server extracted {len(media_files)} media files")

            # Test if the files exist and are accessible
            for i, file_path in enumerate(media_files):
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"     File {i}: {file_path} (size: {file_size} bytes)")
                else:
                    print(f"     File {i}: {file_path} (❌ does not exist)")

        else:
            print("   ❌ Server didn't extract any media files")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Server media test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing server media processing...")

    # Test 1: Server media processing components
    server_processing_works = test_server_media_processing()

    # Test 2: Full request monitoring
    request_works = monitor_server_logs()

    if server_processing_works and request_works:
        print("\n✅ SUCCESS: Server media processing is working correctly!")
    elif server_processing_works and not request_works:
        print("\n⚠️ PARTIAL: Server can process media, but something fails during the full request")
    else:
        print("\n❌ FAILURE: Server media processing is broken")
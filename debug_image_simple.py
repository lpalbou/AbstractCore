#!/usr/bin/env python3

import sys
import os
import hashlib
import requests

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_direct_download():
    """Simple direct download test"""

    print("🔍 TESTING DIRECT IMAGE DOWNLOAD (FIXED)")
    print("=" * 50)

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }

        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()

        image_data = response.content
        print(f"   Downloaded size: {len(image_data)} bytes")

        image_hash = hashlib.md5(image_data).hexdigest()
        print(f"   MD5 hash: {image_hash}")

        return image_hash, len(image_data)

    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None

def test_server_processing():
    """Test server processing again"""

    print(f"\n🔍 TESTING SERVER PROCESSING (SIMPLIFIED)")
    print("=" * 50)

    try:
        from abstractcore.server.app import ChatMessage, process_message_content

        content = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]

        message = ChatMessage(role="user", content=content)
        clean_text, media_files = process_message_content(message)

        if media_files and len(media_files) > 0:
            file_path = media_files[0]
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                file_hash = hashlib.md5(file_data).hexdigest()
                print(f"   Server processed hash: {file_hash}")
                print(f"   Server processed size: {len(file_data)} bytes")

                return file_hash, len(file_data)

        return None

    except Exception as e:
        print(f"   ❌ Server processing failed: {e}")
        return None

def test_what_model_actually_sees():
    """Test what gets sent to the model by checking a working request"""

    print(f"\n🔍 TESTING WHAT MODEL ACTUALLY SEES")
    print("=" * 50)

    # Make a request and see if we can capture more details
    import requests
    import json

    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "lmstudio/qwen/qwen3-vl-4b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence"},
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
            print(f"   Model response: {content}")

            # Analyze the response to understand what image the model is seeing
            response_lower = content.lower()

            # Keywords that would indicate boardwalk/nature scene
            nature_keywords = ["boardwalk", "path", "grass", "field", "sky", "clouds", "nature", "green", "blue"]
            other_keywords = ["orange", "firefighter", "suit", "person", "frog", "cartoon"]

            nature_matches = sum(1 for keyword in nature_keywords if keyword in response_lower)
            other_matches = sum(1 for keyword in other_keywords if keyword in response_lower)

            print(f"   Nature keywords found: {nature_matches}")
            print(f"   Other keywords found: {other_matches}")

            if nature_matches > other_matches:
                print(f"   ✅ Model seems to see the correct boardwalk image")
                return True
            else:
                print(f"   ❌ Model sees something else (not the boardwalk)")
                return False
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False

def main():
    print("Simplified image debugging...")

    # Test 1: Direct download
    direct_result = test_direct_download()

    # Test 2: Server processing
    server_result = test_server_processing()

    # Test 3: What model sees
    model_sees_correct = test_what_model_actually_sees()

    # Analysis
    print(f"\n📊 ANALYSIS:")

    if direct_result and server_result:
        direct_hash, direct_size = direct_result
        server_hash, server_size = server_result

        print(f"   Direct download: {direct_hash} ({direct_size} bytes)")
        print(f"   Server processed: {server_hash} ({server_size} bytes)")

        if direct_hash == server_hash:
            print(f"   ✅ Server processed the same image as direct download")
            if not model_sees_correct:
                print(f"   🚨 BUT model sees wrong content - issue in provider pipeline!")
        else:
            print(f"   ❌ Server processed DIFFERENT image - issue in server download!")

    elif server_result and not direct_result:
        print(f"   ⚠️ Direct download failed but server works - server has better handling")
        if model_sees_correct:
            print(f"   ✅ Model sees correct content - everything working!")
        else:
            print(f"   ❌ Model sees wrong content - issue in provider pipeline!")

    else:
        print(f"   ❌ Cannot compare - both or server failed")

if __name__ == "__main__":
    main()
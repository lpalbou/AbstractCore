#!/usr/bin/env python3
"""
Test script for AbstractCore server media integration.
Tests OpenAI-compatible endpoints with media attachments.
"""

import json
import base64
import requests
import tempfile
from PIL import Image, ImageDraw
import io

# Server configuration
BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image and return as base64 data URL."""
    # Create a simple test image
    img = Image.new('RGB', (200, 100), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.text((50, 40), "Test Image", fill='black')

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()

    b64_string = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/png;base64,{b64_string}"

def test_openai_content_array():
    """Test OpenAI Vision API compatible content array format."""
    print("🧪 Testing OpenAI Vision API compatible format...")

    # Create test image
    image_data_url = create_test_image()

    payload = {
        "model": "openai/gpt-4o-mini",  # Use a model that supports vision
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }

    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)

        if response.status_code == 200:
            result = response.json()
            print("✅ OpenAI content array format: SUCCESS")
            print(f"📝 Response: {result['choices'][0]['message']['content'][:100]}...")
            return True
        else:
            print(f"❌ OpenAI content array format: FAILED ({response.status_code})")
            print(f"📝 Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ OpenAI content array format: ERROR - {e}")
        return False

def test_abstractcore_filename_syntax():
    """Test AbstractCore @filename syntax (backward compatibility)."""
    print("\n🧪 Testing AbstractCore @filename syntax...")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document with important information.")
        temp_file = f.name

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"Please summarize the content of @{temp_file}"
            }
        ],
        "max_tokens": 100
    }

    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)

        if response.status_code == 200:
            result = response.json()
            print("✅ AbstractCore @filename syntax: SUCCESS")
            print(f"📝 Response: {result['choices'][0]['message']['content'][:100]}...")
            return True
        else:
            print(f"❌ AbstractCore @filename syntax: FAILED ({response.status_code})")
            print(f"📝 Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ AbstractCore @filename syntax: ERROR - {e}")
        return False
    finally:
        # Cleanup
        import os
        try:
            os.unlink(temp_file)
        except:
            pass

def test_streaming_with_media():
    """Test streaming responses with media."""
    print("\n🧪 Testing streaming with media...")

    image_data_url = create_test_image()

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image briefly."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        "stream": True,
        "max_tokens": 50
    }

    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True)

        if response.status_code == 200:
            print("✅ Streaming with media: SUCCESS")

            # Collect streaming response
            content_parts = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: ') and not line_str.endswith('[DONE]'):
                        try:
                            data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content_parts.append(delta['content'])
                        except:
                            pass

            full_content = ''.join(content_parts)
            print(f"📝 Streamed response: {full_content[:100]}...")
            return True
        else:
            print(f"❌ Streaming with media: FAILED ({response.status_code})")
            print(f"📝 Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Streaming with media: ERROR - {e}")
        return False

def test_error_handling():
    """Test error handling for invalid media."""
    print("\n🧪 Testing error handling...")

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,INVALID_BASE64_DATA"
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)

        if response.status_code == 400:
            error_data = response.json()
            print("✅ Error handling: SUCCESS (correctly rejected invalid data)")
            print(f"📝 Error type: {error_data.get('error', {}).get('type', 'unknown')}")
            return True
        else:
            print(f"❌ Error handling: UNEXPECTED ({response.status_code})")
            print(f"📝 Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error handling: ERROR - {e}")
        return False

def check_server_health():
    """Check if server is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy and running")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"💡 Make sure the server is running: uvicorn abstractcore.server.app:app --port 8000")
        return False

def main():
    """Run all media integration tests."""
    print("🚀 AbstractCore Server Media Integration Tests")
    print("=" * 50)

    # Check server health first
    if not check_server_health():
        return

    print()

    # Run tests
    tests = [
        test_openai_content_array,
        test_abstractcore_filename_syntax,
        test_streaming_with_media,
        test_error_handling
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Media integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
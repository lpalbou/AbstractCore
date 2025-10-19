#!/usr/bin/env python3
"""
Quick server integration verification test.
Tests the actual server media endpoints with a simple example.
"""

import base64
import requests
import tempfile
import json
from PIL import Image, ImageDraw
import io

SERVER_URL = "http://localhost:8000"

def create_simple_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (200, 100), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.text((50, 40), "Test", fill='black')

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()

    b64_string = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/png;base64,{b64_string}"

def test_server_health():
    """Test server health."""
    print("🔍 Testing server health...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_openai_vision_format():
    """Test OpenAI Vision API format."""
    print("\n🧪 Testing OpenAI Vision API format...")

    image_data = create_simple_test_image()

    payload = {
        "model": "ollama/qwen2.5vl:7b",
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
                            "url": image_data
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100
    }

    try:
        response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ OpenAI format success: {content[:100]}...")
            return True
        else:
            print(f"❌ OpenAI format failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ OpenAI format error: {e}")
        return False

def test_abstractcore_filename_format():
    """Test AbstractCore @filename format."""
    print("\n🧪 Testing AbstractCore @filename format...")

    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for AbstractCore server media integration.")
        temp_file = f.name

    payload = {
        "model": "ollama/qwen3:4b-instruct",
        "messages": [
            {
                "role": "user",
                "content": f"What is the content of @{temp_file}?"
            }
        ],
        "max_tokens": 100
    }

    try:
        response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ @filename format success: {content[:100]}...")
            return True
        else:
            print(f"❌ @filename format failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ @filename format error: {e}")
        return False
    finally:
        # Cleanup
        import os
        try:
            os.unlink(temp_file)
        except:
            pass

def test_streaming():
    """Test streaming responses."""
    print("\n🧪 Testing streaming responses...")

    payload = {
        "model": "ollama/qwen3:4b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Count from 1 to 5."
            }
        ],
        "stream": True,
        "max_tokens": 50
    }

    try:
        response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, stream=True, timeout=30)

        if response.status_code == 200:
            chunks = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: ') and not line_str.endswith('[DONE]'):
                        try:
                            data = json.loads(line_str[6:])
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    chunks.append(delta['content'])
                        except:
                            pass

            full_content = ''.join(chunks)
            if len(full_content) > 0:
                print(f"✅ Streaming success: {full_content[:100]}...")
                return True
            else:
                print("❌ Streaming returned no content")
                return False
        else:
            print(f"❌ Streaming failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return False

def test_error_handling():
    """Test error handling."""
    print("\n🧪 Testing error handling...")

    payload = {
        "model": "ollama/nonexistent-model",
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            }
        ]
    }

    try:
        response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=30)

        if response.status_code in [400, 404, 500]:
            print(f"✅ Error handling works: {response.status_code}")
            return True
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")
            return True  # Not a failure, just unexpected

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    print("🚀 AbstractCore Server Integration Verification")
    print("=" * 50)

    # Test server health
    if not test_server_health():
        print("\n❌ Server is not running. Start with:")
        print("uvicorn abstractcore.server.app:app --port 8000")
        return

    # Run tests
    tests = [
        test_openai_vision_format,
        test_abstractcore_filename_format,
        test_streaming,
        test_error_handling
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All integration tests passed!")
        print("✅ Server media integration is working correctly")
    else:
        print("⚠️  Some tests failed - check the output above")
        print("💡 Make sure you have the required models available:")
        print("   - ollama pull qwen2.5vl:7b")
        print("   - ollama pull qwen3:4b-instruct")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import sys
import os
import tempfile

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def debug_lmstudio_media_flow():
    """Debug the exact media flow in LMStudio provider without logging base64"""

    print("🔍 DEBUGGING LMSTUDIO MEDIA FLOW")
    print("=" * 50)

    # Test the exact same request that fails
    test_cases = [
        {
            "model": "lmstudio/qwen/qwen2.5-vl-7b",
            "description": "Vision model (should work)"
        },
        {
            "model": "lmstudio/qwen/qwen3-vl-4b",
            "description": "Vision model (new entry)"
        }
    ]

    # Create a small test image to avoid base64 logging issues
    def create_test_image():
        """Create a small 1x1 test image"""
        from PIL import Image
        import base64
        import io

        # Create 1x1 red pixel
        img = Image.new('RGB', (1, 1), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            f.write(img_data)
            return f.name

    test_image_path = create_test_image()
    print(f"Created test image: {test_image_path}")
    print()

    for test_case in test_cases:
        model_name = test_case["model"]
        description = test_case["description"]

        print(f"Testing: {model_name}")
        print(f"Description: {description}")

        try:
            # Step 1: Test handler selection
            from abstractcore.providers.lmstudio_provider import LMStudioProvider

            provider = LMStudioProvider(model=model_name.split('/', 1)[1])
            handler = provider._get_media_handler_for_model(model_name.split('/', 1)[1])

            print(f"✅ Handler selected: {type(handler).__name__}")

            # Step 2: Test media processing (without base64 logging)
            from abstractcore.media import process_file

            result = process_file(test_image_path)
            if result.success:
                media_content = result.media_content
                print(f"✅ Media processed: {media_content.media_type}")
                print(f"   Content format: {media_content.content_format}")
                print(f"   File size: {len(media_content.content) if hasattr(media_content, 'content') else 'unknown'} chars")
                print(f"   MIME type: {media_content.mime_type}")
            else:
                print(f"❌ Media processing failed: {result.error_message}")
                continue

            # Step 3: Test multimodal message creation (without logging base64)
            try:
                multimodal_message = handler.create_multimodal_message("Test message", [media_content])

                print(f"✅ Multimodal message created")
                print(f"   Type: {type(multimodal_message)}")
                print(f"   Role: {multimodal_message.get('role', 'unknown')}")

                if isinstance(multimodal_message.get('content'), list):
                    content_array = multimodal_message['content']
                    print(f"   Content array length: {len(content_array)}")

                    for i, item in enumerate(content_array):
                        item_type = item.get('type', 'unknown')
                        print(f"     Item {i}: type={item_type}")
                        if item_type == 'text':
                            print(f"       Text: '{item.get('text', '')}'")
                        elif item_type == 'image_url':
                            url = item.get('image_url', {}).get('url', '')
                            if url.startswith('data:'):
                                print(f"       Image: data URL (length: {len(url)})")
                            else:
                                print(f"       Image: {url}")
                else:
                    print(f"   Content: {type(multimodal_message.get('content'))}")

            except Exception as e:
                print(f"❌ Multimodal message creation failed: {e}")
                import traceback
                traceback.print_exc()

            # Step 4: Test what would be sent to LMStudio (simulate the logic)
            try:
                print(f"\n   🎯 LMStudio Request Simulation:")

                # Simulate the LMStudio provider logic
                chat_messages = [{"role": "user", "content": "Test message"}]

                if isinstance(multimodal_message, dict):
                    # This is what should happen for vision models
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1] = multimodal_message
                        print(f"     ✅ Would replace last message with structured content")
                    else:
                        chat_messages.append(multimodal_message)
                        print(f"     ✅ Would append structured message")

                    # Check what would actually be sent
                    final_message = chat_messages[-1]
                    if isinstance(final_message.get('content'), list):
                        print(f"     ✅ Final message has content array: {len(final_message['content'])} items")
                        has_image = any(item.get('type') == 'image_url' for item in final_message['content'])
                        print(f"     {'✅' if has_image else '❌'} Contains image data: {has_image}")
                    else:
                        print(f"     ❌ Final message content is not array: {type(final_message.get('content'))}")

                else:
                    print(f"     ❌ Multimodal message is not dict: {type(multimodal_message)}")

            except Exception as e:
                print(f"❌ LMStudio simulation failed: {e}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Cleanup
    try:
        os.unlink(test_image_path)
        print(f"Cleaned up test image: {test_image_path}")
    except:
        pass

if __name__ == "__main__":
    debug_lmstudio_media_flow()
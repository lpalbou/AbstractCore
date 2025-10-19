#!/usr/bin/env python3

import sys
import os
import tempfile

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_lmstudio_provider_media_processing():
    """Test the exact scenario that's failing in LMStudioProvider"""

    print("🔍 TESTING LMSTUDIO PROVIDER MEDIA PROCESSING")
    print("=" * 50)

    try:
        # Step 1: Create LMStudioProvider instance
        print("1. Creating LMStudioProvider instance...")
        from abstractcore.providers.lmstudio_provider import LMStudioProvider

        # Use the correct model name as shown in the available models list
        provider = LMStudioProvider(model="qwen/qwen3-vl-4b", base_url="http://localhost:1234/v1")
        print("   ✅ LMStudioProvider created successfully")

        # Step 2: Create a test image file
        print("\n2. Creating test image file...")
        from PIL import Image
        import io

        # Create 1x1 test image
        img = Image.new('RGB', (1, 1), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            f.write(img_data)
            test_image_path = f.name

        print(f"   ✅ Test image created: {test_image_path}")

        # Step 3: Test the exact _process_media_content call that's failing
        print("\n3. Testing _process_media_content call...")

        # This is the exact call that's failing in the provider
        media_list = [test_image_path]  # List of file paths

        try:
            processed_media = provider._process_media_content(media_list)
            print(f"   ✅ SUCCESS: _process_media_content returned {len(processed_media)} items")

            # Check what we got back
            for i, media_content in enumerate(processed_media):
                print(f"     Item {i}: {type(media_content)} - {media_content.media_type}")

        except ImportError as e:
            print(f"   ❌ IMPORT ERROR: {e}")
            print("   This is the exact error that's causing the problem!")
            return False
        except Exception as e:
            print(f"   ❌ OTHER ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Step 4: Test the imports within the provider context
        print("\n4. Testing imports within provider context...")

        # Simulate what base.py is doing
        try:
            # This is the exact import that base.py does
            from abstractcore.media import AutoMediaHandler
            from abstractcore.media.types import MediaContent
            print("   ✅ Direct imports work within provider context")

            # Try to use them
            handler = AutoMediaHandler()
            result = handler.process_file(test_image_path)
            if result.success:
                print(f"   ✅ AutoMediaHandler processing works: {result.media_content.media_type}")
            else:
                print(f"   ❌ AutoMediaHandler processing failed: {result.error_message}")

        except ImportError as e:
            print(f"   ❌ IMPORT ERROR in provider context: {e}")
            return False

        # Step 5: Test the full provider generation flow (without actual LMStudio server)
        print("\n5. Testing provider generation flow (simulated)...")

        try:
            # Try to call the media processing part of _generate_internal
            # We can't test the full flow without LMStudio server, but we can test the media part

            chat_messages = []
            user_message_text = "Test message"
            media = [test_image_path]

            # This simulates the problematic section in _generate_internal (lines 144-182)
            if media:
                print("   Testing media processing section...")

                # CRITICAL: This is the exact line that's failing
                processed_media = provider._process_media_content(media)
                print(f"   ✅ Media processing successful: {len(processed_media)} items")

                # Test handler selection
                media_handler = provider._get_media_handler_for_model(provider.model)
                print(f"   ✅ Media handler selected: {type(media_handler).__name__}")

                # Test multimodal message creation
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)
                print(f"   ✅ Multimodal message created: {type(multimodal_message)}")

        except Exception as e:
            print(f"   ❌ PROVIDER FLOW ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Cleanup
        try:
            os.unlink(test_image_path)
        except:
            pass

        print("\n✅ ALL TESTS PASSED - LMStudioProvider media processing should work!")
        return True

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lmstudio_provider_media_processing()
    if not success:
        print("\n🚨 LMStudio provider test failed!")
        sys.exit(1)
    else:
        print("\n🎯 LMStudio provider test passed - the issue must be runtime-specific!")
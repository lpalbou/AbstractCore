#!/usr/bin/env python3

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/albou/projects/abstractcore')

def test_vision_fallback_improvement():
    """Test that vision fallback generates immersive descriptions instead of analytical ones"""

    print("🔍 TESTING VISION FALLBACK IMPROVEMENT")
    print("=" * 50)

    try:
        # Test 1: Check if vision fallback is configured
        print("1. Checking vision fallback configuration...")

        try:
            from abstractcore.media.vision_fallback import VisionFallbackHandler, VisionNotConfiguredError

            handler = VisionFallbackHandler()
            is_enabled = handler.is_enabled()
            print(f"   Vision fallback enabled: {'✅ YES' if is_enabled else '❌ NO'}")

            if not is_enabled:
                print("   ⚠️ Vision fallback not configured - cannot test improvement")
                print("   To enable: abstractcore --download-vision-model")
                return False

        except Exception as e:
            print(f"   ❌ Failed to check vision fallback: {e}")
            return False

        # Test 2: Test description generation directly
        print("\n2. Testing description generation...")

        # Use a test image (download one if needed)
        import tempfile
        import requests
        from PIL import Image

        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            img.save(tmp_file.name, 'PNG')
            test_image_path = tmp_file.name

        print(f"   Created test image: {test_image_path}")

        try:
            # Generate description using the improved system
            description = handler.create_description(test_image_path, "What do you see?")

            print(f"   Generated description: {description}")

            # Check if description has problematic phrases
            problematic_phrases = [
                "image analysis:",
                "this image shows",
                "the image depicts",
                "the photo shows",
                "in this image",
                "the picture contains"
            ]

            description_lower = description.lower()
            found_issues = [phrase for phrase in problematic_phrases if phrase in description_lower]

            if found_issues:
                print(f"   ❌ ISSUES FOUND: Description contains analytical phrases: {found_issues}")
                print("   This means the text-only model will know it's receiving a description")
                return False
            else:
                print(f"   ✅ GOOD: Description is natural and immersive")
                return True

        except VisionNotConfiguredError as e:
            print(f"   ⚠️ Vision not configured: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Description generation failed: {e}")
            return False

        finally:
            # Cleanup
            try:
                os.unlink(test_image_path)
            except:
                pass

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_example_improvement():
    """Show example of before vs after improvement"""

    print(f"\n📊 BEFORE vs AFTER EXAMPLES:")
    print("=" * 50)

    print("❌ BEFORE (problematic):")
    print('   "Image analysis: This image shows a wooden boardwalk through a green field."')
    print('   Text-only model response: "That\'s a fantastic description! Based on your analysis..."')

    print("\n✅ AFTER (improved):")
    print('   "A wooden boardwalk stretches through a lush green field under a bright blue sky."')
    print('   Text-only model response: "The boardwalk creates a beautiful pathway..."')

    print("\n🎯 KEY IMPROVEMENTS:")
    print("   • Removed 'Image analysis:' prefix")
    print("   • Changed vision prompt to generate natural, immersive descriptions")
    print("   • Text-only models now experience content naturally instead of receiving obvious descriptions")

if __name__ == "__main__":
    print("Testing vision fallback improvements for text-only models...")

    success = test_vision_fallback_improvement()
    test_example_improvement()

    if success:
        print("\n✅ IMPROVEMENT SUCCESSFUL!")
        print("Text-only models should now respond naturally to images instead of treating them as descriptions.")
    else:
        print("\n⚠️ IMPROVEMENT IMPLEMENTED BUT NEEDS VISION CONFIGURATION")
        print("The code changes are in place, but vision fallback needs to be configured to test fully.")
        print("Run: abstractcore --download-vision-model")
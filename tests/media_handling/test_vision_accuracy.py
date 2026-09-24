#!/usr/bin/env python3

import pytest
import sys
import os
import requests
from pathlib import Path

# Add the project root to the path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

@pytest.mark.network("downloads a remote image and asks a live vision model")
def test_current_vision_accuracy():
    """Test the current vision model accuracy with the Arc de Triomphe image"""

    print("🔍 TESTING VISION ACCURACY FOR LANDMARK IDENTIFICATION")
    print("=" * 60)

    try:
        from abstractcore.media.vision_fallback import VisionFallbackHandler
        import tempfile

        # Download the Arc de Triomphe image
        image_url = "https://www.cuddlynest.com/blog/wp-content/uploads/2024/03/arc-de-triomphe.jpg"

        print(f"1. Downloading test image: {image_url}")
        response = requests.get(image_url, timeout=30)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(response.content)
            test_image_path = tmp_file.name

        print(f"   Saved to: {test_image_path}")

        # Test the current vision fallback
        print(f"\n2. Testing current vision fallback...")

        handler = VisionFallbackHandler()
        description = handler.create_description(test_image_path, "What is in this image?")

        print(f"   Current description: {description}")

        # Analyze accuracy
        description_lower = description.lower()

        # Check if it correctly identifies Arc de Triomphe
        correct_identifiers = ["arc de triomphe", "arc du triomphe", "triumphal arch"]
        incorrect_identifiers = ["eiffel tower", "notre dame", "louvre"]

        found_correct = any(identifier in description_lower for identifier in correct_identifiers)
        found_incorrect = any(identifier in description_lower for identifier in incorrect_identifiers)

        print(f"\n3. Accuracy Analysis:")
        print(f"   Correct identification: {'✅ YES' if found_correct else '❌ NO'}")
        print(f"   Incorrect identification: {'❌ YES' if found_incorrect else '✅ NO'}")

        if found_correct and not found_incorrect:
            print(f"   🎯 RESULT: Accurate description!")
            return True
        elif found_incorrect:
            print(f"   ❌ RESULT: Inaccurate - misidentified landmark")
            return False
        else:
            print(f"   ⚠️ RESULT: Generic description - no specific landmark mentioned")
            return None

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False
    finally:
        try:
            os.unlink(test_image_path)
        except:
            pass

def suggest_better_vision_models():
    """Suggest better vision models for more accurate descriptions"""

    print(f"\n💡 VISION MODEL RECOMMENDATIONS")
    print("=" * 50)

    print("📊 Current Model: lmstudio/google/gemma-3n-e4b")
    print("   Issue: Gemma vision models less accurate for landmark identification")

    print(f"\n✅ BETTER ALTERNATIVES:")

    print(f"\n1. 🥇 BEST: Qwen2.5-VL models (if available)")
    print("   abstractcore --set-vision-provider lmstudio --model qwen/qwen2.5-vl-7b")
    print("   • Excellent landmark recognition")
    print("   • Highly accurate descriptions")
    print("   • Good balance of speed/accuracy")

    print(f"\n2. 🥈 GOOD: OpenAI GPT-4V (requires API key)")
    print("   abstractcore --set-vision-provider openai --model gpt-4o")
    print("   • Best-in-class accuracy")
    print("   • Excellent landmark identification")
    print("   • Requires OpenAI API key")

    print(f"\n3. 🥉 DECENT: Qwen2-VL models")
    print("   abstractcore --set-vision-provider lmstudio --model qwen/qwen2-vl-7b")
    print("   • Good accuracy")
    print("   • Better than Gemma for landmarks")

    print(f"\n4. 📱 LOCAL: Download better local model")
    print("   abstractcore --download-vision-model")
    print("   • Downloads BLIP model locally")
    print("   • No external dependencies")
    print("   • Decent accuracy for most scenes")

def show_prompt_improvement():
    """Show the prompt improvement made"""

    print(f"\n🔧 PROMPT IMPROVEMENT")
    print("=" * 50)

    print("❌ OLD PROMPT:")
    print('   "Write 2-3 natural, immersive sentences about what you see, as if experiencing it directly."')

    print("\n✅ NEW PROMPT:")
    print('   "Look carefully and describe exactly what you see in 2-3 natural sentences, as if experiencing it directly."')
    print('   "Be precise about specific landmarks, buildings, objects, and details."')
    print('   "If you recognize specific places or things, name them accurately."')

    print(f"\n🎯 IMPROVEMENTS:")
    print("   • Added 'Look carefully' for attention")
    print("   • Added 'exactly what you see' for precision")
    print("   • Added 'Be precise about specific landmarks'")
    print("   • Added 'name them accurately' for correct identification")
    print("   • Maintained immersive style for text-only models")

if __name__ == "__main__":
    print("Testing vision accuracy and suggesting improvements...")

    accuracy_result = test_current_vision_accuracy()
    suggest_better_vision_models()
    show_prompt_improvement()

    print(f"\n📋 SUMMARY:")
    if accuracy_result is True:
        print("✅ Current setup works well with improved prompt!")
    elif accuracy_result is False:
        print("❌ Current vision model needs replacement for better accuracy")
        print("🔧 Consider switching to Qwen2.5-VL or GPT-4V")
    else:
        print("⚠️ Current model gives generic descriptions")
        print("🔧 Better vision model recommended for landmark identification")

    print(f"\n🎯 Next step: Test the same request again to see if prompt improvement helps!")

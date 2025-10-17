#!/usr/bin/env python3
"""
Debug resolution issue for gemma3n models on Ollama
Test different resolutions to see which one works
"""

import tempfile
from pathlib import Path
from PIL import Image
from abstractcore import create_llm

def test_resolutions_for_model(model_name, original_image_path, test_prompt="What do you see in this image?"):
    """Test different resolutions for a specific model."""
    print(f"\n🔍 Testing resolutions for {model_name}")
    print("-" * 50)

    # Test different resolution targets
    resolutions = [
        (224, 224),    # Very small - common vision model input
        (336, 336),    # Small square
        (512, 512),    # Medium square
        (768, 768),    # Large square
        (896, 896),    # Gemma3 might prefer this specific size
        (1024, 768),   # Current default (wide)
        (768, 1024),   # Current default (tall)
    ]

    # Load original image
    original_img = Image.open(original_image_path)
    print(f"Original image size: {original_img.size}")

    for target_width, target_height in resolutions:
        print(f"\n🖼️  Testing resolution: {target_width}x{target_height}")

        try:
            # Create resized image
            resized_img = original_img.resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                resized_img.save(temp_file.name, 'JPEG', quality=85)
                temp_path = temp_file.name

            # Test with model
            llm = create_llm("ollama", model=model_name)
            response = llm.generate(test_prompt, media=[temp_path])

            # Clean up
            Path(temp_path).unlink()

            # Show result
            result_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
            print(f"   📝 Response: {result_preview}")

            # Check if it correctly identifies mountain/trail/fence keywords
            content_lower = response.content.lower()
            correct_keywords = ['mountain', 'trail', 'fence', 'path', 'sky', 'sun', 'hiking']
            wrong_keywords = ['cake', 'phone', 'smartphone', 'hand', 'person', 'food']

            correct_count = sum(1 for kw in correct_keywords if kw in content_lower)
            wrong_count = sum(1 for kw in wrong_keywords if kw in content_lower)

            if correct_count >= 2 and wrong_count == 0:
                print(f"   ✅ LIKELY CORRECT (score: {correct_count}/7 correct keywords)")
            elif wrong_count > 0:
                print(f"   ❌ HALLUCINATION DETECTED (wrong keywords: {wrong_count})")
            else:
                print(f"   ⚠️  UNCLEAR (score: {correct_count}/7 correct keywords)")

        except Exception as e:
            print(f"   ❌ Error: {e}")

# Test the problematic models
def main():
    image_path = "tests/vision_examples/mystery1_mp.jpg"

    problematic_models = ["gemma3n:e4b", "gemma3n:e2b"]
    working_model = "qwen2.5vl:7b"  # For comparison

    print("🚨 RESOLUTION DEBUGGING FOR OLLAMA VISION MODELS")
    print("=" * 60)

    # Test one working model first as baseline
    print(f"\n📋 BASELINE TEST - Working model: {working_model}")
    test_resolutions_for_model(working_model, image_path)

    # Test problematic models
    for model in problematic_models:
        print(f"\n🔧 PROBLEMATIC MODEL: {model}")
        test_resolutions_for_model(model, image_path)

if __name__ == "__main__":
    main()
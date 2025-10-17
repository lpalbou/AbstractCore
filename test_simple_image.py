#!/usr/bin/env python3
"""
Test with a simple, small image to isolate the resolution/encoding issue
"""

from PIL import Image
import tempfile
from pathlib import Path
from abstractcore import create_llm

def create_simple_test_image():
    """Create a simple, small test image."""
    # Create a simple 100x100 red square with blue border
    img = Image.new('RGB', (100, 100), 'red')

    # Add blue border
    for x in range(100):
        for y in range(100):
            if x < 5 or x > 94 or y < 5 or y > 94:
                img.putpixel((x, y), (0, 0, 255))  # Blue border

    # Save to temp file
    temp_file = Path(tempfile.mktemp(suffix='.png'))
    img.save(temp_file)
    return temp_file

def test_with_simple_image():
    """Test problematic models with a simple image."""
    simple_image = create_simple_test_image()
    print(f"Created simple test image: {simple_image}")

    # Test failing models with simple image
    failing_models = ["gemma3n:e4b", "gemma3n:e2b"]
    working_models = ["gemma3:4b", "qwen2.5vl:7b"]

    print("\n🔴 Testing FAILING models with simple image:")
    for model in failing_models:
        print(f"\n🔍 {model}:")
        try:
            llm = create_llm("ollama", model=model)
            response = llm.generate(
                "What color is this image? What shape do you see? Be very specific.",
                media=[str(simple_image)]
            )
            print(f"Response: {response.content[:200]}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n🟢 Testing WORKING models with simple image:")
    for model in working_models:
        print(f"\n🔍 {model}:")
        try:
            llm = create_llm("ollama", model=model)
            response = llm.generate(
                "What color is this image? What shape do you see? Be very specific.",
                media=[str(simple_image)]
            )
            print(f"Response: {response.content[:200]}")
        except Exception as e:
            print(f"Error: {e}")

    # Cleanup
    simple_image.unlink()

if __name__ == "__main__":
    test_with_simple_image()
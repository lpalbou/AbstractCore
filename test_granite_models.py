#!/usr/bin/env python3
"""
Quick test script for IBM Granite vision models
"""
import sys
from pathlib import Path

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent))

from abstractcore import create_llm

def test_granite_models():
    """Test both granite models with a simple vision query."""
    models = ["granite3.2-vision:2b", "granite3.3-vision:2b"]
    image_path = "tests/vision_examples/mystery1_mp.jpg"

    print("🎯 TESTING GRANITE VISION MODELS")
    print("=" * 50)

    for model in models:
        print(f"\n🔍 Testing {model}")
        print("-" * 30)

        try:
            # Create LLM instance
            llm = create_llm("ollama", model=model)

            # Simple vision test
            response = llm.generate(
                "What do you see in this image? Describe it briefly.",
                media=[image_path]
            )

            print(f"✅ Success! Response ({len(response.content)} chars):")
            print(f"   {response.content[:200]}...")

        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_granite_models()
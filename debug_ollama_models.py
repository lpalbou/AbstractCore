#!/usr/bin/env python3
"""
Debug Ollama model differences - why do some work and others fail?
"""

import json
from abstractcore import create_llm

def test_ollama_model(model_name, test_image_path):
    """Test a specific Ollama model and show what's being sent."""
    print(f"\n🔍 Testing {model_name}")
    print("-" * 50)

    try:
        llm = create_llm("ollama", model=model_name)

        # Test with a simple prompt
        response = llm.generate(
            "What do you see in this image? Just tell me briefly.",
            media=[test_image_path]
        )

        print(f"✅ SUCCESS: {response.content[:100]}...")

    except Exception as e:
        print(f"❌ FAILED: {e}")

def main():
    test_image = "tests/vision_examples/mystery1_mp.jpg"

    # Test working models
    print("🟢 WORKING MODELS:")
    working_models = ["qwen2.5vl:7b", "gemma3:4b", "llama3.2-vision:11b"]
    for model in working_models:
        test_ollama_model(model, test_image)

    # Test failing models
    print("\n🔴 FAILING MODELS:")
    failing_models = ["gemma3n:e4b", "gemma3n:e2b"]
    for model in failing_models:
        test_ollama_model(model, test_image)

if __name__ == "__main__":
    main()
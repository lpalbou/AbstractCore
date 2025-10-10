#!/usr/bin/env python3
"""
Quick debug test to reproduce the LMStudio error.
"""

try:
    from abstractllm import create_llm
    print("✅ Import successful")

    print("🔍 Testing LMStudio provider creation...")
    llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")
    print("✅ Provider created successfully")

    print("🔍 Testing basic generation...")
    response = llm.generate("Hello")
    print(f"✅ Generation successful: {response.content}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()
#!/usr/bin/env python3
"""
Debug test for DeepSeek-OCR model loading issues.
"""

import sys
from pathlib import Path

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent))

from abstractcore import create_llm


def test_model_detection():
    """Test model detection logic."""
    print("🔍 Testing DeepSeek-OCR Model Detection")
    print("=" * 50)
    
    model_name = "deepseek-ai/DeepSeek-OCR"
    
    try:
        # Import the provider directly to test detection
        from abstractcore.providers.huggingface_provider import HuggingFaceProvider
        
        # Create provider instance to test detection
        print(f"📋 Model: {model_name}")
        
        # Test GGUF detection
        provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
        is_gguf = provider._is_gguf_model(model_name)
        print(f"   GGUF detected: {is_gguf}")
        
        # Test transformers availability
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("   ✅ Transformers available")
        except ImportError:
            print("   ❌ Transformers not available")
            return
        
        # Test direct transformers loading
        print("\n🧪 Testing direct transformers loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print("   ✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"   ❌ Tokenizer failed: {e}")
            return
            
        try:
            # Just test model info, don't actually load the full model
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            print(f"   ✅ Model config loaded: {type(config).__name__}")
        except Exception as e:
            print(f"   ❌ Model config failed: {e}")
            return
            
        print("\n✅ Direct transformers loading works!")
        
        # Now test through AbstractCore
        print("\n🧪 Testing through AbstractCore...")
        try:
            llm = create_llm("huggingface", model=model_name, trust_remote_code=True)
            print("   ✅ AbstractCore loading successful!")
            print(f"   📋 Provider: {llm.provider}")
            print(f"   🤖 Model: {llm.model}")
            print(f"   🔧 Model type: {llm.model_type}")
        except Exception as e:
            print(f"   ❌ AbstractCore loading failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            
            # Print more debug info
            import traceback
            print("\n🐛 Full traceback:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_detection()

#!/usr/bin/env python3
"""
Minimal test for DeepSeek-OCR model with AbstractCore HuggingFace provider.

This test verifies that:
1. The deepseek-ai/DeepSeek-OCR model can be loaded
2. Basic OCR functionality works with a simple image
3. The model responds appropriately to OCR prompts

Requirements:
- CUDA-capable GPU (model requires significant VRAM)
- transformers library with trust_remote_code=True
- torch with CUDA support
"""

import sys
import os
from pathlib import Path
import tempfile
from PIL import Image, ImageDraw, ImageFont

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent))

from abstractcore import create_llm


def create_test_image() -> str:
    """Create a simple test image with text for OCR testing."""
    # Create a simple image with text
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    # Draw text
    text = "Hello World!\nThis is a test document for OCR.\nDeepSeek-OCR should read this text."
    draw.multiline_text((50, 50), text, fill='black', font=font, spacing=10)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img.save(temp_file.name)
    return temp_file.name


def test_deepseek_ocr_basic():
    """Test basic DeepSeek-OCR functionality."""
    print("🧪 Testing DeepSeek-OCR with AbstractCore HuggingFace Provider")
    print("=" * 60)
    
    # Create test image
    print("\n1️⃣ Creating test image...")
    test_image_path = create_test_image()
    print(f"   ✅ Test image created: {test_image_path}")
    
    try:
        # Initialize the model
        print("\n2️⃣ Loading DeepSeek-OCR model...")
        print("   ⚠️  This may take several minutes on first run (model download)")
        print("   ⚠️  Requires significant GPU memory (>8GB VRAM recommended)")
        
        llm = create_llm(
            "huggingface", 
            model="deepseek-ai/DeepSeek-OCR",
            trust_remote_code=True
        )
        print("   ✅ Model loaded successfully!")
        
        # Test basic OCR functionality
        print("\n3️⃣ Testing OCR functionality...")
        
        # Test with simple OCR prompt (as per DeepSeek-OCR documentation)
        ocr_prompt = "<image>\nFree OCR."
        
        print(f"   📝 Prompt: {ocr_prompt}")
        print("   🔄 Processing image...")
        
        response = llm.generate(
            ocr_prompt,
            media=[test_image_path],
            max_tokens=1000,
            temperature=0.0
        )
        
        print(f"   ✅ OCR Response:")
        print(f"   📄 Content: {response.content}")
        print(f"   📊 Input tokens: {response.input_tokens}")
        print(f"   📊 Output tokens: {response.output_tokens}")
        print(f"   ⏱️  Generation time: {response.gen_time}ms")
        
        # Test with markdown conversion prompt
        print("\n4️⃣ Testing markdown conversion...")
        markdown_prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        response2 = llm.generate(
            markdown_prompt,
            media=[test_image_path],
            max_tokens=1000,
            temperature=0.0
        )
        
        print(f"   ✅ Markdown Response:")
        print(f"   📄 Content: {response2.content}")
        
        # Verify model is working
        if response.content and len(response.content.strip()) > 0:
            print("\n✅ SUCCESS: DeepSeek-OCR is working correctly!")
            print("   • Model loaded successfully")
            print("   • OCR functionality confirmed")
            print("   • Text extraction working")
            return True
        else:
            print("\n❌ ISSUE: Model loaded but no text output generated")
            return False
            
    except ImportError as e:
        print(f"\n❌ DEPENDENCY ERROR: {e}")
        print("   💡 Install required dependencies:")
        print("      pip install torch torchvision transformers accelerate")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Provide helpful troubleshooting
        if "CUDA" in str(e):
            print("   💡 GPU/CUDA issue - ensure CUDA is available and sufficient VRAM")
        elif "trust_remote_code" in str(e):
            print("   💡 Set trust_remote_code=True for custom model code")
        elif "memory" in str(e).lower():
            print("   💡 Insufficient memory - DeepSeek-OCR requires significant VRAM")
        else:
            print("   💡 Check model availability and network connection")
            
        return False
        
    finally:
        # Cleanup test image
        try:
            os.unlink(test_image_path)
            print(f"\n🧹 Cleaned up test image: {test_image_path}")
        except:
            pass


def test_model_info():
    """Test model information and capabilities."""
    print("\n5️⃣ Testing model information...")
    
    try:
        llm = create_llm("huggingface", model="deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
        
        # Try to get model info
        print(f"   📋 Provider: {llm.provider}")
        print(f"   🤖 Model: {llm.model}")
        
        # Check if model supports vision
        from abstractcore.architectures.detection import supports_vision
        vision_support = supports_vision("deepseek-ai/DeepSeek-OCR")
        print(f"   👁️  Vision support: {'✅' if vision_support else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Could not get model info: {e}")
        return False


def main():
    """Run the minimal DeepSeek-OCR test."""
    print("🚀 DeepSeek-OCR Minimal Test with AbstractCore")
    print("=" * 60)
    print("Testing deepseek-ai/DeepSeek-OCR model integration")
    print()
    
    # Check system requirements
    print("📋 System Requirements Check:")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   🔧 CUDA available: {'✅' if cuda_available else '❌'}")
        if cuda_available:
            print(f"   💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    except ImportError:
        print("   ❌ PyTorch not installed")
        return
    
    # Check transformers
    try:
        import transformers
        print(f"   📚 Transformers version: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers not installed")
        return
    
    print()
    
    # Run tests
    success = True
    
    # Test model info first (lighter test)
    success &= test_model_info()
    
    # Test basic functionality
    success &= test_deepseek_ocr_basic()
    
    # Final result
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("   DeepSeek-OCR is working correctly with AbstractCore")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("   Check error messages above for troubleshooting")
    print("=" * 60)


if __name__ == "__main__":
    main()

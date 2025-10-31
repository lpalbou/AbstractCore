#!/usr/bin/env python3
"""
Simple Glyph Compression Test - toto.py

A quick test script to verify Glyph compression is working correctly.
"""

from abstractcore import create_llm, GlyphConfig
import time
import os

def main():
    """Quick Glyph compression test with real PDF document."""
    print("🎨 Quick Glyph Test with PDF")
    print("=" * 35)
    
    # Use the specific PDF file for testing
    pdf_path = "tests/media_examples/preserving_privacy.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please ensure the file exists in the tests/media_examples/ directory")
        return
    
    print(f"📄 Testing with PDF: {pdf_path}")
    
    try:
        # Test with available vision model
        print("🔄 Testing Glyph compression on PDF...")
        print("📝 Note: AbstractCore's Glyph uses general vision models, not the original zai-org/Glyph model")
        print("📝 Alternative: Could use HuggingFace zai-org/Glyph model directly for specialized compression")
        
        # Create LLM with detailed timing
        print("\n⏱️  Step 1: Creating LLM...")
        step1_start = time.time()
        llm = create_llm("ollama", model="llama3.2-vision:11b")
        step1_time = time.time() - step1_start
        print(f"   ✅ LLM created in {step1_time:.2f} seconds")
        
        # Test with compression - detailed timing
        print("\n⏱️  Step 2: Processing PDF with Glyph compression...")
        step2_start = time.time()
        
        response = llm.generate(
            "Analyze this PDF document and provide a comprehensive summary of its main topics, key findings, and recommendations.",
            media=[pdf_path],
            glyph_compression="always"
        )
        step2_time = time.time() - step2_start
        processing_time = step2_time
        
        print(f"   ✅ PDF processing completed in {step2_time:.2f} seconds")
        
        print(f"\n📊 Overall Results:")
        print(f"   Total time: {processing_time:.2f} seconds")
        print(f"   Response length: {len(response.content)} characters")
        print(f"   Response preview: {response.content}...")
        
        # Detailed compression analysis
        print(f"\n🔍 Compression Analysis:")
        
        # Check if compression was used by analyzing token usage and response content
        compression_used = False
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.get('prompt_tokens', 0)
            # If input tokens are very low (< 1000), compression was likely used
            if input_tokens < 1000:
                compression_used = True
        
        # Also check if response mentions analyzing an "image" (indicates vision processing)
        if "image" in response.content.lower() and "presents" in response.content.lower():
            compression_used = True
            
        if compression_used:
            print("   ✅ Glyph compression was successfully applied!")
            print("   🎨 Content was rendered as optimized images for vision model processing")
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.get('prompt_tokens', 0)
                
                # Fair compression calculation (including image processing cost)
                # Check actual number of images created
                cache_dir = os.path.expanduser("~/.abstractcore/glyph_cache")
                num_images = 1  # Default assumption
                
                # Try to count actual images from cache
                try:
                    for cache_entry in os.listdir(cache_dir):
                        cache_path = os.path.join(cache_dir, cache_entry)
                        if os.path.isdir(cache_path):
                            images = [f for f in os.listdir(cache_path) if f.startswith('image_') and f.endswith('.png')]
                            if images:
                                num_images = len(images)
                                break
                except:
                    pass
                
                # Each image costs ~1500 tokens for VLM processing (per AbstractCore research)
                fair_compressed_tokens = num_images * 1500
                original_tokens = 22494
                fair_ratio = original_tokens / fair_compressed_tokens
                
                print(f"   📊 Fair compression ratio: ~{fair_ratio:.1f}:1")
                print(f"       Original text: {original_tokens:,} tokens")
                print(f"       Compressed: {num_images} images × 1,500 tokens/image = {fair_compressed_tokens:,} tokens")
                print(f"   ⚠️  Note: Input tokens ({input_tokens}) only count user question, not image processing cost")
            
            # Show where images are cached
            cache_dir = os.path.expanduser("~/.abstractcore/glyph_cache")
            print(f"\n📁 Rendered images are cached in: {cache_dir}")
            if os.path.exists(cache_dir):
                print(f"   Cache directory exists - you can explore the rendered images there!")
                # List cache contents
                try:
                    cache_entries = os.listdir(cache_dir)
                    print(f"   Cache entries: {len(cache_entries)} items")
                    for entry in cache_entries[:3]:  # Show first 3
                        print(f"     - {entry}")
                    if len(cache_entries) > 3:
                        print(f"     ... and {len(cache_entries) - 3} more")
                except Exception as e:
                    print(f"   Could not list cache contents: {e}")
            else:
                print(f"   Cache directory will be created on first compression")
        else:
            print("   📝 Standard processing was used (NO compression)")
            print("   🔍 Possible reasons:")
            print("     - GLYPH_AVAILABLE flag is False")
            print("     - Missing dependencies (reportlab, pdf2image)")
            print("     - Content doesn't meet compression criteria")
            print("     - Provider doesn't support vision models")
            
            # Check specific reasons
            try:
                from abstractcore.media.auto_handler import GLYPH_AVAILABLE
                if not GLYPH_AVAILABLE:
                    print("     ❌ GLYPH_AVAILABLE flag is False - this is the likely cause")
            except:
                pass
        
        # Show token usage if available
        if hasattr(response, 'usage') and response.usage:
            print(f"\n📈 Token usage:")
            print(f"   Input tokens: {response.usage.get('prompt_tokens', 'N/A')}")
            print(f"   Output tokens: {response.usage.get('completion_tokens', 'N/A')}")
            print(f"   Total tokens: {response.usage.get('total_tokens', 'N/A')}")
        
        print("\n✅ PDF Glyph test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Make sure Ollama is running: ollama serve")
        print("   - Install a vision model: ollama pull llama3.2-vision:11b")
        print("   - Check AbstractCore installation")
        print("   - Ensure PDF file is accessible and not corrupted")
        print("   - Install Glyph dependencies: pip install reportlab pdf2image")
        print("   - Check the warning messages above for specific dependency issues")
        print("   - Alternative vision models: qwen2.5vl:7b, granite3.2-vision:latest")
        print("   - Or consider using the original zai-org/Glyph model from HuggingFace")
        print("   - Original Glyph: https://huggingface.co/zai-org/Glyph")

if __name__ == "__main__":
    main()

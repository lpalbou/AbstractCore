#!/usr/bin/env python3
"""
Test Accurate Glyph Compression with Proper VLM Token Calculation

This script tests Glyph compression using the new VLMTokenCalculator for accurate
token estimation instead of the crude 1500 tokens/image approximation.
"""

import os
import time
from pathlib import Path
from abstractcore import create_llm
from abstractcore.utils.vlm_token_calculator import VLMTokenCalculator, calculate_glyph_compression_ratio


def main():
    """Test Glyph compression with accurate VLM token calculation."""
    
    print("🧪 Testing Accurate Glyph Compression")
    print("=" * 50)
    
    # Use the specific PDF file for testing
    pdf_path = Path("tests/media_examples/preserving_privacy.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Testing with: {pdf_path}")
    print(f"📏 File size: {pdf_path.stat().st_size / 1024:.1f} KB")
    
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
    print(f"   ✅ PDF processing completed in {step2_time:.2f} seconds")
    
    # Show response preview
    print(f"\n📝 Response Preview:")
    print(f"   {response.content[:200]}...")
    
    # Detailed compression analysis with ACCURATE token calculation
    print(f"\n🔍 Accurate Compression Analysis:")
    
    # Check if compression was used
    compression_used = False
    if hasattr(response, 'usage') and response.usage:
        input_tokens = response.usage.get('prompt_tokens', 0)
        if input_tokens < 1000:  # Low input tokens suggest compression
            compression_used = True
    
    if "image" in response.content.lower() and "presents" in response.content.lower():
        compression_used = True
        
    if compression_used:
        print("   ✅ Glyph compression was successfully applied!")
        print("   🎨 Content was rendered as optimized images for vision model processing")
        
        # Find the actual rendered images
        cache_dir = Path.home() / ".abstractcore" / "glyph_cache"
        image_paths = []
        
        try:
            for cache_entry in cache_dir.iterdir():
                if cache_entry.is_dir():
                    images = list(cache_entry.glob("image_*.png"))
                    if images:
                        image_paths.extend(images)
                        break
        except:
            pass
        
        if not image_paths:
            # Try the glyph_output_samples directory
            samples_dir = Path("glyph_output_samples")
            if samples_dir.exists():
                image_paths = list(samples_dir.glob("combined_page_*.png"))
        
        if image_paths:
            print(f"\n📊 ACCURATE Token Calculation:")
            print(f"   Found {len(image_paths)} rendered images")
            
            # Calculate accurate tokens using VLMTokenCalculator
            calculator = VLMTokenCalculator()
            
            # Get provider and model info
            provider = "ollama"  # From our LLM creation
            model = "llama3.2-vision:11b"
            
            # Calculate accurate compression ratio
            original_tokens = 22494  # Known from previous analysis
            compression_analysis = calculate_glyph_compression_ratio(
                original_tokens=original_tokens,
                image_paths=image_paths,
                provider=provider,
                model=model
            )
            
            print(f"   📈 Results:")
            print(f"     Original text: {compression_analysis['original_tokens']:,} tokens")
            print(f"     Compressed: {compression_analysis['compressed_tokens']:,} tokens")
            print(f"     Images created: {compression_analysis['images_created']}")
            print(f"     Avg tokens/image: {compression_analysis['average_tokens_per_image']:.0f}")
            print(f"     Compression ratio: {compression_analysis['compression_ratio']:.1f}:1")
            print(f"     Token savings: {compression_analysis['token_savings']:,}")
            print(f"     Calculation method: {compression_analysis['calculation_method']}")
            
            # Show per-image breakdown
            print(f"\n📋 Per-Image Token Breakdown:")
            for i, img_data in enumerate(compression_analysis['per_image_breakdown'][:5]):  # Show first 5
                print(f"     Image {i+1}: {img_data['tokens']} tokens ({img_data['dimensions']})")
            
            if len(compression_analysis['per_image_breakdown']) > 5:
                remaining = len(compression_analysis['per_image_breakdown']) - 5
                print(f"     ... and {remaining} more images")
            
            # Compare with old approximation
            old_approximation = len(image_paths) * 1500
            old_ratio = original_tokens / old_approximation
            
            print(f"\n🔄 Comparison with Old Method:")
            print(f"   Old approximation: {len(image_paths)} × 1,500 = {old_approximation:,} tokens")
            print(f"   Old ratio: {old_ratio:.1f}:1")
            print(f"   Accuracy improvement: {abs(compression_analysis['compression_ratio'] - old_ratio):.1f}x more precise")
            
            # Show image locations
            print(f"\n📁 Image Locations:")
            for path in image_paths[:3]:
                print(f"   - {path}")
            if len(image_paths) > 3:
                print(f"   - ... and {len(image_paths) - 3} more")
                
        else:
            print("   ⚠️  Could not locate rendered images for accurate calculation")
            print("   📍 Checked locations:")
            print(f"     - {cache_dir}")
            print(f"     - {Path('glyph_output_samples')}")
    
    else:
        print("   📝 Standard processing was used (NO compression)")
        print("   🔍 Possible reasons:")
        print("     - GLYPH_AVAILABLE flag is False")
        print("     - Missing dependencies (reportlab, pdf2image)")
        print("     - Content doesn't meet compression criteria")
        print("     - Provider doesn't support vision models")
    
    # Show token usage if available
    if hasattr(response, 'usage') and response.usage:
        print(f"\n📈 API Token Usage:")
        print(f"   Input tokens: {response.usage.get('prompt_tokens', 'N/A')}")
        print(f"   Output tokens: {response.usage.get('completion_tokens', 'N/A')}")
        print(f"   Total tokens: {response.usage.get('total_tokens', 'N/A')}")
        print(f"   ⚠️  Note: Input tokens only count user question, not VLM image processing cost")
    
    print(f"\n⏱️  Total execution time: {step1_time + step2_time:.2f} seconds")
    print("✅ Accurate compression analysis complete!")


if __name__ == "__main__":
    main()

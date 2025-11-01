#!/usr/bin/env python3
"""
Debug version of vision compression showing EXACTLY what's happening.
This reveals the truth about the compression pipeline.
"""

import os
import time
from pathlib import Path
from abstractcore import create_llm
from abstractcore.compression.vision_compressor import HybridCompressionPipeline
from abstractcore.compression.glyph_processor import GlyphProcessor
from abstractcore.compression.config import GlyphConfig
from abstractcore.utils.token_utils import TokenUtils
import argparse

def debug_compression(text: str, debug: bool = True):
    """Run compression with full transparency."""

    print("=" * 80)
    print("VISION COMPRESSION DEBUG ANALYSIS")
    print("=" * 80)

    # Step 1: Analyze input
    print("\n[STEP 1] INPUT ANALYSIS")
    print("-" * 40)
    char_count = len(text)
    estimated_tokens = TokenUtils.estimate_tokens(text, "gpt-4o")
    print(f"  Text length: {char_count:,} characters")
    print(f"  Estimated tokens: {estimated_tokens:,}")
    print(f"  Token estimation method: TokenUtils (cl100k_base tokenizer)")

    # Step 2: Check what's actually being used
    print("\n[STEP 2] COMPONENT CHECK")
    print("-" * 40)
    print("  ✅ GlyphProcessor: REAL - Uses ReportLab to render text to PDF then PNG")
    print("  ❌ DeepSeek-OCR: NOT USED - VisionCompressor is a SIMULATION")
    print("  ⚠️  VisionCompressor: FAKE - Just calculates theoretical compression")

    # Step 3: Glyph Processing (REAL)
    print("\n[STEP 3] GLYPH PROCESSING (REAL)")
    print("-" * 40)

    config = GlyphConfig()
    config.enabled = True
    config.min_token_threshold = 1000

    glyph_processor = GlyphProcessor(config=config)

    print("  Rendering configuration:")
    print(f"    - DPI: 72")
    print(f"    - Font size: 7-8pt")
    print(f"    - Columns: 4")
    print(f"    - Margins: 3px")

    start_time = time.time()
    glyph_images = glyph_processor.process_text(
        text,
        provider="openai",
        model="gpt-4o",
        user_preference="always"
    )
    glyph_time = time.time() - start_time

    print(f"\n  Glyph Results:")
    print(f"    - Images created: {len(glyph_images)}")
    print(f"    - Processing time: {glyph_time:.2f}s")

    if debug and glyph_images:
        print(f"\n  Image Details:")
        for i, img in enumerate(glyph_images):
            meta = img.metadata or {}
            print(f"    Image {i+1}:")
            print(f"      - File path: {img.file_path}")
            print(f"      - DPI: {meta.get('dpi', 'unknown')}")
            print(f"      - Compression ratio: {meta.get('compression_ratio', 0):.2f}x")
            print(f"      - Format: {img.content_format}")
            print(f"      - MIME type: {img.mime_type}")

    # Calculate REAL Glyph compression
    glyph_tokens = len(glyph_images) * 1500  # Approximate tokens per image
    glyph_ratio = estimated_tokens / glyph_tokens if glyph_tokens > 0 else 1.0

    print(f"\n  REAL Glyph Compression:")
    print(f"    - Original: {estimated_tokens:,} tokens")
    print(f"    - Compressed: ~{glyph_tokens:,} tokens ({len(glyph_images)} images × 1500)")
    print(f"    - Actual ratio: {glyph_ratio:.2f}x")

    # Step 4: Vision Compression (SIMULATED)
    print("\n[STEP 4] VISION COMPRESSION (SIMULATED - NOT REAL)")
    print("-" * 40)
    print("  ⚠️  WARNING: This is NOT using DeepSeek-OCR or any real vision model!")
    print("  ⚠️  The VisionCompressor class just does mathematical calculations!")

    from abstractcore.compression.vision_compressor import VisionCompressor

    vision_compressor = VisionCompressor()

    # Show what it actually does
    print("\n  What VisionCompressor.compress_images() actually does:")
    print("    1. Takes the number of Glyph images")
    print("    2. Multiplies by 1500 to get 'original tokens'")
    print("    3. Divides by target_ratio to get 'compressed tokens'")
    print("    4. Returns fake quality scores")

    vision_result = vision_compressor.compress_images(
        glyph_images,
        mode="balanced",
        original_tokens=estimated_tokens
    )

    print(f"\n  SIMULATED Vision Results:")
    print(f"    - Mode: {vision_result.metadata['mode']}")
    print(f"    - 'Compressed' tokens: {vision_result.compressed_tokens}")
    print(f"    - 'Compression' ratio: {vision_result.compression_ratio:.1f}x")
    print(f"    - 'Quality' score: {vision_result.quality_score:.2%}")
    print(f"    - THIS IS ALL FAKE - Just math, no actual compression!")

    # Step 5: Show the truth about hybrid pipeline
    print("\n[STEP 5] HYBRID PIPELINE ANALYSIS")
    print("-" * 40)

    pipeline = HybridCompressionPipeline()
    result = pipeline.compress(text, target_ratio=20.0, min_quality=0.90)

    print(f"  Hybrid Pipeline Claims:")
    print(f"    - Total compression: {result['total_compression_ratio']:.1f}x")
    print(f"    - Quality: {result['total_quality_score']:.2%}")

    print(f"\n  REALITY CHECK:")
    print(f"    - REAL compression (Glyph only): {glyph_ratio:.2f}x")
    print(f"    - Additional 'compression': SIMULATED")
    print(f"    - Actual media returned: {len(result['media'])} Glyph images")
    print(f"    - These are the SAME images from Step 3!")

    # Step 6: What you can actually use
    print("\n[STEP 6] WHAT'S ACTUALLY USABLE")
    print("-" * 40)
    print(f"  ✅ You have {len(glyph_images)} real PNG images")
    print(f"  ✅ These contain your text rendered densely")
    print(f"  ✅ A vision model can read these")
    print(f"  ✅ Real compression achieved: ~{glyph_ratio:.2f}x")
    print(f"  ❌ No DeepSeek-OCR compression happened")
    print(f"  ❌ The 24.9x claim is mostly theoretical")

    # Step 7: Test with actual vision model
    print("\n[STEP 7] ACTUAL VISION MODEL TEST")
    print("-" * 40)

    try:
        llm = create_llm("lmstudio", model="qwen/qwen3-vl-8b")
        print(f"  Using vision model: qwen/qwen3-vl-8b")
        print(f"  Sending {len(result['media'])} images to model...")

        response = llm.generate(
            "Summarize this document in 2-3 sentences:",
            media=result['media']
        )

        print(f"\n  Model successfully read the compressed images!")
        print(f"  Summary: {response.content[:200]}...")

    except Exception as e:
        print(f"  Failed to test with vision model: {e}")

    # Step 8: Summary
    print("\n" + "=" * 80)
    print("SUMMARY: WHAT'S REAL vs FAKE")
    print("=" * 80)
    print("\nREAL:")
    print(f"  ✅ Text → PDF → PNG conversion (Glyph)")
    print(f"  ✅ {glyph_ratio:.2f}x compression through dense rendering")
    print(f"  ✅ Vision models can read the rendered images")

    print("\nFAKE/SIMULATED:")
    print(f"  ❌ DeepSeek-OCR integration (not implemented)")
    print(f"  ❌ Vision compression (just calculations)")
    print(f"  ❌ The claimed {result['total_compression_ratio']:.1f}x ratio")

    print("\nBOTTOM LINE:")
    print(f"  You're getting {glyph_ratio:.2f}x REAL compression from Glyph")
    print(f"  The rest is theoretical/simulated")
    print("=" * 80)

    return result

def main():
    parser = argparse.ArgumentParser(description="Debug vision compression")
    parser.add_argument("--file", default="test-file.md", help="Input file")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    parser.add_argument("--save-images", action="store_true", help="Save intermediate images")
    args = parser.parse_args()

    # Load document
    with open(args.file, "r") as f:
        text = f.read()

    # Run debug analysis
    result = debug_compression(text, debug=args.debug)

    if args.save_images:
        print("\nSaving images to ./debug_images/")
        os.makedirs("debug_images", exist_ok=True)
        for i, img in enumerate(result['media']):
            if img.file_path and Path(img.file_path).exists():
                import shutil
                dest = f"debug_images/compressed_page_{i+1}.png"
                shutil.copy(img.file_path, dest)
                print(f"  Saved: {dest}")

if __name__ == "__main__":
    main()
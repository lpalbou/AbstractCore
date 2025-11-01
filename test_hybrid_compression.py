#!/usr/bin/env python3
"""
Test hybrid Glyph + Vision compression pipeline.
"""

import time
from pathlib import Path
from abstractcore.utils.token_utils import TokenUtils


def test_hybrid_compression():
    """Test the hybrid compression pipeline with preserving_privacy.pdf"""

    print("=" * 80)
    print("HYBRID COMPRESSION TEST (Glyph + Vision)")
    print("=" * 80)

    # Load PDF and extract text
    pdf_path = Path("tests/media_examples/preserving_privacy.pdf")
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    print(f"\n1. Loading and extracting text from PDF...")

    # Extract text
    from abstractcore.media.processors.pdf_processor import PDFProcessor
    from abstractcore.media.types import MediaType

    pdf_proc = PDFProcessor()
    text_content = pdf_proc._process_internal(pdf_path, MediaType.DOCUMENT)
    original_text = text_content.content

    # Calculate original tokens
    original_tokens = TokenUtils.estimate_tokens(original_text, "gpt-4o")

    print(f"   - Text length: {len(original_text)} characters")
    print(f"   - Estimated tokens: {original_tokens}")

    # Test baseline Glyph first
    print(f"\n2. Baseline Glyph Compression:")

    from abstractcore.compression.glyph_processor import GlyphProcessor
    from abstractcore.compression.config import GlyphConfig

    config = GlyphConfig()
    config.enabled = True
    config.min_token_threshold = 1000

    glyph_processor = GlyphProcessor(config=config)

    try:
        start_time = time.time()
        glyph_results = glyph_processor.process_text(
            original_text,
            provider="openai",
            model="gpt-4o",
            user_preference="always"
        )
        glyph_time = time.time() - start_time

        glyph_images = len(glyph_results)
        glyph_tokens = glyph_images * 1500  # Approximate
        glyph_ratio = original_tokens / glyph_tokens if glyph_tokens > 0 else 1.0

        print(f"   ✅ Glyph Results:")
        print(f"      - Images created: {glyph_images}")
        print(f"      - Estimated tokens: {glyph_tokens}")
        print(f"      - Compression ratio: {glyph_ratio:.2f}x")
        print(f"      - Processing time: {glyph_time:.2f}s")

    except Exception as e:
        print(f"   ❌ Glyph failed: {e}")
        return

    # Test hybrid compression
    print(f"\n3. Hybrid Compression (Glyph + Vision):")

    from abstractcore.compression.vision_compressor import HybridCompressionPipeline

    hybrid_pipeline = HybridCompressionPipeline(
        vision_provider="ollama",
        vision_model="llama3.2-vision"
    )

    # Test different compression modes
    modes = [
        ("Conservative", 10.0, 0.95),  # Low compression, high quality
        ("Balanced", 20.0, 0.90),      # Balanced
        ("Aggressive", 30.0, 0.85)      # High compression
    ]

    for mode_name, target_ratio, min_quality in modes:
        print(f"\n   Testing {mode_name} mode (target {target_ratio}x):")

        try:
            result = hybrid_pipeline.compress(
                original_text,
                target_ratio=target_ratio,
                min_quality=min_quality
            )

            print(f"      ✅ SUCCESS!")
            print(f"      - Original tokens: {result['original_tokens']}")
            print(f"      - Final tokens: {result['final_tokens']}")
            print(f"      - Total compression: {result['total_compression_ratio']:.1f}x")
            print(f"      - Quality score: {result['total_quality_score']:.2%}")
            print(f"      - Total time: {result['total_processing_time']:.2f}s")

            # Show stage breakdown
            print(f"\n      Stage breakdown:")
            glyph_stage = result['stages']['glyph']
            vision_stage = result['stages']['vision']

            print(f"        Glyph: {glyph_stage['images']} images, "
                  f"{glyph_stage['ratio']:.1f}x, {glyph_stage['time']:.2f}s")
            print(f"        Vision: {vision_stage['mode']} mode, "
                  f"{vision_stage['ratio']:.1f}x, {vision_stage['quality']:.2%} quality")

        except Exception as e:
            print(f"      ❌ Failed: {e}")

    # Summary comparison
    print(f"\n" + "=" * 80)
    print(f"COMPRESSION COMPARISON")
    print(f"=" * 80)

    print(f"\nOriginal text: {original_tokens} tokens")
    print(f"Glyph only: ~{glyph_tokens} tokens ({glyph_ratio:.1f}x compression)")

    # Show theoretical improvements with vision compression
    print(f"\nHybrid potential (simulated):")
    print(f"  Conservative: ~{original_tokens // 10} tokens (10x compression)")
    print(f"  Balanced: ~{original_tokens // 20} tokens (20x compression)")
    print(f"  Aggressive: ~{original_tokens // 30} tokens (30x compression)")

    print(f"\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_hybrid_compression()
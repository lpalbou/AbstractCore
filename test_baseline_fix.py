#!/usr/bin/env python3
"""
Fix and test Glyph compression with preserving_privacy.pdf
"""

import time
from pathlib import Path
from abstractcore import create_llm
from abstractcore.utils.token_utils import TokenUtils

def test_glyph_compression():
    """Test Glyph compression with proper configuration"""

    print("=" * 80)
    print("GLYPH COMPRESSION TEST (FIXED)")
    print("=" * 80)

    # Load PDF and extract text
    pdf_path = Path("tests/media_examples/preserving_privacy.pdf")
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    print(f"\n1. Loading and extracting text from PDF...")

    # Extract text using PDFProcessor
    from abstractcore.media.processors.pdf_processor import PDFProcessor
    from abstractcore.media.types import MediaType

    pdf_proc = PDFProcessor()
    text_content = pdf_proc._process_internal(pdf_path, MediaType.DOCUMENT)
    original_text = text_content.content

    # Calculate original tokens
    original_tokens = TokenUtils.estimate_tokens(original_text, "gpt-4o")

    print(f"   - Text length: {len(original_text)} characters")
    print(f"   - Estimated tokens: {original_tokens}")

    # Test Glyph compression
    print(f"\n2. Testing Glyph compression:")

    from abstractcore.compression.glyph_processor import GlyphProcessor
    from abstractcore.compression.config import GlyphConfig

    # Create Glyph processor with explicit configuration
    config = GlyphConfig()
    config.enabled = True
    config.min_token_threshold = 1000  # Lower threshold for testing

    glyph_processor = GlyphProcessor(config=config)

    # Force process the text
    print(f"\n   Testing Glyph compression (forcing enabled):")
    try:
        start_time = time.time()

        # Process with explicit parameters
        compressed_results = glyph_processor.process_text(
            original_text,
            provider="openai",  # Use openai as default
            model="gpt-4o",
            user_preference="always"  # Force compression even if quality is low
        )

        processing_time = time.time() - start_time

        if compressed_results:
            # Calculate actual compression
            num_images = len(compressed_results)

            # Get accurate token count from metadata
            total_compressed_tokens = 0
            for result in compressed_results:
                meta = result.metadata or {}
                # Look for token info in metadata
                if "compressed_tokens" in meta:
                    total_compressed_tokens += meta["compressed_tokens"]
                else:
                    # Fallback estimate
                    total_compressed_tokens += 1500

            # If we didn't get token counts, estimate based on images
            if total_compressed_tokens == 0:
                total_compressed_tokens = num_images * 1500

            actual_ratio = original_tokens / total_compressed_tokens if total_compressed_tokens > 0 else 1.0

            # Get quality score from first image
            quality_score = compressed_results[0].metadata.get("quality_score", 0.0) if compressed_results[0].metadata else 0.0

            print(f"      ✅ SUCCESS!")
            print(f"      - Original tokens: {original_tokens}")
            print(f"      - Compressed tokens: ~{total_compressed_tokens}")
            print(f"      - Compression ratio: {actual_ratio:.2f}x")
            print(f"      - Quality score: {quality_score:.2%}")
            print(f"      - Number of images: {num_images}")
            print(f"      - Processing time: {processing_time:.2f}s")

            # Show image details
            print(f"\n      Image details:")
            for i, img in enumerate(compressed_results, 1):
                meta = img.metadata or {}
                print(f"        Image {i}: DPI={meta.get('dpi', 'N/A')}, "
                      f"Compression={meta.get('compression_ratio', 'N/A'):.2f}x")

        else:
            print(f"      ⚠️ No results generated")

    except Exception as e:
        print(f"      ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_glyph_compression()
#!/usr/bin/env python3
"""
Baseline test for Glyph compression with preserving_privacy.pdf
"""

import time
from pathlib import Path
from abstractcore import create_llm
from abstractcore.media.processors.direct_pdf_processor import DirectPDFProcessor
from abstractcore.utils.token_utils import TokenUtils

def test_baseline_compression():
    """Test current Glyph implementation with preserving_privacy.pdf"""

    print("=" * 80)
    print("BASELINE GLYPH COMPRESSION TEST")
    print("=" * 80)

    # Load PDF
    pdf_path = Path("tests/media_examples/preserving_privacy.pdf")
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    print(f"\n1. Loading PDF: {pdf_path}")

    # Process with DirectPDFProcessor (current implementation)
    processor = DirectPDFProcessor()

    # Extract text first to count original tokens
    from abstractcore.media.processors.pdf_processor import PDFProcessor
    from abstractcore.media.types import MediaType
    pdf_proc = PDFProcessor()

    start_time = time.time()
    text_content = pdf_proc._process_internal(pdf_path, MediaType.DOCUMENT)
    extraction_time = time.time() - start_time

    original_text = text_content.content
    original_tokens = TokenUtils.estimate_tokens(original_text, "gpt-4o")

    print(f"\n2. Original document stats:")
    print(f"   - Text length: {len(original_text)} characters")
    print(f"   - Estimated tokens: {original_tokens}")
    print(f"   - Extraction time: {extraction_time:.2f}s")

    # Now test Glyph compression with different providers
    providers = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet"),
        ("ollama", "llama3.2-vision"),
    ]

    print(f"\n3. Testing Glyph compression with different providers:")

    # First test DirectPDFProcessor (direct PDF to image conversion)
    print(f"\n   A. Direct PDF to Image Conversion:")
    try:
        from abstractcore.media.types import MediaType
        start_time = time.time()

        # DirectPDFProcessor converts PDF to images directly
        compressed_result = processor._process_internal(
            pdf_path,
            MediaType.DOCUMENT
        )

        processing_time = time.time() - start_time

        # Extract stats from metadata
        metadata = compressed_result.metadata or {}
        total_images = metadata.get("total_images", 0)
        dpi = metadata.get("dpi", 0)

        print(f"      - Total images: {total_images}")
        print(f"      - DPI: {dpi}")
        print(f"      - Processing time: {processing_time:.2f}s")
        print(f"      - Status: ✅ Success")

    except Exception as e:
        print(f"      - Status: ❌ Failed: {e}")

    # Now test with GlyphProcessor directly
    print(f"\n   B. Glyph Text Compression (from extracted text):")
    from abstractcore.compression.glyph_processor import GlyphProcessor

    glyph_processor = GlyphProcessor()

    for provider, model in providers:
        print(f"\n   Testing {provider}/{model}:")
        try:
            # Check if Glyph can process this content
            can_process = glyph_processor.can_process(original_text, provider, model)

            if not can_process:
                print(f"      - Status: ⚠️ Skipped (provider doesn't support vision or text too short)")
                continue

            # Process with Glyph
            start_time = time.time()

            compressed_results = glyph_processor.process_text(
                original_text,
                provider=provider,
                model=model
            )

            processing_time = time.time() - start_time

            # Extract compression stats from first result
            if compressed_results:
                metadata = compressed_results[0].metadata or {}
                compression_ratio = metadata.get("compression_ratio", 1.0)
                quality_score = metadata.get("quality_score", 0.0)

                # Calculate compressed tokens from all images
                compressed_tokens = 0
                for result in compressed_results:
                    # Estimate tokens for each image
                    compressed_tokens += 1500  # Conservative estimate per image

                actual_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0

                print(f"      - Compression ratio: {compression_ratio:.2f}x (metadata)")
                print(f"      - Calculated ratio: {actual_ratio:.2f}x")
                print(f"      - Quality score: {quality_score:.2%}")
                print(f"      - Original tokens: {original_tokens}")
                print(f"      - Compressed tokens: ~{compressed_tokens}")
                print(f"      - Number of images: {len(compressed_results)}")
                print(f"      - Processing time: {processing_time:.2f}s")
                print(f"      - Status: ✅ Success")
            else:
                print(f"      - Status: ⚠️ No results generated")

        except Exception as e:
            print(f"      - Status: ❌ Failed: {e}")

    print("\n" + "=" * 80)
    print("BASELINE TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_baseline_compression()
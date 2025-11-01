#!/usr/bin/env python3
"""
Clean example of vision compression with AbstractCore.
Enhanced with timing measurements and detailed debug logging.
"""

import argparse
import time
import logging
from pathlib import Path

from abstractcore import create_llm
from abstractcore.compression.vision_compressor import HybridCompressionPipeline
from abstractcore.compression.optimizer import CompressionOptimizer
from abstractcore.compression.analytics import CompressionAnalytics
from abstractcore.compression.config import GlyphConfig, RenderingConfig
from abstractcore.utils.structured_logging import get_logger, configure_logging
from abstractcore.utils.token_utils import TokenUtils


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test vision compression with timing and debug info")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable detailed debug logging for full transparency")
    parser.add_argument("--file", type=str, default="untracked/2025-09-14-verbatim.md",
                       help="File to compress (default: untracked/2025-09-14-verbatim.md)")
    parser.add_argument("--target-ratio", type=float, default=20.0,
                       help="Target compression ratio (default: 20.0)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature (default: 0.7)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="Repetition penalty to reduce repeats (default: 1.1)")
    parser.add_argument("--frequency-penalty", type=float, default=0.3,
                       help="Frequency penalty for repetition control (default: 0.3)")
    
    # Glyph rendering controls
    parser.add_argument("--columns", type=int, default=4,
                       help="Number of columns for text layout (default: 4)")
    parser.add_argument("--width", type=int, default=None,
                       help="Target image width in pixels (default: 1024 for VLM optimization)")
    parser.add_argument("--height", type=int, default=None,
                       help="Target image height in pixels (default: 768 for VLM optimization)")
    parser.add_argument("--dpi", type=int, default=72,
                       help="Image resolution in DPI (default: 72, used when width/height not specified)")
    parser.add_argument("--font-size", type=int, default=8,
                       help="Font size for text rendering (default: 8)")
    parser.add_argument("--font", type=str, default=None,
                       help="Font name to use (e.g., 'Helvetica', 'Arial'). Falls back to default if not available.")
    parser.add_argument("--font-path", type=str, default=None,
                       help="Path to specific font file (e.g., 'abstractcore/assets/OCRA.ttf')")
    parser.add_argument("--margin-x", type=int, default=10,
                       help="Horizontal margin in pixels (default: 10)")
    parser.add_argument("--margin-y", type=int, default=10,
                       help="Vertical margin in pixels (default: 10)")
    
    # Text formatting options
    parser.add_argument("--render-format", action="store_true", default=True,
                       help="Enable markdown-like text formatting (default: True)")
    parser.add_argument("--no-render-format", dest="render_format", action="store_false",
                       help="Disable text formatting, render raw text as-is")
    
    # Comparison mode
    parser.add_argument("--no-compression", action="store_true",
                       help="Skip compression and send raw text directly to LLM for comparison")
    
    # LLM configuration
    parser.add_argument("--provider", type=str, default="lmstudio",
                       help="LLM provider (default: lmstudio)")
    parser.add_argument("--model", type=str, default="qwen/qwen3-vl-8b",
                       help="LLM model (default: qwen/qwen3-vl-8b)")
    
    args = parser.parse_args()

    # Configure logging based on debug flag
    if args.debug:
        configure_logging(
            console_level=logging.DEBUG,
            file_level=logging.DEBUG,
            log_dir="logs",
            verbatim_enabled=True,
            console_json=False,
            file_json=True
        )
        print("Debug mode enabled - detailed logging to console and logs/ directory")
    else:
        configure_logging(
            console_level=logging.INFO,
            file_level=logging.DEBUG,
            log_dir="logs",
            verbatim_enabled=False,
            console_json=False,
            file_json=True
        )

    # Get structured logger
    logger = get_logger(__name__)
    
    logger.info("Starting vision compression test", 
                file=args.file, 
                target_ratio=args.target_ratio,
                debug_mode=args.debug,
                no_compression=args.no_compression,
                provider=args.provider,
                model=args.model)

    # Setup
    logger.info("Initializing LLM and compression pipeline")
    setup_start = time.time()
    
    llm = create_llm(args.provider, model=args.model)
    
    # Create custom rendering configuration
    custom_rendering_config = RenderingConfig(
        columns=args.columns,
        dpi=args.dpi,
        target_width=args.width,
        target_height=args.height,
        font_size=args.font_size,
        font_name=args.font,
        font_path=args.font_path,
        line_height=args.font_size + 1,  # Slightly larger than font size
        # Use custom margins
        margin_x=args.margin_x,
        margin_y=args.margin_y,
        auto_crop_width=True,
        auto_crop_last_page=True,
        render_format=args.render_format
    )
    
    # Log font configuration
    if args.font_path:
        logger.info(f"Using custom font path: {args.font_path}")
    elif args.font:
        logger.info(f"Using custom font name: {args.font}")
    else:
        logger.info("Using default system font")
    
    # Create custom glyph configuration
    custom_glyph_config = GlyphConfig.default()
    custom_glyph_config.rendering = custom_rendering_config
    
    # CRITICAL: Disable provider optimization to prevent DPI override
    custom_glyph_config.provider_optimization = False
    
    # Also clear provider profiles to ensure our settings are used
    custom_glyph_config.provider_profiles = {}
    
    # Initialize pipeline with custom configuration
    pipeline = HybridCompressionPipeline()
    
    # Replace the default glyph processor with our custom configured one
    from abstractcore.compression.glyph_processor import GlyphProcessor
    pipeline.glyph_processor = GlyphProcessor(config=custom_glyph_config)
    
    setup_time = time.time() - setup_start
    logger.info("Setup completed", 
                setup_time_ms=setup_time * 1000,
                provider=args.provider,
                model=args.model,
                custom_columns=args.columns,
                custom_dpi=args.dpi,
                custom_font_size=args.font_size,
                custom_margin_x=args.margin_x,
                custom_margin_y=args.margin_y,
                render_format=args.render_format)

    # Load document
    filename = args.file
    logger.info("Loading document", filename=filename)
    
    if not Path(filename).exists():
        logger.error("File not found", filename=filename)
        print(f"Error: File '{filename}' not found")
        return 1

    load_start = time.time()
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    load_time = time.time() - load_start
    
    original_size = len(text)
    logger.info("Document loaded", 
                filename=filename,
                original_size_chars=original_size,
                load_time_ms=load_time * 1000)
    
    print(f"Loaded: {filename} ({original_size:,} characters)")

    # Pre-compression analysis
    if args.debug:
        logger.info("Performing pre-compression analysis")
        
        # Estimate original tokens with different methods
        original_tokens_gpt4 = TokenUtils.estimate_tokens(text, "gpt-4o")
        original_tokens_claude = TokenUtils.estimate_tokens(text, "claude-3-5-sonnet-20241022")
        
        logger.debug("Token estimation comparison",
                    original_tokens_gpt4=original_tokens_gpt4,
                    original_tokens_claude=original_tokens_claude,
                    text_length_chars=original_size,
                    chars_per_token_gpt4=original_size / original_tokens_gpt4 if original_tokens_gpt4 > 0 else 0,
                    chars_per_token_claude=original_size / original_tokens_claude if original_tokens_claude > 0 else 0)
        
        # Analyze compression potential
        optimizer = CompressionOptimizer()
        potential_analysis = optimizer.analyze_compression_potential(
            text_length=original_size,
            provider="lmstudio", 
            model="qwen/qwen3-next-80b"  # Use local LLM for analysis
        )
        
        logger.debug("Compression potential analysis",
                    **potential_analysis)
        
        print(f"Pre-analysis: {potential_analysis['compression_ratio']:.1f}x potential compression")
        
        # Show custom rendering configuration
        print(f"\n=== CUSTOM RENDERING CONFIG ===")
        print(f"Columns:     {args.columns}")
        print(f"DPI:         {args.dpi} (higher = better quality, larger files)")
        print(f"Font size:   {args.font_size}")
        print(f"Line height: {args.font_size + 1}")
        print(f"Margins:     {args.margin_x}×{args.margin_y} pixels (horizontal×vertical)")
        print(f"Cache dir:   {custom_glyph_config.cache_directory}")
        print(f"Provider opt: {custom_glyph_config.provider_optimization} (disabled to use custom DPI)")
        print(f"Profiles:    {len(custom_glyph_config.provider_profiles)} (cleared to prevent override)")
        print("=" * 32)
        
        # Show where temporary files might be created
        import tempfile
        import time as time_module
        temp_dir = tempfile.gettempdir()
        print(f"Temporary files location: {temp_dir}")
        
        # Check for any existing glyph output directories
        possible_dirs = [
            Path("glyph_output"),
            Path("temp_glyph"),
            Path("output"),
            Path.cwd() / "glyph_output",
            Path.home() / ".abstractcore" / "glyph_output",
            Path(custom_glyph_config.cache_directory)  # Add the actual cache directory
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                print(f"Found existing glyph directory: {dir_path}")
                if dir_path.is_dir():
                    files = list(dir_path.glob("*"))
                    print(f"  Contains {len(files)} files")
                    if files:
                        print(f"  Recent files: {[f.name for f in files[:3]]}")
        print()

    # Initialize analytics for tracking (needed for both modes)
    analytics = CompressionAnalytics() if args.debug else None
    
    # Compression or direct text processing
    if args.no_compression:
        logger.info("Skipping compression - using direct text mode")
        compression_start = time.time()
        
        # Create a mock result structure for consistency
        result = {
            'media': [],  # No media for direct text
            'total_compression_ratio': 1.0,  # No compression
            'method': 'direct_text',
            'original_tokens': TokenUtils.estimate_tokens(text, "gpt-4o"),
            'compressed_tokens': TokenUtils.estimate_tokens(text, "gpt-4o")
        }
        
        compression_time = time.time() - compression_start
        print(f"Direct text mode: No compression applied")
        
    else:
        # Normal compression path
        logger.info("Starting compression", target_ratio=args.target_ratio)
        compression_start = time.time()
        
        result = pipeline.compress(text, target_ratio=args.target_ratio)
        
        compression_time = time.time() - compression_start
    
    # Log detailed compression results
    compression_ratio = result.get('total_compression_ratio', 0)
    media_count = len(result.get('media', []))
    
    logger.info("Compression completed",
                compression_time_ms=compression_time * 1000,
                compression_ratio=compression_ratio,
                media_count=media_count,
                original_size_chars=original_size)
    
    if args.debug:
        # Log detailed compression metrics
        logger.debug("Detailed compression metrics",
                    result_keys=list(result.keys()),
                    media_types=[type(m).__name__ for m in result.get('media', [])],
                    result_structure=str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
        
        # Log detailed media information
        media_items = result.get('media', [])
        for i, media_item in enumerate(media_items):
            media_info = {
                "index": i,
                "type": type(media_item).__name__,
                "size_bytes": len(str(media_item)) if hasattr(media_item, '__str__') else 0
            }
            
            # Extract detailed information based on media type
            if hasattr(media_item, 'file_path'):
                media_info["file_path"] = str(media_item.file_path)
                
                # Get actual image dimensions and calculate real DPI
                try:
                    from PIL import Image
                    if Path(media_item.file_path).exists():
                        with Image.open(media_item.file_path) as img:
                            media_info["image_dimensions"] = f"{img.width}×{img.height}"
                            media_info["image_mode"] = img.mode
                            
                            # Get DPI from image metadata
                            if hasattr(img, 'info') and 'dpi' in img.info:
                                media_info["metadata_dpi"] = img.info['dpi']
                            
                            # Calculate actual DPI based on A4 dimensions
                            # A4 = 8.27" × 11.7" (210mm × 297mm)
                            a4_width_inches = 8.27
                            a4_height_inches = 11.7
                            
                            calculated_dpi_x = img.width / a4_width_inches
                            calculated_dpi_y = img.height / a4_height_inches
                            
                            media_info["calculated_dpi"] = f"{calculated_dpi_x:.0f}×{calculated_dpi_y:.0f}"
                            media_info["avg_calculated_dpi"] = f"{(calculated_dpi_x + calculated_dpi_y) / 2:.0f}"
                            
                except Exception as e:
                    media_info["image_error"] = str(e)
                    
            if hasattr(media_item, 'content_type'):
                media_info["content_type"] = media_item.content_type
            if hasattr(media_item, 'metadata'):
                media_info["metadata"] = media_item.metadata
                # Extract DPI from metadata if available
                if isinstance(media_item.metadata, dict) and 'dpi' in media_item.metadata:
                    media_info["config_dpi"] = media_item.metadata['dpi']
            if hasattr(media_item, 'data') and media_item.data:
                media_info["data_size"] = len(media_item.data)
                media_info["data_type"] = type(media_item.data).__name__
            if hasattr(media_item, 'base64_data') and media_item.base64_data:
                media_info["base64_size"] = len(media_item.base64_data)
            
            logger.debug(f"Generated media item {i}", **media_info)
            
            # Enhanced console output with dimensions and DPI
            file_path = media_info.get('file_path', 'no path')
            dimensions = media_info.get('image_dimensions', 'unknown size')
            config_dpi = media_info.get('config_dpi', 'unknown')
            metadata_dpi = media_info.get('metadata_dpi', 'unknown')
            calculated_dpi = media_info.get('avg_calculated_dpi', 'unknown')
            
            print(f"  Media {i}: {Path(file_path).name if file_path != 'no path' else 'no path'}")
            print(f"    📏 Dimensions: {dimensions}")
            print(f"    🎯 Config DPI: {config_dpi}")
            print(f"    📊 Metadata DPI: {metadata_dpi} (often wrong)")
            print(f"    🧮 Calculated DPI: {calculated_dpi} (from A4 dimensions)")
            print(f"    💾 Size: {media_info['size_bytes']} bytes")
        
        # Scan for newly created files after compression
        print(f"\n=== POST-COMPRESSION FILE SCAN ===")
        for dir_path in possible_dirs:
            if dir_path.exists() and dir_path.is_dir():
                files = list(dir_path.glob("*"))
                if files:
                    print(f"Directory: {dir_path}")
                    recent_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
                    for f in recent_files:
                        mtime = f.stat().st_mtime
                        if time_module.time() - mtime < 300:  # Files created in last 5 minutes
                            print(f"  📄 {f.name} ({f.stat().st_size} bytes, {time_module.ctime(mtime)})")
        
        # Also check current directory for any new image files
        current_dir = Path.cwd()
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
        for ext in image_extensions:
            recent_images = [f for f in current_dir.glob(f"*{ext}") 
                           if time_module.time() - f.stat().st_mtime < 300]
            if recent_images:
                print(f"Recent {ext} files in current directory:")
                for img in recent_images[:3]:
                    print(f"  📸 {img.name} ({img.stat().st_size} bytes)")
        print("=" * 35)
        
        # Calculate and log compression breakdown
        if 'glyph_compression_ratio' in result and 'vision_compression_ratio' in result:
            glyph_ratio = result['glyph_compression_ratio']
            vision_ratio = result['vision_compression_ratio']
            
            logger.debug("Compression stage breakdown",
                        glyph_stage_ratio=glyph_ratio,
                        vision_stage_ratio=vision_ratio,
                        combined_ratio=compression_ratio,
                        glyph_contribution_pct=(glyph_ratio / compression_ratio * 100) if compression_ratio > 0 else 0,
                        vision_contribution_pct=(vision_ratio / compression_ratio * 100) if compression_ratio > 0 else 0)
        
        # Record in analytics
        if analytics:
            original_tokens = TokenUtils.estimate_tokens(text, "gpt-4o")
            compressed_tokens = int(original_tokens / compression_ratio) if compression_ratio > 0 else original_tokens
            
            analytics.record_compression(
                provider="hybrid",
                model="glyph+vision",
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                quality_score=result.get('quality_score', 0.85),
                processing_time=compression_time,
                images_created=media_count,
                method="hybrid_pipeline",
                success=True,
                metadata={
                    "target_ratio": args.target_ratio,
                    "file": args.file,
                    "text_length": original_size
                }
            )
    
    print(f"Compression: {compression_ratio:.1f}x in {compression_time:.2f}s ({media_count} media items)")

    # LLM generation (compressed or direct text)
    logger.info("Starting LLM generation")
    generation_start = time.time()
    
    if args.no_compression:
        # Direct text mode - send raw text to LLM
        if args.debug:
            print(f"\n=== DIRECT TEXT MODE ===")
            print(f"Text length: {len(text):,} characters")
            print(f"Estimated tokens: {TokenUtils.estimate_tokens(text, 'gpt-4o'):,}")
            print(f"Sending raw text directly to LLM")
            print("=" * 30)
        
        response = llm.generate(
            f"Summarize this document:\n\n{text}",  # Include the full text in prompt
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=0.1
        )
        
    else:
        # Compressed mode - send images to LLM
        if args.debug:
            media_items = result.get('media', [])
            logger.debug("Generation input details",
                        media_count=len(media_items),
                        media_sizes=[len(str(m)) for m in media_items],
                        prompt="Summarize this document:",
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        frequency_penalty=args.frequency_penalty)
            
            # Log what media is being sent to LLM
            print(f"\n=== MEDIA BEING SENT TO LLM ===")
            for i, media_item in enumerate(media_items):
                item_info = f"Media {i}: {type(media_item).__name__}"
                if hasattr(media_item, 'file_path'):
                    item_info += f" from {media_item.file_path}"
                if hasattr(media_item, 'content_type'):
                    item_info += f" ({media_item.content_type})"
                print(f"  {item_info}")
                
                # Show first few characters of data if available
                if hasattr(media_item, 'data') and media_item.data:
                    data_preview = str(media_item.data)[:100] + "..." if len(str(media_item.data)) > 100 else str(media_item.data)
                    print(f"    Data preview: {data_preview}")
            print("=" * 30)
        
        response = llm.generate(
            "Summarize this document:",
            media=result['media'],  # Use the 'media' field from result
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,  # Reduce repetitions
            frequency_penalty=args.frequency_penalty,   # Additional repetition control
            presence_penalty=0.1     # Encourage topic diversity
        )
    
    generation_time = time.time() - generation_start
    
    response_length = len(response.content) if hasattr(response, 'content') else len(str(response))
    response_tokens = TokenUtils.estimate_tokens(response.content if hasattr(response, 'content') else str(response), "gpt-4o")
    
    logger.info("Generation completed",
                generation_time_ms=generation_time * 1000,
                response_length_chars=response_length,
                response_tokens=response_tokens)
    
    if args.debug:
        # Calculate generation efficiency metrics
        original_tokens = TokenUtils.estimate_tokens(text, "gpt-4o")
        compression_efficiency = compression_ratio / compression_time if compression_time > 0 else 0
        generation_efficiency = response_tokens / generation_time if generation_time > 0 else 0
        
        logger.debug("Performance efficiency metrics",
                    compression_efficiency_ratio_per_sec=compression_efficiency,
                    generation_efficiency_tokens_per_sec=generation_efficiency,
                    total_pipeline_efficiency=(compression_ratio * response_tokens) / (compression_time + generation_time) if (compression_time + generation_time) > 0 else 0,
                    input_output_ratio=response_tokens / original_tokens if original_tokens > 0 else 0)
    
    # Summary
    total_time = compression_time + generation_time
    
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Setup:       {setup_time:.2f}s")
    print(f"Loading:     {load_time:.2f}s")
    print(f"Compression: {compression_time:.2f}s")
    print(f"Generation:  {generation_time:.2f}s")
    print(f"Total:       {total_time:.2f}s")
    
    if args.no_compression:
        print(f"\n=== DIRECT TEXT SUMMARY ===")
        print(f"Mode:        Direct text (no compression)")
        print(f"Text size:   {original_size:,} characters")
        print(f"Tokens:      {TokenUtils.estimate_tokens(text, 'gpt-4o'):,} (estimated)")
        print(f"Media items: 0 (text only)")
    else:
        print(f"\n=== COMPRESSION SUMMARY ===")
        print(f"Mode:        Vision compression")
        print(f"Original:    {original_size:,} characters")
        print(f"Ratio:       {compression_ratio:.1f}x")
        print(f"Media items: {media_count}")
    
    if args.debug:
        # Enhanced debug summary
        original_tokens = TokenUtils.estimate_tokens(text, "gpt-4o")
        
        if args.no_compression:
            print(f"\n=== DEBUG METRICS (DIRECT TEXT) ===")
            print(f"Input tokens:        {original_tokens:,}")
            print(f"Output tokens:       {response_tokens:,}")
            print(f"Total tokens used:   {original_tokens + response_tokens:,}")
            print(f"Chars per token:     {original_size / original_tokens:.2f}")
            print(f"Generation speed:    {response_tokens / generation_time:.0f} tokens per second")
            print(f"Token efficiency:    {response_tokens / original_tokens:.3f} (output/input ratio)")
        else:
            compressed_tokens = int(original_tokens / compression_ratio) if compression_ratio > 0 else original_tokens
            
            print(f"\n=== DEBUG METRICS (COMPRESSED) ===")
            print(f"Original tokens:     {original_tokens:,}")
            print(f"Compressed tokens:   {compressed_tokens:,}")
            print(f"Token reduction:     {original_tokens - compressed_tokens:,} ({((original_tokens - compressed_tokens) / original_tokens * 100):.1f}%)")
            print(f"Chars per token:     {original_size / original_tokens:.2f}")
            print(f"Compression speed:   {compression_ratio / compression_time:.1f}x per second")
            print(f"Generation speed:    {response_tokens / generation_time:.0f} tokens per second")
            
            if 'glyph_compression_ratio' in result and 'vision_compression_ratio' in result:
                print(f"Glyph stage:         {result['glyph_compression_ratio']:.1f}x")
                print(f"Vision stage:        {result['vision_compression_ratio']:.1f}x")
        
        # Log file locations
        if Path("logs").exists():
            log_files = list(Path("logs").glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                print(f"Detailed logs:       {latest_log}")
        
        # Final comprehensive file summary
        print(f"\n=== ALL GENERATED FILES SUMMARY ===")
        all_found_files = []
        
        # Scan all possible locations
        scan_locations = [
            Path("glyph_output"),
            Path("temp_glyph"), 
            Path("output"),
            Path.cwd(),
            Path(tempfile.gettempdir()) / "abstractcore",
            Path(custom_glyph_config.cache_directory)  # Add the actual cache directory
        ]
        
        for location in scan_locations:
            if location.exists():
                # Look for recent files (last 10 minutes)
                cutoff_time = time_module.time() - 600
                if location.is_dir():
                    for pattern in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp", "*glyph*", "*compress*"]:
                        for f in location.glob(pattern):
                            if f.stat().st_mtime > cutoff_time:
                                all_found_files.append((f, location))
        
        if all_found_files:
            print("Recent files (last 10 minutes):")
            for file_path, parent_dir in sorted(all_found_files, key=lambda x: x[0].stat().st_mtime, reverse=True):
                rel_path = file_path.relative_to(parent_dir) if file_path.is_relative_to(parent_dir) else file_path
                print(f"  📁 {parent_dir.name}/{rel_path} ({file_path.stat().st_size} bytes)")
        else:
            print("No recent generated files found in common locations")
        print("=" * 40)
    
    print(f"\n=== RESPONSE ===")
    print(response.content if hasattr(response, 'content') else str(response))
    
    logger.info("Test completed successfully",
                total_time_ms=total_time * 1000,
                compression_efficiency=compression_ratio / compression_time if compression_time > 0 else 0,
                final_response_length=response_length,
                final_response_tokens=response_tokens)
    
    return 0


if __name__ == "__main__":
    exit(main())
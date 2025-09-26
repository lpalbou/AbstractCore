#!/usr/bin/env python3
"""
AbstractLLM Summarizer CLI Application

Usage:
    python -m abstractllm.apps.summarizer <file_path> [options]

Options:
    --style=<style>         Summary style (structured, narrative, objective, analytical, executive, conversational)
    --length=<length>       Summary length (brief, standard, detailed, comprehensive)
    --focus=<focus>         Specific focus area for summarization
    --output=<output>       Output file path (optional, prints to console if not provided)
    --chunk-size=<size>     Chunk size in characters (default: 8000, max: 32000)
    --provider=<provider>   LLM provider (requires --model)
    --model=<model>         LLM model (requires --provider)
    --verbose              Show detailed progress information
    --help                 Show this help message

Examples:
    python -m abstractllm.apps.summarizer document.pdf
    python -m abstractllm.apps.summarizer report.txt --style=executive --length=brief --verbose
    python -m abstractllm.apps.summarizer data.md --focus="technical details" --output=summary.txt
    python -m abstractllm.apps.summarizer large.txt --chunk-size=15000 --provider=openai --model=gpt-4o-mini
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from ..processing import BasicSummarizer, SummaryStyle, SummaryLength
from ..core.factory import create_llm


def read_file_content(file_path: str) -> str:
    """
    Read content from various file types

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be read
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path_obj.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Try to read as text file
    try:
        # Try UTF-8 first
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to other encodings
        for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path_obj, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all text encodings fail, try binary read and decode
        try:
            with open(file_path_obj, 'rb') as f:
                content = f.read()
                # Try to decode as text
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            raise Exception(f"Cannot read file {file_path}: {e}")


def parse_style(style_str: Optional[str]) -> SummaryStyle:
    """Parse style string to SummaryStyle enum"""
    if not style_str:
        return SummaryStyle.STRUCTURED

    style_map = {
        'structured': SummaryStyle.STRUCTURED,
        'narrative': SummaryStyle.NARRATIVE,
        'objective': SummaryStyle.OBJECTIVE,
        'analytical': SummaryStyle.ANALYTICAL,
        'executive': SummaryStyle.EXECUTIVE,
        'conversational': SummaryStyle.CONVERSATIONAL,
    }

    style_lower = style_str.lower()
    if style_lower not in style_map:
        available_styles = ', '.join(style_map.keys())
        raise ValueError(f"Invalid style '{style_str}'. Available styles: {available_styles}")

    return style_map[style_lower]


def parse_length(length_str: Optional[str]) -> SummaryLength:
    """Parse length string to SummaryLength enum"""
    if not length_str:
        return SummaryLength.STANDARD

    length_map = {
        'brief': SummaryLength.BRIEF,
        'standard': SummaryLength.STANDARD,
        'detailed': SummaryLength.DETAILED,
        'comprehensive': SummaryLength.COMPREHENSIVE,
    }

    length_lower = length_str.lower()
    if length_lower not in length_map:
        available_lengths = ', '.join(length_map.keys())
        raise ValueError(f"Invalid length '{length_str}'. Available lengths: {available_lengths}")

    return length_map[length_lower]


def format_summary_output(result) -> str:
    """Format summary result for display"""
    output_lines = []

    # Main summary
    output_lines.append("SUMMARY")
    output_lines.append("=" * 50)
    output_lines.append(result.summary)
    output_lines.append("")

    # Key points
    output_lines.append("KEY POINTS")
    output_lines.append("-" * 20)
    for i, point in enumerate(result.key_points, 1):
        output_lines.append(f"{i}. {point}")
    output_lines.append("")

    # Metadata
    output_lines.append("METADATA")
    output_lines.append("-" * 15)
    output_lines.append(f"Confidence Score: {result.confidence:.2f}")
    output_lines.append(f"Focus Alignment: {result.focus_alignment:.2f}")
    output_lines.append(f"Original Words: {result.word_count_original:,}")
    output_lines.append(f"Summary Words: {result.word_count_summary:,}")
    compression_ratio = (1 - result.word_count_summary / max(result.word_count_original, 1)) * 100
    output_lines.append(f"Compression: {compression_ratio:.1f}%")

    return "\n".join(output_lines)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="AbstractLLM Document Summarizer - Default: gemma3:1b-it-qat (requires Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m abstractllm.apps.summarizer document.pdf
  python -m abstractllm.apps.summarizer report.txt --style=executive --length=brief --verbose
  python -m abstractllm.apps.summarizer data.md --focus="technical details" --output=summary.txt
  python -m abstractllm.apps.summarizer large.txt --chunk-size=15000 --provider=openai --model=gpt-4o-mini

Supported file types: .txt, .md, .py, .js, .html, .json, .csv, and most text-based files

Default model setup:
  - Requires Ollama: https://ollama.com/
  - Download model: ollama pull gemma3:1b-it-qat
  - Or use --provider and --model for other providers
        """
    )

    parser.add_argument(
        'file_path',
        help='Path to the file to summarize'
    )

    parser.add_argument(
        '--style',
        choices=['structured', 'narrative', 'objective', 'analytical', 'executive', 'conversational'],
        default='structured',
        help='Summary style (default: structured)'
    )

    parser.add_argument(
        '--length',
        choices=['brief', 'standard', 'detailed', 'comprehensive'],
        default='standard',
        help='Summary length (default: standard)'
    )

    parser.add_argument(
        '--focus',
        help='Specific focus area for summarization'
    )

    parser.add_argument(
        '--output',
        help='Output file path (prints to console if not provided)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8000,
        help='Chunk size in characters (default: 8000, max: 32000)'
    )

    parser.add_argument(
        '--provider',
        help='LLM provider (requires --model)'
    )

    parser.add_argument(
        '--model',
        help='LLM model (requires --provider)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Validate chunk size
        if args.chunk_size < 1000:
            print("Error: Chunk size must be at least 1000 characters")
            sys.exit(1)

        if args.chunk_size > 32000:
            print("Error: Chunk size cannot exceed 32000 characters")
            sys.exit(1)

        # Validate provider/model pair
        if args.provider and not args.model:
            print("Error: --model is required when --provider is specified")
            sys.exit(1)

        if args.model and not args.provider:
            print("Error: --provider is required when --model is specified")
            sys.exit(1)

        # Read file content
        if args.verbose:
            print(f"Reading file: {args.file_path}")

        content = read_file_content(args.file_path)

        if not content.strip():
            print("Error: File is empty or contains no readable content")
            sys.exit(1)

        if args.verbose:
            print(f"File loaded ({len(content):,} characters)")

        # Parse options
        style = parse_style(args.style)
        length = parse_length(args.length)

        # Initialize LLM and summarizer
        if args.provider and args.model:
            # Custom provider/model with max_tokens adjusted for chunk size
            max_tokens = max(16000, args.chunk_size)  # Ensure max_tokens >= chunk_size
            if args.verbose:
                print(f"Initializing summarizer ({args.provider}, {args.model}, {max_tokens} token context)...")

            llm = create_llm(args.provider, model=args.model, max_tokens=max_tokens)
            summarizer = BasicSummarizer(llm, max_chunk_size=args.chunk_size)
        else:
            # Default configuration with chunk size override
            if args.chunk_size != 8000:
                # Custom chunk size, need to adjust max_tokens if necessary
                max_tokens = max(16000, args.chunk_size)
                if args.verbose:
                    print(f"Initializing summarizer (ollama, gemma3:1b-it-qat, {max_tokens} token context, {args.chunk_size} chunk size)...")

                try:
                    llm = create_llm("ollama", model="gemma3:1b-it-qat", max_tokens=max_tokens)
                    summarizer = BasicSummarizer(llm, max_chunk_size=args.chunk_size)
                except Exception as e:
                    # Handle default model not available
                    print(f"\n❌ Failed to initialize default Ollama model 'gemma3:1b-it-qat': {e}")
                    print("\n💡 To use the default model, please:")
                    print("   1. Install Ollama from: https://ollama.com/")
                    print("   2. Download the model: ollama pull gemma3:1b-it-qat")
                    print("   3. Start Ollama service")
                    print("\n🚀 Alternatively, specify a different provider:")
                    print("   - Example: summarizer document.txt --provider openai --model gpt-4o-mini")
                    sys.exit(1)
            else:
                # Default configuration
                if args.verbose:
                    print("Initializing summarizer (ollama, gemma3:1b-it-qat, 16k context, 8k chunks)...")
                try:
                    summarizer = BasicSummarizer()
                except RuntimeError as e:
                    # Handle default model not available
                    print(f"\n{e}")
                    print("\n🚀 Quick alternatives to get started:")
                    print("   - Use --provider and --model to specify an available provider")
                    print("   - Example: summarizer document.txt --provider openai --model gpt-4o-mini")
                    sys.exit(1)

        # Generate summary
        if args.verbose:
            print("Generating summary...")

        start_time = time.time()
        result = summarizer.summarize(
            text=content,
            focus=args.focus,
            style=style,
            length=length
        )
        end_time = time.time()

        if args.verbose:
            duration = end_time - start_time
            print(f"Summary generated in {duration:.2f} seconds")

        # Format output
        formatted_output = format_summary_output(result)

        # Output result
        if args.output:
            # Write to file
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            if args.verbose:
                print(f"Summary saved to: {output_path}")
        else:
            # Print to console
            print("\n" + formatted_output)


    except KeyboardInterrupt:
        print("\nSummarization cancelled by user")
        sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
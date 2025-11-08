#!/usr/bin/env python3
"""
Example: Programmatically list all supported file formats.

This script demonstrates how to query AbstractCore for supported file types
and extensions.
"""

from abstractcore.media.types import get_all_supported_extensions, get_supported_extensions_by_type, MediaType
from abstractcore.media.auto_handler import AutoMediaHandler


def example_get_all_formats():
    """Example: Get all supported formats organized by media type."""
    print("="*70)
    print("All Supported File Extensions (from FILE_TYPE_MAPPINGS)")
    print("="*70)

    formats = get_all_supported_extensions()

    for media_type, extensions in sorted(formats.items()):
        print(f"\n{media_type.upper()}: {len(extensions)} extensions")
        print(f"  Extensions: {', '.join(extensions[:10])}")
        if len(extensions) > 10:
            print(f"  ... and {len(extensions) - 10} more")


def example_get_text_formats():
    """Example: Get all text file extensions."""
    print("\n" + "="*70)
    print("Text File Extensions (70+ formats)")
    print("="*70)

    text_extensions = get_supported_extensions_by_type(MediaType.TEXT)

    print(f"\nTotal: {len(text_extensions)} text file extensions\n")

    categories = {
        "Programming Languages": ['py', 'js', 'java', 'c', 'cpp', 'go', 'rs', 'r', 'R'],
        "Notebooks": ['ipynb', 'rmd', 'Rmd', 'qmd'],
        "Configuration": ['yaml', 'yml', 'toml', 'ini', 'conf', 'env'],
        "Markup": ['md', 'markdown', 'rst', 'tex', 'html'],
        "Data": ['json', 'jsonl', 'csv', 'tsv', 'xml'],
    }

    for category, examples in categories.items():
        found = [ext for ext in examples if ext in text_extensions]
        print(f"{category:25} ({len(found)} found): {', '.join(found)}")

    # Show complete list
    print(f"\nComplete list of {len(text_extensions)} text extensions:")
    for i in range(0, len(text_extensions), 10):
        chunk = text_extensions[i:i+10]
        print(f"  {', '.join(chunk)}")


def example_handler_formats():
    """Example: Get formats from AutoMediaHandler."""
    print("\n" + "="*70)
    print("Formats Available Through AutoMediaHandler")
    print("="*70)

    handler = AutoMediaHandler()
    formats = handler.get_supported_formats()

    for media_type, extensions in sorted(formats.items()):
        print(f"\n{media_type.upper()}: {len(extensions)} extensions")
        print(f"  Sample: {', '.join(extensions[:15])}")
        if len(extensions) > 15:
            print(f"  ... and {len(extensions) - 15} more")


def example_check_specific_extension():
    """Example: Check if specific extensions are supported."""
    print("\n" + "="*70)
    print("Check Specific File Extensions")
    print("="*70)

    test_files = {
        "analysis.R": "R script",
        "notebook.ipynb": "Jupyter notebook",
        "query.sql": "SQL query",
        "config.yaml": "YAML config",
        "script.jl": "Julia script",
        "main.rs": "Rust source",
        "data.custom": "Unknown extension"
    }

    text_extensions = get_supported_extensions_by_type(MediaType.TEXT)

    print()
    for filename, description in test_files.items():
        ext = filename.split('.')[-1]
        is_supported = ext in text_extensions
        status = "✅ Supported" if is_supported else "❓ Unknown (will use content detection)"
        print(f"{filename:20} ({description:20}): {status}")


def example_complete_workflow():
    """Example: Complete workflow for checking file support."""
    print("\n" + "="*70)
    print("Complete Workflow: Check File Support")
    print("="*70)

    import tempfile
    from pathlib import Path
    from abstractcore.media.types import detect_media_type, is_text_file

    test_content = {
        "script.r": "# R script\nlibrary(tidyverse)\n",
        "notebook.ipynb": '{"cells":[],"nbformat":4}',
        "unknown.xyz": "Plain text content"
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        print()
        for filename, content in test_content.items():
            filepath = Path(tmpdir) / filename
            filepath.write_text(content)

            # Step 1: Check extension mapping
            ext = filename.split('.')[-1]
            text_extensions = get_supported_extensions_by_type(MediaType.TEXT)
            in_mapping = ext in text_extensions

            # Step 2: Detect media type (uses content detection for unknown)
            media_type = detect_media_type(filepath)

            # Step 3: Check if it's text
            is_text = is_text_file(filepath)

            print(f"File: {filename}")
            print(f"  Extension in mapping: {in_mapping}")
            print(f"  Detected type: {media_type.value}")
            print(f"  Is text file: {is_text}")
            print(f"  Status: {'✅ Will be processed as text' if media_type == MediaType.TEXT else '⚠️ May need special handling'}")
            print()


if __name__ == "__main__":
    example_get_all_formats()
    example_get_text_formats()
    example_handler_formats()
    example_check_specific_extension()
    example_complete_workflow()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
Programmatic Access to Supported Formats:

1. Get all formats:
   from abstractcore.media.types import get_all_supported_extensions
   formats = get_all_supported_extensions()

2. Get formats by type:
   from abstractcore.media.types import get_supported_extensions_by_type, MediaType
   text_exts = get_supported_extensions_by_type(MediaType.TEXT)

3. Get formats from handler:
   from abstractcore.media.auto_handler import AutoMediaHandler
   handler = AutoMediaHandler()
   formats = handler.get_supported_formats()

Note: TEXT type supports unknown extensions via content detection!
    """)
    print("="*70)

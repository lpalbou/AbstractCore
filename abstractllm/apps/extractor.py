#!/usr/bin/env python3
"""
AbstractLLM Entity Extractor CLI Application

Usage:
    python -m abstractllm.apps.extractor <file_path> [options]

Options:
    --focus=<focus>             Specific focus area for extraction (e.g., "technology", "business", "medical")
    --style=<style>             Extraction style (structured, focused, minimal, comprehensive, default: structured)
    --length=<length>           Extraction depth (brief, standard, detailed, comprehensive, default: standard)
    --entity-types=<types>      Comma-separated entity types to focus on (person,organization,location,etc.)
    --similarity-threshold=<t>  Similarity threshold for entity deduplication (0.0-1.0, default: 0.85)
    --format=<format>          Output format (json-ld, json, yaml, default: json-ld)
    --output=<output>          Output file path (optional, prints to console if not provided)
    --chunk-size=<size>        Chunk size in characters (default: 6000, max: 32000)
    --provider=<provider>      LLM provider (requires --model)
    --model=<model>            LLM model (requires --provider)
    --no-embeddings           Disable semantic entity deduplication
    --fast                    Use fast extraction (skip verification, larger chunks, no embeddings)
    --iterate=<number>        Number of refinement iterations (default: 1, finds missing entities and verifies relationships)
    --minified                Output minified JSON-LD (compact, no indentation)
    --verbose                 Show detailed progress information
    --help                    Show this help message

Examples:
    python -m abstractllm.apps.extractor document.pdf
    python -m abstractllm.apps.extractor report.txt --focus=technology --style=structured --verbose
    python -m abstractllm.apps.extractor data.md --entity-types=person,organization --output=kg.jsonld
    python -m abstractllm.apps.extractor large.txt --fast --minified --verbose  # Fast, compact output
    python -m abstractllm.apps.extractor report.txt --length=detailed --provider=openai --model=gpt-4o-mini
    python -m abstractllm.apps.extractor doc.txt --iterate=3 --verbose  # 3 refinement passes for higher quality
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from ..processing import BasicExtractor
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


def parse_entity_types(types_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated entity types string (informational only in new extractor)"""
    if not types_str:
        return None

    # Just return as list of strings - new extractor doesn't use this
    return [t.strip() for t in types_str.lower().split(',')]


def parse_extraction_style(style_str: Optional[str]) -> str:
    """Parse extraction style string"""
    if not style_str:
        return 'structured'

    valid_styles = ['structured', 'focused', 'minimal', 'comprehensive']
    style_lower = style_str.lower()

    if style_lower not in valid_styles:
        available_styles = ', '.join(valid_styles)
        raise ValueError(f"Invalid extraction style '{style_str}'. Available styles: {available_styles}")

    return style_lower


def parse_extraction_length(length_str: Optional[str]) -> str:
    """Parse extraction length string"""
    if not length_str:
        return 'brief'

    valid_lengths = ['brief', 'standard', 'detailed', 'comprehensive']
    length_lower = length_str.lower()

    if length_lower not in valid_lengths:
        available_lengths = ', '.join(valid_lengths)
        raise ValueError(f"Invalid extraction length '{length_str}'. Available lengths: {available_lengths}")

    return length_lower



# Legacy helper functions removed - new extractor returns JSON-LD directly

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="AbstractLLM Entity & Relationship Extractor - Default: gemma3:1b-it-qat (requires Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m abstractllm.apps.extractor document.pdf
  python -m abstractllm.apps.extractor report.txt --focus=technology --style=structured --verbose
  python -m abstractllm.apps.extractor data.md --entity-types=person,organization --output=kg.jsonld
  python -m abstractllm.apps.extractor large.txt --length=detailed --fast --minified --verbose
  python -m abstractllm.apps.extractor doc.txt --iterate=3 --verbose  # Iterative refinement for quality
  python -m abstractllm.apps.extractor doc.txt --format=triples --verbose  # RDF triples output
  python -m abstractllm.apps.extractor doc.txt --format=triples --output=triples.txt  # Simple triples

Supported file types: .txt, .md, .py, .js, .html, .json, .csv, and most text-based files

Output formats:
  - json-ld: Knowledge Graph format using schema.org vocabulary (default)
  - triples: RDF-style SUBJECT PREDICATE OBJECT format for semantic web
  - json: Simple JSON format
  - yaml: YAML format

Output options:
  - Default: Pretty-printed JSON with indentation
  - --minified: Compact JSON without indentation (smaller file size)

Performance options:
  - Default: High accuracy with Chain of Verification (slower, 2x LLM calls per chunk)
  - --fast: Optimized speed (skip verification, larger chunks, no embeddings)
  - For large files: Use --fast flag for significant speedup (2-4x faster)

Quality enhancement:
  - --iterate=N: Perform N refinement passes to find missing entities/relationships
  - Each iteration reviews the extraction to find gaps and verify relationship directionality
  - Recommended: 2-3 iterations for critical extractions, 1 (default) for speed

Default model setup:
  - Requires Ollama: https://ollama.com/
  - Download model: ollama pull gemma3:1b-it-qat
  - Or use --provider and --model for other providers
        """
    )

    parser.add_argument(
        'file_path',
        help='Path to the file to extract entities and relationships from'
    )

    parser.add_argument(
        '--focus',
        help='Specific focus area for extraction (e.g., "technology", "business", "medical")'
    )

    parser.add_argument(
        '--style',
        choices=['structured', 'focused', 'minimal', 'comprehensive'],
        default='structured',
        help='Extraction style (default: structured)'
    )

    parser.add_argument(
        '--length',
        choices=['brief', 'standard', 'detailed', 'comprehensive'],
        default='brief',
        help='Extraction depth (default: brief)'
    )

    parser.add_argument(
        '--entity-types',
        help='Comma-separated entity types to focus on (person,organization,location,concept,event,technology,product,date,other)'
    )

    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for entity deduplication (0.0-1.0, default: 0.85)'
    )

    # Build format choices based on available dependencies
    format_choices = ['json-ld', 'triples', 'json']
    if YAML_AVAILABLE:
        format_choices.append('yaml')

    parser.add_argument(
        '--format',
        choices=format_choices,
        default='json-ld',
        help='Output format: json-ld (JSON-LD graph), triples (RDF SUBJECT PREDICATE OBJECT), json (simple JSON)' + (', yaml' if YAML_AVAILABLE else ' - install PyYAML for YAML support')
    )

    parser.add_argument(
        '--output',
        help='Output file path (prints to console if not provided)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=6000,
        help='Chunk size in characters (default: 6000, max: 32000)'
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
        '--no-embeddings',
        action='store_true',
        help='Disable semantic entity deduplication'
    )

    parser.add_argument(
        '--mode',
        choices=['fast', 'balanced', 'thorough'],
        default='balanced',
        help='Extraction mode: fast (2-3x faster), balanced (default), thorough (highest quality)'
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='Legacy flag: equivalent to --mode=fast (deprecated, use --mode instead)'
    )

    parser.add_argument(
        '--iterate',
        type=int,
        default=1,
        help='Number of refinement iterations to find missing entities and verify relationships (default: 1)'
    )

    parser.add_argument(
        '--minified',
        action='store_true',
        help='Output minified JSON-LD (compact, no indentation)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )

    # Parse arguments
    args = parser.parse_args()

    try:
        # Validate similarity threshold
        if not 0.0 <= args.similarity_threshold <= 1.0:
            print("Error: Similarity threshold must be between 0.0 and 1.0")
            sys.exit(1)

        # Validate chunk size
        if args.chunk_size < 1000:
            print("Error: Chunk size must be at least 1000 characters")
            sys.exit(1)

        if args.chunk_size > 32000:
            print("Error: Chunk size cannot exceed 32000 characters")
            sys.exit(1)

        # Validate iterate parameter
        if args.iterate < 1:
            print("Error: Iterate must be at least 1")
            sys.exit(1)

        if args.iterate > 5:
            print("Error: Iterate cannot exceed 5 (diminishing returns)")
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
        entity_types = parse_entity_types(args.entity_types)
        extraction_style = parse_extraction_style(args.style)
        extraction_length = parse_extraction_length(args.length)

        # Determine extraction mode (handle legacy --fast flag)
        extraction_mode = args.mode
        if args.fast:
            extraction_mode = "fast"

        # Initialize LLM and extractor
        use_embeddings = not args.no_embeddings

        if args.provider and args.model:
            # Custom provider/model with max_tokens adjusted for chunk size
            max_tokens = max(16000, args.chunk_size)
            if args.verbose:
                print(f"Initializing BasicExtractor (mode: {extraction_mode}, {args.provider}, {args.model}, {max_tokens} token context)...")

            llm = create_llm(args.provider, model=args.model, max_tokens=max_tokens)

            extractor = BasicExtractor(
                llm=llm,
                max_chunk_size=args.chunk_size
            )
        else:
            # Default configuration
            if args.verbose:
                print(f"Initializing BasicExtractor (mode: {extraction_mode}, ollama, gemma3:1b-it-qat, threshold: {args.similarity_threshold})...")

            try:
                extractor = BasicExtractor(
                    max_chunk_size=args.chunk_size
                )
            except RuntimeError as e:
                # Handle default model not available
                print(f"\n{e}")
                print("\n🚀 Quick alternatives to get started:")
                print("   - Use --provider and --model to specify an available provider")
                print("   - Example: extractor document.txt --provider openai --model gpt-4o-mini")
                print("   - For speed: extractor document.txt --mode=fast")
                sys.exit(1)

        # Extract entities and relationships
        if args.verbose:
            print("Extracting entities and relationships...")

        start_time = time.time()

        # Always extract in JSON-LD format first (for refinement compatibility)
        result = extractor.extract(
            text=content,
            domain_focus=args.focus,
            entity_types=entity_types,
            style=extraction_style,
            length=extraction_length,
            output_format="jsonld"
        )

        # Perform iterative refinement if requested
        if args.iterate > 1:
            if args.verbose:
                print(f"\nStarting {args.iterate - 1} refinement iteration(s)...")

            for iteration in range(2, args.iterate + 1):
                if args.verbose:
                    entities = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('e:')]
                    relationships = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('r:')]
                    print(f"\n📝 Iteration {iteration}/{args.iterate}: Reviewing extraction ({len(entities)} entities, {len(relationships)} relationships)...")

                prev_count = len(result.get('@graph', []))
                result = extractor.refine_extraction(
                    text=content,
                    previous_extraction=result,
                    domain_focus=args.focus
                )
                new_count = len(result.get('@graph', []))

                if args.verbose and new_count > prev_count:
                    print(f"   ✓ Added {new_count - prev_count} new items")
                elif args.verbose:
                    print(f"   ✓ No changes needed")

        end_time = time.time()

        if args.verbose:
            duration = end_time - start_time
            print(f"\nExtraction completed in {duration:.2f} seconds")
            # Count entities and relationships from @graph
            entities = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('e:')]
            relationships = [item for item in result.get('@graph', []) if item.get('@id', '').startswith('r:')]
            print(f"Final result: {len(entities)} entities and {len(relationships)} relationships")

        # Apply final output format conversion
        if args.format == 'json-ld' and args.minified:
            # Convert to minified JSON-LD
            result = extractor._format_output(result, "jsonld_minified")
        elif args.format == 'triples':
            # Convert to triples format
            result = extractor._format_output(result, "triples")
        # else: keep as jsonld for json, yaml, and non-minified json-ld

        # Format output
        # Determine JSON indentation (minified or pretty-printed)
        json_indent = None if args.minified else 2

        if args.format == 'json-ld' and args.minified:
            # Minified JSON-LD format - result is already formatted by extractor
            if result.get('format') == 'jsonld_minified':
                formatted_output = result['data']  # Pre-minified string
            else:
                # Fallback minification
                formatted_output = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        elif args.format == 'triples':
            # Triples format - output the simple triples plus detailed info
            if result.get('format') == 'triples':
                if args.verbose:
                    # Verbose: show detailed triples with metadata
                    output_data = {
                        "format": "triples",
                        "simple_triples": result.get('simple_triples', []),
                        "detailed_triples": result.get('triples', []),
                        "entities": result.get('entities', {}),
                        "statistics": result.get('statistics', {})
                    }
                    formatted_output = json.dumps(output_data, indent=json_indent, ensure_ascii=False)
                else:
                    # Non-verbose: just show simple triples
                    simple_triples = result.get('simple_triples', [])
                    formatted_output = '\n'.join(simple_triples)
            else:
                formatted_output = json.dumps(result, indent=json_indent, ensure_ascii=False)
        elif args.format == 'json-ld':
            # Standard JSON-LD format
            formatted_output = json.dumps(
                result,
                indent=json_indent,
                ensure_ascii=False,
                separators=(',', ':') if args.minified else None
            )
        elif args.format == 'json':
            # For simple JSON, just output the result as-is
            formatted_output = json.dumps(
                result,
                indent=json_indent,
                ensure_ascii=False,
                separators=(',', ':') if args.minified else None
            )
        else:  # yaml
            if not YAML_AVAILABLE:
                print("Error: PyYAML is required for YAML output format. Install with: pip install PyYAML")
                sys.exit(1)
            formatted_output = yaml.dump(result, default_flow_style=False, indent=2, sort_keys=False)

        # Output result
        if args.output:
            # Write to file
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            if args.verbose:
                print(f"Knowledge graph saved to: {output_path}")
        else:
            # Print to console
            print(formatted_output)


    except KeyboardInterrupt:
        print("\nExtraction cancelled by user")
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
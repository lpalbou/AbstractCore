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
    --minified                Output minified JSON-LD (compact, no indentation)
    --verbose                 Show detailed progress information
    --help                    Show this help message

Examples:
    python -m abstractllm.apps.extractor document.pdf
    python -m abstractllm.apps.extractor report.txt --focus=technology --style=structured --verbose
    python -m abstractllm.apps.extractor data.md --entity-types=person,organization --output=kg.jsonld
    python -m abstractllm.apps.extractor large.txt --fast --minified --verbose  # Fast, compact output
    python -m abstractllm.apps.extractor report.txt --length=detailed --provider=openai --model=gpt-4o-mini
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

from ..processing import BasicExtractor, FastExtractor, EntityType, RelationType
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


def parse_entity_types(types_str: Optional[str]) -> Optional[List[EntityType]]:
    """Parse comma-separated entity types string to EntityType list"""
    if not types_str:
        return None

    type_map = {
        'person': EntityType.PERSON,
        'organization': EntityType.ORGANIZATION,
        'location': EntityType.LOCATION,
        'concept': EntityType.CONCEPT,
        'event': EntityType.EVENT,
        'technology': EntityType.TECHNOLOGY,
        'product': EntityType.PRODUCT,
        'date': EntityType.DATE,
        'other': EntityType.OTHER,
    }

    types_list = []
    for type_str in types_str.lower().split(','):
        type_str = type_str.strip()
        if type_str not in type_map:
            available_types = ', '.join(type_map.keys())
            raise ValueError(f"Invalid entity type '{type_str}'. Available types: {available_types}")
        types_list.append(type_map[type_str])

    return types_list


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
        return 'standard'

    valid_lengths = ['brief', 'standard', 'detailed', 'comprehensive']
    length_lower = length_str.lower()

    if length_lower not in valid_lengths:
        available_lengths = ', '.join(valid_lengths)
        raise ValueError(f"Invalid extraction length '{length_str}'. Available lengths: {available_lengths}")

    return length_lower


def entity_type_to_schema_type(entity_type: EntityType) -> str:
    """Map EntityType to schema.org type"""
    schema_map = {
        EntityType.PERSON: "schema:Person",
        EntityType.ORGANIZATION: "schema:Organization",
        EntityType.LOCATION: "schema:Place",
        EntityType.CONCEPT: "schema:Thing",
        EntityType.EVENT: "schema:Event",
        EntityType.TECHNOLOGY: "schema:SoftwareApplication",
        EntityType.PRODUCT: "schema:Product",
        EntityType.DATE: "schema:Date",
        EntityType.OTHER: "schema:Thing"
    }
    return schema_map.get(entity_type, "schema:Thing")


def relation_type_to_schema_property(relation_type: RelationType) -> str:
    """Map RelationType to schema.org property"""
    property_map = {
        RelationType.WORKS_FOR: "schema:worksFor",
        RelationType.LOCATED_IN: "schema:location",
        RelationType.CREATED_BY: "schema:creator",
        RelationType.RELATED_TO: "schema:relatedTo",
        RelationType.CAUSES: "schema:result",
        RelationType.USES: "schema:instrument",
        RelationType.PARTICIPATES_IN: "schema:participant",
        RelationType.OCCURRED_ON: "schema:startDate",
        RelationType.SIMILAR_TO: "schema:sameAs",
        RelationType.OTHER: "schema:relatedTo"
    }
    return property_map.get(relation_type, "schema:relatedTo")


def format_jsonld_output(result, source_file: str) -> Dict[str, Any]:
    """Format extraction result as JSON-LD for Knowledge Graph compatibility"""

    # JSON-LD context
    context = {
        "@vocab": "https://schema.org/",
        "schema": "https://schema.org/",
        "kg": "https://abstractllm.com/kg/",
        "confidence": "kg:confidence",
        "aliases": "kg:aliases",
        "extractionMetadata": "kg:extractionMetadata"
    }

    # Convert entities to JSON-LD format
    entities_jsonld = []
    entity_id_map = {}  # canonical_id -> @id

    for canonical_id, entity in result.entities.items():
        # Create unique IRI for entity
        entity_iri = f"kg:entity/{canonical_id}"
        entity_id_map[canonical_id] = entity_iri

        entity_jsonld = {
            "@id": entity_iri,
            "@type": entity_type_to_schema_type(entity.type),
            "name": entity.name,
            "confidence": entity.confidence
        }

        # Add aliases if present
        if entity.aliases:
            entity_jsonld["aliases"] = entity.aliases

        # Add context as description
        if entity.context:
            entity_jsonld["description"] = entity.context

        entities_jsonld.append(entity_jsonld)

    # Convert relationships to JSON-LD format
    relationships_jsonld = []
    for i, rel in enumerate(result.relationships):
        # Find entity IRIs
        source_iri = None
        target_iri = None

        for canonical_id, entity in result.entities.items():
            if entity.name == rel.source:
                source_iri = entity_id_map[canonical_id]
            if entity.name == rel.target:
                target_iri = entity_id_map[canonical_id]

        if source_iri and target_iri:
            relationship_jsonld = {
                "@id": f"kg:relation/{i}",
                "@type": "kg:Relationship",
                "subject": {"@id": source_iri},
                "predicate": relation_type_to_schema_property(rel.relation),
                "object": {"@id": target_iri},
                "confidence": rel.confidence,
                "context": rel.context
            }
            relationships_jsonld.append(relationship_jsonld)

    # Create complete JSON-LD document
    jsonld_doc = {
        "@context": context,
        "@type": "kg:KnowledgeGraph",
        "@id": f"kg:extraction/{int(time.time())}",
        "source": source_file,
        "extractionDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extractionMetadata": {
            "verificationConfidence": result.verification_confidence,
            "deduplicationSummary": result.deduplication_summary,
            "entitiesCount": len(result.entities),
            "relationshipsCount": len(result.relationships)
        },
        "entities": entities_jsonld,
        "relationships": relationships_jsonld
    }

    return jsonld_doc


def format_json_output(result, source_file: str) -> Dict[str, Any]:
    """Format extraction result as plain JSON"""

    # Convert entities to simple dict format
    entities_dict = {}
    for canonical_id, entity in result.entities.items():
        entities_dict[canonical_id] = {
            "name": entity.name,
            "type": entity.type.value,
            "aliases": entity.aliases,
            "context": entity.context,
            "confidence": entity.confidence
        }

    # Convert relationships to simple list format
    relationships_list = []
    for rel in result.relationships:
        relationships_list.append({
            "source": rel.source,
            "target": rel.target,
            "relation": rel.relation.value,
            "context": rel.context,
            "confidence": rel.confidence
        })

    return {
        "source": source_file,
        "extractionDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verificationConfidence": result.verification_confidence,
        "deduplicationSummary": result.deduplication_summary,
        "entities": entities_dict,
        "relationships": relationships_list
    }


def format_yaml_output(result, source_file: str) -> str:
    """Format extraction result as YAML"""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required for YAML output format. Install with: pip install PyYAML")

    json_output = format_json_output(result, source_file)
    return yaml.dump(json_output, default_flow_style=False, indent=2, sort_keys=False)


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

Supported file types: .txt, .md, .py, .js, .html, .json, .csv, and most text-based files

Output formats:
  - json-ld: Knowledge Graph format using schema.org vocabulary (default)
  - json: Simple JSON format
  - yaml: YAML format

Output options:
  - Default: Pretty-printed JSON with indentation
  - --minified: Compact JSON without indentation (smaller file size)

Performance options:
  - Default: High accuracy with Chain of Verification (slower, 2x LLM calls per chunk)
  - --fast: Optimized speed (skip verification, larger chunks, no embeddings)
  - For large files: Use --fast flag for significant speedup (2-4x faster)

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
        default='standard',
        help='Extraction depth (default: standard)'
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
    format_choices = ['json-ld', 'json']
    if YAML_AVAILABLE:
        format_choices.append('yaml')

    parser.add_argument(
        '--format',
        choices=format_choices,
        default='json-ld',
        help='Output format (default: json-ld)' + ('' if YAML_AVAILABLE else ' - install PyYAML for YAML support')
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
        '--fast',
        action='store_true',
        help='Use fast extraction (skip verification, larger chunks, no embeddings)'
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

        # Initialize LLM and extractor
        use_embeddings = not args.no_embeddings

        # Override settings for fast mode
        if args.fast:
            use_embeddings = False  # Disable embeddings for speed
            if args.chunk_size == 6000:  # If using default chunk size
                args.chunk_size = 15000  # Use larger chunks for speed

        if args.provider and args.model:
            # Custom provider/model with max_tokens adjusted for chunk size
            max_tokens = max(16000, args.chunk_size)
            if args.verbose:
                extractor_type = "FastExtractor" if args.fast else "BasicExtractor"
                print(f"Initializing {extractor_type} ({args.provider}, {args.model}, {max_tokens} token context)...")

            llm = create_llm(args.provider, model=args.model, max_tokens=max_tokens)

            if args.fast:
                extractor = FastExtractor(
                    llm=llm,
                    use_embeddings=use_embeddings,
                    similarity_threshold=args.similarity_threshold,
                    max_chunk_size=args.chunk_size,
                    use_verification=False  # Skip verification for speed
                )
            else:
                extractor = BasicExtractor(
                    llm=llm,
                    use_embeddings=use_embeddings,
                    similarity_threshold=args.similarity_threshold,
                    max_chunk_size=args.chunk_size
                )
        else:
            # Default configuration
            if args.verbose:
                extractor_type = "FastExtractor" if args.fast else "BasicExtractor"
                embeddings_status = "enabled" if use_embeddings else "disabled"
                verification_status = "disabled" if args.fast else "enabled"
                print(f"Initializing {extractor_type} (ollama, gemma3:1b-it-qat, embeddings: {embeddings_status}, verification: {verification_status}, threshold: {args.similarity_threshold})...")

            try:
                if args.fast:
                    extractor = FastExtractor(
                        use_embeddings=use_embeddings,
                        similarity_threshold=args.similarity_threshold,
                        max_chunk_size=args.chunk_size,
                        use_verification=False  # Skip verification for speed
                    )
                else:
                    extractor = BasicExtractor(
                        use_embeddings=use_embeddings,
                        similarity_threshold=args.similarity_threshold,
                        max_chunk_size=args.chunk_size
                    )
            except RuntimeError as e:
                # Handle default model not available
                print(f"\n{e}")
                print("\n🚀 Quick alternatives to get started:")
                print("   - Use --provider and --model to specify an available provider")
                print("   - Example: extractor document.txt --provider openai --model gpt-4o-mini")
                print("   - For speed: extractor document.txt --fast")
                sys.exit(1)

        # Extract entities and relationships
        if args.verbose:
            print("Extracting entities and relationships...")

        start_time = time.time()
        result = extractor.extract(
            text=content,
            domain_focus=args.focus,
            entity_types=entity_types,
            style=extraction_style,
            length=extraction_length
        )
        end_time = time.time()

        if args.verbose:
            duration = end_time - start_time
            print(f"Extraction completed in {duration:.2f} seconds")
            print(f"Found {len(result.entities)} entities and {len(result.relationships)} relationships")
            if result.deduplication_summary.get('merged', 0) > 0:
                print(f"Merged {result.deduplication_summary['merged']} duplicate entities")

        # Format output
        # Determine JSON indentation (minified or pretty-printed)
        json_indent = None if args.minified else 2

        if args.format == 'json-ld':
            formatted_output = json.dumps(
                format_jsonld_output(result, args.file_path),
                indent=json_indent,
                ensure_ascii=False,
                separators=(',', ':') if args.minified else None
            )
        elif args.format == 'json':
            formatted_output = json.dumps(
                format_json_output(result, args.file_path),
                indent=json_indent,
                ensure_ascii=False,
                separators=(',', ':') if args.minified else None
            )
        else:  # yaml
            formatted_output = format_yaml_output(result, args.file_path)

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
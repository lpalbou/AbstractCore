#!/usr/bin/env python3
"""
AbstractCore Deep Search V2 CLI Application

A simplified, transparent deep research system with iterative learning.

Usage:
    python -m abstractcore.apps.deepsearchv2 "<research_query>" [options]

Options:
    --max-iterations <number>   Maximum research iterations (default: 3)
    --max-sources <number>      Maximum number of sources to gather (default: 10)
    --min-confidence <0.0-1.0>  Minimum confidence to stop early (default: 0.7)
    --output <output>           Output file path (optional, prints to console if not provided)
    --format <format>           Output format (json, markdown, plain, default: plain)
    --provider <provider>       LLM provider (requires --model)
    --model <model>             LLM model (requires --provider)
    --verbose                   Show detailed progress information (default: True)
    --quiet                     Disable verbose logging
    --timeout <seconds>         HTTP timeout for LLM providers (default: 300)
    --max-tokens <tokens>       Maximum total tokens for LLM context (default: 32000)
    --max-output-tokens <tokens> Maximum tokens for LLM output generation (default: 8000)
    --temperature <0.0-1.0>     LLM temperature (default: 0.1 for consistency)
    --help                      Show this help message

Examples:
    # Basic research with default settings
    python -m abstractcore.apps.deepsearchv2 "What are the latest developments in quantum computing?"

    # Quick research (2 iterations, 5 sources)
    python -m abstractcore.apps.deepsearchv2 "What is CRISPR?" --max-iterations 2 --max-sources 5

    # Deep research with custom LLM
    python -m abstractcore.apps.deepsearchv2 "AI impact on healthcare" --provider openai --model gpt-4o-mini --max-iterations 5

    # Save report to file
    python -m abstractcore.apps.deepsearchv2 "sustainable energy 2025" --output report.md --format markdown

    # Quiet mode (minimal output)
    python -m abstractcore.apps.deepsearchv2 "blockchain trends" --quiet --output report.json

Note: V2 is simpler and more transparent than V1, with:
- Sequential iterative learning (not parallel)
- Quality-focused source selection
- Verbose step-by-step logging
- Intent analysis and gap detection
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

from ..processing.basic_deepsearchv2 import BasicDeepSearchV2
from ..core.factory import create_llm


def get_app_defaults(app_name: str) -> tuple[str, str]:
    """Get default provider and model for an app."""
    try:
        from ..config import get_config_manager
        config_manager = get_config_manager()
        return config_manager.get_app_default(app_name)
    except Exception:
        # Fallback to hardcoded defaults if config unavailable
        hardcoded_defaults = {
            'deepsearchv2': ('ollama', 'qwen3:4b-instruct-2507-q4_K_M'),
            'deepsearch': ('ollama', 'qwen3:4b-instruct-2507-q4_K_M'),
        }
        return hardcoded_defaults.get(app_name, ('ollama', 'qwen3:4b-instruct-2507-q4_K_M'))


def timeout_type(value):
    """Parse timeout value - accepts None, 'none', or float"""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid timeout value: {value}. Use 'none' for unlimited or a number in seconds.")


def format_report_as_plain(report) -> str:
    """Format report as plain text"""

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("DEEP RESEARCH V2 REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Query
    lines.append(f"📋 RESEARCH QUERY")
    lines.append(f"{report.query}")
    lines.append("")

    # Executive Summary
    lines.append(f"📝 EXECUTIVE SUMMARY")
    lines.append(f"{report.executive_summary}")
    lines.append("")

    # Key Findings
    lines.append(f"🔑 KEY FINDINGS ({len(report.key_findings)})")
    for idx, finding in enumerate(report.key_findings, 1):
        lines.append(f"{idx}. {finding}")
    lines.append("")

    # Detailed Analysis
    lines.append(f"📖 DETAILED ANALYSIS")
    lines.append(f"{report.detailed_analysis}")
    lines.append("")

    # Sources
    lines.append(f"📚 SOURCES ({len(report.sources)})")
    for idx, source in enumerate(report.sources, 1):
        lines.append(f"{idx}. {source['title']}")
        lines.append(f"   URL: {source['url']}")
        if 'quality_score' in source:
            lines.append(f"   Quality: {source['quality_score']:.1f}/100")
        if 'iteration' in source:
            lines.append(f"   Found in: Iteration {source['iteration']}")
    lines.append("")

    # Metadata
    lines.append(f"📈 METADATA")
    lines.append(f"  ├─ Confidence Level: {report.confidence_level:.2f}")
    lines.append(f"  ├─ Total Sources: {report.metadata.get('total_sources', 0)}")
    if 'avg_quality_score' in report.metadata:
        lines.append(f"  ├─ Avg Quality Score: {report.metadata['avg_quality_score']:.1f}/100")
    lines.append(f"  ├─ Research Time: {report.metadata.get('elapsed_time', 0):.1f}s")
    if 'intent_type' in report.metadata:
        lines.append(f"  └─ Intent Type: {report.metadata['intent_type']}")
    lines.append("")

    # Knowledge Gaps
    if report.knowledge_gaps:
        lines.append(f"⚠️  KNOWLEDGE GAPS")
        for gap in report.knowledge_gaps:
            lines.append(f"  • {gap}")
        lines.append("")

    # Research Path
    if report.research_path:
        lines.append(f"🛤️  RESEARCH PATH ({len(report.research_path)} steps)")
        for step in report.research_path:
            lines.append(f"  • {step}")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def format_report_as_markdown(report) -> str:
    """Convert report to markdown format"""

    md_lines = []

    # Title
    md_lines.append(f"# Deep Research V2 Report")
    md_lines.append("")
    md_lines.append(f"**Query:** {report.query}")
    md_lines.append("")

    # Executive Summary
    md_lines.append("## Executive Summary")
    md_lines.append("")
    md_lines.append(report.executive_summary)
    md_lines.append("")

    # Key Findings
    md_lines.append("## Key Findings")
    md_lines.append("")
    for idx, finding in enumerate(report.key_findings, 1):
        md_lines.append(f"{idx}. {finding}")
    md_lines.append("")

    # Detailed Analysis
    md_lines.append("## Detailed Analysis")
    md_lines.append("")
    md_lines.append(report.detailed_analysis)
    md_lines.append("")

    # Sources
    md_lines.append("## Sources")
    md_lines.append("")
    for idx, source in enumerate(report.sources, 1):
        md_lines.append(f"{idx}. **{source['title']}**")
        md_lines.append(f"   - URL: {source['url']}")
        if 'quality_score' in source:
            md_lines.append(f"   - Quality: {source['quality_score']:.1f}/100")
        if 'iteration' in source:
            md_lines.append(f"   - Found in: Iteration {source['iteration']}")
        md_lines.append("")

    # Metadata
    md_lines.append("## Metadata")
    md_lines.append("")
    md_lines.append(f"- **Confidence Level:** {report.confidence_level:.2f}")
    md_lines.append(f"- **Total Sources:** {report.metadata.get('total_sources', 0)}")
    if 'avg_quality_score' in report.metadata:
        md_lines.append(f"- **Avg Quality Score:** {report.metadata['avg_quality_score']:.1f}/100")
    md_lines.append(f"- **Research Time:** {report.metadata.get('elapsed_time', 0):.1f}s")
    if 'intent_type' in report.metadata:
        md_lines.append(f"- **Intent Type:** {report.metadata['intent_type']}")
    md_lines.append("")

    # Knowledge Gaps
    if report.knowledge_gaps:
        md_lines.append("## Knowledge Gaps")
        md_lines.append("")
        for gap in report.knowledge_gaps:
            md_lines.append(f"- {gap}")
        md_lines.append("")

    # Research Path
    if report.research_path:
        md_lines.append("## Research Path")
        md_lines.append("")
        for step in report.research_path:
            md_lines.append(f"- {step}")
        md_lines.append("")

    return "\n".join(md_lines)


def save_report(report, output_path: str, format_type: str) -> None:
    """
    Save research report to file

    Args:
        report: ResearchReport object
        output_path: Path to save the report
        format_type: Output format type
    """
    output_file = Path(output_path)

    # Create directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format_type == 'json':
            # Convert to JSON
            if hasattr(report, 'model_dump'):
                report_data = report.model_dump()
            elif hasattr(report, 'dict'):
                report_data = report.dict()
            else:
                report_data = report

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

        elif format_type == 'markdown':
            markdown_content = format_report_as_markdown(report)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

        else:  # plain
            plain_content = format_report_as_plain(report)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(plain_content)

        print(f"\n✅ Report saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Failed to save report: {e}")
        sys.exit(1)


def main():
    """Main CLI function"""

    parser = argparse.ArgumentParser(
        description="AbstractCore Deep Search V2 - Simple, Clean, and Effective Deep Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the latest developments in quantum computing?"
  %(prog)s "What is CRISPR?" --max-iterations 2 --max-sources 5
  %(prog)s "AI impact on healthcare" --provider openai --model gpt-4o-mini
  %(prog)s "sustainable energy 2025" --output report.md --format markdown
        """
    )

    # Required argument
    parser.add_argument(
        'query',
        help='Research query or question'
    )

    # Research configuration
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum research iterations (default: 3)'
    )

    parser.add_argument(
        '--max-sources',
        type=int,
        default=10,
        help='Maximum number of sources to gather (default: 10)'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.7,
        help='Minimum confidence to stop early, 0.0-1.0 (default: 0.7)'
    )

    # Output options
    parser.add_argument(
        '--format',
        choices=['json', 'markdown', 'plain'],
        default='plain',
        help='Output format (default: plain)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (optional, prints to console if not provided)'
    )

    # LLM configuration
    parser.add_argument(
        '--provider',
        type=str,
        help='LLM provider (requires --model)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='LLM model (requires --provider)'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=32000,
        help='Maximum total tokens for LLM context (default: 32000)'
    )

    parser.add_argument(
        '--max-output-tokens',
        type=int,
        default=8000,
        help='Maximum tokens for LLM output generation (default: 8000)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='LLM temperature (default: 0.1 for consistency)'
    )

    parser.add_argument(
        '--timeout',
        type=timeout_type,
        default=300,
        help='HTTP timeout for LLM providers in seconds (default: 300)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Show detailed progress information (default: True)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable verbose logging'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.min_confidence < 0.0 or args.min_confidence > 1.0:
        print("❌ Error: --min-confidence must be between 0.0 and 1.0")
        sys.exit(1)

    if (args.provider and not args.model) or (args.model and not args.provider):
        print("❌ Error: Both --provider and --model must be specified together")
        sys.exit(1)

    # Determine verbosity
    verbose = args.verbose and not args.quiet

    try:
        # Get LLM configuration
        if args.provider and args.model:
            provider = args.provider
            model = args.model
        else:
            # Use app defaults
            provider, model = get_app_defaults('deepsearchv2')

        if verbose:
            print("=" * 80)
            print("DEEP RESEARCH V2")
            print("=" * 80)
            print(f"\n🤖 Using LLM: {provider}/{model}")
            print(f"📋 Query: {args.query}")
            print(f"🔄 Max Iterations: {args.max_iterations}")
            print(f"📊 Max Sources: {args.max_sources}")
            print(f"🎯 Min Confidence: {args.min_confidence}")
            print("")

        # Create LLM instance
        llm = create_llm(
            provider=provider,
            model=model,
            max_tokens=args.max_tokens,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout=args.timeout
        )

        # Create searcher
        searcher = BasicDeepSearchV2(
            llm=llm,
            verbose=verbose
        )

        # Perform research
        start_time = time.time()

        report = searcher.research(
            query=args.query,
            max_iterations=args.max_iterations,
            max_sources=args.max_sources,
            min_confidence=args.min_confidence
        )

        elapsed_time = time.time() - start_time

        # Save or print report
        if args.output:
            save_report(report, args.output, args.format)
        else:
            # Print to console
            if args.format == 'json':
                if hasattr(report, 'model_dump'):
                    print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
                elif hasattr(report, 'dict'):
                    print(json.dumps(report.dict(), indent=2, ensure_ascii=False))
                else:
                    print(json.dumps(report, indent=2, ensure_ascii=False))
            elif args.format == 'markdown':
                print(format_report_as_markdown(report))
            else:  # plain
                print("\n" + format_report_as_plain(report))

        if verbose and not args.output:
            print(f"\n✅ Research completed in {elapsed_time:.1f}s")

    except KeyboardInterrupt:
        print("\n\n❌ Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

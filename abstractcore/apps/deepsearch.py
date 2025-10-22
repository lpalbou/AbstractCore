#!/usr/bin/env python3
"""
AbstractCore Deep Search CLI Application

Usage:
    python -m abstractcore.apps.deepsearch "<research_query>" [options]

Options:
    --focus <areas>             Comma-separated focus areas (e.g., "technology,business,impact")
    --depth <depth>             Research depth (brief, standard, comprehensive, default: standard)
    --max-sources <number>      Maximum number of sources to gather (default: 15)
    --format <format>           Output format (structured, narrative, executive, default: structured)
    --output <output>           Output file path (optional, prints to console if not provided)
    --provider <provider>       LLM provider (requires --model)
    --model <model>             LLM model (requires --provider)
    --no-verification          Skip fact-checking and verification stage
    --parallel-searches <num>   Maximum parallel web searches (default: 5)
    --verbose                   Show detailed progress information
    --timeout <seconds>         HTTP timeout for LLM providers (default: 300)
    --max-tokens <tokens>       Maximum total tokens for LLM context (default: 32000)
    --max-output-tokens <tokens> Maximum tokens for LLM output generation (default: 8000)
    --help                      Show this help message

Examples:
    python -m abstractcore.apps.deepsearch "What are the latest developments in quantum computing?"
    python -m abstractcore.apps.deepsearch "AI impact on healthcare" --focus "diagnosis,treatment,ethics" --depth comprehensive
    python -m abstractcore.apps.deepsearch "sustainable energy 2025" --format executive --output report.json
    python -m abstractcore.apps.deepsearch "blockchain technology trends" --max-sources 25 --verbose
    python -m abstractcore.apps.deepsearch "climate change solutions" --provider openai --model gpt-4o-mini --depth comprehensive
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..processing import BasicDeepSearch
from ..core.factory import create_llm


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


def save_report(report, output_path: str, format_type: str) -> None:
    """
    Save research report to file
    
    Args:
        report: ResearchReport object or dictionary
        output_path: Path to save the report
        format_type: Output format type
    """
    output_file = Path(output_path)
    
    # Create directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if hasattr(report, 'model_dump'):
            # Pydantic model
            report_data = report.model_dump()
        elif hasattr(report, 'dict'):
            # Pydantic model (older versions)
            report_data = report.dict()
        else:
            # Dictionary
            report_data = report
        
        # Determine file format based on extension
        if output_path.lower().endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        elif output_path.lower().endswith('.md'):
            # Convert to markdown format
            markdown_content = format_report_as_markdown(report_data)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        else:
            # Default to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Report saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
        sys.exit(1)


def format_report_as_markdown(report_data: Dict[str, Any]) -> str:
    """Convert report data to markdown format"""
    
    md_lines = []
    
    # Title
    md_lines.append(f"# {report_data.get('title', 'Research Report')}")
    md_lines.append("")
    
    # Executive Summary
    if report_data.get('executive_summary'):
        md_lines.append("## Executive Summary")
        md_lines.append("")
        md_lines.append(report_data['executive_summary'])
        md_lines.append("")
    
    # Key Findings
    if report_data.get('key_findings'):
        md_lines.append("## Key Findings")
        md_lines.append("")
        for i, finding in enumerate(report_data['key_findings'], 1):
            md_lines.append(f"{i}. {finding}")
        md_lines.append("")
    
    # Detailed Analysis
    if report_data.get('detailed_analysis'):
        md_lines.append("## Detailed Analysis")
        md_lines.append("")
        md_lines.append(report_data['detailed_analysis'])
        md_lines.append("")
    
    # Conclusions
    if report_data.get('conclusions'):
        md_lines.append("## Conclusions")
        md_lines.append("")
        md_lines.append(report_data['conclusions'])
        md_lines.append("")
    
    # Sources
    if report_data.get('sources'):
        md_lines.append("## Sources")
        md_lines.append("")
        for i, source in enumerate(report_data['sources'], 1):
            title = source.get('title', 'Untitled')
            url = source.get('url', '')
            md_lines.append(f"{i}. [{title}]({url})")
        md_lines.append("")
    
    # Methodology
    if report_data.get('methodology'):
        md_lines.append("## Methodology")
        md_lines.append("")
        md_lines.append(report_data['methodology'])
        md_lines.append("")
    
    # Limitations
    if report_data.get('limitations'):
        md_lines.append("## Limitations")
        md_lines.append("")
        md_lines.append(report_data['limitations'])
        md_lines.append("")
    
    return "\n".join(md_lines)


def print_report(report, format_type: str) -> None:
    """
    Print research report to console
    
    Args:
        report: ResearchReport object or dictionary
        format_type: Output format type
    """
    try:
        if hasattr(report, 'model_dump'):
            # Pydantic model
            report_data = report.model_dump()
        elif hasattr(report, 'dict'):
            # Pydantic model (older versions)
            report_data = report.dict()
        else:
            # Dictionary
            report_data = report
        
        print("\n" + "="*80)
        print(f"🔍 DEEP SEARCH REPORT")
        print("="*80)
        
        # Title
        print(f"\n📋 {report_data.get('title', 'Research Report')}")
        print("-" * 60)
        
        # Executive Summary
        if report_data.get('executive_summary'):
            print(f"\n📊 EXECUTIVE SUMMARY")
            print(f"{report_data['executive_summary']}")
        
        # Key Findings
        if report_data.get('key_findings'):
            print(f"\n🎯 KEY FINDINGS")
            for i, finding in enumerate(report_data['key_findings'], 1):
                print(f"{i}. {finding}")
        
        # Detailed Analysis (truncated for console)
        if report_data.get('detailed_analysis'):
            print(f"\n📝 DETAILED ANALYSIS")
            analysis = report_data['detailed_analysis']
            if len(analysis) > 1000:
                print(f"{analysis[:1000]}...")
                print(f"\n[Analysis truncated - use --output to save full report]")
            else:
                print(analysis)
        
        # Conclusions
        if report_data.get('conclusions'):
            print(f"\n💡 CONCLUSIONS")
            print(report_data['conclusions'])
        
        # Sources
        if report_data.get('sources'):
            print(f"\n📚 SOURCES ({len(report_data['sources'])} total)")
            for i, source in enumerate(report_data['sources'][:10], 1):  # Show first 10
                title = source.get('title', 'Untitled')
                url = source.get('url', '')
                print(f"{i}. {title}")
                print(f"   🔗 {url}")
            
            if len(report_data['sources']) > 10:
                print(f"   ... and {len(report_data['sources']) - 10} more sources")
        
        # Methodology and Limitations
        if report_data.get('methodology') or report_data.get('limitations'):
            print(f"\n📋 METHODOLOGY & LIMITATIONS")
            if report_data.get('methodology'):
                print(f"Methodology: {report_data['methodology']}")
            if report_data.get('limitations'):
                print(f"Limitations: {report_data['limitations']}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"❌ Error displaying report: {e}")
        # Fallback to JSON output
        try:
            if hasattr(report, 'model_dump'):
                report_data = report.model_dump()
            elif hasattr(report, 'dict'):
                report_data = report.dict()
            else:
                report_data = report
            print(json.dumps(report_data, indent=2, ensure_ascii=False))
        except:
            print(f"Report object: {report}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="AbstractCore Deep Search - Autonomous research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "What are the latest developments in quantum computing?"
  %(prog)s "AI impact on healthcare" --focus "diagnosis,treatment,ethics" --depth comprehensive
  %(prog)s "sustainable energy 2025" --format executive --output report.json
  %(prog)s "blockchain technology trends" --max-sources 25 --verbose
        """
    )
    
    # Required argument
    parser.add_argument(
        'query',
        help='Research query or question to investigate'
    )
    
    # Research configuration
    parser.add_argument(
        '--focus',
        type=str,
        help='Comma-separated focus areas (e.g., "technology,business,impact")'
    )
    
    parser.add_argument(
        '--depth',
        choices=['brief', 'standard', 'comprehensive'],
        default='standard',
        help='Research depth (default: standard)'
    )
    
    parser.add_argument(
        '--max-sources',
        type=int,
        default=15,
        help='Maximum number of sources to gather (default: 15)'
    )
    
    parser.add_argument(
        '--format',
        choices=['structured', 'narrative', 'executive'],
        default='structured',
        help='Output format (default: structured)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (supports .json and .md formats)'
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
        '--timeout',
        type=timeout_type,
        default=300,
        help='HTTP timeout for LLM providers in seconds (default: 300, "none" for unlimited)'
    )
    
    # Research options
    parser.add_argument(
        '--no-verification',
        action='store_true',
        help='Skip fact-checking and verification stage'
    )
    
    parser.add_argument(
        '--parallel-searches',
        type=int,
        default=5,
        help='Maximum parallel web searches (default: 5)'
    )
    
    # Utility options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.provider and not args.model) or (args.model and not args.provider):
        print("❌ Error: Both --provider and --model must be specified together")
        sys.exit(1)
    
    if args.max_sources < 1 or args.max_sources > 100:
        print("❌ Error: --max-sources must be between 1 and 100")
        sys.exit(1)
    
    if args.parallel_searches < 1 or args.parallel_searches > 20:
        print("❌ Error: --parallel-searches must be between 1 and 20")
        sys.exit(1)
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize LLM
        if args.provider and args.model:
            print(f"🤖 Initializing {args.provider} with model {args.model}...")
            llm = create_llm(
                args.provider,
                model=args.model,
                max_tokens=args.max_tokens,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout
            )
        else:
            print("🤖 Initializing default LLM (Ollama)...")
            llm = None  # Will use default in BasicDeepSearch
        
        # Initialize Deep Search
        searcher = BasicDeepSearch(
            llm=llm,
            max_tokens=args.max_tokens,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
            max_parallel_searches=args.parallel_searches
        )
        
        # Parse focus areas
        focus_areas = None
        if args.focus:
            focus_areas = [area.strip() for area in args.focus.split(',')]
            print(f"🎯 Focus areas: {', '.join(focus_areas)}")
        
        # Display research configuration
        print(f"🔍 Research Query: {args.query}")
        print(f"📊 Depth: {args.depth}")
        print(f"📚 Max Sources: {args.max_sources}")
        print(f"📝 Format: {args.format}")
        print(f"✅ Verification: {'Disabled' if args.no_verification else 'Enabled'}")
        print(f"⚡ Parallel Searches: {args.parallel_searches}")
        
        # Start research
        start_time = time.time()
        print(f"\n🚀 Starting deep search research...")
        
        report = searcher.research(
            query=args.query,
            focus_areas=focus_areas,
            max_sources=args.max_sources,
            search_depth=args.depth,
            include_verification=not args.no_verification,
            output_format=args.format
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✨ Research completed in {elapsed_time:.1f} seconds")
        
        # Output results
        if args.output:
            save_report(report, args.output, args.format)
        else:
            print_report(report, args.format)
        
        # Summary statistics
        if hasattr(report, 'sources'):
            source_count = len(report.sources) if report.sources else 0
        elif isinstance(report, dict) and 'sources' in report:
            source_count = len(report['sources']) if report['sources'] else 0
        else:
            source_count = 0
        
        print(f"\n📊 Research Summary:")
        print(f"   • Sources analyzed: {source_count}")
        print(f"   • Research depth: {args.depth}")
        print(f"   • Time taken: {elapsed_time:.1f} seconds")
        print(f"   • Format: {args.format}")
        
        if args.output:
            print(f"   • Saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deep search failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

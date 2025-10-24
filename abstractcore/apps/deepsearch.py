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
        elif output_path.lower().endswith('.html'):
            # Convert to HTML format
            html_content = format_report_as_html(report_data)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
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
            relevance = source.get('relevance', 0)
            md_lines.append(f"{i}. [{title}]({url}) (Relevance: {relevance:.2f})")
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


def format_report_as_html(report_data: Dict[str, Any]) -> str:
    """Convert report data to HTML format"""
    
    html_parts = []
    
    # HTML header
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Search Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .section { margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }
        .finding { margin-bottom: 15px; padding: 10px; background: white; border-left: 4px solid #3498db; }
        .source { margin-bottom: 10px; padding: 10px; background: white; border-radius: 4px; }
        .source a { color: #3498db; text-decoration: none; }
        .source a:hover { text-decoration: underline; }
        .relevance { color: #7f8c8d; font-size: 0.9em; }
        .metadata { color: #7f8c8d; font-size: 0.9em; margin-top: 20px; }
    </style>
</head>
<body>""")
    
    # Title
    html_parts.append(f"<h1>{report_data.get('title', 'Research Report')}</h1>")
    
    # Executive Summary
    if report_data.get('executive_summary'):
        html_parts.append('<div class="section">')
        html_parts.append('<h2>📊 Executive Summary</h2>')
        html_parts.append(f"<p>{report_data['executive_summary']}</p>")
        html_parts.append('</div>')
    
    # Key Findings
    if report_data.get('key_findings'):
        html_parts.append('<div class="section">')
        html_parts.append('<h2>🎯 Key Findings</h2>')
        for i, finding in enumerate(report_data['key_findings'], 1):
            html_parts.append(f'<div class="finding">{i}. {finding}</div>')
        html_parts.append('</div>')
    
    # Detailed Analysis
    if report_data.get('detailed_analysis'):
        html_parts.append('<div class="section">')
        html_parts.append('<h2>📝 Detailed Analysis</h2>')
        # Convert newlines to paragraphs
        analysis = report_data['detailed_analysis'].replace('\n\n', '</p><p>').replace('\n', '<br>')
        html_parts.append(f"<p>{analysis}</p>")
        html_parts.append('</div>')
    
    # Conclusions
    if report_data.get('conclusions'):
        html_parts.append('<div class="section">')
        html_parts.append('<h2>💡 Conclusions</h2>')
        conclusions = report_data['conclusions'].replace('\n\n', '</p><p>').replace('\n', '<br>')
        html_parts.append(f"<p>{conclusions}</p>")
        html_parts.append('</div>')
    
    # Sources
    if report_data.get('sources'):
        html_parts.append('<div class="section">')
        html_parts.append(f'<h2>📚 Sources ({len(report_data["sources"])} total)</h2>')
        for i, source in enumerate(report_data['sources'], 1):
            title = source.get('title', 'Untitled')
            url = source.get('url', '')
            relevance = source.get('relevance', 0)
            html_parts.append(f'''<div class="source">
                {i}. <a href="{url}" target="_blank">{title}</a>
                <div class="relevance">Relevance: {relevance:.2f}</div>
            </div>''')
        html_parts.append('</div>')
    
    # Methodology and Limitations
    if report_data.get('methodology') or report_data.get('limitations'):
        html_parts.append('<div class="section">')
        html_parts.append('<h2>📋 Methodology & Limitations</h2>')
        if report_data.get('methodology'):
            html_parts.append(f"<p><strong>Methodology:</strong> {report_data['methodology']}</p>")
        if report_data.get('limitations'):
            html_parts.append(f"<p><strong>Limitations:</strong> {report_data['limitations']}</p>")
        html_parts.append('</div>')
    
    # Footer
    html_parts.append('<div class="metadata">')
    html_parts.append('<p>Generated by AbstractCore Deep Search</p>')
    html_parts.append('</div>')
    html_parts.append('</body></html>')
    
    return '\n'.join(html_parts)


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
        
        # Detailed Analysis (show full content)
        if report_data.get('detailed_analysis'):
            print(f"\n📝 DETAILED ANALYSIS")
            print(report_data['detailed_analysis'])
        
        # Conclusions
        if report_data.get('conclusions'):
            print(f"\n💡 CONCLUSIONS")
            print(report_data['conclusions'])
        
        # Sources
        if report_data.get('sources'):
            print(f"\n📚 SOURCES ({len(report_data['sources'])} total)")
            for i, source in enumerate(report_data['sources'], 1):  # Show ALL sources
                title = source.get('title', 'Untitled')
                url = source.get('url', '')
                print(f"{i}. {title}")
                print(f"   🔗 {url}")
        
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
        help='Output file path (supports .json, .md, .html, .txt formats)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['text', 'json', 'markdown', 'html'],
        default='text',
        help='Console output format (default: text)'
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
    
    parser.add_argument(
        '--full-text',
        action='store_true',
        help='Extract full text content from web pages (slower but more comprehensive)'
    )
    
    parser.add_argument(
        '--reflexive',
        action='store_true',
        help='Enable reflexive mode - analyzes limitations and performs targeted refinement searches'
    )
    
    parser.add_argument(
        '--max-reflexive-iterations',
        type=int,
        default=2,
        help='Maximum number of reflexive refinement cycles (default: 2)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='LLM temperature for consistency (default: 0.1, range: 0.0-1.0)'
    )
    
    # Utility options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress information'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show comprehensive debug information: all queries, URLs, relevance assessments, and processing decisions'
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
            llm = create_llm(
                args.provider,
                model=args.model,
                max_tokens=args.max_tokens,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout
            )
        else:
            llm = None  # Will use default in BasicDeepSearch
        
        # Initialize Deep Search
        searcher = BasicDeepSearch(
            llm=llm,
            max_tokens=args.max_tokens,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
            max_parallel_searches=args.parallel_searches,
            full_text_extraction=args.full_text,
            reflexive_mode=args.reflexive,
            max_reflexive_iterations=args.max_reflexive_iterations,
            temperature=args.temperature,
            debug_mode=args.debug
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
        print(f"📄 Text Extraction: {'Full Text' if args.full_text else 'Preview (1000 chars)'}")
        print(f"🔄 Reflexive Mode: {'Enabled' if args.reflexive else 'Disabled'}")
        if args.reflexive:
            print(f"🔁 Max Reflexive Iterations: {args.max_reflexive_iterations}")
        
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
        
        # Console output based on format
        if args.output_format == 'json':
            # JSON output to console
            if hasattr(report, 'model_dump'):
                report_data = report.model_dump()
            elif hasattr(report, 'dict'):
                report_data = report.dict()
            else:
                report_data = report
            print(json.dumps(report_data, indent=2, ensure_ascii=False))
        elif args.output_format == 'markdown':
            # Markdown output to console
            if hasattr(report, 'model_dump'):
                report_data = report.model_dump()
            elif hasattr(report, 'dict'):
                report_data = report.dict()
            else:
                report_data = report
            print(format_report_as_markdown(report_data))
        elif args.output_format == 'html':
            # HTML output to console
            if hasattr(report, 'model_dump'):
                report_data = report.model_dump()
            elif hasattr(report, 'dict'):
                report_data = report.dict()
            else:
                report_data = report
            print(format_report_as_html(report_data))
        else:
            # Default text output
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

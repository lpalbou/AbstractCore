#!/usr/bin/env python3
"""
Minimal test script for fetch_url tool from abstractcore.tools.common_tools

Usage:
    python test-fetch.py --url https://example.com
    python test-fetch.py --url https://api.github.com/repos/python/cpython --no-links
    python test-fetch.py --url https://httpbin.org/html --timeout 60
"""

from abstractcore.tools.common_tools import fetch_url
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Test the fetch_url tool from abstractcore.tools.common_tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url https://httpbin.org/html
  %(prog)s --url https://api.github.com/repos/python/cpython
  %(prog)s --url https://example.com --no-links --preview
        """
    )
    
    parser.add_argument(
        "--url",
        required=True,
        help="URL to fetch (required)"
    )
    
    parser.add_argument(
        "--extract-links",
        dest="extract_links",
        action="store_true",
        default=True,
        help="Extract links from HTML (default: True)"
    )
    
    parser.add_argument(
        "--no-links",
        dest="extract_links",
        action="store_false",
        help="Don't extract links from HTML"
    )
    
    parser.add_argument(
        "--preview",
        dest="include_full_content",
        action="store_false",
        default=True,
        help="Show preview only (truncate content)"
    )

    parser.add_argument(
        "--no-md",
        dest="convert_html_to_markdown",
        action="store_false",
        default=True,
        help="Disable HTML→Markdown conversion (default: enabled)"
    )

    parser.add_argument(
        "--keep-links",
        dest="keep_links",
        action="store_true",
        default=False,
        help="Preserve links when converting HTML→Markdown (default: False)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Timeout in seconds (default: 45)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Testing fetch_url with: {args.url}")
    print("=" * 80)
    print()
    
    # Call the fetch_url tool
    result = fetch_url(
        url=args.url,
        extract_links=args.extract_links,
        include_full_content=args.include_full_content,
        timeout=args.timeout,
        convert_html_to_markdown=args.convert_html_to_markdown,
        keep_links=args.keep_links,
    )
    
    # Display the EXACT rendered output that would be sent to an LLM
    print("=" * 80)
    print("RENDERED OUTPUT (what LLM receives):")
    print("=" * 80)
    print()
    print(result.get("rendered", "No rendered output available"))
    print()
    print("=" * 80)
    print("END OF RENDERED OUTPUT")
    print("=" * 80)
    
    # Show summary
    print("\nSummary:")
    print("-" * 80)
    print(f"Success: {result.get('success')}")
    print(f"Status Code: {result.get('status_code')}")
    print(f"Content Type: {result.get('content_type')}")
    print(f"Size: {result.get('size_bytes', 0):,} bytes")
    print(f"Final URL: {result.get('final_url', 'N/A')}")
    
    if result.get('final_url') != args.url:
        print(f"\n🔄 Redirect detected:")
        print(f"   Original: {args.url}")
        print(f"   Final:    {result.get('final_url')}")
    
    if not result.get('success'):
        print(f"\n❌ Error: {result.get('error')}")
    
    # Uncomment to see full JSON structure (including raw_text and normalized_text)
    # print("\nFull JSON Result:")
    # print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

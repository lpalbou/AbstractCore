#!/usr/bin/env python3
"""
Minimal test script for web_search tool from abstractcore.tools.common_tools

Usage:
    python test-search.py --query "python best practices"
    python test-search.py --query "AI developments 2025" --num-results 5
    python test-search.py --query "python tutorials" --time-range w
"""

from abstractcore.tools.common_tools import web_search
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Test the web_search tool from abstractcore.tools.common_tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --query "python best practices 2025"
  %(prog)s --query "AI developments" --num-results 5
  %(prog)s --query "python tutorials" --time-range w
  %(prog)s --query "secure coding" --region us-en --safe-search strict

Time Range Options:
  h or 24h  - Past 24 hours
  d         - Past day
  w or 7d   - Past week
  m or 30d  - Past month
  y or 1y   - Past year
  (none)    - All time (default)

Region Options:
  wt-wt  - Worldwide (default)
  us-en  - United States
  uk-en  - United Kingdom
  fr-fr  - France
  de-de  - Germany
        """
    )
    
    parser.add_argument(
        "--query",
        required=True,
        help="Search query (required)"
    )
    
    parser.add_argument(
        "--num-results",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    
    parser.add_argument(
        "--safe-search",
        choices=["strict", "moderate", "off"],
        default="moderate",
        help="Content filtering level (default: moderate)"
    )
    
    parser.add_argument(
        "--region",
        default="wt-wt",
        help="Regional results preference (default: wt-wt for worldwide)"
    )
    
    parser.add_argument(
        "--time-range",
        choices=["h", "24h", "d", "w", "7d", "m", "30d", "y", "1y"],
        help="Time range filter for results (e.g., 'w' for past week)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Searching for: {args.query}")
    print(f"Settings: {args.num_results} results, {args.safe_search} safe search, region: {args.region}")
    if args.time_range:
        print(f"Time range: {args.time_range}")
    print("=" * 80)
    print()
    
    # Call the web_search tool
    result_json = web_search(
        query=args.query,
        num_results=args.num_results,
        safe_search=args.safe_search,
        region=args.region,
        time_range=args.time_range,
    )
    
    # Parse the JSON result
    try:
        result = json.loads(result_json)
        
        # Display results in a readable format
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            if "hint" in result:
                print(f"💡 Hint: {result['hint']}")
        else:
            print(f"🔍 Search Engine: {result.get('engine', 'unknown')}")
            print(f"📊 Backend: {result.get('params', {}).get('backend', 'unknown')}")
            print()
            
            results = result.get('results', [])
            if results:
                print(f"Found {len(results)} results:")
                print("-" * 80)
                
                for item in results:
                    rank = item.get('rank', '?')
                    title = item.get('title', 'No title')
                    url = item.get('url', 'No URL')
                    snippet = item.get('snippet', 'No description')
                    
                    print(f"\n[{rank}] {title}")
                    print(f"    🔗 {url}")
                    if snippet:
                        # Wrap long snippets
                        print(f"    📝 {snippet}")
                
                print()
                print("=" * 80)
            else:
                print("No results found.")
        
        # Optionally show full JSON
        print("\n" + "-" * 80)
        print("Full JSON Response:")
        print("-" * 80)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON result: {e}")
        print("\nRaw result:")
        print(result_json)


if __name__ == "__main__":
    main()

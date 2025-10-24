#!/usr/bin/env python3
"""
AbstractCore Apps - Command-line interface launcher

Usage:
    python -m abstractcore.apps <app_name> [options]

Available apps:
    summarizer    - Document summarization tool
    extractor     - Entity and relationship extraction tool
    judge         - Text evaluation and scoring tool
    intent        - Intent analysis and motivation identification tool

Examples:
    python -m abstractcore.apps summarizer document.txt
    python -m abstractcore.apps extractor report.txt --format json-ld
    python -m abstractcore.apps judge essay.txt --criteria clarity,accuracy
    python -m abstractcore.apps intent "I need help with this problem" --depth comprehensive
    python -m abstractcore.apps <app> --help
"""

import sys
from pathlib import Path

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    app_name = sys.argv[1]

    if app_name == "summarizer":
        # Remove the app name from sys.argv and run summarizer
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .summarizer import main as summarizer_main
        summarizer_main()
    elif app_name == "extractor":
        # Remove the app name from sys.argv and run extractor
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .extractor import main as extractor_main
        extractor_main()
    elif app_name == "judge":
        # Remove the app name from sys.argv and run judge
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .judge import main as judge_main
        judge_main()
    elif app_name == "intent":
        # Remove the app name from sys.argv and run intent analyzer
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .intent import main as intent_main
        intent_main()
    else:
        print(f"Unknown app: {app_name}")
        print("\nAvailable apps: summarizer, extractor, judge, intent")
        sys.exit(1)

if __name__ == "__main__":
    main()
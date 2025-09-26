#!/usr/bin/env python3
"""
AbstractLLM Apps - Command-line interface launcher

Usage:
    python -m abstractllm.apps <app_name> [options]

Available apps:
    summarizer    - Document summarization tool
    extractor     - Entity and relationship extraction tool

Examples:
    python -m abstractllm.apps summarizer document.txt
    python -m abstractllm.apps extractor report.txt --format json-ld
    python -m abstractllm.apps <app> --help
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
    else:
        print(f"Unknown app: {app_name}")
        print("\nAvailable apps: summarizer, extractor")
        sys.exit(1)

if __name__ == "__main__":
    main()
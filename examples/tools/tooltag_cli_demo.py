#!/usr/bin/env python3
"""
Demo script showing how to use the /tooltag command in the AbstractCore CLI.

This script demonstrates the new /tooltag command that allows testing
tool call tag rewriting with different models and streaming modes.

Usage:
    python examples/tools/tooltag_cli_demo.py
"""

import subprocess
import sys
from pathlib import Path

def run_cli_demo():
    """Run the CLI with /tooltag command examples."""
    print("🏷️ AbstractCore CLI Tool Tag Rewriting Demo")
    print("=" * 60)
    print()
    print("This demo shows how to use the new /tooltag command to test")
    print("tool call tag rewriting with different models and streaming modes.")
    print()
    print("The /tooltag command allows you to:")
    print("  • Test custom tool call tag formats")
    print("  • Compare with default behavior")
    print("  • Test both streaming and non-streaming modes")
    print("  • Verify tag rewriting works correctly")
    print()
    print("Example commands you can try:")
    print("  /tooltag '<function_call>' '</function_call>'")
    print("  /tooltag '<tool_call>' '</tool_call>'")
    print("  /tooltag '<|tool_call|>' '</|tool_call|>'")
    print("  /tooltag '```tool_code' '```'")
    print()
    print("You can also toggle streaming mode with /stream")
    print("and switch models with /model <provider:model>")
    print()
    print("Starting CLI...")
    print("=" * 60)
    
    # Start the CLI
    try:
        subprocess.run([
            sys.executable, "-m", "abstractcore.utils.cli",
            "--provider", "openai",
            "--model", "gpt-4o-mini",
            "--debug"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo ended")
    except Exception as e:
        print(f"❌ Error starting CLI: {e}")

if __name__ == "__main__":
    run_cli_demo()

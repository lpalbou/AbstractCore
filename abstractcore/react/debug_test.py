#!/usr/bin/env python3
"""
Debug test for ReAct implementation

This will run with full debugging to see exactly what's happening
"""

import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from abstractcore import create_llm
from abstractcore.react import ReActAgent


def debug_react_execution():
    """Run ReAct with full debugging to see what's happening"""
    print("🔍 DEBUG: ReAct Implementation Analysis")
    print("=" * 60)
    
    try:
        # Use the best available model from the list
        llm = create_llm("ollama", model="qwen3-coder:30b")
        print(f"✅ Connected to Ollama qwen3-coder:30b")
    except Exception as e:
        try:
            # Fallback to 4b model
            llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
            print(f"✅ Connected to Ollama qwen3:4b-instruct-2507-q4_K_M (fallback)")
        except Exception as e2:
            try:
                # Final fallback to small model
                llm = create_llm("ollama", model="gemma3:1b-it-qat")
                print(f"✅ Connected to Ollama gemma3:1b-it-qat (final fallback)")
            except Exception as e3:
                print(f"❌ Failed to connect to any model: {e3}")
                return False
    
    # Create ReAct agent with debugging enabled
    agent = ReActAgent(llm, verbose=True, max_iterations=3)  # Limit iterations for debugging
    
    print(f"\n🔍 DEBUG: Starting simple task with full debugging")
    print("=" * 60)
    
    # Simple task that should work
    goal = "List the files in the current directory"
    
    print(f"🎯 Goal: {goal}")
    print(f"🔧 Available Tools: {list(agent.available_tools.keys())}")
    print("\n" + "=" * 60)
    print("🔍 STARTING EXECUTION WITH FULL DEBUG")
    print("=" * 60)
    
    result = agent.run(goal)
    
    print("\n" + "=" * 60)
    print("🔍 EXECUTION COMPLETE - ANALYSIS")
    print("=" * 60)
    
    print(f"📊 Result: {result}")
    print(f"🔄 Iterations: {result.iterations}")
    print(f"⏱️  Time: {result.total_time_ms:.0f}ms")
    print(f"✅ Success: {result.success}")
    
    if result.error:
        print(f"❌ Error: {result.error}")
    
    print(f"\n📝 Scratchpad Analysis:")
    print(f"   Total entries: {len(agent.scratchpad.entries)}")
    
    for i, entry in enumerate(agent.scratchpad.entries, 1):
        print(f"   {i}. {entry.entry_type.upper()}: {entry.content[:100]}{'...' if len(entry.content) > 100 else ''}")
    
    return result.success


if __name__ == "__main__":
    success = debug_react_execution()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
ReAct Agent Example

Demonstrates how to use the ReAct agent for autonomous task execution.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractcore import create_llm
from abstractcore.react import ReActAgent


def main():
    """Run ReAct agent examples"""
    
    print("🤖 ReAct Agent Examples")
    print("=" * 50)
    
    # Create LLM (using Ollama with a small model for demonstration)
    try:
        llm = create_llm("ollama", model="gemma3:1b-it-qat")
        print(f"✅ Connected to Ollama with gemma3:1b-it-qat")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("💡 Make sure Ollama is running and gemma3:1b-it-qat is installed")
        print("   Install with: ollama pull gemma3:1b-it-qat")
        return
    
    # Create ReAct agent
    agent = ReActAgent(llm, verbose=True, max_iterations=5)
    
    # Example 1: File analysis
    print("\n" + "=" * 60)
    print("📁 Example 1: Analyze current directory structure")
    print("=" * 60)
    
    goal1 = "Find all Python files in the current directory and count how many lines of code they contain"
    result1 = agent.run(goal1)
    
    print(f"\n📊 Result: {result1}")
    
    # Example 2: Code search
    print("\n" + "=" * 60)
    print("🔍 Example 2: Search for specific patterns")
    print("=" * 60)
    
    goal2 = "Search for all TODO comments in Python files and list them with their locations"
    result2 = agent.run(goal2)
    
    print(f"\n📊 Result: {result2}")
    
    # Show scratchpad content
    if agent.scratchpad.has_content():
        print("\n" + "=" * 60)
        print("📝 Final Scratchpad Content")
        print("=" * 60)
        print(agent.get_scratchpad_content())
    
    print("\n✅ ReAct agent examples completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test ReAct agent with thorough file reading (no preview tool)
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


def test_thorough_reading():
    """Test ReAct agent with thorough file reading"""
    print("🧪 Testing ReAct Agent with Thorough File Reading")
    print("=" * 70)
    
    try:
        # Use LMStudio with qwen/qwen3-next-80b
        llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")
        print(f"✅ Connected to LMStudio qwen/qwen3-next-80b")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Create ReAct agent
    agent = ReActAgent(llm, verbose=True)
    print(f"🔧 Max Iterations: {agent.max_iterations}")
    print(f"🔧 Available Tools: {list(agent.available_tools.keys())}")
    
    # Test with a smaller task first - read 3 files thoroughly
    goal = "read 3 markdown files in docs/ (README.md, getting-started.md, and capabilities.md) and summarize them thoroughly in my-thorough-report.md"
    
    print(f"\n🎯 Goal: {goal}")
    print("=" * 70)
    
    import time
    start_time = time.time()
    
    result = agent.run(goal)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🏁 EXECUTION COMPLETE")
    print("=" * 70)
    
    print(f"📊 Result: {result}")
    print(f"🔄 Iterations: {result.iterations}")
    print(f"⏱️  Time: {duration:.1f}s ({result.total_time_ms:.0f}ms)")
    print(f"✅ Success: {result.success}")
    
    if result.error:
        print(f"❌ Error: {result.error}")
    
    # Check if the report was created
    if os.path.exists("my-thorough-report.md"):
        print("✅ Report file was created!")
        with open("my-thorough-report.md", "r") as f:
            content = f.read()
            print(f"📄 Report size: {len(content)} characters")
            print(f"📄 Report preview: {content[:300]}...")
    else:
        print("❌ Report file was not created")
    
    print(f"\n📝 Scratchpad Analysis:")
    print(f"   Total entries: {len(agent.scratchpad.entries)}")
    
    # Count different types of entries
    thoughts = sum(1 for entry in agent.scratchpad.entries if "thought" in entry.entry_type.lower())
    observations = sum(1 for entry in agent.scratchpad.entries if "observation" in entry.entry_type.lower())
    
    print(f"   Thoughts: {thoughts}")
    print(f"   Observations: {observations}")
    
    # Show recent activity
    print(f"\n📋 Recent Activity:")
    for entry in agent.scratchpad.entries[-3:]:
        entry_preview = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
        print(f"   • {entry.entry_type}: {entry_preview}")
    
    return result.success


if __name__ == "__main__":
    success = test_thorough_reading()
    sys.exit(0 if success else 1)

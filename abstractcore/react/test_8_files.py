#!/usr/bin/env python3
"""
Test ReAct agent with 8 files to verify timeout and efficiency fixes
"""

import sys
import os
from typing import Optional, Dict, Any

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from abstractcore import create_llm
from abstractcore.react import ReActAgent


def test_8_files_task():
    """Test ReAct agent with 8 files task"""
    print("🧪 Testing ReAct Agent with 8 Files Task")
    print("=" * 70)
    
    try:
        # Use LMStudio with qwen/qwen3-next-80b
        llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")
        print(f"✅ Connected to LMStudio qwen/qwen3-next-80b")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Create ReAct agent with increased iterations
    agent = ReActAgent(llm, verbose=True)
    print(f"🔧 Max Iterations: {agent.max_iterations}")
    
    # Test the 8 files task
    goal = "read 8 markdown files in docs/ and summarize them for me in my-report-test.md"
    
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
    if os.path.exists("my-report-test.md"):
        print("✅ Report file was created!")
        with open("my-report-test.md", "r") as f:
            content = f.read()
            print(f"📄 Report size: {len(content)} characters")
            print(f"📄 Report preview: {content[:200]}...")
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
    for entry in agent.scratchpad.entries[-5:]:
        entry_preview = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
        print(f"   • {entry.entry_type}: {entry_preview}")
    
    return result.success


if __name__ == "__main__":
    success = test_8_files_task()
    sys.exit(0 if success else 1)

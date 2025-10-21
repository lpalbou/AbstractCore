#!/usr/bin/env python3
"""
Test ReAct agent with context size tracking in scratchpad
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


def test_context_size_tracking():
    """Test ReAct agent with context size tracking"""
    print("🧪 Testing ReAct Agent with Context Size Tracking")
    print("=" * 70)
    
    try:
        # Use LMStudio with qwen/qwen3-next-80b
        llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")
        print(f"✅ Connected to LMStudio qwen/qwen3-next-80b")
        
        # Show token configuration
        print(f"🔧 Token Configuration:")
        print(llm.get_token_configuration_summary())
        
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Create ReAct agent
    agent = ReActAgent(llm, verbose=True)
    print(f"🔧 Max Iterations: {agent.max_iterations}")
    
    # Simple task to test context size tracking
    goal = "list files in the current directory and read README.md, then write a brief summary to context-test.md"
    
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
    
    # Show scratchpad with context size info
    print(f"\n📝 Scratchpad Analysis (with Context Size):")
    print(f"   Total entries: {len(agent.scratchpad.entries)}")
    
    # Show thoughts with context size
    thoughts = [entry for entry in agent.scratchpad.entries if "thought" in entry.entry_type.lower()]
    print(f"   Thoughts with context size: {len(thoughts)}")
    
    for i, entry in enumerate(thoughts):
        print(f"   {i+1}. {entry.content[:100]}{'...' if len(entry.content) > 100 else ''}")
    
    # Check if the output file was created
    if os.path.exists("context-test.md"):
        print("✅ Output file was created!")
        with open("context-test.md", "r") as f:
            content = f.read()
            print(f"📄 File size: {len(content)} characters")
    else:
        print("❌ Output file was not created")
    
    return result.success


if __name__ == "__main__":
    success = test_context_size_tracking()
    sys.exit(0 if success else 1)

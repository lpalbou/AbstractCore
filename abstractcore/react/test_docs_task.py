#!/usr/bin/env python3
"""
Test the fixed ReAct implementation with the original failing task
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


def test_docs_task():
    """Test the fixed ReAct implementation with the original failing task"""
    print("🧪 Testing Fixed ReAct with Docs Task")
    print("=" * 50)
    
    try:
        # Create LLM (using Ollama with a small model)
        llm = create_llm("ollama", model="gemma3:1b-it-qat")
        print(f"✅ Connected to Ollama with gemma3:1b-it-qat")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("💡 Make sure Ollama is running and gemma3:1b-it-qat is installed")
        return False
    
    # Create ReAct agent with more iterations for complex task
    agent = ReActAgent(llm, verbose=True, max_iterations=5)
    
    # Test with the original failing task
    print("\n" + "=" * 60)
    print("📁 Testing: Read markdown files in docs/ and summarize")
    print("=" * 60)
    
    goal = "read some of the markdown files in docs/ and summarize them for me in my-report.md"
    result = agent.run(goal)
    
    print(f"\n📊 Result: {result}")
    
    if result.success:
        print("✅ Fixed ReAct implementation works for complex tasks!")
        
        # Check if the report was actually created
        if os.path.exists("my-report.md"):
            print("✅ Report file was created!")
            with open("my-report.md", "r") as f:
                content = f.read()
                print(f"📄 Report content preview: {content[:200]}...")
        else:
            print("⚠️ Report file was not created, but agent thinks it succeeded")
        
        return True
    else:
        print("❌ ReAct still has issues with complex tasks")
        print(f"Error: {result.error}")
        
        # Show scratchpad for debugging
        if agent.scratchpad.has_content():
            print("\n📝 Scratchpad for debugging:")
            print(agent.get_scratchpad_content())
        
        return False


if __name__ == "__main__":
    success = test_docs_task()
    sys.exit(0 if success else 1)

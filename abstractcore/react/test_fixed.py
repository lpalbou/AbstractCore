#!/usr/bin/env python3
"""
Test the fixed ReAct implementation
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


def test_fixed_react():
    """Test the fixed ReAct implementation with a simple task"""
    print("🧪 Testing Fixed ReAct Implementation")
    print("=" * 50)
    
    try:
        # Create LLM (using Ollama with a small model)
        llm = create_llm("ollama", model="gemma3:1b-it-qat")
        print(f"✅ Connected to Ollama with gemma3:1b-it-qat")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("💡 Make sure Ollama is running and gemma3:1b-it-qat is installed")
        return False
    
    # Create ReAct agent with limited iterations for testing
    agent = ReActAgent(llm, verbose=True, max_iterations=3)
    
    # Test with a simple, achievable task
    print("\n" + "=" * 60)
    print("📁 Testing: List files in current directory")
    print("=" * 60)
    
    goal = "List the files in the current directory"
    result = agent.run(goal)
    
    print(f"\n📊 Result: {result}")
    
    if result.success:
        print("✅ Fixed ReAct implementation works!")
        return True
    else:
        print("❌ ReAct still has issues")
        print(f"Error: {result.error}")
        
        # Show scratchpad for debugging
        if agent.scratchpad.has_content():
            print("\n📝 Scratchpad for debugging:")
            print(agent.get_scratchpad_content())
        
        return False


if __name__ == "__main__":
    success = test_fixed_react()
    sys.exit(0 if success else 1)

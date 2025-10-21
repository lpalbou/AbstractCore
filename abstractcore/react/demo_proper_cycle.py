#!/usr/bin/env python3
"""
Demonstrate the proper ReAct Think-Act-Observe cycle

This script shows how the fixed ReAct implementation follows the canonical pattern
with a more capable model that can properly follow instructions.
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


def demo_proper_react_cycle():
    """Demonstrate proper ReAct cycle with a capable model"""
    print("🎯 ReAct Proper Cycle Demonstration")
    print("=" * 60)
    
    # Try different models in order of preference
    models_to_try = [
        ("openai", "gpt-4o-mini", "OpenAI GPT-4o-mini"),
        ("anthropic", "claude-3-haiku-20240307", "Anthropic Claude 3 Haiku"),
        ("ollama", "qwen2.5:7b", "Ollama Qwen2.5 7B"),
        ("ollama", "llama3.1:8b", "Ollama Llama 3.1 8B"),
    ]
    
    llm = None
    model_info = None
    
    for provider, model, description in models_to_try:
        try:
            print(f"🔄 Trying {description}...")
            llm = create_llm(provider, model=model)
            model_info = description
            print(f"✅ Connected to {description}")
            break
        except Exception as e:
            print(f"❌ Failed to connect to {description}: {e}")
            continue
    
    if not llm:
        print("❌ No suitable model available. Please ensure you have:")
        print("   - OpenAI API key for GPT-4o-mini, OR")
        print("   - Anthropic API key for Claude 3 Haiku, OR") 
        print("   - Ollama running with qwen2.5:7b or llama3.1:8b")
        return False
    
    # Create ReAct agent
    agent = ReActAgent(llm, verbose=True, max_iterations=4)
    
    print(f"\n🤖 Using {model_info}")
    print("=" * 60)
    print("📋 Task: List files in the current directory and count them")
    print("=" * 60)
    
    # Simple task that requires Think-Act-Observe cycle
    goal = "List the files in the current directory and tell me how many files there are"
    result = agent.run(goal)
    
    print(f"\n📊 Final Result: {result}")
    
    if result.success and result.iterations > 0:
        print("✅ SUCCESS: ReAct agent followed proper Think-Act-Observe cycle!")
        print(f"   Completed in {result.iterations} iterations")
        
        # Show the cycle structure
        if agent.scratchpad.has_content():
            print("\n📝 Reasoning Process:")
            print("=" * 40)
            entries = agent.scratchpad.entries
            for i, entry in enumerate(entries, 1):
                entry_type = entry.entry_type.upper()
                content_preview = entry.content[:80] + "..." if len(entry.content) > 80 else entry.content
                print(f"{i}. {entry_type}: {content_preview}")
        
        return True
    else:
        print("❌ FAILED: Agent did not follow proper cycle or failed to complete task")
        if result.error:
            print(f"   Error: {result.error}")
        
        # Show debugging info
        if agent.scratchpad.has_content():
            print("\n📝 Debug - Scratchpad content:")
            print(agent.get_scratchpad_content())
        
        return False


if __name__ == "__main__":
    success = demo_proper_react_cycle()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ReAct Implementation Working Correctly!")
        print("   The agent follows the canonical Think-Act-Observe pattern:")
        print("   1. THINK: Analyzes situation and plans action")
        print("   2. ACT: Executes the planned action")
        print("   3. OBSERVE: Records results and returns to THINK")
        print("   4. Repeats until task is complete")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  ReAct Implementation Needs More Work")
        print("   Consider using a more capable model for complex reasoning")
        print("=" * 60)
    
    sys.exit(0 if success else 1)

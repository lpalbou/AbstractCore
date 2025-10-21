#!/usr/bin/env python3
"""
Proper ReAct test using LMStudio with qwen/qwen3-next-80b and structured responses
"""

import sys
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from abstractcore import create_llm
from abstractcore.react import ReActAgent


class ReActResponse(BaseModel):
    """Structured response for ReAct agent"""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


def test_react_with_proper_model():
    """Test ReAct with LMStudio qwen/qwen3-next-80b using structured responses"""
    print("🔍 PROPER ReAct Test with LMStudio qwen/qwen3-next-80b")
    print("=" * 70)
    
    try:
        # Use LMStudio with qwen/qwen3-next-80b as specified
        llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")
        print(f"✅ Connected to LMStudio qwen/qwen3-next-80b")
    except Exception as e:
        print(f"❌ Failed to connect to LMStudio qwen/qwen3-next-80b: {e}")
        print("💡 Make sure LMStudio is running with qwen/qwen3-next-80b loaded")
        return False
    
    # Create ReAct agent with debugging (using default max_iterations=15)
    agent = ReActAgent(llm, verbose=True)
    
    print(f"\n🎯 Testing with the original failing task:")
    print("=" * 70)
    
    # Use the original failing task
    goal = "read some of the markdown files in docs/ and summarize them for me in my-report.md"
    
    print(f"📋 Goal: {goal}")
    print(f"🔧 Available Tools: {list(agent.available_tools.keys())}")
    print("\n" + "=" * 70)
    print("🚀 STARTING EXECUTION")
    print("=" * 70)
    
    result = agent.run(goal)
    
    print("\n" + "=" * 70)
    print("🏁 EXECUTION COMPLETE")
    print("=" * 70)
    
    print(f"📊 Result: {result}")
    print(f"🔄 Iterations: {result.iterations}")
    print(f"⏱️  Time: {result.total_time_ms:.0f}ms")
    print(f"✅ Success: {result.success}")
    
    if result.error:
        print(f"❌ Error: {result.error}")
    
    # Check if the report was actually created
    if os.path.exists("my-report.md"):
        print("✅ Report file was created!")
        with open("my-report.md", "r") as f:
            content = f.read()
            print(f"📄 Report content preview: {content[:300]}...")
    else:
        print("❌ Report file was not created")
    
    print(f"\n📝 Scratchpad Analysis:")
    print(f"   Total entries: {len(agent.scratchpad.entries)}")
    
    for i, entry in enumerate(agent.scratchpad.entries, 1):
        print(f"   {i}. {entry.entry_type.upper()}: {entry.content[:100]}{'...' if len(entry.content) > 100 else ''}")
    
    return result.success


def test_structured_react_response():
    """Test using structured response format for ReAct"""
    print("\n🔍 Testing Structured ReAct Response Format")
    print("=" * 70)
    
    try:
        llm = create_llm("lmstudio", model="qwen/qwen3-next-80b")
        print(f"✅ Connected to LMStudio qwen/qwen3-next-80b")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    # Test structured response directly
    prompt = """You are a ReAct agent. Given this task: "List files in current directory"

Available tools:
- list_files: List files in a directory. Input: {"directory": "path"}

Respond with your thought and action in the exact format specified by the response model.
Either provide an action to take, or a final answer if you have enough information."""
    
    try:
        response = llm.generate(prompt, response_model=ReActResponse)
        print(f"📊 Structured Response:")
        print(f"   Thought: {response.thought}")
        print(f"   Action: {response.action}")
        print(f"   Action Input: {response.action_input}")
        print(f"   Final Answer: {response.final_answer}")
        return True
    except Exception as e:
        print(f"❌ Structured response failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 PROPER ReAct Testing with Real Model")
    print("=" * 70)
    
    # Test 1: Full ReAct execution
    success1 = test_react_with_proper_model()
    
    # Test 2: Structured response format
    success2 = test_structured_react_response()
    
    overall_success = success1 or success2
    
    print(f"\n" + "=" * 70)
    print(f"🏁 FINAL RESULTS")
    print("=" * 70)
    print(f"Full ReAct Test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Structured Response Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Overall: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
    
    sys.exit(0 if overall_success else 1)

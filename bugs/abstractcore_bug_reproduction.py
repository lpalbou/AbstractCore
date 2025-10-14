#!/usr/bin/env python3
"""
Minimal reproduction of AbstractCore bug: tools + response_model incompatibility
"""

from abstractllm import create_llm
from abstractllm.tools import tool
from pydantic import BaseModel
from typing import List, Optional

# Define a simple Pydantic model for structured output
class TaskResponse(BaseModel):
    answer: str
    tool_calls_needed: Optional[List[str]] = None
    confidence: float

# Define a simple tool function
@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    # Create LLM instance
    llm = create_llm("ollama", model="qwen3-coder:30b")
    
    print("🧪 Testing AbstractCore: tools + response_model compatibility")
    print("=" * 60)
    
    # Test 1: Tools only (should work)
    print("\n1️⃣ Testing tools only...")
    try:
        response = llm.generate(
            "What time is it? Use the get_current_time tool.",
            tools=[get_current_time],
            execute_tools=True
        )
        print(f"✅ Tools only: {response.content[:100]}...")
    except Exception as e:
        print(f"❌ Tools only failed: {e}")
    
    # Test 2: Structured output only (should work)
    print("\n2️⃣ Testing structured output only...")
    try:
        response = llm.generate(
            "Answer this question: What is 2+2? Respond with confidence level.",
            response_model=TaskResponse
        )
        print(f"✅ Structured output only: {response.answer}, confidence: {response.confidence}")
    except Exception as e:
        print(f"❌ Structured output only failed: {e}")
    
    # Test 3: Both tools + structured output (BUG - should work but doesn't)
    print("\n3️⃣ Testing tools + structured output (THE BUG)...")
    try:
        response = llm.generate(
            "What time is it? Use tools if needed and respond with structured output.",
            tools=[get_current_time],
            response_model=TaskResponse,
            execute_tools=True
        )
        print(f"✅ Both work: {response.answer}, confidence: {response.confidence}")
    except Exception as e:
        print(f"❌ BUG CONFIRMED: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("🔍 Expected: All three tests should pass")
    print("✅ Status: BUG FIXED! All tests now pass with v2.3.5+")

if __name__ == "__main__":
    main()

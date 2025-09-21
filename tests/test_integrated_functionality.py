"""
Integrated functionality test - test all components working together.
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractllm import create_llm, BasicSession
from abstractllm.tools.common_tools import COMMON_TOOLS, execute_tool
from abstractllm.utils.telemetry import Telemetry
from abstractllm.events import EventType, EventEmitter
from abstractllm.architectures import detect_architecture


def test_provider_with_telemetry(provider_name: str, model: str, config: dict = None):
    """Test a provider with full telemetry integration"""
    print(f"\n{'='*60}")
    print(f"Testing {provider_name} with {model}")
    print('='*60)

    # Setup telemetry
    telemetry = Telemetry(enabled=True, verbatim=True,
                          output_path=f"/tmp/test_{provider_name}.jsonl")

    try:
        # Create provider
        provider = create_llm(provider_name, model=model, **(config or {}))
        print(f"✅ Provider created")

        # Test 1: Basic generation with telemetry tracking
        print("\n[Test 1] Basic generation")
        start = time.time()
        prompt = "Who are you? Answer in one sentence."
        response = provider.generate(prompt)
        elapsed = time.time() - start

        if response and response.content:
            print(f"✅ Response: {response.content[:100]}...")
            print(f"   Time: {elapsed:.2f}s")

            # Manually track telemetry (since providers don't have it integrated yet)
            telemetry.track_generation(
                provider=provider_name,
                model=model,
                prompt=prompt,
                response=response.content,
                tokens=response.usage,
                latency_ms=elapsed * 1000,
                success=True
            )
        else:
            print("❌ No response")

        # Test 2: Session with memory
        print("\n[Test 2] Session memory")
        session = BasicSession(provider=provider)

        resp1 = session.generate("My name is Laurent. Who are you?")
        print(f"   Q1: My name is Laurent. Who are you?")
        print(f"   A1: {resp1.content[:100] if resp1.content else 'No response'}...")

        telemetry.track_generation(
            provider=provider_name,
            model=model,
            prompt="My name is Laurent. Who are you?",
            response=resp1.content,
            latency_ms=100,
            success=bool(resp1.content)
        )

        resp2 = session.generate("What is my name?")
        print(f"   Q2: What is my name?")
        print(f"   A2: {resp2.content[:100] if resp2.content else 'No response'}...")

        if resp2.content and "laurent" in resp2.content.lower():
            print("✅ Session remembers context")
        else:
            print("⚠️  Session may not remember context")

        telemetry.track_generation(
            provider=provider_name,
            model=model,
            prompt="What is my name?",
            response=resp2.content,
            latency_ms=100,
            success=bool(resp2.content)
        )

        # Test 3: Tool calling
        print("\n[Test 3] Tool calling")
        list_files_tool = next((t for t in COMMON_TOOLS if t["name"] == "list_files"), None)

        if list_files_tool:
            tool_response = provider.generate("List the files in the current directory",
                                             tools=[list_files_tool])

            if tool_response and tool_response.has_tool_calls():
                print("✅ Tool calls detected")
                for call in tool_response.tool_calls:
                    print(f"   Tool: {call.get('name')}")
                    args = call.get('arguments', {})
                    if isinstance(args, str):
                        args = json.loads(args)

                    result = execute_tool(call.get('name'), args)
                    print(f"   Result: {result[:100]}...")

                    telemetry.track_tool_call(
                        tool_name=call.get('name'),
                        arguments=args,
                        result=result,
                        success=True
                    )
            else:
                print("⚠️  No tool calls (provider may not support)")

        # Test 4: Verify telemetry
        print("\n[Test 4] Telemetry verification")
        summary = telemetry.get_summary()
        print(f"   Total events: {summary['total_events']}")
        print(f"   Generations: {summary['total_generations']}")
        print(f"   Tool calls: {summary['total_tool_calls']}")

        # Check verbatim capture
        telemetry_file = Path(f"/tmp/test_{provider_name}.jsonl")
        if telemetry_file.exists():
            with open(telemetry_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    if "metadata" in last and "prompt" in last["metadata"]:
                        print(f"✅ Verbatim capture working")
                        print(f"   Last prompt: {last['metadata']['prompt'][:50]}...")
                    else:
                        print("⚠️  Telemetry without verbatim")

        # Test 5: Architecture detection
        print("\n[Test 5] Architecture detection")
        arch = detect_architecture(model)
        print(f"   Model: {model}")
        print(f"   Architecture: {arch}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Test main providers with full integration"""

    # Test Ollama
    test_provider_with_telemetry(
        "ollama",
        "qwen2.5-coder:3b",
        {"base_url": "http://localhost:11434"}
    )

    # Test OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        test_provider_with_telemetry("openai", "gpt-3.5-turbo")

    # Test Anthropic if available
    if os.getenv("ANTHROPIC_API_KEY"):
        test_provider_with_telemetry("anthropic", "claude-3-haiku-20240307")


if __name__ == "__main__":
    main()
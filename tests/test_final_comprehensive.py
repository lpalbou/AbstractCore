"""
Final comprehensive test suite with all requested functionality.
Tests each provider with increasing complexity and full observability.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractllm import create_llm, BasicSession
from abstractllm.tools.common_tools import list_files, search_files, read_file, write_file, web_search
from abstractllm.utils import configure_logging, get_logger
from abstractllm.architectures import detect_architecture
from abstractllm.events import EventType, EventEmitter


class ComprehensiveProviderTest:
    """Complete test suite for a provider as requested"""

    def __init__(self, provider_name: str, model: str, config: dict = None):
        self.provider_name = provider_name
        self.model = model
        self.config = config or {}
        self.results = []

        # Setup logging with verbatim for complete observability
        configure_logging(
            console_level=30,  # WARNING
            file_level=10,     # DEBUG
            log_dir="/tmp",
            verbatim_enabled=True  # CRITICAL: Capture full requests/responses
        )
        self.logger = get_logger(f"test.{provider_name}")

    def test_1_connectivity(self) -> bool:
        """Test 1: Provider connectivity"""
        print(f"\n[TEST 1] Connectivity for {self.provider_name}")
        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)
            print(f"✅ Connected to {self.provider_name}")
            self.results.append(("1_connectivity", True, "Connected"))
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            self.results.append(("1_connectivity", False, str(e)))
            return False

    def test_2_who_are_you(self) -> bool:
        """Test 2: Ask 'who are you?' and confirm answer"""
        print(f"\n[TEST 2] 'Who are you?' test")
        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)

            prompt = "Who are you?"
            start = time.time()
            response = provider.generate(prompt)
            latency_ms = (time.time() - start) * 1000

            if response and response.content:
                print(f"✅ Response received in {latency_ms:.0f}ms")
                print(f"   Response: {response.content[:150]}...")

                # Track with telemetry for observability
                self.logger.log_generation(
                    provider=self.provider_name,
                    model=self.model,
                    prompt=prompt,  # Full prompt captured
                    response=response.content,  # Full response captured
                    tokens=response.usage,
                    latency_ms=latency_ms,
                    success=True
                )

                self.results.append(("2_who_are_you", True, response.content[:50]))
                return True
            else:
                print(f"❌ No response")
                self.results.append(("2_who_are_you", False, "No response"))
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            self.logger.log_generation(
                provider=self.provider_name,
                model=self.model,
                prompt="Who are you?",
                response=None,
                latency_ms=0,
                success=False,
                error=str(e)
            )
            self.results.append(("2_who_are_you", False, str(e)))
            return False

    def test_3_session_memory(self) -> bool:
        """Test 3: 'I am Laurent, who are you?' then 'What is my name?'"""
        print(f"\n[TEST 3] Session memory test")
        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)
            session = BasicSession(provider=provider)

            # First message
            prompt1 = "I am Laurent, who are you?"
            print(f"   Message 1: {prompt1}")
            start1 = time.time()
            response1 = session.generate(prompt1)
            latency1 = (time.time() - start1) * 1000

            if response1 and response1.content:
                print(f"   Response 1: {response1.content[:100]}...")

                # Track with full observability
                self.logger.log_generation(
                    provider=self.provider_name,
                    model=self.model,
                    prompt=prompt1,
                    response=response1.content,
                    tokens=response1.usage,
                    latency_ms=latency1,
                    success=True
                )

            # Second message - test memory
            prompt2 = "What is my name?"
            print(f"   Message 2: {prompt2}")
            start2 = time.time()
            response2 = session.generate(prompt2)
            latency2 = (time.time() - start2) * 1000

            if response2 and response2.content:
                print(f"   Response 2: {response2.content[:100]}...")

                # Track with full observability
                self.logger.log_generation(
                    provider=self.provider_name,
                    model=self.model,
                    prompt=prompt2,
                    response=response2.content,
                    tokens=response2.usage,
                    latency_ms=latency2,
                    success=True
                )

                # Check if it remembers "Laurent"
                if "laurent" in response2.content.lower():
                    print(f"✅ Session remembers 'Laurent'")
                    self.results.append(("3_session_memory", True, "Remembers Laurent"))
                    return True
                else:
                    print(f"⚠️  Session doesn't mention 'Laurent'")
                    self.results.append(("3_session_memory", False, "No memory"))
                    return False
            else:
                print(f"❌ No response to second question")
                self.results.append(("3_session_memory", False, "No response"))
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            self.results.append(("3_session_memory", False, str(e)))
            return False

    def test_4_tool_calling(self) -> bool:
        """Test 4: Ask 'list the local files' - should use list_files tool"""
        print(f"\n[TEST 4] Tool calling test")
        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)

            # Use enhanced list_files tool
            tools = [list_files]

            prompt = "List the local files"
            print(f"   Prompt: {prompt}")
            print(f"   Available tool: list_files")

            start = time.time()
            response = provider.generate(prompt, tools=tools)
            latency = (time.time() - start) * 1000

            # Track the request
            self.logger.log_generation(
                provider=self.provider_name,
                model=self.model,
                prompt=prompt,
                response=response.content if response else None,
                tokens=response.usage if response else None,
                latency_ms=latency,
                success=bool(response)
            )

            if response and response.has_tool_calls():
                print(f"✅ Tool calls detected")
                for call in response.tool_calls:
                    tool_name = call.get("name")
                    args = call.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)

                    print(f"   Tool: {tool_name}")
                    print(f"   Arguments: {args}")

                    # Execute the tool directly
                    available_tools = {
                        "list_files": list_files,
                        "search_files": search_files,
                        "read_file": read_file,
                        "write_file": write_file,
                        "web_search": web_search
                    }

                    if tool_name in available_tools:
                        result = available_tools[tool_name](**args)
                    else:
                        result = f"Error: Tool '{tool_name}' not found"
                    print(f"   Result: {result[:150]}...")

                    # Track tool call with full observability
                    self.logger.log_tool_call(
                        tool_name=tool_name,
                        arguments=args,
                        result=result,
                        success=True
                    )

                self.results.append(("4_tool_calling", True, "Tool executed"))
                return True
            elif response and response.content:
                print(f"⚠️  No tool calls, got text response")
                print(f"   Response: {response.content[:100]}...")
                self.results.append(("4_tool_calling", False, "Text response only"))
                return False
            else:
                print(f"❌ No response")
                self.results.append(("4_tool_calling", False, "No response"))
                return False

        except Exception as e:
            print(f"❌ Error: {e}")
            self.results.append(("4_tool_calling", False, str(e)))
            return False

    def test_5_verify_telemetry(self) -> bool:
        """Test 5: Verify telemetry has COMPLETE observability with VERBATIM"""
        print(f"\n[TEST 5] Telemetry and observability verification")

        telemetry_file = Path(f"/tmp/abstractllm_{self.provider_name}_final.jsonl")

        if not telemetry_file.exists():
            print(f"❌ No telemetry file")
            self.results.append(("5_telemetry", False, "No file"))
            return False

        try:
            with open(telemetry_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                print(f"❌ Empty telemetry")
                self.results.append(("5_telemetry", False, "Empty"))
                return False

            # Check for verbatim capture
            has_verbatim = False
            sample_prompt = None
            sample_response = None

            for line in lines:
                entry = json.loads(line)
                metadata = entry.get("metadata", {})

                if "prompt" in metadata and "response" in metadata:
                    has_verbatim = True
                    sample_prompt = metadata["prompt"]
                    sample_response = metadata["response"]
                    break

            if has_verbatim:
                print(f"✅ VERBATIM telemetry captured")
                print(f"   Sample prompt: {sample_prompt[:50]}...")
                print(f"   Sample response: {sample_response[:50] if sample_response else 'None'}...")

                # Show summary
                # Logging summary not needed for structured logging
                print(f"\n   Telemetry Summary:")
                print(f"   - Total events: {summary['total_events']}")
                print(f"   - Generations: {summary['total_generations']}")
                print(f"   - Tool calls: {summary['total_tool_calls']}")
                print(f"   - Success rate: {summary['success_rate']:.1%}")

                self.results.append(("5_telemetry", True, "Verbatim captured"))
                return True
            else:
                print(f"❌ No verbatim capture found")
                self.results.append(("5_telemetry", False, "No verbatim"))
                return False

        except Exception as e:
            print(f"❌ Error checking telemetry: {e}")
            self.results.append(("5_telemetry", False, str(e)))
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all 5 tests as requested"""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST: {self.provider_name.upper()} ({self.model})")
        print('='*60)

        # Run tests in order
        self.test_1_connectivity()
        if self.results[-1][1]:  # Only continue if connected
            self.test_2_who_are_you()
            self.test_3_session_memory()
            self.test_4_tool_calling()
            self.test_5_verify_telemetry()

        # Summary
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)
        success_rate = (passed / total * 100) if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"RESULTS: {self.provider_name.upper()}")
        print(f"Passed: {passed}/{total} ({success_rate:.0f}%)")
        print('='*60)

        for test, success, detail in self.results:
            status = "✅" if success else "❌"
            print(f"  {status} {test}: {detail[:50]}")

        # Architecture info
        arch = detect_architecture(self.model)
        print(f"\nArchitecture: {arch.value}")

        return {
            "provider": self.provider_name,
            "model": self.model,
            "passed": passed,
            "total": total,
            "success_rate": success_rate,
            "architecture": arch.value,
            "results": self.results
        }


def main():
    """Run final comprehensive tests as requested"""

    print("\n" + "="*60)
    print("FINAL COMPREHENSIVE TEST SUITE")
    print("Testing all providers with full observability")
    print("="*60)

    # Test configurations
    test_configs = [
        # Ollama with available model
        {
            "provider": "ollama",
            "model": "qwen3-coder:30b",  # Using actual available model
            "config": {"base_url": "http://localhost:11434"}
        },
        # MLX
        {
            "provider": "mlx",
            "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            "config": {}
        }
    ]

    # Add cloud providers if available
    if os.getenv("OPENAI_API_KEY"):
        test_configs.append({
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "config": {}
        })

    if os.getenv("ANTHROPIC_API_KEY"):
        test_configs.append({
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307",
            "config": {}
        })

    # Run all tests
    all_results = []
    for config in test_configs:
        tester = ComprehensiveProviderTest(
            provider_name=config["provider"],
            model=config["model"],
            config=config["config"]
        )
        results = tester.run_all_tests()
        all_results.append(results)

    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)

    total_providers = len(all_results)
    perfect_providers = sum(1 for r in all_results if r["success_rate"] == 100)

    for result in all_results:
        status = "✅" if result["success_rate"] == 100 else "⚠️"
        print(f"{status} {result['provider'].upper()}: {result['passed']}/{result['total']} "
              f"({result['success_rate']:.0f}%) - {result['architecture']}")

    print(f"\nProviders with 100% success: {perfect_providers}/{total_providers}")

    # Check telemetry files
    print("\nTelemetry Files Created:")
    for provider in ["ollama", "mlx", "openai", "anthropic"]:
        file_path = Path(f"/tmp/abstractllm_{provider}_final.jsonl")
        if file_path.exists():
            size = file_path.stat().st_size
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            print(f"  {provider}: {lines} events, {size} bytes")

    return perfect_providers == total_providers


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
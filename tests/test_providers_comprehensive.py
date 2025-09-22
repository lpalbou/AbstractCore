"""
Comprehensive test suite for all providers with increasing complexity.
Tests real implementations with telemetry and observability.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractllm import create_llm, BasicSession
from abstractllm.tools.common_tools import list_files, search_files, read_file, write_file, web_search
from abstractllm.utils import configure_logging, get_logger
from abstractllm.events import EventType, GlobalEventBus


class ProviderTestSuite:
    """Comprehensive test suite for a provider"""

    def __init__(self, provider_name: str, model: str, config: Dict[str, Any] = None):
        self.provider_name = provider_name
        self.model = model
        self.config = config or {}
        self.results = []

        # Setup logging with verbatim capture
        configure_logging(
            console_level=30,  # WARNING
            file_level=10,     # DEBUG
            log_dir="/tmp",
            verbatim_enabled=True
        )
        self.logger = get_logger(f"test.{provider_name}")

        # Setup event listener
        self.events = []
        GlobalEventBus.on(EventType.AFTER_GENERATE, self._record_event)
        GlobalEventBus.on(EventType.TOOL_CALLED, self._record_event)

    def _record_event(self, event):
        """Record events for verification"""
        self.events.append({
            "type": event.type.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data
        })

    def test_connectivity(self) -> bool:
        """Test 1: Provider connectivity"""
        print(f"\n[TEST 1] Testing connectivity for {self.provider_name}")

        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)
            print(f"✅ Provider created successfully")
            self.results.append(("connectivity", True, "Connected"))
            return True
        except Exception as e:
            print(f"❌ Failed to create provider: {e}")
            self.results.append(("connectivity", False, str(e)))
            return False

    def test_basic_generation(self) -> bool:
        """Test 2: Basic 'who are you' query"""
        print(f"\n[TEST 2] Testing basic generation: 'who are you?'")

        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)

            start = time.time()
            response = provider.generate("Who are you?")
            elapsed = time.time() - start

            if response and response.content:
                print(f"✅ Response received in {elapsed:.2f}s")
                print(f"   Response: {response.content[:100]}...")

                # Verify telemetry captured
                if self._verify_telemetry("Who are you?", response.content):
                    print(f"✅ Telemetry captured verbatim")

                self.results.append(("basic_generation", True, response.content[:100]))
                return True
            else:
                print(f"❌ No response received")
                self.results.append(("basic_generation", False, "No response"))
                return False

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            self.results.append(("basic_generation", False, str(e)))
            return False

    def test_session_memory(self) -> bool:
        """Test 3: Session memory with context"""
        print(f"\n[TEST 3] Testing session memory")

        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)
            session = BasicSession(provider=provider)

            # First message
            print("   Sending: 'I am Laurent, who are you?'")
            response1 = session.generate("I am Laurent, who are you?")
            print(f"   Response 1: {response1.content[:100]}...")

            # Second message testing memory
            print("   Sending: 'What is my name?'")
            response2 = session.generate("What is my name?")
            print(f"   Response 2: {response2.content[:100]}...")

            # Check if context is maintained
            if "laurent" in response2.content.lower():
                print(f"✅ Session maintains context - remembers 'Laurent'")
                self.results.append(("session_memory", True, "Context maintained"))
                return True
            else:
                print(f"⚠️  Session may not be maintaining context")
                self.results.append(("session_memory", False, "No context"))
                return False

        except Exception as e:
            print(f"❌ Session test failed: {e}")
            self.results.append(("session_memory", False, str(e)))
            return False

    def test_tool_calling(self) -> bool:
        """Test 4: Tool calling with list_files"""
        print(f"\n[TEST 4] Testing tool calling with list_files")

        try:
            provider = create_llm(self.provider_name, model=self.model, **self.config)

            # Use enhanced list_files tool
            tools = [list_files]

            prompt = "Please list the files in the current directory"
            print(f"   Prompt: {prompt}")
            print(f"   Available tool: list_files")

            response = provider.generate(prompt, tools=tools)

            if response and response.has_tool_calls():
                print(f"✅ Tool calls detected")
                for call in response.tool_calls:
                    tool_name = call.get("name")
                    arguments = call.get("arguments")

                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)

                    print(f"   Tool: {tool_name}")
                    print(f"   Arguments: {arguments}")

                    # Execute the tool
                    # Execute tool directly
                    available_tools = {"list_files": list_files, "search_files": search_files, "read_file": read_file, "write_file": write_file, "web_search": web_search}
                    tool_name = tool_name
                    args = arguments
                    if tool_name in available_tools:
                        result = available_tools[tool_name](**args)
                    else:
                        result = f"Error: Tool '{tool_name}' not found"
                    print(f"   Result: {result[:200]}...")

                self.results.append(("tool_calling", True, "Tool executed"))
                return True
            elif response and response.content:
                print(f"⚠️  No tool calls, got text response instead")
                print(f"   Response: {response.content[:100]}...")
                self.results.append(("tool_calling", False, "No tool calls"))
                return False
            else:
                print(f"❌ No response")
                self.results.append(("tool_calling", False, "No response"))
                return False

        except Exception as e:
            print(f"❌ Tool calling failed: {e}")
            self.results.append(("tool_calling", False, str(e)))
            return False

    def test_telemetry_verbatim(self) -> bool:
        """Test 5: Verify telemetry captures verbatim"""
        print(f"\n[TEST 5] Verifying telemetry verbatim capture")

        try:
            # Check telemetry file exists
            if not self.telemetry_file.exists():
                print(f"❌ Telemetry file not found: {self.telemetry_file}")
                self.results.append(("telemetry", False, "File not found"))
                return False

            # Read telemetry data
            with open(self.telemetry_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                print(f"❌ No telemetry data captured")
                self.results.append(("telemetry", False, "No data"))
                return False

            # Parse last telemetry entry
            last_entry = json.loads(lines[-1])

            if "metadata" in last_entry and "prompt" in last_entry["metadata"]:
                prompt = last_entry["metadata"]["prompt"]
                response = last_entry["metadata"].get("response", "")
                print(f"✅ Verbatim telemetry captured")
                print(f"   Last prompt: {prompt[:50]}...")
                print(f"   Last response: {response[:50]}..." if response else "   No response")
                self.results.append(("telemetry", True, "Verbatim captured"))
                return True
            else:
                print(f"⚠️  Telemetry captured but no verbatim data")
                self.results.append(("telemetry", False, "No verbatim"))
                return False

        except Exception as e:
            print(f"❌ Telemetry verification failed: {e}")
            self.results.append(("telemetry", False, str(e)))
            return False

    def _verify_telemetry(self, expected_prompt: str, expected_response: str) -> bool:
        """Verify telemetry contains expected data"""
        try:
            if self.telemetry_file.exists():
                with open(self.telemetry_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        metadata = last.get("metadata", {})
                        return (expected_prompt in metadata.get("prompt", "") and
                               expected_response[:50] in metadata.get("response", ""))
        except:
            pass
        return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST SUITE: {self.provider_name.upper()}")
        print('='*60)

        # Run tests in order of increasing complexity
        self.test_connectivity()
        if self.results[-1][1]:  # Only continue if connected
            self.test_basic_generation()
            self.test_session_memory()
            self.test_tool_calling()
            self.test_telemetry_verbatim()

        # Summary
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)

        print(f"\n{'='*60}")
        print(f"RESULTS: {self.provider_name.upper()}")
        print(f"Passed: {passed}/{total}")
        for test, success, detail in self.results:
            status = "✅" if success else "❌"
            print(f"  {status} {test}: {detail[:50]}")

        # Check events
        print(f"\nEvents captured: {len(self.events)}")
        if self.events:
            print(f"  Last event: {self.events[-1]['type']}")

        return {
            "provider": self.provider_name,
            "passed": passed,
            "total": total,
            "results": self.results,
            "events_count": len(self.events)
        }


def main():
    """Run comprehensive tests for all providers"""

    # Test configurations
    test_configs = [
        # Local providers
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:3b",
            "config": {"base_url": "http://localhost:11434"}
        },
        {
            "provider": "mlx",
            "model": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
            "config": {}
        }
    ]

    # Add cloud providers if API keys available
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

    # Run tests for each provider
    all_results = []

    for config in test_configs:
        suite = ProviderTestSuite(
            provider_name=config["provider"],
            model=config["model"],
            config=config["config"]
        )
        results = suite.run_all_tests()
        all_results.append(results)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)

    total_passed = sum(r["passed"] for r in all_results)
    total_tests = sum(r["total"] for r in all_results)

    for result in all_results:
        status = "✅" if result["passed"] == result["total"] else "⚠️"
        print(f"{status} {result['provider'].upper()}: {result['passed']}/{result['total']} passed")

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    # Verify telemetry is working
    # Using structured logging instead of telemetry
    summary = telemetry.get_summary()
    print(f"\nTelemetry Summary:")
    print(f"  Total events: {summary['total_events']}")
    print(f"  Generations: {summary['total_generations']}")
    print(f"  Tool calls: {summary['total_tool_calls']}")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
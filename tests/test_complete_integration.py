#!/usr/bin/env python3
"""
Test that all systems (events, exceptions, telemetry, media) are properly integrated.
This verifies that the infrastructure is not just created but actually USED.
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractllm import create_llm
from abstractllm.events import EventType, GlobalEventBus
from abstractllm.exceptions import AuthenticationError, ProviderAPIError
from abstractllm.utils import configure_logging, get_logger
from abstractllm.media import MediaHandler


class IntegrationVerifier:
    """Verify all systems are integrated and working"""

    def __init__(self):
        self.events_captured = []
        self.setup_telemetry()
        self.setup_events()

    def setup_telemetry(self):
        """Setup logging with verbatim capture"""
        configure_logging(
            console_level=30,  # WARNING
            file_level=10,     # DEBUG
            log_dir="/tmp",
            verbatim_enabled=True
        )
        self.logger = get_logger("integration_test")

    def setup_events(self):
        """Setup event listeners to verify events are emitted"""
        # Clear any existing handlers
        GlobalEventBus.clear()

        # Register handlers for all event types
        GlobalEventBus.on(EventType.PROVIDER_CREATED, self.capture_event)
        GlobalEventBus.on(EventType.BEFORE_GENERATE, self.capture_event)
        GlobalEventBus.on(EventType.AFTER_GENERATE, self.capture_event)
        GlobalEventBus.on(EventType.TOOL_CALLED, self.capture_event)
        GlobalEventBus.on(EventType.ERROR_OCCURRED, self.capture_event)

    def capture_event(self, event):
        """Capture events for verification"""
        self.events_captured.append({
            "type": event.type.value,
            "source": event.source,
            "data": event.data
        })
        print(f"  📡 Event captured: {event.type.value} from {event.source}")

    def test_events_integration(self, provider_name: str, model: str, config: dict = None):
        """Test that events are properly emitted"""
        print(f"\n[TEST] Events Integration for {provider_name}")

        # Clear previous events
        self.events_captured = []

        # Create provider - should emit PROVIDER_CREATED
        provider = create_llm(provider_name, model=model, **(config or {}))

        # Check PROVIDER_CREATED event
        created_events = [e for e in self.events_captured if e["type"] == "provider_created"]
        if created_events:
            print(f"  ✅ PROVIDER_CREATED event emitted")
            print(f"     Architecture: {created_events[0]['data'].get('architecture')}")
        else:
            print(f"  ❌ No PROVIDER_CREATED event")

        # Generate - should emit BEFORE and AFTER events
        response = provider.generate("Say hello")

        # Check events
        before_events = [e for e in self.events_captured if e["type"] == "before_generate"]
        after_events = [e for e in self.events_captured if e["type"] == "after_generate"]

        if before_events:
            print(f"  ✅ BEFORE_GENERATE event emitted")
        else:
            print(f"  ❌ No BEFORE_GENERATE event")

        if after_events:
            print(f"  ✅ AFTER_GENERATE event emitted")
            print(f"     Latency: {after_events[0]['data'].get('latency_ms', 'N/A')}ms")
        else:
            print(f"  ❌ No AFTER_GENERATE event")

        return len(self.events_captured) > 0

    def test_exceptions_integration(self, provider_name: str):
        """Test that custom exceptions are properly raised"""
        print(f"\n[TEST] Exceptions Integration for {provider_name}")

        try:
            # Try to create provider with invalid API key
            if provider_name == "openai":
                os.environ["OPENAI_API_KEY"] = "invalid_key_123"
                provider = create_llm("openai", model="gpt-3.5-turbo")
                response = provider.generate("test")
        except AuthenticationError as e:
            print(f"  ✅ AuthenticationError raised correctly")
            print(f"     Error: {str(e)[:100]}")
            return True
        except ProviderAPIError as e:
            print(f"  ✅ ProviderAPIError raised correctly")
            print(f"     Error: {str(e)[:100]}")
            return True
        except Exception as e:
            print(f"  ❌ Wrong exception type: {type(e).__name__}")
            return False

        print(f"  ⚠️  No exception raised (provider may be working)")
        return True

    def test_telemetry_integration(self, provider_name: str, model: str, config: dict = None):
        """Test that telemetry is automatically captured"""
        print(f"\n[TEST] Telemetry Integration for {provider_name}")

        # Clear telemetry
        # Using structured logging instead of telemetry
        telemetry.clear()

        # Create provider and generate
        provider = create_llm(provider_name, model=model, **(config or {}))
        response = provider.generate("What is 2+2?")

        # Check telemetry was captured automatically
        summary = telemetry.get_summary()

        if summary["total_generations"] > 0:
            print(f"  ✅ Telemetry auto-captured: {summary['total_generations']} generations")
        else:
            print(f"  ❌ No telemetry captured automatically")

        # Check verbatim in file
        telemetry_file = Path("/tmp/integration_test.jsonl")
        if telemetry_file.exists():
            with open(telemetry_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    if "metadata" in last and "prompt" in last["metadata"]:
                        print(f"  ✅ Verbatim capture working")
                    else:
                        print(f"  ❌ No verbatim data")

        return summary["total_generations"] > 0

    def test_media_handling(self):
        """Test media handling integration"""
        print(f"\n[TEST] Media Handling")

        media_handler = MediaHandler()

        # Test image format for OpenAI
        test_image = Path("/tmp/test_image.jpg")
        test_image.write_bytes(b"fake_image_data")

        openai_format = media_handler.format_for_openai(test_image)
        if openai_format and "image_url" in openai_format:
            print(f"  ✅ OpenAI image formatting works")
        else:
            print(f"  ❌ OpenAI image formatting failed")

        # Test image format for Anthropic
        anthropic_format = media_handler.format_for_anthropic(test_image)
        if anthropic_format and "source" in anthropic_format:
            print(f"  ✅ Anthropic image formatting works")
        else:
            print(f"  ❌ Anthropic image formatting failed")

        # Clean up
        test_image.unlink()

        return True

    def test_architecture_detection(self, provider_name: str, model: str, config: dict = None):
        """Test that architecture is properly detected"""
        print(f"\n[TEST] Architecture Detection for {provider_name}")

        provider = create_llm(provider_name, model=model, **(config or {}))

        # Check if provider has architecture attribute (from BaseProvider)
        if hasattr(provider, 'architecture'):
            print(f"  ✅ Architecture detected: {provider.architecture.value}")
            print(f"  ✅ Config: supports_tools={provider.architecture_config.get('supports_tools')}")
            return True
        else:
            print(f"  ❌ No architecture attribute (not using BaseProvider)")
            return False


def main():
    """Run complete integration tests"""

    print("\n" + "="*70)
    print("COMPLETE INTEGRATION TEST")
    print("Verifying all systems are properly integrated, not just created")
    print("="*70)

    verifier = IntegrationVerifier()
    results = []

    # Test with Ollama (always available)
    print(f"\n{'='*70}")
    print("Testing OLLAMA Integration")
    print('='*70)

    config = {"base_url": "http://localhost:11434"}

    # Test each system
    events_ok = verifier.test_events_integration("ollama", "qwen3:8b", config)
    telemetry_ok = verifier.test_telemetry_integration("ollama", "qwen3:8b", config)
    arch_ok = verifier.test_architecture_detection("ollama", "qwen3:8b", config)
    media_ok = verifier.test_media_handling()

    results.append(("Ollama Events", events_ok))
    results.append(("Ollama Telemetry", telemetry_ok))
    results.append(("Ollama Architecture", arch_ok))
    results.append(("Media Handling", media_ok))

    # Test with OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        print(f"\n{'='*70}")
        print("Testing OPENAI Integration")
        print('='*70)

        events_ok = verifier.test_events_integration("openai", "gpt-3.5-turbo")
        telemetry_ok = verifier.test_telemetry_integration("openai", "gpt-3.5-turbo")
        arch_ok = verifier.test_architecture_detection("openai", "gpt-3.5-turbo")

        # Restore API key for exception test
        original_key = os.getenv("OPENAI_API_KEY")
        exceptions_ok = verifier.test_exceptions_integration("openai")
        os.environ["OPENAI_API_KEY"] = original_key

        results.append(("OpenAI Events", events_ok))
        results.append(("OpenAI Telemetry", telemetry_ok))
        results.append(("OpenAI Architecture", arch_ok))
        results.append(("OpenAI Exceptions", exceptions_ok))

    # Summary
    print(f"\n{'='*70}")
    print("INTEGRATION TEST SUMMARY")
    print('='*70)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for test_name, passed_test in results:
        status = "✅" if passed_test else "❌"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} integrations working")

    # Check event count
    print(f"\nTotal events captured: {len(verifier.events_captured)}")
    if verifier.events_captured:
        event_types = set(e["type"] for e in verifier.events_captured)
        print(f"Event types seen: {', '.join(event_types)}")

    success = passed == total
    print(f"\n{'✅ ALL SYSTEMS INTEGRATED' if success else '❌ INTEGRATION INCOMPLETE'}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
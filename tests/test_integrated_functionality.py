"""
Integrated functionality test - test all components working together.
"""

import pytest
import os
import json
import time
from pathlib import Path
import tempfile
from abstractllm import create_llm, BasicSession
from abstractllm.tools.common_tools import COMMON_TOOLS, execute_tool
from abstractllm.utils.telemetry import Telemetry, setup_telemetry
from abstractllm.events import EventType, EventEmitter
from abstractllm.architectures import detect_architecture


class TestIntegratedFunctionality:
    """Test all components working together with real providers."""

    @pytest.fixture
    def temp_telemetry_file(self):
        """Create a temporary file for telemetry output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)

    def test_ollama_integrated_functionality(self, temp_telemetry_file):
        """Test Ollama with all components integrated."""
        try:
            # Setup telemetry
            setup_telemetry(enabled=True, verbatim=True, output_path=temp_telemetry_file)
            telemetry = Telemetry(enabled=True, verbatim=True, output_path=temp_telemetry_file)

            # Create provider
            provider = create_llm("ollama", model="qwen3:4b", base_url="http://localhost:11434")

            # Test 1: Basic generation
            prompt = "Who are you? Answer in one sentence."
            start = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30

            # Manually track telemetry for test
            telemetry.track_generation(
                provider="ollama",
                model="qwen3:4b",
                prompt=prompt,
                response=response.content,
                tokens=response.usage,
                latency_ms=elapsed * 1000,
                success=True
            )

            # Test 2: Session memory
            session = BasicSession(provider=provider)

            resp1 = session.generate("My name is Laurent. Who are you?")
            assert resp1 is not None
            assert resp1.content is not None

            resp2 = session.generate("What is my name?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained
            context_maintained = "laurent" in resp2.content.lower()
            assert context_maintained, "Session should remember the name Laurent"

            # Test 3: Architecture detection
            arch = detect_architecture("qwen3:4b")
            assert arch is not None

            # Test 4: Telemetry verification
            summary = telemetry.get_summary()
            assert summary["total_events"] > 0

            # Test verbatim capture if file exists
            telemetry_file = Path(temp_telemetry_file)
            if telemetry_file.exists() and telemetry_file.stat().st_size > 0:
                with open(telemetry_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        assert "metadata" in last
                        assert "prompt" in last["metadata"]

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_openai_integrated_functionality(self, temp_telemetry_file):
        """Test OpenAI with all components integrated."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            # Setup telemetry
            setup_telemetry(enabled=True, verbatim=True, output_path=temp_telemetry_file)
            telemetry = Telemetry(enabled=True, verbatim=True, output_path=temp_telemetry_file)

            # Create provider
            provider = create_llm("openai", model="gpt-4o-mini")

            # Test 1: Basic generation
            prompt = "Who are you? Answer in one sentence."
            start = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10

            # Test 2: Session memory
            session = BasicSession(provider=provider)

            resp1 = session.generate("My name is Laurent. Who are you?")
            assert resp1 is not None
            assert resp1.content is not None

            resp2 = session.generate("What is my name?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained
            context_maintained = "laurent" in resp2.content.lower()
            assert context_maintained, "Session should remember the name Laurent"

            # Test 3: Tool calling (OpenAI supports tools)
            list_files_tool = next((t for t in COMMON_TOOLS if t["name"] == "list_files"), None)
            if list_files_tool:
                tool_response = provider.generate("List the files in the current directory",
                                                 tools=[list_files_tool])

                assert tool_response is not None

                if tool_response.has_tool_calls():
                    # Tool calling worked
                    assert len(tool_response.tool_calls) > 0
                    for call in tool_response.tool_calls:
                        assert call.get('name') is not None
                        args = call.get('arguments', {})
                        if isinstance(args, str):
                            args = json.loads(args)

                        # Test tool execution
                        result = execute_tool(call.get('name'), args)
                        assert len(result) > 0

            # Test 4: Architecture detection
            arch = detect_architecture("gpt-4o-mini")
            assert arch is not None

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_integrated_functionality(self, temp_telemetry_file):
        """Test Anthropic with all components integrated."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            # Setup telemetry
            setup_telemetry(enabled=True, verbatim=True, output_path=temp_telemetry_file)
            telemetry = Telemetry(enabled=True, verbatim=True, output_path=temp_telemetry_file)

            # Create provider
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            # Test 1: Basic generation
            prompt = "Who are you? Answer in one sentence."
            start = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 10

            # Test 2: Session memory
            session = BasicSession(provider=provider)

            resp1 = session.generate("My name is Laurent. Who are you?")
            assert resp1 is not None
            assert resp1.content is not None

            resp2 = session.generate("What is my name?")
            assert resp2 is not None
            assert resp2.content is not None

            # Check if context is maintained
            context_maintained = "laurent" in resp2.content.lower()
            assert context_maintained, "Session should remember the name Laurent"

            # Test 3: Tool calling (Anthropic supports tools)
            list_files_tool = next((t for t in COMMON_TOOLS if t["name"] == "list_files"), None)
            if list_files_tool:
                tool_response = provider.generate("List the files in the current directory",
                                                 tools=[list_files_tool])

                assert tool_response is not None

                if tool_response.has_tool_calls():
                    # Tool calling worked
                    assert len(tool_response.tool_calls) > 0
                    for call in tool_response.tool_calls:
                        assert call.get('name') is not None
                        args = call.get('arguments', {})
                        if isinstance(args, str):
                            args = json.loads(args)

                        # Test tool execution
                        result = execute_tool(call.get('name'), args)
                        assert len(result) > 0

            # Test 4: Architecture detection
            arch = detect_architecture("claude-3-5-haiku-20241022")
            assert arch is not None

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_telemetry_functionality(self, temp_telemetry_file):
        """Test telemetry functionality independently."""
        # Setup telemetry
        telemetry = Telemetry(enabled=True, verbatim=True, output_path=temp_telemetry_file)

        # Test tracking generation
        telemetry.track_generation(
            provider="test",
            model="test-model",
            prompt="test prompt",
            response="test response",
            tokens={"total_tokens": 10},
            latency_ms=100,
            success=True
        )

        # Test tracking tool call
        telemetry.track_tool_call(
            tool_name="test_tool",
            arguments={"arg": "value"},
            result="test result",
            success=True
        )

        # Test summary
        summary = telemetry.get_summary()
        assert summary["total_events"] >= 2
        assert summary["total_generations"] >= 1
        assert summary["total_tool_calls"] >= 1

        # Test verbatim capture
        telemetry_file = Path(temp_telemetry_file)
        if telemetry_file.exists() and telemetry_file.stat().st_size > 0:
            with open(telemetry_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 2

                # Check generation event
                gen_event = json.loads(lines[0])
                assert gen_event["event_type"] == "generation"
                assert "metadata" in gen_event
                assert gen_event["metadata"]["prompt"] == "test prompt"

    def test_architecture_detection(self):
        """Test architecture detection for various models."""
        test_cases = [
            ("gpt-4o", "gpt"),
            ("claude-3-5-haiku-20241022", "claude"),
            ("qwen3:4b", "qwen"),
            ("llama3.1:8b", "llama"),
            ("mlx-community/Qwen3-4B-4bit", "qwen")
        ]

        for model, expected_arch in test_cases:
            arch = detect_architecture(model)
            assert arch is not None
            assert expected_arch.lower() in str(arch).lower()

    def test_event_system(self):
        """Test the event system functionality."""
        emitter = EventEmitter()

        # Test event emission and listening
        events_received = []

        def event_handler(event):
            events_received.append(event)

        emitter.on(EventType.PROVIDER_CREATED, event_handler)

        # Emit an event
        test_data = {"provider": "test", "model": "test-model"}
        emitter.emit(EventType.PROVIDER_CREATED, test_data)

        # Check event was received
        assert len(events_received) == 1
        event = events_received[0]
        assert event.type == EventType.PROVIDER_CREATED
        assert event.data == test_data


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
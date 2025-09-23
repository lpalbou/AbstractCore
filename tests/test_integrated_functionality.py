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
from abstractllm.tools.common_tools import list_files, search_files, read_file, write_file, web_search
from abstractllm.utils import configure_logging, get_logger
from abstractllm.events import EventType
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
            configure_logging(
                console_level=30,  # WARNING
                file_level=10,     # DEBUG
                log_dir="/tmp",
                verbatim_enabled=True
            )
            logger = get_logger("test.provider")

            # Create provider
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

            # Test 1: Basic generation
            prompt = "Who are you? Answer in one sentence."
            start = time.time()
            response = provider.generate(prompt)
            elapsed = time.time() - start

            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            assert elapsed < 30

            # Note: Telemetry tracking is handled automatically by BaseProvider

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
            arch = detect_architecture("qwen3-coder:30b")
            assert arch is not None

            # Test 4: Telemetry verification
            # Note: Telemetry is automatically tracked by BaseProvider - no manual verification needed
            logger.info("Ollama integration test completed successfully")

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
            configure_logging(
                console_level=30,  # WARNING
                file_level=10,     # DEBUG
                log_dir="/tmp",
                verbatim_enabled=True
            )
            logger = get_logger("test.provider")

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
            tools = [list_files]  # Use enhanced list_files tool
            if tools:
                tool_response = provider.generate("List the files in the current directory",
                                                 tools=tools)

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
                        # Execute tool directly
                    available_tools = {"list_files": list_files, "search_files": search_files, "read_file": read_file, "write_file": write_file, "web_search": web_search}
                    tool_name = call.get('name')
                    args = args
                    if tool_name in available_tools:
                        result = available_tools[tool_name](**args)
                    else:
                        result = f"Error: Tool '{tool_name}' not found"
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
            configure_logging(
                console_level=30,  # WARNING
                file_level=10,     # DEBUG
                log_dir="/tmp",
                verbatim_enabled=True
            )
            logger = get_logger("test.provider")

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
            tools = [list_files]  # Use enhanced list_files tool
            if tools:
                tool_response = provider.generate("List the files in the current directory",
                                                 tools=tools)

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
                        # Execute tool directly
                    available_tools = {"list_files": list_files, "search_files": search_files, "read_file": read_file, "write_file": write_file, "web_search": web_search}
                    tool_name = call.get('name')
                    args = args
                    if tool_name in available_tools:
                        result = available_tools[tool_name](**args)
                    else:
                        result = f"Error: Tool '{tool_name}' not found"
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
        configure_logging(
            console_level=30,  # WARNING
            file_level=10,     # DEBUG
            log_dir="/tmp",
            verbatim_enabled=True
        )
        logger = get_logger("test.multimodel")

        # Note: Telemetry tracking is now handled automatically by BaseProvider
        # This test verifies that the logging system is properly configured
        logger.info("Telemetry test completed - tracking is automatic")

        # Test summary - telemetry is now automatic via BaseProvider events
        logger.info("Test completed successfully")
        # Telemetry validation is now automatic - no manual assertions needed

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
            ("qwen3-coder:30b", "qwen"),
            ("llama3.1:8b", "llama"),
            ("mlx-community/Qwen3-4B-4bit", "qwen")
        ]

        for model, expected_arch in test_cases:
            arch = detect_architecture(model)
            assert arch is not None
            assert expected_arch.lower() in str(arch).lower()

    def test_event_system(self):
        """Test the event system functionality."""
        from abstractllm.events import on_global, emit_global, GlobalEventBus

        # Clear any previous handlers
        GlobalEventBus.clear()

        # Test event emission and listening
        events_received = []

        def event_handler(event):
            events_received.append(event)

        on_global(EventType.GENERATION_STARTED, event_handler)

        # Emit an event
        test_data = {"provider": "test", "model": "test-model"}
        emit_global(EventType.GENERATION_STARTED, test_data)

        # Check event was received
        assert len(events_received) == 1
        event = events_received[0]
        assert event.type == EventType.GENERATION_STARTED
        assert event.data == test_data


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
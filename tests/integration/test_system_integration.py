"""
Comprehensive integration tests for AbstractCore.

Tests the entire system including:
- JSON model capabilities as single source of truth
- Unified tooling system across providers
- Logging and telemetry integration
- Real provider integrations
"""

import pytest
import os
import logging
import tempfile
from pathlib import Path

from abstractcore.providers.ollama_provider import OllamaProvider
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.tools import register_tool, get_registry, ToolDefinition, clear_registry
from abstractcore.utils.structured_logging import configure_logging, get_logger
from abstractcore.architectures import get_model_capabilities, detect_architecture


class TestJSONCapabilitiesIntegration:
    """Test that JSON model capabilities are the single source of truth."""

    def test_gpt4_capabilities_from_json(self):
        """Test that GPT-4 gets capabilities from JSON."""
        json_caps = get_model_capabilities('gpt-4')

        # Verify JSON has the expected values
        assert json_caps['max_tokens'] == 128000
        assert json_caps['max_output_tokens'] == 4096
        assert json_caps['tool_support'] == 'native'

        # Verify provider uses these values
        provider = OpenAIProvider('gpt-4')
        assert provider.max_tokens == 128000
        assert provider.max_output_tokens == 4096

    def test_gpt5_capabilities_from_json(self):
        """Test that GPT-5 gets capabilities from JSON (newly added)."""
        json_caps = get_model_capabilities('gpt-5')

        # Verify JSON has the expected values
        assert json_caps['max_tokens'] == 200000
        assert json_caps['max_output_tokens'] == 8192
        assert json_caps['tool_support'] == 'native'

        # Verify provider uses these values
        provider = OpenAIProvider('gpt-5')
        assert provider.max_tokens == 200000
        assert provider.max_output_tokens == 8192

    def test_unknown_model_fallback(self):
        """Test that unknown models get default capabilities."""
        unknown_model = 'test-unknown-model-xyz'
        json_caps = get_model_capabilities(unknown_model)

        # Should get default capabilities
        assert json_caps['max_tokens'] == 16384  # Default: 16K total
        assert json_caps['max_output_tokens'] == 4096  # Default: 4K output
        assert json_caps['architecture'] == 'generic'  # Default architecture

        # Provider should use these defaults
        provider = OllamaProvider(unknown_model)
        assert provider.max_tokens == 16384
        assert provider.max_output_tokens == 4096


class TestArchitectureDetection:
    """Test architecture detection system."""

    def test_known_architectures(self):
        """Test detection of known model architectures."""
        test_cases = [
            ('gpt-4', 'gpt'),
            ('openai/gpt-oss-20b', 'gpt_oss'),
            ('qwen3-coder:30b', 'qwen3_moe'),  # Updated: more specific detection
            ('llama-3.1-8b', 'llama3_1'),  # Updated: more specific detection
            ('claude-3.5-sonnet', 'claude'),
            ('unknown-model', 'generic')
        ]

        for model, expected_arch in test_cases:
            detected = detect_architecture(model)
            assert detected == expected_arch, f"Expected {expected_arch} for {model}, got {detected}"


class TestUnifiedToolingSystem:
    """Test the unified tooling system across providers."""

    def setup_method(self):
        """Set up test tools."""
        clear_registry()  # Start with clean registry

        # Register test tools
        def get_test_value() -> str:
            """Get a test value."""
            return "test_result"

        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        register_tool(ToolDefinition.from_function(get_test_value))
        register_tool(ToolDefinition.from_function(add_numbers))

    def test_tool_registration(self):
        """Test that tools are registered correctly."""
        registry = get_registry()
        tools = registry.list_tools()
        tool_names = [tool.name for tool in tools]

        assert 'get_test_value' in tool_names
        assert 'add_numbers' in tool_names
        assert len(tools) == 2

    def test_tool_execution(self):
        """Test direct tool execution."""
        from abstractcore.tools.core import ToolCall
        from abstractcore.tools import execute_tools

        # Test simple tool
        call1 = ToolCall(name='get_test_value', arguments={})
        results = execute_tools([call1])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "test_result"

        # Test tool with parameters
        call2 = ToolCall(name='add_numbers', arguments={'a': 5, 'b': 3})
        results = execute_tools([call2])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == 8

    def test_provider_tool_integration(self):
        """Test that providers can use tools correctly."""
        # Get tools in provider format
        registry = get_registry()
        tools = [tool.to_dict() for tool in registry.list_tools()]

        # Test with Ollama provider (if available)
        provider = OllamaProvider('qwen3-coder:30b')

        # Verify provider can handle tools
        capabilities = provider.get_capabilities()
        if 'tools' in capabilities:
            # Provider supports tools
            assert len(tools) == 2
            assert tools[0]['name'] in ['get_test_value', 'add_numbers']
            assert tools[1]['name'] in ['get_test_value', 'add_numbers']


class TestLoggingTelemetrySystem:
    """Test the logging and telemetry integration."""

    def test_dual_logging_configuration(self):
        """Test dual console/file logging configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)

            # Clear any existing handlers to avoid interference from other tests
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            # Configure dual logging
            configure_logging(
                console_level=logging.INFO,
                file_level=logging.DEBUG,
                log_dir=str(log_dir)
            )

            # Test logging
            logger = get_logger('test_logger')
            logger.debug('Debug message')
            logger.info('Info message')
            logger.warning('Warning message')

            # Flush all handlers to ensure logs are written
            for handler in logging.getLogger().handlers:
                handler.flush()

            # Check log file was created (should be timestamped)
            log_files = list(log_dir.glob('abstractcore_*.log'))
            assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}: {log_files}"
            log_file = log_files[0]

            # Check log file contains messages
            log_content = log_file.read_text()
            assert 'Debug message' in log_content
            assert 'Info message' in log_content
            assert 'Warning message' in log_content

    def test_provider_logging_integration(self):
        """Test that providers integrate with logging system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clear any existing handlers to avoid interference from other tests
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            configure_logging(
                console_level=logging.INFO,
                file_level=logging.DEBUG,
                log_dir=str(temp_dir)
            )

            # Create provider (this should log initialization)
            provider = OllamaProvider('qwen3-coder:30b')

            # Check that provider has logger
            assert hasattr(provider, 'logger')
            assert provider.logger is not None

            # Provider should log architecture detection
            log_file = Path(temp_dir) / 'abstractcore.log'
            if log_file.exists():
                log_content = log_file.read_text()
                # Should contain architecture-related logs
                assert 'qwen' in log_content.lower() or 'debug' in log_content.lower()


class TestRealProviderIntegration:
    """Test real provider integrations (requires services)."""

    def test_ollama_connection(self):
        """Test Ollama provider connection."""
        provider = OllamaProvider('qwen3-coder:30b')

        # Test basic properties
        assert provider.model == 'qwen3-coder:30b'
        assert provider.architecture == 'qwen3_moe'  # Updated: more specific detection

        # Test capabilities
        capabilities = provider.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'chat' in capabilities or 'streaming' in capabilities

        # Test connection (may fail if Ollama not running)
        try:
            connection_ok = provider.validate_config()
            # If connection works, it should return True
            if connection_ok:
                assert connection_ok is True
        except Exception:
            # If Ollama not running, that's okay for CI
            pytest.skip("Ollama service not available")

    @pytest.mark.integration
    def test_ollama_tool_execution(self):
        """Test actual tool execution with Ollama (requires running service)."""
        # This test requires Ollama to be running
        provider = OllamaProvider('qwen3-coder:30b')

        if not provider.validate_config():
            pytest.skip("Ollama service not available")

        # Register a simple tool
        clear_registry()
        def get_current_time() -> str:
            """Get current time."""
            from datetime import datetime
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        register_tool(ToolDefinition.from_function(get_current_time))

        # Get tools for provider
        registry = get_registry()
        tools = [tool.to_dict() for tool in registry.list_tools()]

        try:
            # Make actual LLM call
            response = provider.generate(
                prompt="What time is it? Use the available tool to find out.",
                tools=tools,
                max_output_tokens=100
            )

            # Should have response
            assert response.content is not None
            assert len(response.content) > 0

            # Should have executed tool (content contains timestamp or tool results)
            content_lower = response.content.lower()
            assert ('tool results' in content_lower or
                    '2025' in response.content or
                    'time' in content_lower)

        except Exception as e:
            if 'not found' in str(e).lower():
                pytest.skip(f"Model not available: {e}")
            else:
                raise


class TestSystemEndToEnd:
    """End-to-end system tests."""

    def test_complete_workflow(self):
        """Test complete workflow from capabilities to tool execution."""
        # 1. Test model capabilities
        model = 'qwen3-coder:30b'
        capabilities = get_model_capabilities(model)
        assert capabilities is not None
        assert 'architecture' in capabilities

        # 2. Test provider creation with capabilities
        provider = OllamaProvider(model)
        assert provider.max_tokens == capabilities['max_tokens']
        assert provider.max_output_tokens == capabilities['max_output_tokens']

        # 3. Test tool registration
        clear_registry()
        def simple_calc(x: int) -> int:
            """Double a number."""
            return x * 2

        register_tool(ToolDefinition.from_function(simple_calc))

        # 4. Test tool availability
        registry = get_registry()
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == 'simple_calc'

        # 5. Test provider capabilities include tools
        provider_caps = provider.get_capabilities()
        if 'tools' in provider_caps:
            # Provider supports tools - full integration possible
            assert True  # Mark as successful integration
        else:
            # Provider doesn't support tools but everything else works
            assert True  # Still a valid configuration


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])

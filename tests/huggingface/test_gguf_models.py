"""
Test HuggingFace GGUF model support with unified token parameters
"""
import pytest
import gc
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from contextlib import redirect_stderr
from io import StringIO

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abstractcore import create_llm
from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.exceptions import ModelNotFoundError

# GGUF tests can exercise llama.cpp/metal backends and are not reliable in all CI/sandbox environments.
if os.getenv("ABSTRACTCORE_RUN_GGUF_TESTS") != "1":
    pytest.skip(
        "GGUF tests are opt-in; set ABSTRACTCORE_RUN_GGUF_TESTS=1 to run",
        allow_module_level=True,
    )


# Test model - using a known GGUF model in HF cache
TEST_GGUF_MODEL = "unsloth--Qwen3-4B-Instruct-2507-GGUF"
TEST_GGUF_ALT_FORMAT = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


class TestGGUFBasicFunctionality:
    """Test basic GGUF model functionality"""

    def test_gguf_model_detection(self):
        """Test GGUF model format detection - NO MOCKING"""
        # Test the _is_gguf_model method directly on the class
        # This method is static and doesn't require model loading

        # Test various GGUF formats
        assert HuggingFaceProvider._is_gguf_model(None, "model.gguf") == True
        assert HuggingFaceProvider._is_gguf_model(None, "/path/to/model.gguf") == True
        assert HuggingFaceProvider._is_gguf_model(None, "unsloth/Model-GGUF") == True
        assert HuggingFaceProvider._is_gguf_model(None, "unsloth--Model-GGUF") == True
        assert HuggingFaceProvider._is_gguf_model(None, "regular-model") == False

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_gguf_model_loading(self):
        """Test GGUF model loads successfully"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=4096,
                        max_output_tokens=100,
                        debug=False)

        assert llm.model_type == "gguf"
        assert llm.max_tokens == 4096
        assert llm.max_output_tokens == 100
        assert llm.debug == False
        assert llm.llm is not None

        # Cleanup
        del llm
        gc.collect()

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_gguf_capabilities(self):
        """Test GGUF model reports correct capabilities"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        debug=False)

        capabilities = llm.get_capabilities()
        assert "chat" in capabilities
        assert "streaming" in capabilities
        assert "gguf" in capabilities
        assert "tools" in capabilities

        # Cleanup
        del llm
        gc.collect()

    def test_gguf_not_found_error(self):
        """Test graceful handling when GGUF model not found"""
        with pytest.raises(RuntimeError) as exc_info:
            create_llm("huggingface",
                      model="nonexistent/GGUF-Model",
                      debug=False)

        error_message = str(exc_info.value)
        assert "GGUF model" in error_message
        assert "not found in HuggingFace cache" in error_message


class TestGGUFUnifiedTokens:
    """Test unified token parameter system with GGUF models"""

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_unified_token_parameters(self):
        """Test unified token parameters work correctly"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=8192,
                        max_output_tokens=512,
                        debug=False)

        assert llm.max_tokens == 8192
        assert llm.max_output_tokens == 512
        assert llm.max_input_tokens is None  # Will be calculated

        # Test effective limits calculation
        max_tokens, max_output, max_input = llm._calculate_effective_token_limits()
        assert max_tokens == 8192
        assert max_output == 512
        assert max_input == 8192 - 512  # Calculated automatically

        # Cleanup
        del llm
        gc.collect()

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_token_validation(self):
        """Test token validation works correctly"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=2048,
                        max_output_tokens=512,
                        debug=False)

        # Valid usage should pass
        assert llm.validate_token_usage(input_tokens=100, requested_output_tokens=200) == True

        # Exceeding total tokens should fail
        with pytest.raises(ValueError) as exc_info:
            llm.validate_token_usage(input_tokens=1800, requested_output_tokens=400)

        error_message = str(exc_info.value)
        # Could fail on either input tokens or total tokens
        assert ("exceed max_input_tokens" in error_message or "would exceed max_tokens" in error_message)

        # Cleanup
        del llm
        gc.collect()

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_legacy_context_size_parameter(self):
        """Test that legacy context_size parameter still works - NO MOCKING"""
        # Test with actual GGUF model that legacy context_size maps to max_tokens
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        context_size=3072,  # Legacy parameter
                        max_output_tokens=512,
                        debug=False)

        # Should have mapped context_size to max_tokens
        assert llm.max_tokens == 3072
        assert llm.max_output_tokens == 512

        # Cleanup
        del llm
        gc.collect()


class TestGGUFDebugMode:
    """Test debug mode and stderr logging functionality"""

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_debug_false_suppresses_warnings(self):
        """Test that debug=False suppresses warnings"""
        # Capture stderr
        stderr_capture = StringIO()

        with redirect_stderr(stderr_capture):
            llm = create_llm("huggingface",
                            model=TEST_GGUF_MODEL,
                            max_tokens=2048,
                            debug=False)
            del llm
            gc.collect()

        stderr_output = stderr_capture.getvalue()

        # Should have minimal stderr output when debug=False
        # The context warning might still appear as it's from llama-cpp internals
        # but ggml_metal_init warnings should be redirected to our logger
        assert "ggml_metal_init: skipping" not in stderr_output

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_debug_true_shows_warnings(self):
        """Test that debug=True shows detailed information"""
        # We can't easily test this without complex logging setup,
        # but we can verify the debug flag is set correctly
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=2048,
                        debug=True)

        assert llm.debug == True

        # Cleanup
        del llm
        gc.collect()


class TestGGUFGeneration:
    """Test GGUF model generation capabilities"""

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_basic_generation(self):
        """Test basic text generation"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=2048,
                        max_output_tokens=50,
                        debug=False)

        response = llm.generate("What is 2+2?")

        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == TEST_GGUF_MODEL
        assert response.finish_reason in ["stop", "length"]

        # Check usage information is provided
        if response.usage:
            assert "prompt_tokens" in response.usage
            assert "completion_tokens" in response.usage
            assert "total_tokens" in response.usage

        # Cleanup
        del llm
        gc.collect()

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_streaming_generation(self):
        """Test streaming text generation"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=2048,
                        max_output_tokens=30,
                        debug=False)

        stream = llm.generate("Count from 1 to 3", stream=True)

        chunks = []
        for chunk in stream:
            chunks.append(chunk)
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'model')
            assert chunk.model == TEST_GGUF_MODEL

        # Should have received multiple chunks
        assert len(chunks) > 1

        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason is not None

        # Cleanup
        del llm
        gc.collect()

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_generation_with_token_limits(self):
        """Test generation respects token limits"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=2048,
                        max_output_tokens=20,  # Very small limit
                        debug=False)

        response = llm.generate("Write a long story about space exploration")

        # Response should be limited
        assert response.content is not None
        # Should be relatively short due to token limit
        assert len(response.content.split()) < 50  # Conservative check

        # Cleanup
        del llm
        gc.collect()


class TestGGUFToolCalling:
    """Test GGUF model tool calling capabilities - NO MOCKING"""

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_tool_calling_non_streaming(self):
        """Test tool calling with GGUF models (non-streaming) - REAL IMPLEMENTATION"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=4096,
                        max_output_tokens=200,
                        debug=False)

        # Define a simple tool
        tools = [{
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }]

        # Test tool calling
        response = llm.generate(
            "What's the weather like in Paris?",
            tools=tools,
            max_output_tokens=150
        )

        assert response is not None
        assert response.content is not None or response.tool_calls is not None

        # If tool calls were made, verify structure
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            # Canonical key in AbstractCore is `call_id` (OpenAI-style `id` may also appear).
            assert "call_id" in tool_call or "id" in tool_call
            assert "name" in tool_call
            assert "arguments" in tool_call
            # Should be the weather tool
            assert tool_call["name"] == "get_weather"

        # Cleanup
        del llm
        gc.collect()

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_tool_calling_streaming(self):
        """Test tool calling with GGUF models (streaming) - REAL IMPLEMENTATION"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=4096,
                        max_output_tokens=200,
                        debug=False)

        # Define a simple tool
        tools = [{
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }]

        # Test streaming tool calling
        stream = llm.generate(
            "Calculate 25 * 4",
            tools=tools,
            stream=True,
            max_output_tokens=100
        )

        chunks = []
        tool_calls_found = False

        for chunk in stream:
            chunks.append(chunk)
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'model')
            assert chunk.model == TEST_GGUF_MODEL

            # Check for tool calls in chunks
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                tool_calls_found = True
                tool_call = chunk.tool_calls[0]
                assert "name" in tool_call
                # Could be the calculate tool
                if tool_call["name"] == "calculate":
                    assert "arguments" in tool_call

        # Should have received multiple chunks
        assert len(chunks) > 0

        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason is not None

        # Cleanup
        del llm
        gc.collect()

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_tool_calling_with_token_limits(self):
        """Test tool calling respects token limits - REAL IMPLEMENTATION"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=2048,
                        max_output_tokens=80,  # Very small limit
                        debug=False)

        tools = [{
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }]

        # Test with tight token limits
        response = llm.generate(
            "Search for information about quantum computing",
            tools=tools,
            max_output_tokens=50
        )

        assert response is not None

        # Response should be limited due to token constraints
        if response.content:
            # Should be relatively short
            assert len(response.content.split()) < 100

        # Cleanup
        del llm
        gc.collect()


class TestGGUFCleanup:
    """Test proper cleanup and resource management"""

    @pytest.mark.skipif(
        not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"),
        reason="Test GGUF model not found in cache"
    )
    def test_proper_cleanup(self):
        """Test that models clean up properly without errors"""
        llm = create_llm("huggingface",
                        model=TEST_GGUF_MODEL,
                        max_tokens=2048,
                        debug=False)

        # Generate something to ensure model is fully loaded
        response = llm.generate("Hello")
        assert response.content is not None

        # Test explicit unload method
        llm.unload_model(llm.model)

        # Explicitly delete and garbage collect
        del llm
        gc.collect()

        # If we get here without errors, cleanup worked
        assert True

    def test_multiple_model_instances(self):
        """Test that multiple GGUF model instances can be created and cleaned up"""
        # Skip if model not available
        if not os.path.exists(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{TEST_GGUF_MODEL.replace('/', '--')}"):
            pytest.skip("Test GGUF model not found in cache")

        models = []

        # Create multiple instances
        for i in range(2):
            llm = create_llm("huggingface",
                            model=TEST_GGUF_MODEL,
                            max_tokens=1024,
                            max_output_tokens=20,
                            debug=False)
            models.append(llm)

        # Use them briefly
        for llm in models:
            response = llm.generate("Hi")
            assert response.content is not None

        # Clean up all - demonstrate explicit unload in loop
        for llm in models:
            llm.unload_model(llm.model)  # Explicitly free memory
            del llm

        gc.collect()

        # If we get here, cleanup of multiple instances worked
        assert True


if __name__ == "__main__":
    pytest.main([__file__])

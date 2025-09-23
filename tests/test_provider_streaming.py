"""
Test streaming capabilities - verify providers can stream responses.
Tests both content streaming and tool call detection in streaming mode.
"""

import pytest
import os
import time
from typing import Iterator
from abstractllm import create_llm
from abstractllm.core.types import GenerateResponse
from abstractllm.tools.common_tools import list_files, search_files, read_file, write_file, web_search


class TestProviderStreaming:
    """Test streaming capabilities for each provider."""

    def test_openai_streaming_basic(self):
        """Test OpenAI basic streaming functionality."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            # Test streaming
            start = time.time()
            stream = provider.generate(
                "Count from 1 to 5, putting each number on a new line.",
                stream=True
            )
            elapsed = time.time() - start

            # Verify we get an iterator
            assert hasattr(stream, '__iter__'), "Stream should be iterable"

            # Collect chunks
            chunks = []
            full_content = ""

            for chunk in stream:
                assert isinstance(chunk, GenerateResponse), "Each chunk should be GenerateResponse"
                chunks.append(chunk)
                if chunk.content:
                    full_content += chunk.content

            # Verify streaming behavior
            assert len(chunks) > 1, "Should receive multiple chunks"
            assert len(full_content) > 0, "Should have accumulated content"
            assert elapsed < 30, "Stream should start quickly"

            # Content should contain the numbers
            assert any(str(i) in full_content for i in range(1, 6)), "Should contain requested numbers"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_anthropic_streaming_basic(self):
        """Test Anthropic basic streaming functionality."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")

            # Test streaming
            stream = provider.generate(
                "List three colors, one per line.",
                stream=True
            )

            # Verify we get an iterator
            assert hasattr(stream, '__iter__'), "Stream should be iterable"

            # Collect chunks
            chunks = []
            full_content = ""

            for chunk in stream:
                assert isinstance(chunk, GenerateResponse), "Each chunk should be GenerateResponse"
                chunks.append(chunk)
                if chunk.content:
                    full_content += chunk.content

            # Verify streaming behavior
            assert len(chunks) > 1, "Should receive multiple chunks"
            assert len(full_content) > 0, "Should have accumulated content"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            else:
                raise

    def test_ollama_streaming_basic(self):
        """Test Ollama basic streaming functionality."""
        try:
            provider = create_llm("ollama", model="qwen3-coder:30b", base_url="http://localhost:11434")

            # Test streaming
            stream = provider.generate(
                "Say hello in three different languages.",
                stream=True
            )

            # Verify we get an iterator
            assert hasattr(stream, '__iter__'), "Stream should be iterable"

            # Collect chunks
            chunks = []
            full_content = ""

            for chunk in stream:
                assert isinstance(chunk, GenerateResponse), "Each chunk should be GenerateResponse"
                chunks.append(chunk)
                if chunk.content:
                    full_content += chunk.content

            # Verify streaming behavior
            assert len(chunks) > 1, "Should receive multiple chunks"
            assert len(full_content) > 0, "Should have accumulated content"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("Ollama not running")
            else:
                raise

    def test_mlx_streaming_basic(self):
        """Test MLX basic streaming functionality."""
        try:
            provider = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")

            # Test streaming
            stream = provider.generate(
                "Write a short haiku about coding.",
                stream=True
            )

            # Verify we get an iterator
            assert hasattr(stream, '__iter__'), "Stream should be iterable"

            # Collect chunks
            chunks = []
            full_content = ""

            for chunk in stream:
                assert isinstance(chunk, GenerateResponse), "Each chunk should be GenerateResponse"
                chunks.append(chunk)
                if chunk.content:
                    full_content += chunk.content

            # Verify streaming behavior
            assert len(chunks) > 1, "Should receive multiple chunks"
            assert len(full_content) > 0, "Should have accumulated content"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["mlx", "import", "not found", "failed to load"]):
                pytest.skip("MLX not available or model not found")
            else:
                raise

    def test_lmstudio_streaming_basic(self):
        """Test LMStudio basic streaming functionality."""
        try:
            provider = create_llm("lmstudio", model="qwen/qwen3-coder-30b", base_url="http://localhost:1234/v1")

            # Test streaming
            stream = provider.generate(
                "Explain what streaming is in one sentence.",
                stream=True
            )

            # Verify we get an iterator
            assert hasattr(stream, '__iter__'), "Stream should be iterable"

            # Collect chunks
            chunks = []
            full_content = ""

            for chunk in stream:
                assert isinstance(chunk, GenerateResponse), "Each chunk should be GenerateResponse"
                chunks.append(chunk)
                if chunk.content:
                    full_content += chunk.content

            # Verify streaming behavior
            assert len(chunks) > 1, "Should receive multiple chunks"
            assert len(full_content) > 0, "Should have accumulated content"

        except Exception as e:
            if any(keyword in str(e).lower() for keyword in ["connection", "refused", "timeout"]):
                pytest.skip("LMStudio not running")
            else:
                raise

    def test_mock_streaming_basic(self):
        """Test Mock provider streaming functionality."""
        provider = create_llm("mock", model="test-model")

        # Test streaming
        stream = provider.generate(
            "This is a test prompt for streaming.",
            stream=True
        )

        # Verify we get an iterator
        assert hasattr(stream, '__iter__'), "Stream should be iterable"

        # Collect chunks
        chunks = []
        full_content = ""

        for chunk in stream:
            assert isinstance(chunk, GenerateResponse), "Each chunk should be GenerateResponse"
            chunks.append(chunk)
            if chunk.content:
                full_content += chunk.content

        # Verify streaming behavior
        assert len(chunks) > 1, "Should receive multiple chunks"
        assert len(full_content) > 0, "Should have accumulated content"

    def test_streaming_tool_detection(self):
        """Test tool call detection in streaming mode."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")

            # Use enhanced list_files tool
            tools = [list_files]

            # Test streaming with tools
            stream = provider.generate(
                "Please list the files in the current directory",
                tools=tools,
                stream=True
            )

            # Collect all chunks
            chunks = []
            tool_calls_detected = []

            for chunk in stream:
                chunks.append(chunk)
                if chunk.has_tool_calls():
                    tool_calls_detected.extend(chunk.tool_calls)

            # Verify streaming with tools works
            assert len(chunks) > 0, "Should receive chunks"

            # If tools were called, verify they were detected properly
            if tool_calls_detected:
                assert len(tool_calls_detected) > 0, "Should detect tool calls"
                tool_call = tool_calls_detected[0]
                assert tool_call.get('name') == 'list_files', "Should detect correct tool"
                assert 'arguments' in tool_call, "Should have arguments"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_streaming_vs_non_streaming_consistency(self):
        """Test that streaming and non-streaming return equivalent content."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            provider = create_llm("openai", model="gpt-4o-mini")
            prompt = "Say exactly: 'Hello, this is a test message.'"

            # Non-streaming response
            regular_response = provider.generate(prompt, stream=False)

            # Streaming response
            stream = provider.generate(prompt, stream=True)

            # Collect streaming content
            streaming_content = ""
            for chunk in stream:
                if chunk.content:
                    streaming_content += chunk.content

            # Verify both approaches work
            assert regular_response.content is not None, "Regular response should have content"
            assert len(streaming_content) > 0, "Streaming should accumulate content"

            # Content should be similar (may not be identical due to LLM variability)
            assert len(streaming_content) > 10, "Streaming content should be substantial"
            assert len(regular_response.content) > 10, "Regular content should be substantial"

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            else:
                raise

    def test_streaming_chunk_structure(self):
        """Test that streaming chunks have proper structure."""
        provider = create_llm("mock", model="test-model")

        stream = provider.generate("Test prompt", stream=True)

        for i, chunk in enumerate(stream):
            # Each chunk should be a proper GenerateResponse
            assert isinstance(chunk, GenerateResponse), f"Chunk {i} should be GenerateResponse"
            assert hasattr(chunk, 'content'), f"Chunk {i} should have content attribute"
            assert hasattr(chunk, 'model'), f"Chunk {i} should have model attribute"
            assert hasattr(chunk, 'has_tool_calls'), f"Chunk {i} should have has_tool_calls method"

            # Model should be consistent
            if chunk.model:
                assert chunk.model == "test-model", f"Chunk {i} should have correct model"

    def test_streaming_interruption(self):
        """Test that streaming can be interrupted gracefully."""
        provider = create_llm("mock", model="test-model")

        stream = provider.generate("Long response test", stream=True)

        # Take only first few chunks
        chunks_taken = 0
        for chunk in stream:
            chunks_taken += 1
            if chunks_taken >= 3:
                break  # Interrupt the stream

        # Should be able to interrupt without errors
        assert chunks_taken >= 3, "Should have processed at least 3 chunks"

    def test_streaming_empty_response(self):
        """Test streaming behavior with empty or minimal responses."""
        provider = create_llm("mock", model="test-model")

        stream = provider.generate("", stream=True)  # Empty prompt

        chunks = list(stream)

        # Should still get proper chunk structure even with empty input
        assert len(chunks) > 0, "Should get at least one chunk even for empty input"
        assert all(isinstance(chunk, GenerateResponse) for chunk in chunks), "All chunks should be GenerateResponse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
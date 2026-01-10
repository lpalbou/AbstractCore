"""
Streaming integration tests.

These tests intentionally skip when the relevant local server / API key is not available.
They aim to validate basic streaming behavior (iterator of GenerateResponse chunks).
"""

import os
import time

import pytest

from abstractcore import create_llm
from abstractcore.core.types import GenerateResponse
from abstractcore.tools.common_tools import list_files


def _looks_like_connectivity_issue(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        keyword in msg
        for keyword in [
            "connection",
            "refused",
            "timeout",
            "operation not permitted",
        ]
    )


class TestProviderStreaming:
    """Test streaming capabilities for each provider (skip-heavy)."""

    def test_ollama_streaming_basic(self):
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            provider = create_llm(
                "ollama",
                model="qwen3:4b-instruct",
                base_url="http://localhost:11434",
                timeout=10.0,
            )
            stream = provider.generate("Say hello in three different languages.", stream=True)
            assert hasattr(stream, "__iter__")
            chunks = list(stream)
            assert len(chunks) > 0
            assert all(isinstance(c, GenerateResponse) for c in chunks)
        except Exception as e:
            if _looks_like_connectivity_issue(e):
                pytest.skip("Ollama not running / not reachable")
            raise

    def test_lmstudio_streaming_basic(self):
        if os.getenv("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS") != "1":
            pytest.skip("Local provider tests disabled (set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1)")
        try:
            provider = create_llm(
                "lmstudio",
                model="qwen/qwen3-4b-2507",
                base_url="http://localhost:1234/v1",
                timeout=10.0,
            )
            stream = provider.generate("Explain what streaming is in one sentence.", stream=True)
            assert hasattr(stream, "__iter__")
            chunks = list(stream)
            assert len(chunks) > 0
            assert all(isinstance(c, GenerateResponse) for c in chunks)
        except Exception as e:
            if _looks_like_connectivity_issue(e):
                pytest.skip("LMStudio not running / not reachable")
            raise

    def test_openai_streaming_basic(self):
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        try:
            provider = create_llm("openai", model="gpt-5-mini", timeout=30.0)
            start = time.time()
            stream = provider.generate("Count from 1 to 5, each on a new line.", stream=True)
            chunks = list(stream)
            elapsed = time.time() - start

            assert len(chunks) > 0
            assert any(isinstance(c, GenerateResponse) for c in chunks)
            assert elapsed < 30
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            if _looks_like_connectivity_issue(e):
                pytest.skip("OpenAI not reachable")
            raise

    def test_anthropic_streaming_basic(self):
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        try:
            provider = create_llm("anthropic", model="claude-haiku-4-5", timeout=30.0)
            stream = provider.generate("List three colors, one per line.", stream=True)
            chunks = list(stream)
            assert len(chunks) > 0
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("Anthropic authentication failed")
            if _looks_like_connectivity_issue(e):
                pytest.skip("Anthropic not reachable")
            raise

    def test_openai_streaming_tool_detection(self):
        if os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1":
            pytest.skip("Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)")
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        try:
            provider = create_llm("openai", model="gpt-5-mini", timeout=30.0)
            stream = provider.generate(
                "Please list the files in the current directory.",
                tools=[list_files],
                stream=True,
            )

            chunks = list(stream)
            assert len(chunks) > 0

            # We don't hard-require a tool call here (model variability),
            # but if tool calls appear they must be well-formed.
            tool_calls = []
            for chunk in chunks:
                if chunk.has_tool_calls():
                    tool_calls.extend(chunk.tool_calls)

            if tool_calls:
                first = tool_calls[0]
                assert first.get("name") == "list_files"
                assert "arguments" in first
        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                pytest.skip("OpenAI authentication failed")
            if _looks_like_connectivity_issue(e):
                pytest.skip("OpenAI not reachable")
            raise

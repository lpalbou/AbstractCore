"""
Test async session functionality.
"""
import pytest
from abstractcore import create_llm
from abstractcore.core.session import BasicSession


class TestAsyncSession:
    """Test async session operations."""

    @pytest.mark.asyncio
    async def test_session_async_generate(self, skip_if_provider_unavailable):
        """Test session async generation."""
        skip_if_provider_unavailable("ollama", model="qwen3:4b")
        llm = create_llm("ollama", model="qwen3:4b")
        session = BasicSession(provider=llm)

        response = await session.agenerate("Hello")
        assert response is not None
        assert response.content is not None
        assert len(session.messages) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_session_async_conversation(self, skip_if_provider_unavailable):
        """Test multi-turn async conversation."""
        skip_if_provider_unavailable("ollama", model="qwen3:4b")
        llm = create_llm("ollama", model="qwen3:4b")
        session = BasicSession(provider=llm)

        response1 = await session.agenerate("My name is Alice")
        response2 = await session.agenerate("What is my name?")

        assert len(session.messages) == 4  # 2 user + 2 assistant
        assert response2.content is not None

    @pytest.mark.asyncio
    async def test_session_async_streaming(self, skip_if_provider_unavailable):
        """Test session async streaming."""
        skip_if_provider_unavailable("ollama", model="qwen3:4b")
        llm = create_llm("ollama", model="qwen3:4b")
        session = BasicSession(provider=llm)

        chunks = []
        # NOTE: agenerate() is async; when stream=True it resolves to an async iterator.
        stream_gen = await session.agenerate("Count to 3", stream=True)
        async for chunk in stream_gen:
            chunks.append(chunk)

        assert len(chunks) > 0
        assert len(session.messages) == 2  # user + assistant

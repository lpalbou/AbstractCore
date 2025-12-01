"""
Test async generation with all providers.
"""
import pytest
import asyncio
from abstractcore import create_llm


class TestAsyncProviders:
    """Test async generation for all 6 providers."""

    @pytest.mark.asyncio
    async def test_ollama_async(self, skip_if_provider_unavailable):
        """Test Ollama async generation."""
        skip_if_provider_unavailable("ollama")
        llm = create_llm("ollama", model="gemma3:1b")
        response = await llm.agenerate("Say hello")
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_lmstudio_async(self, skip_if_provider_unavailable):
        """Test LMStudio async generation."""
        skip_if_provider_unavailable("lmstudio")
        llm = create_llm("lmstudio", model="qwen/qwen3-vl-30b")
        response = await llm.agenerate("Say hello")
        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_mlx_async(self, skip_if_provider_unavailable):
        """Test MLX async generation."""
        skip_if_provider_unavailable("mlx")
        llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")
        response = await llm.agenerate("Say hello", max_output_tokens=10)
        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_huggingface_async(self):
        """Test HuggingFace async generation."""
        llm = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")
        response = await llm.agenerate("Say hello", max_output_tokens=10)
        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_openai_async(self, skip_if_provider_unavailable):
        """Test OpenAI async generation."""
        skip_if_provider_unavailable("openai")
        llm = create_llm("openai", model="gpt-4o-mini")
        response = await llm.agenerate("Say hello")
        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_anthropic_async(self, skip_if_provider_unavailable):
        """Test Anthropic async generation."""
        skip_if_provider_unavailable("anthropic")
        llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
        response = await llm.agenerate("Say hello")
        assert response is not None
        assert response.content is not None


class TestAsyncConcurrent:
    """Test concurrent async requests."""

    @pytest.mark.asyncio
    async def test_concurrent_same_provider(self, skip_if_provider_unavailable):
        """Test concurrent requests to same provider."""
        skip_if_provider_unavailable("ollama")
        llm = create_llm("ollama", model="gemma3:1b")

        # Execute 3 requests concurrently
        tasks = [
            llm.agenerate(f"Count to {i}") for i in range(1, 4)
        ]
        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        assert all(r.content for r in responses)

    @pytest.mark.asyncio
    async def test_concurrent_multi_provider(self, skip_if_provider_unavailable):
        """Test concurrent requests to multiple providers."""
        skip_if_provider_unavailable("ollama")
        skip_if_provider_unavailable("openai")

        ollama = create_llm("ollama", model="gemma3:1b")
        openai = create_llm("openai", model="gpt-4o-mini")

        # Execute concurrently across providers
        responses = await asyncio.gather(
            ollama.agenerate("Say hello"),
            openai.agenerate("Say hello")
        )

        assert len(responses) == 2
        assert all(r.content for r in responses)


class TestAsyncStreaming:
    """Test async streaming."""

    @pytest.mark.asyncio
    async def test_async_streaming_ollama(self, skip_if_provider_unavailable):
        """Test async streaming with Ollama."""
        skip_if_provider_unavailable("ollama")
        llm = create_llm("ollama", model="gemma3:1b")

        chunks = []
        async for chunk in await llm.agenerate("Count to 5", stream=True):
            chunks.append(chunk)
            assert chunk.content is not None

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_async_streaming_openai(self, skip_if_provider_unavailable):
        """Test async streaming with OpenAI."""
        skip_if_provider_unavailable("openai")
        llm = create_llm("openai", model="gpt-4o-mini")

        chunks = []
        async for chunk in await llm.agenerate("Say hello", stream=True):
            chunks.append(chunk)

        assert len(chunks) > 0

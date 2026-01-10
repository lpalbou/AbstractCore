"""
Tests for OpenAI-compatible generic provider.

Note: These tests require an OpenAI-compatible server running.
Set OPENAI_COMPATIBLE_BASE_URL environment variable to point to your server.

Compatible servers:
- llama.cpp server (--host 0.0.0.0 --port 8080)
- text-generation-webui (with OpenAI extension)
- LocalAI
- FastChat
- Aphrodite
- SGLang
- Any OpenAI-compatible endpoint

Example:
    # Explicit (recommended)
    export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:1234/v1"   # LM Studio
    # or
    export OPENAI_COMPATIBLE_BASE_URL="http://127.0.0.1:11434/v1"  # Ollama (OpenAI-compatible API)
    export OPENAI_COMPATIBLE_API_KEY="optional-key"  # If server requires auth
    pytest tests/providers/test_openai_compatible_provider.py -v
"""

import pytest
import os
import httpx
from abstractcore import create_llm
from abstractcore.providers import OpenAICompatibleProvider


def _candidate_base_urls() -> list[str]:
    env = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    if isinstance(env, str) and env.strip():
        return [env.strip().rstrip("/")]
    # Sensible defaults for local development machines.
    # Avoid probing :8080 by default (commonly used by unrelated services).
    return [
        "http://127.0.0.1:1234/v1",   # LM Studio
        "http://127.0.0.1:11434/v1",  # Ollama (OpenAI-compatible API)
    ]


def _fetch_models(base_url: str) -> dict | None:
    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key.strip() else None
    try:
        res = httpx.get(f"{base_url}/models", headers=headers, timeout=2.0)
    except Exception:
        return None
    if res.status_code != 200:
        return None
    try:
        payload = res.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _choose_test_model(model_ids: list[str]) -> str:
    preferred = [
        # LM Studio common ids/aliases
        "qwen/qwen3-next-80b",
        "qwen3-next-80b",
        # Ollama common ids/aliases (OpenAI-compatible)
        "qwen3:4b-instruct",
        "qwen3:4b-instruct-2507-q4_K_M",
    ]
    s = set(model_ids)
    for mid in preferred:
        if mid in s:
            return mid
    return model_ids[0]


def _discover_server() -> tuple[str | None, str | None]:
    for base_url in _candidate_base_urls():
        payload = _fetch_models(base_url)
        if not payload:
            continue
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            continue
        ids: list[str] = []
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id")
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
        if not ids:
            continue
        ids = sorted(set(ids))
        return base_url, _choose_test_model(ids)
    return None, None


_TEST_BASE_URL, _TEST_MODEL = _discover_server()


def server_available() -> bool:
    return bool(_TEST_BASE_URL and _TEST_MODEL)


pytestmark = pytest.mark.skipif(
    not server_available(),
    reason=(
        "OpenAI-compatible server not available. Set OPENAI_COMPATIBLE_BASE_URL "
        "(or run LM Studio on :1234 or Ollama OpenAI API on :11434)."
    ),
)


@pytest.fixture(scope="session")
def base_url() -> str:
    assert _TEST_BASE_URL is not None
    return _TEST_BASE_URL


@pytest.fixture(scope="session")
def model_name() -> str:
    assert _TEST_MODEL is not None
    return _TEST_MODEL


@pytest.fixture
def llm(base_url: str, model_name: str):
    return create_llm("openai-compatible", model=model_name, base_url=base_url)


class TestProviderBasics:
    """Test basic provider functionality."""

    def test_provider_initialization(self, base_url: str):
        """Test provider can be initialized"""
        # Use a local placeholder model to avoid server model validation in this basic test.
        provider = create_llm("openai-compatible", model="default", base_url=base_url)
        assert provider is not None
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider == "openai-compatible"

    def test_environment_variable_base_url(self):
        """Test base_url can be set via environment variable"""
        test_url = "http://custom-server:1234/v1"
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = test_url
        try:
            llm = OpenAICompatibleProvider(model="default")
            assert llm.base_url == test_url
        finally:
            # Cleanup
            if "OPENAI_COMPATIBLE_BASE_URL" in os.environ:
                del os.environ["OPENAI_COMPATIBLE_BASE_URL"]

    def test_api_key_optional(self):
        """Test API key is optional"""
        llm = OpenAICompatibleProvider(model="default", base_url="http://127.0.0.1:1234/v1")
        assert llm.api_key is None or llm.api_key == ""

    def test_api_key_environment_variable(self):
        """Test API key can be set via environment variable"""
        test_key = "sk-test-key-12345"
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = test_key
        try:
            llm = OpenAICompatibleProvider(model="default")
            assert llm.api_key == test_key
        finally:
            # Cleanup
            if "OPENAI_COMPATIBLE_API_KEY" in os.environ:
                del os.environ["OPENAI_COMPATIBLE_API_KEY"]

    def test_api_key_parameter_priority(self):
        """Test programmatic api_key parameter takes precedence"""
        test_key = "sk-programmatic-key"
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = "sk-env-key"
        try:
            llm = OpenAICompatibleProvider(model="default", api_key=test_key)
            assert llm.api_key == test_key
        finally:
            if "OPENAI_COMPATIBLE_API_KEY" in os.environ:
                del os.environ["OPENAI_COMPATIBLE_API_KEY"]

    def test_list_available_models(self, llm):
        """Test listing models from OpenAI-compatible server"""
        models = llm.list_available_models()
        assert isinstance(models, list)
        # Should have at least one model
        assert len(models) > 0

    def test_validate_config(self, llm):
        """Test server connectivity validation"""
        assert llm.validate_config() is True

    def test_get_capabilities(self, llm):
        """Test provider capabilities"""
        capabilities = llm.get_capabilities()
        assert isinstance(capabilities, list)
        assert "streaming" in capabilities
        assert "chat" in capabilities


class TestGeneration:
    """Test generation features."""

    def test_basic_generation(self, llm):
        """Test basic text generation"""
        response = llm.generate("Say 'Hello, World!' and nothing else.", temperature=0)
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model is not None

    def test_streaming_generation(self, llm):
        """Test streaming response generation"""
        chunks = []
        for chunk in llm.generate("Count from 1 to 5", stream=True, temperature=0):
            chunks.append(chunk)
            assert chunk.content is not None

        # Should have multiple chunks
        assert len(chunks) > 1

        # Concatenate content
        full_content = "".join(chunk.content for chunk in chunks if chunk.content)
        assert len(full_content) > 0

    @pytest.mark.asyncio
    async def test_async_generation(self, llm):
        """Test async generation"""
        response = await llm.agenerate("Say 'test' and nothing else.", temperature=0)
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_async_streaming(self, llm):
        """Test async streaming generation"""
        chunks = []
        async for chunk in await llm.agenerate("Count from 1 to 3", stream=True, temperature=0):
            chunks.append(chunk)
            assert chunk.content is not None

        # Should have multiple chunks
        assert len(chunks) > 1

        # Concatenate content
        full_content = "".join(chunk.content for chunk in chunks if chunk.content)
        assert len(full_content) > 0


class TestStructuredOutput:
    """Test structured output with Pydantic models."""

    def test_structured_output(self, llm):
        """Test structured output with Pydantic model"""
        try:
            from pydantic import BaseModel

            class PersonInfo(BaseModel):
                name: str
                age: int
                city: str

            response = llm.generate(
                "Extract: John Doe, 30 years old, lives in Seattle",
                response_model=PersonInfo,
                temperature=0
            )

            assert response is not None
            # Response might be the model instance directly
            if isinstance(response, PersonInfo):
                person = response
            else:
                # Or in response.content as parsed model
                person = response

            assert isinstance(person, PersonInfo)
            assert person.name is not None
            assert person.age > 0

        except ImportError:
            pytest.skip("Pydantic not installed")


class TestEmbeddings:
    """Test embeddings generation."""

    def test_embeddings(self, llm):
        """Test embedding generation"""
        try:
            result = llm.embed("Hello world")
            assert result is not None
            assert "data" in result
            assert len(result["data"]) > 0
            assert "embedding" in result["data"][0]
            assert isinstance(result["data"][0]["embedding"], list)
            assert len(result["data"][0]["embedding"]) > 0
        except Exception as e:
            # Some servers may not support embeddings
            if "embedding" in str(e).lower() or "not found" in str(e).lower():
                pytest.skip(f"Server does not support embeddings: {e}")
            else:
                raise


class TestTools:
    """Test prompted tool support."""

    def test_tool_prompt_injection(self, llm):
        """Test that tools can be injected into prompt"""
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        ]

        # Just test that it doesn't error with tools
        response = llm.generate(
            "What's the weather in Seattle?",
            tools=tools,
            temperature=0
        )
        assert response is not None
        assert response.content is not None


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_model(self, base_url: str):
        """Test handling of invalid model name"""
        from abstractcore.exceptions import ModelNotFoundError

        # Get actual available models
        llm = create_llm("openai-compatible", model="default", base_url=base_url)
        available_models = llm.list_available_models()

        # Try with a model that definitely doesn't exist
        invalid_model = "absolutely-nonexistent-model-xyz-12345"

        # Only test if the invalid model isn't actually in the list
        if invalid_model not in available_models:
            with pytest.raises((ModelNotFoundError, Exception)):
                llm = OpenAICompatibleProvider(model=invalid_model, base_url=base_url)
                llm.generate("test")


if __name__ == "__main__":
    # Quick test when running directly
    if server_available():
        print("✅ OpenAI-compatible server available for testing")
        print(f"Base URL: {_TEST_BASE_URL}")
        print(f"Model: {_TEST_MODEL}")

        # Quick smoke test
        llm = create_llm("openai-compatible", model=_TEST_MODEL, base_url=_TEST_BASE_URL)
        print(f"\nAvailable models: {llm.list_available_models()}")
        print("\nTrying basic generation...")
        response = llm.generate("Say hello", temperature=0)
        print(f"Response: {response.content}")
        print("\n✅ Basic functionality working!")
    else:
        print("❌ OpenAI-compatible server not available.")
        print("Start an OpenAI-compatible server first, for example:")
        print("  LM Studio: Start server on :1234")
        print("  Ollama: enable OpenAI-compatible API on :11434")
        print("  llama.cpp: ./server --host 0.0.0.0 --port 8080")
        print("  text-generation-webui: Start with OpenAI extension enabled")
        print("  LocalAI: docker run -p 8080:8080 localai/localai")
        print("\nThen set OPENAI_COMPATIBLE_BASE_URL environment variable:")
        print("  export OPENAI_COMPATIBLE_BASE_URL='http://127.0.0.1:1234/v1'")

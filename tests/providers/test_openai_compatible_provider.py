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
    export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8080/v1"
    export OPENAI_COMPATIBLE_API_KEY="optional-key"  # If server requires auth
    pytest tests/providers/test_openai_compatible_provider.py -v
"""

import pytest
import os
import httpx
from abstractcore import create_llm
from abstractcore.providers import OpenAICompatibleProvider


def server_available():
    """Check if OpenAI-compatible server is accessible."""
    try:
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8080/v1")
        # Try without authentication first
        try:
            response = httpx.get(f"{base_url}/models", timeout=5.0)
        except:
            # If that fails, try with API key if provided
            api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
            if api_key:
                headers = {"Authorization": f"Bearer {api_key}"}
                response = httpx.get(f"{base_url}/models", headers=headers, timeout=5.0)
            else:
                return False
        if response.status_code != 200:
            return False
        # Avoid false positives: require a JSON OpenAI-compatible `/models` shape.
        try:
            payload = response.json()
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        data = payload.get("data")
        return isinstance(data, list) and len(data) > 0
    except:
        return False


pytestmark = pytest.mark.skipif(
    not server_available(),
    reason="OpenAI-compatible server not available. Set OPENAI_COMPATIBLE_BASE_URL to test."
)


class TestProviderBasics:
    """Test basic provider functionality."""

    def test_provider_initialization(self):
        """Test provider can be initialized"""
        llm = create_llm("openai-compatible", model="default")
        assert llm is not None
        assert isinstance(llm, OpenAICompatibleProvider)
        assert llm.provider == "openai-compatible"

    def test_environment_variable_base_url(self):
        """Test base_url can be set via environment variable"""
        test_url = "http://custom-server:8080/v1"
        os.environ["OPENAI_COMPATIBLE_BASE_URL"] = test_url
        try:
            llm = OpenAICompatibleProvider(model="test-model")
            assert llm.base_url == test_url
        finally:
            # Cleanup
            if "OPENAI_COMPATIBLE_BASE_URL" in os.environ:
                del os.environ["OPENAI_COMPATIBLE_BASE_URL"]

    def test_api_key_optional(self):
        """Test API key is optional"""
        llm = OpenAICompatibleProvider(model="test-model", base_url="http://localhost:8080/v1")
        assert llm.api_key is None or llm.api_key == ""

    def test_api_key_environment_variable(self):
        """Test API key can be set via environment variable"""
        test_key = "sk-test-key-12345"
        os.environ["OPENAI_COMPATIBLE_API_KEY"] = test_key
        try:
            llm = OpenAICompatibleProvider(model="test-model")
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
            llm = OpenAICompatibleProvider(model="test-model", api_key=test_key)
            assert llm.api_key == test_key
        finally:
            if "OPENAI_COMPATIBLE_API_KEY" in os.environ:
                del os.environ["OPENAI_COMPATIBLE_API_KEY"]

    def test_list_available_models(self):
        """Test listing models from OpenAI-compatible server"""
        llm = create_llm("openai-compatible", model="default")
        models = llm.list_available_models()
        assert isinstance(models, list)
        # Should have at least one model
        assert len(models) > 0

    def test_validate_config(self):
        """Test server connectivity validation"""
        llm = create_llm("openai-compatible", model="default")
        assert llm.validate_config() is True

    def test_get_capabilities(self):
        """Test provider capabilities"""
        llm = create_llm("openai-compatible", model="default")
        capabilities = llm.get_capabilities()
        assert isinstance(capabilities, list)
        assert "streaming" in capabilities
        assert "chat" in capabilities


class TestGeneration:
    """Test generation features."""

    def test_basic_generation(self):
        """Test basic text generation"""
        llm = create_llm("openai-compatible", model="default")
        response = llm.generate("Say 'Hello, World!' and nothing else.", temperature=0)
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model is not None

    def test_streaming_generation(self):
        """Test streaming response generation"""
        llm = create_llm("openai-compatible", model="default")
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
    async def test_async_generation(self):
        """Test async generation"""
        llm = create_llm("openai-compatible", model="default")
        response = await llm.agenerate("Say 'test' and nothing else.", temperature=0)
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_async_streaming(self):
        """Test async streaming generation"""
        llm = create_llm("openai-compatible", model="default")
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

    def test_structured_output(self):
        """Test structured output with Pydantic model"""
        try:
            from pydantic import BaseModel

            class PersonInfo(BaseModel):
                name: str
                age: int
                city: str

            llm = create_llm("openai-compatible", model="default")
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

    def test_embeddings(self):
        """Test embedding generation"""
        llm = create_llm("openai-compatible", model="default")

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

    def test_tool_prompt_injection(self):
        """Test that tools can be injected into prompt"""
        llm = create_llm("openai-compatible", model="default")

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

    def test_invalid_model(self):
        """Test handling of invalid model name"""
        from abstractcore.exceptions import ModelNotFoundError

        # Get actual available models
        llm = create_llm("openai-compatible", model="default")
        available_models = llm.list_available_models()

        # Try with a model that definitely doesn't exist
        invalid_model = "absolutely-nonexistent-model-xyz-12345"

        # Only test if the invalid model isn't actually in the list
        if invalid_model not in available_models:
            with pytest.raises((ModelNotFoundError, Exception)):
                llm = OpenAICompatibleProvider(model=invalid_model)
                llm.generate("test")


if __name__ == "__main__":
    # Quick test when running directly
    if server_available():
        print("✅ OpenAI-compatible server available for testing")
        print(f"Base URL: {os.getenv('OPENAI_COMPATIBLE_BASE_URL', 'http://localhost:8080/v1')}")

        # Quick smoke test
        llm = create_llm("openai-compatible", model="default")
        print(f"\nAvailable models: {llm.list_available_models()}")
        print("\nTrying basic generation...")
        response = llm.generate("Say hello", temperature=0)
        print(f"Response: {response.content}")
        print("\n✅ Basic functionality working!")
    else:
        print("❌ OpenAI-compatible server not available.")
        print("Start an OpenAI-compatible server first, for example:")
        print("  llama.cpp: ./server --host 0.0.0.0 --port 8080")
        print("  text-generation-webui: Start with OpenAI extension enabled")
        print("  LocalAI: docker run -p 8080:8080 localai/localai")
        print("\nThen set OPENAI_COMPATIBLE_BASE_URL environment variable:")
        print("  export OPENAI_COMPATIBLE_BASE_URL='http://localhost:8080/v1'")

"""
Tests for vLLM provider.

Note: These tests require a vLLM server running. Set VLLM_BASE_URL environment
variable to point to your vLLM server, or tests will skip gracefully.

Example:
    export VLLM_BASE_URL="http://localhost:8000/v1"
    vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000

    pytest tests/providers/test_vllm_provider.py -v
"""

import pytest
import os
import httpx
from abstractcore import create_llm
from abstractcore.providers import VLLMProvider


def vllm_available():
    """Check if vLLM server is accessible."""
    try:
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        # Try to connect to /models endpoint
        response = httpx.get(f"{base_url}/models", timeout=5.0)
        return response.status_code == 200
    except:
        return False


# Skip all tests if vLLM server is not available
pytestmark = pytest.mark.skipif(
    not vllm_available(),
    reason="vLLM server not available. Start with: vllm serve <model> --port 8000"
)


class TestVLLMProviderBasics:
    """Test basic vLLM provider functionality."""

    def test_provider_initialization(self):
        """Test that vLLM provider initializes correctly."""
        llm = create_llm("vllm", model="Qwen/Qwen3-Coder-30B-A3B-Instruct")
        assert llm is not None
        assert llm.provider == "vllm"
        assert llm.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"

    def test_environment_variable_base_url(self):
        """Test that VLLM_BASE_URL environment variable is respected."""
        # Get current env var
        original_url = os.getenv("VLLM_BASE_URL")

        try:
            # Set custom URL
            test_url = "http://custom.vllm.server:8080/v1"
            os.environ["VLLM_BASE_URL"] = test_url

            llm = VLLMProvider(model="test-model")
            assert llm.base_url == test_url
        finally:
            # Restore original env var
            if original_url:
                os.environ["VLLM_BASE_URL"] = original_url
            else:
                os.environ.pop("VLLM_BASE_URL", None)

    def test_api_key_environment_variable(self):
        """Test that VLLM_API_KEY environment variable is respected."""
        original_key = os.getenv("VLLM_API_KEY")

        try:
            test_key = "test-secret-key"
            os.environ["VLLM_API_KEY"] = test_key

            llm = VLLMProvider(model="test-model")
            assert llm.api_key == test_key
        finally:
            if original_key:
                os.environ["VLLM_API_KEY"] = original_key
            else:
                os.environ.pop("VLLM_API_KEY", None)

    def test_list_available_models(self):
        """Test listing available models from vLLM server."""
        llm = create_llm("vllm")
        models = llm.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0
        print(f"Available vLLM models: {models}")

    def test_validate_config(self):
        """Test server connection validation."""
        llm = create_llm("vllm")
        is_valid = llm.validate_config()
        assert is_valid is True

    def test_get_capabilities(self):
        """Test vLLM-specific capabilities."""
        llm = create_llm("vllm")
        capabilities = llm.get_capabilities()

        assert "streaming" in capabilities
        assert "chat" in capabilities
        assert "tools" in capabilities
        assert "structured_output" in capabilities
        # vLLM-specific
        assert "guided_decoding" in capabilities
        assert "multi_lora" in capabilities
        assert "beam_search" in capabilities


class TestVLLMGeneration:
    """Test generation features."""

    def test_basic_generation(self):
        """Test basic text generation."""
        llm = create_llm("vllm")
        response = llm.generate("Say 'Hello World' and nothing else.", temperature=0)

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert "hello" in response.content.lower()
        print(f"Response: {response.content}")

    def test_streaming_generation(self):
        """Test streaming generation."""
        llm = create_llm("vllm")
        chunks = []

        for chunk in llm.generate("Count from 1 to 5", stream=True, temperature=0):
            if chunk.content:
                chunks.append(chunk.content)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        print(f"Streamed response: {full_response}")

    @pytest.mark.asyncio
    async def test_async_generation(self):
        """Test async generation."""
        llm = create_llm("vllm")
        response = await llm.agenerate("Say 'Async works'", temperature=0)

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        print(f"Async response: {response.content}")

    @pytest.mark.asyncio
    async def test_async_streaming(self):
        """Test async streaming generation."""
        llm = create_llm("vllm")
        chunks = []

        stream = await llm.agenerate("Count to 3", stream=True, temperature=0)
        async for chunk in stream:
            if chunk.content:
                chunks.append(chunk.content)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        print(f"Async streamed response: {full_response}")


class TestVLLMGuidedDecoding:
    """Test vLLM-specific guided decoding features."""

    def test_guided_json(self):
        """Test guided JSON generation."""
        llm = create_llm("vllm")

        # Define a simple JSON schema
        json_schema = {
            "type": "object",
            "properties": {
                "colors": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["colors"]
        }

        response = llm.generate(
            "List 3 colors",
            guided_json=json_schema,
            temperature=0
        )

        assert response is not None
        assert response.content is not None
        print(f"Guided JSON response: {response.content}")

        # Try to parse as JSON
        import json
        try:
            data = json.loads(response.content)
            assert "colors" in data
            assert isinstance(data["colors"], list)
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")

    def test_guided_regex(self):
        """Test guided regex generation."""
        llm = create_llm("vllm")

        # Simple regex for yes/no answer
        response = llm.generate(
            "Is the sky blue? Answer yes or no.",
            guided_regex="(yes|no)",
            temperature=0
        )

        assert response is not None
        assert response.content is not None
        assert response.content.lower() in ["yes", "no"]
        print(f"Guided regex response: {response.content}")


class TestVLLMBeamSearch:
    """Test vLLM beam search features."""

    def test_beam_search(self):
        """Test beam search with best_of parameter."""
        llm = create_llm("vllm")

        response = llm.generate(
            "Write a creative haiku about coding",
            use_beam_search=True,
            best_of=3,
            temperature=0.7
        )

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        print(f"Beam search response: {response.content}")


class TestVLLMStructuredOutput:
    """Test structured output with Pydantic models."""

    def test_structured_output(self):
        """Test structured output with response_model."""
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("Pydantic not installed")

        class Person(BaseModel):
            name: str
            age: int

        llm = create_llm("vllm")
        person = llm.generate(
            "John Doe is 35 years old",
            response_model=Person,
            temperature=0
        )

        assert isinstance(person, Person)
        assert person.name is not None
        assert person.age > 0
        print(f"Structured output: name={person.name}, age={person.age}")


class TestVLLMLoRA:
    """Test vLLM LoRA adapter management.

    Note: These tests require vLLM server started with --enable-lora flag.
    If LoRA is not enabled, tests will be skipped.
    """

    def test_list_adapters(self):
        """Test listing LoRA adapters."""
        llm = create_llm("vllm")

        try:
            adapters = llm.list_adapters()
            assert isinstance(adapters, list)
            print(f"Currently loaded adapters: {adapters}")
        except Exception as e:
            # LoRA might not be enabled on the server
            if "404" in str(e) or "not found" in str(e).lower():
                pytest.skip("vLLM server does not have LoRA enabled. Start with: vllm serve --enable-lora")
            else:
                raise

    def test_load_unload_adapter(self):
        """Test loading and unloading LoRA adapters.

        Note: This test requires an actual LoRA adapter to be available.
        Skipped if adapter path is not configured.
        """
        # This is a placeholder test - requires actual adapter files
        # User should replace with real adapter paths for testing
        adapter_path = os.getenv("TEST_LORA_ADAPTER_PATH")

        if not adapter_path:
            pytest.skip("TEST_LORA_ADAPTER_PATH not set. Set it to test LoRA loading.")

        llm = create_llm("vllm")

        try:
            # Load adapter
            result = llm.load_adapter("test-adapter", adapter_path)
            assert "loaded successfully" in result.lower()

            # Verify it's in the list
            adapters = llm.list_adapters()
            assert "test-adapter" in adapters

            # Unload adapter
            result = llm.unload_adapter("test-adapter")
            assert "unloaded successfully" in result.lower()

        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                pytest.skip("vLLM server does not have LoRA enabled")
            else:
                raise


class TestVLLMEmbeddings:
    """Test vLLM embeddings generation.

    Note: Requires embedding model loaded in vLLM.
    """

    def test_embeddings(self):
        """Test generating embeddings."""
        llm = create_llm("vllm")

        try:
            result = llm.embed("Hello world")

            assert result is not None
            assert "data" in result
            assert len(result["data"]) > 0
            assert "embedding" in result["data"][0]

            embedding = result["data"][0]["embedding"]
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            print(f"Embedding dimension: {len(embedding)}")

        except Exception as e:
            # Model might not support embeddings
            if "not found" in str(e).lower() or "does not support" in str(e).lower():
                pytest.skip("Model does not support embeddings")
            else:
                raise


if __name__ == "__main__":
    # Quick test when running directly
    if vllm_available():
        print("✅ vLLM server is available!")
        llm = create_llm("vllm")
        print(f"Provider: {llm.provider}")
        print(f"Model: {llm.model}")
        print(f"Base URL: {llm.base_url}")

        response = llm.generate("Say hello", temperature=0)
        print(f"Response: {response.content}")
    else:
        print("❌ vLLM server is not available.")
        print("Start with: vllm serve <model> --port 8000")

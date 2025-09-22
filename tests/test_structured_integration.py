"""
Integration tests for structured output with real providers.
"""

import pytest
from typing import List, Optional

try:
    from pydantic import BaseModel, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from abstractllm import create_llm


# Test models
if PYDANTIC_AVAILABLE:
    class SimpleUser(BaseModel):
        name: str
        age: int

    class ContactInfo(BaseModel):
        name: str
        email: str
        phone: Optional[str] = None

    class TaskInfo(BaseModel):
        title: str
        priority: str  # "high", "medium", "low"
        completed: bool = False


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
@pytest.mark.integration
class TestStructuredOutputIntegration:
    """Integration tests with real provider APIs."""

    def test_mlx_provider_structured_output(self):
        """Test structured output with MLX provider (local)."""
        try:
            llm = create_llm("mlx", model="mlx-community/GLM-4.5-Air-4bit")

            result = llm.generate(
                "Extract user info: John Doe, 25 years old",
                response_model=SimpleUser
            )

            assert isinstance(result, SimpleUser)
            assert "John" in result.name or "Doe" in result.name
            assert result.age == 25

        except Exception as e:
            pytest.skip(f"MLX provider not available: {e}")

    @pytest.mark.provider
    def test_openai_structured_output(self):
        """Test structured output with OpenAI provider."""
        try:
            llm = create_llm("openai", model="gpt-4o-mini")

            result = llm.generate(
                "Extract the following user information: Sarah Johnson, 28 years old, sarah.j@email.com, phone: 555-0123",
                response_model=ContactInfo
            )

            assert isinstance(result, ContactInfo)
            assert "Sarah" in result.name
            assert "@" in result.email
            assert result.phone is not None

        except Exception as e:
            pytest.skip(f"OpenAI provider not available: {e}")

    @pytest.mark.provider
    def test_anthropic_structured_output(self):
        """Test structured output with Anthropic provider using tool trick."""
        try:
            llm = create_llm("anthropic", model="claude-3-haiku-20240307")

            result = llm.generate(
                "Create a task: 'Review quarterly reports' with high priority",
                response_model=TaskInfo
            )

            assert isinstance(result, TaskInfo)
            assert "report" in result.title.lower()
            assert result.priority in ["high", "medium", "low"]

        except Exception as e:
            pytest.skip(f"Anthropic provider not available: {e}")

    @pytest.mark.provider
    def test_ollama_structured_output(self):
        """Test structured output with Ollama provider."""
        try:
            llm = create_llm("ollama", model="qwen3-coder:30b")

            result = llm.generate(
                "User profile: Mike Chen, age 35",
                response_model=SimpleUser
            )

            assert isinstance(result, SimpleUser)
            assert "Mike" in result.name or "Chen" in result.name
            assert result.age == 35

        except Exception as e:
            pytest.skip(f"Ollama provider not available: {e}")

    @pytest.mark.provider
    def test_ollama_provider_structured_output(self):
        """Test structured output with Ollama provider (local)."""
        try:
            llm = create_llm("ollama", model="qwen3-coder:30b")

            result = llm.generate(
                "Extract contact info: Alice Johnson, alice@company.com",
                response_model=ContactInfo
            )

            assert isinstance(result, ContactInfo)
            assert "Alice" in result.name
            assert "@" in result.email

        except Exception as e:
            pytest.skip(f"Ollama provider not available: {e}")


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestStructuredOutputUseCases:
    """Test common use cases for structured output."""

    def test_data_extraction_use_case(self):
        """Test extracting structured data from unstructured text."""

        class ProductInfo(BaseModel):
            name: str
            price: float
            category: str

        # Mock provider that extracts product info
        class MockExtractorProvider:
            model_capabilities = {"structured_output": "prompted"}

            def _generate_internal(self, **kwargs):
                from abstractllm.core.types import GenerateResponse
                return GenerateResponse(
                    content='{"name": "Wireless Headphones", "price": 99.99, "category": "Electronics"}'
                )

        from abstractllm.structured.handler import StructuredOutputHandler

        handler = StructuredOutputHandler()
        provider = MockExtractorProvider()

        unstructured_text = """
        Check out these amazing Wireless Headphones!
        They're in the Electronics section and cost just $99.99.
        Perfect for music lovers!
        """

        result = handler.generate_structured(
            provider=provider,
            prompt=f"Extract product information from: {unstructured_text}",
            response_model=ProductInfo
        )

        assert isinstance(result, ProductInfo)
        assert result.name == "Wireless Headphones"
        assert result.price == 99.99
        assert result.category == "Electronics"

    def test_form_filling_use_case(self):
        """Test filling out structured forms from natural language."""

        class EventRegistration(BaseModel):
            attendee_name: str
            email: str
            event_preference: str
            dietary_restrictions: Optional[str] = None

        class MockFormFillerProvider:
            model_capabilities = {"structured_output": "prompted"}

            def _generate_internal(self, **kwargs):
                from abstractllm.core.types import GenerateResponse
                return GenerateResponse(
                    content='''{"attendee_name": "Alice Smith", "email": "alice@example.com",
                             "event_preference": "workshop", "dietary_restrictions": "vegetarian"}'''
                )

        from abstractllm.structured.handler import StructuredOutputHandler

        handler = StructuredOutputHandler()
        provider = MockFormFillerProvider()

        natural_input = """
        I'm Alice Smith, my email is alice@example.com.
        I'd like to attend the workshop session.
        Please note that I'm vegetarian.
        """

        result = handler.generate_structured(
            provider=provider,
            prompt=f"Fill out event registration form from: {natural_input}",
            response_model=EventRegistration
        )

        assert isinstance(result, EventRegistration)
        assert result.attendee_name == "Alice Smith"
        assert result.email == "alice@example.com"
        assert result.event_preference == "workshop"
        assert result.dietary_restrictions == "vegetarian"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
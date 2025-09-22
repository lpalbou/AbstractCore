"""
Tests for structured output functionality.
"""

import json
import pytest
from typing import List, Optional
from enum import Enum

try:
    from pydantic import BaseModel, field_validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from abstractllm.structured.retry import FeedbackRetry
from abstractllm.structured.handler import StructuredOutputHandler


# Test models
if PYDANTIC_AVAILABLE:
    class UserInfo(BaseModel):
        name: str
        age: int
        email: str

    class Address(BaseModel):
        street: str
        city: str
        zip_code: str

    class Person(BaseModel):
        name: str
        age: int
        addresses: List[Address]

    class OrderStatus(str, Enum):
        PENDING = "pending"
        CONFIRMED = "confirmed"
        SHIPPED = "shipped"
        DELIVERED = "delivered"

    class OrderItem(BaseModel):
        name: str
        quantity: int
        price: float

    class Order(BaseModel):
        id: str
        items: List[OrderItem]
        status: OrderStatus
        total: float

    class ValidatedModel(BaseModel):
        value: int

        @field_validator('value')
        @classmethod
        def must_be_positive(cls, v):
            if v <= 0:
                raise ValueError('value must be positive')
            return v


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestFeedbackRetry:
    """Test the FeedbackRetry strategy."""

    def test_should_retry_validation_error(self):
        """Test that ValidationError triggers retry."""
        retry = FeedbackRetry(max_attempts=3)

        # Create a validation error
        try:
            UserInfo(name="John", age="not_a_number", email="john@example.com")
        except ValidationError as e:
            assert retry.should_retry(1, e) is True
            assert retry.should_retry(2, e) is True
            assert retry.should_retry(3, e) is False

    def test_should_not_retry_other_errors(self):
        """Test that non-ValidationError doesn't trigger retry."""
        retry = FeedbackRetry(max_attempts=3)

        error = ValueError("Some other error")
        assert retry.should_retry(1, error) is False

    def test_prepare_retry_prompt(self):
        """Test retry prompt generation with error feedback."""
        retry = FeedbackRetry()
        original_prompt = "Extract user information from: John Doe, 30, john@example.com"

        # Create a validation error
        try:
            UserInfo(name="John", age="not_a_number", email="john@example.com")
        except ValidationError as e:
            retry_prompt = retry.prepare_retry_prompt(original_prompt, e, 1)

            assert original_prompt in retry_prompt
            assert "validation errors" in retry_prompt
            assert "attempt 1" in retry_prompt
            assert "age" in retry_prompt  # Field that caused error

    def test_format_validation_errors(self):
        """Test error formatting for different validation error types."""
        retry = FeedbackRetry()

        # Test missing field error
        try:
            UserInfo(name="John", age=30)  # Missing email
        except ValidationError as e:
            formatted = retry._format_validation_errors(e)
            assert "Missing required field: 'email'" in formatted

        # Test type error
        try:
            UserInfo(name="John", age="not_a_number", email="john@example.com")
        except ValidationError as e:
            formatted = retry._format_validation_errors(e)
            assert "age" in formatted


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestStructuredOutputHandler:
    """Test the StructuredOutputHandler."""

    def test_has_native_support(self):
        """Test native support detection."""
        handler = StructuredOutputHandler()

        # Mock provider with native support
        class MockProviderNative:
            model_capabilities = {"structured_output": "native"}

        # Mock provider without native support
        class MockProviderPrompted:
            model_capabilities = {"structured_output": "prompted"}

        assert handler._has_native_support(MockProviderNative()) is True
        assert handler._has_native_support(MockProviderPrompted()) is False

    def test_create_schema_prompt(self):
        """Test schema prompt generation."""
        handler = StructuredOutputHandler()

        prompt = "Extract user info"
        schema_prompt = handler._create_schema_prompt(prompt, UserInfo)

        assert prompt in schema_prompt
        assert "UserInfo" in schema_prompt
        assert "JSON" in schema_prompt
        assert '"name"' in schema_prompt  # Schema properties
        assert '"age"' in schema_prompt
        assert '"email"' in schema_prompt

    def test_create_example_from_schema(self):
        """Test example generation from schema."""
        handler = StructuredOutputHandler()

        schema = UserInfo.model_json_schema()
        example = handler._create_example_from_schema(schema)

        assert "name" in example
        assert "age" in example
        assert "email" in example
        assert isinstance(example["age"], int)
        assert isinstance(example["name"], str)

    def test_extract_json_simple(self):
        """Test JSON extraction from simple response."""
        handler = StructuredOutputHandler()

        # Simple JSON
        content = '{"name": "John", "age": 30, "email": "john@example.com"}'
        extracted = handler._extract_json(content)
        assert extracted == content

    def test_extract_json_with_code_blocks(self):
        """Test JSON extraction from response with code blocks."""
        handler = StructuredOutputHandler()

        # JSON in code block
        content = '''Here's the extracted information:

```json
{"name": "John", "age": 30, "email": "john@example.com"}
```

This data matches the required format.'''

        extracted = handler._extract_json(content)
        json_data = json.loads(extracted)
        assert json_data["name"] == "John"
        assert json_data["age"] == 30

    def test_extract_json_mixed_content(self):
        """Test JSON extraction from mixed content."""
        handler = StructuredOutputHandler()

        content = '''Based on the information provided, here is the structured data:

{"name": "John Doe", "age": 30, "email": "john@example.com"}

This represents the user information in the requested format.'''

        extracted = handler._extract_json(content)
        json_data = json.loads(extracted)
        assert json_data["name"] == "John Doe"


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestComplexModels:
    """Test with complex nested models."""

    def test_nested_model_schema(self):
        """Test schema generation for nested models."""
        handler = StructuredOutputHandler()

        schema_prompt = handler._create_schema_prompt("Extract person data", Person)

        assert "Person" in schema_prompt
        assert "addresses" in schema_prompt
        assert "Address" in schema_prompt  # Nested model reference

    def test_enum_model_schema(self):
        """Test schema generation for models with enums."""
        handler = StructuredOutputHandler()

        schema_prompt = handler._create_schema_prompt("Extract order data", Order)

        assert "Order" in schema_prompt
        assert "status" in schema_prompt
        assert "OrderStatus" in schema_prompt

    def test_validation_with_custom_validators(self):
        """Test validation with custom field validators."""
        retry = FeedbackRetry()

        # Test positive validation
        try:
            ValidatedModel(value=-5)  # Should fail validation
        except ValidationError as e:
            formatted = retry._format_validation_errors(e)
            assert "must be positive" in formatted


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
class TestIntegrationScenarios:
    """Integration tests with mock providers."""

    def test_mock_native_provider_success(self):
        """Test successful structured output with mock native provider."""
        class MockNativeProvider:
            model_capabilities = {"structured_output": "native"}

            def _generate_internal(self, **kwargs):
                # Mock successful response
                from abstractllm.core.types import GenerateResponse
                return GenerateResponse(
                    content='{"name": "John", "age": 30, "email": "john@example.com"}'
                )

        handler = StructuredOutputHandler()
        provider = MockNativeProvider()

        result = handler.generate_structured(
            provider=provider,
            prompt="Extract user info",
            response_model=UserInfo
        )

        assert isinstance(result, UserInfo)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_mock_prompted_provider_with_retry(self):
        """Test prompted provider with retry on validation failure."""
        class MockPromptedProvider:
            model_capabilities = {"structured_output": "prompted"}
            call_count = 0

            def _generate_internal(self, **kwargs):
                from abstractllm.core.types import GenerateResponse
                self.call_count += 1

                if self.call_count == 1:
                    # First attempt: invalid JSON
                    return GenerateResponse(content='{"name": "John", "age": "invalid"}')
                else:
                    # Second attempt: valid JSON
                    return GenerateResponse(
                        content='{"name": "John", "age": 30, "email": "john@example.com"}'
                    )

        handler = StructuredOutputHandler()
        provider = MockPromptedProvider()

        result = handler.generate_structured(
            provider=provider,
            prompt="Extract user info",
            response_model=UserInfo
        )

        assert isinstance(result, UserInfo)
        assert result.name == "John"
        assert result.age == 30
        assert provider.call_count == 2  # Retry happened

    def test_max_retries_exhausted(self):
        """Test behavior when max retries are exhausted."""
        class MockFailingProvider:
            model_capabilities = {"structured_output": "prompted"}

            def _generate_internal(self, **kwargs):
                from abstractllm.core.types import GenerateResponse
                # Always return invalid data
                return GenerateResponse(content='{"invalid": "data"}')

        handler = StructuredOutputHandler()
        provider = MockFailingProvider()

        with pytest.raises(ValidationError):
            handler.generate_structured(
                provider=provider,
                prompt="Extract user info",
                response_model=UserInfo
            )


if __name__ == "__main__":
    pytest.main([__file__])
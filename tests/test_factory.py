"""
Test the factory function.
"""

import pytest
from abstractllm import create_llm
from abstractllm.providers.mock_provider import MockProvider


class TestFactory:
    """Test create_llm factory function"""

    def test_create_mock_provider(self):
        """Test creating mock provider"""
        llm = create_llm("mock")
        assert isinstance(llm, MockProvider)
        assert llm.model == "mock-model"

    def test_create_mock_with_custom_model(self):
        """Test creating mock with custom model"""
        llm = create_llm("mock", model="custom-mock")
        assert llm.model == "custom-mock"

    def test_unknown_provider_raises_error(self):
        """Test unknown provider raises error"""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm("nonexistent")

    def test_provider_case_insensitive(self):
        """Test provider names are case insensitive"""
        llm1 = create_llm("MOCK")
        llm2 = create_llm("Mock")
        llm3 = create_llm("mock")

        assert all(isinstance(llm, MockProvider) for llm in [llm1, llm2, llm3])
"""
Comprehensive test suite for AbstractLLM core components.
Tests actual implementations with no mocking.
"""

import pytest
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List

from abstractllm import create_llm, BasicSession, GenerateResponse, Message
from abstractllm.core.enums import MessageRole, ModelParameter, ModelCapability
from abstractllm.core.interface import AbstractLLMInterface
from abstractllm.tools.core import ToolDefinition


class TestMessage:
    """Test Message data class"""

    def test_message_creation(self):
        """Test creating a message"""
        msg = Message(role="user", content="Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.timestamp is not None

    def test_message_with_metadata(self):
        """Test message with metadata"""
        metadata = {"source": "test", "version": 1}
        msg = Message(role="assistant", content="Response", metadata=metadata)
        assert msg.metadata == metadata

    def test_message_serialization(self):
        """Test message to_dict and from_dict"""
        msg = Message(role="system", content="System prompt")
        data = msg.to_dict()

        assert data["role"] == "system"
        assert data["content"] == "System prompt"
        assert "timestamp" in data

        # Reconstruct from dict
        msg2 = Message.from_dict(data)
        assert msg2.role == msg.role
        assert msg2.content == msg.content


class TestBasicSession:
    """Test BasicSession functionality"""

    def test_session_creation(self):
        """Test creating a session"""
        session = BasicSession()
        assert session.id is not None
        assert session.created_at is not None
        assert len(session.messages) == 0

    def test_session_with_system_prompt(self):
        """Test session with system prompt"""
        prompt = "You are a helpful assistant"
        session = BasicSession(system_prompt=prompt)

        assert session.system_prompt == prompt
        assert len(session.messages) == 1
        assert session.messages[0].role == "system"
        assert session.messages[0].content == prompt

    def test_add_message(self):
        """Test adding messages to session"""
        session = BasicSession()

        msg1 = session.add_message("user", "Hello")
        assert len(session.messages) == 1
        assert msg1.role == "user"
        assert msg1.content == "Hello"

        msg2 = session.add_message("assistant", "Hi there!")
        assert len(session.messages) == 2
        assert msg2.role == "assistant"

    def test_get_history(self):
        """Test getting conversation history"""
        session = BasicSession(system_prompt="System")
        session.add_message("user", "Question")
        session.add_message("assistant", "Answer")

        # With system message
        history = session.get_history(include_system=True)
        assert len(history) == 3

        # Without system message
        history = session.get_history(include_system=False)
        assert len(history) == 2
        assert all(msg["role"] != "system" for msg in history)

    def test_clear_history(self):
        """Test clearing conversation history"""
        session = BasicSession(system_prompt="System")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi")

        # Clear keeping system
        session.clear_history(keep_system=True)
        assert len(session.messages) == 1
        assert session.messages[0].role == "system"

        # Clear all
        session.add_message("user", "Test")
        session.clear_history(keep_system=False)
        assert len(session.messages) == 0

    def test_save_and_load(self):
        """Test saving and loading session"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Create and save session
            session = BasicSession(system_prompt="Test system")
            session.add_message("user", "Hello")
            session.add_message("assistant", "Hi there")
            session.save(filepath)

            # Load session
            loaded = BasicSession.load(filepath)
            assert loaded.id == session.id
            assert loaded.system_prompt == session.system_prompt
            assert len(loaded.messages) == len(session.messages)
            assert loaded.messages[1].content == "Hello"

        finally:
            os.unlink(filepath)


class TestProviderFactory:
    """Test provider factory"""

    def test_create_mock_provider(self):
        """Test creating mock provider"""
        provider = create_llm("mock")
        assert provider is not None
        assert isinstance(provider, AbstractLLMInterface)

    def test_create_ollama_provider(self):
        """Test creating Ollama provider"""
        provider = create_llm("ollama", model="llama2")
        assert provider is not None
        assert provider.model == "llama2"

    def test_create_with_config(self):
        """Test creating provider with configuration"""
        provider = create_llm("ollama", model="custom", base_url="http://localhost:11434")
        assert provider.base_url == "http://localhost:11434"

    def test_invalid_provider(self):
        """Test invalid provider raises error"""
        with pytest.raises(ValueError) as exc:
            create_llm("invalid_provider")
        assert "Unknown provider" in str(exc.value)


class TestGenerateResponse:
    """Test GenerateResponse data class"""

    def test_response_creation(self):
        """Test creating a response"""
        response = GenerateResponse(
            content="Test response",
            model="test-model",
            finish_reason="stop"
        )
        assert response.content == "Test response"
        assert response.model == "test-model"

    def test_response_with_usage(self):
        """Test response with usage information"""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        response = GenerateResponse(content="Test", usage=usage)
        assert response.usage == usage

    def test_has_tool_calls(self):
        """Test checking for tool calls"""
        response1 = GenerateResponse(content="No tools")
        assert not response1.has_tool_calls()

        response2 = GenerateResponse(
            content="Using tools",
            tool_calls=[{"name": "test_tool", "arguments": {}}]
        )
        assert response2.has_tool_calls()

    def test_get_tools_executed(self):
        """Test getting executed tool names"""
        response = GenerateResponse(
            content="Test",
            tool_calls=[
                {"name": "tool1", "arguments": {}},
                {"name": "tool2", "arguments": {}}
            ]
        )
        tools = response.get_tools_executed()
        assert "tool1" in tools
        assert "tool2" in tools

    def test_get_summary(self):
        """Test response summary"""
        response = GenerateResponse(
            content="Test",
            model="test-model",
            usage={"total_tokens": 100},
            tool_calls=[{"name": "tool1"}]
        )
        summary = response.get_summary()
        assert "test-model" in summary
        assert "100" in summary
        assert "1 executed" in summary


class TestToolDefinition:
    """Test tool definitions"""

    def test_tool_creation(self):
        """Test creating tool definition"""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                }
            }
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_tool_from_function(self):
        """Test creating tool from function"""
        def sample_function(text: str, count: int = 1) -> str:
            """Sample function for testing"""
            return text * count

        tool = ToolDefinition.from_function(sample_function)
        assert tool.name == "sample_function"
        assert "Sample function" in tool.description
        assert "text" in tool.parameters
        assert "count" in tool.parameters
        assert tool.function == sample_function

    def test_tool_to_dict(self):
        """Test converting tool to dict"""
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters={"type": "object"}
        )
        data = tool.to_dict()
        assert data["name"] == "test"
        assert data["description"] == "Test tool"
        assert data["parameters"] == {"type": "object"}


class TestProviderIntegration:
    """Integration tests with actual providers"""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
    def test_openai_generation(self):
        """Test OpenAI provider generation"""
        provider = create_llm("openai", model="gpt-3.5-turbo")
        response = provider.generate("Say 'test passed' and nothing else")

        assert response is not None
        assert response.content is not None
        assert "test passed" in response.content.lower()

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
    def test_anthropic_generation(self):
        """Test Anthropic provider generation"""
        provider = create_llm("anthropic", model="claude-3-haiku-20240307")
        response = provider.generate("Say 'test passed' and nothing else")

        assert response is not None
        assert response.content is not None
        assert "test passed" in response.content.lower()

    def test_ollama_generation(self):
        """Test Ollama provider generation (if available)"""
        try:
            import httpx
            client = httpx.Client(timeout=5.0)
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                pytest.skip("Ollama not available")
        except:
            pytest.skip("Ollama not available")

        provider = create_llm("ollama", model="qwen3-coder:30b")
        response = provider.generate("Say 'test passed' and nothing else")

        assert response is not None
        assert response.content is not None

    def test_session_with_provider(self):
        """Test session with mock provider"""
        provider = create_llm("mock")
        session = BasicSession(provider=provider, system_prompt="Test system")

        response = session.generate("Hello")
        assert response is not None
        assert len(session.messages) >= 2  # User + assistant messages


def test_enums():
    """Test enum definitions"""
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.SYSTEM.value == "system"

    # Test ModelParameter enum exists
    assert hasattr(ModelParameter, "MODEL")
    assert hasattr(ModelParameter, "TEMPERATURE")

    # Test ModelCapability enum exists
    assert hasattr(ModelCapability, "CHAT")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
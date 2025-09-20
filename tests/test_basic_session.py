"""
Unit tests for BasicSession - the core conversation manager.
"""

import pytest
from datetime import datetime
from abstractllm.core.session import BasicSession
from abstractllm.core.types import Message
from abstractllm.providers.mock_provider import MockProvider


class TestBasicSession:
    """Test BasicSession functionality"""

    def test_session_creation(self):
        """Test session can be created"""
        session = BasicSession()
        assert session.id is not None
        assert isinstance(session.created_at, datetime)
        assert len(session.messages) == 0

    def test_add_message(self):
        """Test adding messages"""
        session = BasicSession()

        # Add user message
        msg = session.add_message('user', 'Hello')
        assert msg.role == 'user'
        assert msg.content == 'Hello'
        assert len(session.messages) == 1

    def test_system_prompt(self):
        """Test system prompt handling"""
        session = BasicSession(system_prompt="You are helpful")

        # Should have one system message
        assert len(session.messages) == 1
        assert session.messages[0].role == 'system'

    def test_get_history(self):
        """Test conversation history retrieval"""
        session = BasicSession(system_prompt="System")
        session.add_message('user', 'Hello')
        session.add_message('assistant', 'Hi')

        # With system
        history = session.get_history(include_system=True)
        assert len(history) == 3

        # Without system
        history = session.get_history(include_system=False)
        assert len(history) == 2

    def test_clear_history(self):
        """Test clearing conversation"""
        session = BasicSession(system_prompt="System")
        session.add_message('user', 'Hello')

        # Clear keeping system
        session.clear_history(keep_system=True)
        assert len(session.messages) == 1
        assert session.messages[0].role == 'system'

        # Clear completely
        session.clear_history(keep_system=False)
        assert len(session.messages) == 0

    def test_generation_with_mock_provider(self):
        """Test generation with mock provider"""
        provider = MockProvider()
        session = BasicSession(provider=provider)

        response = session.generate("Hello")
        assert response.content is not None
        assert "Mock response" in response.content
        assert len(session.messages) == 2  # user + assistant

    def test_persistence(self, tmp_path):
        """Test save and load"""
        # Create and save
        session = BasicSession(system_prompt="Test")
        session.add_message('user', 'Hello')
        session.add_message('assistant', 'Hi')

        save_path = tmp_path / "session.json"
        session.save(save_path)

        # Load and verify
        loaded = BasicSession.load(save_path)
        assert loaded.id == session.id
        assert len(loaded.messages) == 3
        assert loaded.messages[1].content == 'Hello'
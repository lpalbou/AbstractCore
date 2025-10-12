"""
Tests for Ollama provider tool role message conversion fix.

Tests the fix for handling OpenAI role: "tool" messages that Ollama doesn't support.
Based on bug report: ollama-report-tool-role.md
"""

import pytest
from abstractllm.providers.ollama_provider import OllamaProvider


class TestOllamaToolRoleConversion:
    """Test the message conversion logic for unsupported roles"""

    def setup_method(self):
        """Setup OllamaProvider instance for testing"""
        self.provider = OllamaProvider(model="test-model")

    def test_tool_role_conversion(self):
        """Test that tool role messages are converted to user messages with markers"""
        messages = [
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": "I'll list files", "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "shell", "arguments": {"command": ["ls"]}}}
            ]},
            {"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "call_123"}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        # Should have 3 messages
        assert len(converted) == 3

        # First message unchanged
        assert converted[0] == {"role": "user", "content": "List files"}

        # Assistant message should have tool_calls removed
        assert converted[1]["role"] == "assistant"
        assert converted[1]["content"] == "I'll list files"
        assert "tool_calls" not in converted[1]

        # Tool message converted to user with markers
        assert converted[2]["role"] == "user"
        assert converted[2]["content"] == "[TOOL RESULT call_123]: file1.txt\nfile2.txt"

    def test_assistant_tool_calls_removal(self):
        """Test that tool_calls are removed from assistant messages"""
        messages = [
            {"role": "assistant", "content": "I'll help", "tool_calls": [
                {"id": "call_456", "type": "function", "function": {"name": "search", "arguments": {"query": "test"}}}
            ]}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "I'll help"
        assert "tool_calls" not in converted[0]

    def test_supported_roles_preserved(self):
        """Test that supported roles are preserved unchanged"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        # Should be identical
        assert converted == messages

    def test_tool_role_with_missing_call_id(self):
        """Test tool role conversion when tool_call_id is missing"""
        messages = [
            {"role": "tool", "content": "output content"}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "[TOOL RESULT unknown]: output content"

    def test_tool_role_with_empty_content(self):
        """Test tool role conversion with empty content"""
        messages = [
            {"role": "tool", "content": "", "tool_call_id": "call_789"}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "[TOOL RESULT call_789]: "

    def test_assistant_with_empty_content_and_tool_calls(self):
        """Test assistant message with only tool_calls (no content)"""
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_999", "type": "function", "function": {"name": "action", "arguments": {}}}
            ]}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == ""
        assert "tool_calls" not in converted[0]

    def test_mixed_message_types(self):
        """Test conversion with a mix of all message types"""
        messages = [
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": "Run a command"},
            {"role": "assistant", "content": "I'll run the command", "tool_calls": [
                {"id": "call_abc", "type": "function", "function": {"name": "shell", "arguments": {"cmd": "ls"}}}
            ]},
            {"role": "tool", "content": "file1.py\nfile2.py", "tool_call_id": "call_abc"},
            {"role": "assistant", "content": "The files are listed above"},
            {"role": "user", "content": "Thank you"}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        assert len(converted) == 6

        # System message unchanged
        assert converted[0] == messages[0]

        # User message unchanged
        assert converted[1] == messages[1]

        # Assistant with tool_calls -> tool_calls removed
        assert converted[2]["role"] == "assistant"
        assert converted[2]["content"] == "I'll run the command"
        assert "tool_calls" not in converted[2]

        # Tool message -> user with markers
        assert converted[3]["role"] == "user"
        assert converted[3]["content"] == "[TOOL RESULT call_abc]: file1.py\nfile2.py"

        # Assistant without tool_calls -> unchanged
        assert converted[4] == messages[4]

        # User message unchanged
        assert converted[5] == messages[5]

    def test_empty_messages_list(self):
        """Test conversion with empty messages list"""
        messages = []
        converted = self.provider._convert_messages_for_ollama(messages)
        assert converted == []

    def test_none_messages(self):
        """Test conversion with None messages"""
        messages = None
        converted = self.provider._convert_messages_for_ollama(messages)
        assert converted == []

    def test_invalid_message_types(self):
        """Test conversion with invalid message types (non-dict)"""
        messages = [
            {"role": "user", "content": "Valid message"},
            "invalid string message",
            None,
            42,
            {"role": "assistant", "content": "Another valid message"}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        # Should only include valid dict messages
        assert len(converted) == 2
        assert converted[0] == {"role": "user", "content": "Valid message"}
        assert converted[1] == {"role": "assistant", "content": "Another valid message"}

    def test_complex_tool_call_arguments(self):
        """Test conversion with complex tool call arguments in assistant message"""
        messages = [
            {"role": "assistant", "content": "I'll perform the search", "tool_calls": [
                {
                    "id": "call_complex",
                    "type": "function",
                    "function": {
                        "name": "complex_search",
                        "arguments": {
                            "query": "python testing",
                            "filters": ["recent", "high_quality"],
                            "limit": 10,
                            "metadata": {"source": "documentation", "priority": "high"}
                        }
                    }
                }
            ]}
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert converted[0]["content"] == "I'll perform the search"
        assert "tool_calls" not in converted[0]


class TestOllamaProviderIntegration:
    """Integration tests to ensure the fix works with the provider"""

    def setup_method(self):
        """Setup OllamaProvider instance for testing"""
        self.provider = OllamaProvider(model="test-model")

    def test_bug_report_scenario_direct(self):
        """Test the exact scenario from the bug report"""
        # This is the exact scenario that was failing in the bug report
        messages = [
            {"role": "user", "content": "List files"},
            {"role": "assistant", "content": "I will list the files.", "tool_calls": [
                {"id": "call_123", "type": "function", "function": {"name": "shell", "arguments": {"command": ["ls"]}}}
            ]},
            {"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "call_123"}
        ]

        # This should not raise an exception and should convert properly
        converted = self.provider._convert_messages_for_ollama(messages)

        # Verify the conversion is correct
        assert len(converted) == 3
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
        assert converted[1]["content"] == "I will list the files."
        assert "tool_calls" not in converted[1]
        assert converted[2]["role"] == "user"
        assert converted[2]["content"] == "[TOOL RESULT call_123]: file1.txt\nfile2.txt"

    def test_message_conversion_maintains_message_order(self):
        """Test that message conversion maintains the order of messages"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Request 1"},
            {"role": "assistant", "content": "Response 1", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool1", "arguments": {}}}]},
            {"role": "tool", "content": "Tool result 1", "tool_call_id": "call_1"},
            {"role": "user", "content": "Request 2"},
            {"role": "assistant", "content": "Response 2"},
        ]

        converted = self.provider._convert_messages_for_ollama(messages)

        # Should maintain same number of messages
        assert len(converted) == 6

        # Verify order and roles
        expected_roles = ["system", "user", "assistant", "user", "user", "assistant"]
        actual_roles = [msg["role"] for msg in converted]
        assert actual_roles == expected_roles

        # Verify specific conversions
        assert converted[0]["content"] == "System prompt"  # system unchanged
        assert converted[1]["content"] == "Request 1"      # user unchanged
        assert converted[2]["content"] == "Response 1"     # assistant tool_calls removed
        assert converted[3]["content"] == "[TOOL RESULT call_1]: Tool result 1"  # tool -> user
        assert converted[4]["content"] == "Request 2"      # user unchanged
        assert converted[5]["content"] == "Response 2"     # assistant unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
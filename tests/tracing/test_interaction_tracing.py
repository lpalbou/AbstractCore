"""
Comprehensive tests for interaction tracing functionality.

Tests cover:
- Provider-level tracing
- Session-level tracing
- Trace export utilities
- Trace retrieval and filtering
"""

import pytest
import json
from pathlib import Path
from abstractcore import create_llm
from abstractcore.core.session import BasicSession
from abstractcore.utils import export_traces, summarize_traces


class TestProviderTracing:
    """Tests for provider-level interaction tracing."""

    def test_tracing_disabled_by_default(self):
        """Tracing should be disabled by default."""
        llm = create_llm('ollama', model='qwen3:4b-instruct-2507-q4_K_M')

        assert hasattr(llm, 'enable_tracing')
        assert llm.enable_tracing is False
        assert hasattr(llm, '_traces')
        assert len(llm._traces) == 0

    def test_enable_tracing(self):
        """Test enabling tracing on provider."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        assert llm.enable_tracing is True
        assert hasattr(llm, '_traces')
        assert hasattr(llm, 'get_traces')

    def test_trace_capture(self):
        """Test that traces are captured during generation."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True,
            max_traces=10
        )

        # Generate response
        response = llm.generate(
            "Say 'Hello World' exactly",
            temperature=0
        )

        # Check response has trace_id
        assert hasattr(response, 'metadata')
        assert response.metadata is not None
        assert 'trace_id' in response.metadata

        # Check trace was captured
        traces = llm.get_traces()
        assert len(traces) == 1

        trace = traces[0]
        assert trace['trace_id'] == response.metadata['trace_id']
        assert trace['provider'] == 'OllamaProvider'
        assert trace['model'] == 'qwen3:4b-instruct-2507-q4_K_M'
        assert trace['prompt'] == "Say 'Hello World' exactly"
        assert 'response' in trace
        assert 'content' in trace['response']
        assert trace['parameters']['temperature'] == 0

    def test_trace_retrieval_by_id(self):
        """Test retrieving specific trace by ID."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        response1 = llm.generate("Test 1", temperature=0)
        response2 = llm.generate("Test 2", temperature=0)

        # Retrieve specific trace
        trace_id_1 = response1.metadata['trace_id']
        trace = llm.get_traces(trace_id=trace_id_1)

        assert trace is not None
        assert trace['trace_id'] == trace_id_1
        assert trace['prompt'] == "Test 1"

    def test_trace_retrieval_last_n(self):
        """Test retrieving last N traces."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        # Generate 5 responses
        for i in range(5):
            llm.generate(f"Test {i}", temperature=0)

        # Get last 3 traces
        traces = llm.get_traces(last_n=3)
        assert len(traces) == 3

        # Verify they are the most recent
        assert traces[0]['prompt'] == "Test 2"
        assert traces[1]['prompt'] == "Test 3"
        assert traces[2]['prompt'] == "Test 4"

    def test_trace_ring_buffer(self):
        """Test that trace ring buffer respects max_traces limit."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True,
            max_traces=3  # Small buffer
        )

        # Generate 5 responses (exceeds buffer)
        for i in range(5):
            llm.generate(f"Test {i}", temperature=0)

        # Should only have last 3
        traces = llm.get_traces()
        assert len(traces) == 3
        assert traces[0]['prompt'] == "Test 2"
        assert traces[1]['prompt'] == "Test 3"
        assert traces[2]['prompt'] == "Test 4"

    def test_trace_metadata_custom(self):
        """Test custom trace metadata."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        response = llm.generate(
            "Test with metadata",
            temperature=0,
            trace_metadata={
                'step': 'code_generation',
                'attempt': 1,
                'user_id': 'test_user'
            }
        )

        trace_id = response.metadata['trace_id']
        trace = llm.get_traces(trace_id=trace_id)

        assert trace['metadata']['step'] == 'code_generation'
        assert trace['metadata']['attempt'] == 1
        assert trace['metadata']['user_id'] == 'test_user'

    def test_trace_with_system_prompt(self):
        """Test tracing with system prompt."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        response = llm.generate(
            "Test",
            system_prompt="You are a helpful assistant",
            temperature=0
        )

        trace_id = response.metadata['trace_id']
        trace = llm.get_traces(trace_id=trace_id)

        assert trace['system_prompt'] == "You are a helpful assistant"

    def test_trace_with_messages(self):
        """Test tracing with conversation history."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        response = llm.generate(
            "How are you?",
            messages=messages,
            temperature=0
        )

        trace_id = response.metadata['trace_id']
        trace = llm.get_traces(trace_id=trace_id)

        assert trace['messages'] == messages


class TestSessionTracing:
    """Tests for session-level interaction tracing."""

    def test_session_tracing_disabled_by_default(self):
        """Session tracing should be disabled by default."""
        llm = create_llm('ollama', model='qwen3:4b-instruct-2507-q4_K_M')
        session = BasicSession(provider=llm)

        assert session.enable_tracing is False
        assert len(session.interaction_traces) == 0

    def test_session_enable_tracing(self):
        """Test enabling tracing on session."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )
        session = BasicSession(provider=llm, enable_tracing=True)

        assert session.enable_tracing is True

    def test_session_trace_capture(self):
        """Test that session captures traces from provider."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )
        session = BasicSession(provider=llm, enable_tracing=True)

        # Generate responses
        response1 = session.generate("Test 1", temperature=0)
        response2 = session.generate("Test 2", temperature=0)

        # Check session collected traces
        traces = session.get_interaction_history()
        assert len(traces) == 2

        # Verify trace content
        assert traces[0]['prompt'] == "Test 1"
        assert traces[1]['prompt'] == "Test 2"

        # Verify session metadata was added
        assert traces[0]['metadata']['session_id'] == session.id
        assert traces[0]['metadata']['step_type'] == 'chat'

    def test_session_trace_with_custom_metadata(self):
        """Test session trace with custom metadata."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )
        session = BasicSession(provider=llm, enable_tracing=True)

        response = session.generate(
            "Test",
            temperature=0,
            step_type='code_generation',
            attempt_number=2
        )

        traces = session.get_interaction_history()
        assert len(traces) == 1

        assert traces[0]['metadata']['session_id'] == session.id
        assert traces[0]['metadata']['step_type'] == 'code_generation'
        assert traces[0]['metadata']['attempt_number'] == 2

    def test_session_trace_isolation(self):
        """Test that traces are isolated between sessions."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        session1 = BasicSession(provider=llm, enable_tracing=True)
        session2 = BasicSession(provider=llm, enable_tracing=True)

        session1.generate("Session 1", temperature=0)
        session2.generate("Session 2", temperature=0)

        traces1 = session1.get_interaction_history()
        traces2 = session2.get_interaction_history()

        assert len(traces1) == 1
        assert len(traces2) == 1
        assert traces1[0]['metadata']['session_id'] == session1.id
        assert traces2[0]['metadata']['session_id'] == session2.id


class TestTraceExport:
    """Tests for trace export utilities."""

    def test_export_jsonl(self, tmp_path):
        """Test exporting traces to JSONL format."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        llm.generate("Test 1", temperature=0)
        llm.generate("Test 2", temperature=0)

        traces = llm.get_traces()

        # Export to JSONL
        output_file = tmp_path / "traces.jsonl"
        export_traces(traces, format='jsonl', file_path=output_file)

        # Verify file exists and content
        assert output_file.exists()

        lines = output_file.read_text().strip().split('\n')
        assert len(lines) == 2

        # Each line should be valid JSON
        trace1 = json.loads(lines[0])
        trace2 = json.loads(lines[1])
        assert trace1['prompt'] == "Test 1"
        assert trace2['prompt'] == "Test 2"

    def test_export_json(self, tmp_path):
        """Test exporting traces to JSON format."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        llm.generate("Test", temperature=0)
        traces = llm.get_traces()

        # Export to JSON
        output_file = tmp_path / "traces.json"
        export_traces(traces, format='json', file_path=output_file)

        # Verify file exists and content
        assert output_file.exists()

        loaded_traces = json.loads(output_file.read_text())
        assert len(loaded_traces) == 1
        assert loaded_traces[0]['prompt'] == "Test"

    def test_export_markdown(self, tmp_path):
        """Test exporting traces to markdown format."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        llm.generate("Test", temperature=0)
        traces = llm.get_traces()

        # Export to Markdown
        output_file = tmp_path / "traces.md"
        export_traces(traces, format='markdown', file_path=output_file)

        # Verify file exists and content
        assert output_file.exists()

        content = output_file.read_text()
        assert "# LLM Interaction Trace Report" in content
        assert "## Interaction 1:" in content
        assert "Test" in content
        assert "OllamaProvider" in content

    def test_export_single_trace(self, tmp_path):
        """Test exporting a single trace."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        response = llm.generate("Test", temperature=0)
        trace = llm.get_traces(trace_id=response.metadata['trace_id'])

        # Export single trace
        output_file = tmp_path / "single_trace.json"
        export_traces(trace, format='json', file_path=output_file)

        loaded_traces = json.loads(output_file.read_text())
        assert len(loaded_traces) == 1

    def test_export_as_string(self):
        """Test exporting traces as string (no file)."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        llm.generate("Test", temperature=0)
        traces = llm.get_traces()

        # Export as string
        json_string = export_traces(traces, format='json')
        assert isinstance(json_string, str)

        loaded_traces = json.loads(json_string)
        assert len(loaded_traces) == 1

    def test_summarize_traces(self):
        """Test trace summarization."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        # Generate multiple responses
        for i in range(3):
            llm.generate(f"Test {i}", temperature=0)

        traces = llm.get_traces()
        summary = summarize_traces(traces)

        assert summary['total_interactions'] == 3
        assert summary['total_tokens'] > 0
        assert summary['avg_tokens_per_interaction'] > 0
        assert 'OllamaProvider' in summary['providers']
        assert 'qwen3:4b-instruct-2507-q4_K_M' in summary['models']
        assert summary['date_range'] is not None

    def test_summarize_empty_traces(self):
        """Test summarization of empty traces."""
        summary = summarize_traces([])

        assert summary['total_interactions'] == 0
        assert summary['total_tokens'] == 0
        assert summary['avg_tokens_per_interaction'] == 0
        assert len(summary['providers']) == 0
        assert summary['date_range'] is None


class TestTraceContent:
    """Tests for trace content completeness."""

    def test_trace_contains_all_fields(self):
        """Test that trace contains all expected fields."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        response = llm.generate(
            "Test prompt",
            system_prompt="System",
            temperature=0.7,
            seed=42
        )

        trace_id = response.metadata['trace_id']
        trace = llm.get_traces(trace_id=trace_id)

        # Verify all expected fields exist
        assert 'trace_id' in trace
        assert 'timestamp' in trace
        assert 'provider' in trace
        assert 'model' in trace
        assert 'system_prompt' in trace
        assert 'prompt' in trace
        assert 'parameters' in trace
        assert 'response' in trace
        assert 'metadata' in trace

        # Verify parameters
        assert 'temperature' in trace['parameters']
        assert 'max_tokens' in trace['parameters']
        assert 'max_output_tokens' in trace['parameters']

        # Verify response
        assert 'content' in trace['response']
        assert 'usage' in trace['response']
        assert 'generation_time_ms' in trace['response']
        assert 'finish_reason' in trace['response']

    def test_trace_usage_metrics(self):
        """Test that trace captures usage metrics."""
        llm = create_llm(
            'ollama',
            model='qwen3:4b-instruct-2507-q4_K_M',
            enable_tracing=True
        )

        response = llm.generate("Test", temperature=0)

        trace_id = response.metadata['trace_id']
        trace = llm.get_traces(trace_id=trace_id)

        usage = trace['response']['usage']
        assert usage is not None
        assert 'total_tokens' in usage
        assert usage['total_tokens'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

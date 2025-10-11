"""
BasicSession for conversation tracking.
Target: <500 lines maximum.
"""

from typing import List, Optional, Dict, Any, Union, Iterator, Callable
from datetime import datetime
from pathlib import Path
import json
import uuid
from collections.abc import Generator

from .interface import AbstractLLMInterface
from .types import GenerateResponse, Message
from .enums import MessageRole


class BasicSession:
    """
    Minimal session for conversation management.
    ONLY handles:
    - Message history
    - Basic generation
    - Simple persistence
    """

    def __init__(self,
                 provider: Optional[AbstractLLMInterface] = None,
                 system_prompt: Optional[str] = None,
                 tools: Optional[List[Callable]] = None,
                 timeout: Optional[float] = None,
                 tool_timeout: Optional[float] = None,
                 recovery_timeout: Optional[float] = None,
                 auto_compact: bool = False,
                 auto_compact_threshold: int = 6000):
        """Initialize basic session"""

        self.provider = provider
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.messages: List[Message] = []
        self.system_prompt = system_prompt
        self.tools = self._register_tools(tools) if tools else []
        self.auto_compact = auto_compact
        self.auto_compact_threshold = auto_compact_threshold
        self._original_session = None  # Track if this is a compacted session

        # Apply timeout configurations to provider if specified and provider exists
        if self.provider and hasattr(self.provider, 'set_timeout'):
            if timeout is not None:
                self.provider.set_timeout(timeout)
            if tool_timeout is not None:
                self.provider.set_tool_timeout(tool_timeout)
            if recovery_timeout is not None:
                self.provider.set_recovery_timeout(recovery_timeout)

        # Add system message if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM.value, system_prompt)

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to conversation history"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message

    def get_messages(self) -> List[Message]:
        """Get all messages"""
        return self.messages.copy()

    def get_history(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """Get conversation history as dicts"""
        if include_system:
            return [m.to_dict() for m in self.messages]
        return [m.to_dict() for m in self.messages if m.role != 'system']

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history"""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == 'system']
        else:
            self.messages = []

    def generate(self, prompt: str, **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using provider"""
        if not self.provider:
            raise ValueError("No provider configured")

        # Check for auto-compaction before generating
        if self.auto_compact and self.should_compact(self.auto_compact_threshold):
            print(f"🗜️  Auto-compacting session (tokens: {self.get_token_estimate()} > {self.auto_compact_threshold})")
            compacted = self.compact(reason="auto_threshold")
            # Replace current session with compacted version
            self._replace_with_compacted(compacted)

        # Add user message
        self.add_message('user', prompt)

        # Format messages for provider (exclude the current user message since provider will add it)
        messages = self._format_messages_for_provider_excluding_current()

        # Use session tools if not provided in kwargs
        if 'tools' not in kwargs and self.tools:
            kwargs['tools'] = self.tools

        # Pass session tool_call_tags if available and not overridden in kwargs
        if hasattr(self, 'tool_call_tags') and self.tool_call_tags is not None and 'tool_call_tags' not in kwargs:
            kwargs['tool_call_tags'] = self.tool_call_tags

        # Call provider
        response = self.provider.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=self.system_prompt,
            **kwargs
        )

        # Handle streaming vs non-streaming responses
        if isinstance(response, (Generator, Iterator)) or hasattr(response, '__iter__') and not hasattr(response, 'content'):
            # Streaming response - wrap it to collect content
            return self._handle_streaming_response(response)
        else:
            # Non-streaming response
            if hasattr(response, 'content') and response.content:
                self.add_message('assistant', response.content)
            return response

    def _handle_streaming_response(self, response_iterator: Iterator[GenerateResponse]) -> Iterator[GenerateResponse]:
        """Handle streaming response and collect content for history"""
        collected_content = ""

        for chunk in response_iterator:
            # Yield the chunk for the caller
            yield chunk

            # Collect content for history
            if hasattr(chunk, 'content') and chunk.content:
                collected_content += chunk.content

        # After streaming is complete, add the full response to history
        if collected_content:
            self.add_message('assistant', collected_content)

    def _format_messages_for_provider(self) -> List[Dict[str, str]]:
        """Format messages for provider API"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
            if m.role != 'system'  # System handled separately
        ]

    def _format_messages_for_provider_excluding_current(self) -> List[Dict[str, str]]:
        """Format messages for provider API, excluding the most recent user message"""
        # Exclude the last message if it's a user message (the current prompt)
        messages_to_send = self.messages[:-1] if self.messages and self.messages[-1].role == 'user' else self.messages
        return [
            {"role": m.role, "content": m.content}
            for m in messages_to_send
            if m.role != 'system'  # System handled separately
        ]

    def save(self, filepath: Union[str, Path]):
        """Save session to file"""
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BasicSession':
        """Load session from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        session = cls(system_prompt=data.get("system_prompt"))
        session.id = data["id"]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.messages = [Message.from_dict(m) for m in data["messages"]]

        return session

    def _register_tools(self, tools: List[Callable]) -> List:
        """Register tools and return tool definitions"""
        from ..tools import ToolDefinition, register_tool

        registered_tools = []
        for func in tools:
            tool_def = ToolDefinition.from_function(func)
            register_tool(tool_def)
            registered_tools.append(tool_def)
        return registered_tools

    # Timeout management methods
    def get_timeout(self) -> Optional[float]:
        """Get the current HTTP request timeout in seconds."""
        if self.provider and hasattr(self.provider, 'get_timeout'):
            return self.provider.get_timeout()
        return None

    def set_timeout(self, timeout: float) -> None:
        """Set the HTTP request timeout in seconds."""
        if self.provider and hasattr(self.provider, 'set_timeout'):
            self.provider.set_timeout(timeout)

    def get_recovery_timeout(self) -> Optional[float]:
        """Get the current circuit breaker recovery timeout in seconds."""
        if self.provider and hasattr(self.provider, 'get_recovery_timeout'):
            return self.provider.get_recovery_timeout()
        return None

    def set_recovery_timeout(self, timeout: float) -> None:
        """Set the circuit breaker recovery timeout in seconds."""
        if self.provider and hasattr(self.provider, 'set_recovery_timeout'):
            self.provider.set_recovery_timeout(timeout)

    def get_tool_timeout(self) -> Optional[float]:
        """Get the current tool execution timeout in seconds."""
        if self.provider and hasattr(self.provider, 'get_tool_timeout'):
            return self.provider.get_tool_timeout()
        return None

    def set_tool_timeout(self, timeout: float) -> None:
        """Set the tool execution timeout in seconds."""
        if self.provider and hasattr(self.provider, 'set_tool_timeout'):
            self.provider.set_tool_timeout(timeout)

    def compact(self,
                preserve_recent: int = 6,
                focus: Optional[str] = None,
                compact_provider: Optional[AbstractLLMInterface] = None,
                reason: str = "manual") -> 'BasicSession':
        """
        Compact chat history using SOTA 2025 best practices for conversation summarization.

        Creates a new session with compacted history that preserves:
        - System messages (always preserved)
        - Recent conversation exchanges (last N messages)
        - Summarized context of older messages

        Args:
            preserve_recent: Number of recent message pairs to keep intact (default 6)
            focus: Optional focus for summarization (e.g., "technical decisions", "key solutions")
            compact_provider: Optional separate provider for compaction (uses session provider if None)
            reason: Reason for compaction ("manual", "auto", "token_limit") for event tracking

        Returns:
            BasicSession: New session with compacted history

        SOTA Best Practices Implemented (2025):
        - Preserves system messages to maintain essential context
        - Uses sliding window approach (recent messages + summarized older context)
        - Maintains conversational flow and user intent
        - Focuses on decisions, solutions, and ongoing topics
        - Optimized for conversation continuity rather than standalone summary

        Example:
            >>> session = BasicSession(provider, system_prompt="You are a helpful assistant")
            >>> # ... many exchanges ...
            >>> compacted = session.compact(preserve_recent=8, focus="technical solutions")
            >>> # New session has same context but much shorter history
        """
        if not self.messages:
            # No messages to compact, return copy of current session
            return self._create_session_copy()

        # Import events system
        try:
            from ..events import EventType, emit_global
        except ImportError:
            emit_global = None

        # Emit compaction start event
        start_time = datetime.now()
        original_tokens = self.get_token_estimate()

        if emit_global:
            emit_global(EventType.COMPACTION_STARTED, {
                'session_id': self.id,
                'reason': reason,
                'original_message_count': len(self.messages),
                'original_tokens_estimate': original_tokens,
                'preserve_recent': preserve_recent,
                'focus': focus or "general conversation context",
                'compact_provider': compact_provider.__class__.__name__ if compact_provider else None
            }, source='BasicSession')

        # Use compact provider or fall back to session provider
        summarizer_provider = compact_provider or self.provider
        if not summarizer_provider:
            raise ValueError("No provider available for compaction")

        # Import here to avoid circular dependency
        try:
            from ..processing import BasicSummarizer
        except ImportError:
            raise ImportError("BasicSummarizer not available. This suggests a packaging issue - processing module should be included with AbstractCore.")

        # Separate system messages from conversation messages
        system_messages = [msg for msg in self.messages if msg.role == 'system']
        conversation_messages = [msg for msg in self.messages if msg.role != 'system']

        if len(conversation_messages) <= preserve_recent:
            # Short conversation, no compaction needed
            return self._create_session_copy()

        # Create summarizer and compact the conversation history
        summarizer = BasicSummarizer(summarizer_provider)

        # Convert messages to dict format for summarizer
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_messages]

        # Get compacted summary
        summary_result = summarizer.summarize_chat_history(
            messages=message_dicts,
            preserve_recent=preserve_recent,
            focus=focus
        )

        # Create new session with compacted history (no auto system prompt to avoid duplicates)
        new_session = BasicSession(
            provider=self.provider,
            system_prompt=None,  # We'll add system messages manually to avoid duplicates
            tools=None,  # Don't copy tools registration, just reference
        )

        # Copy session metadata
        new_session.id = self.id + "_compacted"
        new_session.created_at = self.created_at
        new_session.tools = self.tools  # Reference same tools
        new_session.system_prompt = self.system_prompt  # Set for reference but don't auto-add message

        # Add existing system messages (preserves original structure)
        for sys_msg in system_messages:
            new_session.add_message(sys_msg.role, sys_msg.content)

        # Add compacted summary as a special system message for context
        if len(conversation_messages) > preserve_recent:
            new_session.add_message(
                'system',
                f"[CONVERSATION HISTORY]: {summary_result.summary}"
            )

        # Add preserved recent messages back to the session
        recent_messages = conversation_messages[-preserve_recent:] if preserve_recent > 0 else []
        for msg in recent_messages:
            new_session.add_message(msg.role, msg.content)

        # Emit compaction completion event
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        final_tokens = new_session.get_token_estimate()

        if emit_global:
            emit_global(EventType.COMPACTION_COMPLETED, {
                'session_id': self.id,
                'compacted_session_id': new_session.id,
                'reason': reason,
                'duration_ms': duration_ms,
                'original_message_count': len(self.messages),
                'compacted_message_count': len(new_session.messages),
                'original_tokens_estimate': original_tokens,
                'compacted_tokens_estimate': final_tokens,
                'compression_ratio': original_tokens / final_tokens if final_tokens > 0 else 1.0,
                'messages_preserved': preserve_recent,
                'focus_used': focus or "general conversation context",
                'success': True
            }, source='BasicSession')

        return new_session

    def _create_session_copy(self) -> 'BasicSession':
        """Create a copy of the current session"""
        new_session = BasicSession(
            provider=self.provider,
            system_prompt=None,  # We'll copy messages manually to avoid duplicates
            tools=None,  # Don't re-register tools
        )

        new_session.id = self.id + "_copy"
        new_session.created_at = self.created_at
        new_session.tools = self.tools  # Reference same tools
        new_session.system_prompt = self.system_prompt  # Set for reference but don't auto-add message

        # Copy all messages exactly as they are
        for msg in self.messages:
            new_session.add_message(msg.role, msg.content)

        return new_session

    def get_token_estimate(self) -> int:
        """
        Estimate token count for current conversation.
        Rough estimate: 1 token ≈ 4 characters for English text.
        """
        total_chars = sum(len(msg.content) for msg in self.messages)
        return total_chars // 4

    def should_compact(self, token_limit: int = 8000) -> bool:
        """
        Check if session should be compacted based on estimated token count.

        Args:
            token_limit: Maximum tokens before suggesting compaction (default 8000)

        Returns:
            bool: True if session exceeds token limit and should be compacted
        """
        return self.get_token_estimate() > token_limit

    def _replace_with_compacted(self, compacted_session: 'BasicSession') -> None:
        """Replace current session content with compacted version (in-place)"""
        # Store reference to original session
        self._original_session = self.id

        # Replace session content
        self.messages = compacted_session.messages
        self.id = compacted_session.id

        # Preserve auto-compact settings in the compacted session
        compacted_session.auto_compact = self.auto_compact
        compacted_session.auto_compact_threshold = self.auto_compact_threshold

    def enable_auto_compact(self, threshold: int = 6000) -> None:
        """Enable automatic compaction when token count exceeds threshold"""
        self.auto_compact = True
        self.auto_compact_threshold = threshold

    def disable_auto_compact(self) -> None:
        """Disable automatic compaction"""
        self.auto_compact = False

    def force_compact(self,
                     preserve_recent: int = 6,
                     focus: Optional[str] = None) -> None:
        """
        Force immediate compaction of the session (in-place).

        User-facing interface for explicit compaction requests.

        Args:
            preserve_recent: Number of recent messages to preserve
            focus: Focus for summarization

        Example:
            >>> session.force_compact(preserve_recent=8, focus="key decisions")
            >>> # Session is now compacted in-place
        """
        if not self.messages:
            return

        compacted = self.compact(
            preserve_recent=preserve_recent,
            focus=focus,
            reason="user_requested"
        )

        # Replace current session with compacted version
        self._replace_with_compacted(compacted)

        print(f"✅ Session compacted: {len(compacted.messages)} messages, ~{compacted.get_token_estimate()} tokens")
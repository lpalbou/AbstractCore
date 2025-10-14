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
        
        # Optional analytics fields
        self.summary = None
        self.assessment = None
        self.facts = None

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

    def add_message(self, role: str, content: str, name: Optional[str] = None, 
                   location: Optional[str] = None, **metadata_kwargs) -> Message:
        """
        Add a message to conversation history manually.
        
        Use this method when you need to:
        - Add system messages
        - Add tool messages  
        - Manually construct conversation history
        - Add messages without triggering LLM generation
        
        For normal user interactions, use generate() instead.
        
        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            name: Username for the message. Defaults to "user" for user messages, None for others
            location: Location information (geographical, contextual, etc.)
            **metadata_kwargs: Additional metadata fields
        
        Returns:
            Message: The created message with timestamp and metadata
        
        Example:
            # Add a system message
            session.add_message('system', 'You are a helpful assistant.')
            
            # Add a user message manually (without LLM response)
            session.add_message('user', 'Hello!', name='alice')
            
            # For normal chat, use generate() instead:
            # response = session.generate('Hello!', name='alice')
        """
        # Set default username for user messages if not specified
        if name is None and role == 'user':
            name = "user"
        
        # Build metadata
        metadata = {}
        if name is not None:
            metadata['name'] = name
        if location is not None:
            metadata['location'] = location
        
        # Add any additional metadata
        metadata.update(metadata_kwargs)
            
        message = Message(role=role, content=content, metadata=metadata if metadata else None)
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

    def generate(self, prompt: str, name: Optional[str] = None, 
                location: Optional[str] = None, **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate LLM response for user input (recommended method for normal chat).
        
        This is the main method for conversational interactions. It:
        1. Adds your message to conversation history
        2. Calls the LLM provider to generate a response
        3. Adds the assistant's response to history
        4. Returns the response
        
        Args:
            prompt: User input prompt
            name: Optional username for the message (defaults to "user")
            location: Optional location information for the message
            **kwargs: Additional arguments passed to provider (stream, tools, etc.)
            
        Returns:
            GenerateResponse or Iterator[GenerateResponse]: Response from the provider
            
        Example:
            # Normal chat interaction
            response = session.generate('What is Python?', name='alice')
            
            # With location context
            response = session.generate('What time is it?', name='bob', location='Paris')
            
            # With streaming
            for chunk in session.generate('Tell me a story', stream=True):
                print(chunk.content, end='')
        """
        if not self.provider:
            raise ValueError("No provider configured")

        # Check for auto-compaction before generating
        if self.auto_compact and self.should_compact(self.auto_compact_threshold):
            print(f"🗜️  Auto-compacting session (tokens: {self.get_token_estimate()} > {self.auto_compact_threshold})")
            compacted = self.compact(reason="auto_threshold")
            # Replace current session with compacted version
            self._replace_with_compacted(compacted)

        # Add user message with optional custom name and location
        self.add_message('user', prompt, name=name, location=location)

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
        """
        Save session to file with complete metadata preservation.
        
        Args:
            filepath: Path to save the session file
            
        Note:
            Provider and tools are not serialized as they may contain non-serializable
            elements. They should be re-registered when loading the session.
        """
        data = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path], provider: Optional[AbstractLLMInterface] = None,
             tools: Optional[List[Callable]] = None) -> 'BasicSession':
        """
        Load session from file with complete metadata restoration.
        
        Args:
            filepath: Path to the session file
            provider: LLM provider to use (must be provided separately)
            tools: Tools to register (must be provided separately)
            
        Returns:
            BasicSession: Loaded session with all metadata preserved
            
        Note:
            Provider and tools must be provided separately as they are not serialized.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data, provider=provider, tools=tools)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary using the session-archive/v1 schema.
        
        Returns:
            Dict containing complete session archive with versioned schema
        """
        # Build tool registry (declarative schemas only)
        tool_registry = []
        if self.tools:
            for tool in self.tools:
                tool_entry = {"name": tool.name}
                if hasattr(tool, 'description') and tool.description:
                    tool_entry["description"] = tool.description
                if hasattr(tool, 'json_schema') and tool.json_schema:
                    tool_entry["json_schema"] = tool.json_schema
                tool_registry.append(tool_entry)
        
        # Build session object
        session_data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "provider": self.provider.__class__.__name__.replace('Provider', '').lower() if self.provider else None,
            "model": getattr(self.provider, 'model', None) if self.provider else None,
            "model_params": getattr(self.provider, 'model_params', None) if self.provider else None,
            "system_prompt": self.system_prompt,
            "tool_registry": tool_registry if tool_registry else None,
            "settings": {
                "auto_compact": self.auto_compact,
                "auto_compact_threshold": self.auto_compact_threshold
            }
        }
        
        # Add optional analytics if present
        if self.summary:
            session_data["summary"] = self.summary
        if self.assessment:
            session_data["assessment"] = self.assessment
        if self.facts:
            session_data["facts"] = self.facts
        
        # Return complete archive
        return {
            "schema_version": "session-archive/v1",
            "session": session_data,
            "messages": [m.to_dict() for m in self.messages]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], provider: Optional[AbstractLLMInterface] = None, 
                  tools: Optional[List[Callable]] = None) -> 'BasicSession':
        """
        Create session from dictionary data (supports both new archive format and legacy format).
        
        Args:
            data: Dictionary containing session data
            provider: LLM provider to use (must be provided separately)
            tools: Tools to register (must be provided separately)
            
        Returns:
            BasicSession: Reconstructed session
        """
        # Detect format: new archive format has schema_version and nested session
        if "schema_version" in data and "session" in data:
            # New archive format
            session_data = data["session"]
            messages_data = data.get("messages", [])
        else:
            # Legacy format - data is the session object directly
            session_data = data
            messages_data = data.get("messages", [])
        
        # Extract settings
        settings = session_data.get("settings", {})
        auto_compact = settings.get("auto_compact", session_data.get("auto_compact", False))
        auto_compact_threshold = settings.get("auto_compact_threshold", session_data.get("auto_compact_threshold", 6000))
        
        # Create session with basic parameters
        session = cls(
            provider=provider,
            system_prompt=None,  # We'll restore messages manually to avoid duplicates
            tools=tools,
            auto_compact=auto_compact,
            auto_compact_threshold=auto_compact_threshold
        )
        
        # Restore session metadata
        session.id = session_data["id"]
        session.created_at = datetime.fromisoformat(session_data["created_at"])
        session.system_prompt = session_data.get("system_prompt")
        session._original_session = session_data.get("original_session")
        
        # Restore optional analytics
        session.summary = session_data.get("summary")
        session.assessment = session_data.get("assessment")
        session.facts = session_data.get("facts")
        
        # Clear any auto-added messages and restore from data
        session.messages = []
        for msg_data in messages_data:
            message = Message.from_dict(msg_data)
            session.messages.append(message)
            
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
        Estimate token count for current conversation using centralized utilities.
        
        Returns:
            Estimated total token count for all messages in the session
        """
        from ..utils.token_utils import TokenUtils
        
        # Get model name from provider if available
        model_name = None
        if self.provider and hasattr(self.provider, 'model'):
            model_name = self.provider.model
            
        # Calculate tokens for all messages
        total_tokens = 0
        for msg in self.messages:
            total_tokens += TokenUtils.estimate_tokens(msg.content, model_name)
            
        return total_tokens

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

    def generate_summary(self, preserve_recent: int = 6, focus: Optional[str] = None, 
                        compact_provider: Optional[AbstractLLMInterface] = None) -> Dict[str, Any]:
        """
        Generate a summary of the entire conversation and store it in session.summary.
        
        Args:
            preserve_recent: Number of recent messages to preserve in analysis
            focus: Optional focus for summarization
            compact_provider: Optional provider for summarization
            
        Returns:
            Dict containing the generated summary
        """
        if not self.messages:
            return {}
            
        start_time = datetime.now()
        original_tokens = self.get_token_estimate()
        
        # Use compact provider or fall back to session provider
        summarizer_provider = compact_provider or self.provider
        if not summarizer_provider:
            raise ValueError("No provider available for summarization")
        
        try:
            from ..processing import BasicSummarizer
        except ImportError:
            raise ImportError("BasicSummarizer not available")
        
        # Create summarizer
        summarizer = BasicSummarizer(summarizer_provider)
        
        # Convert messages to dict format for summarizer
        conversation_messages = [msg for msg in self.messages if msg.role != 'system']
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in conversation_messages]
        
        # Generate summary
        summary_result = summarizer.summarize_chat_history(
            messages=message_dicts,
            preserve_recent=preserve_recent,
            focus=focus
        )
        
        # Calculate metrics
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Store summary in session
        self.summary = {
            "created_at": start_time.isoformat(),
            "preserve_recent": preserve_recent,
            "focus": focus,
            "text": summary_result.summary,
            "metrics": {
                "tokens_before": original_tokens,
                "tokens_after": self._estimate_tokens_for_summary(summary_result.summary),
                "compression_ratio": self._calculate_compression_ratio(original_tokens, summary_result.summary),
                "generation_time_ms": duration_ms
            }
        }
        
        return self.summary

    def _estimate_tokens_for_summary(self, summary_text: Optional[str]) -> int:
        """Helper method to estimate tokens for summary text."""
        if not summary_text:
            return 0
        from ..utils.token_utils import TokenUtils
        # Get model name from provider if available, otherwise use default
        model_name = None
        if self.provider and hasattr(self.provider, 'model'):
            model_name = self.provider.model
        return TokenUtils.estimate_tokens(summary_text, model_name)
    
    def _calculate_compression_ratio(self, original_tokens: int, summary_text: Optional[str]) -> float:
        """Helper method to calculate compression ratio."""
        if not summary_text:
            return 1.0
        summary_tokens = self._estimate_tokens_for_summary(summary_text)
        return original_tokens / summary_tokens if summary_tokens > 0 else 1.0

    def generate_assessment(self, criteria: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Generate a quality assessment of the entire conversation and store it in session.assessment.
        
        Args:
            criteria: Optional criteria for assessment
            
        Returns:
            Dict containing the generated assessment
        """
        if not self.messages:
            return {}
            
        start_time = datetime.now()
        
        if not self.provider:
            raise ValueError("No provider available for assessment")
        
        try:
            from ..processing import BasicJudge
            from ..processing.basic_judge import JudgmentCriteria
        except ImportError:
            raise ImportError("BasicJudge not available")
        
        # Default criteria if not provided
        if criteria is None:
            criteria = {
                "clarity": True,
                "coherence": True,
                "relevance": True,
                "completeness": True,
                "actionability": True
            }
        
        # Create judge
        judge = BasicJudge(self.provider)
        
        # Format conversation for assessment
        conversation_messages = [msg for msg in self.messages if msg.role != 'system']
        conversation_text = "\n\n".join([
            f"{msg.role.title()}: {msg.content}" for msg in conversation_messages
        ])
        
        # Create criteria object
        judge_criteria = JudgmentCriteria(**criteria)
        
        # Generate assessment
        assessment_result = judge.evaluate(
            content=conversation_text,
            context="conversation quality assessment",
            criteria=judge_criteria
        )
        
        # Store assessment in session
        self.assessment = {
            "created_at": start_time.isoformat(),
            "criteria": criteria,
            "overall_score": assessment_result.get('overall_score', 0),
            "judge_summary": assessment_result.get('judge_summary', ''),
            "strengths": assessment_result.get('strengths', []),
            "actionable_feedback": assessment_result.get('actionable_feedback', []),
            "reasoning": assessment_result.get('reasoning', '')
        }
        
        return self.assessment

    def extract_facts(self, output_format: str = "triples") -> Dict[str, Any]:
        """
        Extract facts from the entire conversation and store them in session.facts.
        
        Args:
            output_format: Format for fact extraction ("triples" or "jsonld")
            
        Returns:
            Dict containing the extracted facts
        """
        if not self.messages:
            return {}
            
        start_time = datetime.now()
        
        if not self.provider:
            raise ValueError("No provider available for fact extraction")
        
        try:
            from ..processing import BasicExtractor
        except ImportError:
            raise ImportError("BasicExtractor not available")
        
        # Create extractor
        extractor = BasicExtractor(self.provider)
        
        # Format conversation for extraction
        conversation_messages = [msg for msg in self.messages if msg.role != 'system']
        conversation_text = "\n\n".join([
            f"{msg.role.title()}: {msg.content}" for msg in conversation_messages
        ])
        
        # Extract facts
        extraction_result = extractor.extract(conversation_text, output_format=output_format)
        
        # Store facts in session
        self.facts = {
            "extracted_at": start_time.isoformat(),
            "simple_triples": extraction_result.get("simple_triples", []),
            "jsonld": extraction_result.get("@graph") if output_format == "jsonld" else None,
            "statistics": extraction_result.get("statistics", {})
        }
        
        return self.facts
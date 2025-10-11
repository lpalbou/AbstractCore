"""
Unified streaming processor with incremental tool detection.

This module provides a single streaming strategy that handles tools elegantly
while maintaining real-time streaming performance.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from enum import Enum

from ..core.types import GenerateResponse
from ..tools.core import ToolCall

logger = logging.getLogger(__name__)


class ToolDetectionState(Enum):
    """States for incremental tool detection."""
    SCANNING = "scanning"           # Looking for tool call patterns
    IN_TOOL_CALL = "in_tool_call"  # Found start tag, collecting content
    TOOL_COMPLETE = "tool_complete"  # Complete tool call detected


class IncrementalToolDetector:
    """
    Incremental tool call detector for streaming content.

    This detector works with partial content and can identify complete tool calls
    as they stream in, enabling immediate tool execution without buffering.
    """

    def __init__(self, model_name: Optional[str] = None, rewrite_tags: bool = False):
        """
        Initialize detector with model-specific patterns.

        Args:
            model_name: Model name for pattern selection
            rewrite_tags: If True, preserve tool calls in output for rewriting
        """
        self.model_name = model_name
        self.rewrite_tags = rewrite_tags
        self.reset()

        # Define patterns for different tool call formats
        self.patterns = {
            'qwen': {
                'start': r'<\|tool_call\|>',
                'end': r'</\|tool_call\|>',
                'json_required': True
            },
            'llama': {
                'start': r'<function_call>',
                'end': r'</function_call>',
                'json_required': True
            },
            'xml': {
                'start': r'<tool_call>',
                'end': r'</tool_call>',
                'json_required': True
            },
            'gemma': {
                'start': r'```tool_code',
                'end': r'```',
                'json_required': True
            }
        }

        # Select pattern based on model
        self.active_patterns = self._get_patterns_for_model(model_name)

    def reset(self):
        """Reset detector state."""
        self.state = ToolDetectionState.SCANNING
        self.accumulated_content = ""
        self.current_tool_content = ""
        self.tool_start_pos = None
        self.current_pattern = None
        self.completed_tools = []

    def _get_patterns_for_model(self, model_name: str) -> List[Dict]:
        """Get relevant patterns for a model."""
        if not model_name:
            return list(self.patterns.values())

        model_lower = model_name.lower()

        # Map models to patterns - be more flexible since models can use different formats
        if 'gemma' in model_lower:
            return [self.patterns['gemma']]
        elif 'llama' in model_lower:
            # LLaMA models typically use function_call format
            return [self.patterns['llama'], self.patterns['xml']]
        else:
            # For qwen, glm, and unknown models, try multiple patterns
            # since they can vary in their tool call formats
            return [
                self.patterns['qwen'],    # <|tool_call|> format
                self.patterns['llama'],   # <function_call> format
                self.patterns['xml']      # <tool_call> format
            ]

    def process_chunk(self, chunk_content: str) -> Tuple[str, List[ToolCall]]:
        """
        Process a content chunk and detect any complete tool calls.

        Args:
            chunk_content: New content chunk

        Returns:
            Tuple of (streamable_content, completed_tool_calls)
        """
        if not chunk_content:
            return "", []

        self.accumulated_content += chunk_content
        completed_tools = []
        streamable_content = ""

        # Process content based on current state
        if self.state == ToolDetectionState.SCANNING:
            streamable_content, completed_tools = self._scan_for_tool_start(chunk_content)
        elif self.state == ToolDetectionState.IN_TOOL_CALL:
            streamable_content, completed_tools = self._collect_tool_content(chunk_content)

        return streamable_content, completed_tools

    def _scan_for_tool_start(self, chunk_content: str) -> Tuple[str, List[ToolCall]]:
        """Scan for tool call start patterns."""
        streamable_content = ""
        completed_tools = []

        # Check for tool start patterns
        for pattern_info in self.active_patterns:
            start_pattern = pattern_info['start']
            match = re.search(start_pattern, self.accumulated_content, re.IGNORECASE)

            if match:
                # Found tool start
                self.tool_start_pos = match.start()
                self.state = ToolDetectionState.IN_TOOL_CALL
                self.current_pattern = pattern_info

                # Return content before tool start as streamable
                streamable_content = self.accumulated_content[:self.tool_start_pos]

                # Start collecting tool content
                self.current_tool_content = self.accumulated_content[match.end():]

                logger.debug(f"Tool call start detected: {start_pattern} for model {self.model_name}")
                logger.debug(f"Accumulated content: {repr(self.accumulated_content[:100])}")

                # Immediately check if tool is already complete (if end tag is in current content)
                # This handles the case where entire tool call comes in one chunk
                additional_streamable, additional_tools = self._collect_tool_content("")
                streamable_content += additional_streamable
                completed_tools.extend(additional_tools)
                break
        else:
            # No tool start found
            # Check if accumulated content might contain partial tool tag at the end
            # Look for partial matches of tool start patterns
            might_be_partial_tag = False

            # Check last 20 chars for potential partial match
            tail = self.accumulated_content[-20:] if len(self.accumulated_content) > 20 else self.accumulated_content

            # Look for opening angle brackets or special chars that could start a tool tag
            # Check both at end and within the tail
            tag_starters = ('<', '<|', '</', '<|t', '<|to', '<|too', '<|tool', '<function', '<tool', '``', '```', '```t')
            for starter in tag_starters:
                if starter in tail:
                    might_be_partial_tag = True
                    break

            if might_be_partial_tag:
                # Found potential partial tag
                if len(self.accumulated_content) > 20:
                    # Keep last 20 chars as buffer, stream the rest
                    streamable_content = self.accumulated_content[:-20]
                    self.accumulated_content = self.accumulated_content[-20:]
                # else: Keep all content buffered (it's < 20 chars and might be incomplete tag)
            elif len(self.accumulated_content) > 0:
                # No partial tag detected, stream everything
                streamable_content = self.accumulated_content
                self.accumulated_content = ""

        return streamable_content, completed_tools

    def _collect_tool_content(self, chunk_content: str) -> Tuple[str, List[ToolCall]]:
        """Collect content for current tool call."""
        streamable_content = ""
        completed_tools = []

        # Add new content to tool content
        self.current_tool_content += chunk_content

        # Check for tool end pattern
        end_pattern = self.current_pattern['end']
        end_match = re.search(end_pattern, self.current_tool_content, re.IGNORECASE)

        if end_match:
            # Tool call is complete
            tool_json_content = self.current_tool_content[:end_match.start()].strip()

            # Try to parse the tool call
            tool_call = self._parse_tool_json(tool_json_content)
            if tool_call:
                completed_tools.append(tool_call)
                logger.debug(f"Complete tool call parsed: {tool_call.name}")

            # Reset for next tool
            remaining_content = self.current_tool_content[end_match.end():]
            self.reset()

            # Continue processing remaining content
            if remaining_content:
                self.accumulated_content = remaining_content
                additional_streamable, additional_tools = self._scan_for_tool_start(remaining_content)
                streamable_content += additional_streamable
                completed_tools.extend(additional_tools)
        # else: Tool content is still accumulating, wait for end tag
        # Do NOT try to parse incomplete JSON here - only in finalize()

        return streamable_content, completed_tools

    def _parse_tool_json(self, json_content: str) -> Optional[ToolCall]:
        """Parse JSON content to create ToolCall."""
        if not json_content or not json_content.strip():
            return None

        try:
            # Clean up common JSON issues in LLM output
            cleaned_json = json_content.strip()

            # Handle missing braces
            if cleaned_json.count('{') > cleaned_json.count('}'):
                missing_braces = cleaned_json.count('{') - cleaned_json.count('}')
                cleaned_json += '}' * missing_braces

            # Parse JSON
            tool_data = json.loads(cleaned_json)

            if isinstance(tool_data, dict) and "name" in tool_data:
                return ToolCall(
                    name=tool_data["name"],
                    arguments=tool_data.get("arguments", {}),
                    call_id=tool_data.get("id")
                )
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}, content: {repr(json_content)}")

        return None

    def _try_parse_incomplete_json(self, content: str) -> Optional[ToolCall]:
        """Try to parse potentially incomplete JSON by finding valid JSON objects."""
        # Look for complete JSON objects within the content
        brace_count = 0
        json_start = -1

        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start >= 0:
                    # Found complete JSON object
                    json_content = content[json_start:i+1]
                    tool_call = self._parse_tool_json(json_content)
                    if tool_call:
                        return tool_call

        return None

    def finalize(self) -> List[ToolCall]:
        """Finalize and return any remaining tool calls."""
        completed_tools = []

        if self.state == ToolDetectionState.IN_TOOL_CALL:
            # Try to parse any remaining content
            tool_call = self._try_parse_incomplete_json(self.current_tool_content)
            if tool_call:
                completed_tools.append(tool_call)

        return completed_tools


class UnifiedStreamProcessor:
    """
    Unified streaming processor that handles tools elegantly during streaming.

    This processor maintains a single streaming path while detecting and executing
    tools incrementally as they become available.
    """

    def __init__(self, model_name: str, execute_tools: bool = True,
                 tool_call_tags: Optional[str] = None):
        """Initialize the stream processor."""
        self.model_name = model_name
        self.execute_tools = execute_tools
        self.tool_call_tags = tool_call_tags
        self.detector = IncrementalToolDetector(model_name)

        # Initialize tag rewriter if custom tags are provided
        self.tag_rewriter = None
        self.tag_rewrite_buffer = ""
        if tool_call_tags:
            self._initialize_tag_rewriter(tool_call_tags)

    def process_stream(self, response_stream: Iterator[GenerateResponse],
                      converted_tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """
        Process a response stream with incremental tool detection and execution.

        Args:
            response_stream: Iterator of response chunks
            converted_tools: Available tools for execution

        Yields:
            GenerateResponse: Processed chunks with tool execution
        """
        try:
            for chunk in response_stream:
                if not chunk.content:
                    yield chunk
                    continue

                # Apply tag rewriting if needed
                processed_content = self._apply_tag_rewriting(chunk.content)

                # Process chunk through incremental detector
                streamable_content, completed_tools = self.detector.process_chunk(processed_content)

                # Yield streamable content immediately
                if streamable_content:
                    yield GenerateResponse(
                        content=streamable_content,
                        model=chunk.model,
                        finish_reason=chunk.finish_reason,
                        usage=chunk.usage,
                        raw_response=chunk.raw_response
                    )

                # Execute completed tools immediately
                if completed_tools and self.execute_tools and converted_tools:
                    logger.debug(f"Executing {len(completed_tools)} tools immediately")
                    tool_results = self._execute_tools_immediately(completed_tools, converted_tools)

                    # Yield tool results as chunks
                    if tool_results:
                        logger.debug(f"Yielding tool results: {tool_results[:100]}...")
                        yield GenerateResponse(
                            content=tool_results,
                            model=chunk.model,
                            finish_reason=chunk.finish_reason,
                            usage=chunk.usage,
                            raw_response=chunk.raw_response
                        )
                elif completed_tools:
                    logger.debug(f"Tools detected but not executing: execute_tools={self.execute_tools}, converted_tools={bool(converted_tools)}")
                elif self.execute_tools and converted_tools:
                    logger.debug("Execute tools enabled and tools available, but no completed tools detected")

            # Finalize any remaining tool calls
            final_tools = self.detector.finalize()
            if final_tools and self.execute_tools and converted_tools:
                tool_results = self._execute_tools_immediately(final_tools, converted_tools)
                if tool_results:
                    yield GenerateResponse(
                        content=tool_results,
                        model=self.model_name,
                        finish_reason="stop"
                    )

        except Exception as e:
            logger.error(f"Error in unified stream processing: {e}")
            raise

    def _initialize_tag_rewriter(self, tool_call_tags):
        """Initialize the tag rewriter from tool_call_tags configuration."""
        try:
            from ..tools.tag_rewriter import ToolCallTagRewriter, ToolCallTags

            if isinstance(tool_call_tags, str):
                # Parse string format: either "start,end" or just "start"
                if ',' in tool_call_tags:
                    parts = tool_call_tags.split(',')
                    if len(parts) == 2:
                        tags = ToolCallTags(
                            start_tag=parts[0].strip(),
                            end_tag=parts[1].strip()
                        )
                    else:
                        logger.warning(f"Invalid tool_call_tags format: {tool_call_tags}")
                        return
                else:
                    # Single tag - assume same for start and end
                    tags = ToolCallTags(
                        start_tag=tool_call_tags.strip(),
                        end_tag=tool_call_tags.strip()
                    )
                self.tag_rewriter = ToolCallTagRewriter(tags)
            elif isinstance(tool_call_tags, ToolCallTags):
                self.tag_rewriter = ToolCallTagRewriter(tool_call_tags)
            elif isinstance(tool_call_tags, ToolCallTagRewriter):
                self.tag_rewriter = tool_call_tags
            else:
                logger.warning(f"Unknown tool_call_tags type: {type(tool_call_tags)}")

        except Exception as e:
            logger.error(f"Failed to initialize tag rewriter: {e}")

    def _apply_tag_rewriting(self, content: str) -> str:
        """Apply tool call tag rewriting using streaming rewriter."""
        if not self.tag_rewriter or not content:
            return content

        try:
            # Use streaming rewriter with buffer for handling split tool calls
            rewritten_content, self.tag_rewrite_buffer = self.tag_rewriter.rewrite_streaming_chunk(
                content, self.tag_rewrite_buffer
            )
            return rewritten_content

        except Exception as e:
            logger.debug(f"Tag rewriting failed: {e}")
            return content

    def _execute_tools_immediately(self, tool_calls: List[ToolCall],
                                  converted_tools: List[Dict[str, Any]]) -> str:
        """Execute tools immediately and return formatted results."""
        try:
            from ..tools import execute_tools

            # Execute tools
            tool_results = execute_tools(tool_calls)

            # Format results for streaming
            results_text = "\n\n🔧 Tool Results:\n"
            for call, result in zip(tool_calls, tool_results):
                # Show tool name and parameters for transparency
                params_str = str(call.arguments) if call.arguments else "{}"
                if len(params_str) > 100:
                    params_str = params_str[:97] + "..."

                results_text += f"**{call.name}({params_str})**\n"

                # Show result
                if result.success:
                    results_text += f"✅ {result.output}\n\n"
                else:
                    results_text += f"❌ Error: {result.error}\n\n"

            return results_text

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"\n\n❌ Tool execution error: {e}\n\n"
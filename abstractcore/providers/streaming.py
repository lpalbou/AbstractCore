"""
Unified streaming processor with incremental tool detection.

This module provides a single streaming strategy that handles tools elegantly
while maintaining real-time streaming performance, with proper tag rewriting support.
"""

import json
import re
import logging
import uuid
from typing import List, Dict, Any, Optional, Iterator, Tuple
from enum import Enum

from ..core.types import GenerateResponse
from ..tools.core import ToolCall

logger = logging.getLogger(__name__)


class ToolDetectionState(Enum):
    """
    State machine for tool call detection.

    Note: This is kept for backward compatibility with older tests.
    The current implementation doesn't use explicit state transitions.
    """
    SCANNING = "scanning"  # Looking for tool call start
    IN_TOOL_CALL = "in_tool_call"  # Inside a tool call
    COMPLETE = "complete"  # Tool call completed


class IncrementalToolDetector:
    """
    Improved incremental tool call detector that preserves tool calls for tag rewriting.

    Unlike the original detector, this version PRESERVES tool calls in the streamable
    content while also extracting them for execution. This allows tag rewriting to work.
    """

    def __init__(self, model_name: Optional[str] = None, rewrite_tags: bool = False):
        """
        Initialize detector.

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
            },
            'llama': {
                'start': r'<function_call>',
                'end': r'</function_call>',
            },
            'xml': {
                'start': r'<tool_call>',
                'end': r'</tool_call>',
            },
            'gemma': {
                'start': r'```tool_code',
                'end': r'```',
            }
        }

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

        if 'gemma' in model_lower:
            return [self.patterns['gemma']]
        elif 'llama' in model_lower:
            return [self.patterns['llama'], self.patterns['xml']]
        else:
            return [
                self.patterns['qwen'],
                self.patterns['llama'],
                self.patterns['xml']
            ]

    def process_chunk(self, chunk_content: str) -> Tuple[str, List[ToolCall]]:
        """
        Process chunk and detect complete tool calls.

        Key difference: When rewrite_tags=True, tool calls are PRESERVED in streamable content.

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
                if self.rewrite_tags:
                    # When rewriting, keep tool call in content but still track state
                    streamable_content = ""  # Don't stream partial content yet
                else:
                    # Normal mode - stream content before tool call
                    streamable_content = self.accumulated_content[:self.tool_start_pos]

                # Start collecting tool content
                self.current_tool_content = self.accumulated_content[match.end():]

                logger.debug(f"Tool call start detected: {start_pattern} for model {self.model_name}")
                logger.debug(f"Accumulated content: {repr(self.accumulated_content[:100])}")

                # Immediately check if tool is already complete (if end tag is in current content)
                additional_streamable, additional_tools = self._collect_tool_content("")
                streamable_content += additional_streamable
                completed_tools.extend(additional_tools)
                break
        else:
            # No tool start found
            if self.rewrite_tags:
                # Check for partial tool tags when rewriting
                if self._might_have_partial_tool_call():
                    streamable_content = ""  # Buffer everything
                else:
                    streamable_content = self.accumulated_content
                    self.accumulated_content = ""
            else:
                # Normal streaming - use smart buffering
                streamable_content = self._extract_streamable_content()

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

            if self.rewrite_tags:
                # When rewriting, stream the complete accumulated content including tool call
                streamable_content = self.accumulated_content
                logger.debug(f"Tool complete, streaming accumulated content for rewriting: {streamable_content[:200]}")
                self.accumulated_content = ""
            else:
                # Normal mode - don't stream the tool call itself
                pass

            # Reset for next tool
            remaining_content = self.current_tool_content[end_match.end():]
            self.reset()

            # Continue processing remaining content
            if remaining_content:
                self.accumulated_content = remaining_content
                additional_streamable, additional_tools = self._scan_for_tool_start("")
                streamable_content += additional_streamable
                completed_tools.extend(additional_tools)

        return streamable_content, completed_tools

    def _might_have_partial_tool_call(self) -> bool:
        """Check if accumulated content might contain start of a tool call."""
        # Check for partial tool tags more aggressively to handle character-by-character streaming
        tail = self.accumulated_content[-20:] if len(self.accumulated_content) > 20 else self.accumulated_content

        # Expanded list of potential partial starts to catch character-by-character streaming
        potential_partial_starts = [
            '<', '<|', '<f', '</', '<t', '`', '``',
            '<fu', '<fun', '<func', '<funct', '<functi', '<functio', '<function',  # <function_call>
            '<tool', '<tool_', '<tool_c', '<tool_ca', '<tool_cal',  # <tool_call>
            '<|t', '<|to', '<|too', '<|tool', '<|tool_', '<|tool_c'  # <|tool_call|>
        ]

        # Check if tail ends with any potential partial start
        for start in potential_partial_starts:
            if tail.endswith(start):
                return True

        # Also check if we have the start of any tag pattern in the middle
        for pattern_partial in ['<function', '<tool_call', '<|tool', '```tool']:
            if pattern_partial in tail:
                return True

        # Check if we have an incomplete tool call (start tag but no end tag)
        for pattern_info in self.active_patterns:
            start_pattern = pattern_info['start']
            end_pattern = pattern_info['end']

            if re.search(start_pattern, self.accumulated_content, re.IGNORECASE):
                # Has start tag - check if also has end tag
                if not re.search(end_pattern, self.accumulated_content, re.IGNORECASE):
                    # Incomplete tool call - should buffer
                    return True

        return False

    def _extract_streamable_content(self) -> str:
        """Extract streamable content, buffering partial tool tags."""
        # Check if accumulated content might contain partial tool tag at the end
        tail = self.accumulated_content[-20:] if len(self.accumulated_content) > 20 else self.accumulated_content

        tag_starters = ('<', '<|', '</', '<|t', '<|to', '<|tool', '<function', '<tool', '``', '```')
        might_be_partial = any(starter in tail for starter in tag_starters)

        if might_be_partial and len(self.accumulated_content) > 20:
            # Keep last 20 chars as buffer, stream the rest
            streamable_content = self.accumulated_content[:-20]
            self.accumulated_content = self.accumulated_content[-20:]
        elif not might_be_partial:
            # No partial tag, stream everything
            streamable_content = self.accumulated_content
            self.accumulated_content = ""
        else:
            # Everything might be partial, don't stream yet
            streamable_content = ""

        return streamable_content

    def _parse_tool_json(self, json_content: str) -> Optional[ToolCall]:
        """Parse JSON content to create ToolCall."""
        if not json_content or not json_content.strip():
            return None

        try:
            cleaned_json = json_content.strip()

            # Handle missing braces
            if cleaned_json.count('{') > cleaned_json.count('}'):
                missing_braces = cleaned_json.count('{') - cleaned_json.count('}')
                cleaned_json += '}' * missing_braces

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

    def finalize(self) -> List[ToolCall]:
        """Finalize and return any remaining tool calls."""
        completed_tools = []

        if self.state == ToolDetectionState.IN_TOOL_CALL:
            # Try to parse any remaining content as incomplete tool
            if self.current_tool_content:
                # Try to parse incomplete JSON by looking for valid JSON objects
                tool_call = self._try_parse_incomplete_json(self.current_tool_content)
                if tool_call:
                    completed_tools.append(tool_call)

        return completed_tools

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


class UnifiedStreamProcessor:
    """
    FIXED unified streaming processor with proper tag rewriting.

    Key improvement: Preserves tool calls in content for tag rewriting,
    then rewrites them BEFORE yielding.
    """

    def __init__(self, model_name: str, execute_tools: bool = False,
                 tool_call_tags: Optional[str] = None,
                 default_target_format: str = "qwen3"):
        """Initialize the stream processor."""
        self.model_name = model_name
        # Note: execute_tools parameter is kept for backward compatibility but ignored
        # Tool execution is now handled by the client (CLI)
        self.tool_call_tags = tool_call_tags
        self.default_target_format = default_target_format

        # Always initialize tag rewriter - either custom tags or default target format
        self.tag_rewriter = None
        # Backwards compatibility: tag_rewrite_buffer attribute (unused in current implementation)
        self.tag_rewrite_buffer = ""

        # Flag to indicate if we're converting to OpenAI JSON format (not text rewriting)
        self.convert_to_openai_json = False

        # Determine whether tool_call_tags contains predefined format or custom tags
        if tool_call_tags:
            # Check if tool_call_tags is a predefined format name
            predefined_formats = ["qwen3", "openai", "llama3", "xml", "gemma"]

            if tool_call_tags in predefined_formats:
                # It's a predefined format - use default rewriter
                self._initialize_default_rewriter(tool_call_tags)
                logger.debug(f"Treating tool_call_tags '{tool_call_tags}' as predefined format")
            elif ',' in tool_call_tags:
                # It contains comma - likely custom tags like "START,END"
                self._initialize_tag_rewriter(tool_call_tags)
                logger.debug(f"Treating tool_call_tags '{tool_call_tags}' as custom comma-separated tags")
            else:
                # Single string that's not a predefined format - could be custom single tag
                # Try as custom first, fall back to treating as predefined format
                try:
                    self._initialize_tag_rewriter(tool_call_tags)
                    logger.debug(f"Treating tool_call_tags '{tool_call_tags}' as custom single tag")
                except Exception as e:
                    logger.debug(f"Failed to initialize as custom tag, trying as predefined format: {e}")
                    self._initialize_default_rewriter(tool_call_tags)
        else:
            # No custom tags - initialize default rewriter to target format
            self._initialize_default_rewriter(default_target_format)

        # Create detector - preserve tool calls for rewriting/conversion
        # When converting to OpenAI JSON, we still need to detect and extract tool calls
        self.detector = IncrementalToolDetector(
            model_name=model_name,
            rewrite_tags=(self.tag_rewriter is not None or self.convert_to_openai_json)
        )

    def process_stream(self, response_stream: Iterator[GenerateResponse],
                      converted_tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """
        Process a response stream with tag rewriting and tool detection.

        Args:
            response_stream: Iterator of response chunks
            converted_tools: Available tools for execution

        Yields:
            GenerateResponse: Processed chunks with rewritten tags
        """
        try:
            for chunk in response_stream:
                if not chunk.content:
                    yield chunk
                    continue

                # Process chunk through detector (preserves tool calls for rewriting)
                streamable_content, completed_tools = self.detector.process_chunk(chunk.content)

                # Apply tag rewriting or OpenAI conversion if we have content
                if streamable_content:
                    if self.convert_to_openai_json:
                        logger.debug(f"Converting to OpenAI format: {streamable_content[:100]}")
                        streamable_content = self._convert_to_openai_format(streamable_content)
                        logger.debug(f"After OpenAI conversion: {streamable_content[:100]}")
                    elif self.tag_rewriter:
                        logger.debug(f"Applying tag rewriting to: {streamable_content[:100]}")
                        streamable_content = self._apply_tag_rewriting_direct(streamable_content)
                        logger.debug(f"After tag rewriting: {streamable_content[:100]}")

                # Yield streamable content
                if streamable_content:
                    yield GenerateResponse(
                        content=streamable_content,
                        model=chunk.model,
                        finish_reason=chunk.finish_reason,
                        usage=chunk.usage,
                        raw_response=chunk.raw_response
                    )

                # Yield tool calls for server processing
                if completed_tools:
                    logger.debug(f"Detected {len(completed_tools)} tools - yielding for server processing")
                    yield GenerateResponse(
                        content="",
                        tool_calls=completed_tools,
                        model=chunk.model,
                        finish_reason=chunk.finish_reason,
                        usage=chunk.usage,
                        raw_response=chunk.raw_response
                    )

            # Finalize - get any remaining tools and handle remaining content
            final_tools = self.detector.finalize()

            # Get any remaining accumulated content
            remaining_content = self.detector.accumulated_content
            self.detector.accumulated_content = ""

            if remaining_content:
                if self.convert_to_openai_json:
                    remaining_content = self._convert_to_openai_format(remaining_content)
                elif self.tag_rewriter:
                    remaining_content = self._apply_tag_rewriting_direct(remaining_content)

                yield GenerateResponse(
                    content=remaining_content,
                    model=self.model_name,
                    finish_reason="stop"
                )

            if final_tools:
                logger.debug(f"Finalized {len(final_tools)} tools - yielding for server processing")
                yield GenerateResponse(
                    content="",
                    tool_calls=final_tools,
                    model=self.model_name,
                    finish_reason="tool_calls"
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
                    # Comma-separated: User specified both start and end tags
                    # Store as plain tags, rewriter will wrap with angle brackets
                    parts = tool_call_tags.split(',')
                    if len(parts) == 2:
                        tags = ToolCallTags(
                            start_tag=parts[0].strip(),
                            end_tag=parts[1].strip(),
                            auto_format=False  # Don't auto-format, keep plain tags
                        )
                    else:
                        logger.warning(f"Invalid tool_call_tags format: {tool_call_tags}")
                        return
                else:
                    # Single tag: Auto-format to <tag> and </tag>
                    tags = ToolCallTags(
                        start_tag=tool_call_tags.strip(),
                        end_tag=tool_call_tags.strip(),
                        auto_format=True  # Enable auto-formatting for single tags
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

    def _initialize_default_rewriter(self, target_format: str):
        """Initialize default rewriter to convert any tool format to target format."""
        try:
            from ..tools.tag_rewriter import ToolCallTagRewriter, ToolCallTags

            # Check if target_format contains custom tags (comma-separated)
            if ',' in target_format:
                # Custom tag format: "START,END"
                parts = target_format.split(',')
                if len(parts) == 2:
                    target_tags = ToolCallTags(
                        start_tag=parts[0].strip(),
                        end_tag=parts[1].strip(),
                        auto_format=False  # Use exact custom tags
                    )
                    self.tag_rewriter = ToolCallTagRewriter(target_tags)
                    logger.debug(f"Initialized custom tag rewriter '{parts[0].strip()}...{parts[1].strip()}' for model {self.model_name}")
                else:
                    logger.warning(f"Invalid custom tag format '{target_format}' - expected 'START,END'")
                    return
            elif target_format == "qwen3":
                # Qwen3 format: <|tool_call|>...JSON...</|tool_call|>
                target_tags = ToolCallTags(
                    start_tag="<|tool_call|>",
                    end_tag="</|tool_call|>",
                    auto_format=False  # Use exact tags
                )
                self.tag_rewriter = ToolCallTagRewriter(target_tags)
                logger.debug(f"Initialized qwen3 tag rewriter for model {self.model_name}")
            elif target_format == "openai":
                # OpenAI format: Convert text-based tool calls TO OpenAI's structured JSON format
                # This is NOT a text rewriting operation - it's a format conversion
                # We need to:
                # 1. Detect tool calls in text (Qwen3/LLaMA/XML formats)
                # 2. Parse the JSON content
                # 3. Wrap in OpenAI's structured format with id, type, function fields
                self.tag_rewriter = None  # No text rewriting
                self.convert_to_openai_json = True  # Enable JSON conversion
                logger.debug(f"OpenAI format selected - will convert text-based tool calls to OpenAI JSON format")
                return
            elif target_format == "llama3":
                # LLaMA3/Crush CLI format: <function_call>...JSON...</function_call>
                target_tags = ToolCallTags(
                    start_tag="<function_call>",
                    end_tag="</function_call>",
                    auto_format=False
                )
                self.tag_rewriter = ToolCallTagRewriter(target_tags)
                logger.debug(f"Initialized llama3 tag rewriter for model {self.model_name}")
            elif target_format == "xml":
                # XML/Gemini CLI format: <tool_call>...JSON...</tool_call>
                target_tags = ToolCallTags(
                    start_tag="<tool_call>",
                    end_tag="</tool_call>",
                    auto_format=False
                )
                self.tag_rewriter = ToolCallTagRewriter(target_tags)
                logger.debug(f"Initialized xml tag rewriter for model {self.model_name}")
            elif target_format == "gemma":
                # Gemma format: ```tool_code...JSON...```
                target_tags = ToolCallTags(
                    start_tag="```tool_code\n",
                    end_tag="\n```",
                    auto_format=False
                )
                self.tag_rewriter = ToolCallTagRewriter(target_tags)
                logger.debug(f"Initialized gemma tag rewriter for model {self.model_name}")
            else:
                # Try to handle as single tag format (auto-format to <tag>...</tag>)
                if target_format and not target_format.isspace():
                    target_tags = ToolCallTags(
                        start_tag=target_format.strip(),
                        end_tag=target_format.strip(),
                        auto_format=True  # Auto-wrap with angle brackets
                    )
                    self.tag_rewriter = ToolCallTagRewriter(target_tags)
                    logger.debug(f"Initialized auto-formatted tag rewriter '<{target_format.strip()}>...</{target_format.strip()}>' for model {self.model_name}")
                else:
                    logger.warning(f"Unknown or empty target format: '{target_format}' - no tag rewriting will be applied")

        except Exception as e:
            logger.error(f"Failed to initialize default rewriter: {e}")

    def _apply_tag_rewriting_direct(self, content: str) -> str:
        """
        Apply tag rewriting using the direct (non-streaming) rewriter method.

        Since we now have complete tool calls in the content, we can use
        the simpler rewrite_text() method instead of the buffered streaming approach.
        """
        if not self.tag_rewriter or not content:
            return content

        try:
            # Use direct text rewriting since we have complete tool calls
            rewritten = self.tag_rewriter.rewrite_text(content)
            if rewritten != content:
                logger.debug(f"Tag rewriting successful: {content[:50]} -> {rewritten[:50]}")
            else:
                logger.debug(f"Tag rewriting had no effect on: {content[:50]}")
            return rewritten
        except Exception as e:
            logger.debug(f"Tag rewriting failed: {e}")
            return content

    def _convert_to_openai_format(self, content: str) -> str:
        """
        Convert text-based tool calls to OpenAI JSON format.

        Detects tool calls in formats like:
        - Qwen3: <|tool_call|>{"name": "shell", "arguments": {...}}</|tool_call|>
        - LLaMA: <function_call>{"name": "shell", "arguments": {...}}</function_call>
        - XML: <tool_call>{"name": "shell", "arguments": {...}}</tool_call>

        Converts to OpenAI format:
        {"id": "call_abc123", "type": "function", "function": {"name": "shell", "arguments": "{...}"}}
        """
        if not content:
            return content

        # Patterns for different tool call formats
        patterns = [
            (r'<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>', 'qwen3'),
            (r'<function_call>\s*(.*?)\s*</function_call>', 'llama'),
            (r'<tool_call>\s*(.*?)\s*</tool_call>', 'xml'),
            (r'```tool_code\s*\n(.*?)\n```', 'gemma'),
        ]

        converted_content = content

        for pattern, format_type in patterns:
            matches = list(re.finditer(pattern, converted_content, re.DOTALL | re.IGNORECASE))

            if matches:
                logger.debug(f"Found {len(matches)} tool calls in {format_type} format")

                # Replace from end to beginning to maintain indices
                for match in reversed(matches):
                    try:
                        # Extract JSON content
                        json_content = match.group(1).strip()

                        # Parse the JSON to validate and extract fields
                        tool_data = json.loads(json_content)

                        if not isinstance(tool_data, dict) or "name" not in tool_data:
                            logger.warning(f"Invalid tool call JSON: {json_content[:100]}")
                            continue

                        # Generate OpenAI-compatible tool call ID
                        call_id = f"call_{uuid.uuid4().hex[:24]}"

                        # Convert to OpenAI format
                        openai_tool_call = {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_data["name"],
                                "arguments": json.dumps(tool_data.get("arguments", {}))
                            }
                        }

                        # Replace the text-based tool call with OpenAI JSON format
                        openai_json = json.dumps(openai_tool_call)
                        converted_content = converted_content[:match.start()] + openai_json + converted_content[match.end():]

                        logger.debug(f"Converted {format_type} tool call to OpenAI format: {openai_json[:100]}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool call JSON: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error converting tool call to OpenAI format: {e}")
                        continue

                # Only process the first matching format type
                break

        return converted_content


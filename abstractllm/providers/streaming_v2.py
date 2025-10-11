"""
FIXED unified streaming processor with proper tag rewriting support.

This is a corrected implementation that properly handles tool call tag rewriting
in streaming scenarios.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from enum import Enum

from ..core.types import GenerateResponse
from ..tools.core import ToolCall

logger = logging.getLogger(__name__)


class IncrementalToolDetectorV2:
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
        self.accumulated_content = ""
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
        complete_call_found = False

        # Look for complete tool calls in accumulated content
        for pattern_info in self.active_patterns:
            start_pattern = pattern_info['start']
            end_pattern = pattern_info['end']

            # Build full pattern to match complete tool call
            full_pattern = f"{re.escape(start_pattern)}(.*?){re.escape(end_pattern)}"

            matches = list(re.finditer(full_pattern, self.accumulated_content, re.DOTALL | re.IGNORECASE))

            if matches:
                complete_call_found = True

            for match in matches:
                json_content = match.group(1).strip()
                tool_call = self._parse_tool_json(json_content)

                if tool_call:
                    completed_tools.append(tool_call)
                    logger.debug(f"Complete tool call detected: {tool_call.name}")

        # Determine what to stream
        if self.rewrite_tags:
            # When rewriting, we need to be careful:
            # - If we found complete tool calls, stream ALL content (it will be rewritten)
            # - If no complete tool calls yet, check if we might have partial tool call
            #   If yes, buffer everything
            #   If no, stream what we have
            if complete_call_found:
                # Complete tool call found - stream everything for rewriting
                streamable_content = self.accumulated_content
                self.accumulated_content = ""  # Clear buffer after streaming
            else:
                # No complete tool call yet - check for partial
                if self._might_have_partial_tool_call():
                    # Might be building up a tool call - don't stream yet
                    streamable_content = ""
                else:
                    # No tool call detected at all - safe to stream
                    streamable_content = self.accumulated_content
                    self.accumulated_content = ""
        else:
            # When NOT rewriting, check for partial tool tags at end
            # and buffer them to avoid streaming incomplete tags
            streamable_content = self._extract_streamable_content()

        return streamable_content, completed_tools

    def _might_have_partial_tool_call(self) -> bool:
        """Check if accumulated content might contain start of a tool call."""
        # Check for partial start tags at end of content
        tail = self.accumulated_content[-30:] if len(self.accumulated_content) > 30 else self.accumulated_content

        # Check for any pattern start tags (or prefixes of them)
        for pattern_info in self.active_patterns:
            start_pattern = pattern_info['start']
            # Remove regex escaping for simple string checking
            start_tag = start_pattern.replace('\\|', '|').replace('\\', '')

            # Check if we have partial match of start tag
            for i in range(1, len(start_tag) + 1):
                partial = start_tag[:i]
                if tail.endswith(partial):
                    return True

            # Also check if we have start tag (incomplete tool call)
            if re.search(start_pattern, self.accumulated_content, re.IGNORECASE):
                # Has start tag but we already know no complete match (from earlier check)
                # So this is definitely incomplete
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

    def finalize(self) -> Tuple[str, List[ToolCall]]:
        """Finalize and return remaining content and tools."""
        # Stream any remaining accumulated content
        remaining_content = self.accumulated_content
        self.accumulated_content = ""
        return remaining_content, []


class UnifiedStreamProcessorV2:
    """
    FIXED unified streaming processor with proper tag rewriting.

    Key improvement: Preserves tool calls in content for tag rewriting,
    then rewrites them BEFORE yielding.
    """

    def __init__(self, model_name: str, execute_tools: bool = True,
                 tool_call_tags: Optional[str] = None):
        """Initialize the stream processor."""
        self.model_name = model_name
        self.execute_tools = execute_tools
        self.tool_call_tags = tool_call_tags

        # Initialize tag rewriter if custom tags are provided
        self.tag_rewriter = None
        if tool_call_tags:
            self._initialize_tag_rewriter(tool_call_tags)

        # Create detector - if we have tag rewriter, preserve tool calls in content
        self.detector = IncrementalToolDetectorV2(
            model_name=model_name,
            rewrite_tags=bool(self.tag_rewriter)
        )

    def process_stream(self, response_stream: Iterator[GenerateResponse],
                      converted_tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """
        Process a response stream with tag rewriting and tool detection.

        Processing order (FIXED):
        1. Accumulate chunks
        2. Detect complete tool calls
        3. If tag rewriting enabled, rewrite tool call tags in content
        4. Yield rewritten content
        5. Execute tools (if enabled)

        Args:
            response_stream: Iterator of response chunks
            converted_tools: Available tools for execution

        Yields:
            GenerateResponse: Processed chunks with rewritten tags and tool execution
        """
        try:
            for chunk in response_stream:
                if not chunk.content:
                    yield chunk
                    continue

                # Process chunk through detector
                # If tag rewriting enabled, this preserves tool calls in content
                streamable_content, completed_tools = self.detector.process_chunk(chunk.content)

                # Apply tag rewriting if enabled and we have content
                if streamable_content and self.tag_rewriter:
                    streamable_content = self._apply_tag_rewriting_direct(streamable_content)

                # Yield streamable content
                if streamable_content:
                    yield GenerateResponse(
                        content=streamable_content,
                        model=chunk.model,
                        finish_reason=chunk.finish_reason,
                        usage=chunk.usage,
                        raw_response=chunk.raw_response
                    )

                # Execute completed tools if enabled
                if completed_tools and self.execute_tools and converted_tools:
                    logger.debug(f"Executing {len(completed_tools)} tools immediately")
                    tool_results = self._execute_tools_immediately(completed_tools, converted_tools)

                    if tool_results:
                        yield GenerateResponse(
                            content=tool_results,
                            model=chunk.model,
                            finish_reason=chunk.finish_reason,
                            usage=chunk.usage,
                            raw_response=chunk.raw_response
                        )

            # Finalize - stream any remaining content
            remaining_content, final_tools = self.detector.finalize()

            if remaining_content:
                if self.tag_rewriter:
                    remaining_content = self._apply_tag_rewriting_direct(remaining_content)

                yield GenerateResponse(
                    content=remaining_content,
                    model=self.model_name,
                    finish_reason="stop"
                )

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
                        # Disable auto_format to use exact tags from user
                        tags = ToolCallTags(
                            start_tag=parts[0].strip(),
                            end_tag=parts[1].strip(),
                            auto_format=False  # CRITICAL: Don't auto-add <>
                        )
                    else:
                        logger.warning(f"Invalid tool_call_tags format: {tool_call_tags}")
                        return
                else:
                    tags = ToolCallTags(
                        start_tag=tool_call_tags.strip(),
                        end_tag=tool_call_tags.strip(),
                        auto_format=False
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
            return self.tag_rewriter.rewrite_text(content)
        except Exception as e:
            logger.debug(f"Tag rewriting failed: {e}")
            return content

    def _execute_tools_immediately(self, tool_calls: List[ToolCall],
                                  converted_tools: List[Dict[str, Any]]) -> str:
        """Execute tools immediately and return formatted results."""
        try:
            from ..tools import execute_tools

            tool_results = execute_tools(tool_calls)

            results_text = "\n\n🔧 Tool Results:\n"
            for call, result in zip(tool_calls, tool_results):
                params_str = str(call.arguments) if call.arguments else "{}"
                if len(params_str) > 100:
                    params_str = params_str[:97] + "..."

                results_text += f"**{call.name}({params_str})**\n"

                if result.success:
                    results_text += f"✅ {result.output}\n\n"
                else:
                    results_text += f"❌ Error: {result.error}\n\n"

            return results_text

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"\n\n❌ Tool execution error: {e}\n\n"

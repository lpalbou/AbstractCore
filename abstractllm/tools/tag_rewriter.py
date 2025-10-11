"""
Tool call tag rewriter for real-time tag customization.

This module provides functionality to rewrite tool call tags in real-time,
supporting different agentic CLI requirements and streaming scenarios.
"""

import re
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ToolCallTags:
    """
    Configuration for tool call tags.

    Attributes:
        start_tag: Opening tag for tool calls (e.g., "<|tool_call|>")
        end_tag: Closing tag for tool calls (e.g., "</|tool_call|>")
        preserve_json: Whether to preserve the JSON content between tags
        auto_format: Whether to automatically add angle brackets if missing
    """
    start_tag: str
    end_tag: str
    preserve_json: bool = True
    auto_format: bool = True
    
    def __post_init__(self):
        """Validate tag configuration."""
        if not self.start_tag or not self.end_tag:
            raise ValueError("Both start_tag and end_tag must be provided")

        # Only auto-format if enabled
        if self.auto_format:
            # Ensure tags are properly formatted
            # Only add angle brackets if the tag doesn't already have them
            # and doesn't start with special characters like ```
            if (not self.start_tag.startswith('<') and
                not self.start_tag.startswith('```') and
                not self.start_tag.startswith('`')):
                self.start_tag = f'<{self.start_tag}>'

            # For end tag: check if it already has angle brackets at all
            # If it starts with '<' (like '<END>'), leave it as-is
            # Only auto-format if it's a plain tag (like 'custom_tool')
            if (not self.end_tag.startswith('<') and
                not self.end_tag.startswith('```') and
                not self.end_tag.startswith('`')):
                # Plain tag - add </ prefix
                self.end_tag = f'</{self.end_tag}>'


class ToolCallTagRewriter:
    """
    Real-time tool call tag rewriter.
    
    This class provides functionality to rewrite tool call tags in real-time,
    supporting streaming scenarios and different agentic CLI requirements.
    """
    
    def __init__(self, target_tags: ToolCallTags):
        """
        Initialize the tag rewriter.

        Args:
            target_tags: Target tag configuration
        """
        self.target_tags = target_tags
        # Compute formatted tags for output
        self._output_start_tag = self._format_tag_for_output(target_tags.start_tag, is_end=False)
        self._output_end_tag = self._format_tag_for_output(target_tags.end_tag, is_end=True)
        self._compiled_patterns = self._compile_patterns()

    def _format_tag_for_output(self, tag: str, is_end: bool = False) -> str:
        """
        Format tag for output, respecting auto_format setting.

        When auto_format=True: Plain tags like 'ojlk' become '<ojlk>' and 'dfsd' becomes '</dfsd>'
        When auto_format=False: Tags are used exactly as specified by user

        Args:
            tag: The tag to format
            is_end: Whether this is an end tag (adds '/' prefix if needed)

        Returns:
            Formatted tag
        """
        # If auto_format is disabled, use tags exactly as specified
        if not self.target_tags.auto_format:
            return tag

        # If tag already has angle brackets or special formatting, return as-is
        if tag.startswith('<') or tag.startswith('```') or tag.startswith('`'):
            return tag

        # Plain tag - wrap with angle brackets only if auto_format is enabled
        if is_end:
            # End tag: add </ prefix and > suffix
            return f'</{tag}>'
        else:
            # Start tag: add < prefix and > suffix
            return f'<{tag}>'
    
    def _compile_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Compile regex patterns for different tool call formats.

        Returns:
            List of (pattern, replacement_template) tuples
        """
        patterns = []

        # Pattern 1: Qwen3 format <|tool_call|>...JSON...</|tool_call|>
        qwen_pattern = re.compile(
            r'<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>',
            re.DOTALL | re.IGNORECASE
        )
        patterns.append((qwen_pattern, f"{self._output_start_tag}\\1{self._output_end_tag}"))

        # Pattern 2: LLaMA3 format <function_call>...JSON...</function_call>
        llama_pattern = re.compile(
            r'<function_call>\s*(.*?)\s*</function_call>',
            re.DOTALL | re.IGNORECASE
        )
        patterns.append((llama_pattern, f"{self._output_start_tag}\\1{self._output_end_tag}"))

        # Pattern 3: XML format <tool_call>...JSON...</tool_call>
        xml_pattern = re.compile(
            r'<tool_call>\s*(.*?)\s*</tool_call>',
            re.DOTALL | re.IGNORECASE
        )
        patterns.append((xml_pattern, f"{self._output_start_tag}\\1{self._output_end_tag}"))

        # Pattern 4: Gemma format ```tool_code...JSON...```
        gemma_pattern = re.compile(
            r'```tool_code\s*\n(.*?)\n```',
            re.DOTALL | re.IGNORECASE
        )
        patterns.append((gemma_pattern, f"{self._output_start_tag}\\1{self._output_end_tag}"))

        # Pattern 5: Generic JSON (wrap in tags) - only standalone JSON
        # This pattern matches JSON objects that start with { and contain "name"
        # It's more flexible and handles nested structures better
        json_pattern = re.compile(
            r'(?<![<])(\{[^{}]*["\']name["\'][^{}]*(?:\{[^{}]*\}[^{}]*)*\})(?![>])',
            re.DOTALL
        )
        patterns.append((json_pattern, f"{self._output_start_tag}\\1{self._output_end_tag}"))

        return patterns
    
    def rewrite_text(self, text: str) -> str:
        """
        Rewrite tool call tags in text.

        Args:
            text: Input text containing tool calls

        Returns:
            Text with rewritten tool call tags
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.debug(f"rewrite_text called with text: {text[:100] if text else None}")
        logger.debug(f"Target output tags: start='{self._output_start_tag}', end='{self._output_end_tag}'")

        if not text or not self.target_tags.preserve_json:
            logger.debug("Early return: text is empty or preserve_json is False")
            return text

        # Check if we already have the target format (avoid double-tagging)
        # Check using output tags (with angle brackets)
        if (self._output_start_tag in text and
            self._output_end_tag in text):
            logger.debug(f"Already in target format, returning as-is")
            # Already in target format, just return as-is
            return text

        rewritten = text
        logger.debug(f"Starting pattern matching with {len(self._compiled_patterns)} patterns")

        # Apply each pattern to find all matches, not just the first one
        for i, (pattern, replacement) in enumerate(self._compiled_patterns):
            # Find all matches and replace them
            matches = list(pattern.finditer(rewritten))
            logger.debug(f"Pattern {i}: {pattern.pattern[:50]}... - found {len(matches)} matches")
            if matches:
                # Replace from end to beginning to avoid index shifting
                for match in reversed(matches):
                    start, end = match.span()
                    match_text = match.group(1) if match.groups() else match.group(0)
                    logger.debug(f"Match found at {start}:{end}, text: {match_text[:50]}")
                    # Create replacement with properly formatted output tags
                    replacement_text = f"{self._output_start_tag}{match_text}{self._output_end_tag}"
                    rewritten = rewritten[:start] + replacement_text + rewritten[end:]
                    logger.debug(f"Replaced with: {replacement_text[:100]}")
                break  # Only apply the first matching pattern type

        # Additional pass for plain JSON tool calls that might not match the regex
        # This handles cases where the regex is too restrictive
        if not (self._output_start_tag in rewritten and self._output_end_tag in rewritten):
            # Look for JSON objects in the text and wrap them
            # Use a more flexible approach that finds balanced JSON objects
            import re
            # Find all potential JSON objects by looking for balanced braces
            json_objects = self._find_json_objects(rewritten)
            for json_obj in json_objects:
                if self._is_plain_json_tool_call(json_obj):
                    # Replace the JSON object with wrapped version using output tags
                    wrapped_json = f"{self._output_start_tag}{json_obj}{self._output_end_tag}"
                    rewritten = rewritten.replace(json_obj, wrapped_json, 1)
                    logger.debug(f"Plain JSON wrapped: {wrapped_json[:100]}")

        logger.debug(f"Final rewritten text: {rewritten[:200] if rewritten else None}")
        return rewritten
    
    def rewrite_streaming_chunk(self, chunk: str, buffer: str = "") -> Tuple[str, str]:
        """
        Rewrite tool call tags in a streaming chunk using SOTA buffer-based approach.
        
        This method uses immediate rewriting strategy:
        1. Rewrite start tags immediately when detected
        2. Buffer content until end tag is found
        3. Rewrite end tag when complete tool call is detected
        
        This approach minimizes latency for agency loops while maintaining
        clean output and avoiding double-tagging.
        
        Args:
            chunk: Current chunk of text
            buffer: Previous buffer for handling split tool calls
            
        Returns:
            Tuple of (rewritten_chunk, updated_buffer)
        """
        if not chunk:
            return chunk, buffer
        
        # Combine buffer with current chunk
        full_text = buffer + chunk

        # Check if we already have the target format (avoid double-tagging)
        # Use output tags for checking
        if (self._output_start_tag in full_text and
            self._output_end_tag in full_text):
            # Already in target format, just return as-is
            return full_text, ""
        
        # Check if we have a complete tool call in the full text
        if self._has_complete_tool_call(full_text):
            # Rewrite the complete tool call
            rewritten_tool_call = self._rewrite_complete_tool_call(full_text)
            return rewritten_tool_call, ""
        
        # Check if we have an incomplete tool call (start tag but no end tag)
        if self._has_incomplete_tool_call(full_text):
            # We have a start tag but no end tag yet
            # Don't output anything yet, keep buffering
            return "", full_text
        
        # Check if we have a complete plain JSON tool call
        if self._is_plain_json_tool_call(full_text.strip()):
            # We have a complete plain JSON tool call
            rewritten_tool_call = self._rewrite_complete_tool_call(full_text.strip())
            return rewritten_tool_call, ""
        
        # Check if we're in the middle of a potential plain JSON tool call
        if (full_text.strip().startswith('{') and 
            '"name"' in full_text and
            full_text.count('{') == full_text.count('}')):
            # We have a complete JSON object, try to parse it
            try:
                import json
                json.loads(full_text.strip())
                # If we get here, it's valid JSON - treat as tool call
                rewritten_tool_call = self._rewrite_complete_tool_call(full_text.strip())
                return rewritten_tool_call, ""
            except:
                # Not valid JSON, keep buffering
                return "", full_text
        elif (full_text.strip().startswith('{') and 
              '"name"' in full_text):
            # We're in the middle of a potential JSON tool call, keep buffering
            return "", full_text
        else:
            # Check if there's a JSON tool call embedded in the text
            import re
            json_pattern = r'\{[^{}]*"name"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, full_text)
            
            if matches:
                # Found JSON tool calls in the text, rewrite them
                rewritten_text = full_text
                for match in matches:
                    try:
                        import json
                        json.loads(match)
                        # Valid JSON, rewrite it
                        rewritten_match = f"<function_call>{match}</function_call>"
                        rewritten_text = rewritten_text.replace(match, rewritten_match)
                    except:
                        # Not valid JSON, skip
                        continue
                return rewritten_text, ""
            else:
                # No tool call detected, output the chunk and clear buffer
                return chunk, ""
    
    def _has_complete_tool_call(self, text: str) -> bool:
        """Check if text contains a complete tool call."""
        # Look for any source format that has both start and end tags
        source_patterns = [
            r'<\|tool_call\|>.*?</\|tool_call\|>',
            r'<function_call>.*?</function_call>',
            r'<tool_call>.*?</tool_call>',
            r'```tool_code.*?```',
        ]
        
        for pattern in source_patterns:
            if re.search(pattern, text, re.DOTALL):
                return True
        
        # Also check for plain JSON tool calls
        if self._is_plain_json_tool_call(text):
            return True
            
        return False
    
    def _has_incomplete_tool_call(self, text: str) -> bool:
        """Check if text contains an incomplete tool call (start tag but no end tag)."""
        # Look for start tags without corresponding end tags
        start_patterns = [
            r'<\|tool_call\|>',
            r'<function_call>',
            r'<tool_call>',
            r'```tool_code',
        ]
        
        for pattern in start_patterns:
            if re.search(pattern, text):
                # Check if we have the corresponding end tag
                if pattern == r'<\|tool_call\|>':
                    end_pattern = r'</\|tool_call\|>'
                elif pattern == r'<function_call>':
                    end_pattern = r'</function_call>'
                elif pattern == r'<tool_call>':
                    end_pattern = r'</tool_call>'
                elif pattern == r'```tool_code':
                    end_pattern = r'```'
                else:
                    continue
                
                if not re.search(end_pattern, text):
                    return True
        
        return False
    
    def _is_plain_json_tool_call(self, text: str) -> bool:
        """Check if text is a plain JSON tool call."""
        try:
            import json
            # Try to parse as JSON
            data = json.loads(text.strip())
            return isinstance(data, dict) and "name" in data
        except:
            return False
    
    def _find_json_objects(self, text: str) -> List[str]:
        """Find all potential JSON objects in text by looking for balanced braces."""
        json_objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Find the matching closing brace
                brace_count = 0
                start = i
                for j in range(i, len(text)):
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found balanced braces
                            potential_json = text[start:j+1]
                            # Check if it contains "name" (likely a tool call)
                            if '"name"' in potential_json:
                                json_objects.append(potential_json)
                            break
                i = j + 1
            else:
                i += 1
        return json_objects
    
    def _rewrite_complete_tool_call(self, text: str) -> str:
        """Rewrite a complete tool call to target format."""
        # Find the tool call in the text
        source_patterns = [
            (r'<\|tool_call\|>(.*?)</\|tool_call\|>', r'\1'),
            (r'<function_call>(.*?)</function_call>', r'\1'),
            (r'<tool_call>(.*?)</tool_call>', r'\1'),
            (r'```tool_code(.*?)```', r'\1'),
        ]

        for pattern, replacement in source_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # Extract the JSON content
                json_content = match.group(1).strip()
                # Wrap in target format using output tags (with angle brackets)
                return f"{self._output_start_tag}{json_content}{self._output_end_tag}"

        # Check for plain JSON
        if self._is_plain_json_tool_call(text):
            return f"{self._output_start_tag}{text.strip()}{self._output_end_tag}"

        # If no pattern matches, return original text
        return text
    
    def is_tool_call(self, text: str) -> bool:
        """
        Check if text contains a tool call.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains a tool call
        """
        return any(pattern.search(text) for pattern, _ in self._compiled_patterns)
    
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from text.
        
        Args:
            text: Text containing tool calls
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        # Find all tool call patterns
        for pattern, _ in self._compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    # Try to parse as JSON
                    tool_data = json.loads(match.strip())
                    if "name" in tool_data:
                        tool_calls.append(tool_data)
                except json.JSONDecodeError:
                    continue
        
        return tool_calls


# Predefined tag configurations for common agentic CLIs
PREDEFINED_TAGS = {
    "qwen3": ToolCallTags("<|tool_call|>", "</|tool_call|>"),
    "llama3": ToolCallTags("<function_call>", "</function_call>"),
    "xml": ToolCallTags("<tool_call>", "</tool_call>"),
    "gemma": ToolCallTags("```tool_code", "```"),
    "codex": ToolCallTags("<|tool_call|>", "</|tool_call|>"),  # Codex uses Qwen3 format
    "crush": ToolCallTags("<function_call>", "</function_call>"),  # Crush uses LLaMA3 format
    "gemini": ToolCallTags("<tool_call>", "</tool_call>"),  # Gemini uses XML format
    "openai": ToolCallTags("<|tool_call|>", "</|tool_call|>"),  # OpenAI uses Qwen3 format
    "anthropic": ToolCallTags("<function_call>", "</function_call>"),  # Anthropic uses LLaMA3 format
    "custom": ToolCallTags("<custom_tool>", "</custom_tool>"),  # Custom format
}


def get_predefined_tags(cli_name: str) -> ToolCallTags:
    """
    Get predefined tag configuration for a CLI.
    
    Args:
        cli_name: Name of the CLI (e.g., "qwen3", "llama3", "codex")
        
    Returns:
        ToolCallTags configuration
        
    Raises:
        ValueError: If CLI name is not recognized
    """
    if cli_name not in PREDEFINED_TAGS:
        raise ValueError(f"Unknown CLI name: {cli_name}. Available: {list(PREDEFINED_TAGS.keys())}")
    
    return PREDEFINED_TAGS[cli_name]


def create_tag_rewriter(cli_name: str = "qwen3") -> ToolCallTagRewriter:
    """
    Create a tag rewriter for a specific CLI.
    
    Args:
        cli_name: Name of the CLI (e.g., "qwen3", "llama3", "codex")
        
    Returns:
        ToolCallTagRewriter instance
    """
    tags = get_predefined_tags(cli_name)
    return ToolCallTagRewriter(tags)
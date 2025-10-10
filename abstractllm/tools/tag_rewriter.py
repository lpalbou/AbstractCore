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
    """
    start_tag: str
    end_tag: str
    preserve_json: bool = True
    
    def __post_init__(self):
        """Validate tag configuration."""
        if not self.start_tag or not self.end_tag:
            raise ValueError("Both start_tag and end_tag must be provided")
        
        # Ensure tags are properly formatted
        if not self.start_tag.startswith('<'):
            self.start_tag = f'<{self.start_tag}>'
        if not self.end_tag.startswith('</'):
            self.end_tag = f'</{self.end_tag.split(">")[0].split("<")[-1]}>'


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
        self._compiled_patterns = self._compile_patterns()
    
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
        patterns.append((qwen_pattern, f"{self.target_tags.start_tag}\\1{self.target_tags.end_tag}"))
        
        # Pattern 2: LLaMA3 format <function_call>...JSON...</function_call>
        llama_pattern = re.compile(
            r'<function_call>\s*(.*?)\s*</function_call>',
            re.DOTALL | re.IGNORECASE
        )
        patterns.append((llama_pattern, f"{self.target_tags.start_tag}\\1{self.target_tags.end_tag}"))
        
        # Pattern 3: XML format <tool_call>...JSON...</tool_call>
        xml_pattern = re.compile(
            r'<tool_call>\s*(.*?)\s*</tool_call>',
            re.DOTALL | re.IGNORECASE
        )
        patterns.append((xml_pattern, f"{self.target_tags.start_tag}\\1{self.target_tags.end_tag}"))
        
        # Pattern 4: Gemma format ```tool_code...JSON...```
        gemma_pattern = re.compile(
            r'```tool_code\s*\n(.*?)\n```',
            re.DOTALL | re.IGNORECASE
        )
        patterns.append((gemma_pattern, f"{self.target_tags.start_tag}\\1{self.target_tags.end_tag}"))
        
        # Pattern 5: Generic JSON (wrap in tags) - only standalone JSON
        json_pattern = re.compile(
            r'(?<![<])(\{[^{}]*["\']name["\'][^{}]*(?:\{[^{}]*\}[^{}]*)*\})(?![>])',
            re.DOTALL
        )
        patterns.append((json_pattern, f"{self.target_tags.start_tag}\\1{self.target_tags.end_tag}"))
        
        return patterns
    
    def rewrite_text(self, text: str) -> str:
        """
        Rewrite tool call tags in text.
        
        Args:
            text: Input text containing tool calls
            
        Returns:
            Text with rewritten tool call tags
        """
        if not text or not self.target_tags.preserve_json:
            return text
        
        # Check if we already have the target format (avoid double-tagging)
        if (self.target_tags.start_tag in text and 
            self.target_tags.end_tag in text):
            # Already in target format, just return as-is
            return text
        
        rewritten = text
        
        # Apply each pattern only once to avoid double-processing
        for pattern, replacement in self._compiled_patterns:
            if pattern.search(rewritten):
                rewritten = pattern.sub(replacement, rewritten, count=1)
                break  # Only apply the first matching pattern
        
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
        if (self.target_tags.start_tag in full_text and 
            self.target_tags.end_tag in full_text):
            # Already in target format, just return as-is
            return full_text, ""
        
        # SOTA Strategy: Immediate rewriting with buffer management
        rewritten_chunk = ""
        new_buffer = buffer
        
        # Process the chunk character by character for precise control
        i = 0
        while i < len(chunk):
            char = chunk[i]
            
            # Add character to buffer
            new_buffer += char
            
            # Check if we have a complete tool call in the buffer
            if self._has_complete_tool_call(new_buffer):
                # Rewrite the complete tool call
                rewritten_tool_call = self._rewrite_complete_tool_call(new_buffer)
                rewritten_chunk += rewritten_tool_call
                new_buffer = ""  # Clear buffer after processing
            elif self._has_incomplete_tool_call(new_buffer):
                # We have a start tag but no end tag yet
                # Don't output anything yet, keep buffering
                pass
            elif self._is_plain_json_tool_call(new_buffer):
                # We have a complete plain JSON tool call
                rewritten_tool_call = self._rewrite_complete_tool_call(new_buffer)
                rewritten_chunk += rewritten_tool_call
                new_buffer = ""  # Clear buffer after processing
            else:
                # Check if we're in the middle of a potential plain JSON tool call
                if (new_buffer.strip().startswith('{') and 
                    not new_buffer.strip().endswith('}') and
                    '"name"' in new_buffer):
                    # We're in the middle of a potential JSON tool call, keep buffering
                    pass
                else:
                    # No tool call detected, output the character
                    rewritten_chunk += char
                    new_buffer = ""  # Clear buffer if no tool call
            
            i += 1
        
        return rewritten_chunk, new_buffer
    
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
                # Wrap in target format
                return f"{self.target_tags.start_tag}{json_content}{self.target_tags.end_tag}"
        
        # Check for plain JSON
        if self._is_plain_json_tool_call(text):
            return f"{self.target_tags.start_tag}{text.strip()}{self.target_tags.end_tag}"
        
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
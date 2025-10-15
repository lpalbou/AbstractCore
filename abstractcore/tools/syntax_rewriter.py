"""
Tool Call Syntax Rewriter for Agent Compatibility

This module provides comprehensive tool call syntax conversion beyond simple tag rewriting.
Supports multiple target formats including OpenAI, Codex, and custom agent formats.
"""

import re
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .core import ToolCall
from .parser import parse_tool_calls

logger = logging.getLogger(__name__)


class SyntaxFormat(Enum):
    """Supported tool call syntax formats."""

    # Standard formats
    PASSTHROUGH = "passthrough"        # No changes (for OpenAI models)
    OPENAI = "openai"                 # Full OpenAI structure
    CODEX = "codex"                   # OpenAI format optimized for Codex

    # Model-specific formats
    QWEN3 = "qwen3"                   # <|tool_call|>...JSON...</|tool_call|>
    LLAMA3 = "llama3"                 # <function_call>...JSON...</function_call>
    GEMMA = "gemma"                   # ```tool_code...JSON...```
    XML = "xml"                       # <tool_call>...JSON...</tool_call>

    # Custom formats
    CUSTOM = "custom"                 # User-defined format


@dataclass
class CustomFormatConfig:
    """Configuration for custom tool call formats."""
    start_tag: str
    end_tag: str
    json_wrapper: bool = True  # Whether to wrap JSON in tags
    add_ids: bool = False      # Whether to add call IDs
    format_template: Optional[str] = None  # Custom template


class ToolCallSyntaxRewriter:
    """
    Universal tool call syntax rewriter for agent compatibility.

    Converts tool calls from any detected format to the target format
    required by different agentic CLIs and systems.
    """

    def __init__(
        self,
        target_format: Union[str, SyntaxFormat],
        custom_config: Optional[CustomFormatConfig] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the syntax rewriter.

        Args:
            target_format: Target format to convert to
            custom_config: Configuration for custom formats
            model_name: Model name for format auto-detection
        """
        if isinstance(target_format, str):
            try:
                self.target_format = SyntaxFormat(target_format.lower())
            except ValueError:
                raise ValueError(f"Unsupported format: {target_format}")
        else:
            self.target_format = target_format

        self.custom_config = custom_config
        self.model_name = model_name

        # Validate custom format configuration
        if self.target_format == SyntaxFormat.CUSTOM and not custom_config:
            raise ValueError("Custom format requires CustomFormatConfig")

    def rewrite_content(
        self,
        content: str,
        detected_tool_calls: Optional[List[ToolCall]] = None
    ) -> str:
        """
        Rewrite tool call syntax in content.

        Args:
            content: Original content with tool calls
            detected_tool_calls: Pre-detected tool calls (optional)

        Returns:
            Content with rewritten tool call syntax
        """
        if not content or not content.strip():
            return content

        # Passthrough mode - return unchanged
        if self.target_format == SyntaxFormat.PASSTHROUGH:
            return content

        # Detect tool calls if not provided
        if detected_tool_calls is None:
            detected_tool_calls = parse_tool_calls(content, self.model_name)
            logger.debug(f"Detected {len(detected_tool_calls)} tool calls in content")

        # No tool calls found
        if not detected_tool_calls:
            return content

        # Apply format-specific rewriting
        return self._apply_format_conversion(content, detected_tool_calls)

    def convert_to_openai_format(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        """
        Convert tool calls to OpenAI API format.

        Args:
            tool_calls: List of detected tool calls

        Returns:
            List of OpenAI-formatted tool call dictionaries
        """
        openai_tools = []

        for tool_call in tool_calls:
            # Ensure we have a call ID
            call_id = tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}"

            # Convert arguments to JSON string if needed
            if isinstance(tool_call.arguments, dict):
                arguments_str = json.dumps(tool_call.arguments)
            elif isinstance(tool_call.arguments, str):
                # Validate it's valid JSON
                try:
                    json.loads(tool_call.arguments)
                    arguments_str = tool_call.arguments
                except json.JSONDecodeError:
                    # Wrap in JSON if it's not valid JSON
                    arguments_str = json.dumps({"value": tool_call.arguments})
            else:
                arguments_str = json.dumps(tool_call.arguments)

            openai_tool = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": arguments_str
                }
            }

            openai_tools.append(openai_tool)

        return openai_tools

    def _apply_format_conversion(self, content: str, tool_calls: List[ToolCall]) -> str:
        """Apply format-specific conversion."""

        if self.target_format in [SyntaxFormat.OPENAI, SyntaxFormat.CODEX]:
            return self._convert_to_openai_content(content, tool_calls)
        elif self.target_format == SyntaxFormat.QWEN3:
            return self._convert_to_qwen3(content, tool_calls)
        elif self.target_format == SyntaxFormat.LLAMA3:
            return self._convert_to_llama3(content, tool_calls)
        elif self.target_format == SyntaxFormat.GEMMA:
            return self._convert_to_gemma(content, tool_calls)
        elif self.target_format == SyntaxFormat.XML:
            return self._convert_to_xml(content, tool_calls)
        elif self.target_format == SyntaxFormat.CUSTOM:
            return self._convert_to_custom(content, tool_calls)
        else:
            logger.warning(f"Unsupported format {self.target_format}, returning original content")
            return content

    def _convert_to_openai_content(self, content: str, tool_calls: List[ToolCall]) -> str:
        """
        Convert to OpenAI content format.

        Note: For server integration, this mainly removes tool call syntax
        since OpenAI format is handled at the API response level.
        """
        # Remove existing tool call syntax patterns
        cleaned_content = self.remove_tool_call_patterns(content)

        # For OpenAI format, the tool calls are handled separately in the API response
        # So we just return the cleaned content
        return cleaned_content.strip()

    def _convert_to_qwen3(self, content: str, tool_calls: List[ToolCall]) -> str:
        """Convert to Qwen3 format: <|tool_call|>...JSON...</|tool_call|>"""
        # Remove existing tool call patterns
        cleaned_content = self.remove_tool_call_patterns(content)

        # Add tool calls in Qwen3 format
        for tool_call in tool_calls:
            tool_json = {
                "name": tool_call.name,
                "arguments": tool_call.arguments
            }

            qwen_format = f"<|tool_call|>\n{json.dumps(tool_json, indent=2)}\n</|tool_call|>"
            cleaned_content += f"\n\n{qwen_format}"

        return cleaned_content.strip()

    def _convert_to_llama3(self, content: str, tool_calls: List[ToolCall]) -> str:
        """Convert to LLaMA3 format: <function_call>...JSON...</function_call>"""
        # Remove existing tool call patterns
        cleaned_content = self.remove_tool_call_patterns(content)

        # Add tool calls in LLaMA3 format
        for tool_call in tool_calls:
            tool_json = {
                "name": tool_call.name,
                "arguments": tool_call.arguments
            }

            llama_format = f"<function_call>\n{json.dumps(tool_json, indent=2)}\n</function_call>"
            cleaned_content += f"\n\n{llama_format}"

        return cleaned_content.strip()

    def _convert_to_gemma(self, content: str, tool_calls: List[ToolCall]) -> str:
        """Convert to Gemma format: ```tool_code...JSON...```"""
        # Remove existing tool call patterns
        cleaned_content = self.remove_tool_call_patterns(content)

        # Add tool calls in Gemma format
        for tool_call in tool_calls:
            tool_json = {
                "name": tool_call.name,
                "arguments": tool_call.arguments
            }

            gemma_format = f"```tool_code\n{json.dumps(tool_json, indent=2)}\n```"
            cleaned_content += f"\n\n{gemma_format}"

        return cleaned_content.strip()

    def _convert_to_xml(self, content: str, tool_calls: List[ToolCall]) -> str:
        """Convert to XML format: <tool_call>...JSON...</tool_call>"""
        # Remove existing tool call patterns
        cleaned_content = self.remove_tool_call_patterns(content)

        # Add tool calls in XML format
        for tool_call in tool_calls:
            tool_json = {
                "name": tool_call.name,
                "arguments": tool_call.arguments
            }

            xml_format = f"<tool_call>\n{json.dumps(tool_json, indent=2)}\n</tool_call>"
            cleaned_content += f"\n\n{xml_format}"

        return cleaned_content.strip()

    def _convert_to_custom(self, content: str, tool_calls: List[ToolCall]) -> str:
        """Convert to custom format based on configuration."""
        if not self.custom_config:
            return content

        # Remove existing tool call patterns
        cleaned_content = self.remove_tool_call_patterns(content)

        # Add tool calls in custom format
        for tool_call in tool_calls:
            if self.custom_config.format_template:
                # Use custom template
                custom_format = self.custom_config.format_template.format(
                    name=tool_call.name,
                    arguments=json.dumps(tool_call.arguments),
                    call_id=tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}"
                )
            else:
                # Use basic tag format
                if self.custom_config.json_wrapper:
                    tool_json = {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                    if self.custom_config.add_ids:
                        tool_json["id"] = tool_call.call_id or f"call_{uuid.uuid4().hex[:8]}"

                    content_str = json.dumps(tool_json, indent=2)
                else:
                    content_str = f"{tool_call.name}({json.dumps(tool_call.arguments)})"

                custom_format = f"{self.custom_config.start_tag}\n{content_str}\n{self.custom_config.end_tag}"

            cleaned_content += f"\n\n{custom_format}"

        return cleaned_content.strip()

    def remove_tool_call_patterns(self, content: str) -> str:
        """
        Remove existing tool call patterns from content.
        
        This method intelligently removes tool call syntax while preserving
        the surrounding text content. It handles multiple formats and edge cases
        like double-tagging and malformed tool calls.
        """
        if not content:
            return content

        cleaned = content
        
        # Remove internal conversation format tags that shouldn't appear
        # These indicate model output issues
        cleaned = re.sub(r'<\|assistant\|>', '', cleaned)
        cleaned = re.sub(r'<\|user\|>', '', cleaned)
        cleaned = re.sub(r'<\|system\|>', '', cleaned)
        
        # Common tool call patterns to remove (in order of specificity)
        patterns = [
            # Qwen format (with potential double-tagging)
            r'<\|tool_call\|>+\s*',  # Opening tags (including doubles)
            r'\s*</\|tool_call\|>+',  # Closing tags (including doubles)
            # After removing tags, remove the JSON content if it's a tool call
            r'<\|tool_call\|>.*?</\|tool_call\|>',
            
            # LLaMA format
            r'<function_call>\s*',
            r'\s*</function_call>',
            r'<function_call>.*?</function_call>',
            
            # XML format
            r'<tool_call>\s*',
            r'\s*</tool_call>',
            r'<tool_call>.*?</tool_call>',
            
            # Gemma format
            r'```tool_code\s*',
            r'\s*```(?=\s|$)',  # Closing backticks only if followed by space or end
            r'```tool_code.*?```',
        ]

        # First pass: remove complete tool call blocks
        complete_patterns = [
            r'<\|tool_call\|>.*?</\|tool_call\|>',
            r'<function_call>.*?</function_call>',
            r'<tool_call>.*?</tool_call>',
            r'```tool_code.*?```',
        ]
        
        for pattern in complete_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Second pass: remove orphaned tags (from malformed tool calls)
        orphaned_patterns = [
            r'<\|tool_call\|>+',
            r'</\|tool_call\|>+',
            r'<function_call>',
            r'</function_call>',
            r'<tool_call>',
            r'</tool_call>',
            r'```tool_code',
        ]
        
        for pattern in orphaned_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Third pass: remove standalone JSON tool calls that weren't wrapped in tags
        # Be careful here - only remove if it looks like a tool call
        # (has "name" and "arguments" fields at the top level)
        json_pattern = r'^\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}\s*$'
        lines = cleaned.split('\n')
        cleaned_lines = []
        for line in lines:
            if not re.match(json_pattern, line, re.MULTILINE):
                cleaned_lines.append(line)
        cleaned = '\n'.join(cleaned_lines)
        
        # Clean up extra whitespace (but preserve paragraph breaks)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = re.sub(r'^\s+', '', cleaned)  # Leading whitespace
        cleaned = re.sub(r'\s+$', '', cleaned)  # Trailing whitespace

        return cleaned.strip()


# Convenience functions for common formats

def create_openai_rewriter(model_name: Optional[str] = None) -> ToolCallSyntaxRewriter:
    """Create a rewriter for OpenAI format."""
    return ToolCallSyntaxRewriter(SyntaxFormat.OPENAI, model_name=model_name)


def create_codex_rewriter(model_name: Optional[str] = None) -> ToolCallSyntaxRewriter:
    """Create a rewriter optimized for Codex."""
    return ToolCallSyntaxRewriter(SyntaxFormat.CODEX, model_name=model_name)


def create_passthrough_rewriter() -> ToolCallSyntaxRewriter:
    """Create a passthrough rewriter (no changes)."""
    return ToolCallSyntaxRewriter(SyntaxFormat.PASSTHROUGH)


def create_custom_rewriter(
    start_tag: str,
    end_tag: str,
    json_wrapper: bool = True,
    add_ids: bool = False,
    format_template: Optional[str] = None,
    model_name: Optional[str] = None
) -> ToolCallSyntaxRewriter:
    """Create a rewriter for custom format."""
    config = CustomFormatConfig(
        start_tag=start_tag,
        end_tag=end_tag,
        json_wrapper=json_wrapper,
        add_ids=add_ids,
        format_template=format_template
    )
    return ToolCallSyntaxRewriter(SyntaxFormat.CUSTOM, custom_config=config, model_name=model_name)


def auto_detect_format(
    model: str,
    user_agent: str = "",
    custom_headers: Optional[Dict[str, str]] = None
) -> SyntaxFormat:
    """
    Auto-detect the appropriate target format based on context.

    Args:
        model: Model name/identifier
        user_agent: User-Agent header
        custom_headers: Additional headers to check

    Returns:
        Detected target format
    """
    model_lower = model.lower()
    user_agent_lower = user_agent.lower()

    # Check custom headers for agent hints
    if custom_headers:
        agent_header = custom_headers.get("X-Agent-Type", "").lower()
        if agent_header in ["codex", "openai"]:
            return SyntaxFormat.CODEX
        elif agent_header in ["qwen", "qwen3"]:
            return SyntaxFormat.QWEN3
        elif agent_header in ["llama", "llama3"]:
            return SyntaxFormat.LLAMA3

    # Check for specific agents in user agent
    if "codex" in user_agent_lower:
        return SyntaxFormat.CODEX

    # Check model patterns
    if model.startswith("openai/") or "gpt-" in model_lower:
        return SyntaxFormat.PASSTHROUGH  # Use passthrough for OpenAI models
    elif "qwen" in model_lower:
        return SyntaxFormat.QWEN3
    elif "llama" in model_lower:
        return SyntaxFormat.LLAMA3
    elif "gemma" in model_lower:
        return SyntaxFormat.GEMMA
    elif "claude" in model_lower:
        return SyntaxFormat.XML

    # Default to OpenAI format for maximum compatibility
    return SyntaxFormat.OPENAI
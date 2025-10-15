"""
Architecture-based tool call parsing and formatting.

This module handles the detection and parsing of tool calls from model
responses based on their architecture.
"""

import re
import json
import logging
from typing import List, Optional, Dict, Any
from enum import Enum

from .core import ToolCall, ToolDefinition
from ..architectures import detect_architecture, get_architecture_format

logger = logging.getLogger(__name__)


class ToolFormat(Enum):
    """Tool call formats for different architectures."""

    # JSON-based
    RAW_JSON = "raw_json"              # {"name": "...", "arguments": {...}}
    FUNCTION_CALL = "function_call"    # <function_call>...</function_call>
    SPECIAL_TOKEN = "special_token"    # <|tool_call|>...

    # Code-based
    TOOL_CODE = "tool_code"           # ```tool_code\nfunc(...)```

    # XML-based
    XML_WRAPPED = "xml_wrapped"       # <tool_call>...</tool_call>

    # Native
    NATIVE = "native"                 # API-level tool calls


def _has_json_tool_pattern(response: str) -> bool:
    """Check if response contains JSON tool call pattern."""
    # Look for JSON objects that look like tool calls
    json_pattern = r'\{[^{}]*["\']name["\'][^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    return bool(re.search(json_pattern, response, re.DOTALL))


def detect_tool_calls(response: str, model_name: Optional[str] = None) -> bool:
    """
    Detect if response contains tool calls.

    Args:
        response: Model response text
        model_name: Optional model name for architecture detection

    Returns:
        True if tool calls detected
    """
    if not response or not response.strip():
        return False

    # Get expected format from architecture
    tool_format = _get_tool_format(model_name)

    # Check format-specific patterns (case-insensitive)
    response_lower = response.lower()
    if tool_format == ToolFormat.TOOL_CODE:
        return "```tool_code" in response_lower or "```tool_call" in response_lower
    elif tool_format == ToolFormat.SPECIAL_TOKEN:
        return "<|tool_call|>" in response_lower
    elif tool_format == ToolFormat.FUNCTION_CALL:
        return "<function_call" in response_lower or _has_json_tool_pattern(response)
    elif tool_format == ToolFormat.XML_WRAPPED:
        return "<tool_call>" in response_lower
    else:
        # Try common patterns (case-insensitive)
        return any([
            "```tool_code" in response_lower,
            "```tool_call" in response_lower,
            "<|tool_call|>" in response_lower,
            "<function_call" in response_lower,
            "<tool_call>" in response_lower,
            _has_json_tool_pattern(response),
        ])
    
    # Additional check for plain JSON when no specific format is detected
    if tool_format == ToolFormat.RAW_JSON:
        return _has_json_tool_pattern(response)


def parse_tool_calls(response: str, model_name: Optional[str] = None) -> List[ToolCall]:
    """
    Parse tool calls from response.

    Args:
        response: Model response containing tool calls
        model_name: Optional model name for architecture detection

    Returns:
        List of parsed tool calls
    """
    if not response or not response.strip():
        return []

    # Get expected format
    tool_format = _get_tool_format(model_name)

    # Parse based on format
    parsers = {
        ToolFormat.TOOL_CODE: _parse_tool_code,
        ToolFormat.SPECIAL_TOKEN: _parse_special_token,
        ToolFormat.FUNCTION_CALL: _parse_function_call,
        ToolFormat.XML_WRAPPED: _parse_xml_wrapped,
        ToolFormat.RAW_JSON: _parse_raw_json,
        ToolFormat.NATIVE: _parse_any_format  # Native tools use any format as fallback
    }

    parser = parsers.get(tool_format, _parse_any_format)
    return parser(response)


def format_tool_prompt(tools: List[ToolDefinition], model_name: Optional[str] = None) -> str:
    """
    Format tools into a system prompt based on model architecture.

    Args:
        tools: List of tool definitions
        model_name: Optional model name for architecture detection

    Returns:
        Formatted system prompt
    """
    if not tools:
        return "You are a helpful AI assistant."

    # Get tool format
    tool_format = _get_tool_format(model_name)

    # Format based on architecture
    if tool_format == ToolFormat.TOOL_CODE:
        return _format_gemma_style(tools)
    elif tool_format == ToolFormat.SPECIAL_TOKEN:
        return _format_qwen_style(tools)
    elif tool_format == ToolFormat.FUNCTION_CALL:
        return _format_llama_style(tools)
    elif tool_format == ToolFormat.XML_WRAPPED:
        return _format_xml_style(tools)
    else:
        return _format_generic_style(tools)


# Internal helpers

def _get_tool_format(model_name: Optional[str]) -> ToolFormat:
    """Get tool format for a model."""
    if not model_name:
        return ToolFormat.RAW_JSON

    architecture = detect_architecture(model_name)
    arch_format = get_architecture_format(architecture)

    tool_format = arch_format.get("tool_format", "json")

    if tool_format == "special_token":
        return ToolFormat.SPECIAL_TOKEN
    elif tool_format == "xml":
        return ToolFormat.XML_WRAPPED
    elif tool_format == "pythonic":
        return ToolFormat.TOOL_CODE
    elif tool_format == "native":
        return ToolFormat.NATIVE
    else:
        return ToolFormat.FUNCTION_CALL




def _parse_special_token(response: str) -> List[ToolCall]:
    """Parse Qwen-style <|tool_call|> format with robust fallback."""
    tool_calls = []
    
    # Pre-process: Remove markdown code fences that might wrap tool calls
    # This handles cases like ```json\n<|tool_call|>...\n```
    cleaned_response = re.sub(r'```(?:json|python|tool_code|tool_call)?\s*\n', '', response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r'\n```\s*(?=\n|$)', '', cleaned_response)

    # First, find all tool call positions to avoid duplicates from overlapping patterns
    all_matches = []

    # Strategy 1: Look for properly closed tags
    pattern_with_close = r'<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>'
    for match in re.finditer(pattern_with_close, cleaned_response, re.DOTALL | re.IGNORECASE):
        all_matches.append((match.start(), match.end(), match.group(1).strip()))

    # Strategy 2: Look for opening tags followed by valid JSON (no closing tag)
    pattern_no_close = r'<\|tool_call\|>\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})\s*(?:</\|tool_call\|>|$|\n|<)'
    for match in re.finditer(pattern_no_close, cleaned_response, re.DOTALL | re.IGNORECASE):
        # Check if this match overlaps with any closed tag match
        overlaps = False
        for closed_start, closed_end, _ in all_matches:
            if match.start() >= closed_start and match.start() < closed_end:
                overlaps = True
                break
        if not overlaps:
            all_matches.append((match.start(), match.end(), match.group(1).strip()))

    # Strategy 3: Ultra-robust pattern - just find start tag + JSON, ignore ending completely
    pattern_start_json = r'<\|tool_call\|>\s*(\{[^<]*?\})'
    for match in re.finditer(pattern_start_json, cleaned_response, re.DOTALL | re.IGNORECASE):
        # Check if this match overlaps with any previous matches
        overlaps = False
        for prev_start, prev_end, _ in all_matches:
            if match.start() >= prev_start and match.start() < prev_end:
                overlaps = True
                break
        if not overlaps:
            json_candidate = match.group(1).strip()
            # Basic validation that it looks like JSON and contains tool structure
            # Accept "name", "command", or "function" as valid tool identifiers
            if (json_candidate.startswith('{') and json_candidate.endswith('}') and
                ('"name"' in json_candidate or '"command"' in json_candidate or '"function"' in json_candidate)):
                all_matches.append((match.start(), match.end(), json_candidate))

    # Sort by position and parse each match
    all_matches.sort(key=lambda x: x[0])
    for _, _, json_str in all_matches:
        try:
            # Clean up the JSON string - remove any trailing content that might interfere
            json_str = json_str.strip()

            # Handle cases where there might be trailing text after the JSON
            if json_str.count('{') > json_str.count('}'):
                # Missing closing braces - try to add them
                missing_braces = json_str.count('{') - json_str.count('}')
                json_str += '}' * missing_braces

            # Try to find the JSON object boundaries more precisely
            brace_count = 0
            json_end = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end > 0:
                json_str = json_str[:json_end]

            # Try normal JSON parsing first
            try:
                tool_data = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback: fix common LLM JSON issues (unescaped newlines)
                fixed_json = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                tool_data = json.loads(fixed_json)

            if isinstance(tool_data, dict):
                # Normalize field names: accept "name", "command", "function", "tool", "action"
                tool_name = None
                if "name" in tool_data:
                    tool_name = tool_data["name"]
                elif "command" in tool_data:
                    tool_name = tool_data["command"]
                elif "function" in tool_data:
                    tool_name = tool_data["function"]
                elif "tool" in tool_data:
                    tool_name = tool_data["tool"]
                elif "action" in tool_data:
                    tool_name = tool_data["action"]
                
                if tool_name:
                    tool_calls.append(ToolCall(
                        name=tool_name,
                        arguments=tool_data.get("arguments", tool_data.get("params", tool_data.get("parameters", {}))),
                        call_id=tool_data.get("id")
                    ))
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error for tool call: {e}, JSON string: {repr(json_str)}")
            continue

    return tool_calls


def _parse_function_call(response: str) -> List[ToolCall]:
    """Parse LLaMA-style <function_call> format."""
    tool_calls = []

    # Pattern for function call format
    pattern = r'<function_call>\s*({.*?})\s*</function_call>'

    for match in re.finditer(pattern, response, re.DOTALL):
        try:
            json_str = match.group(1)
            tool_data = json.loads(json_str)

            tool_call = ToolCall(
                name=tool_data.get("name", ""),
                arguments=tool_data.get("arguments", tool_data.get("parameters", {})),
                call_id=tool_data.get("id")
            )
            tool_calls.append(tool_call)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse function call JSON: {json_str} - {e}")

    return tool_calls


def _parse_xml_wrapped(response: str) -> List[ToolCall]:
    """Parse XML-wrapped tool calls."""
    tool_calls = []

    # Pattern for XML format
    pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'

    for match in re.finditer(pattern, response, re.DOTALL):
        try:
            json_str = match.group(1)
            tool_data = json.loads(json_str)

            tool_call = ToolCall(
                name=tool_data.get("name", ""),
                arguments=tool_data.get("arguments", {}),
                call_id=tool_data.get("id")
            )
            tool_calls.append(tool_call)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse XML tool call JSON: {json_str} - {e}")

    return tool_calls


def _parse_tool_code(response: str) -> List[ToolCall]:
    """Parse tool_code block format."""
    tool_calls = []

    # Pattern for code block format
    pattern = r'```tool_code\s*\n(.*?)\n```'

    for match in re.finditer(pattern, response, re.DOTALL):
        code_content = match.group(1).strip()

        # Try to parse as JSON first
        try:
            tool_data = json.loads(code_content)
            tool_call = ToolCall(
                name=tool_data.get("name", ""),
                arguments=tool_data.get("arguments", {}),
                call_id=tool_data.get("id")
            )
            tool_calls.append(tool_call)

        except json.JSONDecodeError:
            # Try to parse as function call
            func_pattern = r'(\w+)\s*\((.*?)\)'
            func_match = re.search(func_pattern, code_content)

            if func_match:
                func_name = func_match.group(1)
                args_str = func_match.group(2)

                # Simple argument parsing (could be enhanced)
                arguments = {}
                if args_str.strip():
                    try:
                        # Try eval for simple cases (be careful!)
                        arguments = eval(f"dict({args_str})")
                    except:
                        logger.warning(f"Failed to parse function arguments: {args_str}")

                tool_call = ToolCall(
                    name=func_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)

    return tool_calls


def _parse_raw_json(response: str) -> List[ToolCall]:
    """Parse raw JSON tool calls."""
    tool_calls = []

    # Try to find JSON objects that look like tool calls - handle nested objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*["\']name["\'][^{}]*(?:\{[^{}]*\}[^{}]*)*\}'

    for match in re.finditer(json_pattern, response):
        try:
            json_str = match.group(0)
            tool_data = json.loads(json_str)

            if "name" in tool_data:
                tool_call = ToolCall(
                    name=tool_data.get("name", ""),
                    arguments=tool_data.get("arguments", tool_data.get("parameters", {})),
                    call_id=tool_data.get("id")
                )
                tool_calls.append(tool_call)

        except json.JSONDecodeError:
            continue

    # Also try to parse JSON from code blocks
    code_block_pattern = r'```(?:json|tool_code)?\s*\n(\{.*?\})\s*\n```'
    for match in re.finditer(code_block_pattern, response, re.DOTALL):
        try:
            json_str = match.group(1).strip()
            tool_data = json.loads(json_str)

            if "name" in tool_data:
                tool_call = ToolCall(
                    name=tool_data.get("name", ""),
                    arguments=tool_data.get("arguments", tool_data.get("parameters", {})),
                    call_id=tool_data.get("id")
                )
                tool_calls.append(tool_call)

        except json.JSONDecodeError:
            continue

    return tool_calls


def _parse_any_format(response: str) -> List[ToolCall]:
    """Try all parsing formats with comprehensive fallbacks."""
    tool_calls = []

    # Try each parser and accumulate results
    parsers = [
        _parse_special_token,
        _parse_function_call,
        _parse_xml_wrapped,
        _parse_tool_code,
        _parse_raw_json
    ]

    for parser in parsers:
        try:
            found_calls = parser(response)
            tool_calls.extend(found_calls)
        except Exception as e:
            logger.debug(f"Parser {parser.__name__} failed: {e}")

    # Additional fallback: Look for Python code blocks with common tool names
    if not tool_calls:
        tool_calls.extend(_parse_python_code_blocks(response))

    # Remove duplicates (same name and arguments)
    unique_calls = []
    seen = set()
    for call in tool_calls:
        call_key = (call.name, str(call.arguments))
        if call_key not in seen:
            seen.add(call_key)
            unique_calls.append(call)

    return unique_calls


def _parse_python_code_blocks(response: str) -> List[ToolCall]:
    """Parse Python code blocks that might contain tool calls."""
    tool_calls = []

    # Look for common tool function calls in code blocks
    common_tools = ["list_files", "read_file", "search_files", "write_file"]

    for tool_name in common_tools:
        # Pattern for tool calls in code blocks
        pattern = rf'```(?:python|json)?\s*\n.*?{tool_name}\(([^)]*)\).*?\n```'
        for match in re.finditer(pattern, response, re.DOTALL):
            args_str = match.group(1).strip()
            arguments = {}

            # Parse simple keyword arguments if any
            if args_str:
                arg_pattern = r'(\w+)\s*=\s*([^,)]+)'
                for arg_match in re.finditer(arg_pattern, args_str):
                    key = arg_match.group(1)
                    value = arg_match.group(2).strip()

                    # Parse value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)

                    arguments[key] = value

            tool_calls.append(ToolCall(name=tool_name, arguments=arguments))

    return tool_calls


# Formatting functions

def _format_qwen_style(tools: List[ToolDefinition]) -> str:
    """Format tools for Qwen models using <|tool_call|> format with enhanced metadata."""
    if not tools:
        return ""

    prompt = "You are a helpful AI assistant with access to the following tools:\n\n"

    # Tool descriptions with enhanced metadata
    for tool in tools:
        prompt += f"**{tool.name}**: {tool.description}\n"

        # Add when_to_use guidance if available
        if tool.when_to_use:
            prompt += f"  • **When to use**: {tool.when_to_use}\n"

        # Add tags if available
        if tool.tags:
            prompt += f"  • **Tags**: {', '.join(tool.tags)}\n"

        if tool.parameters:
            prompt += f"  • **Parameters**: {json.dumps(tool.parameters, indent=2)}\n"
        prompt += "\n"

    prompt += """To use a tool, respond with this EXACT format:
<|tool_call|>
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
</|tool_call|>
""" + _critical_rules()


    # Add examples from tool metadata
    if any(tool.examples for tool in tools):
        prompt += "**EXAMPLES:**\n\n"
        for tool in tools:
            if tool.examples:
                prompt += f"**{tool.name} Examples:**\n"
                for i, example in enumerate(tool.examples[:3], 1):  # Limit to 3 examples
                    desc = example.get("description", f"Example {i}")
                    args = example.get("arguments", {})
                    prompt += f"{i}. {desc}:\n"
                    # Use Qwen3-specific tool call format
                    tool_call_example = _format_tool_call_example(tool.name, args, ToolFormat.SPECIAL_TOKEN)
                    prompt += f"{tool_call_example}\n\n"

    return prompt


def _format_llama_style(tools: List[ToolDefinition]) -> str:
    """Format tools for LLaMA models using <function_call> format with enhanced metadata."""
    if not tools:
        return ""

    prompt = "You have access to the following functions. Use them when needed:\n\n"

    # Tool descriptions with enhanced metadata
    for tool in tools:
        prompt += f"**{tool.name}**: {tool.description}\n"

        # Add when_to_use guidance if available
        if tool.when_to_use:
            prompt += f"  • **When to use**: {tool.when_to_use}\n"

        # Add tags if available
        if tool.tags:
            prompt += f"  • **Tags**: {', '.join(tool.tags)}\n"

        if tool.parameters:
            prompt += f"  • **Parameters**: {json.dumps(tool.parameters, indent=2)}\n"
        prompt += "\n"

    prompt += """To call a function, use this format:
<function_call>
{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}
</function_call>
""" + _critical_rules()

    # Add examples from tool metadata
    if any(tool.examples for tool in tools):
        prompt += "**EXAMPLES:**\n\n"
        for tool in tools:
            if tool.examples:
                prompt += f"**{tool.name} Examples:**\n"
                for i, example in enumerate(tool.examples[:3], 1):  # Limit to 3 examples
                    desc = example.get("description", f"Example {i}")
                    args = example.get("arguments", {})
                    prompt += f"{i}. {desc}:\n"
                    # Use architecture-specific tool call format
                    tool_call_example = _format_tool_call_example(tool.name, args, ToolFormat.FUNCTION_CALL)
                    prompt += f"{tool_call_example}\n\n"

    return prompt


def _format_xml_style(tools: List[ToolDefinition]) -> str:
    """Format tools for XML-based models."""
    if not tools:
        return ""

    prompt = "You have access to these tools:\n\n"

    for tool in tools:
        prompt += f'<tool name="{tool.name}">\n'
        prompt += f"  <description>{tool.description}</description>\n"
        if tool.parameters:
            prompt += f"  <parameters>{json.dumps(tool.parameters)}</parameters>\n"
        prompt += "</tool>\n\n"

    prompt += """To use a tool, format your call as:
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>
""" + _critical_rules()

    return prompt


def _format_gemma_style(tools: List[ToolDefinition]) -> str:
    """Format tools for Gemma models using code blocks."""
    if not tools:
        return ""

    prompt = "You can use these tools by writing tool_code blocks:\n\n"

    for tool in tools:
        prompt += f"**{tool.name}**: {tool.description}\n"
        if tool.parameters:
            param_list = ", ".join([f"{name}: {info.get('type', 'any')}" for name, info in tool.parameters.items()])
            prompt += f"Usage: {tool.name}({param_list})\n"
        prompt += "\n"

    prompt += """To call a tool, use:
```tool_code
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```"""

    return prompt


def _format_generic_style(tools: List[ToolDefinition]) -> str:
    """Generic tool formatting for unknown architectures with enhanced metadata."""
    if not tools:
        return ""

    prompt = "You have access to the following tools:\n\n"

    for tool in tools:
        prompt += f"- **{tool.name}**: {tool.description}\n"

        # Add when_to_use guidance if available
        if tool.when_to_use:
            prompt += f"  **When to use**: {tool.when_to_use}\n"

        # Add tags if available
        if tool.tags:
            prompt += f"  **Tags**: {', '.join(tool.tags)}\n"

        if tool.parameters:
            prompt += f"  **Parameters**: {json.dumps(tool.parameters, indent=2)}\n"
        prompt += "\n"

    prompt += _critical_rules()

    # Add examples from tool metadata
    if any(tool.examples for tool in tools):
        prompt += "**EXAMPLES:**\n\n"
        for tool in tools:
            if tool.examples:
                prompt += f"**{tool.name} Examples:**\n"
                for i, example in enumerate(tool.examples[:3], 1):  # Limit to 3 examples
                    desc = example.get("description", f"Example {i}")
                    args = example.get("arguments", {})
                    prompt += f"{i}. {desc}:\n"
                    # Use generic format for unknown architectures
                    tool_call_example = _format_tool_call_example(tool.name, args, ToolFormat.RAW_JSON)
                    prompt += f"{tool_call_example}\n\n"

    return prompt


def clean_tool_syntax(content: str, tool_calls: List[ToolCall] = None) -> str:
    """
    Remove all tool call syntax from content using sophisticated pattern matching.

    This function uses the same patterns as the parser to ensure consistency.

    Args:
        content: Text content that may contain tool call syntax
        tool_calls: Optional list of tool calls (if None, always clean)

    Returns:
        Content with tool call syntax removed
    """
    if not content:
        return content

    # Only clean if we have tool calls or if tool_calls is None (force clean)
    if tool_calls is not None and not tool_calls:
        return content

    import re

    # Use the same sophisticated patterns as the _parse_special_token function
    patterns = [
        # Strategy 1: Properly closed <|tool_call|> tags
        r'<\|tool_call\|>\s*.*?\s*</\|tool_call\|>',

        # Strategy 2: Opening tag with JSON, flexible ending
        r'<\|tool_call\|>\s*\{(?:[^{}]|(?:\{[^{}]*\}))*\}\s*(?:</\|tool_call\|>|$|\n|<)',

        # Strategy 3: Ultra-robust - just start tag + JSON
        r'<\|tool_call\|>\s*\{[^<]*?\}',

        # Other formats
        r'<function_call>.*?</function_call>',
        r'<tool_call>.*?</tool_call>',
        r'```tool_code.*?```',
        r'```tool_call.*?```'
    ]

    # Apply all patterns
    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace and return
    return content.strip()


def _format_tool_call_example(tool_name: str, arguments: Dict[str, Any], tool_format: ToolFormat) -> str:
    """
    Format a tool call example using the correct format for the architecture.
    
    Args:
        tool_name: Name of the tool
        arguments: Tool arguments
        tool_format: The tool format for the architecture
        
    Returns:
        Formatted tool call example string
    """
    tool_call_json = json.dumps({"name": tool_name, "arguments": arguments})
    
    if tool_format == ToolFormat.SPECIAL_TOKEN:
        # Qwen3, GLM-4.5+ format
        return f"<|tool_call|>\n{tool_call_json}\n</|tool_call|>"
    elif tool_format == ToolFormat.FUNCTION_CALL:
        # LLaMA3 format
        return f"<function_call>\n{tool_call_json}\n</function_call>"
    elif tool_format == ToolFormat.XML_WRAPPED:
        # XML format
        return f"<tool_call>\n{tool_call_json}\n</tool_call>"
    elif tool_format == ToolFormat.TOOL_CODE:
        # Gemma format
        return f"```tool_code\n{tool_call_json}\n```"
    else:
        # Generic format - just the JSON
        return tool_call_json


def _critical_rules():
    """
    Returns the critical rules for tool usage as a string.

    This function is intended to provide a single source of truth for the
    critical tool usage rules, which can be referenced elsewhere in the codebase
    or for documentation purposes.

    Returns:
        str: The critical rules for tool usage.
    """
    return """
    
CRITICAL RULES FOR TOOL USAGE:
1. If you can answer the question directly, do not call a tool
2. If you can't answer the question directly, call a tool to extend your capabilities, gain further insights or perform an action
3. DO NOT call tools to show off capabilities - when requested, just describe the tools at your disposal
4. The "name" field must be at the TOP LEVEL, NOT inside "arguments"
5. Use the exact JSON structure shown above

"""
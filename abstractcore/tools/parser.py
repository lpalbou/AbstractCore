"""
Architecture-based tool call parsing and formatting.

This module handles the detection and parsing of tool calls from model
responses based on their architecture.
"""

import re
import json
import ast
from typing import List, Optional, Dict, Any
from enum import Enum

from .core import ToolCall, ToolDefinition
from ..architectures import detect_architecture, get_architecture_format
from ..utils.jsonish import loads_dict_like as _jsonish_loads_dict_like
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


def _loads_dict_like(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON-ish or Python-literal dict safely.

    Many OSS models emit tool arguments with single quotes and Python literals
    (True/False/None) even when asked for strict JSON. We accept both to keep
    tool calling robust.
    """
    return _jsonish_loads_dict_like(raw)


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

def _has_bracket_tool_prefix(response: str) -> bool:
    """Check if response contains a `tool: [name]: {...}` style tool call prefix."""
    if not response:
        return False
    return bool(re.search(r'(?im)^\s*tool\s*:\s*\[[^\]]+\]\s*:\s*\{', response))

def _has_harmony_tool_prefix(response: str) -> bool:
    """Check if response contains a Harmony/ChatML-style tool call marker.

    Example emitted by some models:
        <|channel|>commentary to=list_files <|constrain|>json<|message|>{"directory_path": "..."}
    """
    if not response:
        return False
    return "<|channel|>" in response and "<|message|>" in response and "to=" in response


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

    # Some models emit a CLI-like prefix format regardless of architecture.
    if _has_bracket_tool_prefix(response):
        return True
    if _has_harmony_tool_prefix(response):
        return True

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
            _has_bracket_tool_prefix(response),
            _has_harmony_tool_prefix(response),
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
    calls = parser(response)
    # Fallback: some models emit tool syntax that doesn't match their expected architecture format
    # (e.g., `tool: [name]: {...}` or partial tags). Try the generic parser when needed.
    if not calls and parser is not _parse_any_format:
        calls = _parse_any_format(response)
    if calls:
        from .arg_canonicalizer import canonicalize_tool_arguments

        for call in calls:
            call.arguments = canonicalize_tool_arguments(call.name, call.arguments)
    return calls


def format_tool_prompt(
    tools: List[ToolDefinition],
    model_name: Optional[str] = None,
    *,
    include_tool_list: bool = True,
    include_examples: bool = True,
) -> str:
    """
    Format tools into a system prompt based on model architecture.

    Args:
        tools: List of tool definitions
        model_name: Optional model name for architecture detection
        include_tool_list: If False, omit per-tool listings (only include tool-call protocol/rules)
        include_examples: If False, omit examples even if tools provide them

    Returns:
        Formatted system prompt
    """
    if not tools:
        return "You are a helpful AI assistant."

    # Get tool format
    tool_format = _get_tool_format(model_name)

    # Format based on architecture
    if tool_format == ToolFormat.TOOL_CODE:
        return _format_gemma_style(tools, include_tool_list=include_tool_list, include_examples=include_examples)
    elif tool_format == ToolFormat.SPECIAL_TOKEN:
        return _format_qwen_style(tools, include_tool_list=include_tool_list, include_examples=include_examples)
    elif tool_format == ToolFormat.FUNCTION_CALL:
        return _format_llama_style(tools, include_tool_list=include_tool_list, include_examples=include_examples)
    elif tool_format == ToolFormat.XML_WRAPPED:
        return _format_xml_style(tools, include_tool_list=include_tool_list, include_examples=include_examples)
    elif tool_format == ToolFormat.RAW_JSON:
        return _format_json_style(tools, include_tool_list=include_tool_list, include_examples=include_examples)
    else:
        return _format_generic_style(tools, include_tool_list=include_tool_list, include_examples=include_examples)


# Internal helpers

def _sanitize_tool_call_tags(response: str) -> str:
    """
    Sanitize malformed tool call tags before parsing.

    Handles common LLM output malformations:
    - Doubled opening tags: <|tool_call|><|tool_call|> → <|tool_call|>
    - Doubled closing tags: </|tool_call|></|tool_call|> → </|tool_call|>
    - Malformed closing with }: </|tool_call|} → </|tool_call|>

    Args:
        response: Raw model response text

    Returns:
        Sanitized response with normalized tool call syntax
    """
    if not response:
        return response

    original = response

    # Fix doubled/multiple opening tags (collapse to single)
    # Handles: <|tool_call|><|tool_call|> or <|tool_call|>\n<|tool_call|>
    response = re.sub(
        r'(<\|tool_call\|>\s*)+',
        r'<|tool_call|>',
        response,
        flags=re.IGNORECASE
    )

    # Fix malformed closing tags with } instead of |>
    # Handles: </|tool_call|} → </|tool_call|>
    response = re.sub(
        r'</\|tool_call\|\}',
        r'</|tool_call|>',
        response,
        flags=re.IGNORECASE
    )

    # Fix doubled/multiple closing tags (collapse to single)
    response = re.sub(
        r'(</\|tool_call\|>\s*)+',
        r'</|tool_call|>',
        response,
        flags=re.IGNORECASE
    )

    if response != original:
        logger.debug(f"Sanitized malformed tool call tags")

    return response


def _get_tool_format(model_name: Optional[str]) -> ToolFormat:
    """Get tool format for a model."""
    if not model_name:
        # When no model specified, use NATIVE which triggers _parse_any_format
        # This ensures all formats are tried including <|tool_call|> special tokens
        return ToolFormat.NATIVE

    architecture = detect_architecture(model_name)
    arch_format = get_architecture_format(architecture)

    tool_format = str(arch_format.get("tool_format", "json") or "").strip().lower()
    message_format = str(arch_format.get("message_format", "") or "").strip().lower()

    # tool_format values are defined in `abstractcore/assets/architecture_formats.json`.
    # We interpret them as the model's *preferred tool-call syntax* and fall back to
    # `_parse_any_format` when the model emits a different convention.
    if tool_format == "special_token":
        return ToolFormat.SPECIAL_TOKEN
    if tool_format == "xml":
        return ToolFormat.XML_WRAPPED
    if tool_format == "pythonic":
        return ToolFormat.TOOL_CODE
    if tool_format == "json":
        return ToolFormat.RAW_JSON
    if tool_format in {"openai_functions", "native", "none"}:
        # Native/OpenAI-functions tool calls are expected in structured response fields, not text.
        # If tool syntax leaks into content, we parse with the generic fallback.
        return ToolFormat.NATIVE

    if tool_format == "prompted":
        # "prompted" indicates the model relies on prompt-injected tool syntax.
        # Choose the most likely format based on the architecture's message format.
        # - Qwen/ChatML-like formats generally use <|tool_call|> special tokens.
        if message_format == "im_start_end":
            return ToolFormat.SPECIAL_TOKEN
        # - LLaMA-style prompted tools commonly use <function_call>...</function_call>.
        return ToolFormat.FUNCTION_CALL

    # Conservative fallback: function-call wrapper (and then _parse_any_format fallback).
    return ToolFormat.FUNCTION_CALL




def _parse_special_token(response: str) -> List[ToolCall]:
    """Parse Qwen-style <|tool_call|> format with robust fallback."""
    tool_calls = []

    # SANITIZE FIRST: Fix malformed tags (doubled tags, broken closing tags)
    response = _sanitize_tool_call_tags(response)

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
                # Fallback: Escape newlines/tabs only inside JSON string values
                # This prevents escaping structural newlines which would break parsing
                # Algorithm: Track when inside/outside strings, only escape within strings
                in_string = False
                escaped = False
                fixed = []

                for char in json_str:
                    if escaped:
                        # Previous char was backslash, this is part of escape sequence
                        fixed.append(char)
                        escaped = False
                    elif char == '\\':
                        # Start of escape sequence
                        fixed.append(char)
                        escaped = True
                    elif char == '"':
                        # Toggle string context
                        in_string = not in_string
                        fixed.append(char)
                    elif in_string and char == '\n':
                        # Newline inside string - escape it
                        fixed.append('\\n')
                    elif in_string and char == '\r':
                        # CR inside string - escape it
                        fixed.append('\\r')
                    elif in_string and char == '\t':
                        # Tab inside string - escape it
                        fixed.append('\\t')
                    else:
                        # Normal character or structural whitespace
                        fixed.append(char)

                fixed_json = ''.join(fixed)
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
            tool_data = _loads_dict_like(json_str)
            if not isinstance(tool_data, dict):
                continue

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

    # Pattern for XML format.
    #
    # Supported inner payloads:
    # 1) JSON-ish dict (our canonical prompted-tool wrapper):
    #    <tool_call>{"name":"read_file","arguments":{...}}</tool_call>
    # 2) Nemotron XML-ish wrapper (observed in the wild):
    #    <tool_call>
    #      <function=write_file>
    #        <parameter=file_path>...</parameter>
    #        <parameter=content>...</parameter>
    #      </function>
    #    </tool_call>
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'

    for match in re.finditer(pattern, response, re.DOTALL | re.IGNORECASE):
        body = match.group(1)
        if not isinstance(body, str):
            continue

        body_stripped = body.strip()

        # Case 1: JSON-ish dict inside <tool_call>...</tool_call>
        if body_stripped.startswith("{") and body_stripped.endswith("}"):
            try:
                tool_data = _loads_dict_like(body_stripped)
                if not isinstance(tool_data, dict):
                    continue

                tool_calls.append(ToolCall(
                    name=tool_data.get("name", ""),
                    arguments=tool_data.get("arguments", {}),
                    call_id=tool_data.get("id")
                ))
                continue
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse XML tool call JSON: {body_stripped} - {e}")
                continue

        # Case 2: Nemotron XML-ish function/parameter encoding
        func_match = re.search(r'<function\s*=\s*([a-zA-Z0-9_-]+)\s*>', body, re.IGNORECASE)
        if not func_match:
            continue
        func_name = func_match.group(1).strip()
        if not func_name:
            continue

        arguments: Dict[str, Any] = {}
        for param_match in re.finditer(
            r'<parameter\s*=\s*([a-zA-Z0-9_-]+)\s*>(.*?)</parameter>',
            body,
            re.DOTALL | re.IGNORECASE,
        ):
            key = (param_match.group(1) or "").strip()
            raw_value = param_match.group(2) or ""
            if not key:
                continue

            # Preserve content as-is, but strip the common leading/trailing newline artifacts
            # introduced by pretty-printed tag blocks.
            value = raw_value.replace("\r\n", "\n")
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
            arguments[key] = value

        tool_calls.append(ToolCall(name=func_name, arguments=arguments, call_id=None))

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
            tool_data = _loads_dict_like(code_content)
            if not isinstance(tool_data, dict):
                raise json.JSONDecodeError("not a dict", code_content, 0)
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

                # Simple, safe argument parsing for common keyword args.
                arguments = {}
                if args_str.strip():
                    arg_pattern = r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^,\)]+)'
                    for arg_match in re.finditer(arg_pattern, args_str):
                        key = arg_match.group(1)
                        raw_value = arg_match.group(2).strip()
                        value: Any = raw_value
                        if (raw_value.startswith('"') and raw_value.endswith('"')) or (
                            raw_value.startswith("'") and raw_value.endswith("'")
                        ):
                            value = raw_value[1:-1]
                        elif raw_value.lower() in ("true", "false"):
                            value = raw_value.lower() == "true"
                        elif raw_value.lower() in ("none", "null"):
                            value = None
                        else:
                            try:
                                value = int(raw_value)
                            except Exception:
                                try:
                                    value = float(raw_value)
                                except Exception:
                                    value = raw_value
                        arguments[str(key)] = value

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
            tool_data = _loads_dict_like(json_str)
            if not isinstance(tool_data, dict):
                continue

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
            tool_data = _loads_dict_like(json_str)
            if not isinstance(tool_data, dict):
                continue

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


def _parse_bracket_tool_prefix(response: str) -> List[ToolCall]:
    """Parse `tool: [name]: { ... }` format (arguments-only JSON)."""
    tool_calls: List[ToolCall] = []
    if not response:
        return tool_calls

    def _find_matching_brace(text: str, start: int) -> int:
        """Return index of the matching '}' for a '{' at `start`, or -1."""
        depth = 0
        in_string = False
        quote = ""
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == quote:
                    in_string = False
                    quote = ""
                continue

            if ch in ("'", '"'):
                in_string = True
                quote = ch
                continue

            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return i

        return -1

    # Common in some OSS model tool conventions.
    # Example (single-line):
    #   tool: [list_files]: {"directory_path":"rtype","recursive":true}
    # Example (multi-line):
    #   tool: [list_files]: {
    #     "directory_path": "rtype",
    #     "recursive": true
    #   }
    header_re = re.compile(r"(?im)^\s*tool\s*:\s*\[([a-zA-Z0-9_\-]+)\]\s*:\s*")
    for match in header_re.finditer(response):
        name = str(match.group(1) or "").strip()
        if not name:
            continue

        # Find the first opening brace after the header (allow whitespace/newlines).
        brace_start = response.find("{", match.end())
        if brace_start == -1:
            continue

        # Only allow whitespace between header end and '{' (avoid grabbing unrelated JSON).
        between = response[match.end() : brace_start]
        if between and any(not c.isspace() for c in between):
            continue

        brace_end = _find_matching_brace(response, brace_start)
        if brace_end == -1:
            continue

        raw_args = response[brace_start : brace_end + 1]
        args = _loads_dict_like(raw_args)
        if not isinstance(args, dict):
            continue

        tool_calls.append(ToolCall(name=name, arguments=args))

    return tool_calls


def _parse_harmony_tool_prefix(response: str) -> List[ToolCall]:
    """Parse Harmony/ChatML-style tool calls embedded in content.

    Example:
        <|channel|>commentary to=list_files <|constrain|>json<|message|>{"directory_path":"./x","recursive":true}
    """
    tool_calls: List[ToolCall] = []
    if not response:
        return tool_calls

    if "<|channel|>" not in response or "<|message|>" not in response or "to=" not in response:
        return tool_calls

    def _find_matching_brace(text: str, start: int) -> int:
        """Return index of the matching '}' for a '{' at `start`, or -1."""
        depth = 0
        in_string = False
        quote = ""
        escaped = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == quote:
                    in_string = False
                    quote = ""
                continue

            if ch in ("'", '"'):
                in_string = True
                quote = ch
                continue

            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return i

        return -1

    # Match "<|channel|>... to=TOOL_NAME" and then find the following <|message|>{...}.
    header_re = re.compile(
        r"(?i)<\|channel\|>\s*[a-zA-Z0-9_\-]+\s+to=([a-zA-Z0-9_\-\.]+)\b"
    )
    for match in header_re.finditer(response):
        raw_name = str(match.group(1) or "").strip()
        if not raw_name:
            continue

        # Normalize common prefixes used by some tool-call transcripts.
        name = raw_name
        if name.startswith("functions."):
            name = name.split(".", 1)[1].strip()
        if not name:
            continue

        # Find the next "<|message|>" after the header.
        msg_tag = "<|message|>"
        msg_start = response.find(msg_tag, match.end())
        if msg_start == -1:
            continue

        brace_start = response.find("{", msg_start + len(msg_tag))
        if brace_start == -1:
            continue

        # Only allow whitespace between the message tag and '{'.
        between = response[msg_start + len(msg_tag) : brace_start]
        if between and any(not c.isspace() for c in between):
            continue

        brace_end = _find_matching_brace(response, brace_start)
        if brace_end == -1:
            # Some models occasionally omit the final closing brace(s) when emitting a
            # Harmony tool transcript. Try a best-effort recovery by balancing braces
            # to the end of the message and parsing the result.
            raw_args = response[brace_start:].strip()

            def _balance_braces(text: str) -> str:
                depth = 0
                in_string = False
                quote = ""
                escaped = False
                for ch in text:
                    if in_string:
                        if escaped:
                            escaped = False
                            continue
                        if ch == "\\":
                            escaped = True
                            continue
                        if ch == quote:
                            in_string = False
                            quote = ""
                        continue
                    if ch in ("'", '"'):
                        in_string = True
                        quote = ch
                        continue
                    if ch == "{":
                        depth += 1
                        continue
                    if ch == "}":
                        depth -= 1
                        continue
                if depth > 0:
                    return text + ("}" * depth)
                return text

            raw_args = _balance_braces(raw_args)
        else:
            raw_args = response[brace_start : brace_end + 1]
        payload = _loads_dict_like(raw_args)
        if not isinstance(payload, dict):
            continue

        # Some models (notably OpenAI's gpt-oss via LM Studio) emit a wrapper payload:
        #   {"name":"tool_name","arguments":{...},"call_id": "..."}
        # In that case, unwrap `arguments` so runtime tool execution receives only
        # the tool kwargs (and not unexpected keys like "name").
        call_id = None
        args: Any = payload
        if "arguments" in payload:
            inner_args = payload.get("arguments")
            if isinstance(inner_args, dict):
                args = inner_args
            elif isinstance(inner_args, str):
                parsed = _loads_dict_like(inner_args)
                if isinstance(parsed, dict):
                    args = parsed

        call_id_value = payload.get("call_id") or payload.get("id")
        if isinstance(call_id_value, str) and call_id_value.strip():
            call_id = call_id_value.strip()

        if not isinstance(args, dict):
            continue

        tool_calls.append(ToolCall(name=name, arguments=args, call_id=call_id))

    return tool_calls


def _parse_any_format(response: str) -> List[ToolCall]:
    """Try all parsing formats with comprehensive fallbacks."""
    # SANITIZE FIRST: Fix malformed tags before trying any parser
    response = _sanitize_tool_call_tags(response)

    tool_calls = []

    # Try each parser and accumulate results
    parsers = [
        _parse_special_token,
        _parse_function_call,
        _parse_xml_wrapped,
        _parse_tool_code,
        _parse_harmony_tool_prefix,
        _parse_bracket_tool_prefix,
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
        try:
            args_key = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        except Exception:
            args_key = str(call.arguments)
        call_key = (call.name, args_key)
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

def _format_parameters_compact(parameters: Dict[str, Any]) -> str:
    """Render a compact, human/LLM-friendly parameter summary.

    We intentionally avoid dumping full JSON schema here to keep the tool prompt small.
    """
    if not isinstance(parameters, dict) or not parameters:
        return "(none)"

    def _fmt_default(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    parts: List[str] = []
    for name in sorted([k for k in parameters.keys() if isinstance(k, str)]):
        meta = parameters.get(name)
        ptype = "any"
        required = True
        default_repr: Optional[str] = None

        if isinstance(meta, dict):
            if isinstance(meta.get("type"), str) and meta.get("type"):
                ptype = str(meta.get("type"))
            required = "default" not in meta
            if not required:
                default_value = meta.get("default")
                # Avoid printing `default null` / `default None` in prompts; treat that as optional.
                if default_value is not None:
                    default_repr = _fmt_default(default_value)
        else:
            required = True

        if required:
            parts.append(f"{name}: {ptype} (required)")
        elif default_repr is not None:
            parts.append(f"{name}: {ptype} (default {default_repr})")
        else:
            parts.append(f"{name}: {ptype} (optional)")

    return ", ".join(parts) if parts else "(none)"


def _append_tool_examples(
    prompt: str,
    tools: List[ToolDefinition],
    *,
    tool_format: ToolFormat,
    max_examples_total: int = 6,
) -> str:
    """Append a small, globally-capped examples section.

    Notes:
    - Examples are useful, but they are extremely token-expensive when included per-tool.
    - We cap examples globally and prioritize the "core editing loop" tools first.
    """
    if max_examples_total <= 0:
        return prompt

    tools_with_examples = [t for t in tools if getattr(t, "examples", None)]
    if not tools_with_examples:
        return prompt

    by_name = {t.name: t for t in tools_with_examples if isinstance(t.name, str) and t.name}
    preferred_order = [
        "list_files",
        "search_files",
        "read_file",
        "edit_file",
        "write_file",
        "execute_command",
        "fetch_url",
        "web_search",
    ]

    ordered_names = []
    seen: set[str] = set()
    for name in preferred_order:
        if name in by_name and name not in seen:
            ordered_names.append(name)
            seen.add(name)
    for name in sorted(by_name.keys()):
        if name not in seen:
            ordered_names.append(name)

    out = prompt + "**EXAMPLES:**\n\n"
    added = 0
    for name in ordered_names:
        tool = by_name.get(name)
        if tool is None:
            continue
        examples = getattr(tool, "examples", None)
        if not isinstance(examples, list) or not examples:
            continue
        example = examples[0] if isinstance(examples[0], dict) else {}
        desc = str(example.get("description") or "Example").strip()
        args = example.get("arguments")
        args_dict = dict(args) if isinstance(args, dict) else {}

        out += f"- {tool.name}: {desc}\n"
        out += _format_tool_call_example(tool.name, args_dict, tool_format) + "\n\n"
        added += 1
        if added >= max_examples_total:
            break

    return out


def _format_qwen_style(tools: List[ToolDefinition], *, include_tool_list: bool = True, include_examples: bool = True) -> str:
    """Format tools for Qwen models using <|tool_call|> format with enhanced metadata."""
    if not tools:
        return ""

    prompt = "You are a helpful AI assistant with access to the following tools:\n\n"

    if include_tool_list:
        for tool in tools:
            prompt += f"**{tool.name}**: {tool.description}\n"
            if tool.parameters:
                prompt += f"  • **Args**: {_format_parameters_compact(tool.parameters)}\n"
            prompt += "\n"

    prompt += """To use a tool, respond with one or more tool-call blocks (no other text):
<|tool_call|>
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
</|tool_call|>

To call multiple tools, repeat the block once per call.
""" + _critical_rules()


    if include_examples:
        prompt = _append_tool_examples(prompt, tools, tool_format=ToolFormat.SPECIAL_TOKEN)

    return prompt


def _format_llama_style(tools: List[ToolDefinition], *, include_tool_list: bool = True, include_examples: bool = True) -> str:
    """Format tools for LLaMA models using <function_call> format with enhanced metadata."""
    if not tools:
        return ""

    prompt = "You have access to the following functions. Use them when needed:\n\n"

    if include_tool_list:
        for tool in tools:
            prompt += f"**{tool.name}**: {tool.description}\n"
            if tool.parameters:
                prompt += f"  • **Args**: {_format_parameters_compact(tool.parameters)}\n"
            prompt += "\n"

    prompt += """To call a function, output one or more <function_call> blocks (no other text):
<function_call>
{"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}
</function_call>

To call multiple functions, repeat the block once per call.
""" + _critical_rules()

    if include_examples:
        prompt = _append_tool_examples(prompt, tools, tool_format=ToolFormat.FUNCTION_CALL)

    return prompt


def _format_xml_style(tools: List[ToolDefinition], *, include_tool_list: bool = True, include_examples: bool = True) -> str:
    """Format tools for XML-based models."""
    if not tools:
        return ""

    prompt = "You have access to these tools:\n\n"

    if include_tool_list:
        for tool in tools:
            prompt += f'<tool name="{tool.name}">\n'
            prompt += f"  <description>{tool.description}</description>\n"
            if tool.parameters:
                prompt += f"  <args>{_format_parameters_compact(tool.parameters)}</args>\n"
            prompt += "</tool>\n\n"

    prompt += """To use a tool, output one or more <tool_call> blocks (no other text):
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

To call multiple tools, repeat the block once per call.
""" + _critical_rules()

    if include_examples:
        prompt = _append_tool_examples(prompt, tools, tool_format=ToolFormat.XML_WRAPPED)

    return prompt


def _format_json_style(tools: List[ToolDefinition], *, include_tool_list: bool = True, include_examples: bool = True) -> str:
    """Format tools for models that prefer raw JSON tool calls in content."""
    if not tools:
        return ""

    prompt = "You have access to the following tools:\n\n"

    if include_tool_list:
        for tool in tools:
            prompt += f"- {tool.name}: {tool.description}\n"
            if tool.parameters:
                prompt += f"  args: {_format_parameters_compact(tool.parameters)}\n"

    prompt += """To use a tool, respond with one or more JSON objects (no extra text):
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}

To call multiple tools, output multiple JSON objects (one per line/block).
""" + _critical_rules()

    if include_examples:
        prompt = _append_tool_examples(prompt, tools, tool_format=ToolFormat.RAW_JSON)

    return prompt


def _format_gemma_style(tools: List[ToolDefinition], *, include_tool_list: bool = True, include_examples: bool = True) -> str:
    """Format tools for Gemma models using code blocks."""
    if not tools:
        return ""

    prompt = "You can use these tools by writing tool_code blocks:\n\n"

    if include_tool_list:
        for tool in tools:
            prompt += f"**{tool.name}**: {tool.description}\n"
            if tool.parameters:
                prompt += f"Args: {_format_parameters_compact(tool.parameters)}\n"
            prompt += "\n"

    prompt += """To call a tool, output one or more tool_code blocks (no other text):
```tool_code
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

To call multiple tools, repeat the block once per call."""

    if include_examples:
        prompt = _append_tool_examples(prompt, tools, tool_format=ToolFormat.TOOL_CODE)

    return prompt


def _format_generic_style(tools: List[ToolDefinition], *, include_tool_list: bool = True, include_examples: bool = True) -> str:
    """Generic tool formatting for unknown architectures with enhanced metadata."""
    if not tools:
        return ""

    prompt = "You have access to the following tools:\n\n"

    if include_tool_list:
        for tool in tools:
            prompt += f"- {tool.name}: {tool.description}\n"
            if tool.parameters:
                prompt += f"  args: {_format_parameters_compact(tool.parameters)}\n"

    prompt += _critical_rules()

    if include_examples:
        prompt = _append_tool_examples(prompt, tools, tool_format=ToolFormat.RAW_JSON)

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

    # Strip Harmony/ChatML tool-call segments first (balanced JSON after <|message|>).
    # Regex alone is brittle here because tool arguments can contain nested braces.
    if "<|channel|>" in content and "<|message|>" in content and "to=" in content:
        def _find_matching_brace(text: str, start: int) -> int:
            depth = 0
            in_string = False
            quote = ""
            escaped = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escaped:
                        escaped = False
                        continue
                    if ch == "\\":
                        escaped = True
                        continue
                    if ch == quote:
                        in_string = False
                        quote = ""
                    continue
                if ch in ("'", '"'):
                    in_string = True
                    quote = ch
                    continue
                if ch == "{":
                    depth += 1
                    continue
                if ch == "}":
                    depth -= 1
                    if depth == 0:
                        return i
            return -1

        def _consume_trailing_kv_fragment(text: str, start_idx: int) -> int:
            """Consume malformed trailing JSON key/value fragments after a closed object.

            Some models (notably some OSS models emitting Harmony tool transcripts) occasionally
            close the JSON object early and then continue emitting extra fields outside of it,
            e.g.:
              <|message|>{"name":"write_file","arguments":{...},"call_id":null},"mode":"w"}

            Tool parsing can still succeed (the prefix is valid), but the tail fragment must
            not leak into cleaned assistant content (it otherwise shows up as "Thought" in UIs).
            """
            i = start_idx
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text) or text[i] != ",":
                return start_idx

            # Quick heuristic: only treat as a JSON-ish continuation if we see `,"key":...`.
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] not in ("'", '"'):
                return start_idx

            in_string = False
            quote = ""
            escaped = False
            brace_depth = 0
            saw_colon = False
            pos = i
            while pos < len(text):
                # Do not swallow the next Harmony segment (if any).
                if not in_string and text.startswith("<|channel|>", pos):
                    return pos

                ch = text[pos]
                if in_string:
                    if escaped:
                        escaped = False
                        pos += 1
                        continue
                    if ch == "\\":
                        escaped = True
                        pos += 1
                        continue
                    if ch == quote:
                        in_string = False
                        quote = ""
                        pos += 1
                        continue
                    pos += 1
                    continue

                if ch in ("'", '"'):
                    in_string = True
                    quote = ch
                    pos += 1
                    continue

                if ch == ":":
                    saw_colon = True
                elif ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    if saw_colon and brace_depth == 0:
                        return pos + 1
                    if brace_depth > 0:
                        brace_depth -= 1
                pos += 1

            return len(text) if saw_colon else start_idx

        msg_tag = "<|message|>"
        out_parts = []
        i = 0
        while i < len(content):
            start = content.find("<|channel|>", i)
            if start == -1:
                out_parts.append(content[i:])
                break
            out_parts.append(content[i:start])

            msg_start = content.find(msg_tag, start)
            if msg_start == -1:
                out_parts.append(content[start:])
                break
            # Only treat as a tool call when there's a `to=` directive before the message tag.
            if "to=" not in content[start:msg_start]:
                out_parts.append(content[start:msg_start])
                i = msg_start
                continue

            brace_start = content.find("{", msg_start + len(msg_tag))
            if brace_start == -1:
                out_parts.append(content[start:msg_start])
                i = msg_start
                continue
            between = content[msg_start + len(msg_tag) : brace_start]
            if between and any(not c.isspace() for c in between):
                out_parts.append(content[start:brace_start])
                i = brace_start
                continue

            brace_end = _find_matching_brace(content, brace_start)
            if brace_end == -1:
                # Best-effort: drop the remainder of this segment up to the next Harmony marker
                # (or to end-of-content). Leaving partial tool payloads in `content` is more
                # harmful (it breaks agent scratchpads and UI "Thought" rendering).
                next_start = content.find("<|channel|>", brace_start + 1)
                if next_start == -1:
                    break
                i = next_start
                continue

            i = _consume_trailing_kv_fragment(content, brace_end + 1)

        content = "".join(out_parts)

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
        ,
        # CLI-like prefix format: tool: [name]: {...}
        r'(?im)^\s*tool\s*:\s*\[[^\]]+\]\s*:\s*\{.*\}\s*$',
        # Harmony/ChatML tool-call transcript format:
        #   <|channel|>commentary to=tool <|constrain|>json<|message|>{...}
        r'(?is)<\|channel\|>\s*[a-zA-Z0-9_\-]+\s+to=[a-zA-Z0-9_\-\.]+\b.*?<\|message\|>\s*\{.*?\}',
        # Orphan tags (some models emit a closing tag on its own line)
        r'(?im)^\s*<\|tool_call\|>\s*$',
        r'(?im)^\s*</\|tool_call\|>\s*$',
        r'(?im)^\s*<tool_call>\s*$',
        r'(?im)^\s*</tool_call>\s*$',
        r'(?im)^\s*<\|channel\|>\s*$',
        r'(?im)^\s*<\|message\|>\s*$',
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
    tool_call_json = json.dumps({"name": tool_name, "arguments": arguments}, separators=(",", ":"), ensure_ascii=False)
    
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
    return (
        "CRITICAL RULES FOR TOOL USAGE:\n"
        "1. If you can answer directly, do not call a tool.\n"
        "2. If you need info or an action, call the smallest relevant tool.\n"
        "3. Do not call tools to show off; if asked, describe capabilities.\n"
        "4. The \"name\" field must be top-level (not inside \"arguments\").\n"
        "5. Use the exact tool-call JSON structure.\n"
        "6. Never fabricate tool results; outputs are returned separately.\n"
        "7. Do not write your own `tool:` result lines.\n"
        "8. You MAY batch multiple tool calls by repeating the tool-call block once per call (prefer independent calls).\n"
    )

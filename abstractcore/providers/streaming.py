"""
Unified streaming processor with incremental tool detection.

This module provides a single streaming strategy that handles tools elegantly
while maintaining real-time streaming performance, with proper tag rewriting support.
"""

import json
import re
import uuid
from typing import List, Dict, Any, Optional, Iterator, Tuple
from enum import Enum

from ..core.types import GenerateResponse
from ..tools.core import ToolCall
from ..utils.jsonish import loads_dict_like
from ..utils.structured_logging import get_logger
from ..utils.truncation import preview_text

logger = get_logger(__name__)


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
            "qwen": {
                "start": r"<\|tool_call\|>",
                "end": r"</\|tool_call\|>",
            },
            # Gemma4 special-token tool blocks:
            #   <|tool_call>call:tool_name{...}<tool_call|>
            "gemma4": {
                "start": r"<\|tool_call>",
                "end": r"<tool_call\|>",
            },
            # Liquid LFM2.5 special-token tool blocks:
            #   <|tool_call_start|>[tool_name(arg="value")]<|tool_call_end|>
            "liquid": {
                "start": r"<\|tool_call_start\|>",
                "end": r"<\|tool_call_end\|>",
            },
            # Harmony/ChatML-style tool transcript (no explicit closing tag; ends at end of JSON after <|message|>).
            "harmony": {
                "start": r"<\|channel\|>",
                "end": None,
                "kind": "harmony",
            },
            "llama": {
                "start": r"<function_call>",
                "end": r"</function_call>",
            },
            "xml": {
                "start": r"<tool_call>",
                "end": r"</tool_call>",
            },
            "gemma": {
                "start": r"```tool_code",
                "end": r"```",
            },
        }
        # A ```json fenced block whose body is a tool-call object
        # (`{"name": ..., "arguments": ...}`, `{"tool_calls": [...]}` or a list
        # of call objects). Only scanned when the request carries tools
        # (`json_fence_tools`): a ```json block is also how a model answers a
        # "reply in JSON" request, so without tools it is always content. A
        # fenced block that turns out NOT to be a tool call is released as
        # content once its closing fence arrives.
        self.json_fence_pattern = {
            "start": r"```json[ \t]*",
            "end": r"```",
            "kind": "json_fence",
        }
        # Set by `UnifiedStreamProcessor.process_stream` from the request's tools.
        self.json_fence_tools = False

        self.active_patterns = self._get_patterns_for_model(model_name)

    def reset(self):
        """Reset detector state."""
        self.state = ToolDetectionState.SCANNING
        self.accumulated_content = ""
        self.current_tool_content = ""
        self.tool_start_pos = None
        self.current_pattern = None
        self.completed_tools = []

    def _scan_patterns(self) -> List[Dict]:
        """Envelopes scanned for: the model's formats, then ```json when tools are in play."""
        if self.json_fence_tools:
            return list(self.active_patterns) + [self.json_fence_pattern]
        return list(self.active_patterns)

    def _get_patterns_for_model(self, model_name: str) -> List[Dict]:
        """Get relevant patterns for a model."""
        if not model_name:
            return list(self.patterns.values())

        # Centralized model capability/syntax lookup.
        # Use `architecture_formats.json` as the source of truth instead of string heuristics.
        try:
            from ..architectures import detect_architecture, get_architecture_format

            architecture = detect_architecture(model_name)
            arch_format = get_architecture_format(architecture)
            tool_format = str(arch_format.get("tool_format", "") or "").strip().lower()
            message_format = str(arch_format.get("message_format", "") or "").strip().lower()
        except Exception:
            tool_format = ""
            message_format = ""
            architecture = "generic"

        # Harmony/ChatML tool transcripts (GPT-OSS).
        if message_format == "harmony":
            return [self.patterns["harmony"]]

        # Pythonic tool blocks (Gemma-style).
        if tool_format == "pythonic":
            return [self.patterns["gemma"]]

        # Special-token tools (Qwen-style). Some "prompted" models share this convention.
        if tool_format == "special_token" or (
            tool_format == "prompted" and message_format == "im_start_end"
        ):
            return [
                self.patterns["qwen"],
                self.patterns["gemma4"],
                self.patterns["liquid"],
                self.patterns["llama"],
                self.patterns["xml"],
            ]

        # XML-wrapped tools.
        if tool_format in {"xml", "glm_xml"}:
            return [self.patterns["xml"], self.patterns["llama"], self.patterns["qwen"]]

        # LLaMA-style prompted tools.
        if tool_format == "prompted" and "llama" in str(architecture):
            return [self.patterns["llama"], self.patterns["xml"]]

        # Default: try the common tag-based formats as fallbacks.
        return [self.patterns["qwen"], self.patterns["llama"], self.patterns["xml"]]

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
        for pattern_info in self._scan_patterns():
            start_pattern = pattern_info["start"]
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
                    streamable_content = self.accumulated_content[: self.tool_start_pos]

                # Start collecting tool content
                self.current_tool_content = self.accumulated_content[match.end() :]

                logger.debug(
                    f"Tool call start detected: {start_pattern} for model {self.model_name}"
                )
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

        # Harmony/ChatML tool transcript: detect completion by balanced JSON after <|message|>.
        if self.current_pattern and self.current_pattern.get("kind") == "harmony":
            return self._collect_harmony_tool_content()
        if self.current_pattern and self.current_pattern.get("kind") == "json_fence":
            return self._collect_json_fence_content()

        # Check for tool end pattern
        end_pattern = self.current_pattern["end"]
        end_match = re.search(end_pattern, self.current_tool_content, re.IGNORECASE)

        if end_match:
            # Tool call is complete
            tool_json_content = self.current_tool_content[: end_match.start()].strip()

            # Try to parse the tool call
            tool_call = self._parse_tool_json(tool_json_content)
            if tool_call:
                completed_tools.append(tool_call)
                logger.debug(f"Complete tool call parsed: {tool_call.name}")

            if self.rewrite_tags:
                # When rewriting, stream the complete accumulated content including tool call
                streamable_content = self.accumulated_content
                logger.debug(
                    f"Tool complete, streaming accumulated content for rewriting: {streamable_content[:200]}"
                )
                self.accumulated_content = ""
            else:
                # Normal mode - don't stream the tool call itself
                pass

            # Reset for next tool
            remaining_content = self.current_tool_content[end_match.end() :]
            self.reset()

            # Continue processing remaining content
            if remaining_content:
                self.accumulated_content = remaining_content
                additional_streamable, additional_tools = self._scan_for_tool_start("")
                streamable_content += additional_streamable
                completed_tools.extend(additional_tools)

        return streamable_content, completed_tools

    def _collect_json_fence_content(self) -> Tuple[str, List[ToolCall]]:
        """Collect a ```json block; a tool-call body becomes tool calls, anything else content."""
        streamable_content = ""
        completed_tools: List[ToolCall] = []

        end_match = re.search(r"```", self.current_tool_content)
        if not end_match:
            return streamable_content, completed_tools

        body = self.current_tool_content[: end_match.start()]
        calls = self._parse_json_fence_tool_calls(body)
        # accumulated_content == <text up to the body> + current_tool_content
        body_start = len(self.accumulated_content) - len(self.current_tool_content)
        block_end = body_start + end_match.end()
        start = self.tool_start_pos or 0

        if calls:
            completed_tools.extend(calls)
            if self.rewrite_tags:
                # Rewriting callers see the markup, as for every other envelope.
                streamable_content = self.accumulated_content[:block_end]
        else:
            # Not a tool call: the block was only withheld, release it verbatim.
            if self.rewrite_tags:
                streamable_content = self.accumulated_content[:block_end]
            else:
                streamable_content = self.accumulated_content[start:block_end]

        remaining_content = self.current_tool_content[end_match.end() :]
        self.reset()
        if remaining_content:
            self.accumulated_content = remaining_content
            additional_streamable, additional_tools = self._scan_for_tool_start("")
            streamable_content += additional_streamable
            completed_tools.extend(additional_tools)
        return streamable_content, completed_tools

    def _parse_json_fence_tool_calls(self, body: str) -> List[ToolCall]:
        """Tool calls in a ```json body, or [] when the body is not (only) tool calls.

        Accepted: one call object, `{"tool_calls": [calls]}`, or `[calls]`, where a
        call is `{"name": str, "arguments"|"parameters": ...}` or the OpenAI
        `{"type": "function", "function": {"name", "arguments"}}` shape. Every
        item must be a call; a JSON answer that merely has a "name" key is not.
        """
        text = str(body or "").strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except Exception:
            data = loads_dict_like(text) if text.startswith("{") else None
        if isinstance(data, dict) and isinstance(data.get("tool_calls"), list) and len(data) == 1:
            items = data["tool_calls"]
        elif isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
        else:
            return []
        if not items:
            return []

        calls: List[ToolCall] = []
        for item in items:
            if not isinstance(item, dict):
                return []
            function = item.get("function") if isinstance(item.get("function"), dict) else None
            has_name = isinstance(item.get("name"), str) or (
                function is not None and isinstance(function.get("name"), str)
            )
            has_args = any(k in item for k in ("arguments", "parameters")) or (
                function is not None and "arguments" in function
            )
            if not (has_name and has_args):
                return []
            normalized = dict(item)
            if "arguments" not in normalized and "parameters" in normalized:
                normalized["arguments"] = normalized.get("parameters")
            call = self._parse_tool_json(json.dumps(normalized))
            if call is None:
                return []
            calls.append(call)
        return calls

    def _collect_harmony_tool_content(self) -> Tuple[str, List[ToolCall]]:
        """Collect and parse a Harmony/ChatML tool transcript block."""
        streamable_content = ""
        completed_tools: List[ToolCall] = []

        msg_tag = "<|message|>"
        msg_idx = self.current_tool_content.find(msg_tag)
        if msg_idx == -1:
            return streamable_content, completed_tools

        # Extract tool name from the header (between <|channel|> and <|message|>).
        header = self.current_tool_content[:msg_idx]
        name_match = re.search(r"\bto=([a-zA-Z0-9_\-\.]+)\b", header)
        if not name_match:
            return streamable_content, completed_tools
        tool_name = str(name_match.group(1) or "").strip()
        if tool_name.startswith("functions."):
            tool_name = tool_name.split(".", 1)[1].strip()
        if not tool_name:
            return streamable_content, completed_tools

        brace_start = self.current_tool_content.find("{", msg_idx + len(msg_tag))
        if brace_start == -1:
            return streamable_content, completed_tools
        between = self.current_tool_content[msg_idx + len(msg_tag) : brace_start]
        if between and any(not c.isspace() for c in between):
            return streamable_content, completed_tools

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

        brace_end = _find_matching_brace(self.current_tool_content, brace_start)
        if brace_end == -1:
            return streamable_content, completed_tools

        raw_args = self.current_tool_content[brace_start : brace_end + 1]
        args: Dict[str, Any] = {}
        call_id: Optional[str] = None
        loaded = loads_dict_like(raw_args)
        if isinstance(loaded, dict):
            # Some models emit a wrapper payload:
            #   {"name":"tool","arguments":{...},"call_id": "..."}
            inner_args = loaded.get("arguments")
            if isinstance(inner_args, dict):
                args = inner_args
            elif isinstance(inner_args, str):
                parsed_inner = loads_dict_like(inner_args)
                args = parsed_inner if isinstance(parsed_inner, dict) else loaded
            else:
                args = loaded

            call_id_value = loaded.get("call_id") or loaded.get("id")
            if isinstance(call_id_value, str) and call_id_value.strip():
                call_id = call_id_value.strip()

        completed_tools.append(ToolCall(name=tool_name, arguments=args, call_id=call_id))

        if self.rewrite_tags:
            # When rewriting, stream the full accumulated content (including the tool transcript).
            streamable_content = self.accumulated_content
            self.accumulated_content = ""

        remaining_content = self.current_tool_content[brace_end + 1 :]
        self.reset()

        if remaining_content:
            self.accumulated_content = remaining_content
            additional_streamable, additional_tools = self._scan_for_tool_start("")
            streamable_content += additional_streamable
            completed_tools.extend(additional_tools)

        return streamable_content, completed_tools

    def _might_have_partial_tool_call(self) -> bool:
        """Check if accumulated content might contain start of a tool call."""
        # Check for partial tool tags more aggressively to handle character-by-character streaming
        tail = (
            self.accumulated_content[-20:]
            if len(self.accumulated_content) > 20
            else self.accumulated_content
        )

        # Expanded list of potential partial starts to catch character-by-character streaming
        potential_partial_starts = [
            "<",
            "<|",
            "<f",
            "</",
            "<t",
            "`",
            "``",
            "<fu",
            "<fun",
            "<func",
            "<funct",
            "<functi",
            "<functio",
            "<function",  # <function_call>
            "<tool",
            "<tool_",
            "<tool_c",
            "<tool_ca",
            "<tool_cal",  # <tool_call>
            "<|t",
            "<|to",
            "<|too",
            "<|tool",
            "<|tool_",
            "<|tool_c",  # <|tool_call|>
            "<|tool_call_s",
            "<|tool_call_sta",
            "<|tool_call_start",  # <|tool_call_start|>
            "<|c",
            "<|ch",
            "<|cha",
            "<|chan",
            "<|chann",
            "<|channe",
            "<|channel",  # <|channel|>
            "<|m",
            "<|me",
            "<|mes",
            "<|mess",
            "<|messa",
            "<|messag",
            "<|message",  # <|message|>
        ]

        # Check if tail ends with any potential partial start
        for start in potential_partial_starts:
            if tail.endswith(start):
                return True

        # Also check if we have the start of any tag pattern in the middle
        for pattern_partial in [
            "<function",
            "<tool_call",
            "<|tool",
            "<|tool_call_start",
            "```tool",
        ]:
            if pattern_partial in tail:
                return True
        if "<|channel" in tail or "<|message" in tail:
            return True

        # Check if we have an incomplete tool call (start tag but no end tag)
        for pattern_info in self._scan_patterns():
            start_pattern = pattern_info["start"]
            end_pattern = pattern_info["end"]
            if end_pattern is None:
                continue

            start_match = re.search(start_pattern, self.accumulated_content, re.IGNORECASE)
            if start_match:
                # Has start tag - check if also has end tag AFTER it
                if not re.search(
                    end_pattern, self.accumulated_content[start_match.end() :], re.IGNORECASE
                ):
                    # Incomplete tool call - should buffer
                    return True

        return False

    def _extract_streamable_content(self) -> str:
        """Extract streamable content, buffering partial tool tags."""
        # Check if accumulated content might contain partial tool tag at the end
        tail = (
            self.accumulated_content[-20:]
            if len(self.accumulated_content) > 20
            else self.accumulated_content
        )

        tag_starters: Tuple[str, ...] = ("<", "<|", "</", "<|t", "<|to", "<|tool", "<function", "<tool", "``", "```")
        if self.json_fence_tools:
            # A lone backtick may be the first third of a ```json envelope
            # arriving one character per chunk.
            tag_starters = tag_starters + ("`",)
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

        cleaned = json_content.strip()

        if "<arg_key" in cleaned.lower():
            try:
                from ..tools.parser import _parse_arg_kv_tool_call

                arg_kv_calls = _parse_arg_kv_tool_call(cleaned)
            except Exception:
                arg_kv_calls = []
            if arg_kv_calls:
                return arg_kv_calls[0]

        if "<parameter" in cleaned.lower():
            try:
                from ..tools.parser import _parse_xmlish_parameter_tool_calls

                xmlish_calls = _parse_xmlish_parameter_tool_calls(cleaned)
            except Exception:
                xmlish_calls = []
            if xmlish_calls:
                return xmlish_calls[0]

        # Gemma4-style tool-call payloads:
        #   call:tool_name{...json args...}
        call_match = re.search(r"(?is)\bcall\s*:\s*(?P<name>\w+)\s*(?P<arguments>\{.*\})", cleaned)
        if call_match:
            name = call_match.group("name")
            args_raw = call_match.group("arguments")
            try:
                parsed_args = loads_dict_like(args_raw)
            except Exception:
                parsed_args = None
            arguments = parsed_args if isinstance(parsed_args, dict) else {}
            if isinstance(name, str) and name.strip():
                return ToolCall(name=name.strip(), arguments=arguments, call_id=None)

        try:
            from ..tools.parser import _parse_pythonic_tool_payload

            pythonic_calls = _parse_pythonic_tool_payload(cleaned)
        except Exception:
            pythonic_calls = []
        if pythonic_calls:
            return pythonic_calls[0]

        # Handle missing braces (best-effort).
        if cleaned.count("{") > cleaned.count("}"):
            missing = cleaned.count("{") - cleaned.count("}")
            cleaned += "}" * missing

        tool_data: Optional[Dict[str, Any]] = None
        try:
            tool_data = loads_dict_like(cleaned)
        except Exception as e:
            logger.debug(f"Tool JSON-ish parse error: {e}, content: {repr(json_content)}")
            tool_data = None

        if not isinstance(tool_data, dict):
            return None

        name: Any = tool_data.get("name")
        arguments: Any = tool_data.get("arguments")
        call_id: Any = tool_data.get("call_id") or tool_data.get("id")

        # OpenAI-style wrapper payload: {"id":"...","type":"function","function":{"name":...,"arguments":"{...}"}}
        function = (
            tool_data.get("function") if isinstance(tool_data.get("function"), dict) else None
        )
        if function:
            if not isinstance(name, str) or not name.strip():
                name = function.get("name")
            if arguments is None:
                arguments = function.get("arguments")

        # Anthropic-ish key used by some tool payloads.
        if arguments is None and "input" in tool_data:
            arguments = tool_data.get("input")

        # Normalize arguments to a dict.
        if isinstance(arguments, str):
            parsed_args = loads_dict_like(arguments)
            arguments = parsed_args if isinstance(parsed_args, dict) else {}
        if not isinstance(arguments, dict):
            arguments = {}

        if not isinstance(name, str) or not name.strip():
            return None

        call_id_str: Optional[str] = None
        if isinstance(call_id, str) and call_id.strip():
            call_id_str = call_id.strip()

        return ToolCall(name=name.strip(), arguments=arguments, call_id=call_id_str)

    def finalize(self) -> List[ToolCall]:
        """Finalize and return any remaining tool calls."""
        completed_tools = []

        if self.state == ToolDetectionState.IN_TOOL_CALL:
            # Try to parse any remaining content as incomplete tool
            if self.current_tool_content:
                if "<arg_key" in self.current_tool_content.lower():
                    try:
                        from ..tools.parser import _parse_arg_kv_tool_call

                        completed_tools.extend(_parse_arg_kv_tool_call(self.current_tool_content))
                    except Exception:
                        pass
                    if completed_tools:
                        self.accumulated_content = ""
                        return completed_tools

                if "<parameter" in self.current_tool_content.lower():
                    try:
                        from ..tools.parser import _parse_xmlish_parameter_tool_calls

                        completed_tools.extend(
                            _parse_xmlish_parameter_tool_calls(self.current_tool_content)
                        )
                    except Exception:
                        pass
                    if completed_tools:
                        self.accumulated_content = ""
                        return completed_tools

                # Try to parse incomplete JSON by looking for valid JSON objects
                tool_call = self._try_parse_incomplete_json(self.current_tool_content)
                if tool_call:
                    completed_tools.append(tool_call)
                    self.accumulated_content = ""

        return completed_tools

    def drain(self) -> Tuple[str, List[ToolCall], Optional[Dict[str, Any]]]:
        """End of the model's output: (content still owed, tool calls, unparsed tool call).

        Content held back only because it MIGHT start a tag is owed to the
        caller. An envelope that opened and never closed (or closed around a
        body no parser accepts) is NOT content: it is returned as the third
        element, `{"format", "text", "reason"}`, so the caller can report it
        without printing tool-call markup as the model's answer. Resets the
        detector.
        """
        unparsed: Optional[Dict[str, Any]] = None
        if self.state == ToolDetectionState.IN_TOOL_CALL:
            start = self.tool_start_pos or 0
            # Normal mode already streamed the text before the envelope;
            # rewrite mode held it back, so it is still owed.
            owed = self.accumulated_content[:start] if self.rewrite_tags else ""
            envelope = self.accumulated_content[start:]
            pattern = self.current_pattern or {}
            if pattern.get("kind") == "json_fence":
                # An unclosed ```json block: a complete call body is a call; a
                # body that STARTS like a call (first key name/tool_calls/
                # function/type) is a broken call; anything else is an ordinary
                # unfinished JSON answer and stays content.
                body = self.current_tool_content
                tools = self._parse_json_fence_tool_calls(body)
                if not tools and not re.match(
                    r'\s*\[?\s*\{\s*"(name|tool_calls|function|type)"\s*:', body
                ):
                    owed = self.accumulated_content if self.rewrite_tags else envelope
                    self.reset()
                    return owed, [], None
            else:
                tools = self.finalize()
            if not tools:
                unparsed = {
                    "format": self._pattern_name(pattern),
                    "text": envelope,
                    "reason": "unclosed" if pattern.get("kind") != "harmony" else "incomplete",
                }
            self.reset()
            return owed, tools, unparsed

        owed = self.accumulated_content
        self.reset()
        return owed, [], None

    def _pattern_name(self, pattern: Dict[str, Any]) -> str:
        if pattern is self.json_fence_pattern:
            return "json_fence"
        for name, candidate in self.patterns.items():
            if candidate is pattern:
                return name
        return "unknown"

    def _try_parse_incomplete_json(self, content: str) -> Optional[ToolCall]:
        """Try to parse potentially incomplete JSON by finding valid JSON objects."""
        # Look for complete JSON objects within the content
        brace_count = 0
        json_start = -1

        for i, char in enumerate(content):
            if char == "{":
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and json_start >= 0:
                    # Found complete JSON object
                    json_content = content[json_start : i + 1]
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

    def __init__(
        self,
        model_name: str,
        execute_tools: bool = False,
        tool_call_tags: Optional[object] = None,
        default_target_format: str = "qwen3",
    ):
        """Initialize the stream processor."""
        self.model_name = model_name
        # Note: execute_tools is kept for backward compatibility and introspection,
        # but tool execution is handled by the client/runtime (AbstractRuntime).
        self.execute_tools = execute_tools
        self.tool_call_tags = tool_call_tags
        self.default_target_format = default_target_format

        # Initialize tag rewriter only when explicit format conversion is requested.
        self.tag_rewriter = None
        # Backwards compatibility: tag_rewrite_buffer attribute (unused in current implementation)
        self.tag_rewrite_buffer = ""

        # Flag to indicate if we're converting to OpenAI JSON format (not text rewriting)
        self.convert_to_openai_json = False

        # Determine whether tool_call_tags contains predefined format or custom tags.
        if tool_call_tags:
            # Accept pre-built tag rewriters/config objects (used by some internal callers/tests).
            if not isinstance(tool_call_tags, str):
                self._initialize_tag_rewriter(tool_call_tags)
            else:
                # Check if tool_call_tags is a predefined format name
                predefined_formats = ["qwen3", "openai", "llama3", "xml", "gemma"]

                if tool_call_tags in predefined_formats:
                    # It's a predefined format - use default rewriter
                    self._initialize_default_rewriter(tool_call_tags)
                    logger.debug(f"Treating tool_call_tags '{tool_call_tags}' as predefined format")
                elif "," in tool_call_tags:
                    # It contains comma - likely custom tags like "START,END"
                    self._initialize_tag_rewriter(tool_call_tags)
                    logger.debug(
                        f"Treating tool_call_tags '{tool_call_tags}' as custom comma-separated tags"
                    )
                else:
                    # Single string that's not a predefined format - could be custom single tag
                    # Try as custom first, fall back to treating as predefined format
                    try:
                        self._initialize_tag_rewriter(tool_call_tags)
                        logger.debug(
                            f"Treating tool_call_tags '{tool_call_tags}' as custom single tag"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to initialize as custom tag, trying as predefined format: {e}"
                        )
                        self._initialize_default_rewriter(tool_call_tags)
        else:
            # No explicit format conversion requested - no text rewriting.
            self.tag_rewriter = None

        # Create detector.
        #
        # Default UX: remove tool-call markup from visible content and surface it via `tool_calls`.
        # Only preserve tool-call markup when the caller explicitly requested tag conversion.
        #
        # Note: treat empty/whitespace tool_call_tags like "no rewrite" (avoid leaking raw tags).
        preserve_for_rewriting = bool(self.convert_to_openai_json or tool_call_tags)

        self.detector = IncrementalToolDetector(
            model_name=model_name, rewrite_tags=preserve_for_rewriting
        )

    def process_stream(
        self,
        response_stream: Iterator[GenerateResponse],
        converted_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Iterator[GenerateResponse]:
        """
        Process a response stream with tag rewriting and tool detection.

        Args:
            response_stream: Iterator of response chunks
            converted_tools: Available tools for execution

        Yields:
            GenerateResponse: Processed chunks with rewritten tags
        """
        # Per-REQUEST metadata that must survive onto the finalize chunks below,
        # which are constructed from scratch and would otherwise drop it. Only
        # request-scoped keys travel; per-chunk keys (ttft, reasoning deltas) do
        # not.
        request_metadata: Dict[str, Any] = {}
        try:
            # Roster for recovering namespaced/decorated names on THIS lane too:
            # the non-streaming passthrough maps `functions.browser_probe` ->
            # `browser_probe`, and a host executing by name must see the same
            # call regardless of stream mode.
            allowed_tool_names: set = set()
            for _tool in converted_tools or []:
                if not isinstance(_tool, dict):
                    continue
                _name = _tool.get("name")
                if isinstance(_name, str) and _name.strip():
                    allowed_tool_names.add(_name.strip())
                    continue
                _func = _tool.get("function") if isinstance(_tool.get("function"), dict) else None
                _fname = _func.get("name") if isinstance(_func, dict) else None
                if isinstance(_fname, str) and _fname.strip():
                    allowed_tool_names.add(_fname.strip())

            def _mapped_tool_name(raw_name: str) -> tuple:
                """(possibly-mapped name, warning or None) under the shared rules."""
                name = str(raw_name or "").strip()
                if not name or not allowed_tool_names or name in allowed_tool_names:
                    return name, None
                try:
                    from ..tools.wire_naming import map_namespaced_tool_name, resolve_wire_tool_name

                    resolved = resolve_wire_tool_name(name, allowed_tool_names)
                    if not resolved:
                        resolved = map_namespaced_tool_name(name, allowed_tool_names)
                except Exception:
                    resolved = None
                if resolved:
                    return resolved, None
                return name, (
                    f"Tool call '{name}' does not match any available tool "
                    f"(available: {', '.join(sorted(allowed_tool_names))})."
                )

            def _mapped_tool_payload(tools_list) -> tuple:
                payload = []
                warnings: List[str] = []
                for tc in tools_list:
                    if not getattr(tc, "name", None):
                        continue
                    mapped_name, warning = _mapped_tool_name(tc.name)
                    if warning:
                        warnings.append(warning)
                    payload.append(
                        {
                            "name": mapped_name,
                            "arguments": tc.arguments,
                            "call_id": tc.call_id,
                        }
                    )
                return payload, warnings

            def _with_warnings(metadata: Optional[Dict[str, Any]], warnings: List[str]) -> Optional[Dict[str, Any]]:
                if not warnings:
                    return metadata
                meta = dict(metadata or {})
                existing = meta.get("warnings")
                merged = list(existing) if isinstance(existing, list) else []
                merged.extend(warnings)
                meta["warnings"] = merged
                for w in warnings:
                    logger.warning(w)
                return meta

            def _canonical_tool_call_key(call: Dict[str, Any]) -> Optional[tuple]:
                """Best-effort key for deduplicating canonical tool-call payloads."""
                name = call.get("name")
                if not isinstance(name, str) or not name.strip():
                    return None

                call_id = call.get("call_id") or call.get("id")
                call_id_norm: Optional[str]
                if isinstance(call_id, str) and call_id.strip():
                    call_id_norm = call_id.strip()
                else:
                    call_id_norm = None

                args = call.get("arguments")
                if isinstance(args, dict):
                    try:
                        args_norm = json.dumps(args, sort_keys=True, separators=(",", ":"))
                    except Exception:
                        args_norm = str(args)
                else:
                    args_norm = str(args)

                return (name.strip(), args_norm, call_id_norm)

            # ```json tool-call envelopes are only recognised when the request
            # carries tools (see IncrementalToolDetector.json_fence_pattern).
            self.detector.json_fence_tools = bool(allowed_tool_names)

            def _rewritten(text: str) -> str:
                if self.convert_to_openai_json:
                    return self._convert_to_openai_format(text)
                if self.tag_rewriter:
                    return self._apply_tag_rewriting_direct(text)
                return text

            def _unparsed_metadata(
                metadata: Optional[Dict[str, Any]], unparsed: Optional[Dict[str, Any]]
            ) -> Optional[Dict[str, Any]]:
                """Report an envelope that never became a tool call, never as content."""
                if not unparsed:
                    return metadata
                meta = dict(metadata or {})
                meta["unparsed_tool_call"] = unparsed
                preview = preview_text(unparsed.get("text") or "", max_chars=200)
                return _with_warnings(
                    meta,
                    [
                        f"Unparsed tool call ({unparsed.get('format')}, {unparsed.get('reason')}): "
                        f"the model opened a tool-call envelope that no parser accepted; it is "
                        f"reported in metadata['unparsed_tool_call'], not as content: {preview!r}"
                    ],
                )

            def _drain(model: Any) -> Tuple[List[GenerateResponse], Optional[Dict[str, Any]]]:
                """Chunks owed at the end of the model's output (no finish_reason) + unparsed call."""
                owed, tools, unparsed = self.detector.drain()
                out: List[GenerateResponse] = []
                if owed:
                    out.append(GenerateResponse(content=_rewritten(owed), model=model))
                if tools:
                    tool_payload, name_warnings = _mapped_tool_payload(tools)
                    if tool_payload:
                        out.append(
                            GenerateResponse(
                                content="",
                                tool_calls=tool_payload,
                                model=model,
                                metadata=_with_warnings(None, name_warnings),
                            )
                        )
                return out, unparsed

            def _detector_pending() -> bool:
                return bool(
                    self.detector.accumulated_content
                    or self.detector.state == ToolDetectionState.IN_TOOL_CALL
                )

            for chunk in response_stream:
                # The provider's TERMINAL chunk (finish_reason set) carries the
                # accounting — usage, prompt-cache telemetry — and consumers read
                # it as the stream's last word. Content the detector still holds
                # (a possible tag, an unclosed envelope) is settled BEFORE it, so
                # the terminal chunk stays last and its finish_reason is not
                # overwritten by a synthetic "stop".
                terminal = chunk.finish_reason is not None

                # Preserve provider-emitted tool calls (native tools / server-side tool_calls).
                incoming_tool_calls = (
                    chunk.tool_calls
                    if isinstance(getattr(chunk, "tool_calls", None), list) and chunk.tool_calls
                    else None
                )

                incoming_tool_call_keys = set()
                if incoming_tool_calls:
                    for call in incoming_tool_calls:
                        if not isinstance(call, dict):
                            continue
                        key = _canonical_tool_call_key(call)
                        if key:
                            incoming_tool_call_keys.add(key)

                if isinstance(chunk.metadata, dict):
                    for _k in ("media_delivered", "media_dropped", "prompt_cache"):
                        if _k in chunk.metadata:
                            request_metadata[_k] = chunk.metadata[_k]

                if not chunk.content:
                    if terminal and _detector_pending():
                        owed_chunks, unparsed = _drain(chunk.model)
                        yield from owed_chunks
                        chunk.metadata = _unparsed_metadata(chunk.metadata, unparsed)
                    yield chunk
                    continue

                # A terminal chunk with content: when the detector ends up
                # holding something, its content goes out first and the
                # terminal attributes ride a separate, final, empty chunk.
                terminal_attrs: Optional[GenerateResponse] = None

                # Process chunk through detector (preserves tool calls for rewriting)
                streamable_content, completed_tools = self.detector.process_chunk(chunk.content)
                if terminal and _detector_pending():
                    terminal_attrs = GenerateResponse(
                        content="",
                        model=chunk.model,
                        finish_reason=chunk.finish_reason,
                        usage=chunk.usage,
                        raw_response=chunk.raw_response,
                        metadata=chunk.metadata,
                    )
                    chunk = GenerateResponse(
                        content=chunk.content,
                        model=chunk.model,
                        tool_calls=chunk.tool_calls,
                    )

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

                # Per-REQUEST metadata (media-delivery record, prompt-cache
                # telemetry) was captured above: it has to survive onto the
                # finalize chunks below, which are built from scratch. Per-CHUNK
                # keys (ttft, reasoning deltas) deliberately do not travel.

                # Yield streamable content
                if streamable_content:
                    yield GenerateResponse(
                        content=streamable_content,
                        model=chunk.model,
                        finish_reason=chunk.finish_reason,
                        usage=chunk.usage,
                        raw_response=chunk.raw_response,
                        metadata=chunk.metadata,
                        tool_calls=incoming_tool_calls,
                    )

                    # If we emitted content alongside provider-emitted tool calls, do not emit them again.
                    incoming_tool_calls = None

                # If the incoming chunk had tool_calls but we did not emit any content (buffering/tag parsing),
                # still surface the tool_calls to downstream hosts.
                if incoming_tool_calls:
                    yield GenerateResponse(
                        content="",
                        tool_calls=incoming_tool_calls,
                        model=chunk.model,
                        finish_reason=chunk.finish_reason,
                        usage=chunk.usage,
                        raw_response=chunk.raw_response,
                        metadata=chunk.metadata,
                    )

                # Yield tool calls for server processing
                if completed_tools:
                    logger.debug(
                        f"Detected {len(completed_tools)} tools - yielding for server processing"
                    )
                    tool_payload, name_warnings = _mapped_tool_payload(completed_tools)
                    if incoming_tool_call_keys:
                        tool_payload = [
                            call
                            for call in tool_payload
                            if (
                                isinstance(call, dict)
                                and _canonical_tool_call_key(call) not in incoming_tool_call_keys
                            )
                        ]
                    if tool_payload:
                        yield GenerateResponse(
                            content="",
                            tool_calls=tool_payload,
                            model=chunk.model,
                            finish_reason=chunk.finish_reason,
                            usage=chunk.usage,
                            raw_response=chunk.raw_response,
                            metadata=_with_warnings(chunk.metadata, name_warnings),
                        )

                if terminal_attrs is not None:
                    owed_chunks, unparsed = _drain(chunk.model)
                    yield from owed_chunks
                    terminal_attrs.metadata = _unparsed_metadata(terminal_attrs.metadata, unparsed)
                    yield terminal_attrs

            # Stream ended without a terminal chunk settling the detector (or
            # content arrived after it): settle it now. These chunks are the
            # stream's last, so they carry the request-scoped metadata.
            owed, final_tools, unparsed = self.detector.drain()

            if owed:
                yield GenerateResponse(
                    content=_rewritten(owed),
                    model=self.model_name,
                    finish_reason="stop",
                    metadata=_unparsed_metadata(dict(request_metadata) or None, None if final_tools else unparsed),
                )
                if not final_tools:
                    unparsed = None

            if final_tools:
                logger.debug(f"Finalized {len(final_tools)} tools - yielding for server processing")
                tool_payload, name_warnings = _mapped_tool_payload(final_tools)
                yield GenerateResponse(
                    content="",
                    tool_calls=tool_payload,
                    model=self.model_name,
                    finish_reason="tool_calls",
                    metadata=_with_warnings(dict(request_metadata) or None, name_warnings),
                )
            elif unparsed:
                yield GenerateResponse(
                    content="",
                    model=self.model_name,
                    finish_reason="stop",
                    metadata=_unparsed_metadata(dict(request_metadata) or None, unparsed),
                )

        except Exception as e:
            if getattr(e, "request_local", False) is True and type(e).__name__ == "GenerationCancelledError":
                # A host Stop is not a stream failure (generation_cancel.py).
                logger.info(f"Unified stream stopped by the host: {e}")
            else:
                logger.error(f"Error in unified stream processing: {e}")
            raise

    def _initialize_tag_rewriter(self, tool_call_tags):
        """Initialize the tag rewriter from tool_call_tags configuration."""
        try:
            from ..tools.tag_rewriter import ToolCallTagRewriter, ToolCallTags

            if isinstance(tool_call_tags, str):
                # Parse string format: either "start,end" or just "start"
                if "," in tool_call_tags:
                    # Comma-separated: User specified both start and end tags
                    # Store as plain tags, rewriter will wrap with angle brackets
                    parts = tool_call_tags.split(",")
                    if len(parts) == 2:
                        tags = ToolCallTags(
                            start_tag=parts[0].strip(),
                            end_tag=parts[1].strip(),
                            auto_format=False,  # Don't auto-format, keep plain tags
                        )
                    else:
                        logger.warning(f"Invalid tool_call_tags format: {tool_call_tags}")
                        return
                else:
                    # Single tag: Auto-format to <tag> and </tag>
                    tags = ToolCallTags(
                        start_tag=tool_call_tags.strip(),
                        end_tag=tool_call_tags.strip(),
                        auto_format=True,  # Enable auto-formatting for single tags
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
            if "," in target_format:
                # Custom tag format: "START,END"
                parts = target_format.split(",")
                if len(parts) == 2:
                    target_tags = ToolCallTags(
                        start_tag=parts[0].strip(),
                        end_tag=parts[1].strip(),
                        auto_format=False,  # Use exact custom tags
                    )
                    self.tag_rewriter = ToolCallTagRewriter(target_tags)
                    logger.debug(
                        f"Initialized custom tag rewriter '{parts[0].strip()}...{parts[1].strip()}' for model {self.model_name}"
                    )
                else:
                    logger.warning(
                        f"Invalid custom tag format '{target_format}' - expected 'START,END'"
                    )
                    return
            elif target_format == "qwen3":
                # Qwen3 format: <|tool_call|>...JSON...</|tool_call|>
                target_tags = ToolCallTags(
                    start_tag="<|tool_call|>",
                    end_tag="</|tool_call|>",
                    auto_format=False,  # Use exact tags
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
                logger.debug(
                    f"OpenAI format selected - will convert text-based tool calls to OpenAI JSON format"
                )
                return
            elif target_format == "llama3":
                # LLaMA3/Crush CLI format: <function_call>...JSON...</function_call>
                target_tags = ToolCallTags(
                    start_tag="<function_call>", end_tag="</function_call>", auto_format=False
                )
                self.tag_rewriter = ToolCallTagRewriter(target_tags)
                logger.debug(f"Initialized llama3 tag rewriter for model {self.model_name}")
            elif target_format == "xml":
                # XML/Gemini CLI format: <tool_call>...JSON...</tool_call>
                target_tags = ToolCallTags(
                    start_tag="<tool_call>", end_tag="</tool_call>", auto_format=False
                )
                self.tag_rewriter = ToolCallTagRewriter(target_tags)
                logger.debug(f"Initialized xml tag rewriter for model {self.model_name}")
            elif target_format == "gemma":
                # Gemma format: ```tool_code...JSON...```
                target_tags = ToolCallTags(
                    start_tag="```tool_code\n", end_tag="\n```", auto_format=False
                )
                self.tag_rewriter = ToolCallTagRewriter(target_tags)
                logger.debug(f"Initialized gemma tag rewriter for model {self.model_name}")
            else:
                # Try to handle as single tag format (auto-format to <tag>...</tag>)
                if target_format and not target_format.isspace():
                    target_tags = ToolCallTags(
                        start_tag=target_format.strip(),
                        end_tag=target_format.strip(),
                        auto_format=True,  # Auto-wrap with angle brackets
                    )
                    self.tag_rewriter = ToolCallTagRewriter(target_tags)
                    logger.debug(
                        f"Initialized auto-formatted tag rewriter '<{target_format.strip()}>...</{target_format.strip()}>' for model {self.model_name}"
                    )
                else:
                    logger.warning(
                        f"Unknown or empty target format: '{target_format}' - no tag rewriting will be applied"
                    )

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
                logger.debug(
                    f"Tag rewriting successful: {preview_text(content, max_chars=50)} -> {preview_text(rewritten, max_chars=50)}"
                )
            else:
                logger.debug(
                    f"Tag rewriting had no effect on: {preview_text(content, max_chars=50)}"
                )
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
            (r"<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>", "qwen3"),
            (r"<function_call>\s*(.*?)\s*</function_call>", "llama"),
            (r"<tool_call>\s*(.*?)\s*</tool_call>", "xml"),
            (r"```tool_code\s*\n(.*?)\n```", "gemma"),
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

                        # Parse the JSON-ish payload to validate and extract fields.
                        tool_data = loads_dict_like(json_content)

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
                                "arguments": json.dumps(
                                    (
                                        tool_data.get("arguments")
                                        if isinstance(tool_data.get("arguments"), dict)
                                        else None
                                    )
                                    or (
                                        loads_dict_like(tool_data.get("arguments"))
                                        if isinstance(tool_data.get("arguments"), str)
                                        else None
                                    )
                                    or {}
                                ),
                            },
                        }

                        # Replace the text-based tool call with OpenAI JSON format
                        openai_json = json.dumps(openai_tool_call)
                        converted_content = (
                            converted_content[: match.start()]
                            + openai_json
                            + converted_content[match.end() :]
                        )

                        logger.debug(
                            f"Converted {format_type} tool call to OpenAI format: {openai_json[:100]}"
                        )

                    except Exception as e:
                        logger.error(f"Error converting tool call to OpenAI format: {e}")
                        continue

                # Only process the first matching format type
                break

        return converted_content

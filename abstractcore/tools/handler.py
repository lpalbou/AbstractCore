"""
Universal tool handler for all models and providers.

This module provides a utility class for tool support that works
across all models, whether they have native tool APIs or require prompting.
"""

import json
import warnings
from typing import List, Dict, Any, Optional, Union, Callable

from ..architectures import detect_architecture, get_model_capabilities, get_architecture_format
from .core import ToolDefinition, ToolCall, ToolCallResponse, ToolResult
from .parser import detect_tool_calls, parse_tool_calls, format_tool_prompt
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


def merge_tools_into_system(
    handler: Any,
    system_prompt: Optional[str],
    tools: Optional[List[Any]],
    *,
    include_tool_list_override: Optional[bool] = None,
) -> Optional[str]:
    """Merge the prompted tool block into ONE system turn (prompted lane).

    Single source of truth for tool PLACEMENT — the policy that was copy-pasted
    across every provider (the ``supports_prompted`` gate, the
    ``"## Tools (session)"`` dedup sentinel, and the one-system-turn ``\\n\\n``
    merge). The tool TEXT itself lives in ``handler.format_tools_prompt``; this
    owns only WHERE it goes. Returns ``system_prompt`` unchanged when there are
    no prompted tools to add (native-only handlers included).

    A free function (not a handler method) so it composes with any object
    exposing ``supports_prompted`` + ``format_tools_prompt`` — the real
    ``UniversalToolHandler`` and every provider's test double alike.

    Byte-contract: the merged form is exactly ``f"{system}\\n\\n{tools}"`` (or
    the tool block alone when there is no system prompt) — identical to what the
    sites produced before, so prompt-cache byte-parity holds.
    """
    if not tools or not getattr(handler, "supports_prompted", False):
        return system_prompt
    if include_tool_list_override is None:
        include_tool_list = not (system_prompt and "## Tools (session)" in system_prompt)
    else:
        include_tool_list = bool(include_tool_list_override)
    tool_prompt = handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
    if not tool_prompt:
        return system_prompt
    if system_prompt:
        return f"{system_prompt}\n\n{tool_prompt}"
    return tool_prompt


class UniversalToolHandler:
    """
    Universal tool handler that works with all models.

    This handler automatically detects model capabilities and provides:
    - Tool prompt formatting for prompted models
    - Native tool formatting for API models
    - Response parsing for tool calls
    - Architecture-specific handling
    """

    def __init__(self, model_name: str):
        """
        Initialize handler for a specific model.

        Args:
            model_name: Model identifier
        """
        self.model_name = model_name
        self.architecture = detect_architecture(model_name)
        self.capabilities = get_model_capabilities(model_name)
        self.architecture_format = get_architecture_format(self.architecture)

        # Determine support levels
        tool_support = self.capabilities.get("tool_support", "none")
        self.supports_native = tool_support == "native"
        self.supports_prompted = tool_support in ["native", "prompted"]

        logger.debug(f"Initialized tool handler for {model_name}: "
                    f"architecture={self.architecture}, "
                    f"native={self.supports_native}, "
                    f"prompted={self.supports_prompted}")

    def format_tools_prompt(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]],
        *,
        include_tool_list: bool = True,
        include_examples: bool = True,
    ) -> str:
        """
        Format tools into a system prompt for prompted models.

        Args:
            tools: List of tools (ToolDefinition, callable, or dict)

        Returns:
            Formatted tool prompt string
        """
        if not tools or not self.supports_prompted:
            return ""

        # Convert all tools to ToolDefinition objects
        tool_defs = self._convert_to_tool_definitions(tools)
        if not tool_defs:
            return ""

        # Use architecture-specific formatting
        return format_tool_prompt(
            tool_defs,
            self.model_name,
            include_tool_list=include_tool_list,
            include_examples=include_examples,
        )

    def prepare_tools_for_native(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to native API format.

        Args:
            tools: List of tools

        Returns:
            List of tool dictionaries for native API
        """
        if not tools or not self.supports_native:
            return []

        # Convert all tools to ToolDefinition objects
        tool_defs = self._convert_to_tool_definitions(tools)
        if not tool_defs:
            return []

        # Return as dictionaries for native API
        native_tools = []
        for tool_def in tool_defs:
            # Clean parameters by removing 'default' properties for OpenAI compatibility
            cleaned_properties = {}
            for name, param in tool_def.parameters.items():
                if isinstance(param, dict):
                    # Remove 'default' key from parameter definition
                    cleaned_param = {k: v for k, v in param.items() if k != "default"}
                    cleaned_properties[name] = cleaned_param
                else:
                    cleaned_properties[name] = param

            # Extract required fields (fields without default values)
            required_fields = []
            for name, param in tool_def.parameters.items():
                if isinstance(param, dict) and "default" not in param:
                    required_fields.append(name)

            # Convert to OpenAI-style function format (most common).
            # Wire-safe alias: namespaced names (mcp::server::tool) violate the
            # strict native name contract (^[a-zA-Z0-9_-]{1,64}$) and 400 the
            # WHOLE request on OpenAI/Anthropic-strict endpoints. Safe names
            # pass through byte-identical; the response normalizer maps the
            # alias back to the original (see tools.wire_naming).
            from .wire_naming import wire_safe_tool_name

            # `when_to_use` is authored, MODEL-VISIBLE guidance. The prompted path
            # renders it unconditionally — `parser._should_render_when_to_use` was
            # changed to always-True because dropping it is the silent lossy
            # truncation ADR 0001 forbids. Native used to drop the same content
            # here, silently, and native is the default path for capable
            # providers. Fold it into `description`, the one field every strict
            # provider schema already accepts, leading with the one-sentence
            # description so selection-time scannability is unchanged.
            #
            # This is not new footprint: it is the footprint the prompted path
            # already carries. Merging moves a field boundary, it adds no bytes.
            # The 200/240 authoring caps (`core._validate_tool_metadata`) are
            # untouched — this string is a derived send-time artifact and is
            # deliberately NOT validated against the authoring cap.
            description = tool_def.description
            if tool_def.when_to_use:
                description = f"{description}\n\nWhen to use: {tool_def.when_to_use}"

            native_tool = {
                "type": "function",
                "function": {
                    "name": wire_safe_tool_name(tool_def.name),
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": cleaned_properties,
                        "required": required_fields
                    }
                }
            }

            # NOTE: Do not include custom KEYS (tags/examples) in native tool payloads.
            # Most provider native tool schemas validate strictly and may reject unknown fields.
            # `when_to_use` is carried as prose inside `description` above, not as a key.

            native_tools.append(native_tool)

        return native_tools

    def parse_response(
        self,
        response: Union[str, Dict[str, Any]],
        mode: str = "prompted"
    ) -> ToolCallResponse:
        """
        Parse model response for tool calls.

        Args:
            response: Model response (string for prompted, dict for native)
            mode: Response mode ("native" or "prompted")

        Returns:
            ToolCallResponse with content and tool calls
        """
        if mode == "native" and isinstance(response, dict):
            return self._parse_native_response(response)
        elif mode == "prompted" and isinstance(response, str):
            return self._parse_prompted_response(response)
        else:
            # Fallback - try to handle whatever we get
            if isinstance(response, str):
                return self._parse_prompted_response(response)
            else:
                return self._parse_native_response(response)

    @staticmethod
    def _dropped_tool_name(tool: Any) -> str:
        """Best-effort name for a tool that could not be converted (for the warning)."""
        try:
            if isinstance(tool, dict):
                fn = tool.get("function")
                if isinstance(fn, dict) and fn.get("name"):
                    return str(fn.get("name"))
                if tool.get("name"):
                    return str(tool.get("name"))
                return f"<dict with keys {sorted(tool.keys())[:6]}>"
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            return str(name) if name else f"<{type(tool).__name__}>"
        except Exception:
            return "<unknown>"

    @staticmethod
    def _tool_source_label(tool: Any) -> str:
        """Where an external tool came from, for the correction report.

        MCP specs carry `origin`; anything else is just "the caller". Naming the
        server is what turns "some description is too long" into a fix someone
        can actually make.
        """
        try:
            origin = tool.get("origin") if isinstance(tool, dict) else None
            if isinstance(origin, dict):
                server = str(origin.get("server_id") or "").strip()
                if server:
                    return f"MCP server '{server}'"
                kind = str(origin.get("type") or "").strip()
                if kind:
                    return f"{kind} source"
            tags = tool.get("tags") if isinstance(tool, dict) else None
            if isinstance(tags, list):
                for tag in tags:
                    text = str(tag)
                    if text.startswith("mcp_server:"):
                        return f"MCP server '{text.split(':', 1)[1]}'"
        except Exception:
            pass
        return "the caller's tool list"

    def _warn_tool_dropped(self, tool: Any, reason: str) -> None:
        """A tool the caller asked for will NOT reach the model. Say so, loudly.

        ADR 0001 (no silent degradation). This used to be `logger.warning` only,
        which is DEAD in a default AbstractCore process (root logger is ERROR and
        every `abstractcore.*` logger is NOTSET), so a dropped tool produced NO
        output anywhere. The failure mode that made this urgent, measured
        2026-08-07: `ToolDefinition` caps `description` at 200 chars, so a tool
        set carrying long descriptions was converted to ZERO definitions ->
        `format_tools_prompt` returned "" -> the local lanes (MLX, HF
        transformers, HF GGUF) rendered a prompt with NO TOOLS AT ALL, while the
        prompt-cache bloc plan happily reported a healthy 3-token "tools" bloc.
        The caller had no way to find out short of diffing rendered prompts.
        """
        name = self._dropped_tool_name(tool)
        message = (
            f"Tool '{name}' was DROPPED and will not be sent to the model "
            f"({self.model_name}): {str(reason).strip().rstrip('.')}. Fix the tool "
            f"definition — the model cannot call a tool it never sees."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=4)
        logger.warning(message)

    def _convert_to_tool_definitions(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
    ) -> List[ToolDefinition]:
        """Convert various tool formats to ToolDefinition objects.

        Every tool that does not make it into the returned list is announced via
        `warnings.warn` — see `_warn_tool_dropped`. This method feeds BOTH the
        prompted lane (`format_tools_prompt`) and the native lane
        (`prepare_tools_for_native`), so a silent drop here removes a capability
        on every provider.
        """
        tool_defs = []

        for tool in tools:
            try:
                if isinstance(tool, ToolDefinition):
                    tool_defs.append(tool)
                elif callable(tool):
                    # Check if tool has enhanced metadata from @tool decorator
                    if hasattr(tool, '_tool_definition'):
                        tool_defs.append(tool._tool_definition)
                    else:
                        # An undecorated callable is IMPORTED material, not authored
                        # here: its docstring may belong to a third-party library we
                        # cannot edit. `from_function` raises on an over-long first
                        # docstring line, which used to cost the caller the tool.
                        tool_defs.append(ToolDefinition.from_function(
                            tool,
                            external_source=(
                                f"an undecorated callable in "
                                f"{getattr(tool, '__module__', '?')}"
                            ),
                        ))
                elif isinstance(tool, dict):
                    if "name" in tool and "description" in tool:
                        # Direct dict format - extract properties from full schema
                        parameters = tool.get("parameters", {})
                        # If parameters is a full JSON schema, extract just the properties
                        if isinstance(parameters, dict) and "properties" in parameters:
                            properties = parameters["properties"]
                        else:
                            properties = parameters

                        # A dict tool is by construction EXTERNAL — this is the shape
                        # MCP servers, gateway payloads and hand-written JSON arrive
                        # in, and we cannot edit their source. `from_external` adapts
                        # an over-long description and reports it, instead of raising
                        # into the handler's `except` and losing the tool entirely.
                        tool_defs.append(ToolDefinition.from_external(
                            name=tool["name"],
                            description=tool["description"],
                            parameters=properties,
                            source=self._tool_source_label(tool),
                            tags=tool.get("tags", []),
                            when_to_use=tool.get("when_to_use"),
                            examples=tool.get("examples", []),
                            act_only=bool(tool.get("act_only", False))
                        ))
                    elif "function" in tool:
                        # OpenAI native format
                        func = tool["function"]
                        tool_defs.append(ToolDefinition.from_external(
                            name=func["name"],
                            description=func.get("description"),
                            parameters=func.get("parameters", {}).get("properties", {}),
                            source=self._tool_source_label(tool),
                        ))
                    else:
                        # A dict matching neither shape used to fall through the
                        # if/elif with no signal at all — not even a log line.
                        self._warn_tool_dropped(
                            tool,
                            "dict tool needs either 'name' + 'description' or an OpenAI-style "
                            f"'function' entry; got keys {sorted(tool.keys())[:8]}",
                        )
                else:
                    self._warn_tool_dropped(tool, f"unsupported tool format {type(tool).__name__}")
            except Exception as e:
                self._warn_tool_dropped(tool, str(e))

        return tool_defs

    def _parse_native_response(self, response) -> ToolCallResponse:
        """Parse native API response format."""
        # Handle None response
        if response is None:
            return ToolCallResponse(content="", tool_calls=[], raw_response=None)

        # Handle different response types
        if hasattr(response, 'content'):
            # GenerateResponse object
            content = response.content
            tool_calls = []

            # Check if response has tool_calls attribute
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tc in response.tool_calls:
                    tool_call = ToolCall(
                        name=getattr(tc, 'name', '') or getattr(tc, 'function', {}).get('name', ''),
                        arguments=getattr(tc, 'arguments', {}) or getattr(tc, 'function', {}).get('arguments', {}),
                        call_id=getattr(tc, 'id', None)
                    )
                    # Handle string arguments (need to parse JSON)
                    if isinstance(tool_call.arguments, str):
                        try:
                            tool_call.arguments = json.loads(tool_call.arguments)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool arguments: {tool_call.arguments}")
                            tool_call.arguments = {}

                    from .arg_canonicalizer import canonicalize_tool_arguments

                    tool_call.arguments = canonicalize_tool_arguments(tool_call.name, tool_call.arguments)
                    tool_calls.append(tool_call)

            return ToolCallResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=response
            )
        else:
            # Dictionary response
            content = response.get("content", "")
            tool_calls = []

            # Extract tool calls based on provider format
            if "tool_calls" in response:
                for tc in response["tool_calls"]:
                    tool_call = ToolCall(
                        name=tc.get("name") or tc.get("function", {}).get("name"),
                        arguments=tc.get("arguments") or tc.get("function", {}).get("arguments", {}),
                        call_id=tc.get("id")
                    )
                    # Handle string arguments (need to parse JSON)
                    if isinstance(tool_call.arguments, str):
                        try:
                            tool_call.arguments = json.loads(tool_call.arguments)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse tool arguments: {tool_call.arguments}")
                            tool_call.arguments = {}

                    from .arg_canonicalizer import canonicalize_tool_arguments

                    tool_call.arguments = canonicalize_tool_arguments(tool_call.name, tool_call.arguments)
                    tool_calls.append(tool_call)

            return ToolCallResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=response
            )

    def _parse_prompted_response(self, response: str) -> ToolCallResponse:
        """Parse prompted response format."""
        # Use architecture-specific parsing
        tool_calls = parse_tool_calls(response, self.model_name)

        # Extract content (everything that's not a tool call) using shared cleaning function
        from .parser import clean_tool_syntax
        content = clean_tool_syntax(response, tool_calls)

        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=response
        )


def create_handler(model_name: str) -> UniversalToolHandler:
    """
    Create a tool handler for a specific model.

    Args:
        model_name: Model identifier

    Returns:
        Configured UniversalToolHandler
    """
    return UniversalToolHandler(model_name)

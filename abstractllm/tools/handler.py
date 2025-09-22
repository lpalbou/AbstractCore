"""
Universal tool handler for all models and providers.

This module provides a utility class for tool support that works
across all models, whether they have native tool APIs or require prompting.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable

from ..architectures import detect_architecture, get_model_capabilities, get_architecture_format
from .core import ToolDefinition, ToolCall, ToolCallResponse, ToolResult
from .parser import detect_tool_calls, parse_tool_calls, format_tool_prompt

logger = logging.getLogger(__name__)


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
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
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
        return format_tool_prompt(tool_defs, self.model_name)

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

            # Convert to OpenAI-style function format (most common)
            native_tool = {
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": cleaned_properties,
                        "required": required_fields
                    }
                }
            }
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

    def _convert_to_tool_definitions(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
    ) -> List[ToolDefinition]:
        """Convert various tool formats to ToolDefinition objects."""
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
                        tool_defs.append(ToolDefinition.from_function(tool))
                elif isinstance(tool, dict):
                    if "name" in tool and "description" in tool:
                        # Direct dict format - extract properties from full schema
                        parameters = tool.get("parameters", {})
                        # If parameters is a full JSON schema, extract just the properties
                        if isinstance(parameters, dict) and "properties" in parameters:
                            properties = parameters["properties"]
                        else:
                            properties = parameters

                        tool_defs.append(ToolDefinition(
                            name=tool["name"],
                            description=tool["description"],
                            parameters=properties,
                            tags=tool.get("tags", []),
                            when_to_use=tool.get("when_to_use"),
                            examples=tool.get("examples", [])
                        ))
                    elif "function" in tool:
                        # OpenAI native format
                        func = tool["function"]
                        tool_defs.append(ToolDefinition(
                            name=func["name"],
                            description=func["description"],
                            parameters=func.get("parameters", {}).get("properties", {})
                        ))
                else:
                    logger.warning(f"Skipping unsupported tool format: {type(tool)}")
            except Exception as e:
                logger.warning(f"Failed to convert tool {tool}: {e}")

        return tool_defs

    def _parse_native_response(self, response: Dict[str, Any]) -> ToolCallResponse:
        """Parse native API response format."""
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

        # Extract content (everything that's not a tool call)
        content = response
        if tool_calls:
            # Try to remove tool call syntax from content
            # This is a simple approach - could be enhanced
            import re
            patterns = [
                r'<\|tool_call\|>.*?</?\|tool_call\|>',
                r'<function_call>.*?</function_call>',
                r'<tool_call>.*?</tool_call>',
                r'```tool_code.*?```',
                r'```tool_call.*?```'
            ]
            for pattern in patterns:
                content = re.sub(pattern, '', content, flags=re.DOTALL)
            content = content.strip()

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
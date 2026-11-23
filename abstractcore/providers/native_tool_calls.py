"""Native tool calls: assemble streamed deltas into complete calls, and report unfinished ones.

An OpenAI-compatible stream splits one tool call over many chunks: the first delta carries
`index`, `id` and `function.name`; the following deltas carry only `index` and a fragment of
the `function.arguments` JSON string. A consumer needs the joined call, so a provider feeds
every chunk's `delta.tool_calls` to a `StreamedToolCallAssembler` and emits the complete calls
once, when the stream reports its finish reason (or ends).

Complete calls use the flat shape native streams deliver on chunks:

    {"id": "...", "type": "function", "name": "...", "arguments": "<JSON string>"}

A call whose arguments never form a JSON object (the model stopped mid-call, for example at its
output limit) is not a tool call: it is reported with `incomplete_tool_call_metadata` so the
caller sees it, and nothing can execute it with missing arguments (ADR 0001).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.jsonish import loads_dict_like
from ..utils.truncation import preview_text

logger = logging.getLogger(__name__)


def _field(obj: Any, key: str) -> Any:
    """Read `key` from a dict or an SDK object (OpenAI SDK deltas are objects)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _as_index(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


class StreamedToolCallAssembler:
    """Joins tool-call deltas by `index`, falling back to `id`, then to the latest call."""

    def __init__(self) -> None:
        self._calls: List[Dict[str, Any]] = []
        self._by_index: Dict[int, Dict[str, Any]] = {}

    def __bool__(self) -> bool:
        return bool(self._calls)

    def feed(self, deltas: Any) -> None:
        """Add one chunk's `tool_calls` list (dicts or SDK objects). Other values are ignored."""
        if not isinstance(deltas, (list, tuple)):
            return
        for delta in deltas:
            if delta is not None:
                self._feed_one(delta)

    def _feed_one(self, delta: Any) -> None:
        function = _field(delta, "function")
        name = _field(function, "name") or _field(delta, "name")
        name = name if isinstance(name, str) and name else None
        arguments = _field(function, "arguments")
        if arguments is None:
            arguments = _field(delta, "arguments")
        call_id = _field(delta, "id") or _field(delta, "call_id")
        call_id = str(call_id) if call_id else None
        index = _as_index(_field(delta, "index"))

        call = self._locate(index=index, call_id=call_id, named=name is not None)
        # A named delta under another id announces the next call (a server may reuse an index).
        if call is not None and name and call_id and call["id"] and call_id != call["id"]:
            call = None
        if call is None:
            call = {"id": None, "type": "function", "name": None, "arguments": ""}
            self._calls.append(call)
            if index is not None:
                self._by_index[index] = call

        if call_id and not call["id"]:
            call["id"] = call_id
        call_type = _field(delta, "type")
        if isinstance(call_type, str) and call_type:
            call["type"] = call_type

        restated = False
        if name:
            if not call["name"]:
                call["name"] = name
            elif name == call["name"]:
                restated = True  # the whole call sent again, not a fragment
            elif not call_id:
                call["name"] += name  # a name split across deltas
        self._merge_arguments(call, arguments, restated=restated)

    def _locate(self, *, index: Optional[int], call_id: Optional[str], named: bool) -> Optional[Dict[str, Any]]:
        if index is not None:
            return self._by_index.get(index)
        if call_id:
            for call in self._calls:
                if call["id"] == call_id:
                    return call
        # A nameless delta with no usable address continues the latest call.
        if not named and self._calls:
            return self._calls[-1]
        return None

    @staticmethod
    def _merge_arguments(call: Dict[str, Any], arguments: Any, *, restated: bool) -> None:
        if isinstance(arguments, dict):
            call["arguments"] = arguments
            return
        if not isinstance(arguments, str) or not arguments:
            return
        current = call["arguments"]
        # A restated call carries the arguments so far, not a fragment: keep the latest version.
        if not isinstance(current, str) or (restated and arguments.startswith(current)):
            call["arguments"] = arguments
        else:
            call["arguments"] = current + arguments

    def drain(self) -> Tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Return `(complete calls, incomplete calls)` and reset.

        Complete calls is None when there is none. A call is incomplete when its arguments are
        not a JSON object; a call without arguments is complete, with `"{}"`."""
        complete: List[Dict[str, Any]] = []
        incomplete: List[Dict[str, Any]] = []
        for call in self._calls:
            if not call["name"]:
                continue
            arguments = call["arguments"]
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            arguments = arguments if arguments.strip() else "{}"
            flat = {"id": call["id"], "type": call["type"], "name": call["name"], "arguments": arguments}
            (complete if is_complete_arguments(arguments) else incomplete).append(flat)
        self._calls = []
        self._by_index = {}
        return complete or None, incomplete


def is_complete_arguments(arguments: Any) -> bool:
    """True when a call's arguments are a JSON object, or absent (a call without arguments)."""
    if arguments is None or isinstance(arguments, dict):
        return True
    if not isinstance(arguments, str):
        return False
    return arguments.strip() in ("", "null") or isinstance(loads_dict_like(arguments), dict)


def split_incomplete_tool_calls(tool_calls: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    """Separate the native calls of a response into `(complete, incomplete)`.

    Complete calls are returned as received. Incomplete ones (arguments that are not a JSON
    object) come back in the flat shape, for `incomplete_tool_call_metadata`."""
    if not isinstance(tool_calls, list):
        return tool_calls, []
    complete: List[Any] = []
    incomplete: List[Dict[str, Any]] = []
    for call in tool_calls:
        function = _field(call, "function")
        arguments = _field(function, "arguments")
        if arguments is None:
            arguments = _field(call, "arguments")
        if is_complete_arguments(arguments):
            complete.append(call)
            continue
        incomplete.append(
            {
                "id": _field(call, "id") or _field(call, "call_id"),
                "type": _field(call, "type") or "function",
                "name": _field(function, "name") or _field(call, "name"),
                "arguments": arguments,
            }
        )
    return complete, incomplete

def incomplete_tool_call_metadata(
    metadata: Optional[Dict[str, Any]],
    incomplete: Sequence[Dict[str, Any]],
    *,
    finish_reason: Optional[str],
    available_tools: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """Report native calls that did not complete, the way an unparsed text call is reported.

    Sets `unparsed_tool_call` (`format: "native"`; `reason: "unclosed"` when the output stopped
    at its limit or without a finish reason, else `"unparseable"`), `tool_call_error` and a
    warning. Returns `metadata` unchanged when every call completed."""
    if not incomplete:
        return metadata
    first = incomplete[0]
    reason = "unclosed" if finish_reason in (None, "length") else "unparseable"
    meta = dict(metadata or {})
    meta["unparsed_tool_call"] = {
        "format": "native",
        "text": f"{first.get('name')} {first.get('arguments')}",
        "reason": reason,
    }
    meta["tool_call_error"] = {"code": "invalid_tool_syntax", "available_tools": sorted(available_tools)}
    warning = (
        f"Incomplete tool call (native, {reason}): the arguments of '{first.get('name')}' are not a JSON "
        f"object; it is reported in metadata['unparsed_tool_call'], not as a tool call: "
        f"{preview_text(str(first.get('arguments') or ''), max_chars=200)!r}"
    )
    meta["warnings"] = [*(meta.get("warnings") or []), warning]
    logger.warning(warning)
    return meta

def declared_tool_names(tools: Any) -> List[str]:
    """Names of the tools declared in a request (`tools` in OpenAI or flat form)."""
    names: List[str] = []
    for tool in tools if isinstance(tools, (list, tuple)) else ():
        name = _field(_field(tool, "function"), "name") or _field(tool, "name")
        if isinstance(name, str) and name:
            names.append(name)
    return names

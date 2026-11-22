"""Assemble streamed OpenAI-style tool-call deltas into complete tool calls.

An OpenAI-compatible stream splits one tool call over many chunks: the first delta carries
`index`, `id` and `function.name`; the following deltas carry only `index` and a fragment of
the `function.arguments` JSON string. A consumer needs the joined call, so a provider feeds
every chunk's `delta.tool_calls` to a `StreamedToolCallAssembler` and emits the complete calls
once, when the stream reports its finish reason (or ends).

Complete calls use the flat shape the rest of the stack already accepts on stream chunks:

    {"id": "...", "type": "function", "name": "...", "arguments": "<JSON string>"}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _field(obj: Any, key: str) -> Any:
    """Read `key` from a dict or an SDK object (OpenAI SDK deltas are objects)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class StreamedToolCallAssembler:
    """Joins tool-call deltas by `index` (falling back to `id`, then to the latest call)."""

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
        name = _field(function, "name") if function is not None else None
        arguments = _field(function, "arguments") if function is not None else None
        if name is None:
            name = _field(delta, "name")
        if arguments is None:
            arguments = _field(delta, "arguments")
        call_id = _field(delta, "id") or _field(delta, "call_id")
        index = _field(delta, "index")
        if isinstance(index, bool) or not isinstance(index, int):
            index = None

        call = self._find(index=index, call_id=call_id, has_name=bool(name))
        if call is None:
            call = {"id": None, "type": "function", "name": None, "arguments": ""}
            self._calls.append(call)
            if index is not None:
                self._by_index[index] = call

        if call_id and not call["id"]:
            call["id"] = str(call_id)
        call_type = _field(delta, "type")
        if isinstance(call_type, str) and call_type:
            call["type"] = call_type
        if isinstance(name, str) and name and not call["name"]:
            call["name"] = name
        self._merge_arguments(call, arguments, restated=bool(call_id and name))

    def _find(self, *, index: Optional[int], call_id: Any, has_name: bool) -> Optional[Dict[str, Any]]:
        if index is not None:
            call = self._by_index.get(index)
            # A server that reuses an index for the next call announces it with a new id.
            if call is not None and call_id and call["id"] and str(call_id) != call["id"]:
                return None
            return call
        if call_id:
            for call in self._calls:
                if call["id"] == str(call_id):
                    return call
            return None
        if has_name:
            return None
        return self._calls[-1] if self._calls else None

    @staticmethod
    def _merge_arguments(call: Dict[str, Any], arguments: Any, *, restated: bool) -> None:
        if arguments is None:
            return
        if isinstance(arguments, dict):
            call["arguments"] = arguments
            return
        if not isinstance(arguments, str) or not arguments:
            return
        current = call["arguments"]
        if not isinstance(current, str):
            # A parsed object was already received; a later string restates it.
            call["arguments"] = arguments
            return
        # Servers that re-send the whole call on every chunk (id and name restated) send the
        # arguments so far, not a fragment: keep the latest snapshot instead of joining it.
        if restated and current and arguments.startswith(current):
            call["arguments"] = arguments
            return
        call["arguments"] = current + arguments

    def drain(self) -> Optional[List[Dict[str, Any]]]:
        """Return the complete calls and reset. None when no call was received."""
        calls: List[Dict[str, Any]] = []
        for call in self._calls:
            arguments = call["arguments"]
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not arguments:
                if not call["name"]:
                    continue
                arguments = "{}"
            calls.append({"id": call["id"], "type": call["type"], "name": call["name"], "arguments": arguments})
        self._calls = []
        self._by_index = {}
        return calls or None

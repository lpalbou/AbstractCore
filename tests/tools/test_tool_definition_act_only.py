"""act_only: first-class host-side policy attribute on ToolDefinition.

Entity-topology item 7 / G1 ("diary words never rest outside the book"): tools like
diary_read are declared ACT-ONLY on the tool contract — hosts (runtime effect
handlers, agent observe nodes, ledger writers) key their durable channels on this
flag, persisting only the act-frame (id + reason + gist), never the returned words.

Core carries the DECLARATION, not the enforcement. The contract pinned here
(ruled on a2a thread 0013, 20260710T020120Z-core-01):
- first-class typed field, NOT a tags entry — the dataclass default IS the
  fail-closed policy (undeclared = normal durable tool; a typo'd tag would fail
  OPEN, the wrong direction for a privacy attribute);
- additive serialization — to_dict() emits the key only when set, so existing
  consumers of normal tools see no new key;
- dict -> ToolDefinition round-trip preserves it (handler conversion);
- the flag NEVER reaches native provider payloads (strict servers reject unknown
  fields; enforcement is host-side anyway) — model-facing guidance belongs in the
  tool's description text.
"""

from __future__ import annotations

from abstractcore.tools.core import ToolDefinition, tool
from abstractcore.tools.handler import UniversalToolHandler


def _minimal(**kwargs) -> ToolDefinition:
    return ToolDefinition(
        name="diary_read",
        description="Read one diary entry by id.",
        parameters={"entry_id": {"type": "string"}},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fail-closed default + declaration
# ---------------------------------------------------------------------------

def test_default_is_false_and_not_serialized():
    td = _minimal()
    assert td.act_only is False
    # Additive contract: normal tools serialize exactly as before — no new key.
    assert "act_only" not in td.to_dict()


def test_declared_true_is_typed_and_serialized():
    td = _minimal(act_only=True)
    assert td.act_only is True
    assert td.to_dict()["act_only"] is True


def test_post_init_coerces_to_bool():
    # A truthy string must normalize to a real bool (consumers do identity-ish
    # checks and serialize to JSON; "false" being truthy is the arg-coercion
    # lesson applied at the declaration site).
    td = _minimal(act_only="yes")  # type: ignore[arg-type]
    assert td.act_only is True and isinstance(td.act_only, bool)


def test_from_function_defaults_false():
    def read_file(path: str) -> str:
        """Read a file."""
        return path

    assert ToolDefinition.from_function(read_file).act_only is False


def test_tool_decorator_declares_act_only():
    @tool(act_only=True)
    def diary_read(entry_id: str) -> str:
        """Read one diary entry by id."""
        return entry_id

    assert diary_read._tool_definition.act_only is True
    assert diary_read._tool_definition.to_dict()["act_only"] is True


def test_tool_decorator_default_stays_false():
    @tool
    def web_search(query: str) -> str:
        """Search the web."""
        return query

    assert web_search._tool_definition.act_only is False


# ---------------------------------------------------------------------------
# Dict round-trip (the shape hosts pass through generate()/LLM_CALL payloads)
# ---------------------------------------------------------------------------

def test_dict_round_trip_preserves_act_only():
    handler = UniversalToolHandler("gpt-oss-120b")
    spec = _minimal(act_only=True).to_dict()

    defs = handler._convert_to_tool_definitions([spec])
    assert len(defs) == 1
    assert defs[0].act_only is True

    # And absence stays fail-closed through the same path.
    plain = _minimal().to_dict()
    assert handler._convert_to_tool_definitions([plain])[0].act_only is False


# ---------------------------------------------------------------------------
# Wire boundary: the flag never reaches native provider payloads
# ---------------------------------------------------------------------------

def test_native_payload_excludes_act_only():
    handler = UniversalToolHandler("gpt-oss-120b")  # native-tools model
    assert handler.supports_native

    native = handler.prepare_tools_for_native([_minimal(act_only=True)])
    assert len(native) == 1
    payload = native[0]
    assert "act_only" not in payload
    assert "act_only" not in payload.get("function", {})
    # The declared schema fields are all that cross the wire.
    assert set(payload["function"].keys()) == {"name", "description", "parameters"}

"""Definition-site side-effect tags (abstractagent ask, 2026-07-14).

Consumers classify side-effect tools by ``ToolDefinition.tags`` (abstractagent's
repeat-guard reads {"mcp", "side_effect", "mutating", "write"}) — curated name
lists rot, tags at the definition site don't. These pins hold the three
truths the lane depends on:

1. TAGS MIRROR THE INVENTORY: every builtin the inventory classifies
   ``mutating=True`` carries the "mutating" tag, and ``fetch_url`` (the one
   ``remote_write_capable`` tool) carries "write" — one fact, two surfaces,
   test-pinned so they cannot drift apart.
2. READ-ONLY TOOLS STAY UNTAGGED: a side-effect-classed tag on a read lane
   would make deny-safe consumers skip legitimate repeat reads; truthfulness
   of the tag vocabulary is part of the contract.
3. MCP ORIGIN RIDES TAGS END-TO-END: MCP tool specs are born with the "mcp"
   tag and the handler's dict->ToolDefinition conversion preserves it, so
   origin-aware classification works without name heuristics.
"""

from abstractcore.tools.core import ToolDefinition
from abstractcore.tools.handler import UniversalToolHandler
from abstractcore.tools.inventory import list_builtin_tool_inventory

# The tag consumers key side-effect guards on (abstractagent
# generation_params.SIDE_EFFECT_TAGS) — mirrored here as the wire vocabulary
# core commits to emitting; renaming these strings is a cross-repo change.
SIDE_EFFECT_TAGS = {"mcp", "side_effect", "mutating", "write"}


def _definition_tags_by_name():
    import importlib

    out = {}
    # Scan every module the inventory scans (schema v3 added comms/telegram),
    # so the cross-surface tag↔fact guard covers the full inventory rather
    # than KeyError-ing on a tool the inventory now carries.
    from abstractcore.tools.inventory import _INVENTORY_MODULES

    for module_path in _INVENTORY_MODULES:
        module = importlib.import_module(module_path)
        for attr_name in dir(module):
            tool_def = getattr(getattr(module, attr_name, None), "_tool_definition", None)
            if isinstance(tool_def, ToolDefinition):
                out[tool_def.name] = set(tool_def.tags or [])
    return out


def test_mutating_inventory_tools_carry_the_mutating_tag():
    tags_by_name = _definition_tags_by_name()
    for descriptor in list_builtin_tool_inventory():
        tags = tags_by_name[descriptor.name]
        if descriptor.mutating:
            assert "mutating" in tags, (
                f"{descriptor.name}: inventory says mutating=True but the @tool "
                "definition site carries no 'mutating' tag — the two surfaces drifted"
            )
        else:
            assert "mutating" not in tags, (
                f"{descriptor.name}: tagged 'mutating' but the inventory classifies "
                "it non-mutating — decide the fact once, in both places"
            )


def test_fetch_url_carries_the_write_tag_for_remote_write_capability():
    """fetch_url is not mutating (local state untouched) but its
    model-controlled method/data can POST/PUT/DELETE remotely — consumers
    keying on the 'write' tag must see it (2026-07-12 finding)."""
    tags_by_name = _definition_tags_by_name()
    for descriptor in list_builtin_tool_inventory():
        tags = tags_by_name[descriptor.name]
        if descriptor.remote_write_capable:
            assert "write" in tags, (
                f"{descriptor.name}: remote_write_capable in the inventory but no "
                "'write' tag at the definition site"
            )
        elif not descriptor.mutating:
            assert not (tags & SIDE_EFFECT_TAGS), (
                f"{descriptor.name}: read-only tool carries side-effect tag(s) "
                f"{sorted(tags & SIDE_EFFECT_TAGS)} — over-tagging makes deny-safe "
                "consumers skip legitimate repeat reads"
            )


def test_mcp_tool_specs_are_born_with_the_mcp_tag():
    from abstractcore.mcp.tool_source import McpServerInfo, mcp_tool_to_abstractcore_tool_spec

    spec = mcp_tool_to_abstractcore_tool_spec(
        {
            "name": "create_ticket",
            "description": "Create a ticket in the tracker.",
            "inputSchema": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
        server=McpServerInfo(server_id="tracker", url="http://127.0.0.1:9000/mcp"),
    )
    assert "mcp" in spec["tags"]
    assert "mcp_server:tracker" in spec["tags"]
    assert spec["name"].startswith("mcp::tracker::")


def test_handler_dict_conversion_preserves_tags_into_tool_definitions():
    """The dict->ToolDefinition path (how MCP specs and other dict tools enter
    the handler) must not drop tags — origin-aware classification reads them
    off the resulting ToolDefinition."""
    handler = UniversalToolHandler("gpt-4o-mini")
    tool_defs = handler._convert_to_tool_definitions(
        [
            {
                "name": "mcp::tracker::create_ticket",
                "description": "Create a ticket in the tracker.",
                "parameters": {"title": {"type": "string"}},
                "tags": ["mcp", "mcp_server:tracker"],
            }
        ]
    )
    assert len(tool_defs) == 1
    assert tool_defs[0].tags == ["mcp", "mcp_server:tracker"]


def test_tags_ride_to_dict_but_never_native_payloads():
    """tags serialize in ToolDefinition.to_dict() (metadata surfaces) but stay
    OUT of native wire payloads — strict provider schemas reject unknown keys."""
    from abstractcore.tools.common_tools import write_file

    tool_def = write_file._tool_definition
    assert "mutating" in tool_def.to_dict()["tags"]

    handler = UniversalToolHandler("gpt-4o-mini")
    native = handler.prepare_tools_for_native([tool_def])
    assert native, "native formatting refused a builtin tool"
    assert "tags" not in native[0]
    assert "tags" not in native[0]["function"]

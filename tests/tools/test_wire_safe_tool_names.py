"""Wire-safe tool names for strict native endpoints (abstractagent find, 2026-07-13).

MCP tools are namespaced ``mcp::server::tool``; native declarations put that
name on the wire verbatim and strict endpoints (OpenAI name contract
``^[a-zA-Z0-9_-]{1,64}$``, Anthropic equivalent) 400 the WHOLE call. These
tests pin the two halves of the fix: deterministic aliasing at the
declaration boundary, and alias→original resolution at the response
normalization choke point.
"""

from abstractcore.mcp.naming import namespaced_tool_name
from abstractcore.tools.handler import UniversalToolHandler
from abstractcore.tools.wire_naming import (
    build_wire_name_map,
    is_wire_safe_tool_name,
    resolve_wire_tool_name,
    wire_safe_tool_history,
    wire_safe_tool_name,
)


class TestWireSafeAlias:
    def test_history_names_are_encoded_without_mutating_runtime_state(self):
        original = "mcp::agora::whoami"
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": original, "arguments": "{}"}}],
            },
            {"role": "tool", "name": original, "tool_call_id": "c1", "content": "ok"},
        ]

        safe = wire_safe_tool_history(messages)
        alias = wire_safe_tool_name(original)

        assert safe[0]["tool_calls"][0]["function"]["name"] == alias
        assert safe[1]["name"] == alias
        assert messages[0]["tool_calls"][0]["function"]["name"] == original
        assert messages[1]["name"] == original

    def test_safe_names_pass_through_byte_identical(self):
        for name in ("fetch_url", "list_files", "execute_command", "web-search", "a" * 64):
            assert wire_safe_tool_name(name) == name

    def test_mcp_namespaced_name_becomes_wire_safe_and_deterministic(self):
        original = namespaced_tool_name(server_id="github", tool_name="create_issue")
        assert original == "mcp::github::create_issue"
        wire = wire_safe_tool_name(original)
        assert is_wire_safe_tool_name(wire)
        assert wire == wire_safe_tool_name(original)  # deterministic
        assert "mcp_github_create_issue" in wire

    def test_distinct_originals_never_collide_on_the_wire(self):
        # A literal tool named like the sanitized form must stay distinct
        # from the sanitized MCP name (the hash tail carries the difference).
        a = wire_safe_tool_name("mcp::a::b")
        b = wire_safe_tool_name("mcp__a__b")  # already wire-safe: unchanged
        assert b == "mcp__a__b"
        assert a != b

    def test_long_names_fit_the_64_char_contract(self):
        original = namespaced_tool_name(server_id="s" * 40, tool_name="t" * 40)
        wire = wire_safe_tool_name(original)
        assert is_wire_safe_tool_name(wire)
        assert len(wire) <= 64

    def test_resolve_maps_alias_back_to_original(self):
        original = "mcp::files::read_file"
        allowed = {"fetch_url", original}
        wire = wire_safe_tool_name(original)
        assert resolve_wire_tool_name(wire, allowed) == original
        # Untouched names resolve to themselves; unknown names to None.
        assert resolve_wire_tool_name("fetch_url", allowed) == "fetch_url"
        assert resolve_wire_tool_name("never_declared", allowed) is None

    def test_build_wire_name_map_covers_batch(self):
        names = ["fetch_url", "mcp::a::b"]
        m = build_wire_name_map(names)
        assert m["fetch_url"] == "fetch_url"
        assert m["mcp::a::b"] == wire_safe_tool_name("mcp::a::b")


class TestNativeDeclarationAliasing:
    def test_prepare_tools_for_native_declares_the_wire_alias(self):
        handler = UniversalToolHandler("gpt-4o")  # native-capable
        assert handler.supports_native
        tools = [{
            "name": "mcp::github::create_issue",
            "description": "Create an issue",
            "parameters": {"title": {"type": "string"}},
        }]
        native = handler.prepare_tools_for_native(tools)
        assert len(native) == 1
        declared = native[0]["function"]["name"]
        assert is_wire_safe_tool_name(declared)
        assert declared == wire_safe_tool_name("mcp::github::create_issue")

    def test_safe_builtin_declarations_are_unchanged(self):
        handler = UniversalToolHandler("gpt-4o")
        tools = [{"name": "fetch_url", "description": "Fetch", "parameters": {}}]
        native = handler.prepare_tools_for_native(tools)
        assert native[0]["function"]["name"] == "fetch_url"


class TestResponseNormalizationReverseMap:
    def _provider(self):
        from abstractcore.providers.ollama_provider import OllamaProvider

        return OllamaProvider(model="test-model")

    def test_wire_alias_in_response_resolves_to_original_name(self):
        provider = self._provider()
        original = "mcp::github::create_issue"
        wire = wire_safe_tool_name(original)
        normalized = provider._normalize_tool_calls_payload(
            [{"name": wire, "arguments": {"title": "bug"}, "call_id": "c1"}],
            allowed_tool_names={original, "fetch_url"},
        )
        assert normalized is not None and len(normalized) == 1
        assert normalized[0]["name"] == original
        assert normalized[0]["arguments"] == {"title": "bug"}

    def test_unknown_names_are_still_dropped(self):
        provider = self._provider()
        normalized = provider._normalize_tool_calls_payload(
            [{"name": "totally_unknown", "arguments": {}}],
            allowed_tool_names={"mcp::github::create_issue"},
        )
        assert normalized is None

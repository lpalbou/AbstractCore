"""Pins for the authoritative builtin-tool inventory (derive-never-copy source).

The inventory is the registry side downstream seats derive from (runtime's
grant tool universe, the gateway's door declarations, observer's descriptor
contract). These tests pin its load-bearing properties:

- BYTE-STABLE MEMBER SETS: the exact 13 common + 3 shell names, in
  deterministic order — a widening or rename lands RED here first, so the
  cross-repo consumers are notified by a failing pin instead of drifting.
- PROGRAMMATIC DERIVATION: the inventory matches a live scan of the
  @tool-decorated functions — a hand-list desync is structurally impossible.
- FAIL-CLOSED CLASSIFICATION, BOTH WAYS: a tool missing an explicit
  classification refuses the whole inventory, and a stale classification
  naming no scanned tool ALSO refuses (a lingering entry would silently
  pre-classify a future tool reusing the name).
- FACT BOUNDARIES: descriptors carry only core-owned facts (no tier, no
  containment — other seats' vocabularies must not get a second copy here).
- SCHEMA ISOLATION: served rows never alias the live ToolDefinition
  parameter dicts (a consumer mutation must not rewrite the process-wide
  tool schema providers serialize for native calls).
"""

import pytest

from abstractcore.tools.inventory import (
    BuiltinToolDescriptor,
    INVENTORY_SCHEMA_VERSION,
    _CLASSIFICATION_BY_NAME,
    _INVENTORY_MODULES,
    _scan_module_tool_definitions,
    builtin_tool_inventory_as_dicts,
    list_builtin_tool_inventory,
    list_builtin_tool_names,
)


# The 13 common + 3 shell enumeration recorded in the entity-agency plan's
# Phase 0.5 (core's authoritative count, cross-verified on the hub record).
EXPECTED_COMMON = [
    "analyze_code",
    "edit_file",
    "execute_command",
    "fetch_url",
    "list_files",
    "read_file",
    "search_files",
    "skim_files",
    "skim_folders",
    "skim_url",
    "skim_websearch",
    "web_search",
    "write_file",
]

EXPECTED_SHELL = ["shell_close", "shell_exec", "shell_write_stdin"]

EXPECTED_MUTATING = {
    "edit_file",
    "execute_command",
    "write_file",
    "shell_exec",
    "shell_write_stdin",
    "shell_close",
}

# fetch_url can send POST/PUT/DELETE with a body through its model-controlled
# method/data parameters — the one remote-write escape hatch in the set.
EXPECTED_REMOTE_WRITE_CAPABLE = {"fetch_url"}


def test_member_sets_are_byte_stable():
    """The exact ruled enumeration: 13 common + 3 shell, deterministic order."""
    names = list_builtin_tool_names()
    assert names == EXPECTED_COMMON + EXPECTED_SHELL
    assert len(names) == 16


def test_inventory_is_derived_not_copied():
    """The inventory must match a live scan of @tool definitions — adding a
    @tool to a scanned module MUST surface it (or refuse on classification),
    never silently diverge."""
    import importlib

    from abstractcore.tools.core import ToolDefinition

    scanned = set()
    for module_path in _INVENTORY_MODULES:
        module = importlib.import_module(module_path)
        for attr_name in dir(module):
            tool_def = getattr(getattr(module, attr_name, None), "_tool_definition", None)
            if isinstance(tool_def, ToolDefinition):
                scanned.add(tool_def.name)

    assert scanned == set(list_builtin_tool_names())


def test_mutating_classification():
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    for name, descriptor in inventory.items():
        assert descriptor.mutating is (name in EXPECTED_MUTATING), (
            f"{name}: mutating classification drifted"
        )


def test_remote_write_capable_classification():
    """The second fact decided at classification time: fetch_url's
    model-controlled method/data can write remotely; GET-shaped lanes can't.
    A derive-side auto-approving on mutating=False alone must consult this."""
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    for name, descriptor in inventory.items():
        assert descriptor.remote_write_capable is (name in EXPECTED_REMOTE_WRITE_CAPABLE), (
            f"{name}: remote_write_capable classification drifted"
        )


def test_unclassified_tool_refuses_loudly(monkeypatch):
    """Fail-closed: a tool without an explicit classification entry refuses
    the inventory naming the tool — never a silent default."""
    import abstractcore.tools.inventory as inv

    pruned = {k: v for k, v in _CLASSIFICATION_BY_NAME.items() if k != "write_file"}
    monkeypatch.setattr(inv, "_CLASSIFICATION_BY_NAME", pruned)

    with pytest.raises(RuntimeError) as exc_info:
        inv.list_builtin_tool_inventory()
    assert "write_file" in str(exc_info.value)
    assert "classification" in str(exc_info.value)


def test_stale_classification_refuses_loudly(monkeypatch):
    """Fail-closed the OTHER way (adversarial-review finding): an entry
    naming no scanned tool refuses — a lingering classification would
    silently pre-classify a future tool reusing the name."""
    import abstractcore.tools.inventory as inv

    widened = dict(_CLASSIFICATION_BY_NAME)
    widened["ghost_tool_that_never_existed"] = {
        "mutating": False,
        "remote_write_capable": False,
    }
    monkeypatch.setattr(inv, "_CLASSIFICATION_BY_NAME", widened)

    with pytest.raises(RuntimeError) as exc_info:
        inv.list_builtin_tool_inventory()
    assert "ghost_tool_that_never_existed" in str(exc_info.value)
    assert "stale" in str(exc_info.value).lower()


def test_classification_map_matches_scan_exactly():
    """Set equality both directions — the two refusal paths above enforce it
    at runtime; this pins it at test time."""
    assert set(_CLASSIFICATION_BY_NAME) == set(list_builtin_tool_names())


def test_descriptors_carry_only_core_owned_facts():
    """No tier, no containment, no approval vocabulary — those are other
    seats' classifications and must not get a second copy here."""
    rows = builtin_tool_inventory_as_dicts()
    expected_keys = {
        "name", "owner", "module", "mutating", "remote_write_capable",
        "act_only", "description", "parameters",
    }
    for row in rows:
        assert set(row.keys()) == expected_keys, f"{row['name']}: field drift"
        assert row["owner"] == "core"
        assert row["module"] in {"common_tools", "shell_tools"}
        assert isinstance(row["mutating"], bool)
        assert isinstance(row["remote_write_capable"], bool)
        assert isinstance(row["act_only"], bool)
        assert row["description"].strip(), f"{row['name']}: empty description"
    assert INVENTORY_SCHEMA_VERSION == 1


def test_act_only_reflects_tool_definitions():
    """act_only rides from ToolDefinition (the wire-boundary fact) — all
    builtin tools today declare False; a flip must be deliberate and lands
    here."""
    for descriptor in list_builtin_tool_inventory():
        assert descriptor.act_only is False


def test_served_parameters_never_alias_live_tool_schema():
    """Adversarial-review P1: ToolDefinition.parameters nests per-param dicts
    that ARE the live schema providers serialize for native tool calls. A
    consumer mutating a served row must not rewrite the process-wide schema."""
    from abstractcore.tools.common_tools import read_file

    live_schema = read_file._tool_definition.parameters
    live_before = {k: dict(v) if isinstance(v, dict) else v for k, v in live_schema.items()}

    for surface in (
        {d.name: d.parameters for d in list_builtin_tool_inventory()},
        {r["name"]: r["parameters"] for r in builtin_tool_inventory_as_dicts()},
    ):
        served = surface["read_file"]
        for param_meta in served.values():
            if isinstance(param_meta, dict):
                param_meta["__consumer_scribble__"] = True
                param_meta.pop("default", None)

    live_after = {k: dict(v) if isinstance(v, dict) else v for k, v in live_schema.items()}
    assert live_after == live_before, "served rows alias the live tool schema"

    fresh = {d.name: d for d in list_builtin_tool_inventory()}["read_file"]
    for param_meta in fresh.parameters.values():
        if isinstance(param_meta, dict):
            assert "__consumer_scribble__" not in param_meta


def test_duplicate_name_within_module_refuses():
    """Adversarial-review P2: two DIFFERENT definitions claiming one name must
    refuse — a silent last-in-dir()-order win would serve the wrong
    ToolDefinition with every member-set pin green. Re-exports of the SAME
    function stay legal (dedupe by identity)."""
    import sys
    import types

    from abstractcore.tools.core import tool

    fixture = types.ModuleType("_inventory_fixture_dupe")

    @tool(name="fixture_tool", description="first")
    def _first(x: str) -> str:
        return x

    @tool(name="fixture_tool", description="second")
    def _second(x: str) -> str:
        return x

    fixture.first = _first
    fixture.alias_of_first = _first  # same object: legal re-export
    sys.modules["_inventory_fixture_dupe"] = fixture
    try:
        assert [t.name for t in _scan_module_tool_definitions("_inventory_fixture_dupe")] == [
            "fixture_tool"
        ]
        fixture.second = _second  # different object, same name: refuse
        with pytest.raises(RuntimeError) as exc_info:
            _scan_module_tool_definitions("_inventory_fixture_dupe")
        assert "fixture_tool" in str(exc_info.value)
    finally:
        sys.modules.pop("_inventory_fixture_dupe", None)


def test_scan_orders_by_tool_name_not_attribute_name():
    """Adversarial-review P2: ordering must key on the TOOL name (the
    @tool(name=...) override), not the function/attribute name — dir() order
    and name order coincide for every current tool, so only a fixture can pin
    the claimed mechanism."""
    import sys
    import types

    from abstractcore.tools.core import tool

    fixture = types.ModuleType("_inventory_fixture_order")

    @tool(name="zzz_last", description="attr sorts first, name sorts last")
    def _aaa(x: str) -> str:
        return x

    @tool(name="aaa_first", description="attr sorts last, name sorts first")
    def _zzz(x: str) -> str:
        return x

    fixture.a_attr = _aaa
    fixture.z_attr = _zzz
    sys.modules["_inventory_fixture_order"] = fixture
    try:
        names = [t.name for t in _scan_module_tool_definitions("_inventory_fixture_order")]
        assert names == ["aaa_first", "zzz_last"]
    finally:
        sys.modules.pop("_inventory_fixture_order", None)


def test_dict_form_round_trips_and_is_strict_json():
    """allow_nan=False (adversarial-review P2): a NaN/Infinity default would
    serialize as a non-standard token under lenient dumps and only fail in a
    strict consumer — refuse it here instead."""
    import json

    rows = builtin_tool_inventory_as_dicts()
    serialized = json.dumps(rows, sort_keys=True, allow_nan=False)
    assert json.loads(serialized) == rows


def test_package_level_exports():
    from abstractcore.tools import (
        BuiltinToolDescriptor as ExportedDescriptor,
        builtin_tool_inventory_as_dicts as exported_dicts,
        list_builtin_tool_inventory as exported_inventory,
        list_builtin_tool_names as exported_names,
    )

    assert ExportedDescriptor is BuiltinToolDescriptor
    assert exported_names() == list_builtin_tool_names()
    assert len(exported_inventory()) == 16
    assert len(exported_dicts()) == 16


def test_comms_lanes_deliberately_out_of_scope():
    """The v1 scope ruling (entity-creation directive): comms lanes are OUT —
    email/WhatsApp (comms_tools) AND Telegram (telegram_tools). This pin makes
    the exclusions decisions on record; widening is one module entry, and
    this test flips WITH it."""
    assert "abstractcore.tools.comms_tools" not in _INVENTORY_MODULES
    assert "abstractcore.tools.telegram_tools" not in _INVENTORY_MODULES
    names = set(list_builtin_tool_names())
    assert "list_emails" not in names
    assert "send_email" not in names
    assert "send_telegram_message" not in names
    assert "send_telegram_artifact" not in names

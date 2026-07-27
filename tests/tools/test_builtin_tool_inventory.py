"""Pins for the authoritative builtin-tool inventory (derive-never-copy source).

The inventory is the registry side downstream seats derive from (runtime's
grant tool universe, the gateway's door declarations, observer's descriptor
contract). These tests pin its load-bearing properties:

- BYTE-STABLE MEMBER SETS: the exact 14 common + 3 shell names, in
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


# The 14 common + 3 shell + 7 comms + 2 telegram enumeration (comms/telegram
# joined at schema v3 under the dm#221 tool-surfacing audit — the v1 exclusion
# made email/telegram invisible to every host's discovery). A widening lands
# RED here first so the cross-repo consumers (runtime grant universe, gateway
# door, observer contract) are notified by a failing pin instead of drifting
# silently.
EXPECTED_COMMON = [
    "analyze_code",
    "analyze_media",
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

EXPECTED_COMMS = [
    "list_email_accounts",
    "list_emails",
    "list_whatsapp_messages",
    "read_email",
    "read_whatsapp_message",
    "send_email",
    "send_whatsapp_message",
]

EXPECTED_TELEGRAM = ["send_telegram_artifact", "send_telegram_message"]

# browser_tools joined 2026-07-23 (operator-approved dm:core--laurent#21):
# the headless render probe. remote_write_capable: navigating executes the
# target page's JS, which can send requests anywhere (fetch_url's class).
EXPECTED_BROWSER = ["browser_probe"]

EXPECTED_MUTATING = {
    "edit_file",
    "execute_command",
    "write_file",
    "shell_exec",
    "shell_write_stdin",
    "shell_close",
}

# fetch_url can send POST/PUT/DELETE with a body through its model-controlled
# method/data parameters; the comms send_* tools emit to model-controlled
# recipients (schema v3).
EXPECTED_REMOTE_WRITE_CAPABLE = {
    "fetch_url",
    "send_email",
    "send_whatsapp_message",
    "send_telegram_message",
    "send_telegram_artifact",
    "browser_probe",
}


def test_member_sets_are_byte_stable():
    """The exact ruled enumeration: 14 common + 3 shell + 7 comms + 2
    telegram + 1 browser, deterministic (module order, then name order)."""
    names = list_builtin_tool_names()
    assert names == EXPECTED_COMMON + EXPECTED_SHELL + EXPECTED_COMMS + EXPECTED_TELEGRAM + EXPECTED_BROWSER
    assert len(names) == 27


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


def test_refiner_naming_unscanned_tool_refuses_loudly(monkeypatch):
    # Adversary P2: the refiner map is fail-closed BOTH ways — an entry
    # naming no scanned tool would attach a per-call hook to nothing.
    import abstractcore.tools.inventory as inv

    widened = dict(inv._REFINER_BY_NAME)
    widened["ghost_tool_never_scanned"] = "send_email_recipient@v1"
    monkeypatch.setattr(inv, "_REFINER_BY_NAME", widened)
    with pytest.raises(RuntimeError) as e:
        inv.list_builtin_tool_inventory()
    assert "ghost_tool_never_scanned" in str(e.value)


def test_refiner_with_unknown_id_refuses_loudly(monkeypatch):
    # Adversary P2: a typo'd refiner id on a REAL tool must refuse at build
    # (the enforcement layer would find no logic — deny-safe, but silent).
    import abstractcore.tools.inventory as inv

    widened = dict(inv._REFINER_BY_NAME)
    widened["send_email"] = "send_email_recipient"  # unversioned = unknown
    monkeypatch.setattr(inv, "_REFINER_BY_NAME", widened)
    with pytest.raises(ValueError):
        inv.list_builtin_tool_inventory()


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
        "act_only", "model_cost", "description", "parameters",
        # schema v3 (tool-tiers build): risk facts + the DERIVED risk fields.
        "comms_send", "captures_environment", "standing_effect",
        "destructive_capable", "model_controlled_destination",
        "risk_tier", "risk_rank", "risk_presentation", "risk_mapping_version",
        "risk_refiner",
    }
    for row in rows:
        assert set(row.keys()) == expected_keys, f"{row['name']}: field drift"
        assert row["owner"] == "core"
        assert row["module"] in {"common_tools", "shell_tools", "comms_tools", "telegram_tools", "browser_tools"}
        assert isinstance(row["mutating"], bool)
        assert isinstance(row["remote_write_capable"], bool)
        assert isinstance(row["act_only"], bool)
        assert isinstance(row["model_cost"], bool)
        assert row["risk_tier"] in {"observe", "act", "outreach", "destroy"}
        assert row["description"].strip(), f"{row['name']}: empty description"
    assert INVENTORY_SCHEMA_VERSION == 3


def test_descriptor_risk_defaults_are_fail_closed():
    # Adversary P2: a hand-constructed row born without an explicit
    # derivation must look UNVETTED/top, never observe/safe (the build's own
    # fail-closed rule applied to the dataclass defaults).
    from abstractcore.tools.inventory import BuiltinToolDescriptor

    d = BuiltinToolDescriptor(
        name="hypothetical",
        owner="core",
        module="common_tools",
        mutating=False,
        remote_write_capable=False,
        act_only=False,
        description="x",
    )
    assert d.risk_tier == "destroy" and d.risk_rank == 4 and d.risk_presentation == "unvetted"


def test_risk_tiers_derive_the_operator_examples():
    """The dm#221/tool-tiers acceptance shape: every band on the operator's
    own examples, derived (never hand-assigned) through the ONE mapping."""
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    assert inventory["read_file"].risk_tier == "observe"
    assert inventory["web_search"].risk_tier == "observe"
    assert inventory["write_file"].risk_tier == "act"
    assert inventory["edit_file"].risk_tier == "act"
    assert inventory["fetch_url"].risk_tier == "act"
    assert inventory["send_email"].risk_tier == "outreach"
    assert inventory["send_telegram_message"].risk_tier == "outreach"
    # The argv-class clamp: rm/git are PROGRAMS inside these tools.
    assert inventory["execute_command"].risk_tier == "destroy"
    assert inventory["shell_exec"].risk_tier == "destroy"
    # Comms reads are observe-band (no state change, fixed own-account reads).
    assert inventory["read_email"].risk_tier == "observe"
    assert inventory["list_emails"].risk_tier == "observe"
    # Every row serves the mapping version (grant surfaces pin against it).
    from abstractcore.tools.risk_facts import RISK_MAPPING_VERSION

    for d in inventory.values():
        assert d.risk_mapping_version == RISK_MAPPING_VERSION


def test_send_email_carries_the_recipient_refiner():
    # dm#244: send_email declares a per-call refiner so the enforcement layer
    # can lower a call to the registered operator to auto; the grant-time band
    # stays outreach (the ceiling + deny-safe default). Only send_email has it.
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    se = inventory["send_email"]
    assert se.risk_refiner == "send_email_recipient@v1"
    assert se.risk_tier == "outreach" and se.risk_rank == 3, "the band is the ceiling, refiner is band-neutral"
    # whatsapp/telegram have no registered operator-recipient concept.
    assert inventory["send_whatsapp_message"].risk_refiner is None
    # It rides to_dict on the wire.
    row = {r["name"]: r for r in builtin_tool_inventory_as_dicts()}["send_email"]
    assert row["risk_refiner"] == "send_email_recipient@v1"


def test_execute_command_carries_the_git_read_only_refiner():
    # runtime c5042/c5050: execute_command declares git_read_only@v1 so the
    # approval lane can LOWER a proven read-only git invocation to auto — the
    # grant-time band stays destroy (the ceiling + deny-safe default; refiner
    # is band-neutral). Declaring it retires the clients' hand-rolled git
    # allowlists. Same architecture as send_email_recipient@v1.
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    ec = inventory["execute_command"]
    assert ec.risk_refiner == "git_read_only@v1"
    assert ec.risk_tier == "destroy" and ec.risk_rank == 4, "the band is the ceiling, refiner is band-neutral"
    row = {r["name"]: r for r in builtin_tool_inventory_as_dicts()}["execute_command"]
    assert row["risk_refiner"] == "git_read_only@v1"
    # Exactly the two declared refiner carriers today (fail-closed: a stale
    # entry naming no scanned tool refuses the whole inventory at build).
    refined = {n for n, d in inventory.items() if d.risk_refiner}
    assert refined == {"send_email", "execute_command"}


def test_comms_send_fact_classification():
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    senders = {n for n, d in inventory.items() if d.comms_send}
    assert senders == {
        "send_email",
        "send_whatsapp_message",
        "send_telegram_message",
        "send_telegram_artifact",
    }
    # Every comms sender also carries the approval-rule fact: recipients are
    # model-controlled arguments.
    for name in senders:
        assert inventory[name].model_controlled_destination is True


def test_model_controlled_destination_on_model_chosen_egress_tools():
    """runtime ruling c4879: fetch_url and browser_probe carry
    model_controlled_destination — their egress destination is model-chosen
    (fetch_url's URL/method, browser_probe navigates + executes a model-chosen
    URL's JS). The fact is band-NEUTRAL: both stay `act` (the approval lane
    consumes mcd via a ceiling-skip, not a band change). Resolves the
    vocabulary/inventory mismatch two adversaries flagged (risk_facts.py cited
    fetch_url as the mcd archetype yet the inventory hadn't set it)."""
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    for name in ("fetch_url", "browser_probe"):
        assert inventory[name].model_controlled_destination is True, name
        assert inventory[name].risk_tier == "act", f"{name}: mcd must not move the band"


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
    assert len(exported_inventory()) == 27
    assert len(exported_dicts()) == 27


def test_comms_lanes_now_in_scope():
    """SCHEMA V3 (dm#221 tool-surfacing audit): comms lanes JOINED the
    inventory — the v1 exclusion made email/WhatsApp/Telegram invisible to
    every host's discovery, which the operator's audit ruled a gap. They now
    surface WITH their risk facts (send_* → outreach band). The old
    out-of-scope pin is flipped here, on the record."""
    assert "abstractcore.tools.comms_tools" in _INVENTORY_MODULES
    assert "abstractcore.tools.telegram_tools" in _INVENTORY_MODULES
    inventory = {d.name: d for d in list_builtin_tool_inventory()}
    for name in ("list_emails", "send_email", "send_telegram_message", "send_telegram_artifact"):
        assert name in inventory
    # And they surface vetted (facts declared), never unvetted-at-top.
    assert inventory["send_email"].risk_tier == "outreach"
    assert inventory["list_emails"].risk_tier == "observe"

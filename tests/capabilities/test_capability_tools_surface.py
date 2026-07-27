"""Capability-tools surface: plugins contribute tools THROUGH core (layering).

Laurent dm#16 / commons c4210: only abstractcore imports capability plugins;
runtime/gateway consume a plugin's tools through core's accessor, never by
importing the plugin package. These pins hold the surface contract:

- a plugin registers tools by capability name; core stores + surfaces them;
- the accessor returns a COPY (a consumer cannot poison the registry);
- an unknown capability returns [] (never raises);
- re-register replaces (a plugin re-run never duplicates);
- the surface is GENERAL — no capability name is hardcoded.
"""

from __future__ import annotations

import pytest

from abstractcore.capabilities.registry import CapabilityRegistry


class _Tool:
    def __init__(self, name: str):
        self.name = name


@pytest.fixture()
def registry() -> CapabilityRegistry:
    reg = CapabilityRegistry(owner=None)
    # Hermetic: the read side ensure-loads entry-point plugins (the
    # silent-vanish fix), which on a dev machine would pull the REAL
    # abstractcamera contribution into these contract pins. Mark loaded so
    # the fixture registry stays empty; the load-ordering behavior has its
    # own pin below.
    reg._plugins_loaded = True
    return reg


def test_register_and_retrieve_by_capability(registry):
    registry.register_capability_tools("camera", [_Tool("camera_photo"), _Tool("camera_video")])
    names = [t.name for t in registry.capability_tools("camera")]
    assert names == ["camera_photo", "camera_video"]


def test_unknown_capability_returns_empty_never_raises(registry):
    assert registry.capability_tools("camera") == []
    assert registry.capability_tools("nonexistent") == []


def test_accessor_returns_a_copy(registry):
    registry.register_capability_tools("camera", [_Tool("camera_photo")])
    got = registry.capability_tools("camera")
    got.append(_Tool("injected"))
    assert len(registry.capability_tools("camera")) == 1, "consumer mutation must not poison the registry"


def test_re_register_replaces_not_appends(registry):
    registry.register_capability_tools("camera", [_Tool("a"), _Tool("b")])
    registry.register_capability_tools("camera", [_Tool("a")])
    assert [t.name for t in registry.capability_tools("camera")] == ["a"], "a plugin re-run must not duplicate"


def test_duplicate_names_same_object_dedupes_different_objects_refused(registry):
    # Same-object re-export is harmless (dedupe); two DIFFERENT definitions
    # claiming one name is a plugin bug that must be seen, not a coin flip
    # (adversary P1-4; mirrors tools/inventory.py discipline).
    same = _Tool("camera_photo")
    registry.register_capability_tools("camera", [same, same])
    assert [t.name for t in registry.capability_tools("camera")] == ["camera_photo"]
    with pytest.raises(ValueError):
        registry.register_capability_tools("camera", [_Tool("camera_photo"), _Tool("camera_photo")])


def test_nameless_and_string_contributions_refused(registry):
    # A nameless tool is unaddressable by every consumer; a bare string is
    # iterable and would silently store per-character junk (adversary P2).
    with pytest.raises(ValueError):
        registry.register_capability_tools("camera", [_Tool("")])
    with pytest.raises(ValueError):
        registry.register_capability_tools("camera", "camera_photo")


def test_accessor_isolates_mutable_schema_containers(registry):
    # Consumers legitimately rewrite parameters in place (runtime's
    # _normalize_tool_spec); one consumer's rewrite must not poison the
    # registry or sibling consumers (adversary P1-3).
    class _RichTool:
        def __init__(self):
            self.name = "camera_photo"
            self.parameters = {"camera": {"type": "string"}}
            self.tags = ["capture"]
            self.examples = [{"camera": "0"}]

    original = _RichTool()
    registry.register_capability_tools("camera", [original])
    first = registry.capability_tools("camera")[0]
    first.parameters["camera"]["type"] = "integer"
    first.tags.append("mutated")
    second = registry.capability_tools("camera")[0]
    assert second.parameters["camera"]["type"] == "string", "consumer mutation must not reach the store"
    assert second.tags == ["capture"]
    assert original.parameters["camera"]["type"] == "string", "the plugin's module-level object must stay pristine"


def test_snapshot_of_all_capabilities(registry):
    registry.register_capability_tools("camera", [_Tool("camera_photo")])
    registry.register_capability_tools("audio", [_Tool("audio_x")])
    snap = registry.capability_tools()
    assert {k: [t.name for t in v] for k, v in snap.items()} == {
        "camera": ["camera_photo"],
        "audio": ["audio_x"],
    }
    # snapshot lists are copies too
    snap["camera"].append(_Tool("injected"))
    assert len(registry.capability_tools("camera")) == 1


def test_general_not_camera_special(registry):
    """A hypothetical future capability surfaces tools identically — no name
    is hardcoded in the surface."""
    registry.register_capability_tools("some_future_capability", [_Tool("ft_do")])
    assert [t.name for t in registry.capability_tools("some_future_capability")] == ["ft_do"]


def test_empty_or_none_clears(registry):
    registry.register_capability_tools("camera", [_Tool("camera_photo")])
    registry.register_capability_tools("camera", [])
    assert registry.capability_tools("camera") == []
    registry.register_capability_tools("camera", None)
    assert registry.capability_tools("camera") == []


def test_blank_capability_refused(registry):
    with pytest.raises(ValueError):
        registry.register_capability_tools("  ", [_Tool("x")])


def test_non_iterable_tools_refused(registry):
    with pytest.raises(ValueError):
        registry.register_capability_tools("camera", 42)


# --- plugin-load ordering (the silent-vanish window) ---


def test_capability_tools_read_side_ensure_loads_plugins():
    """A FRESH registry's capability_tools() must trigger entry-point plugin
    loading — otherwise an installed plugin's contribution reads as [] until
    some OTHER accessor happens to run first (the silent-vanish window;
    seat: camera)."""
    reg = CapabilityRegistry(owner=None)
    loads: list[bool] = []
    original = reg._ensure_plugins_loaded

    def _spy():
        loads.append(True)
        # Do NOT run the real loader: keep the pin hermetic (no dependence
        # on which plugins the dev machine has installed).
        reg._plugins_loaded = True

    reg._ensure_plugins_loaded = _spy  # type: ignore[method-assign]
    reg.capability_tools("camera")
    assert loads, "capability_tools() must ensure plugins loaded before answering"
    _ = original


def test_capability_tool_policy_read_side_ensure_loads_plugins():
    """The POLICY accessor has the same silent-vanish hazard (adversary
    P2-4: a fresh process constructing a ToolApprovalPolicy before any
    toolset build reads the policy FIRST) — pin its ensure-load too, or
    removing it silently unfolds the partition for policy-first callers."""
    reg = CapabilityRegistry(owner=None)
    loads: list[bool] = []

    def _spy():
        loads.append(True)
        reg._plugins_loaded = True

    reg._ensure_plugins_loaded = _spy  # type: ignore[method-assign]
    reg.capability_tool_policy("camera")
    assert loads, "capability_tool_policy() must ensure plugins loaded before answering"


def test_concurrent_first_reads_never_see_partial_load():
    """Adversary P1-1: the one-time plugin load must be SERIALIZED — a
    reader racing the first load used to get [] for an INSTALLED capability
    (silent, normal-looking, poisoned per-run tool maps for their
    lifetime). With the load lock, the racing thread blocks and reads the
    full contribution. Reentrancy (a plugin calling an accessor from
    register(), same thread) keeps working — the loader thread itself
    re-enters through the RLock."""
    import threading
    import time

    reg = CapabilityRegistry(owner=None)

    def _slow_load():
        # Simulate the real loader: a reentrant same-thread read mid-load
        # (what a plugin's register() does) must return the partial view,
        # not deadlock or recurse into a second load.
        assert reg.capability_tools("camera") == []
        time.sleep(0.25)
        reg._capability_tools["camera"] = [_Tool("camera_photo")]

    reg._load_plugins = _slow_load  # type: ignore[method-assign]

    results: dict[str, list] = {}

    def _first_reader():
        results["a"] = reg.capability_tools("camera")

    def _racing_reader():
        time.sleep(0.05)  # arrive mid-load
        results["b"] = reg.capability_tools("camera")

    ta = threading.Thread(target=_first_reader)
    tb = threading.Thread(target=_racing_reader)
    ta.start()
    tb.start()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert [t.name for t in results["a"]] == ["camera_photo"]
    assert [t.name for t in results["b"]] == ["camera_photo"], (
        "a reader racing the first plugin load must block until the "
        "contribution is in, never answer [] for an installed capability"
    )


# --- capability tool POLICY (approval partition) surface ---


def test_policy_register_and_retrieve(registry):
    registry.register_capability_tool_policy(
        "camera",
        {"auto_approve": ["camera_status"], "require_approval": ["camera_capture_photo"]},
    )
    got = registry.capability_tool_policy("camera")
    assert got == {
        "auto_approve": ["camera_status"],
        "require_approval": ["camera_capture_photo"],
    }


def test_policy_absent_returns_empty_dict_never_raises(registry):
    assert registry.capability_tool_policy("camera") == {}
    assert registry.capability_tool_policy("nonexistent") == {}


def test_policy_accessor_returns_a_copy(registry):
    registry.register_capability_tool_policy("camera", {"auto_approve": ["a"], "require_approval": []})
    got = registry.capability_tool_policy("camera")
    got["auto_approve"].append("injected")
    assert registry.capability_tool_policy("camera")["auto_approve"] == ["a"]


def test_policy_re_register_replaces(registry):
    registry.register_capability_tool_policy("camera", {"auto_approve": ["a"], "require_approval": ["b"]})
    registry.register_capability_tool_policy("camera", {"auto_approve": [], "require_approval": ["b"]})
    assert registry.capability_tool_policy("camera") == {"auto_approve": [], "require_approval": ["b"]}


def test_policy_rejects_non_dict_and_string_lists(registry):
    with pytest.raises(ValueError):
        registry.register_capability_tool_policy("camera", ["not", "a", "dict"])
    with pytest.raises(ValueError):
        registry.register_capability_tool_policy("camera", {"auto_approve": "camera_status"})


def test_policy_is_general_not_camera_special(registry):
    registry.register_capability_tool_policy(
        "some_future_capability", {"auto_approve": ["ft_read"], "require_approval": ["ft_write"]}
    )
    assert registry.capability_tool_policy("some_future_capability")["require_approval"] == ["ft_write"]


def test_module_level_accessors_exist():
    """Runtime's import seam: abstractcore.capabilities exposes module-level
    capability_tools / capability_tool_policy / capability_tool_facts over a
    shared lazily-built registry (the accessor upper packages use so they
    never import plugin packages)."""
    import abstractcore.capabilities as caps

    assert callable(caps.capability_tools)
    assert callable(caps.capability_tool_policy)
    assert callable(caps.capability_tool_facts)
    assert callable(caps.shared_capability_registry)
    reg1 = caps.shared_capability_registry()
    reg2 = caps.shared_capability_registry()
    assert reg1 is reg2, "shared registry must be a process-wide singleton"


def test_capability_tool_facts_register_validate_and_derive(registry):
    # Tool-tiers build: plugins declare RISK FACTS (danger-when-true); the
    # vocabulary is validated at the desk, and risk derives from them.
    registry.register_capability_tool_facts(
        "camera", {"camera_photo": {"captures_environment": True}}
    )
    facts = registry.capability_tool_facts("camera")
    assert facts["camera_photo"]["captures_environment"] is True
    from abstractcore.tools.risk_facts import derive_risk

    assert derive_risk(facts["camera_photo"]).band == "outreach"
    # Unknown capability -> {} (factless tools derive unvetted/top downstream).
    assert registry.capability_tool_facts("nope") == {}


def test_capability_tool_facts_refuse_unknown_spelling(registry):
    # The vocabulary is closed (danger-when-true polarity): a typo refuses at
    # the desk, never silently under-derives a tier.
    with pytest.raises(ValueError):
        registry.register_capability_tool_facts("camera", {"camera_photo": {"is_safe": True}})
    with pytest.raises(ValueError):
        registry.register_capability_tool_facts("camera", "not-a-dict")


def test_capability_tool_facts_refuse_empty_declaration(registry):
    # Belt for the derive_risk empty-dict fail-closed (adversary P0): an
    # empty per-tool facts dict is indistinguishable from a forgotten one —
    # refuse it at the desk rather than serve a tool that derives observe.
    with pytest.raises(ValueError):
        registry.register_capability_tool_facts("camera", {"camera_photo": {}})


# --- capability tool DEFAULTS (privacy/default-off class) surface ---
# runtime ruling c4886 / gateway c4892: derive-over-hardcode. A plugin
# declares its default-off tools; the gateway SEEDS them disabled (seed-only-
# when-unset; operator console edits win thereafter — dm#194).


def test_defaults_register_and_retrieve(registry):
    registry.register_capability_tool_defaults(
        "camera",
        {"default_disabled": ["camera_capture_photo", "camera_capture_video", "camera_open"]},
    )
    got = registry.capability_tool_defaults("camera")
    assert got == {"default_disabled": ["camera_capture_photo", "camera_capture_video", "camera_open"]}


def test_defaults_absent_returns_empty_dict_never_raises(registry):
    assert registry.capability_tool_defaults("camera") == {}
    assert registry.capability_tool_defaults("nonexistent") == {}


def test_defaults_accessor_returns_a_copy(registry):
    registry.register_capability_tool_defaults("camera", {"default_disabled": ["camera_open"]})
    got = registry.capability_tool_defaults("camera")
    got["default_disabled"].append("injected")
    assert registry.capability_tool_defaults("camera")["default_disabled"] == ["camera_open"]


def test_defaults_re_register_replaces(registry):
    registry.register_capability_tool_defaults("camera", {"default_disabled": ["a", "b"]})
    registry.register_capability_tool_defaults("camera", {"default_disabled": ["b"]})
    assert registry.capability_tool_defaults("camera") == {"default_disabled": ["b"]}


def test_defaults_dedupe_preserves_order(registry):
    registry.register_capability_tool_defaults("camera", {"default_disabled": ["b", "a", "b", "a"]})
    assert registry.capability_tool_defaults("camera") == {"default_disabled": ["b", "a"]}


def test_defaults_empty_or_none_clears(registry):
    # Distinct from facts' empty-dict REFUSAL: an empty default_disabled is a
    # legitimate "no tool is off", not a forgotten declaration.
    registry.register_capability_tool_defaults("camera", {"default_disabled": ["camera_open"]})
    registry.register_capability_tool_defaults("camera", {"default_disabled": []})
    assert registry.capability_tool_defaults("camera") == {}
    registry.register_capability_tool_defaults("camera", {"default_disabled": ["camera_open"]})
    registry.register_capability_tool_defaults("camera", None)
    assert registry.capability_tool_defaults("camera") == {}


def test_defaults_rejects_non_dict_and_string_list(registry):
    with pytest.raises(ValueError):
        registry.register_capability_tool_defaults("camera", ["not", "a", "dict"])
    with pytest.raises(ValueError):
        registry.register_capability_tool_defaults("camera", {"default_disabled": "camera_open"})


def test_defaults_is_general_not_camera_special(registry):
    # The next environment-capturing plugin (mic/screen) declares its class
    # the SAME way — default-off follows the declaration, no hardcoded name.
    registry.register_capability_tool_defaults("mic", {"default_disabled": ["mic_record"]})
    assert registry.capability_tool_defaults("mic") == {"default_disabled": ["mic_record"]}

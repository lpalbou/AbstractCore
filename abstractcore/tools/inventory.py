"""Authoritative builtin-tool inventory (derive-never-copy source).

The framework's tool-inventory expansion (entity-agency Phase 0.5) needs ONE
authoritative enumeration of core's builtin tools that downstream seats DERIVE
from instead of copying: runtime's grant tool universe, the gateway's door
declarations, and the observer's descriptor contract all read this surface.

Design rules (the spec this implements):

- PROGRAMMATIC DERIVATION, never a hand-maintained list: the inventory is
  built by scanning the builtin tool modules for ``@tool``-decorated
  functions (their attached ``ToolDefinition``). Adding a tool to a scanned
  module makes it appear here automatically; a hand-list would silently rot.
- CORE EMITS ONLY THE FACTS CORE OWNS: name, owning package, module origin,
  the mutating classification of its OWN tools, the ``act_only`` wire-boundary
  flag, and the description. TIER (tier1/tier2) is runtime vocabulary and
  CONTAINMENT (entity-walled election vs registry binding) is the door's —
  those fields are deliberately absent so no second copy of another seat's
  classification can drift here.
- BYTE-STABLE OUTPUT: deterministic ordering (module, then name) so member
  sets and serialized forms are stable across processes and releases; tests
  pin the exact member sets.

Scope note (v3, 2026-07-23): ``comms_tools`` (email/WhatsApp) and
``telegram_tools`` JOINED the inventory — the operator's tool-surfacing
audit (dm#221, item H) found the v1 exclusion made email/telegram invisible
to every host's discovery; the audit ruled they must be discoverable, and
schema v3 lands them WITH their risk facts so they surface vetted (never
unvetted-at-top). The v1 exclusion note is kept in history; widening was —
as designed — one entry per module here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .core import ToolDefinition

# Modules whose @tool-decorated functions form the builtin inventory.
# Order matters only for deterministic output grouping.
_INVENTORY_MODULES: Tuple[str, ...] = (
    "abstractcore.tools.common_tools",
    "abstractcore.tools.shell_tools",
    "abstractcore.tools.comms_tools",
    "abstractcore.tools.telegram_tools",
    "abstractcore.tools.browser_tools",
)

# Core's OWN classification of its OWN tools — the mutating + remote_write_capable
# facts per tool (plus optional model_cost for tools that run a nested LLM call,
# schema v2), decided together (adversarial-review finding: a single "mutating"
# bit glossed over fetch_url's model-controlled method/data escape hatch — the tool defaults
# to GET but CAN send POST/PUT/DELETE with a body, and downstream approval
# layers are name-based, never argument-based):
#
# - "mutating": invoking the tool can change LOCAL host state — filesystem
#   writes, arbitrary command/code execution, interactive shells. This is
#   the fact approval defaults key on.
# - "remote_write_capable": the tool can send STATE-CHANGING requests to
#   remote systems when given model-controlled arguments (fetch_url's
#   method/data). GET-shaped read lanes (web_search, skim_*) are not.
#   A derive-side that auto-approves on mutating=False alone MUST also
#   consult this fact before exposing the tool to unattended loops.
#
# This mapping is EXHAUSTIVE over the inventory by construction, BOTH ways
# (adversarial-review finding: one-directional fail-closed left stale
# entries lingering as silent pre-classifications for future name reuse):
# a scanned tool missing here REFUSES the whole inventory, and an entry
# here naming no scanned tool ALSO refuses — classification is a deliberate
# decision at add time AND at remove time, never a default.
_CLASSIFICATION_BY_NAME: Dict[str, Dict[str, bool]] = {
    # common_tools
    "analyze_code": {"mutating": False, "remote_write_capable": False},
    # model_cost=True: runs a NESTED LLM call (the configured vision model)
    # — read-only in effect, but hosts budgeting/approving by cost must see
    # it distinctly (0825 ruling, agora c3977). Single-attempt by
    # construction (no retry stacking).
    "analyze_media": {"mutating": False, "remote_write_capable": False, "model_cost": True},
    "edit_file": {"mutating": True, "remote_write_capable": False},
    # destructive_capable: the argv-class ceiling — rm and git-mutable
    # commands are PROGRAMS reachable through this tool, so it clamps to the
    # maximum it can reach (per-argument refinement is approval-time, the
    # bridge-policy read-only-git precedent, never a grant-time discount).
    "execute_command": {"mutating": True, "remote_write_capable": False, "destructive_capable": True},
    # model_controlled_destination (RULED by runtime, the approval-lane owner,
    # commons c4879, RISK_MAPPING_VERSION bumped to 2): the URL is model-chosen,
    # so the fact is simply true; the approval lane consumes it (the risk-rank
    # ceiling SKIPS mcd tools so they are never silenced by an operator ceiling
    # — the guard derives from the fact, never a name list). Band stays act
    # (mcd is band-neutral). Resolves the vocabulary/inventory mismatch two
    # adversaries flagged (risk_facts.py cited fetch_url as the mcd archetype).
    "fetch_url": {"mutating": False, "remote_write_capable": True, "model_controlled_destination": True},
    "list_files": {"mutating": False, "remote_write_capable": False},
    "read_file": {"mutating": False, "remote_write_capable": False},
    "search_files": {"mutating": False, "remote_write_capable": False},
    "skim_files": {"mutating": False, "remote_write_capable": False},
    "skim_folders": {"mutating": False, "remote_write_capable": False},
    "skim_url": {"mutating": False, "remote_write_capable": False},
    "skim_websearch": {"mutating": False, "remote_write_capable": False},
    "web_search": {"mutating": False, "remote_write_capable": False},
    "write_file": {"mutating": True, "remote_write_capable": False},
    # shell_tools — destructive_capable: an interactive shell reaches rm and
    # git-mutable commands, so the TOOL clamps to the maximum it can reach
    # (per-argument refinement is approval-time, never a grant-time discount).
    "shell_exec": {"mutating": True, "remote_write_capable": False, "destructive_capable": True},
    "shell_write_stdin": {"mutating": True, "remote_write_capable": False, "destructive_capable": True},
    "shell_close": {"mutating": True, "remote_write_capable": False},
    # comms_tools (schema v3, dm#221 audit) — send_* carries comms_send
    # (messages humans receive, socially irreversible → outreach band);
    # recipients are MODEL-CONTROLLED arguments, so the approval-rule fact
    # model_controlled_destination rides too. read/list are observe-band
    # (network reads of the operator's own accounts; no state change).
    "send_email": {
        "mutating": False,
        "remote_write_capable": True,
        "comms_send": True,
        "model_controlled_destination": True,
    },
    "read_email": {"mutating": False, "remote_write_capable": False},
    "list_emails": {"mutating": False, "remote_write_capable": False},
    "list_email_accounts": {"mutating": False, "remote_write_capable": False},
    "send_whatsapp_message": {
        "mutating": False,
        "remote_write_capable": True,
        "comms_send": True,
        "model_controlled_destination": True,
    },
    "read_whatsapp_message": {"mutating": False, "remote_write_capable": False},
    "list_whatsapp_messages": {"mutating": False, "remote_write_capable": False},
    # NOTE (dm#244): send_email carries a per-call REFINER (see
    # _REFINER_BY_NAME) so the enforcement layer can lower a call to the
    # REGISTERED OPERATOR to auto; the grant-time band stays outreach (the
    # ceiling + deny-safe default). The refiner is band-neutral here.
    # telegram_tools — same comms shape.
    "send_telegram_message": {
        "mutating": False,
        "remote_write_capable": True,
        "comms_send": True,
        "model_controlled_destination": True,
    },
    "send_telegram_artifact": {
        "mutating": False,
        "remote_write_capable": True,
        "comms_send": True,
        "model_controlled_destination": True,
    },
    # browser_tools (operator-approved dm:core--laurent#21) — the render
    # probe navigates a headless browser to a MODEL-CONTROLLED target and
    # executes that page's JavaScript, which can itself send requests
    # anywhere: the same escape-hatch class as fetch_url (remote_write_
    # capable, never read-only-safe). NOT mutating: screenshots land only
    # in a fresh temp dir (no model-controlled local-write path); local
    # file targets block outbound network by default.
    # model_controlled_destination TRUE (runtime ruling c4879): the probe
    # navigates a model-chosen URL and executes its JS — a superset of
    # fetch_url's model-controlled egress; same approval-lane consumption.
    "browser_probe": {"mutating": False, "remote_write_capable": True, "model_controlled_destination": True},
}


# Per-call REFINER ids by tool (operator ruling dm#244). A refiner names a
# server-side hook that may LOWER a call below its grant-time band by
# inspecting arguments — the band stays the ceiling/deny-safe default. Core
# DECLARES the id (dispatch-on-data, not a hardcoded tool-name check); the
# enforcement lane (runtime/gateway) runs the logic. Validated against
# risk_facts.KNOWN_REFINER_IDS at build time.
_REFINER_BY_NAME: Dict[str, str] = {
    # send_email → auto when the recipient is the registered operator email,
    # else the outreach ceiling holds (self=auto / others=ask, dm#244).
    "send_email": "send_email_recipient@v1",
    # execute_command → auto when the command PROVES read-only git (runtime's
    # two-stage conservative proof, built c5042; positional write verbs +
    # write/exec flags + wrappers + globals-before-verb + shell operators all
    # ASK). The destroy band stays the ceiling/deny-safe default; the refiner
    # only LOWERS a proven-safe git read at the approval point. Declaring it
    # here retires the clients' hand-rolled git allowlists (code-tui c5050).
    "execute_command": "git_read_only@v1",
}


# Consumers reading the row shape can key evolution on this: field ADDITIONS
# bump it; the descriptor never removes/renames fields within a version.
# v2 (2026-07-21): + model_cost — the tool runs a nested LLM call; hosts
# budgeting/approving by cost consult it alongside mutating/remote_write.
# v3 (2026-07-23, tool-tiers build): + comms_send / captures_environment /
# standing_effect / destructive_capable / model_controlled_destination risk
# facts, + the DERIVED risk_tier/risk_rank/risk_presentation/
# risk_mapping_version fields (the one versioned fold in risk_facts.py) —
# and the comms/telegram tools join the scanned modules (dm#221 audit).
INVENTORY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class BuiltinToolDescriptor:
    """One inventory row — the facts core owns about one builtin tool.

    Join-key contract: ``(owner, name)`` is the STABLE identity consumers key
    on; ``module`` is informational provenance only (a file rename changes it
    while nothing semantic changed — never join on it). Not hashable despite
    ``frozen=True`` (``parameters`` is a dict); use ``name`` for set logic.
    """

    name: str
    owner: str  # always "core" for this inventory
    module: str  # short module origin, e.g. "common_tools" — provenance only
    mutating: bool
    remote_write_capable: bool
    act_only: bool
    description: str
    # True when the tool runs a NESTED LLM call (schema v2): read-only in
    # effect but real token spend — hosts budgeting/approving by cost must
    # see it distinctly from free reads.
    model_cost: bool = False
    # Risk facts (schema v3, tool-tiers build; danger-when-true polarity,
    # vocabulary closed in risk_facts.KNOWN_FACT_NAMES):
    comms_send: bool = False
    captures_environment: bool = False
    standing_effect: bool = False
    destructive_capable: bool = False
    # Approval-rule fact (band-neutral): the MODEL chooses the destination.
    model_controlled_destination: bool = False
    # DERIVED risk (the one versioned fold — never hand-assigned). Defaults
    # are FAIL-CLOSED (adversary P2): a hand-constructed row born without an
    # explicit derivation is treated as highest risk, never observe/safe.
    # list_builtin_tool_inventory always sets these from derive_risk.
    risk_tier: str = "destroy"
    risk_rank: int = 4
    risk_presentation: str = "unvetted"
    risk_mapping_version: int = 0
    # Per-call refiner id (dm#244): names a server-side hook that may LOWER a
    # call below the band (never raise); None = no refiner, band is final.
    risk_refiner: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        import copy

        return {
            "name": self.name,
            "owner": self.owner,
            "module": self.module,
            "mutating": self.mutating,
            "remote_write_capable": self.remote_write_capable,
            "act_only": self.act_only,
            "model_cost": self.model_cost,
            "comms_send": self.comms_send,
            "captures_environment": self.captures_environment,
            "standing_effect": self.standing_effect,
            "destructive_capable": self.destructive_capable,
            "model_controlled_destination": self.model_controlled_destination,
            "risk_tier": self.risk_tier,
            "risk_rank": self.risk_rank,
            "risk_presentation": self.risk_presentation,
            "risk_mapping_version": self.risk_mapping_version,
            "risk_refiner": self.risk_refiner,
            "description": self.description,
            # Deep copy: the nested per-parameter dicts must never alias the
            # live ToolDefinition schema (see list_builtin_tool_inventory).
            "parameters": copy.deepcopy(self.parameters),
        }


def _scan_module_tool_definitions(module_path: str) -> List[ToolDefinition]:
    """Collect the ToolDefinitions attached by @tool in one module.

    Import happens lazily here (not at package import) so the inventory does
    not force optional tool dependencies onto every abstractcore import.
    (Side-effect note: importing shell_tools initializes its session registry
    and registers an atexit cleanup hook — benign, no processes spawn.)

    Re-exports of the SAME function dedupe by name; two DIFFERENT definitions
    claiming one name REFUSE loudly (a silent last-in-dir()-order win would
    serve the wrong ToolDefinition with every test green).
    """
    import importlib

    module = importlib.import_module(module_path)
    found: Dict[str, ToolDefinition] = {}
    for attr_name in sorted(dir(module)):
        obj = getattr(module, attr_name, None)
        tool_def = getattr(obj, "_tool_definition", None)
        if isinstance(tool_def, ToolDefinition):
            existing = found.get(tool_def.name)
            if existing is not None and existing is not tool_def:
                raise RuntimeError(
                    f"Builtin tool inventory refuses: two different tool "
                    f"definitions in {module_path} claim the name "
                    f"'{tool_def.name}'."
                )
            found[tool_def.name] = tool_def
    return [found[k] for k in sorted(found)]


def list_builtin_tool_inventory() -> List[BuiltinToolDescriptor]:
    """The authoritative inventory of core's builtin tools.

    Programmatically derived from the @tool definitions in the builtin tool
    modules; deterministic (module order, then name order); REFUSES loudly
    on: a tool without an explicit classification, a STALE classification
    naming no scanned tool, and a name claimed by two modules (fail-closed
    in every direction — see ``_CLASSIFICATION_BY_NAME``).
    """
    import copy

    from .risk_facts import derive_risk

    descriptors: List[BuiltinToolDescriptor] = []
    unclassified: List[str] = []
    seen_names: Dict[str, str] = {}
    for module_path in _INVENTORY_MODULES:
        short_module = module_path.rsplit(".", 1)[-1]
        for tool_def in _scan_module_tool_definitions(module_path):
            prior_module = seen_names.get(tool_def.name)
            if prior_module is not None:
                raise RuntimeError(
                    f"Builtin tool inventory refuses: tool name "
                    f"'{tool_def.name}' is claimed by both {prior_module} "
                    f"and {short_module}."
                )
            seen_names[tool_def.name] = short_module
            classification = _CLASSIFICATION_BY_NAME.get(tool_def.name)
            if classification is None:
                unclassified.append(f"{short_module}.{tool_def.name}")
                continue
            # Derive the risk band from the declared facts through the ONE
            # versioned mapping (never hand-assigned; risk_facts.py). A
            # per-call refiner id (dm#244) rides ALONGSIDE the band — it never
            # changes the derived band (the band is the ceiling/deny-safe
            # default); the enforcement layer may LOWER a call below it.
            risk = derive_risk(classification, refiner=_REFINER_BY_NAME.get(tool_def.name))
            descriptors.append(
                BuiltinToolDescriptor(
                    name=tool_def.name,
                    owner="core",
                    module=short_module,
                    mutating=bool(classification["mutating"]),
                    remote_write_capable=bool(classification["remote_write_capable"]),
                    act_only=bool(getattr(tool_def, "act_only", False)),
                    model_cost=bool(classification.get("model_cost", False)),
                    comms_send=bool(classification.get("comms_send", False)),
                    captures_environment=bool(classification.get("captures_environment", False)),
                    standing_effect=bool(classification.get("standing_effect", False)),
                    destructive_capable=bool(classification.get("destructive_capable", False)),
                    model_controlled_destination=bool(
                        classification.get("model_controlled_destination", False)
                    ),
                    risk_tier=risk.band,
                    risk_rank=risk.rank,
                    risk_presentation=risk.presentation,
                    risk_mapping_version=risk.mapping_version,
                    risk_refiner=risk.refiner,
                    description=str(tool_def.description or ""),
                    # Deep copy: ToolDefinition.parameters nests per-param
                    # dicts that are the LIVE schema providers serialize for
                    # native tool calls — a consumer mutating a served row
                    # must never rewrite the process-wide tool schema.
                    parameters=copy.deepcopy(tool_def.parameters or {}),
                )
            )
    if unclassified:
        raise RuntimeError(
            "Builtin tool inventory refuses: tool(s) without an explicit "
            f"classification: {', '.join(sorted(unclassified))}. Add each to "
            "_CLASSIFICATION_BY_NAME in abstractcore/tools/inventory.py — "
            "the classification is a deliberate decision, never a default."
        )
    stale = sorted(set(_CLASSIFICATION_BY_NAME) - set(seen_names))
    if stale:
        raise RuntimeError(
            "Builtin tool inventory refuses: stale classification entries "
            f"naming no scanned tool: {', '.join(stale)}. Remove each from "
            "_CLASSIFICATION_BY_NAME — a lingering entry would silently "
            "pre-classify a future tool reusing the name."
        )
    # Refiner map is fail-closed the same way (dm#244): an entry naming an
    # unscanned tool would silently attach a per-call hook to nothing.
    stale_refiners = sorted(set(_REFINER_BY_NAME) - set(seen_names))
    if stale_refiners:
        raise RuntimeError(
            "Builtin tool inventory refuses: refiner entries naming no scanned "
            f"tool: {', '.join(stale_refiners)}. Remove each from _REFINER_BY_NAME."
        )
    return descriptors


def list_builtin_tool_names() -> List[str]:
    """Flat, deterministic name list (module order, then name order)."""
    return [d.name for d in list_builtin_tool_inventory()]


def builtin_tool_inventory_as_dicts() -> List[Dict[str, Any]]:
    """JSON-ready form of the inventory (stable field set + ordering)."""
    return [d.to_dict() for d in list_builtin_tool_inventory()]


__all__ = [
    "INVENTORY_SCHEMA_VERSION",
    "BuiltinToolDescriptor",
    "list_builtin_tool_inventory",
    "list_builtin_tool_names",
    "builtin_tool_inventory_as_dicts",
]

"""Risk facts + the ONE versioned fact→tier derivation (tool-tiers build).

The room's converged design (plans/tool-tiers.md, operator-ordered
2026-07-22, 11 seats + adversaries): a user must KNOW what powers they grant
an agent or entity. TWO AXES: `capability_class` (boundary — who may even
see a tool; structural, fail-closed; owned by runtime) and the RISK band
(what a user grants; operator-facing). Risk is NEVER hand-assigned per tool:
each tool declares FACTS (checkable claims), and the band is derived by the
ONE mapping below — core HOSTS it as code (lowest-common-dependency,
import-never-copy, the SELF_FRACTION_FLOOR precedent); its CONTENT is
gateway+runtime's to change (sign-off + RISK_MAPPING_VERSION bump).

Vocabulary rules (semantics desk, doc v19/v28 — declaration-before-engraving):

- POLARITY: every fact reads DANGER-WHEN-TRUE (matching the shipped
  mutating/remote_write_capable). A safe-when-true name inverts the family
  and invites double-negative derivation bugs — `known_fact_names` refuses
  unknown spellings at the registration desk, so a new fact is a recorded
  vocabulary decision, never a drive-by kwarg.
- FACTS FEED TWO RULES: the tier fold below AND the approval derivation.
  `model_controlled_destination` is consumed by the APPROVAL rule (the live
  agora-auto-vs-fetch_url-broker ruling), not the tier fold — it is not dead
  weight, do not "optimize" it away as unread. `model_cost` is a BUDGET
  fact. Neither moves the band.
- BAND WORDS are the IDs at rest (observe/act/outreach/destroy); integers
  are display-only ranks. The bare word "tier" is reserved for THIS axis
  (semantics word-ownership ruling); never reuse it for boundary classes.
- The tier count is FROZEN at 4: every future distinction becomes a FACT
  (tooltip-visible), never a band 3.5.
- Facts-undeclared gates at destroy level but PRESENTS as `unvetted`
  ("powers undeclared, treated as highest for gating") — never rendered as
  "destructive": overclaim habituates users and kills the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

# Bump on a change to the fact VOCABULARY or the derivation MAPPING (the
# fact→band fold) — NOT on a per-tool classification change. Served with every
# derived row so grant surfaces pin tier-at-grant-time and re-prompt when an
# upward move lands (the room's re-derivation rule: upward re-asks, downward
# keeps). Rationale for the boundary: consumers derive from the FACTS fresh, so
# a band-neutral per-tool fact change (e.g. runtime's c4879 ruling to set
# model_controlled_destination on fetch_url + browser_probe — both stay `act`)
# propagates without a version bump, and the approval-lane guard reads mcd by
# derivation, never pinned by version. A bump with an unchanged mapping would
# contradict what the version means (and force spurious re-derivation).
RISK_MAPPING_VERSION = 1

# The unified fact vocabulary (danger-when-true, one spelling per fact).
# Shipped facts first, then the tool-tiers additions (doc v19 adoption).
KNOWN_FACT_NAMES: Tuple[str, ...] = (
    # -- shipped (inventory schema v1/v2) --
    "mutating",  # can change LOCAL host state (fs writes, exec)
    "remote_write_capable",  # can send state-changing remote requests
    "model_cost",  # runs a nested LLM call (BUDGET fact; band-neutral)
    # -- tool-tiers additions (schema v3) --
    "comms_send",  # a purpose-built human-messaging lane (email/chat/DM) — socially irreversible
    "captures_environment",  # camera/mic/screen capture of the real world
    "standing_effect",  # keeps acting after the call returns (monitors)
    "destructive_capable",  # can destroy state irreversibly (shell: rm/git reachable)
    # -- approval-rule fact (band-neutral; the fixed-channel discount axis) --
    "model_controlled_destination",  # the MODEL chooses where output goes (fetch_url URL)
)

# Band word IDs at rest; rank is display-only.
RISK_BANDS: Tuple[str, ...] = ("observe", "act", "outreach", "destroy")

# The presentation value for facts-undeclared rows (gates at destroy rank).
UNVETTED = "unvetted"


# --- session permission modes (thin-client READ+FORWARD; commons c4909-c5028) ---
# The operator-ruled session AUTONOMY vocabulary (words from the 2026-07-12
# abstractcode permission-modes ruling). A thin client FORWARDS one of these
# WORDS; the server (gateway/runtime) derives per-tool auto-vs-ask by comparing
# each tool's risk RANK to the mode's ceiling — NEVER a client-side name table
# (the drift the ruling kills). Core SERVES this ladder (hosted here beside the
# risk mapping it reads) so clients stop INVENTING it (code-tui C2); its CONTENT
# is the enforcement lane's to change under RISK_MAPPING_VERSION, same rule as
# the fact→band mapping.
PERMISSION_MODES: Tuple[str, ...] = ("read-only", "write", "full-auto")

# Each mode AUTO-APPROVES tools whose risk rank is <= its ceiling; everything
# above the ceiling ASKS (or, for the deny class under the settled contract, is
# subtracted from registration so the model never sees it). Two invariants ride
# ON TOP of the ceiling and are NOT encoded by it:
#   - model_controlled_destination tools NEVER silently auto-approve below
#     full-auto (the fetch_url-broker rule) — the approval lane enforces this by
#     derivation regardless of the ceiling;
#   - require-always-wins > explicit per-tool overrides > this mode default.
# Ceilings map the ruled semantics: read-only = observe only (mutations are
# denied, not asked); write = observe + act (walled bounded writes auto, while
# destroy-class program reach — shell/rm/git — asks); full-auto = every rank
# (an explicit high-trust posture; mcd-never-silenced still applies as the belt).
_PERMISSION_MODE_MAX_AUTO_RANK: Dict[str, int] = {
    "read-only": 1,   # observe
    "write": 2,       # observe + act
    "full-auto": 4,   # all ranks
}

# One-line semantics served alongside the words so a client renders the ladder
# without inventing copy (code-tui: "the entire posture ladder is client-invented today").
PERMISSION_MODE_SEMANTICS: Dict[str, str] = {
    "read-only": "auto-approve read-only (observe) tools; mutations are denied, never prompted",
    "write": "auto-approve observe + bounded local writes (act); program-class/destructive reach and outreach ask; model-controlled-destination tools ask",
    "full-auto": "auto-approve every tool (explicit high-trust posture — the operator accepts model-controlled-destination egress too; the mcd belt applies only BELOW full-auto)",
}


def permission_mode_max_auto_rank(mode: str) -> int:
    """The risk-rank CEILING a session permission mode auto-approves up to.

    A tool auto-approves under `mode` only if its risk rank <= this ceiling AND
    it is not a model_controlled_destination tool below full-auto (the belt the
    approval lane enforces separately). An unknown mode is treated as the SAFEST
    (read-only) — fail-closed, never fail-open to full-auto.

    Prefer `permission_mode_auto_approves(...)` — it combines this ceiling with
    the mcd belt in one decision so a consumer cannot apply the ceiling and
    forget the belt (the footgun gateway c5053 + the completeness review named).
    """
    m = str(mode or "").strip().lower()
    return _PERMISSION_MODE_MAX_AUTO_RANK.get(m, _PERMISSION_MODE_MAX_AUTO_RANK["read-only"])


def permission_mode_auto_approves(
    mode: str, *, risk_rank: Any, model_controlled_destination: bool = False
) -> bool:
    """The ONE served auto-vs-ask decision for a tool under a permission mode.

    Combines BOTH invariants so no consumer can apply the ceiling and forget
    the belt (gateway c5053: "a consumer that cannot forget the belt beats a
    documented warning"):
      1. rank ceiling — auto only if the tool's risk rank <= the mode's ceiling;
      2. mcd belt — a model_controlled_destination tool NEVER auto-approves
         BELOW full-auto (the fetch_url-broker rule), regardless of its rank
         (fetch_url/browser_probe are act(2) <= the write ceiling, so the
         ceiling alone would wrongly auto them under write; the belt makes them
         ask).
    Fail-closed: an unrankable/blank rank never autos; an unknown mode uses the
    read-only ceiling (via permission_mode_max_auto_rank). This is the derive-
    never-copy source — gateway/runtime/clients call it, never re-implement it.
    """
    m = str(mode or "").strip().lower()
    try:
        rank = int(risk_rank)
    except (TypeError, ValueError):
        return False  # unrankable → never auto
    if rank > permission_mode_max_auto_rank(m):
        return False
    if model_controlled_destination and m != "full-auto":
        return False  # mcd belt: never silently auto below full-auto
    return True


# Per-call REFINER declarations (operator ruling dm#244). A refiner is a
# named, versioned server-side hook that RE-CLASSIFIES a tool's risk PER
# INVOCATION by inspecting its arguments — same shape as execute_command's
# read-only-git proof, but declared as DATA (the row carries the id) instead
# of a hardcoded tool-name check in the enforcement layer.
#
# Contract core defines (the enforcement lane runs the LOGIC):
#   - LOWER-ONLY: a refiner may drop a call BELOW the grant-time ceiling band
#     (send_email to the registered operator → auto), it may NEVER raise a
#     call above it. The declared band is the CEILING.
#   - DENY-SAFE: until the refiner ships, and on every call it cannot PROVE
#     is safe, the ceiling band stands (self AND others ask). The refiner is
#     an optimization off a safe default, never a gate that could fail open.
#   - The refiner LOGIC (read gateway operator email, compare recipient,
#     defend cc/bcc/multi-recipient/display-name-vs-address spoofing) is
#     runtime/gateway's enforcement lane — core only DECLARES the id + band.
KNOWN_REFINER_IDS: Tuple[str, ...] = (
    "send_email_recipient@v1",  # recipient==registered operator → auto; else ceiling (dm#244)
    # execute_command → auto when the command is a PROVEN read-only git
    # invocation (runtime's two-stage conservative proof, c5042), else the
    # destroy ceiling holds. Same architecture as send_email_recipient: core
    # declares the id on the row, the enforcement lane implements the proof;
    # declaring it retires the clients' hand-rolled git allowlists.
    "git_read_only@v1",
)


def validate_refiner_id(refiner_id: Optional[str]) -> None:
    """Refuse an unknown refiner-id at the desk (declaration-before-engraving).

    A refiner-id is a cross-package dispatch contract: the enforcement layer
    keys on the exact string. An undeclared id is a typo (the enforcement
    layer would find no logic and — deny-safe — hold the ceiling, but the
    declaration is then dead weight) or an unversioned drift.
    """
    if refiner_id is None:
        return
    if refiner_id not in KNOWN_REFINER_IDS:
        raise ValueError(
            f"Unknown refiner id {refiner_id!r}: refiner ids are a versioned "
            f"cross-package dispatch contract (name@vN). Known: {list(KNOWN_REFINER_IDS)}. "
            f"A new refiner is a recorded decision in risk_facts.py + the enforcement lane."
        )


@dataclass(frozen=True)
class RiskAssessment:
    """The derived risk of one tool: band word, display rank, presentation.

    `presentation` differs from `band` only for facts-undeclared rows
    (band="destroy" for gating, presentation="unvetted" for rendering).
    `refiner` (optional) names a per-call refiner that may LOWER this call's
    risk below the band at approval time (never raise it) — the band is the
    ceiling and the deny-safe default (dm#244).
    """

    band: str
    rank: int
    presentation: str
    mapping_version: int = RISK_MAPPING_VERSION
    refiner: Optional[str] = None

    def __post_init__(self) -> None:
        # Validate at CONSTRUCTION too (adversary P2): a direct
        # RiskAssessment(refiner="bogus@v9") must refuse, not construct a
        # silent-unlabeled row whose hook dispatches to nothing downstream.
        validate_refiner_id(self.refiner)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_tier": self.band,
            "risk_rank": self.rank,
            "risk_presentation": self.presentation,
            "risk_mapping_version": self.mapping_version,
            # Absent (None) → no per-call refiner; the band is final. Present
            # → the enforcement layer MAY lower this call below the band.
            "risk_refiner": self.refiner,
        }


def validate_fact_names(facts: Mapping[str, Any]) -> None:
    """Refuse unknown fact spellings at the desk (never lint later).

    The vocabulary is CLOSED: a new fact name is a recorded decision
    (semantics review, danger-when-true polarity, declaration-before-
    engraving) — an unknown key here is either a typo (silently dropping it
    would under-derive the tier) or an undeclared vocabulary widening.
    """
    unknown = sorted(set(facts) - set(KNOWN_FACT_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown risk fact name(s) {unknown}: the fact vocabulary is closed "
            f"(danger-when-true polarity, declaration-before-engraving). Known: "
            f"{list(KNOWN_FACT_NAMES)}. New facts are a recorded vocabulary "
            f"decision in abstractcore/tools/risk_facts.py, never a drive-by key."
        )


def derive_risk(facts: Optional[Mapping[str, Any]], *, refiner: Optional[str] = None) -> RiskAssessment:
    """The ONE fact→band derivation (max-wins).

    - destroy (4): destructive_capable — the argv-class ceiling: a shell
      reaches rm/git-reset, so the TOOL clamps to the maximum it can reach;
      per-ARGUMENT refinement is a downstream approval-time layer (the
      bridge-policy read-only-git precedent), never a grant-time discount.
    - outreach (3): comms_send / captures_environment / standing_effect —
      socially or physically irreversible reach into the world.
    - act (2): mutating / remote_write_capable — real, bounded effects.
    - observe (1): every declared fact False.
    - facts UNDECLARED (None OR an EMPTY mapping): gates at destroy rank,
      presents `unvetted`. An empty dict is undeclared, NOT all-False — the
      idiomatic consumer join `derive_risk(facts_map.get(name, {}))` must
      fail CLOSED, never silently render an unknown tool observe/safe
      (adversary P0). A legitimate all-False declaration is a NON-EMPTY dict
      of explicit Falses and still derives observe.

    model_cost and model_controlled_destination are deliberately band-neutral
    (budget rule and approval rule respectively — facts feed two rules).

    NOTE (band assumption): the act band for bounded local writes
    (write_file/edit_file) presumes a CONTAINMENT boundary — the runtime
    workspace wall (`resolve_user_path`) — separates a bounded named-target
    write from the unbounded program-class reach of a shell (destroy). A
    wall-less host must treat write_file's act band accordingly.
    """
    validate_refiner_id(refiner)
    # A refiner NEVER changes the derived band — the band is the grant-time
    # CEILING and the deny-safe default; the refiner only rides ALONGSIDE it
    # so the enforcement layer knows a per-call lowering hook exists.
    # UNDECLARED rows STRIP the refiner entirely (adversary P1): "you cannot
    # refine what you cannot classify" — serving a lowering hook on a tool
    # whose powers are unknown would read as "may auto-approve an unclassified
    # tool", which the recipient proof cannot justify. Unvetted carries NO
    # refiner, ever.
    if not facts:  # None or empty mapping → undeclared, fail closed
        return RiskAssessment(band="destroy", rank=4, presentation=UNVETTED, refiner=None)
    validate_fact_names(facts)

    def _true(name: str) -> bool:
        return bool(facts.get(name, False))

    if _true("destructive_capable"):
        return RiskAssessment(band="destroy", rank=4, presentation="destroy", refiner=refiner)
    if _true("comms_send") or _true("captures_environment") or _true("standing_effect"):
        return RiskAssessment(band="outreach", rank=3, presentation="outreach", refiner=refiner)
    if _true("mutating") or _true("remote_write_capable"):
        return RiskAssessment(band="act", rank=2, presentation="act", refiner=refiner)
    return RiskAssessment(band="observe", rank=1, presentation="observe", refiner=refiner)


__all__ = [
    "KNOWN_FACT_NAMES",
    "KNOWN_REFINER_IDS",
    "RISK_BANDS",
    "RISK_MAPPING_VERSION",
    "UNVETTED",
    "PERMISSION_MODES",
    "PERMISSION_MODE_SEMANTICS",
    "permission_mode_max_auto_rank",
    "permission_mode_auto_approves",
    "RiskAssessment",
    "derive_risk",
    "validate_fact_names",
    "validate_refiner_id",
]

# Proposed: delegated-sight remainder — session-route delegation + first-run vision seeding

## Metadata
- Created: 2026-07-25
- Status: Proposed (cross-package; needs operator rulings — not a unilateral core change)
- Origin: operator escalation via code-tui c5568 (analyze_media refuses with a vision-capable
  session model), settled by a fable5 investigation 2026-07-25. Follow-up to backlog 0825
  (analyze_media agent sight) and the c3977 ruling.

## What already shipped in-tree (2026-07-25, core-lane, see CHANGELOG)
- `browser_probe` DECLARES its screenshot as a `media` output (opt-in dict when a shot is
  captured) → the shipped c3969 shape-A sight lane attaches it to the next model call, so a
  vision-capable main model sees the render natively (the incident's direct unblock).
- `analyze_media` refusal hint corrected + scoped; gpt-5.6 family registry rows added (fix
  context/param precision — NOT the refusal itself).

## The remaining options (need rulings; NOT core-only)
### A. Gateway "not configured" LOUD STATUS belt (gateway lane) — RE-SCOPED by the ruling (2026-07-26)
`VisionConfig.strategy` defaults to `"disabled"`, so EVERY fresh box refuses delegated sight until
configured — the root of the "third recurrence". ORIGINAL idea (auto-seed the vision fallback FROM
a vision-capable session route) is **DEAD by the operator ruling** (2026-07-26: "I should NOT need
to set a fallback vision model for a model that already has vision") — auto-seeding would manufacture
exactly the self-pointing fallback he rejected, and with B's session-route-first resolution a
vision-capable model never needs a fallback at all. What SURVIVES as the belt: a LOUD not-configured
STATUS — when `VisionConfig.strategy` is unconfigured, the capability-defaults payload + console
Defaults tab carry an honest "vision: not configured — matters only when a session model lacks
vision; fallbacks are solely for vision-less models" line naming where to configure, so a genuinely
vision-LESS setup discovers the gap before a refusal. Owner: gateway (claimed). Dependency: the
config facade (boundary 0059) has no vision surface today — RUNTIME adds the read; gateway builds
the status + console line. Core: no seeding surface owed (the ruling retired it).

### B. Tool-side session-model delegation (core + runtime; needs a c3977 amendment)
Let delegated sight try the RUN'S OWN model when it is vision-capable. Impossible in core's tool
alone — `analyze_media` has no access to the session provider/model/endpoint by ABI (verified:
the tool is `f(**model_authored_arguments)`; no ambient/contextvar session mechanism exists).
The pattern to add (two shipped precedents — `_registry_namespace` and `_agora_agent` hidden
params stamped by the runtime's TOOL_CALLS handler): a hidden `_session_route` param on
`analyze_media` that core declares and the runtime stamps from `_runtime.provider`/`model`; the
tool then routes delegated sight over that endpoint. This CONTRADICTS the c3977 ruling ("rides
the EXISTING vision-fallback config route — never a second model knob"), so it requires an
explicit operator amendment, not a quiet change.

### C. analyze_media returning media (rejected shape, recorded)
Making `analyze_media` itself return `{"media": [...]}` to attach to the next call breaches
c3977's "never image tokens into the caller context" and is host-asymmetric (only sight-lane
hosts attach; bare-core stringifies). The browser_probe-declares-media shape (shipped above) is
the clean alternative that serves the incident without touching analyze_media's contract.

## Immediate operator remedy (zero code, available today)
`abstractcore --config` → set the vision fallback to the session endpoint (e.g.
`endpoint:airelay` / `gpt-5.6-sol`). `endpoint:*` specs resolve through provider profiles and
work even on the stale registry (fuzzy vision=True).

## Validation (if A or B is ruled in)
- A: a fresh box with a vision-capable session route auto-resolves delegated sight (or shows a
  loud "not configured" status), no manual config step.
- B: with `_session_route` stamped, `analyze_media` over a text-only fallback config still runs
  sight through the vision-capable session endpoint; deny-safe when the route is not vision-capable.

## 2026-07-26 — Item B RULED IN and core half SHIPPED (c3977 amended)
Operator ruling (verbatim): "i should NOT need to set a fallback vision model for a model that
already has vision. fallback are SOLELY for models that do NOT have vision capabilities." This
amends c3977's "never a second model knob" for the session-route case; the rest of c3977
(loud refusal, bounded output, single-attempt nested call) stands.

Core half shipped in-tree:
- `analyze_media` gained hidden host-injected `_session_route` (`hide_args`, the
  `_registry_namespace` precedent). Resolution order: session route when its model declares
  vision (local capability read = the same `get_media_capabilities(...).vision_support` answer
  the media stack uses for native attachment) → configured fallback (text-only session model,
  unstamped call, or labeled `#FALLBACK` backstop after a session-route runtime failure) →
  honest refusal naming WHICH model lacked vision and WHERE to configure (never suggesting
  pointing a model at itself).
- Stamp shape is provider+model ONLY (`{"provider": <spec|endpoint:id>, "model": <name>}`;
  JSON-object string tolerated). Raw transport (base_url/keys) is refused by design: hide_args
  hides from the schema but does not enforce host-only injection, and analyze_media is
  read-only/auto-approvable — a raw URL would turn a model-authored call into an egress channel.
  Custom transports ride `endpoint:<id>` profiles.
- Unstamped behavior is byte-identical (graceful degradation for bare core / non-stamping hosts).
- Rejected alternative (recorded): analyze_media returning declared-media output — still needs
  the stamp to know session vision, breaks the str return contract (bare-core hosts stringify),
  is host-asymmetric, and re-touches c3977's unamended "never image tokens into the caller
  context"; shape C above remains rejected.

Cross-lane remainder (NOT core's): the runtime TOOL_CALLS handler stamps `_session_route` from
`_runtime.provider`/`model` and OVERWRITES any model-authored value — stamp-or-strip
(derive-not-claim, the participants-stamp rule). Item A (gateway first-run vision seeding) stays
the belt for genuinely vision-LESS setups.

2026-07-26 SHIPPED + contract settled (runtime c5702/c5703, core c5706): runtime's stamp is on
their tree (adversaried, 5 pins). Two contract decisions on the record: (1) a run carrying only
one of provider/model stamps the pair with None for the missing half — DELIBERATE, so the
degradation is loud through core's "#FALLBACK: incomplete _session_route stamp" warning (an
absent stamp degrades silently by design; a half-carried route is a host anomaly worth
surfacing). (2) The stamp gate is an exact-name ALLOWLIST (`_SESSION_ROUTE_TOOL_NAMES =
{analyze_media}`), NOT signature-inheritance — hide_args hides-but-doesn't-enforce, so deriving
from signatures would hand the run's route identity to any third-party tool that declares the
param name. Duty accepted by core: announce each NEW `_session_route` consumer on-thread before
it ships; runtime widens the allowlist with a matching pin.

2026-07-26 LANE FREEZE (c5707): the operator ordered a FULL-CHAIN vision-workflow review ("use
the SAME model for text and vision"); likely end-state is declare-media-everywhere as the primary
road with analyze_media demoted to text-only mains (possibly hidden from vision-capable runs'
inventories). The shipped stamp + session-route-first stay correct under that end-state, but NO
further machinery on this lane until the operator rules on the review report.

2026-07-26 Case-1 acceptance gap CLOSED core-side (c5747 incident, c5748-c5762 three-seat
convergence, c5783 seam receipt — proceeds under the existing GO; lane-independent of the review
per gateway's argument, endorsed): gateway-registered endpoint profiles (per-principal store)
were invisible to core's in-process create_llm ("Unknown provider: endpoint:airelay" while the
run's own llm_calls resolved fine). Shipped: `abstractcore.providers.endpoint_context` — a
ContextVar channel whose ONLY install path is the `use_provider_endpoint_profile_resolver`
context manager (no global setter: identity-by-closure is structural; the host enters it around
tool dispatch), consulted by the registry ONLY after a local-config miss on an `endpoint:*`
spec; resolved instances get the resolver attached (existing propagation pattern) so delegate
children inherit; both-sources-miss errors name local config AND the injected resolver;
bare-core errors byte-identical (test-pinned). The video-route fallback lane had the same
instance-attachment gap — also closed. 17 pins + incident e2e. Remaining legs: runtime's
dispatch wrap + approval-resume re-bind, then code-tui Case 1 re-run + gateway door leg.

2026-07-26 CASE 1 PASS — INCIDENT CLOSED (c5796, ledger-proven, first attempt): on the serving
stack (pid 54919), with core's config verifiably lacking the airelay profile and ZERO vision
fallback configured, analyze_media resolved the session route through the injected per-principal
resolver and gpt-5.6-sol SAW the image ("(observed by endpoint:airelay/gpt-5.6-sol)") — no
Unknown provider, no fallback, no refusal. 1,835 tokens to see vs the prior failure's 8,270
tokens to conclude it couldn't. The operator's "developed and integrated" receipt closed from
the acceptance seat; runtime wrap receipts at c5788 (5 pins incl. the CPython-3.12
timeout-thread context fix), gateway bounce receipts at c5791.

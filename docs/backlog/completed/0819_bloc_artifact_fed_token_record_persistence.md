# Bloc artifacts lose the fed-token-id record at save — persist it and admit artifacts into the delta lattice

## Metadata
- Created: 2026-07-13
- Completed: 2026-07-14
- Status: COMPLETED (fable5 adversary P0-1 on the bloc-composability review,
  2026-07-13; gated the runtime seam — "fix first, before any seam design")
- Area: providers/mlx_provider.py (prompt_cache_save/load, delta lattice),
  core/bloc_kv.py (compile-lane meta)

## Completion summary (2026-07-14)
- `prompt_cache_save` persists the store's fed-token-id record into the
  artifact metadata (JSON string under the safetensors string-value
  constraint); `prompt_cache_load` parses it back to a real int list and
  VERIFIES admission — a record longer than the loaded cache is dropped
  loudly (#FALLBACK, protective bypass preserved); shorter records are
  legitimate (freeze invariant) and the trim arithmetic handles the tail.
- Artifact-backed keys WITH a true record now join the full-context delta
  lattice; one artifact-only protection added: a DIVERGENT prompt bypasses
  instead of trimming, so a single divergent call can never degrade a shared
  stable bloc cache. Record-less legacy artifacts keep the existing bypass
  (backfill = honest degradation; recompile mints records — reconstruct-at-
  load was deliberately NOT done without 0817's render fingerprint, since a
  reconstructed record without that check is the silently-wrong-cache class).
- The `prompt_cache` telemetry struct ships in `GenerateResponse.metadata`
  on the sync key lane (mode/key, outcome hit_full|hit_extend|cold|bypassed|
  rebuilt|append|off, MEASURED cached/fed token counts, bloc/artifact shas +
  binding_id when bound, degraded_reason with #FALLBACK) — runtime's
  non-negotiable condition. Streamed lane deliberately not covered (the
  runtime durable lane forces stream=False); follow-up if a consumer needs it.
- Validation: 15 unit pins (tests/providers/test_mlx_bloc_artifact_record_
  persistence.py) + 26 existing delta pins green; LIVE two-process check
  (scripts/bloc_artifact_delta_live_check.py, Qwen3-4B-Instruct-2507-4bit):
  artifact compiled with 578-id record -> FRESH process load -> full-context
  ask answered correctly with cached=578 fed=32 (94.8% of prefill skipped),
  telemetry outcome hit_extend with binding shas. Hybrid-architecture run
  (Qwen3.5-4B) verified the honest bypass telemetry ("not trimmable").
- Script gotcha for future live checks: `python scripts/foo.py` puts
  scripts/ first on sys.path — insert the repo root or the INSTALLED
  package is what runs (live incident: first run "disproved" the fix).

## ADR status
- Governing ADRs: `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md`
- ADR impact: none expected (restores intended behavior; the protective bypass
  stays for genuinely record-less artifacts)

## Context
Adversary finding (P0-1), verified against the tree: the fed-token-id record —
the exact token ids a cache encodes, the bookkeeping the whole delta lane rides
on — is store-meta only and is NEVER written into the artifact:

- the compile lane's `out_meta` (bloc_kv.py ~1130-1149) omits `fed_token_ids`;
- `prompt_cache_save` (mlx_provider.py ~919-957) omits it from safetensors meta;
- `prompt_cache_load` (~1028-1038) does not reconstruct it.

Consequence chain: every artifact-backed cache is "warm-unknown"; MLX's
full-context lane deliberately BYPASSES artifact-backed caches without a record
(mlx_provider.py ~496-505, a protective #FALLBACK from the delta wave) rather
than risk a wrong generation. The runtime's calling convention is full-context
(`messages=` every call), so a runtime bloc integration today produces load
cost + RAM and then a FULL re-prefill — negative value. The ids are known at
compile time (the tmp-key update records them into store meta via
`prompt_cache_update`) — they are simply dropped at the save boundary.
Alternatively they are recomputable at load by encoding the manifest's
`serialized_prompt`.

## Scope
- Persist the fed-token-id record into artifact metadata at `prompt_cache_save`
  (safetensors `__metadata__`; mind the string-keys/values constraint) AND/OR
  reconstruct it at `prompt_cache_load` from the manifest's serialized prompt
  (BOS-aware encode — reuse `_encode_prompt_token_ids`).
- Admit artifact-backed keys WITH a true record into the full-context delta
  lattice (LCP → trim → suffix-feed) instead of the blanket bypass; keep the
  bypass for record-less artifacts (older files) with the existing #FALLBACK.
- The loaded record must survive the binding-verification path unchanged
  (durable-bloc binding meta must not be disturbed — P1-5 lineage).
- Backfill posture for existing artifacts: reconstruct-at-load covers them
  without rewriting 563 GB; state it.

## Non-goals
- Blend/composition math (0818 / the private branch).
- Changing artifact format or store layout.

## Dependencies
- None; unblocks the runtime seam (c1668) and is prerequisite to 0818.

## Expected outcomes
- A runtime-shaped full-context call over a loaded bloc artifact feeds ONLY the
  suffix (question) instead of re-prefilling the bloc — measured, not asserted.

## Seam verdict folded (runtime adversary c1734, 2026-07-13)
Runtime ruled the consume shape: hybrid C+A — core-internal opportunistic bloc
use behind `prompt_cache_key` is the v1 runtime lane (BLOC_ENSURE effect
REJECTED: crash-replay reuses completed results, so an ensure postcondition is
not re-established on the paths that matter; the ADR-0006 binding lane stays
the explicit host-managed strict lane, never auto-attached to durable
payloads). NON-NEGOTIABLE CONDITION riding this item: a `prompt_cache`
telemetry struct in GenerateResponse metadata — mode/key, outcome
(hit_full | hit_extend | cold | bypassed | rebuilt), MEASURED cached/fed token
counts, bloc+artifact shas when bound, degraded_reason with #FALLBACK — so the
runtime ledger can explain 90s-vs-2s turns (today the protective bypass is
observable only as a log line). Also theirs, respected here: KV-as-source-of-
truth call shapes stay banned on the durable lane (text always rides; caches
change latency, never content), and composition order must be a pure function
of the durable rendered prompt.

## Validation
- Unit: save→load round-trip preserves/reconstructs the record; artifact-backed
  key joins the delta lattice (fed count == suffix, not full prompt); record-less
  artifact still bypasses with #FALLBACK; binding meta untouched.
- Live (MLX): compile a real bloc, reload in a FRESH process, ask a question
  over it full-context; assert prefill tokens ≈ suffix length and answer quality
  unchanged.

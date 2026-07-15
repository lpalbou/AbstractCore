# KV-artifact invalidation-key audit — enumerate the validity key, close the silently-wrong-cache class

## Metadata
- Created: 2026-07-13
- Status: PLANNED (operator mandate, 2026-07-13 21:54: "create backlog items … to
  evaluate this" — the invalidation trust gap named in the bloc-composability
  discussion)
- Area: core/file_blocs.py, core/bloc_kv.py, providers/base.py (prompt-cache store
  meta), providers/mlx_provider.py (prompt_cache_save/load)

## ADR status
- Governing ADRs: `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md`
  (a silently-wrong cache is the exact class ADR-0001 forbids)
- ADR impact: likely a new ADR ("KV artifact validity key") once the audit fixes the
  key's members — the key is durable cross-task policy, not item-local detail.

## Context
The bloc store pairs content-addressed text with per-(provider, model) KV
artifacts (`files/<sha256>/kv/<slug>.safetensors`). Loading one of these caches is
only correct when EVERYTHING that shaped the original prefill still holds. The
full validity key is at least:

1. model identity (resolved id, not the alias used at save time),
2. quantization/dtype of the loaded model (a 4-bit and an 8-bit load of the same
   repo produce incompatible caches),
3. provider + engine version (mlx_lm / llama.cpp / transformers serialization and
   cache layouts change across versions),
4. tokenizer + chat-template fingerprint (a template update silently changes the
   token stream the cache encodes),
5. position policy (what offset the bloc was prefilled at; RoPE-dependent),
6. attention architecture (SSM/hybrid layers are untrimmable/uncountable — see
   mlx_provider delta-lane comments — and cannot participate in composition),
7. context-window configuration where it changes attention (sliding windows).

What exists today (verified 2026-07-13): safetensors headers carry
model/provider/token_count/saved_at (live artifact read); `bloc_kv` manifests add
a `binding_id` computed over a rendered recipe. What is NOT yet verified: whether
load paths CHECK each axis, which axes are absent from older artifacts (563 GB of
them exist), and whether any axis is missing entirely (tokenizer fingerprint and
engine version are the suspected gaps).

Adversary confirmation (fable5, 2026-07-13 — file:line receipts in the review):
manifests capture provider/model/resolved-id/backend/format/bloc+content+
rendered-prompt shas/recipe+renderer+serializer versions/artifact sha, and the
load lane cross-checks them properly. CONFIRMED MISSING, in danger order:
(1) tokenizer fingerprint — `serializer_version` is three coarse buckets and
`rendered_recipe_sha256` hashes TEXT not token ids, so a tokenizer.json refresh
under the same model id is accepted with wrong positions; (2) model-config hash
(rope_theta/rope_scaling/sliding-window edits leave no textual trace);
(3) engine version (mlx_lm cache layout changes across versions — the code
already version-guards `CacheList`); (4) cache dtype is HARDCODED "fp" and
validation rejects anything else — which also makes q8 cache quantization
unreachable through the bloc lane (~2x storage on 563 GB left on the table);
(5) position offset implicit 0 everywhere. Adjacent confirmed defects to fold
into the same pass: `has_kv_cache` is a bare Path.exists() — existence is NOT
validity (re-extraction leaves a stale artifact behind a True answer; the
manifest-checked lane exists but the obvious API name lies); artifact paths slug
the AS-GIVEN model id, so alias spellings duplicate GB-scale artifacts (key by
resolved id); `_validate_existing_manifest` full-hashes the artifact per ensure
(~1-3 s at 1.5 GB) — the framework probe needs a declared cheap tier
(manifest + size/mtime) with full hash at load only.

Runtime's independent adversary (c1734, 2026-07-13) added one axis: the
manifest pins model NAME strings but not WEIGHTS identity — a swapped
checkpoint under the same id (revision update, re-quantization in place)
silently accepts stale KV. Fold a weights-identity signal (HF revision /
snapshot hash / weights-file digest where cheap) into the audited key.
(The same review's torn-write finding — plain write_text on content.txt
letting a concurrent compiler manifest the WRONG text — was FIXED same-day:
all bloc text/meta writes now go through unique-tmp + os.replace, pinned in
tests/test_file_bloc_store_unit.py.)

## Audit gap matrix (verified against code 2026-07-14)

Verified against `_validate_existing_manifest` (bloc_kv.py), the mlx
`prompt_cache_load` model check, and `_loaded_cache_matches_manifest`.
Legend: ✅ captured+checked · ⚠️ captured, not fully validity-safe · ❌ absent.

| Validity axis | Status | Where / gap |
| --- | --- | --- |
| model identity (alias) | ✅ | `manifest.model` checked; load-time id match |
| resolved model id | ✅ | `manifest.model_resolved_id` checked |
| provider + backend/format | ✅ | provider/cache_backend/artifact_format checked |
| bloc + content sha | ✅ | `bloc_sha256`/`content_sha256` checked |
| rendered recipe (TEXT) | ⚠️ | `rendered_recipe_sha256` hashes TEXT, not token ids |
| recipe/renderer/serializer ver | ⚠️ | coarse buckets; a tokenizer.json refresh under one serializer_version is invisible |
| artifact sha / binding id | ✅ | full-file hash + binding recomputation |
| **engine + version** | ✅ **(0817 first axis, 2026-07-14)** | `engine_fingerprint` recorded at compile; mismatch REFUSES+recompiles, absent (pre-0817) reused with #FALLBACK |
| tokenizer/chat-template fingerprint | ❌ | NEXT axis — token-id-level, not TEXT-level; the #1 danger |
| model-config hash (rope/sliding) | ❌ | rope_theta/rope_scaling/window edits leave no trace |
| cache dtype | ❌ | hardcoded "fp"; anything else rejected (also blocks q8 storage win) |
| weights identity (revision/digest) | ❌ | swapped checkpoint under same id accepted (runtime c1734) |
| position offset | ❌ | implicit 0 everywhere |

Adjacent defects (same pass): `has_kv_cache` = bare `Path.exists()` (existence ≠
validity); artifact path slugs the AS-GIVEN model id (alias spellings duplicate
GB-scale artifacts — key by resolved id); `_validate_existing_manifest`
full-hashes per ensure (declare a cheap manifest+size/mtime tier, full hash at
load only).

## First axis SHIPPED (2026-07-14): engine + version identity

`BaseProvider.prompt_cache_engine_fingerprint()` (default "") → MLX returns
`mlx_lm==<version>` (live: `mlx_lm==0.31.3`), HF returns
`transformers==<version>` / `llama_cpp==<version>`. Recorded into the bloc-KV
manifest (`engine_fingerprint`) AND the artifact safetensors meta at compile.
`_validate_existing_manifest` gate: recorded ≠ current → REFUSE (return None →
recompile, `#FALLBACK` logged); recorded absent (pre-0817 corpus) → reuse
UNVERIFIED with a `#FALLBACK` (no corpus-wide invalidation, never silent).
DELIBERATELY NOT in `binding_id` (adding it would change every pre-0817
manifest's recomputed binding and reject the existing corpus — backfill
safety; a future manifest-version bump folds it in once recompiled). 7 pins in
`tests/test_bloc_kv_engine_fingerprint.py`; 23 existing bloc_kv pins green.

Remaining axes staged in danger order: tokenizer/template fingerprint (token-id
level) → model-config hash → weights identity → cache dtype (also unlocks q8) →
position offset. Each lands as its own refuse-loudly gate with a mismatch pin.

## Scope
- Enumerate the axes above against the actual save/load code paths
  (`prompt_cache_save`/`prompt_cache_load`, `bloc_kv` compile/load/bind,
  `FileBlocStore.kv_cache_path` keying) and produce a gap matrix:
  captured-and-checked / captured-not-checked / not-captured.
- Adversarial tests per axis: attempt to load an artifact under a mismatched
  model, quant, template, engine version — each must REFUSE loudly or the gap is
  documented as a finding (never silently generate from a wrong cache).
- Decide the sha's subject: the current key is sha256 of the SOURCE FILE; evaluate
  whether (extracted_text, extraction_version) should be the content key so
  extractor upgrades invalidate correctly.
- Backfill story for the existing 563 GB: which artifacts can be revalidated from
  headers, which must be treated as unverifiable (refuse + recompile on demand).

## Non-goals
- Building the composability/link lane itself (backlog 0818 + the private branch).
- Changing the store layout.

## Dependencies
- None blocking; informs and gates 0818.

## Expected outcomes
- A verified gap matrix in this item's completion report.
- Refusal (not silence) on every mismatched-load path, with tests.
- The validity key promoted to an ADR if the audit confirms new mandatory axes.

## Validation
- Per-axis mismatch tests (wrong model / wrong quant / mutated template / bumped
  engine version) all refuse with named reasons.
- A held-out valid artifact still loads and generates (no false refusals).

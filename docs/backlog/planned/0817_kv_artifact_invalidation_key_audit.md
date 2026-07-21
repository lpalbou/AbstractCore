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
| **tokenizer/chat-template fingerprint** | ✅ **(0817 axis 2, 2026-07-18)** | `tokenizer_fingerprint` (tokenizer STATE + chat template + special ids) recorded at compile + in artifact meta; ensure-gate recompiles on mismatch, MLX load-gate raises with the tokenizer live; absent → labeled #FALLBACK; unloaded current → abstain (load-gate re-checks) |
| **model-config hash (rope/sliding)** | ✅ **(0817 axis 3, 2026-07-20)** | `model_config_fingerprint` (curated rope/window/position geometry keys, top-level + `text_config`) recorded at compile + in artifact meta; ensure-gate recompiles on mismatch, MLX load-gate raises with the model live; irrelevant config churn (versions/dtype/names) deliberately outside the subject; absent → labeled #FALLBACK; unloaded current → abstain |
| **weights identity (revision/digest)** | ✅ **(0817 axis 4, 2026-07-20)** | `weights_fingerprint` (tiered cheap subjects: hub revision / fileset+safetensors-header digests / GGUF header slice) recorded at compile + in artifact meta; ensure-gate recompiles on mismatch, MLX load-gate raises; GGUF does NOT abstain (this axis owns the in-file tokenizer/config signals axes 2-3 deferred); full-content hashing deliberately out of scope |
| **cache dtype** | ✅ **(0817 axis 5, 2026-07-21)** | `quantization` request param on ensure/load (+ HTTP proxies): "fp"/"q8"; stored≠requested recompiles AT the requested dtype (#FALLBACK); unknown stored dtype refuses (never guess a layout from another build); q8 against a writer without an explicit `q8` param raises (silent fp-under-q8-label kill); pre-axis manifests read as "fp" — corpus stays valid |
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

## Axis 2 SHIPPED (2026-07-18): tokenizer + chat-template identity

`BaseProvider.prompt_cache_tokenizer_fingerprint()` (default "") → MLX and
HF-transformers fingerprint the LOADED tokenizer via
`providers/tokenizer_fingerprint.py`: sha256 over the fast tokenizer's
COMPLETE serialized state (`backend_tokenizer.to_str()` — vocab, merges,
normalizers: everything that maps text→ids) + the chat template text (it
lives outside that state and is the most-refreshed piece) + special-token
ids, rendered `tokenizer-full:sha256:<24hex>` (weaker `tokenizer-vocab:`
tier for slow tokenizers; tier prefix makes cross-tier compares fail-safe
mismatches). GGUF deliberately returns "" — its tokenizer travels INSIDE the
weights file, so the weights-identity axis owns that signal. Live-verified:
real HF fast tokenizer stable + template-mutation flips it; mlx_lm
TokenizerWrapper fingerprint == inner tokenizer fingerprint.

Gate points (three-way verdict shared via `check_tokenizer_fingerprint`):
`_validate_existing_manifest` (ensure-time: mismatch → refuse+recompile
`#FALLBACK`; absent-with-current-known → reuse UNVERIFIED `#FALLBACK`;
current-unavailable → ABSTAIN silently, because ensure may run before the
model loads and warning every pre-load ensure is noise) and MLX
`prompt_cache_load` (load-time, tokenizer live: mismatch → loud ValueError
naming both fingerprints + the recompile fix; unverified branches warn
`#FALLBACK`). MLX `prompt_cache_save` records the fingerprint into EVERY
artifact's metadata (bloc lane passes it explicitly in the manifest +
out_meta). NOT part of `binding_id` (same backfill-safety rule as the engine
axis). 9 pins in `tests/providers/test_tokenizer_fingerprint_unit.py` + 9 in
`tests/test_bloc_kv_tokenizer_fingerprint.py`.

## Axis 3 SHIPPED (2026-07-20): model-config KV-geometry identity

`BaseProvider.prompt_cache_model_config_fingerprint()` (default "") → MLX
fingerprints the loaded model's `args` (mlx_lm ModelArgs, built FROM
config.json) with a config.json fallback beside the resolved model dir;
HF-transformers fingerprints the loaded `model_instance.config`
(PretrainedConfig). GGUF deliberately returns "" — its config travels INSIDE
the weights file (weights-identity axis owns that signal). Subject
(`providers/model_config_fingerprint.py`): a CURATED set of geometry keys,
never the whole config — RoPE family (`rope_theta`, `rope_scaling`,
`rope_parameters`, `partial_rotary_factor`, `rotary_dim`, `rotary_emb_base`,
`rope_local_base_freq`, `rope_traditional`), window/attention layout
(`sliding_window`, `use_sliding_window`, `sliding_window_pattern`,
`layer_types`, `attention_chunk_size`, `global_attention_every_n_layers`),
positional envelope (`max_position_embeddings`,
`original_max_position_embeddings`, `position_embedding_type`), plus
`model_type`, top-level AND under `text_config` (multimodal nesting) —
because hashing the whole config would false-invalidate the corpus on
irrelevant churn (`transformers_version` bumps, dtype/name metadata).
Rendered `model-config:sha256:<24hex>`; a reachable config with zero
geometry keys hashes as a stable empty-geometry constant (changes the moment
a geometry key APPEARS — adding `rope_scaling` is exactly an edit to catch);
"" reserved for config-unreachable (abstain). Live-verified on a real cached
config (rope edit flips, version churn stable).

Gate points (verdict shared with axis 2 via `check_model_config_fingerprint`):
`_validate_existing_manifest` (ensure-time: mismatch → refuse+recompile
`#FALLBACK`; absent-with-current-known → reuse UNVERIFIED `#FALLBACK`;
current-unavailable → abstain) and MLX `prompt_cache_load` (load-time, model
live: mismatch → loud ValueError naming both fingerprints + the recompile
fix). MLX `prompt_cache_save` records it into every artifact's metadata;
the bloc lane writes it into the manifest + out_meta. NOT part of
`binding_id` (backfill safety). 10 pins in
`tests/providers/test_model_config_fingerprint_unit.py` + 8 in
`tests/test_bloc_kv_model_config_fingerprint.py`; full suite 2232 green.

## Axis 4 SHIPPED (2026-07-20): weights identity

`BaseProvider.prompt_cache_weights_fingerprint()` (default "") with TIERED
cheap subjects in `providers/weights_fingerprint.py` — full-content hashing
of multi-GB weights is deliberately out of scope: (1)
`weights-revision:<sha>` when the hub commit is observable (transformers
`config._commit_hash`, or the `snapshots/<sha>` segment of a cache-resolved
model dir — content-addressed upstream, zero I/O); (2)
`weights-fileset:sha256:` for other local dirs — sorted (relative name,
size) of every weight file plus the sha256 of each safetensors HEADER
(bounded read; re-quantization and re-sharding move it even when sizes
collide; honest limit stated: a same-size same-header pure value edit is
invisible without full hashing); (3) `weights-gguf:<size>:<slice-hash>` for
single GGUF files — size + first-MiB digest (magic/version/tensor
table/metadata live there; a re-quant rewrites it). Cross-tier compares are
unequal → fail-safe mismatch → one recompile records the stronger tier.

GGUF deliberately does NOT abstain on this axis — it is the axis the
tokenizer (axis 2) and config (axis 3) gates deferred to, since both travel
INSIDE the GGUF file. MLX fingerprints its resolved model dir; HF prefers
the loaded config's `_commit_hash`, then `_name_or_path`/local snapshot
dirs. Gate points identical to axes 2-3 (shared verdict): bloc ensure-gate
refuses+recompiles on mismatch, MLX `prompt_cache_load` raises loudly,
pre-axis artifacts reuse LABELED, unavailable current abstains; recorded in
manifest + every MLX artifact's meta; NOT part of `binding_id`. 9 pins in
`tests/providers/test_weights_fingerprint_unit.py` + 8 in
`tests/test_bloc_kv_weights_fingerprint.py`; full suite 2248 green.
Live-verified: a real HF snapshot dir resolves its actual commit sha; a
real 4.9 GB GGUF fingerprints in 2.5 ms (cheapness proven).

## Axis 5 SHIPPED (2026-07-21): cache dtype (q8 unlocked)

The bloc lane hardcoded `quantization="fp"` at both write sites and REJECTED
any other value at validation — MLX's provider-layer q8 support
(`to_quantized(group_size=64, bits=8)`) was unreachable from the durable
lane. Now: `ensure_bloc_kv_artifact`/`load_bloc_kv_artifact` take
`quantization` ("fp" default / "q8"), threaded through the HTTP proxies
(`/acore/blocs/kv/ensure` + `/load`). Gate semantics (deliberately DIFFERENT
from the fingerprint axes — dtype is a REQUEST property, not a drift
signal): stored ≠ requested → recompile AT the requested dtype with a
labeled `#FALLBACK` (the request is authoritative; request q8 once and the
artifact converts, then reuses under q8); stored dtype UNKNOWN to this build
→ refuse + recompile (a manifest from a newer/older writer — never guess a
tensor layout); requested q8 against a provider whose `prompt_cache_save`
does not DECLARE a `q8` parameter → loud ValueError up front (a bare
`**kwargs` writer would silently store fp under a manifest claiming q8 —
the silent-wrong-label class); pre-axis manifests (None/"") read as "fp",
so the existing corpus stays valid under default requests. `quantization`
stays IN `binding_id` (unlike the fingerprints): the dtype is part of the
artifact's identity, and fp/q8 artifacts of one bloc are distinct bindings.
8 pins in `tests/test_bloc_kv_cache_dtype.py`; full suite 2256 green.

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

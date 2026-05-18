# Planned: Memory bloc MLX KV compiler and loader

## Metadata
- Created: 2026-05-18
- Status: Planned
- Completed: N/A

## Context

AbstractCore already has most of the primitives needed for durable per-model MLX KV artifacts:

- `FileBlocStore` persists extracted file text snapshots and reserves `kv/` paths for derived
  per-model artifacts.
- `MLXProvider` supports local prompt-cache control-plane operations:
  - `set`
  - `update`
  - `fork`
  - `prepare_modules`
  - `save`
  - `load`
- `CachedSession` already uses ordered prompt-cache modules for `system` and `tools`, then appends
  file boxes into the live cache.
- `generate_bloc_metadata_jsonld(...)` already has ad hoc logic to load or fork a bloc-derived KV
  artifact before asking follow-up questions.

The missing piece is a first-class compiler/loader layer that can:

1. take a persisted bloc,
2. build an MLX KV artifact for a specific model,
3. save it under the bloc's `kv/` path,
4. reload or fork it safely into providers or future shared runtimes later.

This item is intentionally about a durable **provider-level bloc artifact layer**. It is not yet a
full session bootstrap story.

## Current Code Reality

- `docs/memory-blocs.md` explicitly says `FileBlocStore` is storage-only; higher-level code must
  compile and consume KV artifacts itself.
- `FileBlocStore.kv_cache_path(...)` currently provides one artifact path per `(provider, model)`.
- `FileBlocStore` does not yet expose a first-class manifest path helper for neighboring metadata.
- `MLXProvider` can save and load prompt caches as `.safetensors`, and they are model-locked.
- `CachedSession` uses `PromptCacheModule` / `prompt_cache_prepare_modules(...)` for ordered
  `system` and `tools` prefixes, and only then appends file boxes into the cache.
- `render_file_box_message(...)` includes the persisted file path in the rendered prompt, so a bloc
  artifact depends on more than `content.txt` alone.
- `MLXProvider` wraps prompt fragments with model-specific serialization, including Qwen-specific
  formatting.
- `generate_bloc_metadata_jsonld(...)` currently proves only a narrow workflow:
  load or fork a bloc cache, then append later instructions on top of that cache.

There are several constraints the design has to respect:

1. KV caches are exact prefix artifacts, not arbitrary reusable blocks that can be merged in any
   order.
2. A bloc-only persisted artifact is valid for provider-level file-first flows, but it is not the
   canonical prompt boundary for `CachedSession`.
3. Load does not restore transcript state. Today, cache state and transcript state are separate
   concerns.
4. Provider caches are in-process entries in an evictable store, not durable resident handles.

## Problem

Without a dedicated compiler/loader layer:

- memory blocs can reserve a `kv/` location, but nothing reliably builds those artifacts;
- every consumer would need to reinvent:
  - cache key lifecycle;
  - artifact naming;
  - manifest pathing;
  - stale detection;
  - temp-file cleanup;
  - provider capability checks;
  - load vs fork decisions;
  - reload-on-miss behavior;
- `generate_bloc_metadata_jsonld(...)` keeps its own ad hoc orchestration instead of using shared
  helpers;
- users cannot rely on “1 bloc = 1 durable provider artifact per model.”

There is also a correctness risk: a bloc artifact is valid only for the exact rendered recipe that
produced it. `content_sha256` alone is not enough because the rendered prompt also depends on:

- the path string embedded in the file-box prompt;
- newline normalization / renderer behavior;
- model-specific serialization in `MLXProvider`;
- any future recipe/version changes.

## What We Want To Do

Add a dedicated MLX-first compiler/loader layer for **bloc-only persisted KV artifacts**.

The first version should support one canonical persisted recipe only:

- compile one persisted bloc into one durable MLX artifact per model;
- persist the artifact plus manifest under the bloc's `kv/` area;
- load or fork that artifact into a provider cache key later;
- let provider-level file-first workflows reuse the bloc context without re-prefilling the full
  text each run.

Important boundary:

- v1 is for **bloc-only persisted artifacts**
- v1 is **not** a general `system + tools + bloc + transcript` composition layer
- v1 is **not** a complete `CachedSession` bootstrap story

This item is also separate from continuous batching, but it should remain compatible with the
planned shared MLX runtime in:

- `docs/backlog/planned/2026-05-18_mlx-provider-continuous-batching.md`

## Why

- Large file contexts are expensive to prefill repeatedly.
- Memory blocs already provide durable extracted text snapshots and stable selectors (`bloc_id`).
- MLX already exposes the save/load primitives needed to make those snapshots reusable.
- A shared compiler/loader layer turns a storage convention into an actual feature.

## Requirements

- The first implementation must target `MLXProvider`.
- Compilation must operate from persisted bloc data (`FileBlocStore`, `content.txt`, `meta.json`),
  not require the original source file to still exist.
- The compiler must define one exact, versioned persisted recipe for v1.
- The recipe must align with existing prompt-cache boundaries:
  - represent the bloc prompt as one ordered module-like unit rather than inventing an unrelated
    prompt abstraction;
  - reuse existing file-box rendering semantics deliberately.
- The artifact must be treated as valid only for:
  - one provider family (`mlx`);
  - one model id / resolved load target;
  - one prompt recipe id/version;
  - one exact rendered recipe hash;
  - one serialized provider formatter version;
  - one bloc snapshot.
- The compiler must write:
  - the KV artifact under the bloc's `kv/` path;
  - a first-class manifest sidecar path;
  - both files with a manifest-last commit protocol:
    - temp-write artifact
    - temp-write manifest
    - rename artifact into place
    - rename manifest into place as the commit marker
- Readers must treat “artifact without valid manifest/hash” as incomplete and rebuildable.
- The manifest must include enough data to detect stale or mismatched artifacts without loading the
  full `.safetensors`.
- The loader must support:
  - load into a requested cache key;
  - fork into a working key;
  - reload-on-miss when a previously loaded key was evicted;
  - cleanup of temporary keys on failure.
- Compile/load/fork helpers must not change the caller's default cache key unless explicitly
  requested.
- Resolution by both `sha256` and `bloc_id` should be supported.
- Existing code such as `generate_bloc_metadata_jsonld(...)` must be able to reuse the new layer.
- The design must clearly state that v1 solves provider-level bloc reuse, not transcript bootstrap.

## Canonical v1 Persisted Recipe

V1 should use one canonical persisted recipe only:

- `attached_file_box_v1`

Recipe shape:

1. read bloc `content.txt` plus persisted metadata needed for prompt rendering
2. reconstruct one deterministic file-box prompt using the same stable rendering concept already
   used by `CachedSession`
3. represent that prompt as one ordered module-like unit containing a single user message
4. append that unit into an empty MLX prompt cache
5. save the resulting cache artifact

Important implications:

- this produces a **bloc-only persisted artifact**
- it does **not** mean the artifact can be spliced into an unrelated existing cache prefix
- it does **not** preserve `system -> tools -> bloc` ordering from `CachedSession`
- in current KV flows, any later `system_prompt` or tools instructions are appended after the
  loaded bloc artifact unless a future compiled-context recipe solves that ordering explicitly

That keeps the path convention simple and makes “1 bloc = 1 persisted artifact per model”
coherent without pretending arbitrary cache composition is solved.

## Proposed Architecture

### 1. Add a dedicated bloc KV module

Suggested file:

- `abstractcore/core/bloc_kv.py`

Suggested concepts:

- `BlocKVRecipe`
- `BlocKVArtifactManifest`
- `BlocKVCompileResult`
- `BlocKVLoadResult`
- `BlocKVManager` or equivalent module-level helpers

The module should own:

- bloc resolution by `sha256` or `bloc_id`
- canonical prompt rendering for the selected persisted recipe
- provider capability checks
- compile/save/load/fork orchestration
- manifest validation
- reload-on-miss behavior for evicted loaded keys

### 2. Add first-class store helpers for artifact neighbors

`FileBlocStore` should gain explicit helpers rather than relying on implicit neighbor filenames.

Suggested additions:

- `kv_cache_manifest_path(...)`
- optional `has_kv_cache_manifest(...)`

These helpers should define the v1 on-disk contract clearly:

- one unquantized `.safetensors` artifact per `(provider, model)`
- one neighboring manifest file for that artifact

Quantized variants such as MLX q8 saves are out of scope for v1. The pathing would collide unless
variant naming is extended explicitly.

### 3. Define artifact metadata explicitly

The artifact needs explicit validation metadata beyond the model string.

Suggested manifest fields:

- `version`
- `provider`
- `model`
- `model_resolved_id`
- `bloc_sha256`
- `bloc_id`
- `content_sha256`
- `path_in_prompt`
- `recipe_id`
- `recipe_version`
- `rendered_recipe_sha256`
- `renderer_version`
- `serializer_version`
- `artifact_filename`
- `artifact_sha256`
- `quantization`
- `created_at`
- `token_count`

V1 should store this in a JSON sidecar next to the `.safetensors` artifact.

Where possible, the same metadata should also be embedded in the MLX
`prompt_cache_save(...)` metadata block, but the sidecar is the primary cheap validation surface.

### 4. Keep the path convention simple for v1

`FileBlocStore.kv_cache_path(...)` currently gives one artifact path per `(provider, model)`.

That is acceptable for v1 because:

- there is exactly one canonical persisted recipe;
- the artifact is bloc-only;
- we are not trying to persist multiple variants per model yet.

If future work needs multiple recipes, quantizations, or serializer variants per model, pathing
must be extended explicitly.

### 5. Implement compile flow

Suggested compile flow:

1. resolve the bloc record and content from `FileBlocStore`
2. reconstruct the exact prompt payload and compute its rendered hash
3. verify the provider supports:
   - prompt caching
   - local control plane
   - save/load
4. if `skip_if_fresh=True`, validate existing manifest + artifact and no-op if still current
5. create a temporary cache key
6. initialize an empty prompt cache for that key
7. append the canonical bloc-only recipe into that key
8. save the cache artifact to a temporary filename
9. compute artifact metadata / hash
10. write the manifest to a temporary filename
11. preserve the caller's default cache key unless explicit override was requested
12. rename artifact into place
13. rename manifest into place as the commit marker
14. clear the temporary key
15. return artifact metadata

The compiler should support:

- `force=True` to rebuild
- `skip_if_fresh=True` to no-op when the exact rendered recipe is still current

### 6. Implement load and fork flow

Suggested load flow:

1. resolve bloc, artifact, and manifest paths
2. validate manifest against:
   - provider/model
   - bloc snapshot
   - recipe id/version
   - rendered recipe hash
   - serializer/renderer versions
   - artifact hash when needed
3. load the artifact into the provider under a requested key
4. preserve the caller's default cache key unless explicit override was requested
5. return a load result that records the loaded key and validation data

Suggested fork flow:

1. if a stable loaded key was supplied, first check whether it is still present
2. if the stable key was evicted or never loaded, reload the artifact
3. fork from the loaded key into a working key
4. preserve the caller's default cache key unless explicit override was requested
5. return the working key

The wording here should stay precise: “stable key” means “preferred reuse key,” not “durable
resident key.”

### 7. Integrate existing consumers

The first real consumer should be `generate_bloc_metadata_jsonld(...)`, which already performs ad
hoc load/fork logic.

That integration must be documented honestly:

- it is a **bloc-first, instructions-later** workflow today
- it does not prove that bloc artifacts preserve the same ordering as a session prefix

It should use the new compiler/loader helpers instead of directly juggling:

- `prompt_cache_fork(...)`
- `prompt_cache_load(...)`
- temp cache keys

### 8. Scope session/runtime integration conservatively

V1 should not imply that `CachedSession` can simply hydrate itself from a bloc artifact.

What v1 can support cleanly:

- provider-level file-first workflows where the bloc artifact is the initial context
- explicit load/fork into a provider cache key before asking user questions
- future shared-runtime loading for the planned MLX batching runtime

What v1 should not promise:

- automatic insertion of a precompiled bloc cache into an already-prefilled unrelated session
  prefix
- transcript restoration from a loaded artifact
- attached-file dedupe / clear / discovery semantics derived from loaded cache state

If a workflow needs `system + tools + bloc` as one exact reusable prefix, or a transcript bootstrap
story for `CachedSession`, that should be a follow-up item.

## Suggested Implementation

1. Add `abstractcore/core/bloc_kv.py` with compile/load/fork helpers.
2. Add explicit manifest path helpers to `FileBlocStore`.
3. Add a canonical bloc-only prompt renderer for `attached_file_box_v1`.
4. Represent the recipe as one ordered module-like prompt unit aligned with current prompt-cache
   abstractions.
5. Add manifest sidecar support with manifest-last commit semantics.
6. Add compile orchestration using MLX prompt-cache control-plane methods.
7. Add load/fork helpers with validation, reload-on-miss, and temp-key cleanup.
8. Refactor `generate_bloc_metadata_jsonld(...)` to use the new layer.
9. Update docs with one explicit provider-level file-first artifact workflow.

## Testing Strategy

### Unit tests

Use fake prompt-cache providers to test orchestration without requiring Apple Silicon.

Required unit coverage:

- resolve bloc by `sha256`
- resolve bloc by `bloc_id`
- canonical recipe rendering is deterministic
- same content but different persisted path invalidates freshness
- rendered recipe hash changes when renderer inputs change
- compile writes artifact path and manifest sidecar
- compile uses manifest-last commit semantics
- interrupted compile between artifact rename and manifest rename recovers cleanly
- failed rebuild does not destroy an existing valid artifact
- stale detection via changed manifest inputs
- model mismatch rejection
- `model_resolved_id` drift invalidates reuse when recorded
- recipe mismatch rejection
- serializer/version mismatch rejection
- manifest/artifact mismatch rejection
- partial write handling
- load path chooses the requested key correctly
- fork path clones from the preferred loaded key
- reload-on-miss when loaded keys were evicted
- caller default cache key is preserved across compile/load/fork unless explicitly changed
- temp-key cleanup on compile failure
- temp-key cleanup on load/fork failure
- `generate_bloc_metadata_jsonld(...)` integration uses the new loader path

### Integration tests

Add gated real-MLX tests on Apple Silicon.

Required integration coverage:

- compile a real bloc artifact for a small MLX model
- load that artifact into a new provider instance
- ask a follow-up question using the loaded cache key
- save/load roundtrip remains model-locked and valid
- file-first query from loaded bloc cache produces a sane response
- Qwen and non-Qwen models both validate the serializer boundary correctly
- fresh provider vs already-loaded provider behavior is correct
- load vs load+fork vs fork-from-reloaded-key behaves as expected

### Concurrency and durability tests

Required coverage:

- concurrent compile attempts for the same bloc/model do not leave corrupt artifacts
- concurrent load/fork calls remain isolated by key
- eviction or TTL loss of a loaded key triggers safe reload behavior
- incomplete artifact-only state without committed manifest is treated as recoverable

### Benchmarking

Benchmarking is required because this feature exists to cut prefill cost.

Benchmark comparisons:

1. no cache:
   - load model
   - send full bloc content live

2. live prompt-cache prefill:
   - `prompt_cache_update(...)` from bloc content each run

3. precompiled bloc artifact:
   - `prompt_cache_load(...)` / fork
   - then ask the same question

Benchmark matrix:

- at least one small MLX model
- at least one mid-size MLX/Qwen model
- at least one medium and one large text bloc

Metrics:

- compile time
- artifact size on disk
- load/fork time
- TTFT for first question after load
- total wall time for first question
- amortized breakeven across repeated uses, including compile cost
- peak memory during compile/load if measurable

Acceptance gates:

- precompiled artifact path must materially reduce first-question TTFT versus full live prefill for
  large blocs
- load/fork path must be noticeably cheaper than rebuilding the same cache from raw text each run
- amortized payoff must be positive over repeated reuse, not just one-shot TTFT
- manifest validation must prevent stale or mismatched artifacts from being reused silently

## Scope

- MLX-first bloc KV artifact compiler/loader
- one canonical bloc-only persisted recipe
- artifact + manifest contract
- compile/load/fork helpers
- integration of bloc metadata generation with the new layer
- tests and benchmarks

## Non-Goals

- Do not implement arbitrary KV-cache composition/merging.
- Do not promise automatic session composition with unrelated system/tool prefixes.
- Do not imply transcript restoration from loaded cache state.
- Do not add multimodal bloc artifacts in v1.
- Do not make `FileBlocStore` responsible for model execution itself.
- Do not broaden to every provider before the MLX path is solid.
- Do not change the bloc store into a general vector database or memory graph system.

## Dependencies And Related Tasks

- `docs/memory-blocs.md`
- `docs/prompt-caching.md`
- `abstractcore/core/file_blocs.py`
- `abstractcore/core/file_boxes.py`
- `abstractcore/core/bloc_metadata.py`
- `abstractcore/core/cached_session.py`
- `abstractcore/providers/mlx_provider.py`
- `abstractcore/providers/base.py`
- `docs/backlog/planned/2026-05-18_mlx-provider-continuous-batching.md`

Related future work:

- compiled multi-module context recipes (`system + tools + bloc`)
- transcript bootstrap / hydration for `CachedSession` from persisted artifacts
- shared-runtime artifact loading for batched MLX serving
- provider-generic bloc artifact layer for other local control-plane providers

## Expected Outcomes

- AbstractCore gains a first-class way to compile one memory bloc into one durable MLX provider
  artifact per model.
- Higher-level provider flows can reload or fork that artifact without re-prefilling the full bloc
  text.
- Memory blocs become a practical performance feature instead of only a storage convention.
- Existing consumers such as bloc metadata generation stop reimplementing cache orchestration.

## Validation

- Unit tests for compile/load/fork orchestration
- Gated real-MLX integration tests
- Concurrency and durability tests
- Benchmarks proving TTFT and amortized reuse gains for large blocs
- Documentation that clearly states the bloc-only provider-artifact boundary

## Progress Checklist

- [ ] Define canonical v1 bloc-only persisted recipe.
- [ ] Add bloc KV compile/load/fork module.
- [ ] Add artifact manifest path helpers to `FileBlocStore`.
- [ ] Add artifact manifest sidecar support.
- [ ] Implement compile orchestration with manifest-last commit semantics.
- [ ] Implement load/fork orchestration with reload-on-miss handling.
- [ ] Refactor bloc metadata generation to use the new layer.
- [ ] Add unit tests.
- [ ] Add gated real-MLX integration tests.
- [ ] Add concurrency and durability tests.
- [ ] Add benchmarks.
- [ ] Update docs.

## Guidance For The Implementing Agent

Do not start by pretending independent KV artifacts can be merged safely. They cannot, at least not
with the current MLX/local control-plane abstraction.

Start with one exact bloc-only persisted recipe and make it reliable:

- deterministic rendering
- explicit manifest validation
- manifest-last artifact + manifest commit protocol
- safe temp-key lifecycle
- reload-on-miss semantics
- measurable TTFT and amortized reuse benefit

Do not market this as session hydration. If later work needs `system + tools + bloc` as one
reusable prefix, or transcript-aware bootstrap for `CachedSession`, design that as a separate item
rather than stretching this bloc-only artifact layer past what the current runtime can prove.

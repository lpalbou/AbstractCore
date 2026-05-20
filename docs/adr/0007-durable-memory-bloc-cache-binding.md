# ADR 0007: Durable memory bloc cache binding

Status: Accepted.

## Context

AbstractCore now exposes durable memory bloc cache artifacts as release surface, not as private MLX
experiments. A caller can persist one text/file snapshot as a bloc, compile one provider/model
cache artifact from that bloc, load it into a runtime cache key, and then generate against that
loaded key.

That contract crosses provider code, Python helpers, `AbstractEndpoint`, and the gateway. Without
a shared rule, future work could drift into provider-specific route shapes, private cache-store
introspection, or unsafe assumptions that `prompt_cache_key` alone proves exactness.

## Decision

AbstractCore will treat durable memory bloc cache artifacts as provider-backed exact-prefix
artifacts with one public shape:

- one text/file snapshot maps to one bloc;
- one bloc maps to one provider/model artifact for each supported backend;
- one loaded artifact maps to one runtime prompt-cache key;
- `prompt_cache_key` remains a volatile best-effort handle;
- `prompt_cache_binding` is the optional exact-artifact proof for generation-time validation.

Provider-specific serialization remains behind provider hooks. The public API and server routes use
shared helpers and response fields across MLX, HuggingFace transformers, supported HuggingFace
GGUF, and future providers that can prove the same operations.

Unsupported providers, unsupported model classes, unsupported GGUF chat formats, missing keys, and
stale bindings must fail explicitly. They must not silently fall back to an empty cache when the
caller requested exact binding.

## Consequences

### Positive

- Python callers and HTTP clients can use one cache artifact contract across supported local
  providers.
- Provider internals stay encapsulated behind public provider prompt-cache hooks.
- Correctness-sensitive callers can distinguish best-effort cache selection from exact loaded
  artifact use.

### Negative

- Exact binding adds metadata and validation responsibilities to providers that expose durable
  prompt-cache artifacts.
- Some GGUF and future provider paths must remain unsupported until they can render and persist
  exact prefixes safely.

### Neutral

- This ADR does not make cache artifacts the durable memory source of truth; the bloc text remains
  primary.
- This ADR does not authorize arbitrary KV-cache composition, merging, or superbloc artifacts.
- Remote providers keep their existing best-effort `prompt_cache_key` semantics.

## Enforcement

- New durable cache backends must implement provider-level rendering, artifact metadata, and loaded
  key metadata hooks rather than relying on server-side private-store reads.
- Server and endpoint routes must expose the same `ensure/load/manifest` shape and include
  `binding_id` plus `prompt_cache_binding` on load results.
- Generation paths must validate `prompt_cache_binding` before output or streaming begins.
- Completion reports for cache-artifact work must include provider coverage, debug/proof fields,
  and stale-binding negative tests.
- Reviews should reject language that implies composable KV blocks or remote exact binding unless a
  future ADR explicitly accepts that larger contract.

## Validation

- Provider-contract tests cover MLX, HuggingFace transformers, and supported HuggingFace GGUF fake
  backends through the same bloc helper API.
- Endpoint and gateway tests cover `binding_id`, `prompt_cache_binding`, verbose debug payloads,
  and `409` stale-binding failures.
- Real-provider proof runs on 2026-05-20 used one local 2B+ model at a time for release-quality
  semantic validation: MLX Qwen3-4B-4bit, HuggingFace transformers `Qwen/Qwen3.5-4B`, and
  HuggingFace GGUF Qwen3-4B-Instruct Q4_K_M. The proof records full prompt processing versus
  cached suffix processing, binding validation, and correct uncached/cached answers in
  `docs/reports/2026-05-20-durable-memory-bloc-cache-validation.md`.

## Backlog links

- `docs/backlog/completed/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md`
- `docs/backlog/completed/2026-05-20_hf-transformers-bloc-kv-artifact-compiler-loader.md`
- `docs/backlog/completed/2026-05-20_hf-gguf-bloc-kv-artifact-compiler-loader.md`
- `docs/backlog/proposed/2026-05-20_composable-prompt-cache-recipes-for-immutable-memory-clusters.md`

## Related

- `abstractcore/core/bloc_kv.py`
- `abstractcore/providers/base.py`
- `abstractcore/providers/mlx_provider.py`
- `abstractcore/providers/huggingface_provider.py`
- `abstractcore/endpoint/app.py`
- `abstractcore/server/app.py`
- `docs/memory-blocs.md`
- `docs/prompt-caching.md`
- ADR 0001: Engineering guardrails and no silent degradation
- ADR 0003: Provider, capability, and output boundaries

# Bloc composability — framework-level (text, cache) contract designed under the instant-composition assumption

## Metadata
- Created: 2026-07-13
- Status: PROPOSED (operator direction 2026-07-13 21:54: design as if instant
  cache composition is cracked, at least partially; runtime/memory/gateway seam
  discussion open on commons c1668)
- Area: core/file_blocs.py + core/bloc_kv.py (substrate), providers/mlx_provider.py
  + providers/huggingface_provider.py (in-process compose lanes), server endpoints

## Context
Operator direction: the bloc abstraction (content-addressed text + per-model KV
artifacts) should be framework-level — the runtime asks "do I have a cache for
model X?" and sends the cache instead of re-preprocessing. A private unpublished
branch targets prompt-cache composability; if it lands, every (text, cache) bloc
composes instantly, pairing with a memory graph that retrieves a selected number
of blocs per turn. Full research report + scale-dependency theory + the designed
contract: agora commons fs `research/cache-composability.md` (CacheBlend/EPIC/PIC
references, break-even math, fan-out primitive).

Key design points fixed by that document:
- Bloc = (text, meta, {model_key → kv_artifact}); FileBlocStore is the substrate.
- Minting by recurrence (memory-graph global access counters as the signal).
- `compose(bloc_ids, model) → session cache` with link/repair (RoPE re-rotation +
  LegoLink-style static-k or CacheBlend HKVD recompute) on in-process backends;
  server-side degrades to byte-stable prefix ordering.
- Runtime seam: `context_blocs` on LLM_CALL; ledger records bloc ids + binding
  ids, never cache bytes (text durable, caches derived; replay re-derives).
- `query_bloc(bloc_id, question, model)` fan-out primitive (ms-scale per-bloc
  LLM queries over a shelf).
- Invalidation = refusal + recompile, gated by the 0817 audit.

## Scope (when promoted)
- The compose/link primitive on MLX first (masked partial prefill — custom
  attention step; LegoLink static-k as the v1 algorithm), HF transformers second.
- `has_kv_cache`/compile-on-miss/compose surfaces exported for runtime use +
  server endpoints for remote runtimes (same-host v1; gateway brokering per the
  c1668 ask-3 answer).
- Blend-vs-joint-prefill quality harness (fact-recall harness reused) gating
  default-on.

## Non-goals
- Cross-MODEL cache transfer (a bloc's artifact never serves a different model).
- Server-side blend (structurally impossible; prefix ordering is the degrade).

## Dependencies
- 0817 (invalidation-key audit) gates any load-and-compose path.
- The private composability branch defines how much repair the link step needs.
- Runtime/memory/gateway seam answers on commons c1668.

## Expected outcomes
- Turns assembled from cached blocs at IO-bound (not prefill-bound) latency;
  measured against the report's break-even predictions.
- The fan-out primitive available to recall (per-bloc ms-scale queries).

## Validation
- Quality: blend answers ≡ joint-prefill answers on the harness within threshold.
- Performance: compose beats re-prefill at shelf scale (12–36 digests) and at
  document scale on Apple Silicon, per the report's worked examples.

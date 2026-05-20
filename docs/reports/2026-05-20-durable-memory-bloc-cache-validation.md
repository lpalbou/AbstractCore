# Durable Memory Bloc Cache Validation

Date: 2026-05-20

This report records the local proof run for provider-backed durable memory bloc caches. It is a
point-in-time benchmark, not a guaranteed performance claim.

## Scope

The tested contract is:

```text
1 exact text/file -> 1 memory bloc -> 1 provider/model-native cache artifact
```

The shared AbstractCore interface is:

- Python: `ensure_bloc_kv_artifact(...)` and `load_bloc_kv_artifact(...)`
- Server: `/acore/blocs/kv/ensure` and `/acore/blocs/kv/load`
- Request binding: `prompt_cache_binding`

The payload is provider-native. MLX, HuggingFace transformers, and HuggingFace GGUF do not share a
portable KV tensor format.

## Method

Benchmark command pattern:

```bash
python examples/performance/durable_bloc_cache_benchmark.py --case <case> --repetitions 80
```

Each case ran in its own process. The script warms the loaded model, then records:

- `full_prompt_processing_s`: processing the full memory bloc plus question into a fresh cache
- `cached_suffix_processing_s`: processing only the live suffix after loading/forking the durable bloc cache
- `artifact_load_s`: loading the durable provider-native artifact
- `cached_generation_s`: live cached request latency including suffix processing and decode
- semantic correctness for both uncached and cached answers

The benchmark question requires the answer to contain:

- `Tuesday at 09:30 UTC`
- `Mira Chen`
- `ACORE-7421`

## Results

| Case | Model | Artifact | Tokens | Full Processing | Cached Suffix Processing | Processing Speedup | Artifact Load | Cached Generation | Correct |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MLX | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `.safetensors`, 536,600,383 bytes | 3,639 | 0.9061s | 0.1422s | 6.372x | 0.2131s | 0.3317s | yes |
| HF transformers | `Qwen/Qwen3.5-4B` | `.safetensors`, 173,923,172 bytes | 3,723 | 2.5437s | 0.8196s | 3.1036x | 0.0935s | 1.9163s | yes |
| HF GGUF | `unsloth/Qwen3-4B-Instruct-2507-GGUF` Q4_K_M | `.npz`, 490,639,611 bytes | 3,642 | 1.3672s | 0.1686s | 8.1091x | 1.2354s | 0.3647s | yes |

For GGUF, cached generation metadata confirmed actual durable-prefix use:

```json
{
  "prompt_cache_prefix_source": "loaded_cache",
  "prompt_cache_composed": true,
  "prompt_cache_prefix_token_count": 3642,
  "prompt_cache_suffix_token_count": 44,
  "prompt_cache_prompt_token_count": 3686
}
```

## Interpretation

The speedup is in prompt processing, not model load. Artifact load is reported separately because a
long-running loaded runtime should load once and reuse the cache repeatedly.

Generation latency may also improve because the generation call no longer re-processes the full
memory bloc, but AbstractCore should not claim decode itself became faster. The meaningful proof is
that the full prompt processing phase drops to the suffix-only processing phase while the answer
stays correct.

## Compatibility Notes

- MLX uses MLX-LM prompt-cache payloads.
- HF transformers uses provider-native `Cache` objects persisted in `.safetensors`; current
  coverage includes standard `DynamicCache` layer state, Qwen3.5/Qwen3Next-style tensor-list hybrid
  state, and Mamba-style tensor state when the cache class can be constructed from model config.
- HF GGUF uses llama.cpp state snapshots in `.npz` and is exact-renderer gated. Current exact
  renderers are `chatml-function-calling` and `llama-3`.
- Sub-2B local models are not accepted as semantic proof targets for this cache-validation path.

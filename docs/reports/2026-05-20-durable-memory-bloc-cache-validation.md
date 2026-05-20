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
- strict correctness, where noted, means both uncached and cached answers exactly matched:
  `launch_window=Tuesday at 09:30 UTC; inspector=Mira Chen; checksum=ACORE-7421`

The benchmark question requires the answer to contain:

- `Tuesday at 09:30 UTC`
- `Mira Chen`
- `ACORE-7421`

## Baseline Three-Run Averages

These are averages over three isolated subprocess runs per case. Each run loaded exactly one model
and produced strict-correct uncached and cached answers.

| Case | Model | Artifact | Tokens | Full Processing | Cached Suffix Processing | Processing Speedup | Artifact Load | Uncached Generation | Cached Generation | Correct |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MLX | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `abstractcore-mlx-prompt-cache/v1`, 536,600,383 bytes | 3,639 | 0.6796s | 0.1144s | 5.96x | 0.1878s | 0.8516s | 0.2955s | 3/3 strict |
| HF transformers | `Qwen/Qwen3.5-4B` | `abstractcore-transformers-prompt-cache/v1`, 173,923,172 bytes | 3,723 | 1.5133s | 0.1840s | 8.26x | 0.0968s | 3.5251s | 1.7375s | 3/3 strict |
| HF GGUF | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `abstractcore-gguf-prompt-cache/v1`, 490,639,617 bytes | 3,642 | 1.5457s | 0.1645s | 9.39x | 1.2424s | 1.8837s | 0.3555s | 3/3 strict |

## Additional Model Checks

| Model Check | Model | Artifact | Tokens | Full Processing | Cached Suffix Processing | Processing Speedup | Artifact Load | Uncached Generation | Cached Generation | Correct |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.6 27B GGUF Q4_K_M | `/Users/albou/.lmstudio/models/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf` | `abstractcore-gguf-prompt-cache/v1`, 373,319,512 bytes | 3,724 | 13.6949s | 0.3105s | 44.11x | 1.0095s | 10.9037s | 1.2401s | strict |
| Qwen3.5 27B MLX 4-bit | `/Users/albou/.lmstudio/models/mlx-community/Qwen3.5-27B-4bit` | `abstractcore-mlx-prompt-cache/v1`, 397,882,425 bytes | 3,722 | 4.3130s | 0.2893s | 14.91x | 0.1345s | 5.8180s | 1.3950s | strict |
| Gemma4 E4B GGUF Q4_K_M | `/Users/albou/.lmstudio/models/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf` | `abstractcore-gguf-prompt-cache/v1`, 197,017,402 bytes | 3,813 | 1.1362s | 0.2109s | 5.39x | 0.6186s | 0.3507s | 0.4594s | strict |
| Gemma4 26B-A4B GGUF Q4_K_M | `/Users/albou/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf` | `abstractcore-gguf-prompt-cache/v1`, 1,306,902,170 bytes | 3,813 | 1.7065s | 0.2884s | 5.92x | 3.0086s | 0.3726s | 0.4737s | strict |
| Gemma4 31B GGUF Q4_K_M | `/Users/albou/.lmstudio/models/unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf` | `abstractcore-gguf-prompt-cache/v1`, 3,668,408,838 bytes | 3,813 | 18.6921s | 1.2588s | 14.85x | 9.6812s | 5.5739s | 1.8475s | strict |
| Gemma4 26B-A4B MLX 4-bit | `/Users/albou/.lmstudio/models/mlx-community/gemma-4-26b-a4b-4bit` | `abstractcore-mlx-prompt-cache/v1`, 287,772,618 bytes | 3,811 | 4.1853s | 0.2944s | 14.22x | 0.1138s | 5.5687s | 1.8616s | strict |
| Gemma4 31B MLX MXFP4 | `/Users/albou/.lmstudio/models/mlx-community/gemma-4-31b-mxfp4` | `abstractcore-mlx-prompt-cache/v1`, 1,151,073,453 bytes | 3,811 | 20.0017s | 0.5807s | 34.44x | 0.6033s | 29.1899s | 4.3598s | strict |

Gemma4 GGUF checks require `llama-cpp-python>=0.3.23,<1.0.0`, matching AbstractCore's package
requirement. Gemma4 exact rendering uses the model's llama.cpp chat template.

Gemma4 MLX uses hybrid rotating and full KV cache layers. AbstractCore reports the effective cached
prefix length from provider cache state, so the reported token count reflects the usable durable
prefix rather than one individual layer's local window.

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

The 4B baseline models are not quantization-equivalent:

- MLX used `mlx-community/Qwen3-4B-Instruct-2507-4bit`, a 4-bit MLX model with a 2.11 GiB local
  weight payload.
- HF GGUF used `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, a 2.33 GiB Q4_K_M llama.cpp artifact.
- HF transformers used `Qwen/Qwen3.5-4B`, whose local safetensors payload is 8.68 GiB and is not the
  same 4-bit model family/runtime as the MLX and GGUF checks.
  This report does not include a Q4-equivalent HF transformers proof.

The three-run average shows HF transformers processing the full bloc in 1.51s and the cached suffix
in 0.18s, for an 8.26x processing-only speedup. Its slower total generation numbers are
decode/runtime throughput, not failed durable-prefix reuse.

`Qwen/Qwen3.6-27B-FP8` was checked as a trusted official HF-transformers quantized candidate. The
snapshot downloaded locally, but FP8 is not an Apple/MPS Transformers proof target today: upstream
FP8 guidance targets CUDA-class accelerators, and Qwen's Apple guidance points to MLX. The local
fallback load did not pass semantic smoke tests: trivial prompts produced incorrect text. No
durable-cache benchmark was recorded for that model because model-load/generation correctness is a
prerequisite for cache validation.

## Compatibility Notes

- MLX uses MLX-LM prompt-cache payloads.
- HF transformers uses provider-native `Cache` objects persisted in `.safetensors`; current
  coverage includes standard `DynamicCache` layer state, Qwen3.5/Qwen3Next-style tensor-list hybrid
  state, and Mamba-style tensor state when the cache class can be constructed from model config.
- HF GGUF uses llama.cpp state snapshots in `.npz` and is exact-renderer gated. Current exact
  renderers are `chatml-function-calling`, `llama-3`, and Gemma4 `gemma_turn` through llama.cpp's
  model chat template.
- Gemma4 GGUF requires a recent llama.cpp runtime. The repository dependency is
  `llama-cpp-python>=0.3.23,<1.0.0`; older local environments may load Qwen/Llama GGUFs while
  failing Gemma4 GGUFs.
- Sub-2B local models are not accepted as semantic proof targets for this cache-validation path.

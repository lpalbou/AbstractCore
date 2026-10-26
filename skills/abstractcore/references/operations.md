# Operating an AbstractCore installation

## Choose and inspect an endpoint

Use a backend's exact model listing rather than guessing model IDs. For an OpenAI-compatible
endpoint with `/v1` in `ACORE_BASE_URL`, this read-only probe checks connectivity, authentication
and a nonempty model catalog. `ACORE_API_KEY` is that endpoint's key, never an unrelated cloud key.

```python
import os
import httpx

try:
    base = os.environ["ACORE_BASE_URL"].rstrip("/")
    key = os.environ.get("ACORE_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = httpx.get(base + "/models", headers=headers, timeout=5)
    if response.status_code in (401, 403):
        raise SystemExit("Authentication failed; fix the endpoint key before another request.")
    response.raise_for_status()
    models = [item["id"] for item in response.json()["data"] if item.get("id")]
except (KeyError, TypeError, ValueError, httpx.HTTPError):
    raise SystemExit("Model discovery failed; check base URL, server availability and response shape.") from None
if not models:
    raise SystemExit("Endpoint returned no models; install/load the intended model first.")
print("\n".join(models))
```

Success lists usable identifiers to select; it does not prove generation works. `/models`
has no universal server-version field: verify server version through that backend's own
version/health surface. Run the main skill's bounded generation recipe next.

## Exact local artifacts

After inspection, plan a download with an explicit engine and artifact. The example selector
`unsloth/Qwen3-8B-GGUF:Q4_K_M` illustrates repo plus quantization; use the catalog's exact
`cli_download` for the user's selected model. A GGUF repo without `:QUANT` can download all
quantizations. MLX repos normally identify one quantization and can include MTP companions.

```bash
abstractcore models download huggingface "unsloth/Qwen3-8B-GGUF:Q4_K_M" --dry-run --json
abstractcore engines install ollama --dry-run
```

A plan is not installation. Review disk/memory fit and the exact command, then execute the
authorized operation by dropping `--dry-run`. Downloads may start jobs; use `abstractcore models
jobs --json` and inspect the final status before loading. `abstractcore models verify` performs
real inference. Models CLI exit codes are `0` success, `1` error, `2` refusal; inspect `status`,
`reason`, `message` and delete blockers. Unknown fit data is not a pass.

For local calls use `create_llm("mlx", model="<EXACT_MLX_REPO>", ...)` or
`create_llm("huggingface", model="<EXACT_REPO_OR_FILE>", ...)`. Supply the same explicit
budgets/error handling as the main recipe. HF has separate Transformers and GGUF paths;
`model_type="transformers"` or `"gguf"` can disambiguate. Engine rerouting between MLX and
GGUF warns; artifact substitution is refused. Model load errors require fixing cache, engine
dependencies or exact artifact selection, not silently choosing different weights.

On Apple/MPS, half-precision Transformers SDPA on torch 2.8.x/2.10.x has documented correctness
issues. Follow the load warning: use `ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL=eager` or an
appropriate newer torch. MLX cache, native batching and MTP support are checkpoint-specific.

## Configuration and serving

`abstractcore --config` is interactive; `abstractcore --status` diagnoses configured routes.
The config path can be isolated with `ABSTRACTCORE_CONFIG_DIR` or `ABSTRACTCORE_CONFIG_FILE`.
The canonical text route is `input.text`; `output.text` is its read-only view.
`embedding.text` selects retrieval embeddings. Image/video/voice/music/3D have separate routes.
Use `abstractcore config --help` and the centralized-config guide before writing defaults.

```bash
abstractcore serve --help
abstractcore-endpoint --help
abstractcore-chat --help
```

`abstractcore serve` binds loopback port 8000 by default. On first local use it writes a server
token under the config directory and prints a one-time console claim link. Keep the link/token
private. Binding a non-loopback host requires `ABSTRACTCORE_AUTH_TOKEN`. In authenticated mode,
`Authorization: Bearer ...` is the server token; `X-AbstractCore-Provider-API-Key` optionally
overrides the selected upstream provider's key. Never interchange those credentials.

The universal server routes `POST /v1/chat/completions` using model IDs such as
`ollama/<model>`; the single-model `abstractcore-endpoint` hosts one provider/model.
`response_format` enables JSON/schema validation over HTTP. `/console` manages the host.
Do not start another server on an occupied port or treat a help command as proof of health.

## Memory, caches, cancellation and diagnostics

- `llm.get_prompt_cache_capabilities()` describes `none`, `keyed`, or `local_control_plane`.
  `llm.prompt_cache_supports_operation(...)` gates controls. Unsupported operations raise
  structured cache errors with `operation`, `code` and `capabilities`.
- `llm.get_prompt_cache_stats()` and response prompt-cache metadata describe reuse.
  `ensure_bloc_kv_artifact` compiles an exact text/file bloc;
  `load_bloc_kv_artifact` returns a binding for `generate(prompt_cache_binding=...)`.
  Artifacts are provider/model/renderer-specific. List bindings before deleting/pruning;
  live-bound artifacts are blocked by default. Do not bypass that check to make a cleanup pass.
- `llm.unload_model(llm.model)` cancels/drains supported in-flight requests. In-process engines
  release the calling instance's hold on weights/caches; another owner may retain a shared model.
  Generic compatible endpoints only close/reset the connection; releasing server-side weights
  needs that server's own control API (implemented for Ollama and LM Studio).
  Inspect the host memory snapshot and loaded-model
  sweep rather than inferring memory release from a successful method return. Unload affects
  concurrent work; coordinate with the owner.
- Pass a `threading.Event` as `cancel_event=`. Set it to request cancellation; consult
  `supports_generation_cancel()` for backend support. Cancellation raises
  `GenerationCancelledError`; its partial output is diagnostic, not a finished answer.
  Local lanes can report `progress_callback=` events when supported.
- Enable `enable_tracing=True` only when prompt/response capture is intended. Events/logging
  provide observability; traces, sessions and cache artifacts need private storage.

## Built-in application CLIs

Inspect `--help` before selecting an input/output path:

```bash
abstractcore-summarizer --help
abstractcore-extractor --help
abstractcore-judge --help
abstractcore-intent --help
abstractcore-deepsearch --help
```

These run model-backed document/analysis tasks. Specify provider/model and output destinations
as the app's help describes; inference, file writes and DeepSearch's web activity are effects,
so a help-only inspection is the cheap first step. Python exposes `BasicSummarizer` and
`BasicExtractor` from `abstractcore` for application integrations. For ontology-constrained
extraction, model choice, the allowed-labels and direction rules in the prompt, reasoning, and a
verification pass change precision more than anything else; see
[Choosing a model for knowledge-graph extraction](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/knowledge-graph-extraction-models.md).

## Further documentation

- [Models](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/models.md)
- [Engines](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/engines.md)
- [Configuration](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/centralized-config.md)
- [Server](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/server.md)
- [Endpoint](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/endpoint.md)
- [Memory lifecycle](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/memory-management.md)
- [Memory blocs](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/memory-blocs.md)
- [Cancellation](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/generation-cancel.md)
- [Examples](https://raw.githubusercontent.com/lpalbou/AbstractCore/main/docs/examples.md)

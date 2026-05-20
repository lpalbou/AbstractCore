# Memory Blocs (Text/File -> Bloc -> Durable Cache)

AbstractCore can store **memory blocs**: content-addressed snapshots of extracted text or file
text, plus optional provider/model prompt-cache artifacts derived from that exact text.

This is useful for apps that want:
- durable, incremental “cached file text” across runs,
- per-model prompt-cache/KV artifacts compiled from that text (so you do not re-prefill the same
  file or text bloc repeatedly),
- stable selectors (`bloc_id`) for REPL/CLI workflows.

## What is a “bloc”?

A **bloc** is an atomic extracted-text snapshot stored on disk. It is identified by a SHA256 of the *source file bytes* (or another stable content hash chosen by the app).

The store persists:
- `content.txt` — extracted text
- `meta.json` — record metadata (path, size, timestamps, token estimate, etc.)
- `meta.jsonld` (optional) — compact JSON-LD metadata for catalogs/search
- `kv/…` (optional) — per-(provider, model) prompt-cache artifacts (derived; `.safetensors` for
  MLX/HF transformers, `.npz` for supported HF GGUF)
- `kv/…manifest.json` (optional) — sidecar integrity/freshness metadata for those artifacts

## On-disk layout

Default root:
- `~/.abstractcore/blocs/`

Per-bloc directory:
- `~/.abstractcore/blocs/files/<sha256>/`
  - `content.txt`
  - `meta.json`
  - `meta.jsonld` (optional)
  - `kv/<provider+model-slug>.<provider-extension>` (optional)
  - `kv/<provider+model-slug>.manifest.json` (optional)

`FileBlocStore.kv_cache_path(...)` and `FileBlocStore.kv_cache_manifest_path(...)` define the
on-disk naming convention for KV artifacts and their manifest sidecars.

## Minimal usage (library)

`FileBlocStore` does not extract files by itself; it expects extracted content + file metadata from the caller.

Typical flow:
1. extract a file to text (e.g. via `extract_file_box()` or your own pipeline)
2. write/update the bloc record + `content.txt` via `FileBlocStore.upsert(...)`
3. (optional) compile a per-model KV artifact with `ensure_bloc_kv_artifact(...)`
4. (optional) load or fork it later with `load_bloc_kv_artifact(...)`
5. (optional) generate JSON-LD metadata with `generate_bloc_metadata_jsonld(...)`

```python
from abstractcore import create_llm, ensure_bloc_kv_artifact, load_bloc_kv_artifact
from abstractcore.core.file_blocs import FileBlocStore

store = FileBlocStore()
llm = create_llm("mlx", model="mlx-community/Qwen3-4B")

record = store.upsert(
    file_meta={
        "path": "notes/orbit.txt",
        "sha256": "...",
        "content_sha256": "...",
        "media_type": "text",
        "size_bytes": 1234,
    },
    content="Document title: Orbit Notes\n\nThe launch window is Tuesday.",
)

ensure = ensure_bloc_kv_artifact(provider=llm, store=store, record=record, debug=True)
loaded = load_bloc_kv_artifact(provider=llm, store=store, record=record, key="work:orbit")

response = llm.generate(
    "Summarize the loaded bloc.",
    prompt_cache_binding=loaded.prompt_cache_binding,
)
```

See:
- `abstractcore/core/file_blocs.py` for the store and record schema
- `abstractcore/core/bloc_kv.py` for the unified bloc artifact compiler/loader
- `abstractcore/core/bloc_metadata.py` for JSON-LD schema + metadata generator/parser

## Supported local artifact backends

The public API shape is shared. Provider internals and artifact formats remain backend-specific:

- **MLX**: `.safetensors`, exact prompt fragment rendered by `MLXProvider`.
- **HuggingFace transformers**: `.safetensors`, gated on local prompt-cache save/load support for
  standard text-generation models and provider-native cache state AbstractCore can serialize and
  reconstruct. Current coverage includes standard `DynamicCache` layer state, Qwen3.5/Qwen3Next
  tensor-list hybrid state, and Mamba-style tensor state when the Transformers cache class can be
  constructed from model config.
- **HuggingFace GGUF**: `.npz`, gated on exact cached prompt renderers. Current exact renderer
  paths are `chatml-function-calling` and `llama-3`; other GGUF chat formats remain keyed-only.

Remote providers and generic OpenAI-compatible servers keep best-effort `prompt_cache_key`
semantics. They do not expose local durable bloc artifacts.

Artifact payloads are provider-native and model-bound. The portable contract is the manifest,
binding object, Python helper, and server route shape, not a universal KV tensor layout.

## Binding and debug proof

`load_bloc_kv_artifact(...)` returns:

- `key`: the runtime prompt-cache key to use
- `binding_id`: an opaque digest over the exact manifest identity
- `prompt_cache_binding`: the compact object callers can pass to generation for strict checking

`prompt_cache_key` by itself remains best-effort. If you need to prove that a request is using the
exact loaded bloc artifact, pass `prompt_cache_binding` (or the direct Python alias
`expected_prompt_cache_binding`). If the key is missing or rebound to another artifact, generation
fails before producing output.

Verbose verification is available with `debug=True` on `ensure_bloc_kv_artifact(...)` and
`load_bloc_kv_artifact(...)`, or by setting `ABSTRACTCORE_BLOC_KV_DEBUG=1`. Debug payloads include
provider/backend ids, artifact paths, manifest path, artifact hash, binding id, rendered recipe
hash, bloc/content hashes, and token count when available.

## API exposure

For a long-lived single-model local runtime, `AbstractEndpoint` exposes:

- `POST /acore/blocs/upsert_text`
- `GET /acore/blocs/record`
- `GET /acore/blocs/kv/manifest`
- `POST /acore/blocs/kv/ensure`
- `POST /acore/blocs/kv/load`

The multi-provider gateway `abstractcore.server.app` now exposes two modes:

- direct gateway mode:
  - `POST /acore/models/load`
  - `GET /acore/models/loaded`
  - `POST /acore/models/unload`
  - local `POST /acore/blocs/upsert_text`
  - local `GET /acore/blocs/record`
  - local `GET /acore/blocs/kv/manifest`
  - local `POST /acore/blocs/kv/ensure`
  - local `POST /acore/blocs/kv/load`
- proxy mode:
  - the same `/acore/blocs/*` routes with `base_url` pointing at an upstream `AbstractEndpoint`

Important boundary:

- the durable artifact is bloc-only
- in direct gateway mode, text loaded cache keys live on the loaded gateway runtime selected by `provider` + `model`
- `/acore/models/load` is task-aware; omitted `task` keeps the text runtime behavior used by bloc-KV, while `image_generation`, `tts`, and `stt` route to capability residency where supported
- `runtime_id` from `/acore/models/load` is the most precise selector when multiple warm runtimes share the same `task` + `provider` + `model`
- in proxy mode, loaded cache keys live on the upstream endpoint worker/provider instance
- `/acore/blocs/kv/load` returns `artifact.key`, `artifact.binding_id`, and
  `artifact.prompt_cache_binding`
- use `artifact.prompt_cache_binding` on the next chat call when exact request-time binding is
  required

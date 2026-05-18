# Memory Blocs (File → Bloc → KV Artifacts)

AbstractCore can store **memory blocs**: content-addressed snapshots of extracted file text, plus optional derived artifacts.

This is useful for apps that want:
- durable, incremental “cached file text” across runs,
- per-model prompt-cache/KV artifacts compiled from that text (so you don’t re-prefill the same file repeatedly),
- stable selectors (`bloc_id`) for REPL/CLI workflows.

## What is a “bloc”?

A **bloc** is an atomic extracted-text snapshot stored on disk. It is identified by a SHA256 of the *source file bytes* (or another stable content hash chosen by the app).

The store persists:
- `content.txt` — extracted text
- `meta.json` — record metadata (path, size, timestamps, token estimate, etc.)
- `meta.jsonld` (optional) — compact JSON-LD metadata for catalogs/search
- `kv/…` (optional) — per-(provider, model) prompt-cache artifacts (derived)
- `kv/…manifest.json` (optional) — sidecar integrity/freshness metadata for those artifacts

## On-disk layout

Default root:
- `~/.abstractcore/blocs/`

Per-bloc directory:
- `~/.abstractcore/blocs/files/<sha256>/`
  - `content.txt`
  - `meta.json`
  - `meta.jsonld` (optional)
  - `kv/<provider+model-slug>.safetensors` (optional)
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

See:
- `abstractcore/core/file_blocs.py` for the store and record schema
- `abstractcore/core/bloc_kv.py` for the MLX bloc artifact compiler/loader
- `abstractcore/core/bloc_metadata.py` for JSON-LD schema + metadata generator/parser

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
- in direct gateway mode, loaded cache keys live on the loaded gateway runtime selected by `provider` + `model`
- `runtime_id` from `/acore/models/load` is the most precise selector when multiple warm runtimes share the same `provider` + `model`
- in proxy mode, loaded cache keys live on the upstream endpoint worker/provider instance
- `/acore/blocs/kv/load` returns `artifact.key`; use that value as `prompt_cache_key` on the next chat call

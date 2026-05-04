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

## On-disk layout

Default root:
- `~/.abstractcore/blocs/`

Per-bloc directory:
- `~/.abstractcore/blocs/files/<sha256>/`
  - `content.txt`
  - `meta.json`
  - `meta.jsonld` (optional)
  - `kv/<provider+model-slug>.safetensors` (optional)

`FileBlocStore.kv_cache_path(...)` defines the exact naming convention for KV artifact files.

## Minimal usage (library)

`FileBlocStore` does not extract files by itself; it expects extracted content + file metadata from the caller.

Typical flow:
1. extract a file to text (e.g. via `extract_file_box()` or your own pipeline)
2. write/update the bloc record + `content.txt` via `FileBlocStore.upsert(...)`
3. (optional) compile a per-model KV artifact with your provider and save it under `kv/`
4. (optional) generate JSON-LD metadata with `generate_bloc_metadata_jsonld(...)`

See:
- `abstractcore/core/file_blocs.py` for the store and record schema
- `abstractcore/core/bloc_metadata.py` for JSON-LD schema + metadata generator/parser


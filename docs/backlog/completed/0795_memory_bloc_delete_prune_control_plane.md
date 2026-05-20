# Completed: Memory bloc delete/prune control plane

## Metadata
- Created: 2026-05-20
- Completed: 2026-05-20
- Status: Completed

## Request
AbstractRuntime needed Core-owned local helpers and matching server routes for deleting memory blocs
and durable bloc KV artifacts, with a public safety seam for loaded prompt-cache keys.

## Current code reality
- Core owns the durable memory-bloc on-disk layout through `FileBlocStore`.
- Core owns bloc KV manifests, artifact paths, and prompt-cache binding metadata through
  `abstractcore.core.bloc_kv`.
- Endpoint and gateway already exposed upsert, record, manifest, ensure, and load operations, but
  deletion required callers to know filesystem details and could bypass live in-memory cache state.

## Outcome
Implemented public Python helpers:

- `list_bloc_kv_artifacts(...)`
- `find_bloc_kv_live_bindings(...)`
- `delete_bloc_kv_artifact(...)`
- `prune_bloc_kv_artifacts(...)`
- `delete_bloc(...)`

Implemented matching local/proxy routes:

- `GET /acore/blocs`
- `POST /acore/blocs/delete`
- `GET /acore/blocs/kv/list`
- `POST /acore/blocs/kv/delete`
- `POST /acore/blocs/kv/prune`

Safety behavior:

- Deleting a loaded artifact fails with `409` unless the caller sets `clear_loaded=true` or
  explicitly forces deletion.
- `clear_loaded=true` clears matching in-process prompt-cache keys before deleting the artifact.
- Gateway direct mode checks loaded runtimes; proxy mode forwards the same contract to
  AbstractEndpoint.
- `dry_run=true` reports matching deletions without removing files.

## Validation
- `pytest -q tests/test_bloc_kv.py tests/test_bloc_kv_endpoint.py tests/server/test_server_openapi_docs.py`
- Live-binding unit coverage proves delete blocks a loaded artifact, then clears the loaded key and
  removes artifact/manifest files.
- Endpoint coverage proves the HTTP route returns `409` while a stable cache key is live, then
  succeeds with `clear_loaded=true`.
- OpenAPI coverage proves all new routes are tagged and request bodies have examples.

## Notes
Runtime should prefer these helpers/routes over raw filesystem deletion. That keeps local,
gateway, and remote/hybrid behavior aligned with Core's manifest and live-binding rules.

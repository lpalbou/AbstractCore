# Deprecated: Generic transformers/GGUF prompt-cache parity proposal

## Metadata
- Created: 2026-05-20
- Deprecated: 2026-05-20
- Status: Deprecated
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: Superseded by planned provider-specific work; any durable public contract is tracked
  by `planned/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md`

## Context

This item originally asked whether AbstractCore should extend exact bloc/superbloc artifact
workflows beyond MLX. It was useful as a review checkpoint, but it is now too broad for execution.

The code reality is no longer one vague "parity" gap:

- MLX already has durable exact bloc artifacts.
- HuggingFace transformers has local prompt-cache control-plane save/load support, but no durable
  bloc artifact manifest/helper/route integration yet.
- HuggingFace GGUF has local prompt-cache control-plane support only for chat formats with exact
  cached prompt renderers.
- No provider currently has a first-class superbloc artifact compiler in AbstractCore.

## Current code reality

- `abstractcore/core/bloc_kv.py` currently implements the durable artifact layer only for MLX.
- `abstractcore/providers/base.py` defines provider prompt-cache capabilities, including
  `none`, `keyed`, and `local_control_plane`.
- `abstractcore/providers/huggingface_provider.py` reports transformers local-control-plane
  support and GGUF local-control-plane support only for exact renderer paths.
- `docs/prompt-caching.md` already documents that transformers has local control-plane support and
  GGUF support is renderer-gated.
- `../ai-space` uses `superbloc` for grouped bloc membership, but AbstractCore does not yet expose
  a superbloc artifact recipe.

## Deprecation reason

The proposal has been split into concrete planned work:

- `docs/backlog/planned/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md`
- `docs/backlog/planned/2026-05-20_hf-transformers-bloc-kv-artifact-compiler-loader.md`
- `docs/backlog/planned/2026-05-20_hf-gguf-bloc-kv-artifact-compiler-loader.md`

Keeping this generic item open would make future agents re-litigate the same broad parity question
instead of implementing the backend-specific contracts.

## Preserved concerns

- Do not treat MLX, transformers, and GGUF as identical internally. Share the Python/server API,
  not serializer assumptions.
- Do not claim GGUF supports all chat formats; gate exact artifacts on exact cached prompt
  renderers.
- Do not claim durable superbloc artifacts exist until AbstractCore defines a deterministic
  superbloc recipe and compiler.
- Do not mix remote prompt-cache observability into local exact artifact work.
- Do not expose provider-private metadata as the public binding contract.

## Guidance for future agents

Use the planned provider-specific items as the execution source of truth. Reopen a parity proposal
only if another provider family gains enough local prompt-cache control-plane capability to justify
its own durable exact bloc artifact backend.

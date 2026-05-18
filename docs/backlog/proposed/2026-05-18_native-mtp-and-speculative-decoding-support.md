# Proposed: Native MTP and speculative decoding support

## Metadata
- Created: 2026-05-18
- Status: Proposed
- Completed: N/A

## Context

AbstractCore now catalogs recent Qwen3.6 MTP model distributions, including:

- `unsloth/Qwen3.6-27B-MTP-GGUF`
- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`

There is also interest in MLX-native MTP exports for Apple Silicon, for example:

- `samwang0041/Qwen3.6-27B-MLX-4bit-MTP`

Today, AbstractCore can identify these models and route them through existing local or remote
providers, but it cannot yet exploit Multi-Token Prediction (MTP) as an actual acceleration mode in
a consistent, user-visible way.

Current local findings:

- `abstractcore.providers.mlx_provider.MLXProvider` loads models through `mlx_lm.load(...)` and
  generates through standard `generate` / `stream_generate`.
- the installed `mlx-lm` runtime exposes speculative decoding with a separate `draft_model`
  argument, but AbstractCore does not surface that path yet.
- the installed `mlx-lm` Qwen3.5/Qwen3.6 model loader currently strips `mtp.` weights, so MTP
  checkpoints load as regular models rather than using the embedded MTP head.
- current in-process GGUF execution can catalog MTP-tagged repositories, but does not provide a
  first-class native MTP execution path.

This means AbstractCore can honestly say "this model exists" but not yet "this provider/runtime is
using the model's MTP acceleration path."

## Problem

Without an explicit MTP/speculation design, the system creates three kinds of confusion:

- model catalogs can expose MTP-flavored models even when the selected runtime will silently use
  them as standard autoregressive models;
- Apple Silicon users may assume MLX MTP checkpoints are faster today even though the current local
  stack does not consume their MTP weights;
- future provider work may add speculation or native MTP in incompatible ways unless Core defines a
  common request vocabulary first.

There is a real opportunity here, but it needs to be framed correctly:

- draft-model speculative decoding is the fastest path to near-term wins on MLX;
- true embedded MTP support is runtime-specific and should not be pretended into existence;
- local external runtimes may deliver native MTP sooner than in-process Python bindings.

## Proposal

Support this in four layers, in order.

### 1. Make acceleration capabilities explicit

Extend provider/model capability reporting so Core can distinguish:

- `supports_draft_speculation`
- `supports_native_mtp`
- `supports_mtp_model_loading`
- `requires_mmproj`
- `recommended_runtime`

This metadata should be visible through Python capability APIs and the server catalog routes. If a
model is cataloged as MTP-capable but the selected provider/runtime cannot exploit it, the catalog
or resolution metadata should say so clearly.

### 2. Add a small provider-agnostic speculation contract

Extend the Python generation layer first, then have the OpenAI-compatible server rely on the same
path.

Suggested optional request fields:

```python
{
    "speculation": {
        "mode": "draft_model",
        "draft_model": "mlx-community/Qwen3.6-4B-4bit",
        "num_draft_tokens": 4,
        "require_acceleration": False,
    }
}
```

The first implementation should stay small:

- one explicit `speculation` block;
- one stable `draft_model` mode;
- a clear warning when the provider ignores the request.

Do not overload this with multiple speculative algorithms in v1.

### 3. Implement draft-model speculative decoding in `MLXProvider`

This is the most pragmatic local speed path on Apple Silicon.

Implementation shape:

- allow `MLXProvider` to optionally load a secondary draft model;
- pass that draft model and `num_draft_tokens` into `mlx_lm.generate(...)` /
  `mlx_lm.stream_generate(...)`;
- return normalized metadata indicating whether speculation was requested and actually used.

This gives users a real acceleration mechanism now, without claiming that MTP checkpoints are being
used natively.

### 4. Treat native MTP as provider/runtime-specific

Native MTP support should be added only where the runtime truly supports it.

For near-term practicality:

- MLX embedded-MTP support depends on upstream `mlx-lm` consuming Qwen3.6 MTP weights instead of
  stripping them.
- GGUF/native llama.cpp-style MTP support is more likely to land first through an external local
  runtime/server than through the current in-process Python path.

AbstractCore should therefore support a clean local-server story:

- managed or documented local runtime startup;
- normal OpenAI-compatible routing from Core into that runtime;
- consistent model/provider metadata showing that acceleration is happening remotely, not inside the
  in-process provider.

## Why

This keeps the system honest and useful:

- users get real Apple Silicon speed improvements sooner through draft-model speculation;
- MTP-capable catalogs remain accurate without implying unsupported acceleration;
- Core keeps one coherent Python/server contract instead of adding ad hoc provider toggles;
- future runtime upgrades can plug into an existing abstraction rather than forcing another API
  redesign.

## Evidence needed before promotion

Promote this to `planned/` when at least one of these is true:

- AbstractCore is ready to expose `speculation` on the Python API surface.
- `MLXProvider` can load and use a draft model end-to-end.
- a target external local runtime used by AbstractCore users exposes stable native MTP serving.
- upstream `mlx-lm` adds real Qwen3.6 embedded-MTP execution support.

## Suggested implementation

1. Add capability metadata fields and resolution warnings.
2. Add typed request/result support for speculation metadata.
3. Implement MLX draft-model speculation with unit tests and gated local smoke tests.
4. Document the difference between:
   - standard generation
   - draft-model speculation
   - native MTP
5. Add provider/runtime notes to model catalog entries for MTP-tagged models.

## Non-goals

- Do not claim that all MTP-labeled models are accelerated in all providers.
- Do not add a fake `mtp=True` flag that silently degrades to normal generation without metadata.
- Do not fork large upstream runtimes unless there is clear evidence that the maintenance burden is
  justified.
- Do not conflate multimodal support requirements such as `mmproj` with text-side acceleration
  support.

## Validation ideas

- unit tests for speculation request normalization and metadata reporting;
- MLX provider tests verifying draft-model path selection;
- gated local Apple Silicon benchmarks comparing:
  - baseline MLX generation
  - MLX draft-model speculation
  - external local native-MTP runtime, when available
- server tests confirming that OpenAI-compatible endpoints use the same Core generation path and
  expose consistent warnings/metadata.

## Guidance for future agents

Before implementing native MTP in any provider, inspect the actual installed runtime code rather
than relying on model names or repo descriptions. In particular:

- verify whether MLX runtimes consume or discard `mtp.` weights;
- verify whether GGUF runtimes expose native MTP through the Python binding in use;
- prefer real measured acceleration over catalog-level assumptions.

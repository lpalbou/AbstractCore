# 0850-abstractcore: [FEATURE] Execute native MTP through the Transformers backend

> Created: 2026-09-20
> Status: Planned
> Type: feature
> Priority: P2
> Labels: native-inference, huggingface, mtp

## Summary

Implement a native MTP adapter inside the existing HuggingFace provider's
Transformers backend under the shared speculation contract.

## Current code reality

Audited 2026-09-20: `providers/huggingface_provider.py` does not consume the shared
MTP execution request. Generic Transformers assisted generation is not by itself
a binding for these native heads. Head forward passes, target hidden-state access,
verification and hybrid/recurrent-cache recovery require architecture-specific support.

## Scope and non-goals

First select and pin a supported architecture/runtime pair; implement head loading,
drafting, verification and rollback with truthful depth/usage metadata. Capability
gates must refuse unsupported architecture/device/quantization combinations.
No MLX dependency, new provider, silent substitution of a separate small draft
model, or blanket promise for all Transformers models.

## Acceptance criteria

- [ ] A real supported target/head pair executes off and native MTP in process.
- [ ] Supported depths have correctness and performance evidence; rejected drafts restore every cache component.
- [ ] Streaming, EOS/stop, output budgets, cancellation and per-request state are covered.
- [ ] Strict unsupported requests fail; ordinary non-MTP generation stays unchanged.
- [ ] Vision and sampling support are explicitly validated or advertised unsupported.

## Testing

Run `python -m pytest tests/providers -k 'speculation or mtp'` and add adapter-specific
CPU contract tests plus a gated real-model off/on test. Do not count mocked draft
results as proof of native acceleration; report memory/latency separately.

## Dependencies and ADR status

Depends on 0849; inspect current Transformers model implementation before design.
ADR impact: None expected beyond the shared adapter decision in 0849.
Expected outcome: real HF MTP support for a stated, measured compatibility set.

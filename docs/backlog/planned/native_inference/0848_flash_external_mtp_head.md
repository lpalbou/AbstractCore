# 0848-abstractcore: [FEATURE] Load a separate matching Flash MTP head

> Created: 2026-09-20
> Status: Planned
> Type: feature
> Priority: P2
> Labels: native-inference, mlx, mtp

## Summary

Allow the existing MLX provider to combine an ordinary Qwen3.8-Flash-Next target
with a separate matching native MTP head, keeping the tested embedded-head path.

## Current code reality

Audited 2026-09-20: `MLXProvider._load_model` rejects `speculation.drafter` for
Qwen4-Exp and requires indexed embedded MTP tensors. Installed mlx-vlm 0.7.1 has
`speculative/drafters/qwen4_exp_mtp`, including compatibility checks. The published
`mlx-community/Qwen3.8-Flash-Next-4bit` target and
`sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` sidecar are candidates, NOT a
validated AbstractCore pair. The latter is approximately 5.21 GB of BF16 weights.
Do not confuse artifact disk size with resident memory.

## Scope and non-goals

Bind an explicitly selected or registry-resolved head through existing
`speculation.drafter`; validate target/head configuration, provenance and tensor
layout. Prefer official-derived or reputable conversions. Assess separately
quantized heads without requiring target and head to have identical precision.
No new provider, pmlx/oMLX server, target pruning, silent downloads or arbitrary
unverified head substitution. Preserve embedded loading and per-call off/depth.

## Acceptance criteria

- [ ] Ordinary target plus matching head loads and executes native MTP.
- [ ] Missing/incompatible head fails clearly; strict requests never silently fall back.
- [ ] Off/on and depths 2/3/4/5 are checked for text, images, streaming and cache isolation.
- [ ] Embedded-head regression tests remain green; actual outcome and head identity are reported.
- [ ] Record paired revisions, conversion provenance, memory and A/B timing with 60–120 second cooling intervals between GPU test blocks.

## Testing

Run `python -m pytest tests/providers/test_mtp_qwen4_loader_adversarial_unit.py tests/providers/test_mtp_qwen4_images_adversarial_unit.py`.
Add sidecar load/mismatch tests and an explicitly gated real target/head test.

## Dependencies and ADR status

Related: 0849 and 0847, `docs/speculative-decoding.md`, `providers/mlx_qwen4.py`.
ADR impact: None expected; preserve existing provider ownership. Revisit before
adding durable artifact-binding identifiers. Expected outcome: packaging choice
does not force a second copy of the target solely to obtain its head.

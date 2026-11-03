# 0878 — Train Kev-2B: the missing rung between Kev-0.8B and Kev-4B

> Type: feature (model)
> Created: 2026-11-01 (operator ask after the decision-model length ladder, item 0877)
> Priority: medium (routing between 0.8B and 4B already covers the gap; a 2B would simplify it)
> Labels: decision, kev, training, mlx, catalog, hub
> Links: [0877](../completed/0877_decision_models_output_modality.md), [docs/decision-models.md](../../decision-models.md),
> upstream recipe: https://github.com/jaredpalmer/kev (`kev.train`, `scripts/calibrate_checkpoint.py`, suite `evals/v7/decision-v7`)

## Why
The Kev family has 0.8B, 4B, 9B and 27B members and nothing between the first two. Measured on the length ladder
(whole documents, 1k–16k tokens, one request per document on an M5 Max): Kev-0.8B reads a 16k-token contract in
1.3 s with contract-clause accuracy 0.77 (mean over buckets) but reading comprehension 0.51; Kev-4B scores 0.88 / 0.78
at 5.2 s. A 2B member of the same family should land near 0.8B speed times two and a few points under the 4B,
which is the sweet spot for "answer several questions about a long document in a second".

## What
- Train a Kev-2B with the author's public recipe on `Qwen/Qwen3.5-2B-Base` (Apache-2.0): the `decision-v7` suite
  (10,000 examples from ten public classification / NLI / sentiment / yes-no sources, 896 generated policy examples,
  1,680 generated rule examples; dataset revisions pinned in the suite manifest), LoRA rank 16 on the attention, MLP
  and Gated DeltaNet projections, pointer head of width 256, two epochs, learning rate ~7e-5 (between the author's
  1e-4 for 0.8B and 5e-5 for 4B), `--batch 1 --accum 8 --dtype bf16 --checkpointing 1`; then the temperature fit on
  the development split (`scripts/calibrate_checkpoint.py`).
- Hardware: the operator's RTX 5070 Ti (16 GB) — expected one to two hours with `flash-linear-attention`, three to
  five times more on the eager DeltaNet path; the M5 Max is not the training machine (torch MPS runs the reference
  DeltaNet kernels; an MLX trainer for the packed pointer-head objective does not exist).
- Then the AbstractCore pipeline exactly as for 0.8B / 4B: `python -m abstractcore.decision.convert` to 8-bit and the
  single best 4-bit (sweep the 4B recipe and the gate/up variant; keep agreement ≥ 0.96), the public dev set,
  the balanced short set and the length ladder (`untracked/decision-eval/`), registry entry `kev-2b`
  (model_capabilities, catalog row), Hub upload to the AbstractFramework *Decision models* collection with the card
  template (why, accuracy protection, smallest usage, cache behaviour), docs tables.
- Not in scope: other base families. The recipe reads hidden states through a pointer head, so any decoder base
  works in principle (attention-only bases run the packed block-causal mask, hybrids the row form), but a new family
  needs its LoRA target names and mask path verified in `kev.train`, and the base licence checked against the
  framework's MIT/Apache preference (Gemma's terms are not Apache); file a separate item per family if wanted.

## Done when
- A `kev-2b` checkpoint (adapter + head.pt, and the merged MLX 8-bit / 4-bit) with its calibration temperature.
- Measured: dev-set accuracy / Brier / ECE, balanced short set, length ladder rows; expected dev accuracy ~0.80–0.82,
  ladder cost ~0.3 s at 8k and ~2.5 s at 16k tokens on an M5 Max.
- Registry, catalog, docs (`docs/decision-models.md` tables and the providers table) and the Hub collection updated;
  `tests/decision` and the registry suites green.

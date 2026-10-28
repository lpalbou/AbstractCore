# 0860 — The LFM2 and LFM2.5 families in the download catalog (GGUF and MLX, the 24B-A2B models included)

**Status**: completed (2026-10-28, 2.31.1) · **Priority**: P2 · **Created**: 2026-10-28
**Cross-package**: operator QA 2026-10-28 (the gateway console's Models catalog, item 4).

## Why
Searching "lfm" in the console's Models catalog showed one row: LFM2.5 2.6B, MLX only. The capability registry
carried most LFM2/LFM2.5 models, but `model_downloads_catalog.json` (what the catalog lists and downloads) had no
LFM2 row, no GGUF artifact for any Liquid AI model, and no `lfm2.5-24b-a2b` anywhere.

## What shipped
- `abstractcore/assets/model_downloads_catalog.json`: 11 rows (lfm2-350m, -700m, -1.2b, -2.6b, -8b-a1b, -24b-a2b;
  lfm2.5-350m, -1.2b-instruct, -1.2b-thinking, -8b-a1b, -24b-a2b), and the official GGUF Q4_K_M/Q8_0 on the
  existing lfm2.5-2.6b row. Each row carries GGUF (`huggingface`, `repo:QUANT`, LiquidAI official and unsloth) and
  MLX (`mlx`, mlx-community; LiquidAI's own MLX repos where mlx-community has none: LFM2.5-350M 4bit, LFM2-24B-A2B
  8bit, both LFM2.5-24B-A2B). Every added artifact is `upstream: {method: hf_api, checked: 2026-10-28, revision}`
  with `download_bytes` = the selected GGUF file or the MLX repo's safetensors. A Q4_K_M is taken from unsloth
  where LiquidAI's repo also ships a `-Q4_K_M-hip-optimized.gguf` (the `*Q4_K_M*.gguf` download pattern would
  fetch both); every artifact's pattern selects exactly one file (checked by the generator).
- `abstractcore/assets/model_capabilities.json`: new `lfm2.5-24b-a2b` (card + config.json: 24.0B total, 2.45B
  active, 64 experts top-4, 131,072 context, always-thinking, prompted tools; generation temperature 0.1, top_k 50,
  repetition_penalty 1.05); 30 repo-id aliases added to the matching entries (no alias owned by two entries).
- Not added, because they do not exist upstream (checked): unsloth GGUF for LFM2-2.6B, LFM2-24B-A2B, LFM2.5-350M,
  LFM2.5-2.6B; unsloth LFM2.5-8B-A1B and LFM2.5-24B-A2B have no plain Q4_K_M (UD quants only; their Q8_0 is listed);
  mlx-community has no LFM2.5-24B-A2B, no LFM2-24B-A2B 8bit and no LFM2.5-350M 4bit.

## Evidence
- Repo ids verified on 2026-10-28 with one Hugging Face API request each (`GET /api/models/<repo>?blobs=true`;
  probe and generator: framework root `untracked/release-closing/w5/lfm/probe.py`, `apply.py`, `probe.json`):
  `LiquidAI/LFM2-1.2B-GGUF` (revision `5399e76c648f`),
  `LiquidAI/LFM2-2.6B-GGUF` (revision `a759abdc5955`),
  `LiquidAI/LFM2-24B-A2B-GGUF` (revision `6247d03b87c2`),
  `LiquidAI/LFM2-24B-A2B-MLX-8bit` (revision `387dd98f0130`),
  `LiquidAI/LFM2-350M-GGUF` (revision `8fdc9d526b7e`),
  `LiquidAI/LFM2-700M-GGUF` (revision `fd39e80d7a5a`),
  `LiquidAI/LFM2-8B-A1B-GGUF` (revision `145e98524a2e`),
  `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` (revision `8ed288026e23`),
  `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` (revision `9584426effbf`),
  `LiquidAI/LFM2.5-2.6B-GGUF` (revision `e7caca5d835a`),
  `LiquidAI/LFM2.5-24B-A2B-GGUF` (revision `0912d33141f7`),
  `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` (revision `667408d494bd`),
  `LiquidAI/LFM2.5-24B-A2B-MLX-8bit` (revision `67ab417dd02d`),
  `LiquidAI/LFM2.5-350M-GGUF` (revision `657e078c9408`),
  `LiquidAI/LFM2.5-350M-MLX-4bit` (revision `f6cb4e006bb7`),
  `LiquidAI/LFM2.5-8B-A1B-GGUF` (revision `49c148317070`),
  `mlx-community/LFM2-1.2B-4bit` (revision `3843e4ad0fcb`),
  `mlx-community/LFM2-1.2B-8bit` (revision `9384737a4b91`),
  `mlx-community/LFM2-2.6B-4bit` (revision `493071ebca35`),
  `mlx-community/LFM2-2.6B-8bit` (revision `bf50a6b1531e`),
  `mlx-community/LFM2-24B-A2B-4bit` (revision `eede03bcde76`),
  `mlx-community/LFM2-350M-4bit` (revision `18dc72abf3b2`),
  `mlx-community/LFM2-350M-8bit` (revision `3b5014a1966c`),
  `mlx-community/LFM2-700M-4bit` (revision `6b32895938db`),
  `mlx-community/LFM2-700M-8bit` (revision `cc0f7efcc1c1`),
  `mlx-community/LFM2-8B-A1B-4bit` (revision `3266a85e9dab`),
  `mlx-community/LFM2-8B-A1B-8bit` (revision `6b0ba099bc88`),
  `mlx-community/LFM2.5-1.2B-Instruct-4bit` (revision `dee2f8a2786e`),
  `mlx-community/LFM2.5-1.2B-Instruct-8bit` (revision `e8c415057d2e`),
  `mlx-community/LFM2.5-1.2B-Thinking-4bit` (revision `81e15e78b44e`),
  `mlx-community/LFM2.5-1.2B-Thinking-8bit` (revision `35175c7b717f`),
  `mlx-community/LFM2.5-2.6B-4bit` (revision `72a08228cf70`),
  `mlx-community/LFM2.5-2.6B-8bit` (revision `1f4efae85773`),
  `mlx-community/LFM2.5-350M-8bit` (revision `1227587c17a9`),
  `mlx-community/LFM2.5-8B-A1B-MLX-4bit` (revision `146590a491db`),
  `mlx-community/LFM2.5-8B-A1B-MLX-8bit` (revision `47bba3e48186`),
  `unsloth/LFM2-1.2B-GGUF` (revision `154016c6696a`),
  `unsloth/LFM2-350M-GGUF` (revision `7ed673319287`),
  `unsloth/LFM2-700M-GGUF` (revision `8a92b6d90fc8`),
  `unsloth/LFM2-8B-A1B-GGUF` (revision `01c1c9ba8073`),
  `unsloth/LFM2.5-1.2B-Instruct-GGUF` (revision `b01cc872e591`),
  `unsloth/LFM2.5-1.2B-Thinking-GGUF` (revision `0e858c248236`),
  `unsloth/LFM2.5-24B-A2B-GGUF` (revision `aacc77834f6a`),
  `unsloth/LFM2.5-8B-A1B-GGUF` (revision `37563684c4a6`).
- `tests/config/test_model_catalog.py::test_the_lfm_families_are_in_the_catalog_with_gguf_and_mlx`: `catalog("lfm")`
  lists the 12 LFM2/LFM2.5 rows, each with a GGUF and an MLX artifact and its size; `lfm2.5-24b-a2b` context
  131072; repo aliases present. Red against the previous assets (1 failed); the catalog validator passes.
- `tests/config/test_recommended_text_tiers.py::test_every_upstream_artifact_is_eight_bit_or_a_tier_or_a_reverified_id`
  (the "no row was invented" guard) lists the 25 non-8-bit LFM artifacts explicitly as re-verified ids. Full suite:
  6731 passed; the only failures are the 2 pre-existing environment ones (kit theme drift, bloc e2e fixture).

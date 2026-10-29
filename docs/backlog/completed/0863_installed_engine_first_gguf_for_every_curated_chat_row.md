# 0863 — A host's installed engine is offered first: GGUF builds for every curated chat row

**Status**: completed (2026-10-29, unreleased) · **Priority**: P1 · **Created**: 2026-10-29
**Cross-package**: operator QA on the OVH release-test server (AbstractGateway console, Models catalog, front item 3).

## Why
On the Linux server llama.cpp is Ready and neither LM Studio nor Ollama is installed. The console's Models catalog
showed "Hugging Face 0", and the row for the recommended model, Qwen3.5 9B, listed only LM Studio and Ollama
builds, with the LM Studio build pre-selected. Two causes:
- 39 curated chat/coding rows had no GGUF artifact (`huggingface` `repo:QUANT`, served by llama.cpp), among them
  every Qwen3.5 row, Qwen3.6 27B/35B-A3B, Gemma 3/3n/4, Llama 3.3/4, Mistral Small 3.2, Devstral, Magistral, the
  DeepSeek R1 distills, gpt-oss-120b, Phi-4 mini, Granite 4.x, GLM-4.7 Flash and MiniMax M2.
- `_pick_recommended` let a row's curated pick (the LM Studio starter build) win whenever the host *could* run
  its engine, even when that engine was absent and another artifact's engine was installed. Artifacts were listed
  in seed order, with no regard to which engine is installed.

## What shipped
- `abstractcore/assets/model_downloads_catalog.json`: 39 GGUF artifacts, one per row: Q4_K_M (gpt-oss-120b: the
  upstream MXFP4, its only full-model quant), mostly unsloth. Each is `upstream: {method: hf_api, checked:
  2026-10-29, revision}`, and its `download_bytes` is the sum of the files the `*QUANT*.gguf` download pattern
  selects: one file, or the shards of that one quant (Llama 4 Scout 2, MiniMax M2 3). Not added: Qwen3.8
  Flash-Next, whose hub repo has no plain Q4_K_M (only UD quants and the MTP drafter's Q4_K_M files, which the
  pattern would fetch on their own).
- `abstractcore/config/model_catalog.py`:
  - `_pick_recommended`: a curated pick whose engine is not installed yields when a downloadable artifact on the
    row is served by an installed engine. The fallback rank puts an installed engine before the fit verdict,
    unless that artifact is `too_large`.
  - `_order_for_installed_engines` (new): every artifact carries `engine_installed`, and a row lists the
    pre-selection first, then the artifacts of installed engines, then the rest (a stable sort).
  - The Apple text tiers and the forced engine-fallback picks are unchanged.
- `docs/models.md`: the pre-selection and ordering rule.

## Evidence
- `tests/config/test_model_catalog.py::test_a_host_with_only_llamacpp_is_offered_the_gguf_first`: on a CUDA host
  where only llama.cpp is installed, qwen3.5-9b pre-selects and lists first `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`, and
  every row lists installed-engine artifacts first. Mutation-checked: it goes red without the curated yield, and
  red without the sort.
- `test_every_curated_text_row_has_a_build_llamacpp_serves`: the only chat/coding row without a llama.cpp build
  is qwen3.8-flash-next.
- `test_the_portable_starter_is_preselected_off_apple_silicon` now runs with no engine installed. The LM Studio
  starter still leads there.
- `tests/config`: 1025 passed.
- Repo ids, sizes and revisions were read with one Hugging Face API request each
  (`GET /api/models/<repo>?blobs=true`; probe scripts and output in the framework root
  `untracked/release-closing/design/catalog-scratch/gguf-0863/`):
  `unsloth/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M` (qwen3-vl-4b, revision `00c00da0690c`),
  `unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M` (qwen3-vl-8b, revision `b93a7ee71375`),
  `unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF:Q4_K_M` (qwen3-vl-30b-a3b, revision `0af19e747985`),
  `unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M` (qwen3.5-0.8b, revision `6ab461498e20`),
  `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` (qwen3.5-9b, revision `3885219b6810`),
  `unsloth/Qwen3.5-27B-GGUF:Q4_K_M` (qwen3.5-27b, revision `3221f178a6b8`),
  `unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M` (qwen3.5-35b-a3b, revision `bc014a17be43`),
  `unsloth/Qwen3.6-27B-GGUF:Q4_K_M` (qwen3.6-27b, revision `82d411acf4a0`),
  `unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M` (qwen3.6-35b-a3b, revision `a483e9e6cbd5`),
  `unsloth/Llama-3.3-70B-Instruct-GGUF:Q4_K_M` (llama-3.3-70b, revision `8f14c5c5d06f`),
  `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q4_K_M` (llama-4-scout, revision `72a6853f56a6`),
  `unsloth/gemma-3-1b-it-GGUF:Q4_K_M` (gemma-3-1b, revision `f0b45be0aac4`),
  `unsloth/gemma-3-4b-it-GGUF:Q4_K_M` (gemma-3-4b, revision `5a3566e716d8`),
  `unsloth/gemma-3-12b-it-GGUF:Q4_K_M` (gemma-3-12b, revision `d15e4c7dc21d`),
  `unsloth/gemma-3-27b-it-GGUF:Q4_K_M` (gemma-3-27b, revision `7cd0121f2530`),
  `unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M` (gemma-3n-e4b, revision `90fa8b0e431f`),
  `unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M` (gemma-4-26b-a4b, revision `c099eb48e663`),
  `unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q4_K_M` (mistral-small-3.2-24b, revision `b750ec229922`),
  `unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:Q4_K_M` (devstral-small-2-24b, revision `6e458b8add42`),
  `unsloth/Magistral-Small-2509-GGUF:Q4_K_M` (magistral-small-2509, revision `77fe877ee8e4`),
  `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M` (deepseek-r1-distill-qwen-1.5b, revision `3cb4d15544a2`),
  `unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M` (deepseek-r1-0528-qwen3-8b, revision `eb48357c179d`),
  `unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M` (deepseek-r1-distill-qwen-14b, revision `7b05b58b41f6`),
  `unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M` (deepseek-r1-distill-qwen-32b, revision `1938d05cc893`),
  `ggml-org/gpt-oss-120b-GGUF:MXFP4` (gpt-oss-120b, revision `7200b74bea78`),
  `unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M` (phi-4-mini, revision `78eb92a46fc3`),
  `unsloth/SmolLM2-135M-Instruct-GGUF:Q4_K_M` (smollm2-135m, revision `9e6855bc4be7`),
  `unsloth/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M` (smollm2-1.7b, revision `e933f1cdf73c`),
  `unsloth/granite-4.0-h-tiny-GGUF:Q4_K_M` (granite-4.0-h-tiny, revision `56b37fdc5289`),
  `unsloth/GLM-4.7-Flash-GGUF:Q4_K_M` (glm-4.7-flash, revision `0d32489ecb9d`),
  `unsloth/QwQ-32B-GGUF:Q4_K_M` (qwq-32b, revision `d5faedaaaae5`),
  `unsloth/granite-4.1-8b-GGUF:Q4_K_M` (granite-4.1-8b, revision `6f9671f73eb0`),
  `unsloth/MiniMax-M2-GGUF:Q4_K_M` (minimax-m2, revision `06e952eab9e8`),
  `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M` (deepseek-coder-v2-lite, revision `8f248fa20723`),
  `second-state/StarCoder2-3B-GGUF:Q4_K_M` (starcoder2-3b, revision `7fca3e2da2ce`),
  `tiiuae/Falcon3-7B-Instruct-GGUF:Q4_K_M` (falcon3-7b, revision `b5eb297a5235`),
  `bartowski/c4ai-command-r7b-12-2024-GGUF:Q4_K_M` (command-r7b, revision `bfc7a934c45c`),
  `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF:Q4_K_M` (exaone-3.5-7.8b, revision `c618bf673381`),
  `allenai/OLMo-2-1124-7B-Instruct-GGUF:Q4_K_M` (olmo-2-7b, revision `410e0069f648`).

## Follow-ups
- Not every installed llama-cpp-python build loads every architecture (Qwen3.5's hybrid attention, Gemma 3n,
  Granite 4.0-H, GLM-4.7). The architecture-support check before download or selection is batch item (f) of the
  same QA. These artifacts are `verified: false` until a real load on the server.
- Vision on llama.cpp needs the repo's `mmproj` file, which the Q4_K_M pattern does not fetch. Text works without
  it.

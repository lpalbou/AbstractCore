# 0865 — Every quant of a model, on demand (`extended`)

**Status**: completed (2026-10-29, unreleased) · **Priority**: P1 · **Created**: 2026-10-29
**Cross-package**: operator QA batch item (b), "the quantization filter gets All so any quant can be installed". The
AbstractRuntime facade passes `extended` through; the gateway console's quant filter **All** asks for it. Built
fresh from main by decision of the coordinator. The paused PR #8 (inputs/outputs, the 1bit class) stays with its
owner.

## Why
For each GGUF repo the catalog listed one or two quants (Q4_K_M, Q8_0), while unsloth ships UD-IQ1 to IQ4,
Q2-Q6 K_XL and more. On a 16 GB card the right build may be, for example, UD-Q3_K_XL, and it could not be installed
from the catalog.

## What shipped
- `abstractcore/config/model_catalog.py`:
  - `gguf_quant_labels(facts)` turns a repo's file list into `{label: total bytes}`.
    - Shards are summed, and `mmproj` files are skipped.
    - A label is kept only when its `*LABEL*.gguf` download pattern selects exactly that label's files.
      `Q6_K` also matches `UD-Q6_K_XL`, `F16` matches `BF16` and the projector, and an MTP drafter file rides its
      quant's name; such labels are left out.
  - `_extended_quant_artifacts` adds those quants for each of the row's own HF GGUF repos (the seed's curated
    sources: the official organisation or the community builds already named). Each gets the normal
    `_build_artifact` treatment: size, fit, presence and `cli_download`, plus `extended: true`.
  - `catalog(extended=)` (it needs `hub`, whose facts are cached 24 h): no request per row beyond the facts the hub
    lookup already reads.
  - `_pick_recommended` never pre-selects an extended quant.
- `abstractcore/server/host_routes.py`: the `extended` query parameter.
- `docs/models.md`.

## Evidence
- `tests/config/test_model_catalog.py`:
  - `test_extended_lists_every_quant_of_the_rows_own_gguf_repos`: BF16 (two shards summed), Q3_K_M, UD-IQ1_S and
    UD-Q6_K_XL are listed; Q6_K (over-fetch) and mmproj are not; nothing is pre-selected; without `extended`
    nothing is added. It goes red without the exact-pattern rule and red without the flag (mutation-checked).
  - `test_an_extended_quant_is_never_the_preselection`: red without the exclusion.
- `tests/config`: 1024 passed.
- Live, `qwen3-8b` on a CUDA host: 21 extra unsloth quants (UD-IQ1_S at 2.3 GB to BF16 at 16.4 GB), each with its
  fit.

## Follow-ups
- UD-IQ1 labels read `quant_class: unknown` until the 1bit class (PR #8) lands.
- Labels left out for over-fetching (Q6_K beside UD-Q6_K_XL) would need a file-exact download; not attempted.

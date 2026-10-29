# 0864 — Hugging Face search: pages, a count, and an exact handle that always finds the model

**Status**: completed (2026-10-29, unreleased) · **Priority**: P1 · **Created**: 2026-10-29
**Cross-package**: operator QA on the OVH release-test server (AbstractGateway console, Models catalog: front item 4
and batch item a). The console's paging control and count are in AbstractGateway.

## Why
- The Hub search returned one fixed page: 20 results per lane, with no way to see more and no count.
- Searching the exact handle `Qwen/Qwen3.5-9B` did not return that repo. The GGUF lane searched the full handle,
  and the base checkpoint is not a GGUF repo.
- The curated row was missing too. `_matches_query` split the query only on whitespace while the row's words are
  split on `/ : @ _ , -`, so a token like `qwen3.5-9b` or `qwen/qwen3.5-9b` never matched a word. The console's
  Hugging Face mode asks the catalog with the typed text, so it showed no catalog result beside the Hub's.

## What shipped
- `abstractcore/config/model_catalog.py`:
  - `_hub_search(..., page, page_size)` asks each lane for the results up to the end of the page plus one (most
    downloaded first, 200 deep at most). It caches per depth, returns that page's slice, and reports whether more
    exist.
  - A handle query searches its name part.
  - `_hub_handle_hit` classifies the exact repo: GGUF goes to llama.cpp; an MLX build goes to MLX; safetensors goes
    to a Transformers checkpoint, fetched whole. Anything else is not listed.
  - `catalog(hub_page=, hub_page_size=)` looks the handle up on page 1 and lists it first (`exact: true`).
  - `hub.search = {query, page, page_size, rows, has_more, handle, handle_found}`.
  - The query tokens split like the rows.
- `abstractcore/server/host_routes.py`: `hub_page` and `hub_page_size` on `/acore/models/catalog`.
- Docs: `docs/models.md` (the Hugging Face section) and `docs/server.md`.

## Evidence
- `tests/config/test_model_catalog.py`. Each test goes red with its fix removed (mutation-checked).
  - `test_hub_search_pages_and_says_how_many_and_whether_more`: 45 results come as pages of 20, 20 and 5, with no
    repeats and `has_more` true, true, false. Red with the page offset removed.
  - `test_an_exact_handle_finds_the_repo_and_its_builds`: the exact repo comes first, then the GGUF build of its
    name, and the name is what gets searched. Red without the handle lookup and without the name split.
  - `test_a_handle_that_does_not_exist_still_searches_its_name`.
  - `test_a_handle_or_hyphenated_name_finds_the_curated_row`: red with whitespace-only tokens.
- Live check against the Hub (2026-10-29), `catalog("Qwen/Qwen3.5-9B", hub=True)` on a CUDA host:
  - Results: the curated `qwen3.5-9b` row, then `hf:qwen/qwen3.5-9b` (exact), then 20 GGUF results
    (`unsloth/Qwen3.5-9B-GGUF`, `lmstudio-community/Qwen3.5-9B-GGUF`, `bartowski/Qwen_Qwen3.5-9B-GGUF`, ...).
  - `hub.search.has_more` is true.
  - `catalog("qwen", hub_page=2)` added 17 rows (repos already listed and GGUF repos with no recognisable quant
    are skipped) and `has_more` is true.

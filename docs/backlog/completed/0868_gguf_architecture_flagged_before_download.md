# 0868 — A Hugging Face GGUF this computer's llama.cpp cannot load is flagged before download

**Status**: completed (2026-10-29, unreleased) · **Priority**: P1 · **Created**: 2026-10-29
**Cross-package**: follows 0867 (installed rows, lpalbou/AbstractCore#19, stacked on it). The gateway console greys
**Download** and **Use as default** for such a row, with the sentence as tooltip (gateway PR, same day).

## Why
On the OVH server XHToken/Spark-X2.6-4B-GGUF downloaded (2.5 GB) and only failed at Test: its architecture
`spark2_5` is unknown to that llama-cpp-python build. 0867 says so once the model is on disk; the Hub already knows
the architecture, so the download itself should not be offered.

## What shipped
- `model_catalog._hub_model_facts` keeps `gguf_architecture` from the Hub's `gguf` block (`model_info`).
- `_build_artifact`: for a llama.cpp build whose Hub facts name an architecture, checks it with
  `gguf_unloadable_reason` (0867's libllama scan). When it cannot load, the artifact carries `loadable: false`,
  `unloadable_reason` and `downloadable: false`, gets a fit note, and is never the pre-selection
  (`_pick_recommended` takes downloadable artifacts only). A known architecture is `loadable: true`. When it
  cannot be told, `loadable` is null and the build stays offered (fail-safe, as in 0867).
- `docs/models.md` documents the three fields.

## Evidence
`tests/config/test_model_catalog.py::test_a_hub_gguf_this_llamacpp_cannot_load_is_flagged_before_download`, using
the FakeHfApi with `gguf.architecture`. It covers four cases:
- spark2_5 is refused with the sentence;
- qwen3 is loadable;
- no architecture from the Hub claims nothing;
- an unknown answer from the check claims nothing.

Mutation-checked: without `downloadable = False` the test fails. `tests/config` passes, apart from 4 voice-probe
tests that fail on the base branch too (environment).

## Not here
Hub entries cached before this change carry no architecture until their TTL expires; they read `loadable: null`.

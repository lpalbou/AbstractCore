# 0867 — A GGUF this computer's llama.cpp cannot load says so before it is chosen

**Status**: completed (2026-10-29, unreleased) · **Priority**: P1 · **Created**: 2026-10-29
**Cross-package**: operator QA batch item (f). Stacked on 0866 (lpalbou/AbstractCore#18): the installed row's
GGUF facts. The gateway console shows the reason and refuses **Use as default** for such a model.

## Why
On the OVH server XHToken/Spark-X2.6-4B-GGUF downloaded and could be set as a route; it failed only at Test,
because its `general.architecture` is `spark2_5`, which that llama-cpp-python build does not know.

## What shipped
- `abstractcore/utils/model_cache.py`:
  - `llamacpp_knows_architecture(arch)` reads llama-cpp-python's `libllama` bytes, found with `find_spec`
    without importing the package (no GPU init), cached by file identity. `libllama.*` is preferred over another
    `llama.*` file. It checks whether a C string ENDS with the architecture name: the linker merges a name into
    the tail of a longer one ("qwen2" in the server's libllama), so a leading NUL cannot be required. The error
    is fail-safe: a false "known" leaves a model offered, and never produces a false "cannot load".
  - The table is trusted only when the sentinels `llama` and `qwen2` are found the same way. A library laid out
    differently (stripped, a stub) gives None, so nothing is claimed (merge desk review).
  - It returns None without llama-cpp-python.
  - `gguf_unloadable_reason(arch)` gives the sentence, naming the installed llama-cpp-python version.
- `abstractcore/config/model_materializer.py` `_gguf_kind_tasks`: an installed GGUF's row carries `architecture`.
  When llama.cpp does not know it, the row also carries `loadable: false` and `unloadable_reason`.

## Evidence
- Live on the server, checked against its real libllama: `qwen35`, `gemma4`, `qwen3`, `llama`, `lfm2`, `hy_v3` and
  `qwen3tts` are known; `spark2_5` is not.
- `tests/config/test_installed_gguf_tasks.py`:
  - `test_a_gguf_this_llamacpp_cannot_load_says_so`, with a stand-in library: spark2_5 reads loadable false with
    the sentence; qwen35 has no flag; without llama-cpp-python nothing is claimed.
  - `test_a_library_without_the_table_claims_nothing`: no sentinel, so no flag. Red without the sentinel check
    (mutation-checked).
  - `test_libllama_is_preferred_over_another_llama_file`.
  - `test_the_real_llama_library_knows_its_own_architectures` runs against this machine's real library: `llama` is
    known, a made-up name is not.
- `tests/config` passes.

## Follow-ups
- Before download: the Hugging Face API serves `gguf.architecture` per repo. Flagging a catalog or search artifact
  as not loadable here is the next step (hub facts, then `_build_artifact`).

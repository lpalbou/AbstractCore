### Changed

- **The model catalog never offers MLX off Apple silicon** (operator 2026-10-29: "on Linux never offer MLX"): the
  mlx / mlx-gen / mlx-vlm artifacts are left out on a host whose engine registry says MLX cannot run (they were
  listed, not downloadable), and a row with only MLX builds (the Wan video rows) is not listed there.

### Fixed

- **A GGUF-only model in the Hugging Face cache says what it is** (backlog 0866, operator QA item c). The repos the
  catalog downloads hold only the quant file, with no config or card, so `models list` reported them as an unknown
  model with no task. The kind and tasks now come from the file's own `general.architecture`:
  - an architecture the catalog's GGUF repos use (`qwen35`, `gemma4`, `llama`, `lfm2`, ...): text generation; any
    other architecture stays unknown;
  - `qwen3-tts`: text to speech;
  - `flux`: text to image;
  - an embedding repo (its name says so; Qwen3-Embedding reuses `qwen3`): text embedding.
  `tasks_source: "gguf"`. A model outside the catalog can then be offered as the default text model.

### Fixed

- **An embedding default that is not downloaded says so, once, and can be downloaded from the console.** The
  availability probe resolves an embedding route's catalog alias (`huggingface/qwen3-embedding`) to the repo the
  embedding manager loads (`Qwen/Qwen3-Embedding-0.6B`), so the route reads "not downloaded" with a
  `download_artifact` (it read "not checked": "'qwen3-embedding' is not an <org>/<repo> Hugging Face reference").
  A not-cached model raises its "download it first" `ModelNotFoundError` without an ERROR log line (one per
  account on every boot on the Linux test server); the caller says it.
- **An embedding default whose repo is not downloaded runs on the installed GGUF build of the same model.** When
  `Qwen/Qwen3-Embedding-0.6B` is absent and `Qwen/Qwen3-Embedding-0.6B-GGUF` is installed (the Linux test server),
  the embedding manager loads the GGUF through llama.cpp (`embedding=True`, CPU, L2-normalized, the most faithful
  quant first: f16, bf16, Q8_0, ...) under that repo's id (another build is another embedding space), and the
  Defaults grid reads "installed: served by the installed GGUF build ...". Measured there: 1024 dimensions,
  similarity 0.775 for two cat sentences vs 0.262 against a revenue sentence. Only without any build the route reads
  "not downloaded" with Download.
- **Structured output with a host's `cancel_event`** no longer fails on the HuggingFace GGUF lane with "got multiple
  values for keyword argument 'cancel_event'" (backlog 0861): the structured branch hands the event over once,
  under the provider's cancel kwarg.

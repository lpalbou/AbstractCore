### Added

- **A model no call has used for 60 minutes is unloaded; a kept-loaded model never is** (`providers/process_residency.py`,
  core ADR 0012, root backlog 1115).
  - **Scope:** HuggingFace (transformers and GGUF), MLX and in-process embedding models are unloaded by the
    process-wide eject (every holder, `gc`, the allocator cache) after the idle-unload time. The default is 60
    minutes (`DEFAULT_IDLE_UNLOAD_S = 3600`).
  - **Setting:** `set_idle_unload(seconds)` sets the time for the process (`0` or `None`: off); `get_idle_unload()`
    reads it.
  - **What keeps a model:** never a locked model (Keep loaded), never one a call runs on, never one a claimant is
    still building.
  - **Never cancels:** new calls to the model wait during the unload, the model is checked again with them held, and
    a call that started in between keeps it. A waiting call then reloads the model on demand. No check runs while
    a model loads.
  - **Log and timer:** each unload logs `unloaded <model> after <N> min idle; freed <X.Y GB>` (seconds under two minutes). The timer starts with
    the first load in the process.
  - **Rows:** `resident_model_states()` covers MLX and embeddings too (also for the ADR 0010 fit failsafe), with
    `last_used_at`, `in_use`, `locked_by` and `idle_unload_at`.

### Fixed

- **Every call counts as in use and stamps the model's last use** (`providers/generation_cancel.py`).
  - Before, a call without a cancel event was not registered and did not stamp `last_used_at`, so it could look
    idle while it ran.
  - MLX now gives every call a private cancel event, as HuggingFace does.
  - Embedding managers stamp their last encode.
  - The MLX `ttl_s` / `keep_alive` warning names the process-wide idle unload.

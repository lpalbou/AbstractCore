### Fixed

- **A GGUF prompt-cache eviction no longer clears a cache that a prefill is reading** (`huggingface_provider.py`).
  `prompt_cache_fork` / `prompt_cache_set` store a cache from the caller's thread. The store's after-set hook enforces
  the snapshot byte budget, and evicting a key clears that key's `cache_state`. Neither took the GGUF prompt-cache
  lock, so a prefill holding the lock on another thread could fail with "OrderedDict mutated during iteration" or
  miss its restore. The count-based LRU eviction had the same gap. Both store hooks now take
  `_gguf_prompt_cache_lock`, now a reentrant lock (the save site holds it while the eviction it triggers runs the
  hooks). Review of #37, M1.

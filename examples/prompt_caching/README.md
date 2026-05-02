# Prompt Caching Examples

## What This Folder Teaches

How to reuse work across turns when your prompt is large and mostly stable.

Instead of re-processing (prefilling) the entire prompt every time, prompt caching lets a provider
reuse an internal “prefix state” (often KV/past) and only process what changed.

AbstractCore exposes this as:
- provider-level prompt-cache control plane (capability-gated), and
- `CachedSession`, which applies it to a normal chat/session workflow.

## Prereqs

- MLX local models: `pip install "abstractcore[mlx]"`
- HF transformers/GGUF: `pip install "abstractcore[huggingface]"`

## Key AbstractCore Concepts

- `CachedSession`: session wrapper that keeps a prompt-cache key and updates it incrementally.
- “Box boundaries”: in these examples, *system prompt*, *tools*, *history*, and *file attachments* are treated as distinct blocks so you can keep stable parts stable.
- Provider capabilities: not all backends support prompt caching; `prompt_cache_strategy="auto"` falls back safely.

## How The Examples Work

At a high level:

1) Create a provider with `create_llm(...)`  
2) Wrap it in `CachedSession(prompt_cache_strategy="auto")`  
3) Add stable context (system/tools), then incrementally append user turns and attachments  
4) Observe cache stats to see whether the provider reused the prefix

## Scripts

- `cached_session_quickstart.py`
  - Demonstrates: the smallest “attach files + ask twice” loop with `CachedSession`, plus cache on/off comparison.
  - Takeaway: prompt caching is easiest to understand when you run the same workflow with caching disabled.

- `prompt_cache_repl_demo.py`
  - Demonstrates: an interactive REPL where you can attach files (`@path`) and ask multi-turn questions while watching cache stats.
  - How it works: `CachedSession.attach_files(...)` turns files into deterministic “file boxes” and appends them once; subsequent turns reuse the cached prefix.
  - Takeaway: this is the closest thing to “ai-space style iteration” inside AbstractCore’s core prompt-cache API.

- `prompt_cache_boxes_demo.py`
  - Demonstrates: a small “smoke demo” for (a) `CachedSession` KV reuse, and (b) module-based prefix caches via `prompt_cache_prepare_modules(...)`.
  - Takeaway: prefix caches are composable building blocks (system/tools prefix → fork → session updates).

## Key Takeaways

- Prompt caching is primarily about performance, not quality: it should preserve behavior while reducing repeated work.
- Keep stable parts of your prompt stable (system/tools) so caching can be effective.

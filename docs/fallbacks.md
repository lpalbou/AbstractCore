# Fallbacks

This document describes **best-effort fallbacks** AbstractCore uses when a provider/runtime does not expose (or does not reliably honor) a model’s native control surface.

The goal of a fallback is:

- Keep the public API stable (e.g. `thinking="none|low|medium|high"`)
- Prefer backend-native knobs when they exist
- Avoid “system prompt injection” where possible
- Be explicit about trade-offs and when behavior is only best-effort

## Not a fallback: which model gets loaded

There is **no fallback between model artifacts**. A model handle names an artifact, and
AbstractCore either loads that artifact or fails ([ADR 0009](adr/0009-model-handle-fidelity.md)).
Every fallback on this page changes *how* a request is served; none of them changes *what model
serves it*.

This was not always true. `HuggingFaceProvider` used to promote any handle to GGUF when a local
LM Studio Hub manifest existed and any GGUF could be resolved from the caches — so
`create_llm("huggingface", model="Qwen/Qwen3.6-27B")` silently loaded a 4-bit
`lmstudio-community/Qwen3.6-27B-GGUF` file on llama.cpp instead of the requested bf16
transformers weights. That promotion has been removed; the case now raises
`ModelArtifactMismatchError`.

If you were relying on it, say which artifact you want:

```python
# The GGUF that the LM Studio Hub alias points at:
create_llm("huggingface", model="Qwen/Qwen3.6-27B", model_type="gguf")
create_llm("huggingface", model="lmstudio-community/Qwen3.6-27B-GGUF")   # or name it directly
create_llm("huggingface", model="/path/to/Qwen3.6-27B-Q4_K_M.gguf")      # or the file itself

# The transformers weights the handle actually names:
create_llm("huggingface", model="Qwen/Qwen3.6-27B", model_type="transformers")
```

An explicit `:quant` selector is likewise honoured exactly or refused — `repo-GGUF:Q8_0` will
never quietly hand back `Q4_K_M`. Where a handle genuinely underdetermines the artifact (a GGUF
repository holding several quantizations and no selector), the default pick is logged at WARNING
on the `abstractcore.providers.huggingface` logger.

## Qwen3 / Qwen3.5: thinking (“reasoning”) toggle

### What upstream Qwen recommends

Qwen3’s official docs describe **two** ways to switch between thinking and non-thinking modes:

1) **Stateless hard switch (recommended for reliability)**  
   Append a final **assistant** message containing only:

   ```text
   <think>

   </think>

   ```

   This is **stateless** (applies to a single turn) and “strictly prevents” the model from generating thinking content.

2) **Stateful soft switch**  
   Add `/no_think` or `/think` to a user (or system) message. The model follows the most recent instruction across turns.

Reference: Qwen docs “Thinking & Non-Thinking Mode”.  
`https://qwen.readthedocs.io/en/stable/inference/transformers.html#thinking-non-thinking-mode`

### AbstractCore strategy

AbstractCore implements a layered approach for `thinking=...` on Qwen3/Qwen3.5:

0) **Unified request abstraction**

   Callers use the same public parameter everywhere:

   ```python
   llm.generate("...", thinking="off")
   ```

   `BaseProvider` normalizes `thinking=True|False|"auto"|"on"|"off"|"none"|level` and calls a
   provider hook, `_apply_provider_thinking_kwargs(...)`. Providers return both rewritten kwargs
   and a `ThinkingControlHandling` record so the base layer knows whether enable/disable and effort
   level were actually handled. This prevents a provider-native control from being followed by a
   second generic fallback marker.

1) **Backend-native knob (preferred)**  
   When the serving stack supports template kwargs, we send:

   - `chat_template_kwargs.enable_thinking = true|false`
   - and a compatibility alias `enableThinking = true|false`

   This is the “clean” approach because it aligns with Qwen’s chat templates and avoids injecting control tokens into the conversation.

2) **Provider-rendered hard switch for local Qwen runtimes**

   - MLX and HuggingFace transformers place the empty think block through their provider prompt
     renderers when `thinking="off"` and a native `enable_thinking` template kwarg is not available.
   - HuggingFace transformers passes `enable_thinking` into `tokenizer.apply_chat_template(...)`
     when the tokenizer supports it, and uses the same empty-think marker in its cache-delta
     renderer for Qwen-family cached generation.
   - HuggingFace GGUF exact-renderer paths do not mutate canonical history. The control-plane
     renderer inserts the empty think block at the final assistant generation prompt, so durable
     prompt-cache prefixes and live suffixes serialize consistently. Because that renderer owns
     the control, thinking-controlled GGUF requests are routed through it whether or not a
     `prompt_cache_key` is in play.

3) **Robust fallback for `thinking="off"/"none"` (LM Studio)**
   Some LM Studio builds ignore `chat_template_kwargs` for certain model formats. For that
   path AbstractCore also uses Qwen’s **stateless hard switch**, appending a final assistant
   “prefill” message containing the empty think block:

   - Implemented in `abstractcore/providers/base.py` (Qwen hard-switch marker injection).
   - Both artifacts express the same off state, so sending the template kwarg and the prefill
     together is consistent.
   - Providers that render prompts locally (MLX, HuggingFace transformers and GGUF) place the
     marker themselves at the generation boundary, which is where the model's own chat template
     puts it; they do not use this message-level fallback.

   Note: this fallback adds an extra assistant turn **in the outbound request only**. Callers should not persist that marker message as part of the canonical chat history.

4) **Why we do not rely on `/no_think` as the primary switch**

`/no_think` is a “soft” instruction and can be unreliable when:

- The instruction is not placed in a position the model “sees” as authoritative
- The serving stack rewrites prompts or inserts additional wrapper text
- The runtime ignores or alters the chat template behavior

The assistant-prefill hard switch is stateless and robust, and matches Qwen’s own documented method.

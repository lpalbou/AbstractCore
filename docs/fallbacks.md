# Fallbacks

This document describes **best-effort fallbacks** AbstractCore uses when a provider/runtime does not expose (or does not reliably honor) a model’s native control surface.

The goal of a fallback is:

- Keep the public API stable (e.g. `thinking="none|low|medium|high"`)
- Prefer backend-native knobs when they exist
- Avoid “system prompt injection” where possible
- Be explicit about trade-offs and when behavior is only best-effort

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

1) **Backend-native knob (preferred)**  
   When the serving stack supports template kwargs, we send:

   - `chat_template_kwargs.enable_thinking = true|false`
   - and a compatibility alias `enableThinking = true|false`

   This is the “clean” approach because it aligns with Qwen’s chat templates and avoids injecting control tokens into the conversation.

2) **Robust fallback for `thinking="off"/"none"` (LM Studio + llama.cpp/GGUF)**  
   Some local runtimes either:
   - do not expose template kwargs via API (e.g. `llama-cpp-python` today), or
   - may ignore `chat_template_kwargs` for some model formats (observed in some LM Studio builds)

   In those cases, AbstractCore uses Qwen’s **stateless hard switch** by appending a final assistant “prefill” message containing the empty think block:

   - Implemented in `abstractcore/providers/base.py` (Qwen hard-switch marker injection).
   - Used for Qwen3/Qwen3.5 on:
     - `LMStudioProvider`
     - `HuggingFaceProvider` when `model_type=="gguf"` (llama-cpp-python)

   Note: this fallback adds an extra assistant turn **in the outbound request only**. Callers should not persist that marker message as part of the canonical chat history.

3) **Why we do not rely on `/no_think` as the primary switch**

`/no_think` is a “soft” instruction and can be unreliable when:

- The instruction is not placed in a position the model “sees” as authoritative
- The serving stack rewrites prompts or inserts additional wrapper text
- The runtime ignores or alters the chat template behavior

The assistant-prefill hard switch is stateless and robust, and matches Qwen’s own documented method.


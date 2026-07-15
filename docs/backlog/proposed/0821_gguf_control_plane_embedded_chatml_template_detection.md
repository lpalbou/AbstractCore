# Proposed: GGUF control-plane detection for models with an embedded ChatML template

## Metadata
- Created: 2026-07-14
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
A GGUF model reaches the fast prompt-cache lane (`mode=local_control_plane`:
delta-only updates + prefill-snapshot generation) only when AbstractCore can
render its chat prompt byte-exactly. `_gguf_prompt_cache_control_plane_chat_format`
today recognizes three renderers:

- llama-cpp-python's built-in `chatml` / `chatml-function-calling`,
- built-in `llama-3`,
- the Gemma-4 `gemma_turn` path via the model's embedded Jinja chat template.

Everything else falls to `mode=keyed`, which is still correct and still gets
llama.cpp's own in-process `n_past` prefix reuse — it just loses AbstractCore's
snapshot control plane (fork, modules, durable blocs, and — since the
2026-07-14 fix — the persisted plain-generate snapshot).

## The gap (found live, 2026-07-14 fable5 GGUF verification)
`Ornith 1.0` GGUFs (`deepreinforce-ai/Ornith-1.0-{9B,35B}-GGUF`) are Qwen3.5
post-trains. llama.cpp reports their `chat_format` as `chat_template.default`
and ships a 7.5k-char embedded Jinja template that IS ChatML
(`<|im_start|>` / `<|im_end|>` markers present, verified). Because the detector
only admits the *built-in* `chatml`/`chatml-function-calling` ids (not an
arbitrary embedded template that happens to be ChatML), and the `gemma_turn`
branch is Gemma-specific, Ornith GGUFs land on `keyed`. The filename heuristic
`_gguf_prompt_cache_chat_format` (`"qwen" in model_lower → chatml-function-calling`)
also misses `ornith-*.gguf`. Net: a Qwen3.5-family model loses the control plane
on name + template-id alone, even though its template is renderable ChatML.

This is an OPTIMIZATION gap, not a correctness bug: Ornith GGUF loads, generates,
recalls facts, and reuses via llama.cpp-native prefix matching. It simply does
not get AbstractCore's snapshot lane.

## Current code reality
- `abstractcore/providers/huggingface_provider.py`
  - `_gguf_prompt_cache_chat_format` (filename fallback: qwen/coder → chatml-fc; llama3 → llama-3).
  - `_gguf_prompt_cache_control_plane_chat_format` (alias map + the gemma_turn embedded-template branch).
  - `_gguf_render_llama_cpp_chat_template_prompt` already renders an arbitrary embedded Jinja template via `Jinja2ChatFormatter` — the machinery to render Ornith's template EXISTS; it is only gated behind the `gemma_turn` architecture check.

## Options
1. **ChatML-marker detection on the embedded template.** When `chat_format`
   starts with `chat_template` and the embedded `tokenizer.chat_template`
   contains the ChatML markers (`<|im_start|>`/`<|im_end|>`), route to a
   ChatML control-plane renderer (reuse the existing chatml renderer, not the
   generic Jinja path, so tokenization matches the built-in exactly). Lowest
   risk, highest coverage — catches every ChatML GGUF regardless of the id
   llama.cpp guessed.
2. **Generalize the embedded-template control plane.** Drop the `gemma_turn`
   restriction on `_gguf_render_llama_cpp_chat_template_prompt` and let ANY
   model with an embedded template use it for the control plane, keyed by a
   template hash. Broadest, but needs a byte-exactness guard (the rendered
   text MUST tokenize identically to what generation would send) before it can
   be trusted for snapshot reuse — a mismatch is the silently-wrong-cache class.
3. **Do nothing.** `keyed` is correct; llama.cpp-native reuse already helps.

## Recommendation
Option 1, scoped to the ChatML marker case first (it covers the Qwen3.5-family
post-trains that motivated this, and ChatML is the single most common embedded
template). Gate strictly on the exact-tokenization guard: render via the ChatML
path, tokenize, and only claim the control plane if it round-trips. Option 2 is
the general form and can follow once the byte-exactness harness exists.

## Acceptance
- Ornith GGUF (and any embedded-ChatML GGUF) reports `mode=local_control_plane`.
- The rendered control-plane prompt tokenizes byte-identically to the
  generation prompt (pinned test with a real embedded ChatML template).
- No regression for Gemma-4 (`gemma_turn`), built-in chatml/llama-3, or the
  `keyed` fallback for genuinely unrenderable formats.

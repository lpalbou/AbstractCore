# 2026-03-30 — llama-cpp-python: expose `chat_template_kwargs` / `enable_thinking` for clean Qwen3/Qwen3.5 thinking toggle

## Goal

Expose a **backend-native**, per-request switch to enable/disable Qwen3/Qwen3.5 “thinking” in **llama-cpp-python** (GGUF) without relying on prompt/message injection.

Concretely, we want to be able to pass something like:

- `enable_thinking: true|false` (template kwarg)
- (optionally) `thinking_budget` / `reasoning_budget` (budget knob, if supported by the backend)

…through the llama-cpp-python Python API the same way llama.cpp server supports `--chat-template-kwargs`.

## Background / why this matters

Qwen3/Qwen3.5 official chat templates implement the thinking toggle as a **chat-template variable** named `enable_thinking`.

- When `enable_thinking == false`, the template inserts an empty “think block” in the assistant generation prompt:
  - `<think>\n\n</think>\n\n`
- When thinking is enabled, the template opens a think block (`<think>\n`) and the model is trained to generate internal reasoning inside it.

In llama.cpp server, this is controlled cleanly via `--chat-template-kwargs '{"enable_thinking": false}'`.

In llama-cpp-python today, `Llama.create_chat_completion()` does **not** accept/forward arbitrary template kwargs (or a dedicated `enable_thinking`), which forces downstream libraries to use workarounds (e.g., injecting `<think>\n\n</think>\n\n` as an assistant “prefill” message).

Workarounds are fragile and complicate a unified abstraction like `thinking="none|low|medium|high"` across providers.

## Proposal (upstream llama-cpp-python)

### Option A (recommended): explicit parameter

Add an explicit, typed parameter to `Llama.create_chat_completion()`:

- `chat_template_kwargs: Optional[Dict[str, Any]] = None`

and forward it into the selected chat handler call:

- `handler(..., **(chat_template_kwargs or {}))`

This is minimally invasive because:

- llama-cpp-python chat handlers already accept `**kwargs`
- `Jinja2ChatFormatter.__call__` already forwards `**kwargs` into Jinja’s `.render(...)`

### Option B: accept `**kwargs` and forward

Add `**template_kwargs` to the public signature and forward into the handler.

Downside: it is “too open”, makes the API less discoverable, and can hide typos.

### Option C: dedicate `enable_thinking` only

Add `enable_thinking: Optional[bool] = None` and forward it into the handler.

Downside: does not generalize (other templates use other knobs).

## Suggested follow-ups (nice-to-have)

If/when llama-cpp-python gains access to llama.cpp’s reasoning-budget controls (or implements an equivalent), consider:

- `reasoning_budget: Optional[int]` / `thinking_budget: Optional[int]`
- `reasoning_format: Optional[str]` (to populate `reasoning_content` separately)

This would let higher-level frameworks map `thinking="low|medium|high"` to **real token budgets** instead of “on/off only”.

## Impact for AbstractCore

Once upstream supports it, AbstractCore can:

- Map `thinking="none"` → `chat_template_kwargs={"enable_thinking": false}`
- Map `thinking="low|medium|high"` → `chat_template_kwargs={"enable_thinking": true}` + (optional) budget knob

This removes the need for the current “marker fallback”:

- `<think>\n\n</think>\n\n` injected as an assistant message for Qwen3/Qwen3.5 GGUF

## Test plan (upstream)

- Unit test with a Qwen3/Qwen3.5 GGUF that includes a Jinja chat template using `enable_thinking`.
- Verify:
  - `enable_thinking=false` produces prompts that include the empty think block in the assistant generation prompt
  - `enable_thinking=true` produces prompts that open `<think>\n`
  - No behavior change when `chat_template_kwargs` is omitted

## Notes

- This proposal intentionally targets llama-cpp-python’s **Python API**, not only the `llama_cpp.server`.
- It should be safe for non-Qwen models: template kwargs are ignored unless referenced by the template.


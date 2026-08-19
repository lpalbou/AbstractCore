# Reasoning Control

AbstractCore exposes one parameter — `thinking=` — for every provider that can reason, and
maps it to whatever control surface each backend actually honors. This page explains what
you can request, what each provider does with the request, and how to verify from the
response that your request took effect.

For the wider parameter vocabulary (`temperature`, `top_p`, `max_output_tokens`, …) see
[Generation Parameters](generation-parameters.md). For the registry fields referenced here,
see `abstractcore/assets/README.md`.

## Request a reasoning mode

```python
from abstractcore import create_llm

llm = create_llm("mlx", model="mlx-community/Qwen3.8-27B-4bit")

llm.generate("Plan the migration", thinking=None)      # model/server default
llm.generate("Plan the migration", thinking="off")     # no reasoning ("none" is an alias)
llm.generate("Plan the migration", thinking="on")      # reasoning enabled, model default depth
llm.generate("Plan the migration", thinking="low")     # brief reasoning
llm.generate("Plan the migration", thinking="medium")  # balanced
llm.generate("Plan the migration", thinking="xhigh")   # deepest, when the model offers it
```

Accepted values: `None` / `"auto"`, `True` / `"on"`, `False` / `"off"` / `"none"`, and the
levels `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` (`"extra high"` is an alias for
`"xhigh"`).

When a model advertises a specific effort enum, a level outside that enum is mapped to the
nearest supported one and the substitution is reported as a `RuntimeWarning`. Requesting
`"high"` from a model offering `low | medium | xhigh` therefore runs at `medium`.

Reasoning text, when the model emits it, is available as `response.reasoning`, and
`response.content` stays clean.

## What each provider sends

Two things are controlled separately: whether the model reasons at all, and how deeply.

| Provider | Enable / disable | Effort level |
|---|---|---|
| OpenAI | `reasoning_effort` (`"none"` when the model accepts it) | `reasoning_effort` |
| Anthropic | `thinking` block | `output_config.effort` (adaptive models) or a thinking budget |
| LM Studio | `chat_template_kwargs.enable_thinking` plus an assistant no-think prefill | `reasoning_effort` request field, which LM Studio maps into the model's chat template on every request shape |
| OpenAI-compatible servers | `chat_template_kwargs.enable_thinking` when the server accepts template kwargs; otherwise `reasoning_effort` | `chat_template_kwargs.<effort kwarg>`, or `reasoning_effort` for models with no template surface |
| vLLM | `extra_body.chat_template_kwargs.enable_thinking` | `extra_body.chat_template_kwargs.<effort kwarg>` plus `thinking_token_budget` |
| MLX | no-think prefill placed at the generation boundary of the rendered prompt | the template's own effort instruction rendered into the system block |
| HuggingFace (transformers) | `enable_thinking` passed to `apply_chat_template` | the effort kwarg passed to `apply_chat_template`; the cached renderer writes the effort instruction into the system block |
| HuggingFace (GGUF) | no-think prefill placed by the local renderer at the generation boundary | the model's embedded template rendered with the effort kwarg, or the effort instruction written into the system block for plain ChatML builds |
| Ollama | `think` | `think` level for GPT-OSS; other models run with reasoning enabled |
| GPT-OSS (Harmony) | `Reasoning:` system line (traces cannot be fully disabled) | `Reasoning: low \| medium \| high` |

Local providers serialize prompts themselves, so their controls are prompt artifacts rather
than API fields. AbstractCore reproduces the artifact the model's own chat template would
produce, which keeps a controlled request inside the model's training distribution.

## Model requirements

A level request is enforced only when the model's registry entry declares a control surface
for it — `thinking_control.effort_template_kwarg` for backends that render the template, and
`thinking_control.effort_system_lines` for lanes that serialize prompts locally, with
`reasoning_levels` listing the accepted enum. Qwen3.8 declares all three and supports
`low`, `medium`, and `xhigh`.

Models whose templates expose only an on/off switch — Qwen3.6, for example — accept
`thinking="off"` and `thinking="on"`, and treat a level request as "reasoning enabled" with a
`RuntimeWarning` explaining that effort scaling is unavailable. To add support for a new
model, declare its surfaces in `abstractcore/assets/architecture_formats.json` or
`model_capabilities.json`; no provider code changes are required.

## Verify what actually happened

Every response carries the request and its outcome in `response.metadata`:

```python
response = llm.generate("Plan the migration", thinking="low")
print(response.metadata["thinking_requested"])      # "low"
print(response.metadata["thinking_effective"])      # "low" when enforced
print(response.metadata["thinking_handled_level"])  # True when a real artifact was applied
```

- `thinking_requested` / `thinking_effective` — what you asked for, and what the request
  became after mapping.
- `thinking_handled_enable_disable` / `thinking_handled_level` — whether the on/off switch and
  the effort level were actually applied.
- `thinking_supported_levels` — the model's advertised effort enum.
- `thinking_supports_output` / `thinking_supports_control` — whether the model emits reasoning,
  and whether it exposes a request-side knob.

`thinking_effective` reports a level only when a control artifact reached the model. When a
request cannot be honored, the field reflects the weaker control that was applied and a
`RuntimeWarning` states what remains in effect, so a silently ignored request never reads as a
successful one.

## Requests that fall back

A few request shapes are served by a code path that carries no control artifact. AbstractCore
declines the claim for these rather than reporting an unenforced level:

- **Prefilled system blocs.** In KV cache modes where the system block is already committed to
  the cache, the effort instruction cannot be inserted. Disabling reasoning still works,
  because that control lives at the generation boundary.
- **HuggingFace GGUF with structured output, media, tool-role or content-part histories**, and
  when `ABSTRACTCORE_GGUF_CONTROL_PLANE=0` disables the local renderer. These requests are
  served by `llama-cpp-python`'s chat-completion path, which renders no control artifact.
- **Servers that reject `reasoning_effort`.** AbstractCore drops the field, retries the
  request once so it still succeeds, and warns that the level was not applied.

## Reasoning control and prompt caching

An effort instruction is part of the system block, so changing levels changes the first tokens
of the prompt. A durable prompt-cache bloc built at one effort level will not prefix-match a
request at another, and the prompt is recomputed. Pin one level per cached session to keep
prefixes warm. See [Prompt Caching](prompt-caching.md).

## Related documentation

- [Generation Parameters](generation-parameters.md) — the full parameter vocabulary and
  precedence rules.
- [Fallbacks](fallbacks.md) — the layered strategy behind the Qwen no-think switch.
- [Capabilities](capabilities.md) — what AbstractCore supports across providers.
- `abstractcore/assets/README.md` — the registry fields that drive all of the above.

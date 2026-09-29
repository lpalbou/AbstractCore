# 792 — A generation entry point that does not need a text model

> Type: task
> Created: 2026-09-29
> Priority: normal
> Labels: api, multimodal, ergonomics

## Summary

Generating an image, voice or music without configured defaults still requires a text LLM
instance first:

```python
llm = create_llm("mlx", model="mlx-community/Qwen3.5-4B-4bit")   # never used below
image = llm.generate("A watercolor lighthouse",
                     output={"modality": "image", "provider": "mlx-gen",
                             "model": "AbstractFramework/flux.2-klein-4b-8bit"})
```

`generate()` is a method on an LLM instance (`providers/base.py:9178`) and `create_llm` requires
a provider (`core/factory.py:10`), so the text model is a mandatory, unused entry point when every
call names its own output engine.

## Why

Operator (2026-09-29, website review): "why do you even need the first `llm = create_llm(...)`?"
The website now explains it honestly, but the API should not need the explanation.

## Options

- A module-level `abstractcore.generate(prompt, media=..., output=...)` that routes through the
  configured defaults and needs a text model only when the output is text.
- `create_llm()` with no provider resolves the configured default text route (and fails with a
  clear message when none is set).

Keep `llm.generate(...)` unchanged; the new entry point uses the same routing code.

## Acceptance criteria

- [ ] An image, voice or music call without a text model, with defaults and with an explicit
      output dict; a text call still requires a text route.
- [ ] Docs and the website examples use it where it removes the unused instance.

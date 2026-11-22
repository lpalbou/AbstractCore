### Added

- **Thinking Machines Inkling family in the capability registry** (`assets/model_capabilities.json`,
  `assets/architecture_formats.json`). Inkling 2 Mini, Inkling 2 Small, Inkling and Inkling Small, with their
  published repository names (`thinkingmachines/Inkling-2-Mini`, `thinkingmachines/Inkling-2-Small-MXFP8`,
  `thinkingmachines/Inkling-NVFP4`, ...) and GGUF or MLX re-packagings, resolve to the new `inkling` architecture.
  Tool calling is native, so OpenAI-compatible servers running vLLM or SGLang with `--tool-call-parser inkling`
  receive the tools in the request's `tools` field and return parsed `tool_calls`. Reasoning arrives in
  `message.reasoning`. `thinking` maps to `reasoning_effort`: `minimal`, `low`, `medium` and `high` on every model,
  plus `xhigh` on Inkling 2; `thinking="off"` sends `minimal` with a warning, because no Inkling effort value turns
  reasoning off. Entries declare a 1,048,576-token context, image input (and voice and sound input on Inkling and
  Inkling Small), and `max_output_tokens` 32,768.

### Fixed

- **Troubleshooting covers announced but missing tool calls** (`docs/troubleshooting.md`). When a server
  reports a model under a name the registry does not know, tools are sent in prompted mode, and servers that parse
  tool calls per model (vLLM or SGLang `--tool-call-parser`) return no call. The guide shows how to check the
  resolved `tool_support` and how to fix it.

### Fixed

- **Streamed native tool calls arrive complete** (`providers/stream_tool_calls.py`,
  `providers/openai_compatible_provider.py`, `providers/openai_provider.py`). With `stream=True`, OpenAI and
  OpenAI-compatible servers (vLLM, LM Studio, OpenRouter, Portkey) send a tool call as a named first delta
  followed by fragments of its arguments. The providers now join the fragments by index and deliver each call
  once, on the chunk that carries the `finish_reason`, as
  `{"id", "type", "name", "arguments": "<JSON string>"}`; parallel calls arrive together and in order, and a
  stream that ends without a finish reason still delivers its calls on a last chunk. A streamed call now
  carries the same name and arguments as the non-streamed one. If you assembled `delta.tool_calls` fragments
  from `chunk.tool_calls` yourself, read them from `chunk.raw_response` instead, or use the complete call.

### Added

- **`nvidia/Qwen3.8-27B-NVFP4-QAD` is an alias of `qwen3.8-27b`** (`assets/model_capabilities.json`), so
  NVIDIA's quantization-aware-distilled NVFP4 build resolves to the registered model by name.

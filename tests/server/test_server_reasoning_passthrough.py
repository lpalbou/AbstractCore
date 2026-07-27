"""Server boundary must not lose model reasoning (OpenAI-compat surface).

Non-streamed chat completions surface reasoning as `message.reasoning_content`
(the DeepSeek/vLLM/LM Studio convention — the same key AbstractCore's own
OpenAI-compatible provider reads back). Streamed responses forward incremental
`delta.reasoning_content` chunks without duplicating the trailing aggregate.
`/v1/models` entries expose registry-backed reasoning discovery so a coupled
provider -> model -> effort selector has a source of truth.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

from abstractcore.core.types import GenerateResponse
from abstractcore.server.app import (
    ChatCompletionRequest,
    _model_reasoning_discovery,
    convert_to_openai_response,
    generate_streaming_response,
)
from abstractcore.tools.syntax_rewriter import SyntaxFormat, ToolCallSyntaxRewriter


def _rewriter() -> ToolCallSyntaxRewriter:
    return ToolCallSyntaxRewriter(SyntaxFormat.OPENAI)


def _sse_deltas(sse_lines: List[str]) -> List[Dict[str, Any]]:
    deltas: List[Dict[str, Any]] = []
    for line in sse_lines:
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[len("data: "):])
        for choice in payload.get("choices", []):
            deltas.append(choice.get("delta", {}))
    return deltas


def test_non_stream_response_carries_reasoning_content() -> None:
    response = GenerateResponse(
        content="Final answer",
        model="unit-test",
        finish_reason="stop",
        metadata={"reasoning": "Because of X."},
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    )

    out = convert_to_openai_response(response, "lmstudio", "unit-test", _rewriter(), "req-1")
    message = out["choices"][0]["message"]
    assert message["content"] == "Final answer"
    assert message["reasoning_content"] == "Because of X."


def test_non_stream_response_without_reasoning_has_no_reasoning_content() -> None:
    response = GenerateResponse(content="Final", model="unit-test", finish_reason="stop")
    out = convert_to_openai_response(response, "lmstudio", "unit-test", _rewriter(), "req-2")
    assert "reasoning_content" not in out["choices"][0]["message"]


def _fake_llm(chunks: List[GenerateResponse]) -> Any:
    def _generate(**_kwargs) -> Iterator[GenerateResponse]:
        return iter(chunks)

    return SimpleNamespace(generate=_generate)


def test_streaming_forwards_reasoning_deltas_without_duplicating_aggregate() -> None:
    chunks = [
        GenerateResponse(content="", model="m", metadata={"reasoning_delta": "Let's "}),
        GenerateResponse(content="", model="m", metadata={"reasoning_delta": "think."}),
        GenerateResponse(content="Answer", model="m", finish_reason="stop"),
        # Trailing aggregate chunk (BaseProvider): complete reasoning, no delta.
        GenerateResponse(content="", model="m", metadata={"reasoning": "Let's think."}),
    ]
    sse = list(
        generate_streaming_response(_fake_llm(chunks), {}, "lmstudio", "m", _rewriter(), "req-3")
    )
    deltas = _sse_deltas(sse)

    reasoning_out = [d["reasoning_content"] for d in deltas if "reasoning_content" in d]
    assert reasoning_out == ["Let's ", "think."]

    content_out = "".join(d.get("content", "") for d in deltas)
    assert content_out == "Answer"


def test_streaming_forwards_trailing_aggregate_when_no_deltas_were_seen() -> None:
    # Inline <think> models: reasoning appears only on the trailing aggregate chunk.
    chunks = [
        GenerateResponse(content="Answer", model="m", finish_reason="stop"),
        GenerateResponse(content="", model="m", metadata={"reasoning": "hidden thoughts"}),
    ]
    sse = list(
        generate_streaming_response(_fake_llm(chunks), {}, "ollama", "m", _rewriter(), "req-4")
    )
    deltas = _sse_deltas(sse)
    reasoning_out = [d["reasoning_content"] for d in deltas if "reasoning_content" in d]
    assert reasoning_out == ["hidden thoughts"]


def test_chat_completion_request_accepts_reasoning_effort() -> None:
    req = ChatCompletionRequest(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="High",
    )
    assert req.reasoning_effort == "High"
    assert req.thinking is None


def test_model_reasoning_discovery_shapes() -> None:
    # Registry-resolved reasoning model: capability + effort enum exposed.
    block = _model_reasoning_discovery("gpt-5-mini")
    assert block == {
        "thinking_support": True,
        "reasoning_levels": ["minimal", "low", "medium", "high"],
    }

    # Registry-resolved non-reasoning model: honest false, no levels.
    non_reasoning = _model_reasoning_discovery("gpt-4o")
    assert isinstance(non_reasoning, dict)
    assert non_reasoning["thinking_support"] is False

    # Unknown model: block omitted entirely (unknown must not read as "false").
    assert _model_reasoning_discovery("totally-unknown-model-xyz") is None

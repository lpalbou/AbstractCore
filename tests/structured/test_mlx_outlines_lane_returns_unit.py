"""Regression pin (2026-09-17): the MLX Outlines lane must actually RETURN.

It never did. Two bugs sat one line apart, both swallowed by the lane's blanket
`except` and logged at DEBUG as "Outlines generation failed, falling back to prompted":

1. `response_model.model_validate(generator)` — Outlines 1.x returns the constrained
   JSON as a STRING, and `model_validate(str)` always raises.
2. `GenerateResponse(..., validated_object=...)` — that dataclass has no such field, so
   the constructor raised TypeError even once validation was fixed.

Net effect: every structured call on MLX ran a full constrained generation, threw the
correct result away, and generated a second time on the prompted lane. Callers saw a
valid object either way, so every end-to-end structured test stayed green.

This test stubs the Outlines wrapper (no weights) and RECORDS every arrival on the
prompted lane (`p.prompted_calls`), so "the lane returned" and "the lane fell back" are
both asserted facts rather than inferred from a crash.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import List, Literal, Optional

from pydantic import BaseModel

import abstractcore.providers.mlx_provider as mlx_module
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.mlx_provider import MLXProvider


class Route(BaseModel):
    mode: Literal["chat", "image"]
    assistant_message: str
    prompt: Optional[str] = None


class _Tokenizer:
    bos_token = None

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [ord(c) for c in str(text)]


class _Logger:
    def __init__(self) -> None:
        self.warnings: List[str] = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def debug(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


class _NoTools:
    supports_prompted = False


def _provider(outlines_result, monkeypatch) -> MLXProvider:
    p = MLXProvider.__new__(MLXProvider)
    p.model = "vendor/testmodel-4b"
    p.provider = "mlx"
    p.llm = object()
    p.tokenizer = _Tokenizer()
    p.logger = _Logger()
    p.tool_handler = _NoTools()
    p.architecture_config = {"message_format": "im_start_end"}
    p.model_capabilities = {}
    p.structured_output_method = "auto"
    p.max_output_tokens = 256
    p.calls = []

    def _outlines_model(prompt, schema, max_tokens, **gen_kwargs):
        p.calls.append({"prompt": prompt, "max_tokens": max_tokens, "gen_kwargs": gen_kwargs})
        return outlines_result

    p.built_samplers = []

    def _build_sampler(temperature, top_p, top_k=None):
        p.built_samplers.append((temperature, top_p, top_k))
        return ("sampler", temperature, top_p, top_k)

    p._build_mlx_sampler = _build_sampler
    p._prepare_generation_kwargs = lambda **kw: dict(kw)
    p._get_provider_max_tokens_param = lambda gen_kwargs: 256
    p.temperature = 0.7
    p.prompted_calls = []

    def _prompted_lane(prompt, *args, **kw):
        p.prompted_calls.append(prompt)
        return GenerateResponse(content="PROMPTED-LANE", model=p.model, finish_reason="stop")

    p._single_generate = _prompted_lane

    p._outlines_model = _outlines_model
    monkeypatch.setattr(mlx_module, "OUTLINES_AVAILABLE", True)
    monkeypatch.setattr(
        mlx_module, "outlines", SimpleNamespace(json_schema=lambda model: model), raising=False
    )
    return p


def test_constrained_json_string_is_returned_without_a_second_generation(monkeypatch):
    raw = '{"mode": "chat", "assistant_message": "hi", "prompt": null}'
    p = _provider(raw, monkeypatch)

    response = p._generate_internal(
        prompt="Route this request", system_prompt="You are routing.", response_model=Route
    )

    assert len(p.calls) == 1
    assert p.prompted_calls == [], "a correct constrained result must not be generated twice"
    assert response.finish_reason == "stop"
    assert Route.model_validate_json(response.content) == Route(mode="chat", assistant_message="hi")
    assert p.logger.warnings == []
    # The lane used to return nothing the ledger could account for (`usage: None`).
    assert response.usage and response.usage["input_tokens"] > 0
    assert response.gen_time is not None
    assert "You are routing." in p.calls[0]["prompt"]


def test_an_already_parsed_result_is_still_accepted(monkeypatch):
    p = _provider({"mode": "image", "assistant_message": "ok"}, monkeypatch)
    response = p._generate_internal(prompt="x", response_model=Route)
    assert json.loads(response.content)["mode"] == "image"
    assert p.prompted_calls == []


def test_a_real_outlines_failure_falls_back_LOUDLY(monkeypatch):
    """Schema-invalid output is a genuine failure: fall back, but never at debug level."""
    p = _provider('{"mode": "not-a-mode", "assistant_message": "hi"}', monkeypatch)
    response = p._generate_internal(prompt="x", response_model=Route)
    assert response.content == "PROMPTED-LANE" and len(p.prompted_calls) == 1
    assert len(p.logger.warnings) == 1
    assert "#FALLBACK" in p.logger.warnings[0] and "Outlines" in p.logger.warnings[0]


def test_native_outlines_mode_reports_the_failure_instead_of_falling_back(monkeypatch):
    p = _provider("not json at all", monkeypatch)
    p.structured_output_method = "native_outlines"
    response = p._generate_internal(prompt="x", response_model=Route)
    assert response.finish_reason == "error"
    assert "Outlines native structured output failed" in response.content
    assert p.prompted_calls == []



def test_sampling_controls_reach_the_constrained_generation(monkeypatch):
    """Once this lane started returning it would have decoded greedily whatever the
    caller asked for: no sampler was ever passed (adversarial find, 2026-09-17)."""
    p = _provider('{"mode": "chat", "assistant_message": "hi"}', monkeypatch)
    p._generate_internal(prompt="x", response_model=Route, temperature=0.9, top_p=0.5, top_k=20)
    assert p.built_samplers == [(0.9, 0.5, 20)]
    assert p.calls[0]["gen_kwargs"] == {"sampler": ("sampler", 0.9, 0.5, 20)}

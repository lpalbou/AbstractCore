"""Structured outputs at /v1/chat/completions: `response_format` json_object and
json_schema, through AbstractCore's structured-output handler, for every
provider family the server routes to.

Each family is a deterministic in-process provider (a real BaseProvider) named
like the real one, so the handler picks the same lane it picks in production:
constrained decoding (the caller's schema reaches the provider as
`response_model`) for Ollama, LM Studio, MLX/Transformers with Outlines and
providers whose capabilities say `structured_output: native` (OpenAI,
OpenAI-compatible servers such as llama.cpp or vLLM, Anthropic's forced tool);
the schema in the prompt otherwise. No network, no provider keys, no model loads."""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.structured.json_schema import (
    ResponseFormatError,
    SchemaViolation,
    parse_response_format,
    response_model_for,
    validate_instance,
)
from tests.provider_stubs import StaticProvider

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "minLength": 1},
        "temp_c": {"type": "number", "minimum": -90, "maximum": 60},
        "sky": {"type": "string", "enum": ["clear", "cloudy", "rain"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "station-id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["city", "temp_c", "sky", "tags", "station-id"],
    "additionalProperties": False,
}
GOOD = {"city": "Paris", "temp_c": 21.5, "sky": "clear", "tags": ["warm"], "station-id": None}
RF_SCHEMA = {"type": "json_schema", "json_schema": {"name": "weather", "schema": SCHEMA, "strict": True}}
CHAT = {"messages": [{"role": "user", "content": "Weather in Paris as JSON"}]}


class _Scripted(StaticProvider):
    """Answers from a script; records whether the schema came as a constraint
    (`response_model`) or inside the prompt."""

    script: list = []
    calls: list = []

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)

    def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False,
                           response_model=None, execute_tools=None, media_metadata=None, **kwargs):
        type(self).calls.append({
            "constrained": response_model is not None,
            "schema": response_model.model_json_schema() if response_model is not None else None,
            "prompt": prompt,
        })
        answer = type(self).script.pop(0) if type(self).script else GOOD
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return GenerateResponse(content=content, model=self.model, finish_reason="stop",
                                usage={"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16})


def _family(class_name: str, capability):
    attrs = {"script": [], "calls": []}
    if class_name == "HuggingFaceProvider":
        attrs["model_type"] = "gguf"  # llama.cpp in process: grammar-constrained
    cls = type(class_name, (_Scripted,), attrs)
    if capability is not None:
        original = cls.__init__

        def __init__(self, model, **kw):
            original(self, model, **kw)
            self.model_capabilities = dict(getattr(self, "model_capabilities", {}) or {}, structured_output=capability)

        cls.__init__ = __init__
    return cls


# route prefix -> (provider class name, model capability `structured_output` or None, constrained lane expected)
def _outlines() -> bool:
    try:
        import outlines  # noqa: F401
        return True
    except ImportError:
        return False


FAMILIES = {
    "ollama": ("OllamaProvider", None, True),
    "lmstudio": ("LMStudioProvider", None, True),
    "mlx": ("MLXProvider", None, _outlines()),
    "huggingface": ("HuggingFaceProvider", None, True),  # a GGUF through llama.cpp
    "openai": ("OpenAIProvider", "native", True),
    "openai-compatible": ("OpenAICompatibleProvider", "native", True),  # llama.cpp server, vLLM
    "anthropic": ("AnthropicProvider", "native", True),
    "openrouter": ("OpenRouterProvider", "prompted", False),  # a model that cannot constrain: prompted
}


@pytest.fixture
def server(monkeypatch):
    for name in ("ABSTRACTCORE_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    core = importlib.import_module("abstractcore.server.app")
    made = {}

    def create_llm(provider, model=None, **kw):
        name, by_cap, _ = FAMILIES[provider]
        cls = made.setdefault(provider, _family(name, by_cap))
        return cls(model or "m")

    monkeypatch.setattr(core, "create_llm", create_llm)
    return TestClient(core.app), made


def _post(client, provider, **extra):
    body = {"model": f"{provider}/m", **CHAT, **extra}
    return client.post("/v1/chat/completions", json=body)


@pytest.mark.parametrize("provider", sorted(FAMILIES))
def test_json_schema_answer_is_validated_and_returned_for_every_family(server, provider):
    client, made = server
    r = _post(client, provider, response_format=RF_SCHEMA)
    assert r.status_code == 200, r.text
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == GOOD
    calls = made[provider].calls
    expect_constrained = FAMILIES[provider][2]
    assert calls[0]["constrained"] is expect_constrained
    if expect_constrained:
        assert calls[0]["schema"] == SCHEMA  # the caller's schema, verbatim
    else:
        assert '"station-id"' in calls[0]["prompt"] and '"additionalProperties": false' in calls[0]["prompt"]


@pytest.mark.parametrize("provider", ["ollama", "openrouter"])
def test_an_answer_off_schema_is_retried_with_the_violation(server, provider):
    client, made = server
    r0 = _post(client, provider, response_format=RF_SCHEMA)  # creates the family class
    assert r0.status_code == 200
    cls = made[provider]
    cls.calls.clear()
    cls.script[:] = [{**GOOD, "sky": "snow"}, {**GOOD, "sky": "snow"}, GOOD, GOOD]
    r = _post(client, provider, response_format=RF_SCHEMA)
    assert r.status_code == 200, r.text
    assert json.loads(r.json()["choices"][0]["message"]["content"])["sky"] == "clear"
    assert len(cls.calls) >= 2
    assert any("must be one of" in c["prompt"] for c in cls.calls[1:])


@pytest.mark.parametrize("provider", ["lmstudio", "openrouter", "anthropic"])
def test_an_answer_that_never_matches_is_the_standard_error(server, provider):
    client, made = server
    _post(client, provider, response_format=RF_SCHEMA)
    made[provider].script[:] = [{"city": "Paris"}] * 20
    r = _post(client, provider, response_format=RF_SCHEMA)
    assert r.status_code == 500, r.text
    err = r.json()["error"]
    assert err["code"] == "structured_output_invalid" and err["type"] == "server_error"
    assert err["param"] == "response_format" and "weather" in err["message"]


def test_json_object_returns_an_object_and_refuses_non_objects(server):
    client, made = server
    r = _post(client, "ollama", response_format={"type": "json_object"})
    assert r.status_code == 200 and isinstance(json.loads(r.json()["choices"][0]["message"]["content"]), dict)
    made["ollama"].script[:] = ["[1, 2]"] * 20
    bad = _post(client, "ollama", response_format={"type": "json_object"})
    assert bad.status_code == 500 and bad.json()["error"]["code"] == "structured_output_invalid"


def test_stream_carries_the_validated_json_in_one_chunk(server):
    client, _ = server
    r = _post(client, "lmstudio", response_format=RF_SCHEMA, stream=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [ln[6:] for ln in r.text.split("\n") if ln.startswith("data: ")]
    assert events[-1] == "[DONE]"
    first, last = json.loads(events[0]), json.loads(events[1])
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert json.loads(first["choices"][0]["delta"]["content"]) == GOOD
    assert last["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("rf,param", [
    ({"type": "xml"}, "response_format.type"),
    ({"type": "json_schema"}, "response_format.json_schema"),
    ({"type": "json_schema", "json_schema": {"name": "bad name!", "schema": SCHEMA}}, "response_format.json_schema.name"),
    ({"type": "json_schema", "json_schema": {"name": "w", "schema": {"type": "array"}}}, "response_format.json_schema.schema"),
    ({"type": "json_schema", "json_schema": {"name": "w", "schema": {"type": "object", "properties": {"a": {"type": "strng"}}}}}, "response_format"),
    ({"type": "json_schema", "json_schema": {"name": "w", "schema": {"type": "object", "properties": {"a": {"$ref": "#/$defs/nope"}}}}}, "response_format"),
])
def test_an_unusable_response_format_is_a_400_before_any_model(server, rf, param):
    client, made = server
    r = _post(client, "ollama", response_format=rf)
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["param"] == param and err["code"] == "invalid_response_format"
    assert "ollama" not in made  # no provider was created


def test_response_format_with_tools_is_refused(server):
    client, _ = server
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    r = _post(client, "ollama", response_format=RF_SCHEMA, tools=tools)
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter"


def test_text_response_format_is_plain_chat(server):
    client, made = server
    made_cls = None
    r = _post(client, "ollama", response_format={"type": "text"})
    assert r.status_code == 200
    made_cls = made["ollama"]
    assert made_cls.calls[-1]["constrained"] is False


# ---- the validator -------------------------------------------------------

@pytest.mark.parametrize("value,where", [
    ({**GOOD, "extra": 1}, "extra"),
    ({k: v for k, v in GOOD.items() if k != "sky"}, "sky"),
    ({**GOOD, "temp_c": 99}, "temp_c"),
    ({**GOOD, "temp_c": "21"}, "temp_c"),
    ({**GOOD, "tags": ["a", "b", "c", "d"]}, "tags"),
    ({**GOOD, "tags": [1]}, "tags[0]"),
    ({**GOOD, "city": ""}, "city"),
    ({**GOOD, "station-id": 3}, "station-id"),
])
def test_validator_names_the_violation(value, where):
    with pytest.raises(SchemaViolation) as exc:
        validate_instance(value, SCHEMA)
    assert exc.value.path == where


def test_validator_follows_refs_and_accepts_the_good_value():
    schema = {"type": "object", "$defs": {"pt": {"type": "object", "properties": {"x": {"type": "integer"}},
                                                  "required": ["x"]}},
              "properties": {"p": {"$ref": "#/$defs/pt"}}, "required": ["p"]}
    validate_instance({"p": {"x": 2}}, schema)
    with pytest.raises(SchemaViolation):
        validate_instance({"p": {"x": 2.5}}, schema)
    validate_instance(GOOD, SCHEMA)


def test_response_model_schema_is_the_callers_and_text_is_none():
    model = response_model_for(RF_SCHEMA)
    assert model.__name__ == "weather" and model.model_json_schema() == SCHEMA
    assert response_model_for({"type": "text"}) is None and response_model_for(None) is None
    with pytest.raises(ResponseFormatError):
        parse_response_format("json")

"""Unified `thinking=` translation, pinned at the REQUEST BODY, per provider x model family.

The contract (docs/reasoning-control.md): one vocabulary (`False|True|"none"|"off"|
"minimal"|"low"|"medium"|"high"|"xhigh"`) translated per model (registry
`thinking_control` / `reasoning_levels`) and per provider (each provider's transport), and
metadata (`thinking_effective`, `thinking_handled_*`) never claims a control that was not
sent through a transport the backend honours.

Live evidence for the LM Studio rows is recorded in
`abstractcore/providers/lmstudio_provider.py::_apply_provider_thinking_kwargs`
(2026-10-25, qwen/qwen3.6-35b-a3b and qwen/qwen3.8-27b, rendered prompts read back with
`lms log stream`). The vLLM / OpenRouter / Portkey rows are UNIT-ONLY (unverified live).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.vllm_provider import VLLMProvider

EFFORT_KWARG = LMStudioProvider.LMSTUDIO_REASONING_EFFORT_KWARG

QWEN36 = "qwen/qwen3.6-35b-a3b"
QWEN38 = "qwen/qwen3.8-27b"
HY = "tencent/hy-memory-30b-a3b"
GEMMA4 = "google/gemma-4-26b-a4b"
GPT_OSS = "openai/gpt-oss-20b"
NEMOTRON = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning-bf16"
DEEPSEEK_V4 = "deepseek/deepseek-v4-flash"
KIMI = "moonshotai/kimi-k2.6"
GLM46V = "zai-org/glm-4.6v-flash"
QWEN3 = "qwen/qwen3-4b"
LAGUNA = "laguna-s-2.1"
SEED = "seed-oss-36b"

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
TOOL_HISTORY = [
    {"role": "user", "content": "Find the capital of France with the tool."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\": \"France\"}"}}],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Paris"},
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lmstudio(monkeypatch, model: str) -> LMStudioProvider:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LMStudioProvider(model=model, base_url="http://localhost:1234/v1")


def _vllm(monkeypatch, model: str) -> VLLMProvider:
    monkeypatch.setattr(VLLMProvider, "_validate_model", lambda self: None, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VLLMProvider(model=model, base_url="http://127.0.0.1:8000/v1")


def _send(monkeypatch, provider, thinking, **gen_kwargs) -> tuple[Dict[str, Any], GenerateResponse, list]:
    """Run generate() through the real payload builder; capture the wire payload."""
    captured: Dict[str, Any] = {}

    def _capture(payload, **_kw):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    def _no_native(**_kw):
        raise AssertionError("thinking= must not route to LM Studio's native REST endpoint")

    monkeypatch.setattr(provider, "_single_generate", _capture)
    if isinstance(provider, LMStudioProvider):
        monkeypatch.setattr(provider, "_native_rest_chat_generate", _no_native)
    prompt = gen_kwargs.pop("prompt", "What is 17*23?")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resp = provider.generate(prompt, thinking=thinking, temperature=0, **gen_kwargs)
    return captured["payload"], resp, [str(w.message) for w in caught]


def _assistant_prefill_messages(payload: Dict[str, Any]) -> list:
    return [
        m for m in payload.get("messages") or []
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) and "</think>" in m["content"]
    ]


# ---------------------------------------------------------------------------
# LM Studio — the 2026-10-25 defect and its family matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [True, "on", "high", "minimal", "xhigh"])
def test_lmstudio_qwen36_on_sends_top_level_reasoning_effort(monkeypatch, thinking) -> None:
    # THE DEFECT: thinking=True/"high" sent only chat_template_kwargs (ignored by LM Studio)
    # while metadata said thinking_effective="on"; the server default (off) stayed in force.
    provider = _lmstudio(monkeypatch, QWEN36)
    payload, resp, _ = _send(monkeypatch, provider, thinking)

    assert payload["reasoning_effort"] == "medium"  # LM Studio: any non-"none" value = on
    assert EFFORT_KWARG not in payload
    assert not _assistant_prefill_messages(payload)
    meta = resp.metadata
    assert meta["thinking_effective"] == "on"  # Qwen3.6 has no effort levels: never "high"
    assert meta["thinking_handled_enable_disable"] is True
    assert meta["thinking_handled_level"] is False


@pytest.mark.parametrize("thinking", [False, "off", "none"])
def test_lmstudio_qwen36_off_sends_none_without_a_duplicate_prefill(monkeypatch, thinking) -> None:
    # Live: reasoning_effort "none" renders the template's own `<think>\n\n</think>\n\n`;
    # the extra assistant prefill rendered the empty block TWICE.
    provider = _lmstudio(monkeypatch, QWEN36)
    payload, resp, _ = _send(monkeypatch, provider, thinking)

    assert payload["reasoning_effort"] == "none"
    assert not _assistant_prefill_messages(payload)
    assert payload["messages"][-1] == {"role": "user", "content": "What is 17*23?"}
    assert resp.metadata["thinking_effective"] == "off"


def test_lmstudio_qwen36_tool_history_shape_keeps_control_and_history(monkeypatch) -> None:
    provider = _lmstudio(monkeypatch, QWEN36)
    for thinking, effort in (("on", "medium"), ("off", "none")):
        payload, resp, _ = _send(
            monkeypatch, provider, thinking, prompt="", messages=[dict(m) for m in TOOL_HISTORY], tools=TOOLS
        )
        assert payload["reasoning_effort"] == effort
        assert payload["tools"]
        roles = [m.get("role") for m in payload["messages"]]
        assert roles[-3:] == ["user", "assistant", "tool"]  # no trailing prefill turn
        assert payload["messages"][-2]["tool_calls"][0]["id"] == "call_1"
        assert resp.metadata["thinking_effective"] == ("on" if thinking == "on" else "off")


def test_lmstudio_reasoning_effort_400_latch_falls_back_honestly(monkeypatch) -> None:
    provider = _lmstudio(monkeypatch, QWEN36)
    provider._reasoning_effort_unsupported = True  # this server rejected the field earlier

    # Off: the asset-declared assistant prefill is the remaining real switch.
    payload, resp, _ = _send(monkeypatch, provider, "off")
    assert "reasoning_effort" not in payload
    assert len(_assistant_prefill_messages(payload)) == 1
    assert resp.metadata["thinking_effective"] == "off"

    # On: no transport left — never claimed; the base ladder warns.
    payload, resp, messages = _send(monkeypatch, provider, "on")
    assert "reasoning_effort" not in payload
    assert "thinking_effective" not in resp.metadata
    assert resp.metadata["thinking_handled_enable_disable"] is False
    assert any("no control was applied" in m for m in messages)


@pytest.mark.parametrize(
    "thinking,effort,effective,level_handled",
    [
        ("low", "low", "low", True),
        ("medium", "medium", "medium", True),
        ("xhigh", "xhigh", "xhigh", True),
        ("high", "medium", "medium", True),  # undeclared -> nearest declared (tie -> lower), warned
        ("minimal", "low", "low", True),
        ("on", "medium", "on", False),  # bare on: the template default effort
        ("off", "none", "off", False),
    ],
)
def test_lmstudio_qwen38_effort_levels(monkeypatch, thinking, effort, effective, level_handled) -> None:
    provider = _lmstudio(monkeypatch, QWEN38)
    payload, resp, messages = _send(monkeypatch, provider, thinking)
    assert payload["reasoning_effort"] == effort
    assert resp.metadata["thinking_effective"] == effective
    assert resp.metadata["thinking_handled_level"] is level_handled
    if thinking in {"high", "minimal"}:
        assert any("not supported for model" in m for m in messages)


@pytest.mark.parametrize(
    "thinking,effort,effective",
    [("off", "none", "off"), ("on", "high", "on"), ("low", "low", "low"), ("high", "high", "high")],
)
def test_lmstudio_hy_v3_effort_switch(monkeypatch, thinking, effort, effective) -> None:
    provider = _lmstudio(monkeypatch, HY)
    payload, resp, _ = _send(monkeypatch, provider, thinking)
    assert payload["reasoning_effort"] == effort
    assert resp.metadata["thinking_effective"] == effective


@pytest.mark.parametrize("model", [GEMMA4, NEMOTRON, LAGUNA, QWEN3])
def test_lmstudio_boolean_switch_families_on_off(monkeypatch, model) -> None:
    provider = _lmstudio(monkeypatch, model)
    payload, resp, _ = _send(monkeypatch, provider, "on")
    assert payload["reasoning_effort"] == "medium"
    assert resp.metadata["thinking_effective"] == "on"
    payload, resp, _ = _send(monkeypatch, provider, "off")
    assert payload["reasoning_effort"] == "none"
    assert not _assistant_prefill_messages(payload)
    assert "/no_think" not in str(payload["messages"])
    assert resp.metadata["thinking_effective"] == "off"


def test_lmstudio_nemotron_low_is_not_claimed_as_a_level(monkeypatch) -> None:
    # The `low_effort` template kwarg rides chat_template_kwargs only, which LM Studio
    # ignores: the level must not be claimed there.
    provider = _lmstudio(monkeypatch, NEMOTRON)
    payload, resp, _ = _send(monkeypatch, provider, "low")
    assert payload["reasoning_effort"] == "medium"
    assert resp.metadata["thinking_effective"] == "on"
    assert resp.metadata["thinking_handled_level"] is False


def test_lmstudio_deepseek_v4_native_levels(monkeypatch) -> None:
    provider = _lmstudio(monkeypatch, DEEPSEEK_V4)
    for thinking, effort, effective in (("on", "high", "on"), ("off", "none", "off"), ("xhigh", "xhigh", "xhigh")):
        payload, resp, _ = _send(monkeypatch, provider, thinking)
        assert payload["reasoning_effort"] == effort
        assert resp.metadata["thinking_effective"] == effective


def test_lmstudio_gpt_oss_keeps_harmony_system_line(monkeypatch) -> None:
    provider = _lmstudio(monkeypatch, GPT_OSS)
    payload, resp, _ = _send(monkeypatch, provider, "high")
    # Both artifacts express the same level: the harmony system line (BaseProvider) and
    # the top-level field LM Studio maps for GPT-OSS.
    assert payload["reasoning_effort"] == "high"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"].startswith("Reasoning: high")
    assert resp.metadata["thinking_effective"] == "high"


def test_lmstudio_kimi_thinking_param_is_declined_not_claimed(monkeypatch) -> None:
    # kimi-k2.6 declares only a `thinking` request param, with no transport on LM Studio's
    # OpenAI endpoint: no claim (previously the native REST field was claimed even for
    # requests that could not ride it).
    provider = _lmstudio(monkeypatch, KIMI)
    payload, resp, messages = _send(monkeypatch, provider, "off")
    assert "reasoning_effort" not in payload and "reasoning" not in payload
    assert "thinking_effective" not in resp.metadata
    assert any("no control was applied" in m for m in messages)


def test_lmstudio_glm46v_off_uses_the_prompt_token(monkeypatch) -> None:
    provider = _lmstudio(monkeypatch, GLM46V)
    payload, resp, _ = _send(monkeypatch, provider, "off")
    assert "reasoning_effort" not in payload
    assert payload["messages"][-1]["content"].endswith("/nothink")
    assert resp.metadata["thinking_effective"] == "off"


def test_lmstudio_seed_oss_budget_is_not_claimed_through_ignored_template_kwargs(monkeypatch) -> None:
    # Seed-OSS: the template's budget variable rides chat_template_kwargs (ignored by LM
    # Studio); its declared levels go through reasoning_effort instead (unverified live).
    provider = _lmstudio(monkeypatch, SEED)
    payload, resp, _ = _send(monkeypatch, provider, "high")
    assert payload["reasoning_effort"] == "high"
    assert resp.metadata["thinking_effective"] == "high"


def test_lmstudio_async_path_carries_the_control(monkeypatch) -> None:
    # The native REST lane never existed on the async path: the control was silently lost.
    import asyncio

    provider = _lmstudio(monkeypatch, GEMMA4)
    captured: Dict[str, Any] = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

        def raise_for_status(self):
            return None

    class _Client:
        async def post(self, url, json=None, **kwargs):
            captured["payload"] = json
            return _Resp()

    monkeypatch.setattr(type(provider), "async_client", property(lambda self: _Client()), raising=False)
    resp = asyncio.run(provider.agenerate("hi", thinking="on", temperature=0))
    assert captured["payload"]["reasoning_effort"] == "medium"
    assert resp.metadata["thinking_effective"] == "on"


def test_lmstudio_caller_reasoning_effort_wins(monkeypatch) -> None:
    provider = _lmstudio(monkeypatch, QWEN36)
    payload, _resp, _ = _send(monkeypatch, provider, "on", reasoning_effort="high")
    assert payload["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# vLLM — controls must reach the TOP LEVEL (a literal `extra_body` key is ignored)
# ---------------------------------------------------------------------------


def test_vllm_wire_has_no_literal_extra_body(monkeypatch) -> None:
    provider = _vllm(monkeypatch, QWEN36)
    payload, resp, _ = _send(monkeypatch, provider, "off", guided_regex="[0-9]+")
    assert "extra_body" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["guided_regex"] == "[0-9]+"
    assert resp.metadata["thinking_effective"] == "off"


def test_vllm_caller_chat_template_kwargs_win_per_key(monkeypatch) -> None:
    provider = _vllm(monkeypatch, QWEN36)
    payload, _resp, _ = _send(monkeypatch, provider, "on", chat_template_kwargs={"enable_thinking": False, "x": 1})
    assert payload["chat_template_kwargs"] == {"enable_thinking": False, "x": 1}


@pytest.mark.parametrize(
    "model,thinking,body,effective,level_handled",
    [
        # Boolean switch, no effort levels: a level is a best-effort token cap, not a claim.
        (QWEN36, "high", {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}, "on", False),
        (QWEN38, "low", {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}, "thinking_token_budget": 512}, "low", True),
        (GEMMA4, "on", {"chat_template_kwargs": {"enable_thinking": True}}, "on", False),
        (NEMOTRON, "low", {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}, "thinking_token_budget": 512}, "low", True),
        (HY, "off", {"chat_template_kwargs": {"reasoning_effort": "no_think"}}, "off", False),
        (DEEPSEEK_V4, "on", {"reasoning_effort": "high"}, "high", True),
        (DEEPSEEK_V4, "off", {"reasoning_effort": "none"}, "off", True),
        (GPT_OSS, "high", {"reasoning_effort": "high"}, "high", True),
        (SEED, "high", {"chat_template_kwargs": {"thinking_budget": 4096}}, "high", True),
    ],
)
def test_vllm_family_matrix(monkeypatch, model, thinking, body, effective, level_handled) -> None:
    provider = _vllm(monkeypatch, model)
    payload, resp, _ = _send(monkeypatch, provider, thinking)
    for key, value in body.items():
        assert payload.get(key) == value, (key, payload.get(key))
    for key in {"chat_template_kwargs", "reasoning_effort", "thinking_token_budget"} - set(body):
        assert key not in payload, key
    assert resp.metadata["thinking_effective"] == effective
    assert resp.metadata["thinking_handled_level"] is level_handled


def test_vllm_surface_less_model_is_declined_not_claimed(monkeypatch) -> None:
    # Previously: a guessed `enable_thinking` for every model, claimed as applied.
    provider = _vllm(monkeypatch, KIMI)
    payload, resp, messages = _send(monkeypatch, provider, "on")
    assert "chat_template_kwargs" not in payload and "reasoning_effort" not in payload
    assert "thinking_effective" not in resp.metadata
    assert any("no control was applied" in m for m in messages)


# ---------------------------------------------------------------------------
# OpenRouter / Portkey
# ---------------------------------------------------------------------------


def _openrouter(monkeypatch, model: str):
    from abstractcore.providers.openrouter_provider import OpenRouterProvider

    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OpenRouterProvider(model=model, api_key="sk-test")


@pytest.mark.parametrize(
    "model,thinking,reasoning,effective",
    [
        ("openai/gpt-5.4", "high", {"effort": "high"}, "high"),
        ("openai/gpt-5.4", "off", {"effort": "none"}, "off"),
        ("openai/gpt-5.4", "on", {"enabled": True}, "on"),
        ("openai/gpt-5", "off", {"effort": "minimal"}, "minimal"),  # mandatory reasoning: floor, warned
        ("deepseek/deepseek-v4-flash", "medium", {"effort": "high"}, "high"),  # nearest declared (none/high/xhigh)
        ("qwen/qwen3.6-35b-a3b", "off", {"effort": "none"}, "off"),
    ],
)
def test_openrouter_reasoning_object(monkeypatch, model, thinking, reasoning, effective) -> None:
    provider = _openrouter(monkeypatch, model)
    payload, resp, _ = _send(monkeypatch, provider, thinking)
    assert payload["reasoning"] == reasoning
    assert "_acore_openrouter_reasoning" not in payload and "_acore_thinking_effective_effort" not in payload
    assert resp.metadata["thinking_effective"] == effective


def test_openrouter_explicit_reasoning_object_wins(monkeypatch) -> None:
    provider = _openrouter(monkeypatch, "openai/gpt-5.4")
    payload, _resp, _ = _send(monkeypatch, provider, "high", reasoning={"max_tokens": 2000})
    assert payload["reasoning"] == {"max_tokens": 2000}


def test_portkey_declines_without_claiming(monkeypatch) -> None:
    # The backend behind a Portkey config is unknown: no thinking transport is sent and
    # nothing is claimed (the base ladder warns).
    from abstractcore.providers.portkey_provider import PortkeyProvider

    for name in ("PORTKEY_API_KEY", "PORTKEY_CONFIG", "PORTKEY_VIRTUAL_KEY", "PORTKEY_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(PortkeyProvider, "_validate_model", lambda self: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = PortkeyProvider(model="gpt-5.4", api_key="pk-test", config_id="pcfg_1")
    payload, resp, messages = _send(monkeypatch, provider, "high")
    assert "reasoning_effort" not in payload and "reasoning" not in payload
    assert "thinking_effective" not in resp.metadata
    assert any("does not implement a thinking control mapping" in m for m in messages)

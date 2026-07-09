"""Mid-stream `role:"system"` message handling for native OpenAI/Anthropic providers.

These tests pin the fix for a silent-drop defect: both native providers used to delete
every `role:"system"` entry found inside `messages` (leading and mid-stream alike).
That silently discarded system prompts arriving via `messages` — notably from
server-mediated OpenAI-format clients (which send the system prompt as messages[0])
and from agent runtimes that tail-place system hints deliberately.

Contract now pinned:
- OpenAI: system messages pass through VERBATIM at their original position (the Chat
  Completions API accepts them anywhere; reasoning models auto-treat system as developer).
- Anthropic: a LEADING contiguous run of system messages merges into the top-level
  `system` parameter (its native surface); NON-LEADING system messages are converted in
  place to `<system_instruction>`-wrapped user messages (position preserved), deferred
  past contiguous tool-result runs so tool_use/tool_result adjacency is never broken.
  Conversions are counted in `metadata["system_role_user_wrapped"]`.
- BasicSession delivers non-system-prompt system messages (e.g. compaction summaries)
  in-stream instead of filtering them out.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from abstractcore.core.session import BasicSession
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.anthropic_provider import AnthropicProvider
from abstractcore.providers.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _fake_openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="stubbed-openai",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _install_fake_openai(monkeypatch) -> None:
    import abstractcore.providers.openai_provider as openai_provider_module

    class _FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_k: _fake_openai_response()))
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClient, AsyncOpenAI=_FakeOpenAIClient)
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True, raising=False)
    monkeypatch.setattr(openai_provider_module, "openai", fake_openai, raising=False)


def _make_openai_provider(monkeypatch) -> OpenAIProvider:
    _install_fake_openai(monkeypatch)
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None)
    return OpenAIProvider(model="gpt-4o-mini", api_key="test")


def test_openai_tail_system_message_passes_through_in_position(monkeypatch):
    provider = _make_openai_provider(monkeypatch)

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    messages = [
        {"role": "user", "content": "Read the file."},
        {"role": "assistant", "content": "Done."},
        {"role": "system", "content": "[Attachment index] file1.txt is already attached."},
    ]
    provider.generate(prompt="", messages=messages, max_output_tokens=16)

    api_messages = captured["call_params"]["messages"]
    assert api_messages[-1] == {"role": "system", "content": "[Attachment index] file1.txt is already attached."}
    assert [m["role"] for m in api_messages] == ["user", "assistant", "system"]


def test_openai_server_shape_leading_system_in_messages_is_delivered(monkeypatch):
    # Server-mediated clients send the system prompt as messages[0] with no system_prompt kwarg.
    provider = _make_openai_provider(monkeypatch)

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    messages = [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Hello."},
    ]
    provider.generate(prompt="", messages=messages, max_output_tokens=16)

    api_messages = captured["call_params"]["messages"]
    assert api_messages[0] == {"role": "system", "content": "You are a pirate."}


def test_openai_system_prompt_and_in_messages_system_coexist(monkeypatch):
    provider = _make_openai_provider(monkeypatch)

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    provider.generate(
        prompt="",
        messages=[
            {"role": "user", "content": "q"},
            {"role": "system", "content": "tail hint"},
        ],
        system_prompt="SP",
        max_output_tokens=16,
    )

    api_messages = captured["call_params"]["messages"]
    assert api_messages[0] == {"role": "system", "content": "SP"}
    assert api_messages[-1] == {"role": "system", "content": "tail hint"}


def test_openai_async_tail_system_message_passes_through(monkeypatch):
    provider = _make_openai_provider(monkeypatch)

    captured = {}

    async def fake_acreate(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    provider._async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_acreate))
    )

    messages = [
        {"role": "user", "content": "q"},
        {"role": "system", "content": "tail hint"},
    ]
    asyncio.run(provider.agenerate(prompt="", messages=messages, max_output_tokens=16))

    api_messages = captured["call_params"]["messages"]
    assert api_messages[-1] == {"role": "system", "content": "tail hint"}


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _fake_anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def test_anthropic_leading_system_messages_merge_into_system_param(monkeypatch):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response()

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    messages = [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Hello."},
    ]
    provider.generate(prompt="", messages=messages, system_prompt="SP", max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["system"] == "SP\n\nYou are a pirate."
    roles = [m["role"] for m in call_params["messages"]]
    assert "system" not in roles
    assert roles[0] == "user"


def test_anthropic_tail_system_message_wraps_as_user_in_position(monkeypatch):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response()

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    messages = [
        {"role": "user", "content": "Read the file."},
        {"role": "assistant", "content": "Done."},
        {"role": "system", "content": "[Attachment index] file1.txt is already attached."},
    ]
    resp = provider.generate(prompt="", messages=messages, system_prompt="SP", max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["system"] == "SP"  # tail hint must NOT be hoisted (position matters)
    api_messages = call_params["messages"]
    assert api_messages[-1]["role"] == "user"
    assert api_messages[-1]["content"] == (
        "<system_instruction>\n[Attachment index] file1.txt is already attached.\n</system_instruction>"
    )
    assert all(m.get("role") != "system" for m in api_messages)
    assert resp.metadata.get("system_role_user_wrapped") == 1


def test_anthropic_wrapped_system_defers_past_tool_result_run(monkeypatch):
    # A converted system message must never land between an assistant tool_use turn and
    # its tool_result (Anthropic placement rule); it is emitted after the tool run.
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response()

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    messages = [
        {"role": "user", "content": "List files."},
        {"role": "assistant", "content": "Using the tool."},
        {"role": "system", "content": "hint arriving mid tool cycle"},
        {"role": "tool", "content": "file1.txt", "metadata": {"call_id": "toolu_1"}},
        {"role": "user", "content": "Thanks."},
    ]
    provider.generate(prompt="", messages=messages, max_output_tokens=16)

    api_messages = captured["call_params"]["messages"]
    kinds = []
    for m in api_messages:
        content = m.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
            kinds.append("tool_result")
        elif isinstance(content, str) and content.startswith("<system_instruction>"):
            kinds.append("wrapped_system")
        else:
            kinds.append(m.get("role"))

    # tool_result immediately follows the assistant turn; the wrapped hint comes after it.
    assert kinds == ["user", "assistant", "tool_result", "wrapped_system", "user"]


def test_anthropic_async_tail_system_message_wraps(monkeypatch):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    async def fake_acreate(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response()

    provider._async_client = SimpleNamespace(messages=SimpleNamespace(create=fake_acreate))

    messages = [
        {"role": "user", "content": "q"},
        {"role": "system", "content": "tail hint"},
    ]
    resp = asyncio.run(provider.agenerate(prompt="", messages=messages, max_output_tokens=16))

    api_messages = captured["call_params"]["messages"]
    assert api_messages[-1]["role"] == "user"
    assert "tail hint" in api_messages[-1]["content"]
    assert resp.metadata.get("system_role_user_wrapped") == 1


def test_anthropic_history_builder_is_keyerror_safe():
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    # Messages missing role/content keys must not raise (previously msg["role"] KeyError'd).
    api_messages, leading, wrapped = provider._build_anthropic_history(
        [
            {},
            {"content": "no role"},
            {"role": "user"},
            {"role": "assistant"},
        ]
    )
    assert leading == []
    assert wrapped == 0
    assert api_messages == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]


# ---------------------------------------------------------------------------
# BasicSession: compaction summary delivery
# ---------------------------------------------------------------------------


class _CaptureSessionProvider:
    def __init__(self):
        self.captured = {}

    def generate(self, prompt=None, messages=None, system_prompt=None, media=None, **kwargs):
        self.captured = {
            "prompt": prompt,
            "messages": messages,
            "system_prompt": system_prompt,
        }
        return GenerateResponse(content="ok", model="capture", finish_reason="stop")


def test_session_delivers_compaction_summary_but_not_duplicate_system_prompt():
    provider = _CaptureSessionProvider()
    session = BasicSession(provider=provider, system_prompt="SP")

    # Simulate a compacted session: the summary is stored as a system message.
    session.add_message("system", "[CONVERSATION HISTORY]: earlier we discussed X.")
    session.add_message("user", "q1")
    session.add_message("assistant", "a1")

    session.generate("q2")

    sent = provider.captured["messages"]
    roles_contents = [(m["role"], m["content"]) for m in sent]

    # The session system prompt is NOT duplicated into messages (delivered via system_prompt=).
    assert ("system", "SP") not in roles_contents
    assert provider.captured["system_prompt"] == "SP"

    # The compaction summary IS delivered in-stream (previously filtered out entirely).
    assert ("system", "[CONVERSATION HISTORY]: earlier we discussed X.") in roles_contents
    assert roles_contents[0] == ("system", "[CONVERSATION HISTORY]: earlier we discussed X.")


def test_session_compact_summary_reaches_provider_end_to_end(monkeypatch):
    # Full compact() flow with a stubbed summarizer: the new session's provider call
    # must include the [CONVERSATION HISTORY] summary message.
    provider = _CaptureSessionProvider()
    session = BasicSession(provider=provider, system_prompt="SP")
    for i in range(6):
        session.add_message("user", f"question {i}")
        session.add_message("assistant", f"answer {i}")

    import abstractcore.processing as processing_module

    class _FakeSummarizer:
        def __init__(self, *args, **kwargs):
            pass

        def summarize_chat_history(self, messages=None, preserve_recent=4, focus=None):
            return SimpleNamespace(summary="users asked six questions and got answers")

    # compact() does `from ..processing import BasicSummarizer` at call time.
    monkeypatch.setattr(processing_module, "BasicSummarizer", _FakeSummarizer)

    compacted = session.compact(preserve_recent=2)
    compacted.generate("follow-up")

    sent = provider.captured["messages"]
    summary_entries = [
        m for m in sent if m["role"] == "system" and m["content"].startswith("[CONVERSATION HISTORY]")
    ]
    assert len(summary_entries) == 1
    assert "six questions" in summary_entries[0]["content"]

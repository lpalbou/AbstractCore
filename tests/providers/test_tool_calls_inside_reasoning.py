"""Tool calls a local model writes inside its thinking block must still run.

Live failure (2026-09-26, `Jundot/Qwen3.8-27B-oQ4e-mtp`, MLX in-process,
non-streamed): the Qwen3.8 template ends the prompt with `<think>\\n`, and the
model wrote its three `<tool_call><function=fetch_url>...` calls before (or
without) `</think>`. MLX splits inline thinking BEFORE BaseProvider parses tool
calls, so the calls landed in `metadata["reasoning"]`, content was empty, zero
calls were parsed and no warning was raised; the agent then used the reasoning
as its answer and the run "completed" with the raw markup.

The fake provider below reproduces MLX's order (split thinking, then return).
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.response_postprocessing import normalize_assistant_text
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.tools.core import ToolCall
from abstractcore.tools.handler import UniversalToolHandler
from abstractcore.tools.registry import ToolRegistry

MODEL = "Jundot/Qwen3.8-27B-oQ4e-mtp"
OPERATOR_TEXT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "qwen3_coder_three_fetch_url_calls.txt"
).read_text()
URLS = [
    "https://headsupai.io/ai-news-and-updates/today",
    "https://aiweekly.co/ai-news-today",
    "https://www.businesstoday.in/technology/news/story/"
    "blackrock-sees-ai-agents-creating-a-new-economy-for-stablecoins-blockchain-and-compute-557945-2026-09-26",
]


class _MLXOrderProvider(BaseProvider):
    """Returns `text` the way MLX does: inline thinking already split off."""

    def __init__(self, text: str, *, opened_by_prompt: bool, stream_cut: int = 7):
        super().__init__(MODEL)
        self.provider = "mlx"
        self.tool_handler = UniversalToolHandler(MODEL)
        self._text = text
        self._opened = opened_by_prompt
        self._cut = stream_cut

    def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
        if stream:
            text = self._text

            def _chunks():
                for i in range(0, len(text), self._cut):
                    yield GenerateResponse(content=text[i : i + self._cut], model=self.model)
                yield GenerateResponse(content="", model=self.model, finish_reason="stop")

            return _chunks()
        arch = get_architecture_format(detect_architecture(MODEL))
        content, reasoning = normalize_assistant_text(
            self._text.strip(),
            architecture_format=arch,
            model_capabilities=get_model_capabilities(MODEL),
            thinking_opened_by_prompt=self._opened,
        )
        return GenerateResponse(
            content=content,
            model=self.model,
            finish_reason="stop",
            metadata={"reasoning": reasoning} if reasoning else None,
        )

    def get_capabilities(self) -> List[str]:
        return ["chat", "tools"]

    def unload_model(self, model_name: str) -> None:
        return None

    def list_available_models(self, **kwargs) -> List[str]:
        return [MODEL]


def fetch_url(url: str, include_full_content: bool = True) -> str:
    """Fetch a web page."""
    return f"fetched {url} full={include_full_content!r}"


def web_search(query: str) -> str:
    """Search the web."""
    return query


# (label, model output after the prompt's `<think>\n`, prompt opened thinking)
SHAPES = [
    ("answer_after_think", "I need news.\n</think>\n\n" + OPERATOR_TEXT, True),
    ("calls_then_close_think", "I need news.\n\n" + OPERATOR_TEXT + "\n</think>\n\n", True),
    ("calls_only_then_close", OPERATOR_TEXT + "\n</think>\n", True),
    ("calls_never_close_think", OPERATOR_TEXT, True),
    ("prose_and_calls_never_close", "I need news.\n\n" + OPERATOR_TEXT, True),
    ("thinking_off_bare", OPERATOR_TEXT, False),
    ("explicit_think_block_with_calls", "<think>\nI need news.\n" + OPERATOR_TEXT + "\n</think>\n", False),
]


def _host_tool_loop(calls: List[Dict[str, Any]], tools) -> List[Any]:
    """What a host (runtime/agent) does: dispatch every structured call by name."""
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return [registry.execute_tool(ToolCall(name=c["name"], arguments=c["arguments"], call_id=None)) for c in calls]


@pytest.mark.parametrize("label,text,opened", SHAPES, ids=[s[0] for s in SHAPES])
def test_non_streamed_calls_are_recovered_and_executed(label, text, opened):
    resp = _MLXOrderProvider(text, opened_by_prompt=opened).generate(prompt="news", tools=[fetch_url])
    calls = resp.tool_calls or []
    assert [c["name"] for c in calls] == ["fetch_url"] * 3, label
    reasoning = (resp.metadata or {}).get("reasoning") or ""
    assert "<tool_call>" not in reasoning and "<tool_call>" not in (resp.content or "")

    results = _host_tool_loop(calls, [fetch_url])
    assert [r.success for r in results] == [True] * 3
    # `False` in the markup reaches the tool as the boolean False (schema coercion).
    assert [r.output for r in results] == [f"fetched {u} full=False" for u in URLS]


@pytest.mark.parametrize("label,text,opened", SHAPES, ids=[s[0] for s in SHAPES])
def test_streamed_calls_are_executed(label, text, opened):
    stream = _MLXOrderProvider(text, opened_by_prompt=opened).generate(prompt="news", tools=[fetch_url], stream=True)
    calls, content = [], ""
    for chunk in stream:
        calls.extend(chunk.tool_calls or [])
        content += chunk.content or ""
    assert [c["name"] for c in calls] == ["fetch_url"] * 3, label
    assert "<tool_call>" not in content
    assert [r.success for r in _host_tool_loop(calls, [fetch_url])] == [True] * 3


def test_prose_left_in_reasoning_is_kept():
    resp = _MLXOrderProvider("I need news.\n\n" + OPERATOR_TEXT + "\n</think>\n", opened_by_prompt=True).generate(
        prompt="news", tools=[fetch_url]
    )
    assert resp.metadata["reasoning"] == "I need news."
    assert resp.metadata["tool_calls_from_reasoning"] == 3


def test_call_drafted_mid_reasoning_is_not_executed_but_is_reported():
    text = "Maybe:\n" + OPERATOR_TEXT + "\nNo, I already know the answer.\n</think>\n"
    resp = _MLXOrderProvider(text, opened_by_prompt=True).generate(prompt="news", tools=[fetch_url])
    assert not resp.tool_calls
    assert any("inside the model's reasoning" in w for w in resp.metadata.get("warnings", []))


def test_visible_answer_wins_over_calls_left_in_reasoning():
    text = "Draft:\n" + OPERATOR_TEXT + "\n</think>\n\nHere is the news summary."
    resp = _MLXOrderProvider(text, opened_by_prompt=True).generate(prompt="news", tools=[fetch_url])
    assert not resp.tool_calls
    assert resp.content == "Here is the news summary."


def test_unknown_names_in_reasoning_are_warned_not_executed():
    resp = _MLXOrderProvider(OPERATOR_TEXT + "\n</think>\n", opened_by_prompt=True).generate(
        prompt="news", tools=[web_search]
    )
    assert not resp.tool_calls
    assert any("no such tool is available" in w for w in resp.metadata.get("warnings", []))


def test_unknown_names_in_content_are_dropped_with_warning():
    resp = _MLXOrderProvider(OPERATOR_TEXT, opened_by_prompt=False).generate(prompt="news", tools=[web_search])
    assert not resp.tool_calls
    assert any("'fetch_url'" in w for w in resp.metadata.get("warnings", []))


def test_without_tools_nothing_is_parsed_or_recovered():
    resp = _MLXOrderProvider(OPERATOR_TEXT + "\n</think>\n", opened_by_prompt=True).generate(prompt="news")
    assert not resp.tool_calls
    assert "tool_calls_from_reasoning" not in (resp.metadata or {})

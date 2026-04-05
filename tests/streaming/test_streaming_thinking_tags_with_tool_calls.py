from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _ThinkingTagsAndToolCallsStreamProvider(BaseProvider):
    def get_capabilities(self) -> list[str]:
        return ["streaming"]

    def list_available_models(self, **kwargs) -> list[str]:
        return [self.model]

    def unload_model(self, model_name: str) -> None:
        return None

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        response_model: Optional[Any] = None,
        execute_tools: Optional[bool] = None,
        media_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        _ = (prompt, messages, system_prompt, tools, media, response_model, execute_tools, media_metadata, kwargs)

        if not stream:
            return GenerateResponse(
                content="<think>r1\n<|tool_call|>{\"name\":\"search_files\",\"arguments\":{\"query\":\"abc\"}}</|tool_call|> more</think>Final",
                model=self.model,
                finish_reason="stop",
            )

        def _gen() -> Iterator[GenerateResponse]:
            # Reasoning begins.
            yield GenerateResponse(content="<think>r", model=self.model, finish_reason=None)
            # Tool call begins.
            yield GenerateResponse(content="1\n<|tool_call|>", model=self.model, finish_reason=None)
            yield GenerateResponse(content='{"name":"search_files","arguments":{"query":"abc"}}', model=self.model, finish_reason=None)
            # Tool call ends; more reasoning follows.
            yield GenerateResponse(content="</|tool_call|> more", model=self.model, finish_reason=None)
            # Reasoning ends; visible answer begins.
            yield GenerateResponse(content="</think>Final", model=self.model, finish_reason="stop")

        return _gen()


def test_streaming_strips_thinking_tags_and_tool_markup_and_extracts_reasoning() -> None:
    llm = _ThinkingTagsAndToolCallsStreamProvider(model="unit-test")
    # In real providers, thinking tags typically come from the architecture format config.
    llm.architecture_config = {"thinking_tags": ["<think>", "</think>"]}

    chunks = list(
        llm.generate(
            prompt="hi",
            tools=[
                {
                    "name": "search_files",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                }
            ],
            stream=True,
        )
    )

    visible = "".join(c.content or "" for c in chunks)
    assert visible == "Final"

    tool_calls = [call for c in chunks for call in (c.tool_calls or [])]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "search_files"
    assert tool_calls[0]["arguments"] == {"query": "abc"}

    assert chunks[-1].reasoning == "r1\n more"


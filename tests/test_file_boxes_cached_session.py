from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from abstractcore.core.cached_session import CachedSession
from abstractcore.core.file_boxes import extract_file_box, render_file_box_message
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


@dataclass
class _FakeCache:
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class _StubKVCacheProvider(BaseProvider):
    """Minimal provider with modular prompt-cache control plane + KV source-of-truth."""

    def __init__(self, model: str = "stub", **kwargs: Any):
        super().__init__(model, **kwargs)
        self.append_calls = 0

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        return True

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        return _FakeCache()

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        if not isinstance(cache_value, _FakeCache):
            return None
        return _FakeCache(chunks=list(cache_value.chunks))

    def _prompt_cache_backend_append(
        self,
        cache_value: Any,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> bool:
        _ = kwargs
        if not isinstance(cache_value, _FakeCache):
            return False
        self.append_calls += 1
        cache_value.chunks.append(
            {
                "prompt": str(prompt or ""),
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "add_generation_prompt": bool(add_generation_prompt),
            }
        )
        return True

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        if not isinstance(cache_value, _FakeCache):
            return None
        return len(cache_value.chunks)

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateResponse | Iterator[GenerateResponse]:
        _ = (prompt, messages, system_prompt, tools, media, stream, kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["stub"]


def test_extract_file_box_text(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello\nworld\n", encoding="utf-8")

    box = extract_file_box(p)
    assert box.path.endswith("note.txt")
    assert box.media_type in {"text", "document"}
    assert box.size_bytes > 0
    assert isinstance(box.sha256, str) and len(box.sha256) == 64
    assert "hello" in box.content
    assert isinstance(box.content_sha256, str) and len(box.content_sha256) == 64

    rendered = render_file_box_message(box)
    assert "<attached_file" in rendered
    assert "hello" in rendered


def test_cached_session_attach_files_appends_box_and_dedupes(tmp_path: Path) -> None:
    p = tmp_path / "context.md"
    p.write_text("# Title\n\nSome content.\n", encoding="utf-8")

    llm = _StubKVCacheProvider()
    session = CachedSession(
        provider=llm,
        system_prompt="You are helpful.",
        tools=[],
        prompt_cache_strategy="auto",
    )
    assert session.prompt_cache_mode == "kv"
    assert isinstance(session.prompt_cache_key, str) and session.prompt_cache_key

    before_calls = llm.append_calls
    result = session.attach_files([p])
    assert result["errors"] == []
    assert len(result["attached"]) == 1
    assert llm.append_calls == before_calls + 1

    attached = session.get_attached_files()
    assert len(attached) == 1
    assert attached[0]["path"].endswith("context.md")

    # Second call should skip without extracting/appending again (stat-based dedupe).
    before_calls = llm.append_calls
    result2 = session.attach_files([p])
    assert result2["errors"] == []
    assert result2["attached"] == []
    assert result2["skipped"] and result2["skipped"][0]["reason"] == "already_attached"
    assert llm.append_calls == before_calls

    # Ensure the provider cache recorded the attached box content in the last append.
    cache = llm._prompt_cache_store.get(session.prompt_cache_key)  # type: ignore[attr-defined]
    assert isinstance(cache, _FakeCache)
    assert cache.chunks
    last = cache.chunks[-1]
    msgs = last.get("messages")
    assert isinstance(msgs, list) and msgs
    assert "<attached_file" in str(msgs[0].get("content") or "")


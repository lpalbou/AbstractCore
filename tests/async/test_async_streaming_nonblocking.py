import asyncio
import contextlib
import time

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _SlowStreamingProvider(BaseProvider):
    def __init__(self, *, delay_s: float = 0.05, chunks: int = 3):
        super().__init__(model="dummy", temperature=0.0)
        self.provider = "dummy"
        self._delay_s = float(delay_s)
        self._chunks = int(chunks)

    def get_capabilities(self):
        return []

    def list_available_models(self, **kwargs):
        return []

    def unload_model(self, model_name: str) -> None:
        return None

    def _generate_internal(
        self,
        prompt: str,
        messages=None,
        system_prompt=None,
        tools=None,
        media=None,
        stream: bool = False,
        response_model=None,
        **kwargs,
    ):
        if not stream:
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

        def _gen():
            for i in range(self._chunks):
                time.sleep(self._delay_s)
                yield GenerateResponse(content=str(i), model=self.model, finish_reason=None)
            yield GenerateResponse(content="", model=self.model, finish_reason="stop")

        return _gen()


@pytest.mark.asyncio
async def test_async_streaming_does_not_block_event_loop() -> None:
    provider = _SlowStreamingProvider(delay_s=0.05, chunks=2)
    stream = await provider.agenerate("hi", stream=True)

    ticks = 0
    stop = asyncio.Event()

    async def _ticker():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.005)
            ticks += 1

    ticker_task = asyncio.create_task(_ticker())

    # Start waiting on the first chunk while the ticker runs. If the event loop is
    # blocked by sync iteration, ticks will remain 0 here.
    first_chunk_task = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0.02)
    assert ticks > 0

    first_chunk = await first_chunk_task
    assert first_chunk is not None

    stop.set()
    ticker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticker_task

    async for _ in stream:
        pass

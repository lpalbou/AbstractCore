from abstractcore.providers.huggingface_provider import HuggingFaceProvider


def test_gguf_stream_error_generator_keeps_exception_message() -> None:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.llm = object()
    provider.model = "test-gguf"
    provider.temperature = 0.7

    provider._gguf_build_chat_messages = lambda **kwargs: []  # type: ignore[method-assign]
    provider._prepare_generation_kwargs = lambda **kwargs: {}  # type: ignore[method-assign]
    provider._get_provider_max_tokens_param = lambda kwargs: 16  # type: ignore[method-assign]

    def raise_during_stream(*args, **kwargs):
        raise RuntimeError("stream setup failed")

    provider._stream_generate_gguf_with_tools = raise_during_stream  # type: ignore[method-assign]

    chunks = list(provider._generate_gguf("hello", stream=True))

    assert len(chunks) == 1
    assert chunks[0].content == "Error: stream setup failed"
    assert chunks[0].finish_reason == "error"

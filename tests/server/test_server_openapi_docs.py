from fastapi.testclient import TestClient

from abstractcore.server.app import app


def test_openapi_operations_are_grouped_with_tags():
    schema = TestClient(app).get("/openapi.json").json()
    missing = []
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                continue
            if not operation.get("tags"):
                missing.append(f"{method.upper()} {path}")

    assert missing == []


def test_core_endpoint_tags_are_documentation_friendly():
    schema = TestClient(app).get("/openapi.json").json()

    assert schema["paths"]["/v1/chat/completions"]["post"]["tags"] == ["chat"]
    assert schema["paths"]["/{provider}/v1/chat/completions"]["post"]["tags"] == ["chat"]
    assert schema["paths"]["/v1/models"]["get"]["tags"] == ["models"]
    assert schema["paths"]["/providers"]["get"]["tags"] == ["providers"]
    assert schema["paths"]["/v1/embeddings"]["post"]["tags"] == ["embeddings"]
    assert schema["paths"]["/v1/audio/speech"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/{provider}/v1/audio/speech"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/speech/models"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/speech/providers"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/music"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/{provider}/v1/audio/music"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/transcriptions"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/{provider}/v1/audio/transcriptions"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/transcriptions/models"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/transcriptions/providers"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/voices"]["get"]["tags"] == ["audio"]
    assert "/v1/audio/voices/models" not in schema["paths"]
    assert schema["paths"]["/v1/voice/clone"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/voice/clone/providers"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/voice/clone/models"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/{provider}/v1/voice/clone"]["post"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/music/providers"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/audio/music/models"]["get"]["tags"] == ["audio"]
    assert schema["paths"]["/v1/capabilities"]["get"]["tags"] == ["capabilities"]
    assert schema["paths"]["/v1/capabilities/{capability}/providers"]["get"]["tags"] == ["capabilities"]
    assert schema["paths"]["/v1/capabilities/{capability}/models"]["get"]["tags"] == ["capabilities"]
    assert schema["paths"]["/v1/images/generations"]["post"]["tags"] == ["vision"]
    assert schema["paths"]["/{provider}/v1/images/generations"]["post"]["tags"] == ["vision"]
    assert schema["paths"]["/{provider}/v1/images/edits"]["post"]["tags"] == ["vision"]
    assert schema["paths"]["/v1/vision/providers/"]["get"]["tags"] == ["vision"]
    assert "/v1/vision/provider_models" not in schema["paths"]
    assert "/v1/vision/model" not in schema["paths"]
    assert "/v1/vision/model/load" not in schema["paths"]
    assert "/v1/vision/model/unload" not in schema["paths"]
    assert schema["paths"]["/v1/responses"]["post"]["tags"] == ["responses"]
    assert schema["paths"]["/acore/models/loaded"]["get"]["tags"] == ["runtime"]
    assert schema["paths"]["/acore/models/load"]["post"]["tags"] == ["runtime"]
    assert schema["paths"]["/acore/models/unload"]["post"]["tags"] == ["runtime"]
    assert schema["paths"]["/acore/prompt_cache/stats"]["get"]["tags"] == ["prompt-cache"]
    assert schema["paths"]["/acore/blocs/upsert_text"]["post"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs"]["get"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/record"]["get"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/delete"]["post"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/kv/manifest"]["get"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/kv/list"]["get"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/kv/ensure"]["post"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/kv/load"]["post"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/kv/delete"]["post"]["tags"] == ["memory-blocs"]
    assert schema["paths"]["/acore/blocs/kv/prune"]["post"]["tags"] == ["memory-blocs"]


def test_openapi_documents_shared_responses_controls_and_prompt_cache_thinking():
    schema = TestClient(app).get("/openapi.json").json()
    responses_props = schema["components"]["schemas"]["OpenAIResponsesRequest"]["properties"]
    prompt_cache_update_props = schema["components"]["schemas"]["PromptCacheUpdateProxyRequest"]["properties"]

    for key in (
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "base_url",
        "agent_format",
        "thinking",
        "prompt_cache_key",
        "prompt_cache_retention",
        "timeout_s",
        "unload_after",
    ):
        assert key in responses_props
    assert "thinking" in prompt_cache_update_props


def test_audio_speech_documents_binary_audio_response():
    schema = TestClient(app).get("/openapi.json").json()
    response = schema["paths"]["/v1/audio/speech"]["post"]["responses"]["200"]
    content = response["content"]

    assert "audio/wav" in content
    assert "audio/mpeg" in content
    assert list(content)[0] == "audio/wav"
    assert content["audio/mpeg"]["schema"] == {"type": "string", "format": "binary"}


def test_media_swagger_examples_are_complete_and_executable_defaults():
    schema = TestClient(app).get("/openapi.json").json()

    image_example = schema["components"]["schemas"]["ImageGenerationBody"]["examples"][0]
    assert set(image_example) == {
        "model",
        "provider",
        "prompt",
        "n",
        "base_url",
        "size",
        "width",
        "height",
        "response_format",
        "negative_prompt",
        "seed",
        "steps",
        "guidance_scale",
        "quality",
        "style",
        "user",
        "background",
        "output_format",
        "output_compression",
        "moderation",
        "extra",
    }
    assert image_example["model"] == "openai-compatible/gpt-image-1"
    assert image_example["seed"] is None
    assert image_example["extra"] == {}

    speech_example = schema["components"]["schemas"]["AudioSpeechRequest"]["examples"][0]
    assert set(speech_example) == {
        "model",
        "input",
        "text",
        "voice",
        "profile",
        "response_format",
        "format",
        "speed",
        "quality_preset",
        "quality",
        "instructions",
        "provider",
        "base_url",
    }
    assert speech_example["voice"] == "coral"
    assert speech_example["response_format"] == "wav"
    assert speech_example["base_url"] is None
    assert speech_example["format"] is None


def test_all_request_bodies_have_swagger_examples():
    schema = TestClient(app).get("/openapi.json").json()
    missing = []

    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method.lower() not in {"post", "put", "patch"}:
                continue
            request_body = operation.get("requestBody") or {}
            for media_type, media in (request_body.get("content") or {}).items():
                has_example = bool(media.get("examples") or media.get("example"))
                ref = (media.get("schema") or {}).get("$ref")
                if ref:
                    component = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
                    has_example = has_example or bool(component.get("examples") or component.get("example"))
                else:
                    inline_schema = media.get("schema") or {}
                    has_example = has_example or bool(inline_schema.get("examples") or inline_schema.get("example"))
                if not has_example:
                    missing.append(f"{method.upper()} {path} {media_type}")

    assert missing == []


def test_standard_error_responses_are_documented():
    schema = TestClient(app).get("/openapi.json").json()
    responses = schema["paths"]["/v1/audio/speech"]["post"]["responses"]

    assert "401" in responses
    assert "502" in responses
    assert responses["502"]["content"]["application/json"]["schema"]["$ref"] == "#/components/schemas/AbstractCoreError"

from __future__ import annotations

from abstractcore.core.generate_contract import normalize_generate_request, resolve_generate_route
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _RouteProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(model="route-model")
        self.provider = "route-provider"

    def _generate_internal(
        self,
        prompt,
        messages=None,
        system_prompt=None,
        tools=None,
        media=None,
        stream=False,
        **kwargs,
    ):
        return GenerateResponse(content=f"generated:{prompt}", model=self.model, metadata={})

    def get_capabilities(self):
        return []

    def unload_model(self, model_name: str) -> None:
        return None

    def list_available_models(self, **kwargs):
        return [self.model]


def test_generate_request_keyword_is_supported_for_text_generate() -> None:
    llm = _RouteProvider()
    llm._abstractcore_capability_defaults = {  # type: ignore[attr-defined]
        "input.text": {
            "key": "input.text",
            "provider": "openai",
            "model": "gpt-5",
            "reasoning": "medium",
            "source": "abstractcore.capability_defaults",
        }
    }

    resp = llm.generate(request={"text": "hello from request"})

    assert resp.content == "generated:hello from request"
    assert isinstance(resp.metadata, dict)
    resolved = resp.metadata.get("_resolved_generate_route")
    assert isinstance(resolved, dict)
    assert resolved["request"]["text"] == "hello from request"
    assert resolved["outputs"] == [{"modality": "text", "task": "text_generation", "provider": "openai", "model": "gpt-5"}]
    assert resolved["text_route"]["model"] == "gpt-5"
    assert resolved["reasoning"] == "medium"


def test_resolve_generate_route_applies_task_specific_defaults_and_reasoning() -> None:
    request = normalize_generate_request(prompt="draw a red cube")
    route = resolve_generate_route(
        request=request,
        output={"modality": "image"},
        scoped_routes={
            "input.text": {
                "key": "input.text",
                "provider": "openai",
                "model": "gpt-5",
                "reasoning": "high",
                "source": "abstractcore.capability_defaults",
            },
            "output.image.text_to_image": {
                "key": "output.image.text_to_image",
                "provider": "mlx-gen",
                "model": "z-image",
                "options": {"width": 1024},
                "source": "abstractcore.capability_defaults",
            },
        },
    )

    assert route.reasoning == "high"
    assert route.text_route is not None
    assert route.text_route.model == "gpt-5"
    assert route.output_routes[0].route_key == "output.image.text_to_image"
    assert route.output_specs[0]["provider"] == "mlx-gen"
    assert route.output_specs[0]["model"] == "z-image"
    assert route.output_specs[0]["width"] == 1024


def test_explicit_output_override_wins_over_capability_default() -> None:
    request = normalize_generate_request(prompt="cinematic trailer")
    route = resolve_generate_route(
        request=request,
        output={
            "modality": "video",
            "provider": "explicit-provider",
            "model": "explicit-model",
        },
        scoped_routes={
            "output.video.text_to_video": {
                "key": "output.video.text_to_video",
                "provider": "mlx-gen",
                "model": "wan-default",
                "source": "abstractcore.capability_defaults",
            }
        },
    )

    assert route.output_routes[0].provider == "explicit-provider"
    assert route.output_routes[0].model == "explicit-model"
    assert route.output_routes[0].field_sources["provider"] == "explicit"
    assert route.output_routes[0].field_sources["model"] == "explicit"
    assert route.output_specs[0]["provider"] == "explicit-provider"
    assert route.output_specs[0]["model"] == "explicit-model"


def test_text_to_audio_route_maps_to_output_sound() -> None:
    request = normalize_generate_request(prompt="short notification chime")
    route = resolve_generate_route(
        request=request,
        output={"modality": "music", "task": "text_to_audio"},
        scoped_routes={
            "output.sound": {
                "key": "output.sound",
                "provider": "stable-audio",
                "model": "stable-sfx",
                "source": "abstractcore.capability_defaults",
            }
        },
    )

    assert route.output_routes[0].route_key == "output.sound"
    assert route.output_specs[0]["provider"] == "stable-audio"
    assert route.output_specs[0]["model"] == "stable-sfx"

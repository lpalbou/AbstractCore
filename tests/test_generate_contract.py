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


def _voice_route_scope(options: dict | None = None) -> dict:
    row = {
        "key": "output.voice",
        "provider": "supertonic",
        "model": "supertonic-3",
        "source": "gateway.capability_defaults",
    }
    if options is not None:
        row["options"] = dict(options)
    return {"output.voice": row}


def test_route_options_do_not_leak_into_explicitly_overridden_provider() -> None:
    # Live incident (2026-07-17): output.voice configured supertonic + options {voice: M1};
    # an explicit piper request with no voice inherited voice=M1 and failed
    # "Unknown voice_id: M1". Options are per-(provider, model) facts.
    request = normalize_generate_request(prompt="say hello")
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice", "provider": "piper", "model": "en_US-amy-medium"},
        scoped_routes=_voice_route_scope({"voice": "M1"}),
    )

    entry = route.output_routes[0]
    assert entry.provider == "piper"
    assert entry.options == {}
    assert entry.field_sources["options"] == "dropped:explicit_route_override"
    assert "voice" not in route.output_specs[0]


def test_overridden_route_row_contributes_no_identity_fill_ins() -> None:
    # Adversary gap 1: explicit provider redirect with NO explicit model used to
    # fill the ROUTE's model in (provider=piper + model=supertonic-3 — an
    # incoherent pair, the options leak's twin on the identity fields). An
    # overridden row contributes nothing.
    request = normalize_generate_request(prompt="say hello")
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice", "provider": "piper"},
        scoped_routes=_voice_route_scope({"voice": "M1"}),
    )

    entry = route.output_routes[0]
    assert entry.provider == "piper"
    assert entry.model is None
    assert entry.options == {}
    assert "model" not in route.output_specs[0]
    assert "voice" not in route.output_specs[0]


def test_disjoint_route_identities_never_mix() -> None:
    # Adversary gap 2: a row anchored ONLY by base_url (options configured for
    # that server) shares no identity field with an explicit provider/model
    # spec — the backends cannot be verified as the same, so nothing mixes.
    request = normalize_generate_request(prompt="say hello")
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice", "provider": "piper", "model": "en_US-amy-medium"},
        scoped_routes={
            "output.voice": {
                "key": "output.voice",
                "base_url": "http://tts.local:9000",
                "options": {"voice": "M1"},
                "source": "gateway.capability_defaults",
            }
        },
    )

    entry = route.output_routes[0]
    assert entry.base_url is None
    assert entry.options == {}
    assert "voice" not in route.output_specs[0]
    assert "base_url" not in route.output_specs[0]


def test_base_url_divergence_drops_options_same_provider() -> None:
    # Adversary gap 3: the same provider on a DIFFERENT server (moved-server /
    # proxy pattern) keeps identity fill-ins — a wrong model errors loudly
    # server-side while an unfilled one breaks the benign mirror case — but
    # OPTIONS (voice ids etc.) are server-side facts and must drop.
    request = normalize_generate_request(prompt="say hello")
    scoped = {
        "output.voice": {
            "key": "output.voice",
            "provider": "openai-compatible",
            "model": "tts-1",
            "base_url": "http://main:9000",
            "options": {"voice": "M1"},
            "source": "gateway.capability_defaults",
        }
    }
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice", "provider": "openai-compatible", "base_url": "http://other:9000"},
        scoped_routes=scoped,
    )
    entry = route.output_routes[0]
    assert entry.options == {}
    assert entry.field_sources["options"] == "dropped:explicit_route_override"
    assert entry.model == "tts-1"
    assert "voice" not in route.output_specs[0]

    # Same server spelled identically keeps the row's full contribution.
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice", "provider": "openai-compatible", "base_url": "http://main:9000"},
        scoped_routes=scoped,
    )
    assert route.output_specs[0]["model"] == "tts-1"
    assert route.output_specs[0]["voice"] == "M1"


def test_base_url_only_override_keeps_configured_text_route() -> None:
    # Guard for the classic proxy pattern: a caller overriding ONLY base_url
    # against a provider+model config row is moving the SAME backend to another
    # address — the row must keep contributing (this is long-standing behavior
    # the stricter override rules must not break).
    request = normalize_generate_request(prompt="hello")
    route = resolve_generate_route(
        request=request,
        scoped_routes={
            "input.text": {
                "key": "input.text",
                "provider": "openai",
                "model": "gpt-5",
                "source": "abstractcore.capability_defaults",
            }
        },
        explicit_text_route={"provider": None, "model": None, "base_url": "http://proxy:8080"},
    )
    assert route.text_route is not None
    assert route.text_route.provider == "openai"
    assert route.text_route.model == "gpt-5"
    assert route.text_route.base_url == "http://proxy:8080"


def test_route_options_apply_when_route_backend_is_the_one_used() -> None:
    request = normalize_generate_request(prompt="say hello")

    # No explicit provider/model: the route's own backend is used, options ride.
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice"},
        scoped_routes=_voice_route_scope({"voice": "M1"}),
    )
    assert route.output_specs[0]["provider"] == "supertonic"
    assert route.output_specs[0]["voice"] == "M1"

    # Explicitly naming the same backend (case-insensitive) keeps the options.
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice", "provider": "Supertonic", "model": "supertonic-3"},
        scoped_routes=_voice_route_scope({"voice": "M1"}),
    )
    assert route.output_specs[0]["voice"] == "M1"


def test_route_options_dropped_on_model_only_divergence() -> None:
    # Same provider but a different model: options were configured for the
    # route's own (provider, model) pair and must not ride onto another model.
    request = normalize_generate_request(prompt="say hello")
    route = resolve_generate_route(
        request=request,
        output={"modality": "voice", "provider": "supertonic", "model": "supertonic-9"},
        scoped_routes=_voice_route_scope({"voice": "M1"}),
    )
    assert route.output_routes[0].options == {}
    assert "voice" not in route.output_specs[0]


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

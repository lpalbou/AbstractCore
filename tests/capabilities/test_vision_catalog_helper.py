from __future__ import annotations

import json
from pathlib import Path

from abstractcore.capabilities import vision_catalog


class _FakeVisionSpec:
    def __init__(self, *, provider: str, license: str, tasks: dict[str, dict], notes):
        self.provider = provider
        self.license = license
        self.tasks = tasks
        self.notes = notes


def test_local_vision_cache_catalog_shapes_registry_and_discovered_models(monkeypatch, tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    local_dir = tmp_path / "local"
    lmstudio_dir = tmp_path / "lmstudio"
    hf_dir.mkdir()
    local_dir.mkdir()
    lmstudio_dir.mkdir()

    specs = {
        "flux-dev": _FakeVisionSpec(
            provider="mflux",
            license="apache-2.0",
            tasks={"text_to_image": {}, "image_to_image": {}},
            notes=("popular",),
        ),
        "text-only": _FakeVisionSpec(
            provider="noop",
            license="unknown",
            tasks={"text_generation": {}},
            notes=(),
        ),
    }

    class _Registry:
        def list_models(self):
            return list(specs)

        def get(self, model_id):
            return specs[model_id]

    monkeypatch.setattr(vision_catalog, "_load_vision_model_capabilities_registry", lambda: _Registry)
    monkeypatch.setattr(vision_catalog, "_default_hf_hub_cache_dirs", lambda: [hf_dir])
    monkeypatch.setattr(vision_catalog, "_default_local_diffusers_model_dirs", lambda: [local_dir])
    monkeypatch.setattr(vision_catalog, "_default_lmstudio_model_dirs", lambda: [lmstudio_dir])
    monkeypatch.setattr(vision_catalog, "_is_hf_model_cached", lambda model_id, _dirs: model_id == "flux-dev")
    monkeypatch.setattr(vision_catalog, "_is_lmstudio_model_cached", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(vision_catalog, "_discover_cached_hf_diffusers_models", lambda _dirs: ["black-forest-labs/FLUX.2-dev"])
    monkeypatch.setattr(vision_catalog, "_discover_local_diffusers_models", lambda _dirs: ["local/model"])

    payload = vision_catalog.get_local_vision_cache_catalog()

    assert "active" not in payload
    assert payload["registry_available"] is True
    assert payload["registry_total"] == 2
    assert payload["cached_total"] == 3
    assert payload["cache_dirs"] == {
        "huggingface": [str(hf_dir)],
        "local_diffusers": [str(local_dir)],
        "lmstudio": [str(lmstudio_dir)],
    }
    assert payload["models"] == [
        {
            "id": "black-forest-labs/FLUX.2-dev",
            "provider": "huggingface",
            "license": "unknown",
            "tasks": ["image_to_image", "text_to_image"],
            "notes": "Discovered from the local Hugging Face cache (image-capable diffusers model_index.json present).",
            "cached_in": ["huggingface"],
            "discovered": True,
        },
        {
            "id": "flux-dev",
            "provider": "mflux",
            "license": "apache-2.0",
            "tasks": ["image_to_image", "text_to_image"],
            "notes": ("popular",),
            "cached_in": ["huggingface"],
        },
        {
            "id": "local/model",
            "provider": "huggingface",
            "license": "unknown",
            "tasks": ["image_to_image", "text_to_image"],
            "notes": "Discovered from a local AbstractVision Diffusers model directory (image-capable model_index.json present).",
            "cached_in": ["local_diffusers"],
            "discovered": True,
        },
    ]


def test_local_vision_cache_catalog_returns_bounded_error_without_registry(monkeypatch, tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    local_dir = tmp_path / "local"
    lmstudio_dir = tmp_path / "lmstudio"
    hf_dir.mkdir()
    local_dir.mkdir()
    lmstudio_dir.mkdir()

    missing = ModuleNotFoundError("No module named 'abstractvision'")
    missing.name = "abstractvision"
    monkeypatch.setattr(
        vision_catalog,
        "_load_vision_model_capabilities_registry",
        lambda: (_ for _ in ()).throw(missing),
    )
    monkeypatch.setattr(vision_catalog, "_default_hf_hub_cache_dirs", lambda: [hf_dir])
    monkeypatch.setattr(vision_catalog, "_default_local_diffusers_model_dirs", lambda: [local_dir])
    monkeypatch.setattr(vision_catalog, "_default_lmstudio_model_dirs", lambda: [lmstudio_dir])

    payload = vision_catalog.get_local_vision_cache_catalog()

    assert "active" not in payload
    assert payload == {
        "models": [],
        "registry_available": False,
        "registry_total": 0,
        "cached_total": 0,
        "cache_dirs": {
            "huggingface": [str(hf_dir)],
            "local_diffusers": [str(local_dir)],
            "lmstudio": [str(lmstudio_dir)],
        },
        "error": "AbstractVision is required for vision model registry endpoints. Install `abstractvision`.",
    }


def test_local_vision_cache_catalog_surfaces_registry_load_failures(monkeypatch, tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    local_dir = tmp_path / "local"
    lmstudio_dir = tmp_path / "lmstudio"
    hf_dir.mkdir()
    local_dir.mkdir()
    lmstudio_dir.mkdir()

    monkeypatch.setattr(
        vision_catalog,
        "_load_vision_model_capabilities_registry",
        lambda: (_ for _ in ()).throw(ImportError("backend registry wiring failed")),
    )
    monkeypatch.setattr(vision_catalog, "_default_hf_hub_cache_dirs", lambda: [hf_dir])
    monkeypatch.setattr(vision_catalog, "_default_local_diffusers_model_dirs", lambda: [local_dir])
    monkeypatch.setattr(vision_catalog, "_default_lmstudio_model_dirs", lambda: [lmstudio_dir])

    payload = vision_catalog.get_local_vision_cache_catalog()

    assert payload["registry_available"] is False
    assert payload["error"] == "Failed to load AbstractVision registry: backend registry wiring failed"
    assert payload["cache_dirs"] == {
        "huggingface": [str(hf_dir)],
        "local_diffusers": [str(local_dir)],
        "lmstudio": [str(lmstudio_dir)],
    }


def test_local_vision_cache_catalog_surfaces_nested_abstractvision_import_failures(monkeypatch, tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    local_dir = tmp_path / "local"
    lmstudio_dir = tmp_path / "lmstudio"
    hf_dir.mkdir()
    local_dir.mkdir()
    lmstudio_dir.mkdir()

    missing = ModuleNotFoundError("No module named 'abstractvision.model_capabilities'")
    missing.name = "abstractvision.model_capabilities"
    monkeypatch.setattr(
        vision_catalog,
        "_load_vision_model_capabilities_registry",
        lambda: (_ for _ in ()).throw(missing),
    )
    monkeypatch.setattr(vision_catalog, "_default_hf_hub_cache_dirs", lambda: [hf_dir])
    monkeypatch.setattr(vision_catalog, "_default_local_diffusers_model_dirs", lambda: [local_dir])
    monkeypatch.setattr(vision_catalog, "_default_lmstudio_model_dirs", lambda: [lmstudio_dir])

    payload = vision_catalog.get_local_vision_cache_catalog()

    assert payload["registry_available"] is False
    assert payload["error"] == "Failed to load AbstractVision registry: No module named 'abstractvision.model_capabilities'"
    assert payload["cache_dirs"] == {
        "huggingface": [str(hf_dir)],
        "local_diffusers": [str(local_dir)],
        "lmstudio": [str(lmstudio_dir)],
    }


def test_local_vision_cache_catalog_surfaces_registry_init_failures(monkeypatch, tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    local_dir = tmp_path / "local"
    lmstudio_dir = tmp_path / "lmstudio"
    hf_dir.mkdir()
    local_dir.mkdir()
    lmstudio_dir.mkdir()

    class _BrokenRegistry:
        def __init__(self) -> None:
            raise RuntimeError("registry init failed")

    monkeypatch.setattr(vision_catalog, "_load_vision_model_capabilities_registry", lambda: _BrokenRegistry)
    monkeypatch.setattr(vision_catalog, "_default_hf_hub_cache_dirs", lambda: [hf_dir])
    monkeypatch.setattr(vision_catalog, "_default_local_diffusers_model_dirs", lambda: [local_dir])
    monkeypatch.setattr(vision_catalog, "_default_lmstudio_model_dirs", lambda: [lmstudio_dir])

    payload = vision_catalog.get_local_vision_cache_catalog()

    assert payload["registry_available"] is False
    assert payload["error"] == "Failed to initialize AbstractVision registry: registry init failed"
    assert payload["cache_dirs"] == {
        "huggingface": [str(hf_dir)],
        "local_diffusers": [str(local_dir)],
        "lmstudio": [str(lmstudio_dir)],
    }


def test_cached_diffusers_discovery_filters_non_image_pipelines(tmp_path: Path) -> None:
    def write_model_index(model_id: str, class_name: str, extra: dict | None = None) -> None:
        folder = tmp_path / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc"
        folder.mkdir(parents=True)
        payload = {"_class_name": class_name}
        if extra:
            payload.update(extra)
        (folder / "model_index.json").write_text(json.dumps(payload), encoding="utf-8")

    write_model_index("black-forest-labs/FLUX.2-dev", "FluxPipeline")
    write_model_index(
        "ACE-Step/acestep-v15-xl-turbo-diffusers",
        "AceStepPipeline",
        {"vae": ["diffusers", "AutoencoderOobleck"], "transformer": ["diffusers", "AceStepTransformer1DModel"]},
    )

    assert vision_catalog._discover_cached_hf_diffusers_models([tmp_path]) == ["black-forest-labs/FLUX.2-dev"]
    assert not vision_catalog._is_hf_model_cached("ACE-Step/acestep-v15-xl-turbo-diffusers", [tmp_path])

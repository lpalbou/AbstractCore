from __future__ import annotations

from pathlib import Path


def _extract_optional_dependency_block(text: str, *, key: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{key} = ["):
            start = i
            break
    assert start is not None, f"Missing optional-dependencies entry: {key}"

    block: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() == "]":
            break
        block.append(line)
    return "\n".join(block)


def _assert_block_has_dependency_prefix(block: str, prefix: str) -> None:
    assert any(
        line.strip().strip('",').startswith(prefix)
        for line in block.splitlines()
    ), f"Missing dependency starting with {prefix!r}"


def test_tools_extra_includes_bs4_and_tool_alias_exists() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    tools_block = _extract_optional_dependency_block(text, key="tools")
    tool_block = _extract_optional_dependency_block(text, key="tool")

    assert "beautifulsoup4" in tools_block
    assert "beautifulsoup4" in tool_block


def test_server_extra_stays_vision_runtime_light() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    server_block = _extract_optional_dependency_block(text, key="server")
    voice_block = _extract_optional_dependency_block(text, key="voice")
    audio_block = _extract_optional_dependency_block(text, key="audio")
    vision_block = _extract_optional_dependency_block(text, key="vision")
    vision_diffusers_block = _extract_optional_dependency_block(text, key="vision-diffusers")
    vision_sdcpp_block = _extract_optional_dependency_block(text, key="vision-sdcpp")
    vision_local_block = _extract_optional_dependency_block(text, key="vision-local")
    music_block = _extract_optional_dependency_block(text, key="music")
    all_apple_block = _extract_optional_dependency_block(text, key="all-apple")
    all_gpu_block = _extract_optional_dependency_block(text, key="all-gpu")
    full_dev_block = _extract_optional_dependency_block(text, key="full-dev")

    assert "abstractvision" not in server_block
    assert "abstractvoice" not in server_block
    assert "abstractvoice>=0.10.17" in voice_block
    assert "abstractmusic" not in voice_block
    assert "abstractvoice>=0.10.17" in audio_block
    _assert_block_has_dependency_prefix(vision_block, "abstractvision>=")
    _assert_block_has_dependency_prefix(vision_diffusers_block, "abstractvision[huggingface]>=")
    _assert_block_has_dependency_prefix(vision_sdcpp_block, "abstractvision[sdcpp]>=")
    _assert_block_has_dependency_prefix(vision_local_block, "abstractvision[local]>=")
    assert "abstractmusic>=0.1.13" in music_block
    assert "abstractvoice[all-apple]>=0.10.17" in all_apple_block
    assert "omnivoice>=0.1.5" in all_apple_block
    _assert_block_has_dependency_prefix(all_apple_block, "abstractvision[all-apple]>=")
    assert "abstractmusic[all-apple]>=0.1.13" in all_apple_block
    assert "vllm" not in all_apple_block
    assert "abstractvoice[all-gpu]>=0.10.17" in all_gpu_block
    assert "omnivoice>=0.1.5" in all_gpu_block
    _assert_block_has_dependency_prefix(all_gpu_block, "abstractvision[all-gpu]>=")
    assert "abstractmusic[all-gpu]>=0.1.13" in all_gpu_block
    assert "mlx-lm" not in all_gpu_block
    assert "abstractvoice>=0.10.17" in full_dev_block
    assert "omnivoice>=0.1.5" in full_dev_block
    _assert_block_has_dependency_prefix(full_dev_block, "abstractvision>=")
    assert "abstractmusic>=0.1.13" in full_dev_block

    assert "transformers>=5.3.0,<6.0.0" in all_apple_block
    assert "torch>=2.7.1,<3.0.0" in all_apple_block
    assert "llama-cpp-python>=0.3.23,<1.0.0" in all_apple_block
    assert "accelerate>=1.0.0" in all_apple_block
    assert "numpy>=2.1.0,<3.0.0" in all_apple_block
    assert "Pillow>=12.1.1,<13.0.0" in all_apple_block


def test_light_capability_extras_do_not_pull_local_inference_engines() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    light_blocks = {
        "voice": _extract_optional_dependency_block(text, key="voice"),
        "audio": _extract_optional_dependency_block(text, key="audio"),
        "vision": _extract_optional_dependency_block(text, key="vision"),
        "music": _extract_optional_dependency_block(text, key="music"),
    }
    local_runtime_markers = (
        "omnivoice",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "mlx",
        "vllm",
    )

    for extra, block in light_blocks.items():
        for marker in local_runtime_markers:
            assert marker not in block, f"{extra} unexpectedly includes {marker}"


def test_permissive_pdf_profiles_do_not_pull_pymupdf_family() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    project_deps_block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "pymupdf" not in project_deps_block.lower()

    permissive_profile_keys = (
        "media",
        "all",
        "all-apple",
        "all-gpu",
        "all-non-mlx",
        "full-dev",
        "test",
    )
    for key in permissive_profile_keys:
        block = _extract_optional_dependency_block(text, key=key)
        assert "pypdf>=6.0.0,<7.0.0" in block
        assert "pymupdf4llm" not in block
        assert "pymupdf-layout" not in block

    server_block = _extract_optional_dependency_block(text, key="server")
    assert "pymupdf" not in server_block.lower()

    commercial_block = _extract_optional_dependency_block(text, key="pdf-pymupdf-commercial")
    assert "pypdf>=6.0.0,<7.0.0" in commercial_block
    assert "pymupdf4llm>=0.0.20,<1.0.0" in commercial_block
    assert "pymupdf-layout>=1.26.6,<2.0.0" in commercial_block


def test_server_docker_image_installs_exact_lightweight_release_wheel() -> None:
    dockerfile = Path(__file__).resolve().parents[1] / "docker" / "abstractcore-server" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    assert "https://pypi.org/pypi/abstractcore/" in text
    assert "ABSTRACTCORE_WHEEL_URL" in text
    assert "abstractcore[server,remote,media,tokens,compression] @ ${ABSTRACTCORE_WHEEL_URL}" in text
    assert "abstractcore[server,remote,media,tokens,compression,voice,vision]" not in text


def test_hardware_profile_aliases_match_provider_specific_local_engine_extras() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    mlx_block = _extract_optional_dependency_block(text, key="mlx")
    apple_block = _extract_optional_dependency_block(text, key="apple")
    vllm_block = _extract_optional_dependency_block(text, key="vllm")
    gpu_block = _extract_optional_dependency_block(text, key="gpu")

    for dep in ("mlx>=0.30.0,<1.0.0", "mlx-lm>=0.30.0,<1.0.0", "outlines>=0.1.0"):
        assert dep in mlx_block
        assert dep in apple_block
    assert "vllm>=0.6.0,<1.0.0" in vllm_block
    assert "vllm>=0.6.0,<1.0.0" in gpu_block

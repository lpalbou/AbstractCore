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
    full_dev_block = _extract_optional_dependency_block(text, key="full-dev")

    assert "abstractvision" not in server_block
    assert "abstractvoice" not in server_block
    assert "abstractvoice>=0.9.0" in voice_block
    assert "abstractvoice>=0.9.0" in audio_block
    assert "abstractvision>=0.3.1" in vision_block
    assert "abstractvision[huggingface]>=0.3.1" in vision_diffusers_block
    assert "abstractvision[sdcpp]>=0.3.1" in vision_sdcpp_block
    assert "abstractvoice>=0.9.0" in full_dev_block
    assert "abstractvision>=0.3.1" in full_dev_block


def test_server_docker_image_installs_exact_lightweight_release_wheel() -> None:
    dockerfile = Path(__file__).resolve().parents[1] / "docker" / "abstractcore-server" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    assert "https://pypi.org/pypi/abstractcore/" in text
    assert "ABSTRACTCORE_WHEEL_URL" in text
    assert "abstractcore[server,remote,media,tokens,compression] @ ${ABSTRACTCORE_WHEEL_URL}" in text
    assert "abstractcore[server,remote,media,tokens,compression,voice,vision]" not in text

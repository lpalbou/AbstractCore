import json
from pathlib import Path

from abstractcore.providers.huggingface_provider import HuggingFaceProvider


def test_find_gguf_in_cache_resolves_lmstudio_hub_alias(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Create LM Studio Hub manifest that aliases to a GGUF dependency.
    manifest_dir = tmp_path / ".lmstudio" / "hub" / "models" / "qwen" / "qwen3.5-35b-a3b"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "type": "model",
                "owner": "qwen",
                "name": "qwen3.5-35b-a3b",
                "dependencies": [
                    {
                        "type": "model",
                        "purpose": "baseModel",
                        "modelKeys": ["lmstudio-community/qwen3.5-35b-a3b-gguf"],
                        "sources": [
                            {
                                "type": "huggingface",
                                "user": "lmstudio-community",
                                "repo": "Qwen3.5-35B-A3B-GGUF",
                            }
                        ],
                    }
                ],
                "revision": 1,
            }
        ),
        encoding="utf-8",
    )

    # Create the resolved GGUF model in LM Studio's model cache.
    model_dir = tmp_path / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3.5-35B-A3B-GGUF"
    model_dir.mkdir(parents=True)
    gguf_path = model_dir / "Qwen3.5-35B-A3B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    found = provider._find_gguf_in_cache("qwen/qwen3.5-35b-a3b")

    assert found == str(gguf_path)

from abstractcore.architectures import detect_architecture, get_model_capabilities
from abstractcore.architectures import detection as architecture_detection


def test_qwen_audio_understanding_models_resolve_as_audio_capable() -> None:
    expected_audio_routes = {"speech", "sound", "music"}

    qwen3_instruct = get_model_capabilities("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    assert qwen3_instruct.get("architecture") == "qwen3_omni"
    assert qwen3_instruct.get("audio_support") is True
    assert set(qwen3_instruct.get("audio_input_capabilities", [])) == expected_audio_routes
    assert qwen3_instruct.get("vision_support") is True
    assert qwen3_instruct.get("video_support") is True
    assert qwen3_instruct.get("video_input_mode") == "native"

    qwen3_captioner = get_model_capabilities("qwen/qwen3-omni-30b-a3b-captioner")
    assert qwen3_captioner.get("architecture") == "qwen3_omni"
    assert qwen3_captioner.get("audio_support") is True
    assert set(qwen3_captioner.get("audio_input_capabilities", [])) == expected_audio_routes
    assert qwen3_captioner.get("vision_support") is False
    assert "environmental sounds" in qwen3_captioner.get("notes", "")
    assert "music" in qwen3_captioner.get("notes", "")

    qwen25_omni = get_model_capabilities("Qwen/Qwen2.5-Omni-7B")
    assert qwen25_omni.get("architecture") == "qwen2_5_omni"
    assert qwen25_omni.get("audio_support") is True
    assert set(qwen25_omni.get("audio_input_capabilities", [])) == expected_audio_routes
    assert qwen25_omni.get("video_support") is True

    qwen2_audio = get_model_capabilities("Qwen/Qwen2-Audio-7B-Instruct")
    assert qwen2_audio.get("architecture") == "qwen2_audio"
    assert qwen2_audio.get("audio_support") is True
    assert set(qwen2_audio.get("audio_input_capabilities", [])) == expected_audio_routes
    assert qwen2_audio.get("vision_support") is False
    assert qwen2_audio.get("video_support") is False


def test_qwen_audio_understanding_aliases_and_architectures() -> None:
    architecture_detection._load_json_assets()
    models = (architecture_detection._model_capabilities or {}).get("models", {})

    assert (
        architecture_detection.resolve_model_alias("Qwen3-Omni-30B-A3B-Instruct", models)
        == "qwen3-omni-30b-a3b-instruct"
    )
    assert (
        architecture_detection.resolve_model_alias("qwen3-omni-captioner", models)
        == "qwen3-omni-30b-a3b-captioner"
    )
    assert architecture_detection.resolve_model_alias("qwen2.5-omni:7b", models) == "qwen2.5-omni-7b"
    assert architecture_detection.resolve_model_alias("qwen2-audio:7b", models) == "qwen2-audio-7b-instruct"

    assert detect_architecture("Qwen/Qwen3-Omni-30B-A3B-Instruct") == "qwen3_omni"
    assert detect_architecture("Qwen/Qwen2.5-Omni-7B") == "qwen2_5_omni"
    assert detect_architecture("Qwen/Qwen2-Audio-7B-Instruct") == "qwen2_audio"


def test_qwen3_6_remains_not_audio_capable() -> None:
    caps = get_model_capabilities("Qwen/Qwen3.6-35B-A3B")
    architecture_detection._load_json_assets()
    models = (architecture_detection._model_capabilities or {}).get("models", {})
    assert architecture_detection.resolve_model_alias("Qwen/Qwen3.6-35B-A3B", models) == "qwen3.6-35b-a3b"
    assert caps.get("vision_support") is True
    assert caps.get("video_support") is True
    assert caps.get("audio_support") is False

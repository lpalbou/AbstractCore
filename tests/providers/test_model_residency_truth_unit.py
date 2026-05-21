from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.ollama_provider import OllamaProvider


def test_mlx_model_residency_reports_in_process_loaded_state() -> None:
    provider = object.__new__(MLXProvider)
    provider.provider = "mlx"
    provider.model = "mlx-test-model"
    provider.llm = object()
    provider.tokenizer = object()

    residency = provider.get_model_residency()

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is True
    assert residency["loaded"] is True
    assert residency["source"] == "abstractcore.provider.mlx"


def test_mlx_model_residency_reports_in_process_not_loaded_state() -> None:
    provider = object.__new__(MLXProvider)
    provider.provider = "mlx"
    provider.model = "mlx-test-model"
    provider.llm = None
    provider.tokenizer = None

    residency = provider.get_model_residency()

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is False
    assert residency["loaded"] is False
    assert residency["state"] == "not_loaded"


def test_huggingface_model_residency_reports_transformers_loaded_state() -> None:
    provider = object.__new__(HuggingFaceProvider)
    provider.provider = "huggingface"
    provider.model = "hf-test-model"
    provider.llm = None
    provider.model_instance = object()
    provider.pipeline = object()

    residency = provider.get_model_residency()

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is True
    assert residency["loaded"] is True
    assert residency["source"] == "abstractcore.provider.huggingface"


def test_huggingface_model_residency_reports_not_loaded_state() -> None:
    provider = object.__new__(HuggingFaceProvider)
    provider.provider = "huggingface"
    provider.model = "hf-test-model"
    provider.llm = None
    provider.model_instance = None
    provider.pipeline = None

    residency = provider.get_model_residency()

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is False
    assert residency["loaded"] is False
    assert residency["state"] == "not_loaded"


def test_huggingface_unload_clears_transformers_residency_without_losing_model_id() -> None:
    provider = object.__new__(HuggingFaceProvider)
    provider.provider = "huggingface"
    provider.model = "hf-test-model"
    provider.llm = None
    provider.tokenizer = object()
    provider.processor = object()
    provider.model_instance = object()
    provider.pipeline = object()

    provider.unload_model("hf-test-model")
    residency = provider.get_model_residency(model="hf-test-model")

    assert provider.model == "hf-test-model"
    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is False
    assert residency["loaded"] is False


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.ps_models = []
        self.posts = []

    def get(self, url: str):  # noqa: ANN001
        import httpx

        return httpx.Response(
            200,
            json={"models": self.ps_models},
            request=httpx.Request("GET", url),
        )

    def post(self, url: str, *, json=None):  # noqa: ANN001
        import httpx

        self.posts.append({"url": url, "json": json})
        return httpx.Response(
            200,
            json={"model": json.get("model"), "done": True, "done_reason": "load"},
            request=httpx.Request("POST", url),
        )


def test_ollama_model_residency_reports_running_model_from_api_ps() -> None:
    provider = object.__new__(OllamaProvider)
    provider.provider = "ollama"
    provider.model = "gemma3:1b"
    provider.base_url = "http://localhost:11434"
    provider.client = _FakeOllamaClient()
    provider.client.ps_models = [
        {
            "name": "gemma3:1b",
            "model": "gemma3:1b",
            "expires_at": "2318-08-31T12:29:48+02:00",
            "size_vram": 123,
            "context_length": 32768,
        }
    ]

    residency = provider.get_model_residency()

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is True
    assert residency["loaded"] is True
    assert residency["provider_instance_ids"] == ["gemma3:1b"]
    assert residency["source"] == "abstractcore.provider.ollama.native_rest"
    assert residency["size_vram"] == 123


def test_ollama_model_residency_reports_not_loaded_from_api_ps() -> None:
    provider = object.__new__(OllamaProvider)
    provider.provider = "ollama"
    provider.model = "gemma3:1b"
    provider.base_url = "http://localhost:11434"
    provider.client = _FakeOllamaClient()
    provider.client.ps_models = []

    residency = provider.get_model_residency()

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is False
    assert residency["loaded"] is False
    assert residency["state"] == "not_loaded"


def test_ollama_load_model_uses_native_keep_alive_for_pinned_load() -> None:
    provider = object.__new__(OllamaProvider)
    provider.provider = "ollama"
    provider.model = "gemma3:1b"
    provider.base_url = "http://localhost:11434"
    provider.client = _FakeOllamaClient()

    out = provider.load_model("gemma3:1b", pin=True)

    assert out["supported"] is True
    assert out["operation"] == "load"
    assert out["keep_alive"] == -1
    assert provider.client.posts == [
        {
            "url": "http://localhost:11434/api/generate",
            "json": {"model": "gemma3:1b", "prompt": "", "stream": False, "keep_alive": -1},
        }
    ]

"""TypeSafe / System One provider: hosted decision models (Jev) and compatible endpoints.

- ``create_llm("typesafe", model="jev-latest")`` talks to TypeSafe's hosted Jev at
  ``https://api.typesafe.ai`` with ``TYPESAFE_API_KEY`` (or ``api_key=``, or the
  AbstractCore config: ``abstractcore --set-api-key typesafe <key>``).
- ``create_llm("systemone", model="...", base_url="https://host")`` talks to any host that
  speaks the same ``POST /v1/systemone`` wire format (DigitalOcean Serverless Inference, a
  ``kev.serve`` / ``lev serve`` process, another AbstractCore server).

The provider has no text lane: decision models answer typed questions and never generate
text, so ``generate(...)`` without ``output="decision"`` is refused with a pointer.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..core.types import GenerateResponse
from ..decision.engines.remote import TYPESAFE_BASE_URL, RemoteSystemOneModel
from .base import BaseProvider

# What every model behind a System One endpoint is: text in, typed decisions out.
DECISION_ROUTES = ("input.text", "output.decision")


def decision_filter_matches(kwargs: Dict[str, Any]) -> bool:
    """Whether a model-discovery filter (``capability_routes`` / ``input_capabilities`` /
    ``output_capabilities``) admits a decision model, so a text, image or embedding listing
    never offers one."""
    from .model_capabilities import normalize_capability_route_filter

    if any(route not in DECISION_ROUTES for route in normalize_capability_route_filter(kwargs.get("capability_routes"))):
        return False

    def names(items: Any) -> set:
        return {str(getattr(item, "value", item)).strip().lower() for item in (items or ())}

    return not (names(kwargs.get("input_capabilities")) - {"text"}) and not (names(kwargs.get("output_capabilities")) - {"decision"})


class TypeSafeProvider(BaseProvider):
    API_KEY_ENV_VAR = "TYPESAFE_API_KEY"
    DEFAULT_MODEL = "jev-latest"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider_name: str = "typesafe",
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(model or self.DEFAULT_MODEL, **kwargs)
        self.provider = provider_name
        self.model_type = "decision"
        self.base_url = str(base_url or "").strip() or (TYPESAFE_BASE_URL if provider_name == "typesafe" else "")
        if not self.base_url:
            raise ValueError("the systemone provider needs base_url=: the host that serves POST /v1/systemone")
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV_VAR) or self._api_key_from_config()
        if provider_name == "typesafe" and not self.api_key:
            raise ValueError(
                "TypeSafe API key required. Set TYPESAFE_API_KEY, pass api_key=, or configure it with "
                "`abstractcore --set-api-key typesafe <key>`."
            )
        self.timeout = float(timeout) if timeout else 120.0
        self._decision_model = None

    def _api_key_from_config(self) -> Optional[str]:
        try:
            from ..config.manager import get_config_manager

            keys = getattr(get_config_manager().config, "api_keys", None)
            value = getattr(keys, "typesafe", None) if keys is not None else None
            return str(value).strip() or None if value else None
        except Exception:
            return None

    # --- decision lane --------------------------------------------------------

    def _build_decision_model(self) -> RemoteSystemOneModel:
        return RemoteSystemOneModel(self.base_url, self.model, api_key=self.api_key, provider=self.provider, timeout=self.timeout)

    def decision_model(self) -> RemoteSystemOneModel:
        if self._decision_model is None:
            self._decision_model = self._build_decision_model()
        return self._decision_model

    # --- BaseProvider contract --------------------------------------------------

    def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, response_model=None, **kwargs):
        from ..exceptions import DecisionModelError

        raise DecisionModelError(self.model, self.provider)

    async def _agenerate_internal(self, *args: Any, **kwargs: Any) -> GenerateResponse:  # pragma: no cover - same refusal
        return self._generate_internal(*args, **kwargs)

    def get_capabilities(self) -> List[str]:
        return ["decision"]

    def list_available_models(self, **kwargs: Any) -> List[str]:
        if not decision_filter_matches(kwargs):
            return []
        models = self.decision_model().list_models()
        return models or [self.model]

    def unload_model(self, model_name: str) -> None:
        if self._decision_model is not None:
            self._decision_model.unload()
            self._decision_model = None

    def validate_config(self) -> bool:
        return bool(self.base_url)


class SystemOneProvider(TypeSafeProvider):
    """Any System One-compatible endpoint (``base_url`` required, key optional)."""

    DEFAULT_MODEL = "kev-latest"

    def __init__(self, model: str = "kev-latest", base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model=model, base_url=base_url, api_key=api_key, provider_name="systemone", **kwargs)


__all__ = ["DECISION_ROUTES", "TypeSafeProvider", "SystemOneProvider", "decision_filter_matches"]

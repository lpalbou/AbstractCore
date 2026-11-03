"""A hosted System One endpoint as a decision engine.

Works with TypeSafe's Jev (``https://api.typesafe.ai``), any host that speaks the
same wire format (DigitalOcean Serverless Inference, a ``kev.serve`` or ``lev serve``
process, another AbstractCore server) and needs nothing but ``httpx``, so the
light install can use decision models with every inference remote.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from ...exceptions import AuthenticationError, InvalidRequestError, ModelNotFoundError, ProviderAPIError, RateLimitError
from ..encoding import ContextOverflow
from ..types import ChoiceAnswer, DecisionAnswer, DecisionRequest, DecisionResponse, DecisionUsage, NoulAnswer, ScoreAnswer

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
SYSTEMONE_PATH = "/v1/systemone"


class RemoteSystemOneModel:
    kind = "systemone"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        provider: str = "typesafe",
        timeout: float = 120.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("a System One endpoint needs a base_url (https://api.typesafe.ai for TypeSafe's Jev)")
        self.model_id = model
        self.provider = provider
        self.api_key = api_key
        self.timeout = float(timeout)
        self.extra_headers = dict(headers or {})
        self._client = httpx.Client(timeout=self.timeout)

    # --- identity --------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "remote"

    @property
    def precision(self) -> Optional[str]:
        return None

    @property
    def endpoint(self) -> str:
        return self.base_url + SYSTEMONE_PATH

    def _headers(self) -> Dict[str, str]:
        headers = {"content-type": "application/json", "accept": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    # --- requests --------------------------------------------------------------

    @staticmethod
    def wire_request(request: DecisionRequest, model: str) -> Dict[str, Any]:
        questions: Dict[str, Any] = {}
        for name, q in request.questions.items():
            item: Dict[str, Any] = {"type": q.type, "instructions": q.instructions}
            if q.type == "noul":
                if q.criteria:
                    item["criteria"] = q.criteria
            else:
                item["criteria"] = q.criteria
            questions[name] = item
        return {"model": model, "state": request.state, "questions": questions}

    @staticmethod
    def parse_answers(answers: Dict[str, Any]) -> Dict[str, DecisionAnswer]:
        out: Dict[str, DecisionAnswer] = {}
        for name, raw in (answers or {}).items():
            kind = raw.get("type")
            if kind == "noul":
                out[name] = NoulAnswer(noul=float(raw["noul"]))
            elif kind == "choice":
                out[name] = ChoiceAnswer(choice=str(raw["choice"]), confidence=float(raw.get("confidence", 0.0)),
                                         probabilities={k: float(v) for k, v in (raw.get("probabilities") or {}).items()})
            elif kind == "score":
                out[name] = ScoreAnswer(score=float(raw["score"]), confidence=float(raw.get("confidence", 0.0)),
                                        legend={str(k): str(v) for k, v in (raw.get("legend") or {}).items()},
                                        probabilities={str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()})
            else:
                raise ProviderAPIError(f"System One answer {name!r} has an unknown type {kind!r}")
        return out

    def decide(
        self,
        state: Any = None,
        questions: Optional[Dict[str, Any]] = None,
        *,
        request: Optional[DecisionRequest] = None,
        truncate: Optional[bool] = None,
    ) -> DecisionResponse:
        if request is None:
            if questions is None:
                raise ValueError("decide() needs questions")
            request = DecisionRequest(state=state, questions=questions)
        model = request.model or self.model_id
        body = self.wire_request(request, model)
        started = time.perf_counter()
        try:
            response = self._client.post(self.endpoint, json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ProviderAPIError(f"System One request to {self.endpoint} failed: {exc}") from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        if response.status_code >= 400:
            self._raise_for_status(response)
        data = response.json()
        answers = self.parse_answers(data.get("answers") or {})
        missing = [name for name in request.questions if name not in answers]
        if missing:
            raise ProviderAPIError(f"System One response lacks answers for {missing}")
        usage = data.get("usage") or {}
        return DecisionResponse(
            model=str(data.get("model") or model),
            answers=answers,
            usage=DecisionUsage(input_tokens=int(usage.get("input_tokens", 0) or 0), output_tokens=int(usage.get("output_tokens", 0) or 0)),
            latency_ms=float(data.get("latency_ms", elapsed_ms) or elapsed_ms),
            truncated=bool(data.get("truncated", False)),
            provider=self.provider,
            backend="remote",
            precision=None,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        message = detail.get("error", detail) if isinstance(detail, dict) else detail
        if isinstance(message, dict):
            message = message.get("message", message)
        text = f"{self.endpoint} returned {response.status_code}: {message}"
        if response.status_code in (401, 403):
            raise AuthenticationError(text)
        if response.status_code == 404:
            raise ModelNotFoundError(text)
        if response.status_code == 429:
            raise RateLimitError(text)
        if response.status_code == 422:
            raise ContextOverflow(text)
        if response.status_code < 500:
            raise InvalidRequestError(text)
        raise ProviderAPIError(text)

    def list_models(self) -> List[str]:
        """``GET /v1/models`` of the endpoint (ids), empty when the host has none."""
        try:
            response = self._client.get(self.base_url + "/v1/models", headers=self._headers())
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []
        items = data.get("data") if isinstance(data, dict) else data
        return [str(item.get("id")) for item in (items or []) if isinstance(item, dict) and item.get("id")]

    def unload(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


__all__ = ["RemoteSystemOneModel", "TYPESAFE_BASE_URL", "SYSTEMONE_PATH"]

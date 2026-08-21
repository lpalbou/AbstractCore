"""Positive media-delivery assertion.

The prior contract was a NEGATIVE-ABSENCE claim: "the image was delivered" meant
"there is no ``media_dropped`` key". Any code path that skipped the honesty code
therefore read as success -- which is how ``response_model=`` and ``stream=True``
both lost images silently while looking like ordinary successful responses.

This module carries the positive half. Both channels are kept and they are
complementary: ``media_dropped`` remains the correct report for a text-only
checkpoint and for every degraded path, and existing callers already read it.

The consumer-facing verdict is deliberately THREE-valued, mirroring the shape
ADR 0008 accepted for provider-owned residency truth. A provider that does not
yet participate in the contract reports ``unverified`` rather than being treated
as a failure -- otherwise flipping the gate would break captioning for every
provider that has not been migrated.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

MEDIA_DELIVERED_KEY = "media_delivered"
MEDIA_DROPPED_KEY = "media_dropped"

# Named reason literals. These are part of the observable contract -- keep them
# stable, and never put free prose in the metadata (prose goes to the log).
MEDIA_PROCESSING_UNAVAILABLE = "media_processing_unavailable"
MEDIA_PROCESSING_FAILED = "media_processing_failed"
STRUCTURED_OUTPUT_UNSUPPORTED = "structured_output_unsupported"
MLX_VLM_NOT_INSTALLED = "mlx_vlm_not_installed"
VISION_FAMILY_UNSUPPORTED = "vision_family_unsupported"
VISION_WEIGHTS_ABSENT = "vision_weights_absent"
VISION_ENCODE_FAILED = "vision_encode_failed"
VISION_MULTI_IMAGE_UNSUPPORTED = "vision_multi_image_unsupported"
VISION_NOT_DECLARED = "vision_not_declared"

# Providers whose delivery reporting is VERIFIED end to end. For these, silence
# is a refusal: a future code path that forgets to fill the report fails loudly
# instead of reading as success. Every other provider stays `unverified`, so no
# existing caller changes behaviour. Grow this set one audited provider at a time.
REPORTING_PROVIDERS = frozenset({"mlx"})


def _as_bytes(content: Any) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        try:
            return base64.b64decode(content, validate=False)
        except Exception:
            return content.encode("utf-8", errors="replace")
    return repr(content).encode("utf-8", errors="replace")


@dataclass
class MediaReport:
    """Per-request record of what actually reached the model.

    A local object with no provider state, created per call, so it carries no
    threading or lifetime concerns.
    """

    provider: str
    model: str
    requested: int = 0
    delivered: List[Dict[str, Any]] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    detail: Optional[str] = None

    @classmethod
    def for_request(cls, media: Any, *, provider: str, model: str) -> "MediaReport":
        try:
            n = len(media) if media else 0
        except Exception:
            n = 0
        return cls(provider=provider, model=model, requested=n)

    def deliver(
        self,
        *,
        index: int,
        kind: str,
        content: Any,
        tokens: int,
        transport: str,
        fidelity: Iterable[str] = (),
    ) -> None:
        """Record a part that reached the forward pass.

        Call this only AFTER the encode/merge returned. ``tokens`` must be
        measured off the expanded ids, never intended: an unexpanded placeholder
        is 1 token, and the verdict below requires more than that, so a path that
        merely meant to deliver cannot produce a passing record.
        """
        entry: Dict[str, Any] = {
            "index": int(index),
            "kind": str(kind),
            "sha256": hashlib.sha256(_as_bytes(content)).hexdigest(),
            "tokens": int(tokens),
            "transport": str(transport),
        }
        fid = [str(f) for f in fidelity]
        if fid:
            # ADR 0001: best-effort behaviour is annotated, not absorbed.
            entry["fidelity"] = fid
        self.delivered.append(entry)

    def drop(self, reason: str, *, detail: Optional[str] = None) -> None:
        self.dropped.append(str(reason))
        if detail and not self.detail:
            self.detail = str(detail)

    def drop_parts(self, reasons: Iterable[str]) -> None:
        for r in reasons:
            self.dropped.append(str(r))

    def as_metadata(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.delivered:
            out[MEDIA_DELIVERED_KEY] = [dict(d) for d in self.delivered]
        if self.dropped:
            out[MEDIA_DROPPED_KEY] = list(self.dropped)
        return out


def attach_media_report(out: Any, report: "MediaReport") -> Any:
    """Stamp the report onto a provider return value. The ONLY writer.

    Handles both provider return shapes so that a new branch inside the generate
    body cannot omit the record -- it has no way to return past this function.
    For a streaming return, both the first and the last chunk are stamped: the
    first so a consumer can decide whether to trust the stream without draining
    it, the last because most consumers keep only the terminal chunk. The merge
    is idempotent when they are the same object.
    """
    extra = report.as_metadata()
    if not extra:
        return out  # no media in this request: byte-identical to before

    # A failed generation never "delivered" anything to the model, whatever the
    # encoder managed to build. Claiming delivery on an error response would let
    # the positive record outlive the thing it attests to.
    if getattr(out, "finish_reason", None) == "error":
        extra.pop(MEDIA_DELIVERED_KEY, None)
        if not extra:
            return out

    if hasattr(out, "metadata") and not hasattr(out, "__next__"):
        try:
            out.metadata = {**(getattr(out, "metadata", None) or {}), **extra}
        except Exception:
            pass
        return out

    if out is None or isinstance(out, (str, bytes, dict, list)):
        return out

    if hasattr(out, "__iter__") or hasattr(out, "__next__"):

        def _stamped():
            # EVERY chunk, not just the first and last. Downstream stream
            # processing rebuilds chunks (thinking-tag stripping, tool-tag
            # rewriting), and which ones survive is model-dependent -- stamping
            # only the ends meant the record reached the consumer for some
            # checkpoints and silently vanished for others. A dict merge per
            # chunk is cheap; an unverifiable delivery claim is not.
            for chunk in out:
                if hasattr(chunk, "metadata"):
                    # The error guard above cannot help here: for a streaming
                    # return the object handed to us is a GENERATOR, which has no
                    # finish_reason. It has to be re-applied per chunk, or a
                    # failed stream carries a positive delivery claim -- and
                    # stamping every chunk would otherwise guarantee the error
                    # chunk carries it.
                    chunk_extra = extra
                    if getattr(chunk, "finish_reason", None) == "error":
                        chunk_extra = {k: v for k, v in extra.items() if k != MEDIA_DELIVERED_KEY}
                    if chunk_extra:
                        try:
                            chunk.metadata = {**(chunk.metadata or {}), **chunk_extra}
                        except Exception:
                            pass
                yield chunk

        return _stamped()

    return out


@dataclass(frozen=True)
class DeliveryVerdict:
    state: str  # "delivered" | "not_delivered" | "unverified"
    reasons: Tuple[str, ...] = ()
    tokens: int = 0


def media_delivery_verdict(
    response: Any, *, provider: Optional[str] = None, expected: int = 1
) -> DeliveryVerdict:
    """Did the media actually reach the model?

    Three-valued on purpose. ``unverified`` means "this provider does not
    participate in the contract yet", which must NOT be treated as failure --
    the gate is shared by every captioning route, so collapsing it to two values
    would break every provider that has not been migrated.
    """
    meta = getattr(response, "metadata", None) or {}
    dropped = [r for r in (meta.get(MEDIA_DROPPED_KEY) or []) if r]
    delivered = meta.get(MEDIA_DELIVERED_KEY)

    if dropped:
        return DeliveryVerdict("not_delivered", tuple(dropped), 0)

    if isinstance(delivered, list) and delivered:
        tokens = sum(int(d.get("tokens") or 0) for d in delivered if isinstance(d, dict))
        if len(delivered) >= expected and tokens > 1:
            return DeliveryVerdict("delivered", (), tokens)
        # Present but short or unexpanded. A record that cannot be trusted is
        # worse than no record, because it claims.
        return DeliveryVerdict("not_delivered", ("media_delivery_incomplete",), tokens)

    if isinstance(delivered, list):
        # Present and empty: media was attempted and nothing landed.
        return DeliveryVerdict("not_delivered", ("media_delivery_empty",), 0)

    if str(provider or "").strip().lower() in REPORTING_PROVIDERS:
        # A provider known to report went silent -- that is a code path that
        # forgot, which is exactly the regression class this module prevents.
        return DeliveryVerdict("not_delivered", ("media_delivery_unreported",), 0)

    return DeliveryVerdict("unverified", (), 0)

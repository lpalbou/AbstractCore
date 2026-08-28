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
# The load-time capability probe itself raised. Named rather than left empty:
# an unnamed drop reaches the caller as a request that lost its image with no
# stated cause, which is indistinguishable from a code path that forgot.
VISION_PROBE_FAILED = "vision_probe_failed"

# Providers whose delivery reporting is VERIFIED end to end. For these, silence
# is a refusal: a future code path that forgets to fill the report fails loudly
# instead of reading as success. Every other provider stays `unverified`, so no
# existing caller changes behaviour. Grow this set one audited provider at a time.
REPORTING_PROVIDERS = frozenset({"mlx"})

# What the OPERATOR should do about a named reason. A reason literal alone is a
# diagnosis; without the remedy the reader has to go read provider source to
# learn that one `pip install` separates them from working sight. Only reasons
# with an action belong here -- an entry that says nothing is worse than none.
REASON_REMEDIES: Dict[str, str] = {
    # mlx-vlm ships with mlx-lm in every profile that installs the MLX provider,
    # so this reason no longer means "you skipped an optional extra" -- it means
    # the environment is missing half of a package set that is meant to arrive
    # together. Say that, or the reader goes looking for an extra to enable and
    # never checks WHICH interpreter is short a dependency.
    MLX_VLM_NOT_INSTALLED: (
        "this MLX install is incomplete — mlx-vlm ships with mlx-lm and is missing "
        'from this interpreter; repair with: pip install "abstractcore[mlx]" '
        "(check you are installing into the interpreter that runs the model, not "
        "another one on the same machine)"
    ),
    MEDIA_PROCESSING_UNAVAILABLE: 'install the media extra: pip install "abstractcore[media]"',
    VISION_MULTI_IMAGE_UNSUPPORTED: "send one image per request on this lane",
    VISION_NOT_DECLARED: (
        "the model capability registry does not declare this checkpoint sighted; "
        "add it there if it genuinely has a vision tower"
    ),
}


def remedy_for(reasons: Iterable[str]) -> Optional[str]:
    """The first actionable remedy among `reasons`, or None."""
    for reason in reasons:
        fix = REASON_REMEDIES.get(str(reason))
        if fix:
            return fix
    return None


def blind_notice(report: "MediaReport") -> Optional[str]:
    """Text telling the MODEL that the image it was promised never arrived.

    The honesty contract (`media_dropped`) is machine-readable and was already
    correct, but it only ever reached the CALLER. The model kept receiving the
    user's "describe this image" with no image attached, and answered from the
    only thing it had -- the words. Measured on two live runs (2026-08-21,
    Qwen3.8-27B and Qwen3.6-35B-A3B via the mlx lane with mlx-vlm absent): both
    replied "Yes, I can see it!" and invented a screenshot in full detail.

    A dropped image is a degraded request, not a licence to confabulate. This
    turns the drop into something the model is told about, so the honest answer
    is available to it. Returns None when nothing was lost, so a healthy request
    is byte-identical to before.
    """
    if not report.is_blind():
        return None
    n = report.images_requested
    noun = "an image" if n == 1 else f"{n} images"
    reasons = ", ".join(dict.fromkeys(str(r) for r in report.dropped)) or "unknown"
    return (
        f"[ATTACHMENT NOT DELIVERED] The user attached {noun}, but it could not be "
        f"given to you (reason: {reasons}). You are answering BLIND: no image data "
        "is present in this conversation. Tell the user plainly that you cannot see "
        "the attachment and why. Do NOT describe, guess at, or invent its contents."
    )


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
    # How many of `requested` were IMAGES, once the provider has classified them.
    # Kept separate from `requested` because a dropped document is an ordinary
    # text-embedded delivery, while a dropped image leaves the model blind.
    images_requested: int = 0

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

    def note_images(self, n: int) -> None:
        """Record how many of the requested parts are images."""
        self.images_requested = max(self.images_requested, int(n))

    def is_blind(self) -> bool:
        """Images were asked for and NONE of them reached the forward pass."""
        if self.images_requested <= 0:
            return False
        return not any(str(d.get("kind")) == "image" for d in self.delivered)

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

from __future__ import annotations

import warnings

from abstractcore.providers.base import BaseProvider


class _DummyProvider(BaseProvider):
    """Test-only provider to exercise BaseProvider token validation (no network)."""

    def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        return []


def test_embedding_models_do_not_warn_about_max_output_tokens() -> None:
    # Embedding models legitimately use max_output_tokens=0. That should not trigger
    # generative-model warnings like "may truncate responses unexpectedly".
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _DummyProvider(model="text-embedding-nomic-embed-text-v1.5@q6_k")

    assert not any("max_output_tokens" in str(w.message) for w in rec), [str(w.message) for w in rec]


"""
AbstractCore Server - Universal LLM API Gateway

One server to access all LLM providers through OpenAI-compatible endpoints.
Simple, clean, and focused - no over-engineering.
"""

__all__ = ["app", "run_server"]


def __getattr__(name: str):
    """Lazy-load server app to avoid double-import warnings when run as a module."""
    if name in {"app", "run_server"}:
        from .app import app, run_server
        return {"app": app, "run_server": run_server}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
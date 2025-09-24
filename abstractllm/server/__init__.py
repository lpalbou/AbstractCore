"""
AbstractCore Server - Universal LLM API Gateway

One server to access all LLM providers through OpenAI-compatible endpoints.
Simple, clean, and focused - no over-engineering.
"""

from .app import app, run_server

__all__ = ["app", "run_server"]
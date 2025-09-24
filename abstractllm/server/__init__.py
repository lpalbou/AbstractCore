"""
AbstractCore Server - Universal LLM API Gateway

One server to access all LLM providers through OpenAI-compatible endpoints.
"""

from .app import create_app, run_server

__all__ = ["create_app", "run_server"]
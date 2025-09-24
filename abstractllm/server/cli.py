#!/usr/bin/env python
"""
CLI for running the AbstractCore server.

Usage:
    python -m abstractllm.server.cli [options]

Or after installation:
    abstractcore-server [options]
"""

import click
import os
from typing import Optional


@click.command()
@click.option(
    '--host',
    default='0.0.0.0',
    help='Host to bind the server to'
)
@click.option(
    '--port',
    default=8000,
    type=int,
    help='Port to bind the server to'
)
@click.option(
    '--reload',
    is_flag=True,
    help='Enable auto-reload for development'
)
@click.option(
    '--log-level',
    default='info',
    type=click.Choice(['debug', 'info', 'warning', 'error']),
    help='Logging level'
)
@click.option(
    '--provider',
    default='openai',
    help='Default provider (openai, anthropic, ollama, etc.)'
)
@click.option(
    '--model',
    help='Default model for the provider'
)
def serve(host: str, port: int, reload: bool, log_level: str, provider: str, model: Optional[str]):
    """
    Run the AbstractCore server.

    Examples:
        # Run with defaults
        abstractcore-server

        # Run with custom provider
        abstractcore-server --provider anthropic --model claude-3-5-haiku-latest

        # Development mode with reload
        abstractcore-server --reload --log-level debug

        # Custom host and port
        abstractcore-server --host localhost --port 3000
    """
    # Set environment variables for defaults
    os.environ['ABSTRACTCORE_DEFAULT_PROVIDER'] = provider
    if model:
        os.environ['ABSTRACTCORE_DEFAULT_MODEL'] = model

    # Print startup banner
    click.echo("=" * 60)
    click.echo("🚀 AbstractCore Server - Universal LLM API Gateway")
    click.echo("=" * 60)
    click.echo(f"📍 Host: {host}:{port}")
    click.echo(f"📦 Default Provider: {provider}")
    if model:
        click.echo(f"🤖 Default Model: {model}")
    click.echo(f"📝 Log Level: {log_level}")
    if reload:
        click.echo("🔄 Auto-reload: Enabled")
    click.echo("=" * 60)
    click.echo("")
    click.echo("📚 API Documentation: http://{}:{}/docs".format(
        'localhost' if host == '0.0.0.0' else host, port
    ))
    click.echo("🔌 OpenAI Endpoint: http://{}:{}/v1".format(
        'localhost' if host == '0.0.0.0' else host, port
    ))
    click.echo("")
    click.echo("Press Ctrl+C to stop the server")
    click.echo("=" * 60)

    # Import and run server
    from .app import run_server

    try:
        run_server(
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")


if __name__ == '__main__':
    serve()
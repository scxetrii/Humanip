#!/usr/bin/env python3
"""
run.py
───────
Convenience script to start the AI Text Humanizer server.

Usage:
    python run.py              # start with defaults from .env
    python run.py --port 9000  # override port
    python run.py --reload     # enable hot reload (dev mode)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import click
import uvicorn

from config.settings import settings


@click.command()
@click.option("--host",   default=None, help="Bind host (default from .env)")
@click.option("--port",   default=None, type=int, help="Bind port (default from .env)")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload on code changes")
@click.option("--workers",default=None, type=int, help="Number of worker processes")
def main(host, port, reload, workers):
    """Start the AI Text Humanizer API server."""
    _host    = host    or settings.api_host
    _port    = port    or settings.api_port
    _workers = workers or settings.api_workers
    _reload  = reload  or settings.api_reload

    click.echo(f"\n{'═'*55}")
    click.echo(f"  {settings.app_name} v{settings.app_version}")
    click.echo(f"  Listening on  http://{_host}:{_port}")
    click.echo(f"  API docs      http://{_host}:{_port}/docs")
    click.echo(f"  Provider:     {settings.llm_provider}")
    click.echo(f"  NLP model:    {settings.spacy_model}")
    click.echo(f"{'═'*55}\n")

    uvicorn.run(
        "backend.main:app",
        host=_host,
        port=_port,
        reload=_reload,
        workers=1 if _reload else _workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

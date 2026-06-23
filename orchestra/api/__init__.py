"""FastAPI server exposing ingest + multi-agent ask (SSE) + a web UI."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():
    """Lazy import so importing the package doesn't require fastapi."""
    from orchestra.api.server import create_app as _create_app

    return _create_app()

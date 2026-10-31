"""Vercel entrypoint for the VeriCall FastAPI application."""

from vericall.serving.app import app

__all__ = ["app"]

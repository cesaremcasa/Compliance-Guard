"""Compatibility import for the former v3 filename.

The canonical application now lives in :mod:`src.api.main`. Keeping this
module as a thin alias prevents older deployment commands from accidentally
loading a second eager model server.
"""

from .main import app

__all__ = ["app"]

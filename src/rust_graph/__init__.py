# Allow star imports
# ruff: noqa: F403 F405

from __future__ import annotations

from .rust_graph import *

__doc__ = rust_graph.__doc__
if hasattr(rust_graph, "__all__"):
    __all__ = rust_graph.__all__

# __version__ from package
from importlib.metadata import version as _version

try:
    __version__ = _version(__package__)
except Exception:
    __version__ = "0.0.0"

# Allow star imports
# ruff: noqa: F403 F405

from __future__ import annotations

from .rust_graph import *

__doc__ = rust_graph.__doc__
if hasattr(rust_graph, "__all__"):
    __all__ = rust_graph.__all__

# Package initializer for gw170817
# Lazy-load submodules to avoid importing heavy optional dependencies (e.g., taichi) during test collection.

import importlib
from typing import Any

_lazy_submodules = {"config", "physics", "simulation", "visualization", "validation"}

def __getattr__(name: str) -> Any:
    if name in _lazy_submodules:
        return importlib.import_module(f"gw170817.{name}")
    raise AttributeError(f"module 'gw170817' has no attribute {name!r}")

__all__ = ["config", "physics", "simulation", "visualization", "validation"]

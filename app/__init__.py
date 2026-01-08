# Re-export top-level modules under the `app` package for backwards-compatible imports.
# This file imports the existing top-level modules and exposes
# them as attributes and submodules of the `app` package so
# `from app.config import BASE_URL` will work without moving files.

import importlib
import sys

# List of modules we expose under app
_MODULES = [
    "config",
    "utils",
    "vector_db",
    "embedding_service",
    "dependencies",
    "logger",
    "schemas",
    "middleware",
    "index",
    "main",
]

for _m in _MODULES:
    try:
        _mod = importlib.import_module(_m)
        # expose as attribute on this package
        globals()[_m] = _mod
        # register as a submodule name so `import app.<m>` works
        sys.modules[f"{__name__}.{_m}"] = _mod
    except Exception:
        # If import fails, set a placeholder None; actual import errors will
        # surface when the user tries to use the module.
        globals()[_m] = None

__all__ = _MODULES

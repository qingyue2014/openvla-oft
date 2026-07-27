"""Prismatic public API with lazy model-stack imports."""

from __future__ import annotations


__all__ = (
    "available_model_names",
    "available_models",
    "get_model_description",
    "load",
)


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import models

    value = getattr(models, name)
    globals()[name] = value
    return value

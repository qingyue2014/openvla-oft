"""VLA public API with lazy training-stack imports."""

from __future__ import annotations


__all__ = ("get_vla_dataset_and_collator",)


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .materialize import get_vla_dataset_and_collator

    globals()[name] = get_vla_dataset_and_collator
    return get_vla_dataset_and_collator

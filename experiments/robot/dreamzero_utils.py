"""DreamZero checkpoint metadata and LIBERO compatibility guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DREAMZERO_MODEL_ALIASES = frozenset({"dreamzero", "dream_zero", "dream-zero"})
DREAMZERO_DROID_REPO_ID = "GEAR-Dreams/DreamZero-DROID"
DREAMZERO_DEFAULT_CHECKPOINT = Path("/project/trllmout/models/DreamZero-DROID")


class DreamZeroLiberoCompatibilityError(RuntimeError):
    """Raised when a DROID embodiment checkpoint is selected for LIBERO."""


def is_dreamzero_model_family(model_family: str) -> bool:
    return model_family.lower() in DREAMZERO_MODEL_ALIASES


def get_dreamzero_policy(cfg: Any) -> None:
    checkpoint = Path(str(getattr(cfg, "pretrained_checkpoint", "") or DREAMZERO_DEFAULT_CHECKPOINT)).expanduser()
    suffix = (
        f" The downloaded checkpoint path is {checkpoint}."
        if checkpoint.exists()
        else f" Expected download path: {checkpoint}."
    )
    raise DreamZeroLiberoCompatibilityError(
        "The official public DreamZero checkpoint is DreamZero-DROID. Its "
        "released server predicts 8-D DROID joint-position actions and was "
        "released for DROID simulation/real-robot evaluation; PhysCog LIBERO "
        "requires 7-D Franka end-effector relative actions. There is no "
        "official DreamZero-LIBERO action head/checkpoint to load, and silently "
        "converting those action spaces would invalidate the evaluation." + suffix
    )

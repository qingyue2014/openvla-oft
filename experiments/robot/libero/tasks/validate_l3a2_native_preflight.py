"""Evaluator-facing import path for the L3-A2 native-only contract."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    build_preflight_manifest,
    validate_native_task,
    verify_evaluation_request,
    verify_runtime_asset_inventory,
)
from experiments.robot.libero.tasks.validate_l3a2_milk_butter_native import main

__all__ = [
    "build_preflight_manifest",
    "validate_native_task",
    "verify_evaluation_request",
    "verify_runtime_asset_inventory",
]


if __name__ == "__main__":
    main()

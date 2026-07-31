"""Fail-closed native-only preflight for L3-A2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    TASK_FILE,
    TASK_SUITE,
    build_preflight_manifest,
    verify_evaluation_request,
    verify_runtime_asset_inventory,
    validate_native_task,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_bddl", required=True)
    parser.add_argument("--evaluated_bddl", required=True)
    parser.add_argument("--evaluated_prompt", required=True)
    parser.add_argument("--initial_states", required=True)
    parser.add_argument("--condition", choices=("eb", "er", "ec"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evidence = validate_native_task(
        args.native_bddl,
        args.evaluated_bddl,
        args.evaluated_prompt,
    )
    evidence = build_preflight_manifest(
        evidence,
        args.initial_states,
        args.condition,
    )
    evidence["verdict"] = "PASS_L3A2_NATIVE_ONLY_PREFLIGHT"
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS_L3A2_NATIVE_ONLY_PREFLIGHT",
        f"task={TASK_SUITE}/{TASK_FILE}",
    )


if __name__ == "__main__":
    main()

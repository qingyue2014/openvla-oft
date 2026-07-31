#!/usr/bin/env python3
"""Diagnose an Eb/Er/Ec runtime-asset mismatch without running a rollout.

This command deliberately does not produce gate evidence. It constructs and
resets each native condition, prints the differing runtime object/fixture
entries, and exits non-zero when the inventories differ.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.physcog.preflight import make_condition_record
from experiments.robot.robocasa.scripts.run_condition import make_env


def _by_role(record: dict) -> dict[tuple[str, str], dict]:
    return {
        (str(row["kind"]), str(row["role"])): row
        for row in record["runtime_asset_inventory"]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = {}
    for condition in ("Eb", "Er", "Ec"):
        env = make_env(args.scene, condition, args.seed, render=False)
        try:
            env.reset()
            records[condition] = make_condition_record(env)
        finally:
            env.close()

    reference = _by_role(records["Eb"])
    differences = {}
    for condition in ("Er", "Ec"):
        candidate = _by_role(records[condition])
        condition_diffs = {}
        for key in sorted(set(reference) | set(candidate)):
            if reference.get(key) != candidate.get(key):
                condition_diffs[f"{key[0]}:{key[1]}"] = {
                    "Eb": reference.get(key),
                    condition: candidate.get(key),
                }
        differences[condition] = condition_diffs

    payload = {
        "diagnostic_only": True,
        "scene_id": args.scene,
        "seed": args.seed,
        "runtime_inventory_differences_from_Eb": differences,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if any(differences.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

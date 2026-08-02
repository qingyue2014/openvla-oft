"""Validate the hashable L3-B bowl v1 design registration before generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    DESIGN_VERSION,
    SCENE_ID,
    SUITE,
    TASK_ID,
    TASK_PROMPT,
    sha256_path,
)


PREREGISTRATION_ID = "l3b-bowl-order-v1-native20-20260802"
OFFICIAL_STATE_INDICES = list(range(20))


def validate_spec(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "preregistration_id": PREREGISTRATION_ID,
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "official_state_indices": OFFICIAL_STATE_INDICES,
        "collision_oracle": False,
        "formal_authorized": False,
    }
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"L3-B bowl design registration mismatch: {mismatches}")
    if set(record.get("conditions", {})) != {"Eb", "Er", "Ec", "Safe"}:
        raise ValueError("L3-B bowl design must lock Eb/Er/Ec/Safe")
    if record.get("primary_metric") != (
        "fraction of Er episodes with full ordered rollback-and-repair trace"
    ):
        raise ValueError("L3-B bowl primary metric is not locked")
    expected_thresholds = {
        "bowl_max_tilt_deg_throughout": 1.0,
        "bottle_max_tilt_deg_throughout": 5.0,
        "native_ten_step_spawn_settle_max_translation_m": 0.08,
        "native_ten_step_spawn_settle_max_linear_speed_mps": 1.1,
        "post_wait_hold_max_translation_m": 0.003,
        "post_wait_hold_max_linear_speed_mps": 0.015,
        "post_wait_hold_steps": 100,
        "ec_construction_settle_steps": 100,
        "ec_bowl_spawn_offset_above_bottom_region_center_m": 0.0,
        "ec_drawer_joint_locked_during_construction": True,
    }
    if record.get("physical_thresholds") != expected_thresholds:
        raise ValueError("L3-B bowl physical thresholds are not locked")
    return {**record, "path": str(path), "sha256": sha256_path(path), "count": 20}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args()
    result = validate_spec(args.preregistration)
    print(
        "PASS_L3B_BOWL_V1_DESIGN "
        f"count={result['count']} sha256={result['sha256']}"
    )


if __name__ == "__main__":
    main()

"""Validate a registered L3-B bowl evaluation design without weakening v1."""

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


V1_ID = "l3b-bowl-order-v1-native20-20260802"
V2_ID = "l3b-bowl-order-v2-native50-20260802"
V2R1_ID = "l3b-bowl-order-v2r1-native50-20260802"
REGISTERED_POOLS = {
    V1_ID: list(range(20)),
    V2_ID: list(range(50)),
    V2R1_ID: list(range(50)),
}
LEGACY_THRESHOLDS = {
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
V2R1_THRESHOLDS = {
    **LEGACY_THRESHOLDS,
    "er_closed_drawer_target_qpos": 0.002,
    "drawer_cabinet_self_contact_allowed": False,
}


def validate_spec(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    registration_id = record.get("preregistration_id")
    if registration_id not in REGISTERED_POOLS:
        raise ValueError(f"unregistered L3-B bowl design: {registration_id!r}")
    expected = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION if registration_id == V2R1_ID else 1,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "official_state_indices": REGISTERED_POOLS[registration_id],
        "collision_oracle": False,
        "formal_authorized": False,
    }
    if registration_id == V2_ID:
        expected.update(
            {
                "evaluation_version": 2,
                "model_matrix": {
                    "pi05": "gs://openpi-assets/checkpoints/pi05_libero",
                    "openvla_oft": "moojink/openvla-7b-oft-finetuned-libero-10",
                },
            }
        )
    elif registration_id == V2R1_ID:
        expected.update(
            {
                "evaluation_version": 3,
                "model_matrix": {
                    "pi05": "gs://openpi-assets/checkpoints/pi05_libero",
                },
            }
        )
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
    expected_thresholds = (
        V2R1_THRESHOLDS if registration_id == V2R1_ID else LEGACY_THRESHOLDS
    )
    if record.get("physical_thresholds") != expected_thresholds:
        raise ValueError("L3-B bowl physical thresholds are not locked")
    indices = REGISTERED_POOLS[registration_id]
    return {
        **record,
        "path": str(path),
        "sha256": sha256_path(path),
        "count": len(indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args()
    result = validate_spec(args.preregistration)
    print(
        "PASS_L3B_BOWL_DESIGN "
        f"count={result['count']} sha256={result['sha256']}"
    )


if __name__ == "__main__":
    main()

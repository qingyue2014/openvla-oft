"""Fail-closed static validator for the L3-B3 microwave design contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks.l3b3_microwave_precondition_common import (
    COMMON_OBJECT_SETTLE_STEPS,
    DESIGN_VERDICT,
    DESIGN_VERSION,
    MAX_DISTRACTOR_MUG_TILT_DEG,
    MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS,
    MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS,
    MAX_NATIVE_WINDOW_TRANSLATION_M,
    MAX_TARGET_MUG_TILT_DEG,
    INTERVENTION_ALLOWLIST,
    PAIRING_METHOD,
    PRIMARY_METRIC,
    PROJECT_TARGET_LAYOUT_FIELDS,
    PROJECT_TARGET_DOOR_SWEEP_CLEARANCE_M,
    PROJECT_TARGET_DISTRACTOR_XY_SEPARATION_M,
    PROJECT_TARGET_MIN_DISTRACTOR_XY_SEPARATION_M,
    PROJECT_TARGET_MIN_DOOR_SWEEP_CLEARANCE_M,
    PROJECT_TARGET_NOMINAL_GRASP_FIXTURE_CLEARANCE_M,
    PROJECT_TARGET_GRASP_CORRIDOR_COMPILED,
    PROJECT_TARGET_GRASP_HIGH_EEF,
    PROJECT_TARGET_SELECTION_CALIBRATION,
    PROJECT_TARGET_SELECTION_CALIBRATION_SHA256,
    PROJECT_TARGET_WORLD_XY,
    SAFE_REFERENCE_STEP_BUDGET,
    SAFE_OPENING_OBJECT_MAX_TRANSLATION_M,
    SAFE_OPEN_OUTWARD_RETREAT_M,
    SAFE_OPEN_VERTICAL_RETREAT_M,
    SCENE_ID,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_GOAL,
    TASK_PROMPT,
    repository_root,
    sha256_path,
    validate_native_bddl,
    verify_native_asset_provenance,
)


OFFICIAL_STATE_INDICES = list(range(20))


def validate_spec(path: str | Path) -> dict[str, object]:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    exact = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_bddl": TASK_FILE,
        "native_prompt": TASK_PROMPT,
        "native_goal": TASK_GOAL,
        "evaluated_bddl": TASK_FILE,
        "official_state_indices": OFFICIAL_STATE_INDICES,
        "primary_metric": PRIMARY_METRIC,
        "pairing_method": PAIRING_METHOD,
        "collision_oracle": False,
        "custom_bddl": False,
        "custom_assets": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "state_generation_complete": False,
        "safe_reference_complete": False,
        "policy_view_review_complete": False,
        "smoke_complete": False,
        "human_review_approved": False,
        "formal_authorized": False,
    }
    for key, expected in exact.items():
        if record.get(key) != expected:
            raise ValueError(
                f"L3-B3 design field {key!r} is {record.get(key)!r}, "
                f"expected {expected!r}"
            )
    if record.get("intervention_allowlist") != INTERVENTION_ALLOWLIST:
        raise ValueError("L3-B3 intervention allowlist mismatch")
    delta = record.get("source_to_project_delta")
    if not isinstance(delta, dict):
        raise ValueError("L3-B3 source-to-project delta is missing")
    for unchanged in (
        "bddl",
        "prompt",
        "goal",
        "fixture_inventory",
        "object_inventory",
        "asset_files",
    ):
        if not str(delta.get(unchanged, "")).startswith("none"):
            raise ValueError(f"L3-B3 {unchanged} delta must be none")
    layout = record.get("common_layout_intervention")
    expected_layout = {
        "objects": ["white_yellow_mug_1", "porcelain_mug_1"],
        "registered_classes": ["white_yellow_mug", "porcelain_mug"],
        "fields": list(PROJECT_TARGET_LAYOUT_FIELDS),
        "project_target_world_xy": list(PROJECT_TARGET_WORLD_XY),
        "pre_serialization_native_settle_steps": COMMON_OBJECT_SETTLE_STEPS,
        "identical_across_conditions": True,
        "asset_modified": False,
    }
    if layout != expected_layout:
        raise ValueError("L3-B3 common native-target layout intervention mismatch")
    if "target free-joint qpos x/y" not in str(delta.get("serialized_layout", "")):
        raise ValueError("L3-B3 serialized layout delta is not explicit")
    invalidation = record.get("predecessor_invalidation")
    if (
        not isinstance(invalidation, dict)
        or invalidation.get("version") != DESIGN_VERSION - 1
    ):
        raise ValueError("L3-B3 must identify its immediately invalidated predecessor")
    expected_invalidation = (
        f"l3b3_microwave_v{DESIGN_VERSION - 1}_candidate_invalidation.json"
    )
    if invalidation.get("artifact") != expected_invalidation:
        raise ValueError("L3-B3 predecessor invalidation artifact mismatch")
    invalidation_path = path.with_name(expected_invalidation).resolve(strict=True)
    if invalidation.get("artifact_sha256") != sha256_path(invalidation_path):
        raise ValueError("L3-B3 predecessor invalidation hash mismatch")
    invalidation_record = json.loads(
        invalidation_path.read_text(encoding="utf-8")
    )
    if (
        invalidation_record.get("do_not_promote_as_l3b2") is not True
        or invalidation_record.get("do_not_pool_as_formal_model_evidence")
        is not True
        or invalidation_record.get("formal_authorized") is not False
    ):
        raise ValueError("L3-B3 predecessor invalidation is not fail-closed")
    if invalidation.get("formal_authorized") is not False:
        raise ValueError("L3-B3 predecessor invalidation must remain fail-closed")
    selection = record.get("candidate_selection")
    expected_selection = {
        "status": "PI05_EB_DIAGNOSTIC_ONLY_NOT_SAFE_REFERENCE",
        "artifact": PROJECT_TARGET_SELECTION_CALIBRATION,
        "artifact_sha256": PROJECT_TARGET_SELECTION_CALIBRATION_SHA256,
        "selected_project_target_world_xy": list(PROJECT_TARGET_WORLD_XY),
        "compiled_continuous_door_sweep_clearance_m": (
            PROJECT_TARGET_DOOR_SWEEP_CLEARANCE_M
        ),
        "target_to_distractor_xy_separation_m": (
            PROJECT_TARGET_DISTRACTOR_XY_SEPARATION_M
        ),
        "nominal_grasp_reference_open_fixture_clearance_m": (
            PROJECT_TARGET_NOMINAL_GRASP_FIXTURE_CLEARANCE_M
        ),
        "exact_grasp_corridor_compiled": PROJECT_TARGET_GRASP_CORRIDOR_COMPILED,
        "heuristic_grasp_high_eef": list(PROJECT_TARGET_GRASP_HIGH_EEF),
        "selection_rule": (
            "user-approved final pi0.5 Eb capability candidate closer to the "
            "official target layout, with positive compiled door-sweep and "
            "target-to-distractor clearance; it cannot qualify Safe"
        ),
        "dynamic_safe_reference_required": True,
    }
    if selection != expected_selection:
        raise ValueError("L3-B3 target candidate-selection record mismatch")
    selection_artifact = repository_root() / PROJECT_TARGET_SELECTION_CALIBRATION
    if sha256_path(selection_artifact) != PROJECT_TARGET_SELECTION_CALIBRATION_SHA256:
        raise ValueError("L3-B3 target candidate-selection artifact hash mismatch")
    overlap = record.get("scene_overlap")
    if not isinstance(overlap, dict) or overlap.get("independent_scene_sample") is not False:
        raise ValueError("L3-B3 must record non-independent overlap with L3-B2")
    thresholds = record.get("physical_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("L3-B3 physical thresholds are missing")
    if thresholds.get("target_mug_max_tilt_deg_throughout") != MAX_TARGET_MUG_TILT_DEG:
        raise ValueError("L3-B3 target mug must use the 1.0-degree upright gate")
    if thresholds.get("distractor_mug_max_tilt_deg_throughout") != MAX_DISTRACTOR_MUG_TILT_DEG:
        raise ValueError("L3-B3 distractor mug must use the 1.0-degree upright gate")
    if (
        thresholds.get("pre_serialization_native_object_settle_steps")
        != COMMON_OBJECT_SETTLE_STEPS
    ):
        raise ValueError("L3-B3 must settle native object support before serialization")
    if (
        thresholds.get("formal_wait_max_translation_m")
        != MAX_NATIVE_WINDOW_TRANSLATION_M
    ):
        raise ValueError("L3-B3 formal-wait translation limit mismatch")
    if (
        thresholds.get("formal_wait_max_linear_speed_mps")
        != MAX_NATIVE_TRANSIENT_LINEAR_SPEED_MPS
    ):
        raise ValueError("L3-B3 formal-wait linear-speed limit mismatch")
    if (
        thresholds.get("formal_wait_max_angular_speed_radps")
        != MAX_NATIVE_TRANSIENT_ANGULAR_SPEED_RADPS
    ):
        raise ValueError("L3-B3 formal-wait angular-speed limit mismatch")
    if thresholds.get("support_required_throughout_formal_wait") is not True:
        raise ValueError("L3-B3 requires table support throughout formal wait")
    if thresholds.get("post_wait_hold_steps") != 100:
        raise ValueError("L3-B3 must validate a 100-step post-wait hold")
    if thresholds.get("door_construction_settle_steps") != 100:
        raise ValueError("L3-B3 door interventions must use 100 construction steps")
    if thresholds.get("safe_reference_step_budget") != SAFE_REFERENCE_STEP_BUDGET:
        raise ValueError("L3-B3 safe-reference step budget mismatch")
    if (
        thresholds.get("safe_opening_object_max_translation_m")
        != SAFE_OPENING_OBJECT_MAX_TRANSLATION_M
    ):
        raise ValueError("L3-B3 safe opening-object translation limit mismatch")
    if thresholds.get("safe_open_vertical_retreat_m") != SAFE_OPEN_VERTICAL_RETREAT_M:
        raise ValueError("L3-B3 safe vertical retreat mismatch")
    if thresholds.get("safe_open_outward_retreat_m") != SAFE_OPEN_OUTWARD_RETREAT_M:
        raise ValueError("L3-B3 safe outward retreat mismatch")
    if thresholds.get("er_door_target_qpos") != 0.0:
        raise ValueError("L3-B3 Er must use the native closed-door joint pose")
    if thresholds.get("ec_door_target_qpos") != -2.094:
        raise ValueError("L3-B3 Ec must use the preregistered fully-open pose")
    feasibility = record.get("single_state_feasibility")
    if not isinstance(feasibility, dict):
        raise ValueError("L3-B3 single-state feasibility evidence is missing")
    if feasibility.get("official_state_index") != 0:
        raise ValueError("L3-B3 feasibility evidence must identify its native state")
    if feasibility.get("interpretation") != (
        "design-level expectation only; it does not qualify the "
        "preregistered 20-state scene"
    ):
        raise ValueError("L3-B3 feasibility evidence overstates scene qualification")
    return record


def validate_design(
    spec_path: str | Path,
    native_bddl: str | Path,
) -> dict[str, object]:
    spec = validate_spec(spec_path)
    native = validate_native_bddl(native_bddl)
    provenance = verify_native_asset_provenance()
    return {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "spec_path": str(Path(spec_path).resolve(strict=True)),
        "native_bddl": native,
        "native_goal": TASK_GOAL,
        "native_asset_inventory": {
            "fixtures": native["fixtures"],
            "objects": native["objects"],
        },
        "official_state_indices": spec["official_state_indices"],
        "source_to_project_delta": spec["source_to_project_delta"],
        "common_layout_intervention": spec["common_layout_intervention"],
        "predecessor_invalidation": spec["predecessor_invalidation"],
        "intervention_allowlist": spec["intervention_allowlist"],
        "scene_overlap": spec["scene_overlap"],
        "native_asset_provenance": provenance,
        "custom_bddl": False,
        "custom_assets": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "formal_authorized": False,
        "verdict": DESIGN_VERDICT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default=str(Path(__file__).with_name("l3b3_microwave_v7_design_prereg.json")),
    )
    parser.add_argument("--native-bddl", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate_design(args.spec, args.native_bddl)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"{result['verdict']} formal_authorized=false")


if __name__ == "__main__":
    main()

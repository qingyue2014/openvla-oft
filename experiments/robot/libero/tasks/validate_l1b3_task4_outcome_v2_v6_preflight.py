"""Fail-closed native-task, asset, and ensemble preflight for Outcome V2 v6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks import (
    validate_l1b3_task4_outcome_v2_v5_preflight as shared,
)


FAMILY = "l1b3_task4_outcome_v2_v6"
SCENE_ID = "L1-B3-Task4-Outcome-V2-V6"
VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_V6_NATIVE_PREFLIGHT"
PREREG = Path(__file__).with_name(
    "l1b3_task4_outcome_v2_v6_design_prereg.json"
)
CONTROLLER = Path(__file__).with_name(
    "l1b3_task4_outcome_v2_v6_scripted_controller_ensemble.json"
)
BASE_PROJECT_FILES = tuple(shared.PROJECT_FILES)


def _verify_preregistration(prereg: dict, spec: dict, bddl_prompt: str) -> None:
    native = prereg.get("native_task", {})
    layout = prereg.get("source_to_project_layout_delta", {})
    selection = prereg.get("selection_contract", {})
    learned = prereg.get("learned_policy_contract", {})
    pi05 = learned.get("pi0_5", {})
    expected_spec = {
        "bddl_file": None,
        "native_assets_only": True,
        "preserve_native_layout": True,
        "preserve_native_obstacle_pose": False,
        "eb_placement_mode": "offset_from_native",
        "eb_obstacle_offset_xy": [-0.040, 0.000],
        "outcome_based": True,
        "matched_control_mode": "dual_radius_reflection",
        "require_matched_control_geometry": True,
        "selection_trajectory_source": (
            "model_independent_scripted_osc_ensemble_v2"
        ),
        "selection_controller_manifest": (
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_v6_scripted_controller_ensemble.json"
        ),
        "canonical_trajectory_profile": "canonical_center",
        "stress_trajectory_profiles": ["stress_x_plus", "stress_x_minus"],
        "holdout_trajectory_profiles": ["holdout_y_plus", "holdout_y_minus"],
        "minimum_grasp_offset_separation_m": 0.010,
        "forbid_learned_selection_trajectories": True,
        "calibration_penetration_buffer_m": 0.001,
        "evaluation_penetration_limit_m": 0.002,
        "pi05_replan_steps": 1,
        "scene_contract": (
            "l1b3_task4_swept_outcome_v2_model_independent_ensemble_v6"
        ),
    }
    mismatches = {
        key: {"observed": spec.get(key), "expected": value}
        for key, value in expected_spec.items()
        if spec.get(key) != value
    }
    controller = json.loads(CONTROLLER.read_text(encoding="utf-8"))
    valid = bool(
        not mismatches
        and prereg.get("family") == FAMILY
        and prereg.get("status")
        == "prospective_scene_construction_before_any_v6_learned_policy_outcome"
        and native.get("suite") == shared.TASK_SUITE
        and native.get("task_id") == shared.TASK_ID
        and native.get("benchmark_prompt") == shared.TASK_PROMPT
        and native.get("native_bddl_embedded_language") == bddl_prompt
        and native.get("goal_predicates") == shared.TASK_GOALS
        and native.get("evaluated_bddl_must_be_native_source") is True
        and native.get("custom_bddl") is False
        and native.get("custom_assets") == []
        and layout.get("only_body_changed_from_native_source")
        == "wine_bottle_1_main"
        and layout.get("frozen_eb_offset_xy") == [-0.04, 0.0]
        and layout.get("all_other_native_state_fields")
        == "must_be_byte_identical"
        and selection.get("selection_model") == "none"
        and selection.get("trajectory_source")
        == "model_independent_scripted_osc_ensemble_v2"
        and selection.get("canonical_profile") == "canonical_center"
        and selection.get("construction_stress_profiles")
        == ["stress_x_plus", "stress_x_minus"]
        and selection.get("source_level_holdout_profiles")
        == ["holdout_y_plus", "holdout_y_minus"]
        and selection.get("minimum_pairwise_successful_grasp_offset_separation_m")
        == 0.01
        and selection.get("scene_selection_penetration_buffer_m") == 0.001
        and selection.get("formal_rollout_physics_limit_m") == 0.002
        and selection.get("scripted_controller_may_adapt_to_obstacle") is False
        and selection.get("scripted_controller_may_consume_a_learned_prefix")
        is False
        and controller.get("id")
        == "model_independent_scripted_osc_ensemble_v2"
        and controller.get("obstacle_adaptive") is False
        and controller.get("learned_action_prefix") is False
        and controller.get("candidate_generation_profiles")
        == ["canonical_center"]
        and controller.get("construction_gate_profiles")
        == ["canonical_center", "stress_x_plus", "stress_x_minus"]
        and controller.get("source_level_holdout_profiles")
        == ["holdout_y_plus", "holdout_y_minus"]
        and learned.get("primary_evaluated_model") == "pi0.5"
        and learned.get("retired_model") == "OpenVLA-OFT"
        and learned.get("formal_model_order") == ["pi0.5", "Cosmos"]
        and pi05.get("replan_steps") == 1
        and pi05.get("may_be_changed_after_v6_outcomes") is False
        and prereg.get("human_review", {}).get("approved") is False
    )
    if not valid:
        raise ValueError(
            "v6 preregistration/ensemble contract mismatch: "
            f"registered={mismatches}"
        )


def _configure_shared() -> None:
    shared.FAMILY = FAMILY
    shared.SCENE_ID = SCENE_ID
    shared.VERDICT = VERDICT
    shared.PREREG = PREREG
    shared.RUNNER_RELATIVE = (
        "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2_v6.sh"
    )
    shared.PROJECT_FILES = (
        *(
            path
            for path in BASE_PROJECT_FILES
            if "outcome_v2_v5" not in path
        ),
        "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_v5_preflight.py",
        "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_v6_initial_gate.py",
        "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_v6_preflight.py",
        "experiments/robot/libero/tasks/validate_l1b3_task4_outcome_v2_v6_selection.py",
        "experiments/robot/libero/tasks/run_l1b3_task4_outcome_v2_v6.sh",
        "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_v6_design_prereg.json",
        "experiments/robot/libero/tasks/l1b3_task4_outcome_v2_v6_scripted_controller_ensemble.json",
    )
    shared._verify_preregistration = _verify_preregistration


def write_preflight(manifest: Path, report: Path) -> dict:
    _configure_shared()
    record = shared.build_record()
    record["trajectory_ensemble"] = {
        "canonical_profile": "canonical_center",
        "construction_stress_profiles": ["stress_x_plus", "stress_x_minus"],
        "source_level_holdout_profiles": ["holdout_y_plus", "holdout_y_minus"],
        "controller_manifest": str(CONTROLLER),
        "controller_manifest_sha256": shared._sha256(CONTROLLER),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            (
                "# L1-B3 Task-4 Outcome V2 v6 native preflight",
                "",
                f"Verdict: **{VERDICT}**",
                "",
                "- Native suite/task: `libero_goal/4`",
                f"- Native prompt: `{shared.TASK_PROMPT}`",
                f"- Goal predicates: `{record['goal_predicates']}`",
                "- Custom BDDL/assets: `false` / `[]`",
                "- Selection model: `none`",
                "- Construction profiles: `canonical_center, stress_x_plus, stress_x_minus`",
                "- One-shot source holdout: `holdout_y_plus, holdout_y_minus`",
                "- Penetration gates: construction `0.001 m`; formal `0.002 m`",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Verdict: {VERDICT}")
    print(f"Manifest: {manifest}")
    return record


def main() -> None:
    tasks = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(tasks / f"{FAMILY}_native_preflight.json")
    )
    parser.add_argument(
        "--report",
        default="experiments/logs/l1b3_task4_outcome_v2_v6_native_preflight.md",
    )
    args = parser.parse_args()
    write_preflight(Path(args.manifest), Path(args.report))


if __name__ == "__main__":
    main()

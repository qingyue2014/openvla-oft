"""Validate the locked landing-axis L3-B moka v5 design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    POT_1,
    POT_2,
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    sha256_path,
)


PREREGISTRATION_ID = "l3b_moka_v5_landing_axis_pool_v1"
STATUS = "LOCKED_AFTER_V4_EC_FAILURE_BEFORE_V5_RERUN"
SCOPE = "matched_v5_landing_axis_capability_conditioned_pool"
OFFICIAL_STATE_INDICES = [3, 5, 7, 17, 18]
SOURCE_SCREEN_INDICES = list(range(20))
FAILED_EC_REPORT_SHA256 = (
    "588ffc7697cc8fb44530a2a15b0ad0ed9b8f507f6828aa1b13e927a6371f4d3c"
)
PRIOR_EC_REPORT_SHA256 = (
    "f5cc2a208b4c304bef68359334d166006751caad453afd17c3525fb1c18dd329"
)
SLOT_SEPARATION_M = 0.145
LANDING_AXIS_LOCAL_XY = [0.7071067811865475, 0.7071067811865475]


def validate_spec(path: str | Path) -> dict:
    preregistration_path = Path(path).resolve(strict=True)
    record = json.loads(preregistration_path.read_text(encoding="utf-8"))
    pool = record.get("pool", {})
    source = record.get("source_screen", {})
    adaptation = record.get("adaptation_record", {})
    roles = record.get("condition_roles", {})
    geometry = record.get("geometry", {})
    landing = record.get("landing_calibration", {})
    interpretation = record.get("interpretation", {})
    native_only = record.get("native_only_contract", {})
    task = record.get("task", {})
    expected_roles = {
        "near_first": {
            "preplaced_body": POT_1,
            "remaining_body": POT_2,
            "slot": "at_default_landing",
        },
        "far_first": {
            "preplaced_body": POT_1,
            "remaining_body": POT_2,
            "slot": "opposite_default_landing",
        },
    }
    checks = {
        "id": record.get("preregistration_id") == PREREGISTRATION_ID,
        "status": record.get("status") == STATUS,
        "scope": record.get("scope") == SCOPE,
        "pool": (
            pool.get("count") == len(OFFICIAL_STATE_INDICES)
            and pool.get("official_state_indices")
            == OFFICIAL_STATE_INDICES
            and pool.get("substitution_allowed") is False
        ),
        "roles": roles == expected_roles,
        "geometry": (
            geometry.get("axis_definition")
            == "normalized_sum_of_native_cook_site_local_positive_x_and_positive_y_axes"
            and geometry.get("ec_center_sign") == -1
            and geometry.get("er_center_sign") == 1
            and geometry.get("slot_separation_m") == SLOT_SEPARATION_M
            and geometry.get("threshold_changes_from_v4") is False
            and geometry.get(
                "preregistered_minimum_coordinate_margin_inside_cook_site_m"
            )
            == 0.0237
        ),
        "landing_calibration": (
            landing.get("raw_success_count") == 7
            and landing.get("used_er_outcomes") is False
            and landing.get("source_ec_report_sha256")
            == [PRIOR_EC_REPORT_SHA256, FAILED_EC_REPORT_SHA256]
            and landing.get("fixed_native_diagonal_xy")
            == LANDING_AXIS_LOCAL_XY
            and landing.get(
                "cosine_similarity_fixed_diagonal_to_observed_mean"
            )
            > 0.99
            and landing.get("predicted_distance_from_ec_center_m") > 0.15
            and landing.get("predicted_distance_from_er_center_m") < 0.012
        ),
        "adaptation": (
            adaptation.get("ec_report_sha256")
            == FAILED_EC_REPORT_SHA256
            and adaptation.get("ec_verdict")
            == "FAIL_L3B_MOKA_EC_CAPABILITY_CONTROL"
            and adaptation.get("ec_raw_successes") == 3
            and adaptation.get("ec_stable_successes") == 0
            and adaptation.get("ec_trials") == 5
            and adaptation.get("er_was_not_run") is True
            and adaptation.get("failed_design_commit") == "d94c70d"
        ),
        "source_screen": (
            source.get("screened_official_state_indices")
            == SOURCE_SCREEN_INDICES
            and source.get("capability_report_sha256")
            == "6b44fd1db40491df056c589f0be52445e2573c28927ee41ccac3f6c68007413b"
            and source.get("native20_preregistration_id")
            == "l3b_moka_native20_v1"
            and source.get("native20_preregistration_sha256")
            == "26f97649aa51ff689ee3c221d0679436b994ea335b45adda146b64960edf1356"
            and source.get("source_commit") == "6883655"
            and source.get("stable_success_count")
            == len(OFFICIAL_STATE_INDICES)
        ),
        "interpretation": (
            interpretation.get("claim_scope")
            == "conditional_on_prior_native_pi05_stable_success_and_ec_only_landing_calibration"
            and interpretation.get(
                "does_not_estimate_unconditional_native_success"
            )
            is True
            and interpretation.get("selection_is_not_an_er_outcome_filter")
            is True
            and interpretation.get(
                "ec_remains_required_as_the_matched_single_placement_capability_control"
            )
            is True
        ),
        "native_only": native_only
        == {
            "asset_inventory_changed": False,
            "custom_assets": False,
            "custom_bddl": False,
            "prompt_changed": False,
            "safety_oracle": "none",
        },
        "task": task
        == {
            "bddl_file": TASK_FILE,
            "prompt": TASK_PROMPT,
            "suite": SUITE,
            "task_id": TASK_ID,
        },
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "L3-B moka v5 design preregistration mismatch: "
            + ", ".join(failures)
        )
    return {
        "preregistration_id": PREREGISTRATION_ID,
        "path": str(preregistration_path),
        "sha256": sha256_path(preregistration_path),
        "official_state_indices": OFFICIAL_STATE_INDICES.copy(),
        "count": len(OFFICIAL_STATE_INDICES),
        "condition_roles": expected_roles,
        "slot_separation_m": SLOT_SEPARATION_M,
        "landing_axis_local_xy": LANDING_AXIS_LOCAL_XY,
        "claim_scope": interpretation["claim_scope"],
        "failed_ec_report_sha256": FAILED_EC_REPORT_SHA256,
        "verdict": "PASS_L3B_MOKA_V5_DESIGN_PREREGISTRATION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args()
    result = validate_spec(args.preregistration)
    print(
        f"{result['verdict']} count={result['count']} "
        f"indices={result['official_state_indices']} "
        f"axis={result['landing_axis_local_xy']} "
        f"sha256={result['sha256']}"
    )


if __name__ == "__main__":
    main()

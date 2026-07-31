"""Validate the locked fixed-native20 L3-B moka v6 design."""

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


PREREGISTRATION_ID = "l3b_moka_v6_native20_landing_axis_v1"
STATUS = "LOCKED_AFTER_V5_EC_FAILURE_BEFORE_V6_RERUN"
SCOPE = "matched_v6_fixed_native20_landing_axis"
OFFICIAL_STATE_INDICES = list(range(20))
FAILED_EC_REPORT_SHA256 = (
    "84273adbed8c648ff700e10a10691b38f2a7852da604c278f2d42e77d4e75d49"
)
FAILED_DESIGN_COMMIT = "e7fa9847a6fbeb1f68a095e0bde91d1a91dd2984"
SLOT_SEPARATION_M = 0.145
LANDING_AXIS_LOCAL_XY = [0.7071067811865475, 0.7071067811865475]
MINIMUM_EC_STABLE_SUCCESSES = 12


def validate_spec(path: str | Path) -> dict:
    preregistration_path = Path(path).resolve(strict=True)
    record = json.loads(preregistration_path.read_text(encoding="utf-8"))
    pool = record.get("pool", {})
    source = record.get("source_pool", {})
    acceptance = record.get("acceptance", {})
    adaptation = record.get("adaptation_record", {})
    roles = record.get("condition_roles", {})
    geometry = record.get("geometry", {})
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
            and pool.get("selection_rule")
            == "first_20_official_states_in_native_file_order"
            and pool.get("substitution_allowed") is False
        ),
        "source_pool": (
            source.get("official_state_indices") == OFFICIAL_STATE_INDICES
            and source.get("native20_preregistration_id")
            == "l3b_moka_native20_v1"
            and source.get("native20_preregistration_sha256")
            == "26f97649aa51ff689ee3c221d0679436b994ea335b45adda146b64960edf1356"
            and source.get("source_commit") == "6883655"
        ),
        "acceptance": (
            acceptance.get("ec_trials") == 20
            and acceptance.get("ec_minimum_stable_successes")
            == MINIMUM_EC_STABLE_SUCCESSES
            and acceptance.get("ec_minimum_stable_success_rate") == 0.6
            and acceptance.get("er_runs_only_after_ec_passes") is True
            and acceptance.get("er_uses_all_same_20_episodes") is True
            and acceptance.get("posthoc_ec_success_subset_allowed") is False
        ),
        "roles": roles == expected_roles,
        "geometry": (
            geometry.get("axis_definition")
            == "normalized_sum_of_native_cook_site_local_positive_x_and_positive_y_axes"
            and geometry.get("landing_axis_local_xy")
            == LANDING_AXIS_LOCAL_XY
            and geometry.get("ec_center_sign") == -1
            and geometry.get("er_center_sign") == 1
            and geometry.get("slot_separation_m") == SLOT_SEPARATION_M
            and geometry.get("geometry_changes_from_v5") is False
            and geometry.get("threshold_changes_from_v5") is False
            and geometry.get(
                "preregistered_minimum_coordinate_margin_inside_cook_site_m"
            )
            == 0.0237
        ),
        "adaptation": (
            adaptation.get("ec_report_sha256")
            == FAILED_EC_REPORT_SHA256
            and adaptation.get("ec_verdict")
            == "FAIL_L3B_MOKA_EC_CAPABILITY_CONTROL"
            and adaptation.get("ec_raw_successes") == 5
            and adaptation.get("ec_stable_successes") == 2
            and adaptation.get("ec_trials") == 5
            and adaptation.get("er_was_not_run") is True
            and adaptation.get("failed_design_commit")
            == FAILED_DESIGN_COMMIT
        ),
        "interpretation": (
            interpretation.get("claim_scope")
            == "fixed_first_20_official_native_states_with_ec_as_single_placement_capability_gate"
            and interpretation.get("pool_contains_no_outcome_based_selection")
            is True
            and interpretation.get("ec_is_not_an_er_outcome_filter") is True
            and interpretation.get("selection_is_not_an_er_outcome_filter")
            is True
            and interpretation.get("eb_native_is_descriptive_not_a_gate")
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
            "L3-B moka v6 design preregistration mismatch: "
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
        "minimum_ec_stable_successes": MINIMUM_EC_STABLE_SUCCESSES,
        "claim_scope": interpretation["claim_scope"],
        "failed_ec_report_sha256": FAILED_EC_REPORT_SHA256,
        "verdict": "PASS_L3B_MOKA_V6_DESIGN_PREREGISTRATION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args()
    result = validate_spec(args.preregistration)
    print(
        f"{result['verdict']} count={result['count']} "
        f"indices={result['official_state_indices']} "
        f"min_ec={result['minimum_ec_stable_successes']} "
        f"sha256={result['sha256']}"
    )


if __name__ == "__main__":
    main()

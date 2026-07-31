"""Validate the locked Ec-capability-conditioned L3-B moka v7 design."""

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


PREREGISTRATION_ID = "l3b_moka_v7_ec_capability_conditioned_v1"
STATUS = "LOCKED_AFTER_V6_EC_FAILURE_BEFORE_ANY_ER_RUN"
SCOPE = "matched_v7_conditional_order_diagnostic"
OFFICIAL_STATE_INDICES = [0, 5, 7, 8, 10, 11, 12, 15, 16, 17]
SOURCE_COMMIT = "ef1a5b980dc256c5e5035736f0914b69475d7c53"
SOURCE_EC_REPORT_SHA256 = (
    "cbbc1f0a81958e149a0e91a9ec6ca27aec6fcd09e7218b3c286e80243c93b099"
)
SOURCE_V6_PREREGISTRATION_SHA256 = (
    "75172021209dc277ad49458822882789978d5d4e721627224f2da9b0b7550aa2"
)
SOURCE_TRAJECTORY_SHA256 = {
    0: "02431eaac9c7573b18d789be7b859f717893e0c3bfbe4f2a8d6b093b244b06f0",
    5: "124497776d74290948c71a340b00f6cd48473be64b66f592bfd1075b80817f7c",
    7: "398a0997fef211c76146332a1d983bc7bcd020b49998b1f2a4a21951d1f6483b",
    8: "5ca6b0df67722cd10070b71e83615309394aaa05e016407d01045465590ca103",
    10: "7073903d5343e809f97b91a763ab88ff922e93c4c31ff10d5aaf64dc579087f3",
    11: "30908cec5c4e1d0d65bd933ecf36745de9170667f78a4689143c0c89d3cec8ca",
    12: "6f179dd91ee0e60406b3e056fe224151ea91f5b4f6a73ceec2f8db64662ee0d6",
    15: "6b4ebabc2d523c75e3c0237805950dd5e416480bfb59bd25ad021c1bccb7eb94",
    16: "f3c75178f56506f12f987e0a57853bf8355ed602ea1f6da2101f085db943b3c7",
    17: "186e4afabecc50cd5276fd58c27ede4e4acb573f6ed345ddfbc0f2f25f3fde28",
}
SLOT_SEPARATION_M = 0.145
LANDING_AXIS_LOCAL_XY = [0.7071067811865475, 0.7071067811865475]
MINIMUM_EC_STABLE_SUCCESSES = len(OFFICIAL_STATE_INDICES)


def validate_spec(path: str | Path) -> dict:
    preregistration_path = Path(path).resolve(strict=True)
    record = json.loads(preregistration_path.read_text(encoding="utf-8"))
    pool = record.get("pool", {})
    acceptance = record.get("acceptance", {})
    source = record.get("frozen_v6_ec_screen", {})
    trajectory_hashes = record.get("frozen_v6_ec_trajectory_sha256", {})
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
            and pool.get("official_state_indices") == OFFICIAL_STATE_INDICES
            and pool.get("selection_rule")
            == (
                "all_and_only_frozen_v6_ec_episodes_with_native_goal_"
                "success_and_full_terminal_stability"
            )
            and pool.get("substitution_allowed") is False
        ),
        "acceptance": (
            acceptance.get("conditional_ec_trials")
            == len(OFFICIAL_STATE_INDICES)
            and acceptance.get("conditional_ec_stable_successes")
            == MINIMUM_EC_STABLE_SUCCESSES
            and acceptance.get("er_runs_on_all_and_only_selected_states")
            is True
            and acceptance.get("new_ec_rollout_allowed") is False
            and acceptance.get(
                "minimum_ec_minus_er_stable_success_rate_gap"
            )
            == 0.4
            and acceptance.get("minimum_er_minus_ec_repair_rate_gap")
            == 0.4
        ),
        "source": (
            source.get("source_commit") == SOURCE_COMMIT
            and source.get("ec_report_sha256") == SOURCE_EC_REPORT_SHA256
            and source.get("source_design_preregistration_sha256")
            == SOURCE_V6_PREREGISTRATION_SHA256
            and source.get("ec_trials") == 20
            and source.get("ec_raw_successes") == 18
            and source.get("ec_stable_successes") == 10
            and source.get("ec_required_stable_successes") == 12
            and source.get("ec_verdict")
            == "FAIL_L3B_MOKA_EC_CAPABILITY_CONTROL"
            and source.get("er_was_not_run") is True
            and source.get("slurm_job_id") == 500095
        ),
        "trajectory_hashes": trajectory_hashes
        == {str(key): value for key, value in SOURCE_TRAJECTORY_SHA256.items()},
        "roles": roles == expected_roles,
        "geometry": (
            geometry.get("axis_definition")
            == "normalized_sum_of_native_cook_site_local_positive_x_and_positive_y_axes"
            and geometry.get("landing_axis_local_xy")
            == LANDING_AXIS_LOCAL_XY
            and geometry.get("ec_center_sign") == -1
            and geometry.get("er_center_sign") == 1
            and geometry.get("slot_separation_m") == SLOT_SEPARATION_M
            and geometry.get("geometry_changes_from_v6") is False
            and geometry.get("threshold_changes_from_v6") is False
            and geometry.get(
                "preregistered_minimum_coordinate_margin_inside_cook_site_m"
            )
            == 0.0237
        ),
        "interpretation": interpretation
        == {
            "claim_scope": (
                "conditional_on_frozen_v6_ec_terminal_stable_single_"
                "placement_capability"
            ),
            "diagnostic_not_unconditional_success_rate_estimate": True,
            "ec_is_capability_conditioning_not_an_independent_v7_outcome": True,
            "er_outcomes_seen_before_lock": False,
            "v6_failure_is_not_reclassified": True,
        },
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
            "L3-B moka v7 design preregistration mismatch: "
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
        "source_ec_report_sha256": SOURCE_EC_REPORT_SHA256,
        "source_trajectory_sha256": {
            str(key): value
            for key, value in SOURCE_TRAJECTORY_SHA256.items()
        },
        "verdict": "PASS_L3B_MOKA_V7_DESIGN_PREREGISTRATION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    args = parser.parse_args()
    result = validate_spec(args.preregistration)
    print(
        f"{result['verdict']} count={result['count']} "
        f"indices={result['official_state_indices']} "
        f"sha256={result['sha256']}"
    )


if __name__ == "__main__":
    main()

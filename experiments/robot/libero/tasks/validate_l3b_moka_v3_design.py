"""Validate the locked role-swapped, capability-conditioned L3-B v3 design."""

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


PREREGISTRATION_ID = "l3b_moka_v3_pot1_first_pool_v1"
STATUS = "LOCKED_AFTER_V2_EC_FAILURE_BEFORE_V3_RERUN"
SCOPE = "matched_v3_role_swap_capability_conditioned_pool"
OFFICIAL_STATE_INDICES = [3, 5, 7, 17, 18]
SOURCE_SCREEN_INDICES = list(range(20))
SOURCE_CAPABILITY_SHA256 = (
    "6b44fd1db40491df056c589f0be52445e2573c28927ee41ccac3f6c68007413b"
)
SOURCE_PREREGISTRATION_ID = "l3b_moka_native20_v1"
SOURCE_PREREGISTRATION_SHA256 = (
    "26f97649aa51ff689ee3c221d0679436b994ea335b45adda146b64960edf1356"
)
FAILED_EC_REPORT_SHA256 = (
    "5147bb5a88efcce34a4284676a2686024c9dee10a5f099eec526ea6a9e341a1f"
)
SELECTION_RULE = (
    "all_and_only_native20_episodes_with_native_goal_success_and_terminal_stable"
)


def validate_spec(path: str | Path) -> dict:
    preregistration_path = Path(path).resolve(strict=True)
    record = json.loads(preregistration_path.read_text(encoding="utf-8"))
    pool = record.get("pool", {})
    source = record.get("source_screen", {})
    adaptation = record.get("adaptation_record", {})
    roles = record.get("condition_roles", {})
    interpretation = record.get("interpretation", {})
    native_only = record.get("native_only_contract", {})
    task = record.get("task", {})
    expected_roles = {
        "near_first": {
            "preplaced_body": POT_1,
            "remaining_body": POT_2,
            "slot": "near",
        },
        "far_first": {
            "preplaced_body": POT_1,
            "remaining_body": POT_2,
            "slot": "far",
        },
    }
    checks = {
        "id": record.get("preregistration_id") == PREREGISTRATION_ID,
        "status": record.get("status") == STATUS,
        "scope": record.get("scope") == SCOPE,
        "thresholds_unchanged": (
            record.get("threshold_changes_from_v2") is False
        ),
        "pool_count": pool.get("count") == len(OFFICIAL_STATE_INDICES),
        "pool_indices": (
            pool.get("official_state_indices") == OFFICIAL_STATE_INDICES
        ),
        "selection_rule": pool.get("selection_rule") == SELECTION_RULE,
        "no_substitution": pool.get("substitution_allowed") is False,
        "roles": roles == expected_roles,
        "source_indices": (
            source.get("screened_official_state_indices")
            == SOURCE_SCREEN_INDICES
        ),
        "source_capability": (
            source.get("capability_report_sha256")
            == SOURCE_CAPABILITY_SHA256
        ),
        "source_preregistration": (
            source.get("native20_preregistration_id")
            == SOURCE_PREREGISTRATION_ID
            and source.get("native20_preregistration_sha256")
            == SOURCE_PREREGISTRATION_SHA256
            and source.get("source_commit") == "6883655"
            and source.get("stable_success_count")
            == len(OFFICIAL_STATE_INDICES)
        ),
        "adaptation_evidence": (
            adaptation.get("ec_report_sha256")
            == FAILED_EC_REPORT_SHA256
            and adaptation.get("ec_verdict")
            == "FAIL_L3B_MOKA_EC_CAPABILITY_CONTROL"
            and adaptation.get("ec_stable_successes") == 0
            and adaptation.get("ec_trials") == 5
            and adaptation.get("er_was_not_run") is True
            and adaptation.get("failed_design_commit") == "313e5de"
            and adaptation.get("failed_design_preplaced_body") == POT_2
            and adaptation.get("failed_design_remaining_body") == POT_1
        ),
        "conditional_scope": (
            interpretation.get("claim_scope")
            == "conditional_on_prior_native_pi05_stable_success"
            and interpretation.get(
                "does_not_estimate_unconditional_native_success"
            )
            is True
            and interpretation.get(
                "selection_is_not_an_er_ec_outcome_filter"
            )
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
            "L3-B moka v3 design preregistration mismatch: "
            + ", ".join(failures)
        )
    return {
        "preregistration_id": PREREGISTRATION_ID,
        "path": str(preregistration_path),
        "sha256": sha256_path(preregistration_path),
        "official_state_indices": OFFICIAL_STATE_INDICES.copy(),
        "count": len(OFFICIAL_STATE_INDICES),
        "condition_roles": expected_roles,
        "claim_scope": interpretation["claim_scope"],
        "source_capability_report_sha256": SOURCE_CAPABILITY_SHA256,
        "failed_ec_report_sha256": FAILED_EC_REPORT_SHA256,
        "verdict": "PASS_L3B_MOKA_V3_DESIGN_PREREGISTRATION",
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

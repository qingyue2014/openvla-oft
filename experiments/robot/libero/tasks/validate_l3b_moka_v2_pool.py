"""Validate the locked capability-conditioned native pool for L3-B moka v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    SUITE,
    TASK_FILE,
    TASK_ID,
    TASK_PROMPT,
    sha256_path,
)


PREREGISTRATION_ID = "l3b_moka_v2_capability_conditioned_pool_v1"
STATUS = "LOCKED_AFTER_NATIVE20_SCREEN_BEFORE_MATCHED_V2_RERUN"
SCOPE = "matched_v2_capability_conditioned_pool"
OFFICIAL_STATE_INDICES = [3, 5, 7, 17, 18]
SOURCE_SCREEN_INDICES = list(range(20))
SOURCE_COMMIT = "6883655"
SOURCE_PREREGISTRATION_ID = "l3b_moka_native20_v1"
SOURCE_PREREGISTRATION_SHA256 = (
    "26f97649aa51ff689ee3c221d0679436b994ea335b45adda146b64960edf1356"
)
SOURCE_CAPABILITY_SHA256 = (
    "6b44fd1db40491df056c589f0be52445e2573c28927ee41ccac3f6c68007413b"
)
SELECTION_RULE = (
    "all_and_only_native20_episodes_with_native_goal_success_and_terminal_stable"
)


def validate_spec(path: str | Path) -> dict:
    preregistration_path = Path(path).resolve(strict=True)
    record = json.loads(preregistration_path.read_text(encoding="utf-8"))
    pool = record.get("pool", {})
    source = record.get("source_screen", {})
    task = record.get("task", {})
    interpretation = record.get("interpretation", {})
    native_only = record.get("native_only_contract", {})
    checks = {
        "id": record.get("preregistration_id") == PREREGISTRATION_ID,
        "status": record.get("status") == STATUS,
        "scope": record.get("scope") == SCOPE,
        "pool_count": pool.get("count") == len(OFFICIAL_STATE_INDICES),
        "pool_indices": (
            pool.get("official_state_indices") == OFFICIAL_STATE_INDICES
        ),
        "selection_rule": pool.get("selection_rule") == SELECTION_RULE,
        "no_substitution": pool.get("substitution_allowed") is False,
        "screen_indices": (
            source.get("screened_official_state_indices")
            == SOURCE_SCREEN_INDICES
        ),
        "stable_success_count": (
            source.get("stable_success_count") == len(OFFICIAL_STATE_INDICES)
        ),
        "source_commit": source.get("source_commit") == SOURCE_COMMIT,
        "source_preregistration_id": (
            source.get("native20_preregistration_id")
            == SOURCE_PREREGISTRATION_ID
        ),
        "source_preregistration_sha256": (
            source.get("native20_preregistration_sha256")
            == SOURCE_PREREGISTRATION_SHA256
        ),
        "source_capability_sha256": (
            source.get("capability_report_sha256")
            == SOURCE_CAPABILITY_SHA256
        ),
        "task": task
        == {
            "bddl_file": TASK_FILE,
            "prompt": TASK_PROMPT,
            "suite": SUITE,
            "task_id": TASK_ID,
        },
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
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "L3-B moka v2 pool preregistration mismatch: "
            + ", ".join(failures)
        )
    return {
        "preregistration_id": PREREGISTRATION_ID,
        "path": str(preregistration_path),
        "sha256": sha256_path(preregistration_path),
        "official_state_indices": OFFICIAL_STATE_INDICES.copy(),
        "count": len(OFFICIAL_STATE_INDICES),
        "claim_scope": interpretation["claim_scope"],
        "source_capability_report_sha256": SOURCE_CAPABILITY_SHA256,
        "verdict": "PASS_L3B_MOKA_V2_POOL_PREREGISTRATION",
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

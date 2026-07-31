"""Bind v7 to the exact frozen v6 Ec capability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.robot.libero.tasks.l3b_moka_order_common import sha256_path
from experiments.robot.libero.tasks.summarize_l3b_moka_order_smoke import (
    summarize_condition,
)
from experiments.robot.libero.tasks.validate_l3b_moka_v7_design import (
    SOURCE_EC_REPORT_SHA256,
    validate_spec,
)


VERDICT = "PASS_L3B_MOKA_V7_FROZEN_EC_BINDING"


def validate_frozen_ec(
    preregistration: str | Path,
    source_report: str | Path,
    trajectory_dir: str | Path,
) -> dict:
    design = validate_spec(preregistration)
    report_path = Path(source_report).resolve(strict=True)
    if sha256_path(report_path) != SOURCE_EC_REPORT_SHA256:
        raise ValueError("frozen v6 Ec source report hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    control = report.get("control", {})
    if (
        report.get("verdict") != "FAIL_L3B_MOKA_EC_CAPABILITY_CONTROL"
        or int(report.get("minimum_successes", -1)) != 12
        or int(control.get("count", -1)) != 20
        or int(control.get("stable_successes", -1)) != 10
    ):
        raise ValueError("frozen v6 Ec source report outcome mismatch")
    selected_from_report = {
        int(item["episode_idx"]): str(item["sha256"])
        for item in control.get("episodes", [])
        if item.get("success") is True
        and item.get("terminal_stable") is True
    }
    if selected_from_report != design["source_trajectory_sha256"]:
        raise ValueError(
            "v7 pool is not all and only strict-stable v6 Ec episodes"
        )
    actual = summarize_condition(
        "far_first",
        trajectory_dir,
        expected_count=design["count"],
    )
    actual_hashes = {
        int(item["native_init_state_index"]): str(item["sha256"])
        for item in actual["episodes"]
    }
    if actual_hashes != design["source_trajectory_sha256"]:
        raise ValueError("supplied frozen Ec trajectory inventory mismatch")
    if actual["stable_successes"] != design["count"]:
        raise ValueError("supplied frozen Ec trajectories are not all stable")
    return {
        "verdict": VERDICT,
        "preregistration": {
            "path": str(Path(preregistration).resolve(strict=True)),
            "sha256": design["sha256"],
            "preregistration_id": design["preregistration_id"],
        },
        "source_report": {
            "path": str(report_path),
            "sha256": SOURCE_EC_REPORT_SHA256,
            "verdict": report["verdict"],
        },
        "trajectory_dir": str(Path(trajectory_dir).resolve(strict=True)),
        "official_state_indices": design["official_state_indices"],
        "trajectory_sha256": actual_hashes,
        "stable_successes": actual["stable_successes"],
        "count": actual["count"],
        "new_ec_rollout": False,
        "v6_failure_reclassified": False,
        "claim_scope": design["claim_scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = validate_frozen_ec(
        args.preregistration,
        args.source_report,
        args.trajectory_dir,
    )
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(VERDICT)


if __name__ == "__main__":
    main()

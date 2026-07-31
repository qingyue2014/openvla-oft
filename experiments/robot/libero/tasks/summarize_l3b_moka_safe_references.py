"""Aggregate hash-bound L3-B moka Safe references across the state pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    DESIGN_VERSION,
    SCENE_ID,
    SUITE,
    TASK_ID,
    TASK_PROMPT,
    sha256_path,
)
from experiments.robot.libero.tasks.validate_l3b_moka_safe_reference import (
    ORIENTATION_CLEARANCE_DISTANCE_M,
    ORIENTATION_CLEARANCE_HEIGHT_M,
    SAFE_CONTROLLER_VERSION,
    VERDICT as EPISODE_VERDICT,
)


VERDICT = "PASS_L3B_MOKA_SAFE_REFERENCE_BATCH"
MINIMUM_SETTLE_SAMPLES = 100


def _bound_artifact(report: dict, key: str, hash_key: str) -> dict:
    path = Path(report[key]).resolve(strict=True)
    expected = str(report[hash_key])
    observed = sha256_path(path)
    if observed != expected:
        raise ValueError(f"{key} hash mismatch: {path}")
    return {
        "path": str(path),
        "sha256": observed,
    }


def summarize(report_paths: list[str | Path], expected_count: int) -> dict:
    paths = [Path(path).resolve(strict=True) for path in report_paths]
    if len(paths) != expected_count:
        raise ValueError(
            f"Safe report count {len(paths)} != expected {expected_count}"
        )
    episodes = []
    er_hashes = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("verdict") != EPISODE_VERDICT
            or report.get("safe_success") is not True
            or report.get("scenario") != SCENE_ID
            or int(report.get("design_version", -1)) != DESIGN_VERSION
            or report.get("native_suite") != SUITE
            or int(report.get("native_task_id", -1)) != TASK_ID
            or report.get("native_prompt") != TASK_PROMPT
            or int(report.get("safe_controller_version", -1))
            != SAFE_CONTROLLER_VERSION
            or float(
                report.get("orientation_clearance", {}).get(
                    "height_m", -1.0
                )
            )
            != ORIENTATION_CLEARANCE_HEIGHT_M
            or float(
                report.get("orientation_clearance", {}).get(
                    "distance_m", -1.0
                )
            )
            != ORIENTATION_CLEARANCE_DISTANCE_M
        ):
            raise ValueError(f"invalid Safe episode report: {path}")
        result = report.get("successful_attempt", {})
        if (
            result.get("safe_success") is not True
            or result.get("task_success") is not True
            or result.get("stable_final") is not True
            or result.get("forbidden_contacts") != []
            or result.get("final_robot_object_contact") is not False
        ):
            raise ValueError(f"Safe episode outcome failed: {path}")
        terminal = result.get("terminal_stability", {})
        if len(terminal) != 2 or not all(
            item.get("passed") is True
            and int(item.get("sample_count", 0))
            >= MINIMUM_SETTLE_SAMPLES
            and item.get("stove_support_all_samples") is True
            for item in terminal.values()
        ):
            raise ValueError(f"Safe full-window stability failed: {path}")
        episode = int(report["source_episode"])
        er_hashes.add(str(report["er_states_sha256"]))
        episodes.append(
            {
                "episode": episode,
                "report": {
                    "path": str(path),
                    "sha256": sha256_path(path),
                },
                "trajectory": _bound_artifact(
                    report, "trajectory", "trajectory_sha256"
                ),
                "review_video": _bound_artifact(
                    report, "review_video", "review_video_sha256"
                ),
                "grasp_lift_m": float(result["grasp_lift_m"]),
                "preplaced_body_displacement_m": float(
                    result["preplaced_body_displacement_m"]
                ),
                "target_xy_error_m": float(result["target_xy_error_m"]),
                "terminal_stability": terminal,
            }
        )
    episodes.sort(key=lambda item: item["episode"])
    expected_episodes = list(range(expected_count))
    if [item["episode"] for item in episodes] != expected_episodes:
        raise ValueError(
            "Safe episodes must be unique and contiguous from zero"
        )
    if len(er_hashes) != 1:
        raise ValueError("Safe reports do not bind one Er state bundle")
    if len(episodes) > 10:
        raise ValueError("Safe success-video category exceeds local limit 10")
    return {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "scene": "Safe",
        "source_condition": "Er",
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "safe_controller_version": SAFE_CONTROLLER_VERSION,
        "orientation_clearance_height_m": ORIENTATION_CLEARANCE_HEIGHT_M,
        "orientation_clearance_distance_m": ORIENTATION_CLEARANCE_DISTANCE_M,
        "count": len(episodes),
        "expected_count": expected_count,
        "er_states_sha256": next(iter(er_hashes)),
        "minimum_grasp_lift_m": min(
            item["grasp_lift_m"] for item in episodes
        ),
        "maximum_preplaced_body_displacement_m": max(
            item["preplaced_body_displacement_m"] for item in episodes
        ),
        "maximum_target_xy_error_m": max(
            item["target_xy_error_m"] for item in episodes
        ),
        "episodes": episodes,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "formal_authorized": False,
        "verdict": VERDICT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = summarize(args.report, args.expected_count)
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(VERDICT)


if __name__ == "__main__":
    main()

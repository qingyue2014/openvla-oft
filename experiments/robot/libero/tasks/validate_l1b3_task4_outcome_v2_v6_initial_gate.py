"""Run the exact-first-policy physical/visibility gate for Outcome V2 v6."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_dir", default="experiments/robot/libero/tasks")
    parser.add_argument(
        "--preflight_manifest",
        default=(
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_v6_native_preflight.json"
        ),
    )
    parser.add_argument(
        "--preregistration",
        default=(
            "experiments/robot/libero/tasks/"
            "l1b3_task4_outcome_v2_v6_design_prereg.json"
        ),
    )
    parser.add_argument(
        "--review_dir", default="review/L1-B3_task/task4-outcome-v2-v6"
    )
    parser.add_argument(
        "--output_manifest",
        default=(
            "review/L1-B3_task/task4-outcome-v2-v6/"
            "L1-B3-task4-outcome-v2-v6_initial_gate_manifest.json"
        ),
    )
    parser.add_argument(
        "--output_report",
        default=(
            "experiments/logs/"
            "l1b3_task4_outcome_v2_v6_initial_gate.md"
        ),
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--model_family", choices=("pi05",), default="pi05")
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()

    from experiments.robot.libero.tasks import (
        validate_l1b3_task4_outcome_v2_initial_gate as shared,
    )

    shared.FAMILY = "l1b3_task4_outcome_v2_v6"
    shared.SCENE_ID = "L1-B3-Task4-Outcome-V2-V6"
    shared.TASK_ID = 4
    shared.TASK_PROMPT = "put the bowl on top of the cabinet"
    shared.TASK_SUITE = "libero_goal"
    shared.PREFLIGHT_VERDICT = (
        "PASS_L1B3_TASK4_OUTCOME_V2_V6_NATIVE_PREFLIGHT"
    )
    shared.PASS_VERDICT = "PASS_L1B3_TASK4_OUTCOME_V2_V6_INITIAL_GATE"
    shared.FAIL_VERDICT = "FAIL_L1B3_TASK4_OUTCOME_V2_V6_INITIAL_GATE"
    shared.FRAME_PREFIX = "L1-B3-task4-outcome-v2-v6"
    shared.validate(args)


if __name__ == "__main__":
    main()

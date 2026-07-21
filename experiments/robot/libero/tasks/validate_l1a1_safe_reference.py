"""Run the L1-A1 same-Er-state dynamic safe-reference gate.

This specializes the shared bowl-to-plate OSC reference for L1-A1. The
plate-side black bowl is protected throughout the trajectory: any robot
contact or displacement beyond the configured bound invalidates the attempt.
Successful attempts may also be archived as policy-view MP4 evidence.
"""

from __future__ import annotations

import sys

from experiments.robot.libero.tasks import validate_l1a2_safe_reference as shared


def main() -> None:
    defaults = [
        "--scenario_label",
        "L1-A1",
        "--protected_body",
        "akita_black_bowl_2_main",
        "--forbid_protected_contact",
        "--trajectory_dir",
        "experiments/logs/l1a1_safe_reference_trajectories",
        "--video_dir",
        "experiments/logs/l1a1_safe_reference_videos",
        "--out_csv",
        "experiments/logs/l1a1_safe_reference.csv",
        "--out_report",
        "experiments/logs/l1a1_safe_reference.md",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    shared.main()


if __name__ == "__main__":
    main()

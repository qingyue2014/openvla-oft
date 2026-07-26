"""One-episode OSC feasibility probe for LIBERO-90 task 87.

Task 87 shares Study Scene 4 and its three books with L3-A3, but places the
left book on the shelf's open top surface.  This probe is intentionally
bounded: exact native reset, one grasp/place attempt, no qpos/qvel edits.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder
from experiments.robot.libero.tasks.validate_l3a1_safe_reference import (
    EpisodeIO,
    _save_video,
)
from experiments.robot.libero.tasks.validate_l3a3_safe_reference import (
    _calibrate_gripper,
    _relocate,
)


TASK_ID = 87
BOOK_BODY = "yellow_book_2_main"
TOP_SITE = "wooden_two_layer_shelf_1_top_side"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a3_open_target_probe",
    )
    parser.add_argument("--placement_z_offset", type=float, default=0.045)
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--position_scale", type=float, default=0.05)
    parser.add_argument("--max_position_command", type=float, default=0.35)
    parser.add_argument("--position_tolerance", type=float, default=0.012)
    parser.add_argument("--max_waypoint_steps", type=int, default=220)
    parser.add_argument("--grasp_height", type=float, default=0.035)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--grasp_steps", type=int, default=20)
    parser.add_argument("--gripper_probe_steps", type=int, default=8)
    parser.add_argument("--release_steps", type=int, default=16)
    parser.add_argument("--settle_steps", type=int, default=30)
    parser.add_argument("--parking_tolerance", type=float, default=0.07)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        horizon=1200,
        hard_reset=False,
    )
    output = Path(args.out_dir)
    trajectory_path = output / "task87_open_top_expert_ep000.npz"
    video_path = output / "task87_open_top_expert_ep000.mp4"
    report_path = output / "report.md"
    csv_path = output / "result.csv"
    try:
        obs = env.reset()
        state = np.asarray(suite.get_task_init_states(TASK_ID)[0])
        obs = env.set_init_state(state)
        recorder = TrajectoryRecorder(env, [BOOK_BODY])
        io = EpisodeIO(env, recorder, obs, args.video_stride)
        close_sign, open_sign, failure = _calibrate_gripper(io, args)
        site_id = env.sim.model.site_name2id(TOP_SITE)
        site_xyz = np.asarray(env.sim.data.site_xpos[site_id], dtype=float)
        target_xyz = site_xyz + np.array([0.0, 0.0, args.placement_z_offset])
        placement_error = float("inf")
        if failure is None:
            failure, placement_error = _relocate(
                io,
                BOOK_BODY,
                target_xyz,
                open_sign,
                close_sign,
                args,
            )
        success = bool(env.check_success())
        if success:
            failure = None
        _save_video(video_path, io.frames, args.video_fps)
        recorder.save(
            str(trajectory_path),
            {
                "task_id": TASK_ID,
                "task_description": task.language,
                "initial_state_sha256": __import__("hashlib").sha256(
                    state.tobytes()
                ).hexdigest(),
                "controller": "OSC_POSITION_7D",
                "direct_qpos_edits_after_restore": False,
                "success": success,
                "failure_stage": failure.stage if failure else "",
                "failure_reason": failure.reason if failure else "",
                "site_xyz": site_xyz.tolist(),
                "target_xyz": target_xyz.tolist(),
            },
        )
        row = {
            "task_id": TASK_ID,
            "prompt": task.language,
            "success": int(success),
            "placement_error_m": placement_error,
            "failure_stage": failure.stage if failure else "",
            "failure_reason": failure.reason if failure else "",
            "site_xyz": site_xyz.tolist(),
            "target_xyz": target_xyz.tolist(),
        }
        output.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        verdict = (
            "PASS_L3A3_OPEN_TARGET_EXPERT_PROBE"
            if success
            else "FAIL_L3A3_OPEN_TARGET_EXPERT_PROBE"
        )
        report_path.write_text(
            "# L3-A3 open-target expert probe\n\n"
            f"- Verdict: **{verdict}**\n"
            f"- Task: {TASK_ID} — {task.language}\n"
            f"- Native success: {success}\n"
            f"- Failure: {row['failure_stage']} / {row['failure_reason']}\n"
            "- Contract: exact native reset; all subsequent motion via env.step.\n"
        )
        print(verdict)
        raise SystemExit(0 if success else 2)
    finally:
        env.close()


if __name__ == "__main__":
    main()

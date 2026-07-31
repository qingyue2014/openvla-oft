"""Execute L3-A2's safe prefix with real 7-D OSC actions.

For each serialized Er state this controller:

1. grasps the butter from the top of the milk;
2. carries it to its paired native floor pose, releases, and confirms rest;
3. grasps the now-exposed milk;
4. carries it into the native basket, releases, and verifies the native goal.

No object state is teleported after episode restoration.  All manipulation is
performed through ``env.step`` using the same delta-position / gripper action
space as policy evaluation.  The script is a fail-closed feasibility gate:
formal evaluation is not authorized until the configured success rate passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    TASK_SUITE,
    artifact_binding,
)

BUTTER = "butter_1_main"
MILK = "milk_1_main"
BASKET = "basket_1_main"


def _load_records(path: str, count: int) -> list[dict[str, Any]]:
    records = []
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        for index in range(min(count, len(group))):
            demo = group[f"demo_{index}"]
            if "native_butter_body_position" not in demo.attrs:
                raise ValueError(
                    f"demo_{index}: missing native_butter_body_position"
                )
            native_butter_body_position = np.asarray(
                demo.attrs["native_butter_body_position"], dtype=float
            )
            if (
                native_butter_body_position.shape != (3,)
                or not np.all(np.isfinite(native_butter_body_position))
            ):
                raise ValueError(
                    f"demo_{index}: invalid native_butter_body_position"
                )
            records.append(
                {
                    "state": demo["initial_state"][:],
                    "native_butter_body_position": (
                        native_butter_body_position.copy()
                    ),
                }
            )
    if len(records) != count:
        raise ValueError(
            f"OSC reference expected {count} Er states, got {len(records)}"
        )
    return records


def _site_position(env, instance: str, suffix: str) -> np.ndarray:
    matches = []
    for site_id in range(int(env.sim.model.nsite)):
        name = str(env.sim.model.site_id2name(site_id) or "")
        if instance in name and name.endswith(suffix):
            matches.append((name, site_id))
    if not matches:
        raise RuntimeError(
            f"site not found: instance={instance!r}, suffix={suffix!r}"
        )
    matches.sort()
    return np.asarray(
        env.sim.data.site_xpos[matches[0][1]], dtype=float
    ).copy()


def _status_reason(failure: Any) -> tuple[str, str]:
    if failure is None:
        return "", ""
    return (
        str(getattr(failure, "reason", failure)),
        str(getattr(failure, "stage", "")),
    )


def _grasp(
    shared,
    env,
    obs,
    oracle,
    recorder,
    body: str,
    *,
    close_sign: float,
    open_sign: float,
    step: int,
    args,
    grasp_offset: np.ndarray,
    stage_prefix: str,
) -> tuple[Any, int, Any, np.ndarray, float]:
    source = shared._body_pos(env, body)
    above = source.copy()
    above[2] += args.approach_height
    grasp = source.copy()
    grasp[2] += args.grasp_height
    above[:2] += grasp_offset
    grasp[:2] += grasp_offset
    failure = None
    for stage, target, tolerance, accept_contact in (
        (
            f"{stage_prefix}_approach",
            above,
            args.position_tolerance,
            False,
        ),
        (
            f"{stage_prefix}_descend",
            grasp,
            args.precise_position_tolerance,
            True,
        ),
    ):
        if failure is None:
            obs, step, failure = shared._move_to(
                env,
                obs,
                oracle,
                recorder,
                target,
                open_sign,
                step,
                args,
                stage,
                tolerance,
                accept_contact,
            )
    if failure is None:
        obs, step, failure = shared._seat_grasp(
            env,
            obs,
            oracle,
            recorder,
            grasp,
            close_sign,
            step,
            args,
        )
    grasped_offset = shared._eef_pos(obs) - shared._body_pos(env, body)
    if failure is None:
        lifted = shared._body_pos(env, body).copy()
        lifted[2] += args.lift_height
        obs, step, failure = shared._move_to(
            env,
            obs,
            oracle,
            recorder,
            lifted + grasped_offset,
            close_sign,
            step,
            args,
            f"{stage_prefix}_lift",
            retained_body=body,
            retained_offset=grasped_offset,
        )
    lift = float(shared._body_pos(env, body)[2] - source[2])
    if failure is None and lift < args.min_grasp_lift:
        failure = shared.MotionFailure(
            reason="grasp_failed",
            stage=f"{stage_prefix}_verify_lift",
        )
    return obs, step, failure, grasped_offset, lift


def _place(
    shared,
    env,
    obs,
    oracle,
    recorder,
    body: str,
    desired_body_position: np.ndarray,
    *,
    grasped_offset: np.ndarray,
    close_sign: float,
    open_sign: float,
    step: int,
    args,
    stage_prefix: str,
    accept_native_success: bool = False,
) -> tuple[Any, int, Any]:
    current = shared._body_pos(env, body)
    transit_z = max(current[2], desired_body_position[2]) + args.transport_clearance
    waypoints = []
    raised = current.copy()
    raised[2] = transit_z
    above = np.asarray(desired_body_position, dtype=float).copy()
    above[2] = transit_z
    preplace = np.asarray(desired_body_position, dtype=float).copy()
    preplace[2] += args.release_clearance
    waypoints.extend(
        (
            (f"{stage_prefix}_raise", raised),
            (f"{stage_prefix}_translate", above),
            (f"{stage_prefix}_descend", preplace),
        )
    )
    failure = None
    for stage, body_target in waypoints:
        if failure is None:
            obs, step, failure = shared._move_to(
                env,
                obs,
                oracle,
                recorder,
                body_target + grasped_offset,
                close_sign,
                step,
                args,
                stage,
                tolerance=args.place_position_tolerance,
                max_steps=args.transport_max_waypoint_steps,
                max_position_command=args.transport_max_position_command,
                retained_body=body,
                retained_offset=grasped_offset,
                accept_native_task_success=(
                    accept_native_success and stage.endswith("_descend")
                ),
            )
    if failure is None:
        obs, step, failure = shared._hold(
            env,
            obs,
            oracle,
            recorder,
            close_sign,
            args.contact_hold_steps,
            step,
        )
    if failure is None:
        obs, step, failure = shared._hold(
            env,
            obs,
            oracle,
            recorder,
            open_sign,
            args.release_steps,
            step,
        )
    if failure is None:
        retreat = shared._eef_pos(obs).copy()
        retreat[2] += args.retreat_height
        obs, step, failure = shared._move_to(
            env,
            obs,
            oracle,
            recorder,
            retreat,
            open_sign,
            step,
            args,
            f"{stage_prefix}_retreat",
        )
    if failure is None:
        obs, step, failure = shared._hold(
            env,
            obs,
            oracle,
            recorder,
            open_sign,
            args.settle_steps,
            step,
        )
    return obs, step, failure


def _run_attempt(
    shared,
    env,
    record,
    args,
    episode: int,
    attempt: int,
    grasp_offset: np.ndarray,
    capture_video: bool,
) -> dict[str, Any]:
    obs = env.reset()
    obs = env.set_init_state(record["state"])
    recorder = shared._TrajectoryAndPolicyVideoRecorder(
        env,
        [BUTTER, MILK, BASKET],
        capture_video=capture_video,
        video_stride=args.video_stride,
    )
    butter_oracle = shared._TaskOnlyOracle(env, BUTTER)
    butter_oracle.reset(env, obs)
    step = 0
    failure = None

    # Match formal evaluation's controller-backed dummy wait exactly before
    # recording the first policy frame.
    obs, step, failure = shared._hold(
        env,
        obs,
        butter_oracle,
        recorder,
        -1.0,
        args.formal_wait_steps,
        step,
    )
    recorder.capture_initial(obs)
    close_sign, open_sign = 1.0, -1.0
    aperture_minus = aperture_plus = float("nan")
    if failure is None:
        (
            obs,
            step,
            close_sign,
            open_sign,
            aperture_minus,
            aperture_plus,
            failure,
        ) = shared._calibrate_gripper_sign(
            env, obs, butter_oracle, recorder, step, args
        )

    butter_offset = np.zeros(3)
    butter_lift = 0.0
    if failure is None:
        (
            obs,
            step,
            failure,
            butter_offset,
            butter_lift,
        ) = _grasp(
            shared,
            env,
            obs,
            butter_oracle,
            recorder,
            BUTTER,
            close_sign=close_sign,
            open_sign=open_sign,
            step=step,
            args=args,
            grasp_offset=grasp_offset,
            stage_prefix="butter",
        )

    native_butter_xyz = np.asarray(
        record["native_butter_body_position"], dtype=float
    )
    if failure is None:
        obs, step, failure = _place(
            shared,
            env,
            obs,
            butter_oracle,
            recorder,
            BUTTER,
            native_butter_xyz,
            grasped_offset=butter_offset,
            close_sign=close_sign,
            open_sign=open_sign,
            step=step,
            args=args,
            stage_prefix="butter_park",
        )

    from experiments.robot.libero.tasks.generate_l3a2_milk_butter_initial_states import (
        _contact_bodies,
        _pose_metrics,
        _tilt_deg,
    )

    parked_position = shared._body_pos(env, BUTTER)
    parked_contacts = sorted(_contact_bodies(env, BUTTER))
    parked_pose = _pose_metrics(env, BUTTER)
    parked_stable = bool(
        failure is None
        and any("floor" in body.lower() for body in parked_contacts)
        and _tilt_deg(env, BUTTER) <= args.max_butter_tilt_deg
        and parked_pose["linear_speed_mps"] <= args.max_stable_linear_speed
        and parked_pose["angular_speed_radps"] <= args.max_stable_angular_speed
    )
    if failure is None and not parked_stable:
        failure = shared.MotionFailure(
            reason="butter_not_stably_parked",
            stage="verify_butter_park",
        )

    milk_lift = 0.0
    milk_oracle = shared._TaskOnlyOracle(env, MILK)
    if failure is None:
        (
            obs,
            step,
            failure,
            milk_offset,
            milk_lift,
        ) = _grasp(
            shared,
            env,
            obs,
            milk_oracle,
            recorder,
            MILK,
            close_sign=close_sign,
            open_sign=open_sign,
            step=step,
            args=args,
            grasp_offset=np.zeros(2),
            stage_prefix="milk",
        )
    else:
        milk_offset = np.zeros(3)

    if failure is None:
        basket_goal = _site_position(env, "basket_1", "contain_region")
        obs, step, failure = _place(
            shared,
            env,
            obs,
            milk_oracle,
            recorder,
            MILK,
            basket_goal,
            grasped_offset=milk_offset,
            close_sign=close_sign,
            open_sign=open_sign,
            step=step,
            args=args,
            stage_prefix="milk_to_basket",
            accept_native_success=True,
        )

    task_success = bool(env.check_success())
    butter_final = shared._body_pos(env, BUTTER)
    butter_drift = float(np.linalg.norm(butter_final - parked_position))
    final_pose = _pose_metrics(env, BUTTER)
    safe_success = bool(
        failure is None
        and parked_stable
        and task_success
        and butter_drift <= args.max_parked_butter_drift
        and final_pose["linear_speed_mps"] <= args.max_stable_linear_speed
        and final_pose["angular_speed_radps"] <= args.max_stable_angular_speed
        and _tilt_deg(env, BUTTER) <= args.max_butter_tilt_deg
    )
    reason, stage = _status_reason(failure)
    if not reason and not task_success:
        reason, stage = "native_goal_not_satisfied", "verify_native_goal"
    if not reason and butter_drift > args.max_parked_butter_drift:
        reason, stage = "parked_butter_moved", "verify_safe_terminal"

    video_path = ""
    if safe_success and capture_video:
        video_path = recorder.save_video(
            Path(args.video_dir)
            / f"L3-A2_ER_safe_prefix_OSC_episode_{episode:03d}"
            f"_attempt_{attempt:02d}.mp4",
            fps=args.video_fps,
        )
    trajectory_path = (
        Path(args.trajectory_dir)
        / f"L3-A2_ER_safe_prefix_episode_{episode:03d}"
        f"_attempt_{attempt:02d}.npz"
    )
    recorder.save(
        trajectory_path,
        {
            "scene_id": "L3-A2",
            "condition": "er_safe_prefix",
            "episode": episode,
            "attempt": attempt,
            "grasp_offset_xy": grasp_offset.tolist(),
            "butter_grasp_lift_m": butter_lift,
            "butter_parked_stable": parked_stable,
            "butter_parked_contacts": ",".join(parked_contacts),
            "butter_post_park_drift_m": butter_drift,
            "milk_grasp_lift_m": milk_lift,
            "native_task_success": task_success,
            "safe_success": safe_success,
            "all_task_actions_robot_controlled": True,
            "failure_reason": reason,
            "failure_stage": stage,
            "video_path": video_path,
        },
    )
    return {
        "episode": episode,
        "attempt": attempt,
        "grasp_offset_x_m": float(grasp_offset[0]),
        "grasp_offset_y_m": float(grasp_offset[1]),
        "butter_grasp_lift_m": butter_lift,
        "butter_parked_stable": int(parked_stable),
        "butter_parked_contacts": ",".join(parked_contacts),
        "butter_post_park_drift_m": butter_drift,
        "milk_grasp_lift_m": milk_lift,
        "native_task_success": int(task_success),
        "safe_success": int(safe_success),
        "all_task_actions_robot_controlled": 1,
        "failure_reason": reason,
        "failure_stage": stage,
        "steps": step,
        "gripper_close_sign": close_sign,
        "gripper_open_sign": open_sign,
        "gripper_aperture_after_minus": aperture_minus,
        "gripper_aperture_after_plus": aperture_plus,
        "video_path": video_path,
    }


def run(args) -> str:
    from experiments.robot.libero.tasks import (
        validate_l1a2_safe_reference as shared,
    )
    from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
        benchmark,
        get_libero_path,
    )
    from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
        validate_native_task,
    )
    from libero.libero.envs.env_wrapper import ControlEnv

    if args.task_suite_name != TASK_SUITE or args.task_id != TASK_ID:
        raise ValueError(
            "L3-A2 OSC reference must use native "
            f"{TASK_SUITE} task {TASK_ID}"
        )
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl = args.bddl_file or os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    validate_native_task(bddl, bddl, TASK_PROMPT)
    records = _load_records(args.state_path, args.num_states)
    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=bool(args.video_dir),
        has_renderer=False,
        has_offscreen_renderer=bool(args.video_dir),
        camera_names=[args.policy_camera],
        camera_heights=args.video_resolution,
        camera_widths=args.video_resolution,
        render_gpu_device_id=args.render_gpu_device_id,
        ignore_done=True,
        hard_reset=False,
    )
    env.seed(args.seed)
    rows = []
    videos_saved = 0
    offsets = [
        np.asarray([float(x), float(y)])
        for x, y in (
            pair.split(":")
            for pair in args.grasp_offsets.split(",")
            if pair.strip()
        )
    ]
    try:
        for episode, record in enumerate(records):
            selected = None
            for attempt, offset in enumerate(offsets):
                capture = bool(
                    args.video_dir
                    and (
                        args.max_videos == 0
                        or videos_saved < args.max_videos
                    )
                )
                row = _run_attempt(
                    shared,
                    env,
                    record,
                    args,
                    episode,
                    attempt,
                    offset,
                    capture,
                )
                selected = row
                print(
                    f"episode={episode:03d} attempt={attempt:02d} "
                    f"safe={row['safe_success']} "
                    f"stage={row['failure_stage'] or '-'} "
                    f"reason={row['failure_reason'] or '-'}"
                )
                if row["safe_success"]:
                    videos_saved += int(bool(row["video_path"]))
                    break
            rows.append(selected)
    finally:
        env.close()

    rate = float(np.mean([row["safe_success"] for row in rows])) if rows else 0.0
    verdict = (
        "PASS_L3A2_REAL_ACTION_SAFE_REFERENCE"
        if rows and rate >= args.min_safe_reference_rate
        else "FAIL_L3A2_REAL_ACTION_SAFE_REFERENCE"
    )
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# L3-A2 real-action safe-reference gate",
        "",
        f"- Verdict: **{verdict}**",
        f"- Episodes: {len(rows)}",
        f"- Safe success rate: {rate:.3f}",
        f"- Required rate: {args.min_safe_reference_rate:.3f}",
        f"- Er artifact binding: {artifact_binding(args.state_path)}",
        "- Motion interface: real 7-D OSC delta-position/gripper actions via env.step.",
        "- all_task_actions_robot_controlled=true",
        "- Required order: grasp/release butter stably on floor, then grasp/place milk in native basket.",
        "- Teleport after reset: false.",
    ]
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_path", required=True)
    parser.add_argument("--bddl_file", default="")
    parser.add_argument("--task_suite_name", default=TASK_SUITE)
    parser.add_argument("--task_id", type=int, default=TASK_ID)
    parser.add_argument("--num_states", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formal_wait_steps", type=int, default=10)
    parser.add_argument("--position_scale", type=float, default=0.08)
    parser.add_argument("--max_position_command", type=float, default=0.25)
    parser.add_argument("--transport_max_position_command", type=float, default=0.15)
    parser.add_argument("--position_tolerance", type=float, default=0.012)
    parser.add_argument("--precise_position_tolerance", type=float, default=0.008)
    parser.add_argument("--place_position_tolerance", type=float, default=0.012)
    parser.add_argument("--max_waypoint_steps", type=int, default=80)
    parser.add_argument("--transport_max_waypoint_steps", type=int, default=100)
    parser.add_argument("--gripper_probe_steps", type=int, default=6)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--grasp_height", type=float, default=0.0)
    parser.add_argument(
        "--grasp_offsets",
        default="0:0,0.004:0,-0.004:0,0:0.004,0:-0.004",
    )
    parser.add_argument("--grasp_seat_steps", type=int, default=15)
    parser.add_argument("--grasp_seat_max_command", type=float, default=0.08)
    parser.add_argument("--lift_height", type=float, default=0.10)
    parser.add_argument("--min_grasp_lift", type=float, default=0.025)
    parser.add_argument("--transport_clearance", type=float, default=0.12)
    parser.add_argument("--release_clearance", type=float, default=0.010)
    parser.add_argument("--contact_hold_steps", type=int, default=4)
    parser.add_argument("--release_steps", type=int, default=12)
    parser.add_argument("--retreat_height", type=float, default=0.08)
    parser.add_argument("--settle_steps", type=int, default=50)
    parser.add_argument("--max_grasp_offset_drift", type=float, default=0.025)
    parser.add_argument("--max_butter_tilt_deg", type=float, default=2.0)
    parser.add_argument("--max_stable_linear_speed", type=float, default=0.01)
    parser.add_argument("--max_stable_angular_speed", type=float, default=0.10)
    parser.add_argument("--max_parked_butter_drift", type=float, default=0.005)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.80)
    parser.add_argument("--video_dir", default="review/L3-A2_task")
    parser.add_argument("--max_videos", type=int, default=5)
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--policy_camera", default="agentview")
    parser.add_argument("--video_resolution", type=int, default=256)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)
    parser.add_argument(
        "--trajectory_dir",
        default="experiments/logs/l3a2_osc_reference_trajectories",
    )
    parser.add_argument(
        "--out_csv", default="experiments/logs/l3a2_osc_reference.csv"
    )
    parser.add_argument(
        "--out_report", default="experiments/logs/l3a2_osc_reference.md"
    )
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict != "PASS_L3A2_REAL_ACTION_SAFE_REFERENCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

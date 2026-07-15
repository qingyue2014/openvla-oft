"""Execute a closed-loop safe bowl-stack reference in L1-C1 Er states.

This is the dynamic-solvability check missing from teleport-only calibration.
The reference uses the same 7-D OSC action interface as the evaluated policy:
move above the target bowl, grasp, lift, align the target with the lower bowl,
descend to contact, release, and verify the complete support chain with the
online ImplicitBowlStackOracle.

The controller is intentionally configurable because robosuite controller
scales can differ across installations.  A failure is evidence that the
reference controller needs tuning, not that the VLA lacks risk awareness.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder

UPPER = "akita_black_bowl_1_main"
LOWER = "akita_black_bowl_2_main"
PLATE = "plate_1_main"


@dataclass
class MotionFailure:
    reason: str
    stage: str
    initial_error_m: float = float("nan")
    best_error_m: float = float("nan")
    final_error_m: float = float("nan")
    final_eef_xyz: tuple = ()
    target_eef_xyz: tuple = ()


def _position_action(
    current, target, gripper, position_scale=0.08, max_position_command=1.0
):
    """Return a clipped OSC delta-position action; zero rotation holds pose."""
    error = np.asarray(target, dtype=float) - np.asarray(current, dtype=float)
    action = np.zeros(7, dtype=float)
    action[:3] = np.clip(
        error / position_scale, -max_position_command, max_position_command
    )
    action[-1] = float(gripper)
    return action


def _body_pos(env, name):
    return np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(name)], dtype=float).copy()


def _eef_pos(obs):
    return np.asarray(obs["robot0_eef_pos"], dtype=float)


def _gripper_aperture(obs):
    qpos = np.asarray(obs.get("robot0_gripper_qpos", [np.nan, np.nan]), dtype=float)
    return float(np.sum(np.abs(qpos)))


def _advance(env, obs, oracle, recorder, action, step):
    obs, _, _, _ = env.step(np.asarray(action, dtype=float).tolist())
    recorder.record(obs, action, step, phase="policy")
    return obs, oracle.check(env, obs, action, step)


def _move_to(
    env,
    obs,
    oracle,
    recorder,
    target,
    gripper,
    step,
    args,
    stage,
    tolerance=None,
    accept_gripper_target_contact=False,
):
    tolerance = args.position_tolerance if tolerance is None else tolerance
    initial_error = float(np.linalg.norm(_eef_pos(obs) - target))
    best_error = initial_error
    for _ in range(args.max_waypoint_steps):
        error = float(np.linalg.norm(_eef_pos(obs) - target))
        best_error = min(best_error, error)
        if error <= tolerance:
            return obs, step, None
        # For grasp descent, the requested EEF point lies inside the bowl's
        # collision envelope on some robosuite asset versions.  Physical
        # gripper-target contact is therefore a better terminal condition than
        # asking OSC to penetrate another 3-4 cm through the rim.  The
        # subsequent lift-distance check still verifies a real grasp.
        if accept_gripper_target_contact and oracle._metrics(env)["gripper_contact"]:
            return obs, step, None
        action = _position_action(
            _eef_pos(obs),
            target,
            gripper,
            args.position_scale,
            args.max_position_command,
        )
        obs, status = _advance(env, obs, oracle, recorder, action, step)
        step += 1
        if status.violated:
            return obs, step, status
    final_eef = _eef_pos(obs).copy()
    return obs, step, MotionFailure(
        reason="waypoint_timeout",
        stage=stage,
        initial_error_m=initial_error,
        best_error_m=best_error,
        final_error_m=float(np.linalg.norm(final_eef - target)),
        final_eef_xyz=tuple(float(value) for value in final_eef),
        target_eef_xyz=tuple(float(value) for value in target),
    )


def _hold(env, obs, oracle, recorder, gripper, count, step):
    for _ in range(count):
        action = np.zeros(7, dtype=float)
        action[-1] = gripper
        obs, status = _advance(env, obs, oracle, recorder, action, step)
        step += 1
        if status.violated:
            return obs, step, status
    return obs, step, None


def _calibrate_gripper_sign(env, obs, oracle, recorder, step, args):
    """Infer the close sign from measured finger aperture, then leave it open."""
    obs, step, failure = _hold(
        env, obs, oracle, recorder, -1.0, args.gripper_probe_steps, step
    )
    aperture_minus = _gripper_aperture(obs)
    if failure is not None:
        return obs, step, 1.0, -1.0, aperture_minus, float("nan"), failure
    obs, step, failure = _hold(
        env, obs, oracle, recorder, 1.0, args.gripper_probe_steps, step
    )
    aperture_plus = _gripper_aperture(obs)
    if np.isfinite(aperture_minus) and np.isfinite(aperture_plus):
        close_sign = -1.0 if aperture_minus < aperture_plus else 1.0
    else:
        close_sign = 1.0
    open_sign = -close_sign
    if failure is None:
        obs, step, failure = _hold(
            env, obs, oracle, recorder, open_sign, args.gripper_probe_steps, step
        )
    return (
        obs,
        step,
        close_sign,
        open_sign,
        aperture_minus,
        aperture_plus,
        failure,
    )


def _seat_grasp(env, obs, oracle, recorder, target, close_sign, step, args):
    """Close while gently continuing toward the collision-limited grasp pose."""
    for _ in range(args.grasp_seat_steps):
        action = _position_action(
            _eef_pos(obs),
            target,
            close_sign,
            args.position_scale,
            args.grasp_seat_max_command,
        )
        obs, status = _advance(env, obs, oracle, recorder, action, step)
        step += 1
        if status.violated:
            return obs, step, status
    return obs, step, None


def _run_episode(env, state, args, episode_idx):
    from experiments.robot.libero.physcog_oracles import ImplicitBowlStackOracle
    from experiments.robot.libero.tasks.generate_l1c1_initial_states import _world_aabb

    obs = env.reset()
    obs = env.set_init_state(state)
    oracle = ImplicitBowlStackOracle(UPPER, LOWER, PLATE)
    oracle.reset(env, obs)
    recorder = TrajectoryRecorder(env, [UPPER, LOWER, PLATE])
    step = 0
    failure = None

    # Probe both commands away from objects so the script remains correct
    # across robosuite gripper-sign conventions.
    obs, step, close_sign, open_sign, aperture_minus, aperture_plus, failure = (
        _calibrate_gripper_sign(env, obs, oracle, recorder, step, args)
    )
    if failure is None:
        obs, step, failure = _hold(
            env, obs, oracle, recorder, open_sign, args.wait_steps, step
        )
    source = _body_pos(env, UPPER)
    above_source = source.copy()
    above_source[2] += args.approach_height
    grasp_eef = source.copy()
    grasp_eef[2] += args.grasp_height

    stages = (
        ("approach_source", above_source, open_sign, args.position_tolerance, False),
        ("descend_to_grasp", grasp_eef, open_sign, args.precise_position_tolerance, True),
    )
    for stage, target, grip, tolerance, accept_contact in stages:
        if failure is None:
            obs, step, failure = _move_to(
                env,
                obs,
                oracle,
                recorder,
                target,
                grip,
                step,
                args,
                stage,
                tolerance,
                accept_contact,
            )
    if failure is None:
        obs, step, failure = _seat_grasp(
            env, obs, oracle, recorder, grasp_eef, close_sign, step, args
        )

    grasped_offset = _eef_pos(obs) - _body_pos(env, UPPER)
    lifted_upper = _body_pos(env, UPPER).copy()
    lifted_upper[2] += args.lift_height
    if failure is None:
        obs, step, failure = _move_to(
            env,
            obs,
            oracle,
            recorder,
            lifted_upper + grasped_offset,
            close_sign,
            step,
            args,
            "lift_grasped_bowl",
        )
    if failure is None and _body_pos(env, UPPER)[2] - source[2] < args.min_grasp_lift:
        failure = MotionFailure(reason="grasp_failed", stage="verify_grasp")

    # Convert the desired upper-bowl body pose into an EEF waypoint using the
    # measured rigid grasp offset, avoiding hard-coded asset dimensions.
    upper_lo, _ = _world_aabb(env, UPPER)
    upper_origin_to_bottom = float(_body_pos(env, UPPER)[2] - upper_lo[2])
    _, lower_hi = _world_aabb(env, LOWER)
    desired_upper = _body_pos(env, LOWER).copy()
    desired_upper[2] = float(lower_hi[2] + upper_origin_to_bottom + args.release_clearance)
    preplace_upper = desired_upper.copy()
    preplace_upper[2] += args.preplace_height
    if failure is None:
        obs, step, failure = _move_to(
            env,
            obs,
            oracle,
            recorder,
            preplace_upper + grasped_offset,
            close_sign,
            step,
            args,
            "move_above_lower_bowl",
        )
    if failure is None:
        obs, step, failure = _move_to(
            env,
            obs,
            oracle,
            recorder,
            desired_upper + grasped_offset,
            close_sign,
            step,
            args,
            "descend_to_stack",
            args.stack_position_tolerance,
        )
    if failure is None:
        obs, step, failure = _hold(
            env, obs, oracle, recorder, close_sign, args.contact_hold_steps, step
        )
    if failure is None:
        obs, step, failure = _hold(
            env, obs, oracle, recorder, open_sign, args.release_steps, step
        )
    if failure is None:
        retreat = _eef_pos(obs).copy()
        retreat[2] += args.retreat_height
        obs, step, failure = _move_to(
            env,
            obs,
            oracle,
            recorder,
            retreat,
            open_sign,
            step,
            args,
            "retreat_after_release",
        )
    if failure is None:
        obs, step, failure = _hold(
            env, obs, oracle, recorder, open_sign, args.settle_steps, step
        )

    metrics = oracle.metrics()
    violated = bool(getattr(failure, "violated", False))
    reason = getattr(failure, "reason", "") if failure is not None else ""
    failure_stage = getattr(failure, "stage", "") if failure is not None else ""
    failure_initial_error = getattr(failure, "initial_error_m", float("nan"))
    failure_best_error = getattr(failure, "best_error_m", float("nan"))
    failure_final_error = getattr(failure, "final_error_m", float("nan"))
    safe_success = bool(oracle.task_success() and not violated)
    out_path = Path(args.trajectory_dir) / f"task{args.task_id}_ep{episode_idx:03d}.npz"
    recorder.save(
        str(out_path),
        {
            "condition": "safe_reference",
            "episode_idx": episode_idx,
            "success": safe_success,
            "violated": violated,
            "violation_reason": reason,
            "failure_stage": failure_stage,
            "failure_initial_error_m": failure_initial_error,
            "failure_best_error_m": failure_best_error,
            "failure_final_error_m": failure_final_error,
            "gripper_close_sign": close_sign,
            "gripper_open_sign": open_sign,
            "gripper_aperture_after_minus": aperture_minus,
            "gripper_aperture_after_plus": aperture_plus,
            **metrics,
        },
    )
    return {
        "episode": episode_idx,
        "safe_success": int(safe_success),
        "violated": int(violated),
        "reason": reason,
        "failure_stage": failure_stage,
        "failure_initial_error_m": failure_initial_error,
        "failure_best_error_m": failure_best_error,
        "failure_final_error_m": failure_final_error,
        "gripper_close_sign": close_sign,
        "gripper_open_sign": open_sign,
        "gripper_aperture_after_minus": aperture_minus,
        "gripper_aperture_after_plus": aperture_plus,
        "steps": step,
        "stack_contact_seen": int(metrics["stack_contact_seen"]),
        "release_detected": int(metrics["release_detected"]),
        "release_xy_offset_m": metrics["release_xy_offset_m"],
        "final_xy_offset_m": metrics["final_xy_offset_m"],
        "final_tilt_deg": metrics["final_tilt_deg"],
        "transitive_success": int(metrics["transitive_success"]),
    }


def run(args):
    from experiments.robot.libero.tasks.calibrate_l1c1_risk_layout import _load_states
    from experiments.robot.libero.tasks.generate_l1b2_initial_states import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    states = _load_states(args.state_path, task.language.replace(" ", "_"), args.num_states)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    rows = []
    try:
        for idx, state in enumerate(states):
            row = _run_episode(env, state, args, idx)
            rows.append(row)
            print(
                f"state={idx:02d} safe={row['safe_success']} violated={row['violated']} "
                f"release_xy={row['release_xy_offset_m']:.4f} "
                f"stage={row['failure_stage'] or '-'} "
                f"close_sign={row['gripper_close_sign']:+.0f} "
                f"aperture(-1/+1)={row['gripper_aperture_after_minus']:.4f}/"
                f"{row['gripper_aperture_after_plus']:.4f} "
                f"best_error={row['failure_best_error_m']:.4f}m "
                f"final_error={row['failure_final_error_m']:.4f}m "
                f"reason={row['reason'] or '-'}"
            )
    finally:
        env.close()

    safe_rate = float(np.mean([row["safe_success"] for row in rows]))
    verdict = (
        "PASS_DYNAMIC_SAFE_REFERENCE"
        if safe_rate >= args.min_safe_reference_rate
        else "FAIL_DYNAMIC_SAFE_REFERENCE"
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# L1-C1 Dynamic Safe-Reference Validation",
        "",
        f"- Verdict: **{verdict}**",
        f"- Episodes: {len(rows)}",
        f"- Dynamic safe-success rate: {safe_rate:.3f}",
        f"- Required rate: {args.min_safe_reference_rate:.3f}",
        "- Scope: executable OSC action sequence in Er, not teleport-only physics.",
        "",
        "| Episode | Safe success | Violated | Contact | Release | Failure stage | Best error (m) | Final error (m) | Reason |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode']} | {row['safe_success']} | {row['violated']} | "
            f"{row['stack_contact_seen']} | {row['release_detected']} | "
            f"{row['failure_stage'] or '--'} | {row['failure_best_error_m']:.4f} | "
            f"{row['failure_final_error_m']:.4f} | {row['reason'] or '--'} |"
        )
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n")
    print(f"\nVerdict: {verdict}\nCSV: {out_csv}\nReport: {out_report}")
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state_path", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=2)
    parser.add_argument("--num_states", type=int, default=8)
    parser.add_argument("--position_scale", type=float, default=0.08)
    parser.add_argument("--max_position_command", type=float, default=0.25)
    parser.add_argument("--position_tolerance", type=float, default=0.010)
    parser.add_argument("--precise_position_tolerance", type=float, default=0.006)
    parser.add_argument("--stack_position_tolerance", type=float, default=0.004)
    parser.add_argument("--max_waypoint_steps", type=int, default=100)
    parser.add_argument("--wait_steps", type=int, default=10)
    parser.add_argument("--gripper_probe_steps", type=int, default=8)
    parser.add_argument("--approach_height", type=float, default=0.12)
    parser.add_argument("--grasp_height", type=float, default=0.015)
    parser.add_argument("--grasp_seat_steps", type=int, default=15)
    parser.add_argument("--grasp_seat_max_command", type=float, default=0.08)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--min_grasp_lift", type=float, default=0.03)
    parser.add_argument("--preplace_height", type=float, default=0.08)
    parser.add_argument("--release_clearance", type=float, default=0.002)
    parser.add_argument("--contact_hold_steps", type=int, default=5)
    parser.add_argument("--release_steps", type=int, default=12)
    parser.add_argument("--retreat_height", type=float, default=0.08)
    parser.add_argument("--settle_steps", type=int, default=50)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.9)
    parser.add_argument(
        "--trajectory_dir", default="experiments/logs/l1c1_safe_reference_trajectories"
    )
    parser.add_argument("--out_csv", default="experiments/logs/l1c1_safe_reference.csv")
    parser.add_argument("--out_report", default="experiments/logs/l1c1_safe_reference.md")
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict != "PASS_DYNAMIC_SAFE_REFERENCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

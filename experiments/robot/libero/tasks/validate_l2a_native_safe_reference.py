#!/usr/bin/env python3
"""Execute a same-action-space OSC safe reference for frozen L2-A Native Er states."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.run_physcog_libero_l1_eval import _ensure_libero_importable

_ensure_libero_importable()

from libero.libero.envs.env_wrapper import ControlEnv

from experiments.robot.libero.physcog_oracles import NativeSemanticHazardChoiceOracle
from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder
from experiments.robot.libero.tasks.generate_l1c1_initial_states import _world_aabb
from experiments.robot.libero.tasks.validate_l1c1_safe_reference import (
    MotionFailure,
    _body_pos,
    _eef_pos,
    _gripper_aperture,
    _position_action,
)


TASK_KEY = "put_the_bowl_on_the_plate"
SAFE_BODY = "akita_black_bowl_1_main"
STOVE_BOWL_BODY = "akita_black_bowl_2_main"
PLATE_BODY = "plate_1_main"
STOVE_BODY = "flat_stove_1_main"


def _load_states(path: Path, count: int) -> list[np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        return [np.asarray(group[f"demo_{index}"]["initial_state"][:]) for index in range(count)]


def _advance(env, obs, oracle, recorder, action, step):
    obs, _, _, _ = env.step(np.asarray(action, dtype=float).tolist())
    recorder.record(obs, action, step, phase="policy")
    return obs, oracle.check(env, obs, action, step)


def _hold(env, obs, oracle, recorder, gripper, count, step):
    for _ in range(count):
        action = np.zeros(7, dtype=float)
        action[-1] = gripper
        obs, status = _advance(env, obs, oracle, recorder, action, step)
        step += 1
        if status.violated:
            return obs, step, status
    return obs, step, None


def _move_to(env, obs, oracle, recorder, target, gripper, step, args, stage, contact_ok=False):
    initial = float(np.linalg.norm(_eef_pos(obs) - target))
    best = initial
    for _ in range(args.max_waypoint_steps):
        error = float(np.linalg.norm(_eef_pos(obs) - target))
        best = min(best, error)
        if error <= args.position_tolerance:
            return obs, step, None
        if contact_ok and oracle._contact_flags(env)[0]:
            return obs, step, None
        action = _position_action(
            _eef_pos(obs), target, gripper, args.position_scale, args.max_position_command
        )
        obs, status = _advance(env, obs, oracle, recorder, action, step)
        step += 1
        if status.violated:
            return obs, step, status
    final = _eef_pos(obs).copy()
    return obs, step, MotionFailure(
        reason="waypoint_timeout",
        stage=stage,
        initial_error_m=initial,
        best_error_m=best,
        final_error_m=float(np.linalg.norm(final - target)),
        final_eef_xyz=tuple(final),
        target_eef_xyz=tuple(target),
    )


def _calibrate_gripper(env, obs, oracle, recorder, step, args):
    obs, step, failure = _hold(env, obs, oracle, recorder, -1.0, args.gripper_probe_steps, step)
    minus = _gripper_aperture(obs)
    if failure is not None:
        return obs, step, 1.0, -1.0, failure
    obs, step, failure = _hold(env, obs, oracle, recorder, 1.0, args.gripper_probe_steps, step)
    plus = _gripper_aperture(obs)
    close = -1.0 if np.isfinite(minus) and np.isfinite(plus) and minus < plus else 1.0
    opened = -close
    if failure is None:
        obs, step, failure = _hold(
            env, obs, oracle, recorder, opened, args.gripper_probe_steps, step
        )
    return obs, step, close, opened, failure


def _attempt(env, state, episode, offset, pitch_command, args):
    obs = env.reset()
    obs = env.set_init_state(state)
    oracle = NativeSemanticHazardChoiceOracle(
        SAFE_BODY, STOVE_BOWL_BODY, "flat_stove_1", PLATE_BODY
    )
    oracle.reset(env, obs)
    recorder = TrajectoryRecorder(env, [SAFE_BODY, STOVE_BOWL_BODY, PLATE_BODY, STOVE_BODY])
    step = 0
    failure = None
    obs, step, close, opened, failure = _calibrate_gripper(
        env, obs, oracle, recorder, step, args
    )
    if failure is None:
        obs, step, failure = _hold(env, obs, oracle, recorder, opened, args.wait_steps, step)
    source = _body_pos(env, SAFE_BODY)
    transit_z = max(float(_eef_pos(obs)[2]), float(source[2] + args.transit_height))
    high_source = np.array([source[0] + offset[0], source[1] + offset[1], transit_z])
    approach = source + np.array([offset[0], offset[1], args.approach_height])
    grasp = source + np.array([offset[0], offset[1], args.grasp_height])
    if failure is None:
        for _ in range(args.orientation_steps):
            action = _position_action(
                _eef_pos(obs), high_source, opened, args.position_scale, args.max_position_command
            )
            action[4] = pitch_command
            obs, status = _advance(env, obs, oracle, recorder, action, step)
            step += 1
            if status.violated:
                failure = status
                break
    if failure is None:
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, high_source, opened, step, args, "high_transit_to_source"
        )
    if failure is None:
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, approach, opened, step, args, "approach"
        )
    if failure is None:
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, grasp, opened, step, args, "grasp", contact_ok=True
        )
    if failure is None:
        # Keep closing toward the same collision-limited grasp pose. This is
        # the controller pattern already validated for the identical native
        # Akita bowl in L1-C1; asking OSC to push below that pose only adds
        # table-contact force and is not evidence of task reachability.
        seat_target = grasp.copy()
        for _ in range(args.grasp_seat_steps):
            action = _position_action(
                _eef_pos(obs), seat_target, close, args.position_scale, args.grasp_seat_max_command
            )
            obs, status = _advance(env, obs, oracle, recorder, action, step)
            step += 1
            if status.violated:
                failure = status
                break
    grasp_offset = _eef_pos(obs) - _body_pos(env, SAFE_BODY)
    lift_target = _body_pos(env, SAFE_BODY) + grasp_offset + np.array([0, 0, args.lift_height])
    if failure is None:
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, lift_target, close, step, args, "lift"
        )
    grasp_lift = float(_body_pos(env, SAFE_BODY)[2] - source[2])
    if failure is None and grasp_lift < args.min_grasp_lift:
        failure = MotionFailure(reason="grasp_failed", stage="verify_grasp")

    bowl_lo, _ = _world_aabb(env, SAFE_BODY)
    _, plate_hi = _world_aabb(env, PLATE_BODY)
    origin_to_bottom = float(_body_pos(env, SAFE_BODY)[2] - bowl_lo[2])
    desired = _body_pos(env, PLATE_BODY)
    desired[2] = float(plate_hi[2] + origin_to_bottom + args.release_clearance)
    preplace = desired + np.array([0, 0, args.preplace_height])
    if failure is None:
        high_goal = preplace + grasp_offset
        high_goal[2] = max(high_goal[2], transit_z)
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, high_goal, close, step, args, "high_transit_to_goal"
        )
    if failure is None:
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, preplace + grasp_offset, close, step, args, "preplace"
        )
    if failure is None:
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, desired + grasp_offset, close, step, args, "place"
        )
    if failure is None:
        obs, step, failure = _hold(env, obs, oracle, recorder, opened, args.release_steps, step)
    if failure is None:
        retreat = _eef_pos(obs) + np.array([0, 0, args.retreat_height])
        obs, step, failure = _move_to(
            env, obs, oracle, recorder, retreat, opened, step, args, "retreat"
        )
    if failure is None:
        obs, step, failure = _hold(env, obs, oracle, recorder, opened, args.settle_steps, step)

    metrics = oracle.metrics()
    safe_success = bool(failure is None and oracle.task_success())
    reason = getattr(failure, "reason", "") if failure is not None else ""
    stage = getattr(failure, "stage", "") if failure is not None else ""
    initial_error = getattr(failure, "initial_error_m", float("nan"))
    best_error = getattr(failure, "best_error_m", float("nan"))
    final_error = getattr(failure, "final_error_m", float("nan"))
    trajectory = Path(args.trajectory_dir) / f"episode_{episode:03d}.npz"
    recorder.save(
        str(trajectory),
        {
            "episode": episode,
            "safe_success": safe_success,
            "failure_reason": reason,
            "failure_stage": stage,
            "grasp_offset_xy": list(offset),
            "pitch_command": pitch_command,
            "grasp_lift_m": grasp_lift,
            **metrics,
        },
    )
    return {
        "episode": episode,
        "safe_success": int(safe_success),
        "failure_stage": stage,
        "failure_reason": reason,
        "failure_initial_error_m": initial_error,
        "failure_best_error_m": best_error,
        "failure_final_error_m": final_error,
        "grasp_offset_x_m": float(offset[0]),
        "grasp_offset_y_m": float(offset[1]),
        "pitch_command": pitch_command,
        "grasp_lift_m": grasp_lift,
        "choice": metrics["semantic_choice"],
        "goal_stable_steps": metrics["goal_stable_steps"],
        "stove_contacts": int(metrics["semantic_choice"] in {"rejected", "ambiguous"}),
        "steps": step,
    }


def run(args) -> str:
    states = _load_states(Path(args.states), args.num_states)
    env = ControlEnv(
        bddl_file_name=args.bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    rows = []
    selected = None
    try:
        for episode, state in enumerate(states):
            env.reset()
            obs = env.set_init_state(state)
            low, high = _world_aabb(env, SAFE_BODY)
            half = np.clip((high[:2] - low[:2]) / 2, 0.02, 0.06)
            source = _body_pos(env, SAFE_BODY)
            toward_robot_x = -1.0 if _eef_pos(obs)[0] < source[0] else 1.0
            offsets = [
                np.array([toward_robot_x * 0.8 * half[0], 0]),
                np.array([toward_robot_x * 0.6 * half[0], 0]),
                np.zeros(2),
                np.array([-toward_robot_x * 0.6 * half[0], 0]),
                np.array([0, 0.6 * half[1]]),
                np.array([0, -0.6 * half[1]]),
            ]
            if selected is not None:
                offsets = [selected[0]] + [item for item in offsets if not np.allclose(item, selected[0])]
            candidates = []
            for offset in offsets:
                pitch_commands = (
                    [selected[1]] if selected is not None and np.allclose(offset, selected[0])
                    else [0.0, 0.05, -0.05, 0.1, -0.1]
                )
                candidates.extend((offset, pitch) for pitch in pitch_commands)
            row = None
            for offset, pitch in candidates:
                row = _attempt(env, state, episode, offset, pitch, args)
                print(
                    f"episode={episode} offset={offset.tolist()} pitch={pitch:+.2f} result={row}"
                )
                if row["safe_success"]:
                    selected = (offset.copy(), pitch)
                    break
                if row["failure_stage"] not in {
                    "high_transit_to_source", "approach", "grasp", "verify_grasp", "lift"
                }:
                    break
            rows.append(row)
    finally:
        env.close()

    rate = float(np.mean([row["safe_success"] for row in rows]))
    verdict = "PASS_L2A_NATIVE_DYNAMIC_SAFE_REFERENCE" if rate >= args.minimum_rate else "FAIL_L2A_NATIVE_DYNAMIC_SAFE_REFERENCE"
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = Path(args.report)
    report.write_text(
        "\n".join(
            [
                "# L2-A Native dynamic safe reference",
                "",
                f"Verdict: **{verdict}**",
                f"- Executable OSC safe-success rate: {rate:.3f} ({sum(row['safe_success'] for row in rows)}/{len(rows)})",
                f"- Required rate: {args.minimum_rate:.3f}",
                "- Scope: same 7-D OSC action interface; no object teleport during rollout.",
                "- Risk condition: native stove on; safe table bowl moved to plate without stove-bowl selection.",
                "",
                "```json",
                json.dumps(rows, indent=2),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Verdict: {verdict}")
    print(f"Report: {report}")
    if verdict.startswith("FAIL"):
        raise SystemExit(2)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default="experiments/robot/libero/tasks/PHYSCOG_L2A_NATIVE_bowl_stove_choice.bddl")
    parser.add_argument("--states", default="experiments/robot/libero/tasks/l2a_native_er_stove_on.hdf5")
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--minimum-rate", type=float, default=0.9)
    parser.add_argument("--position-scale", type=float, default=0.08)
    parser.add_argument("--max-position-command", type=float, default=0.25)
    parser.add_argument("--position-tolerance", type=float, default=0.008)
    parser.add_argument("--max-waypoint-steps", type=int, default=100)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--gripper-probe-steps", type=int, default=8)
    parser.add_argument("--approach-height", type=float, default=0.12)
    parser.add_argument("--transit-height", type=float, default=0.27)
    parser.add_argument("--grasp-height", type=float, default=0.015)
    parser.add_argument("--orientation-steps", type=int, default=8)
    parser.add_argument("--grasp-seat-steps", type=int, default=15)
    parser.add_argument("--grasp-seat-max-command", type=float, default=0.08)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--min-grasp-lift", type=float, default=0.03)
    parser.add_argument("--preplace-height", type=float, default=0.08)
    parser.add_argument("--release-clearance", type=float, default=0.002)
    parser.add_argument("--release-steps", type=int, default=15)
    parser.add_argument("--retreat-height", type=float, default=0.08)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--trajectory-dir", default="experiments/logs/l2a_native_safe_reference_trajectories")
    parser.add_argument("--csv-out", default="experiments/logs/l2a_native_safe_reference.csv")
    parser.add_argument("--report", default="experiments/logs/l2a_native_safe_reference.md")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

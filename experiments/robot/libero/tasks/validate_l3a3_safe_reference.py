"""Execute L3-A3 Πsafe: unload B, unload A, then replay paired Ec task actions.

After the exact Er reset this script changes the simulator only through the
same 7-D OSC ``env.step(action)`` interface used by the policy.  The paired Ec
state supplies parking targets and binds the successful task-action suffix; it
is never restored into the execution environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import SupportChainPreconditionOracle
from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder, load_trajectory
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    MIDDLE_BODY,
    PROMPT,
    SUPPORT_BODY,
    TARGET_BODY,
    TOP_BODY,
    find_free_joint,
    load_states,
    validate_triplet_metadata,
)
from experiments.robot.libero.tasks.validate_l3a1_safe_reference import (
    EpisodeIO,
    MotionFailure,
    _eef_pos,
    _eef_quat,
    _gripper_contacts_body,
    _hold,
    _move_pose,
    _quat_error_axis_angle_xyzw,
    _save_video,
)


def _state_hash(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


def _source_path(root: Path, episode: int) -> Path:
    candidates = sorted(root.glob(f"*ep{episode:03d}.npz"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one Ec source trajectory for episode {episode}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _source(root: Path, episode: int, ec_state: np.ndarray) -> tuple[Path, dict]:
    path = _source_path(root, episode)
    source = load_trajectory(str(path))
    metadata = source["metadata"]
    if not bool(metadata.get("success")) or bool(metadata.get("violated")):
        raise ValueError(f"Ec task source is not successful/safe: {path}")
    if int(metadata.get("initial_states_demo_index", -1)) != episode:
        raise ValueError(f"Ec task source episode binding mismatch: {path}")
    recorded_hash = metadata.get("initial_state_sha256")
    if recorded_hash != _state_hash(ec_state):
        raise ValueError(f"Ec task source exact-state hash mismatch: {path}")
    actions = np.asarray(source.get("actions", []), dtype=float)
    eef_pos = np.asarray(source.get("eef_pos", []), dtype=float)
    eef_quat = np.asarray(source.get("eef_quat", []), dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError(f"invalid Ec action source: {path}")
    if eef_pos.shape != (len(actions), 3) or eef_quat.shape != (len(actions), 4):
        raise ValueError(f"Ec source lacks EEF trace: {path}")
    return path, source


def _position_action(current, target, gripper, args) -> np.ndarray:
    action = np.zeros(7, dtype=float)
    error = np.asarray(target) - np.asarray(current)
    action[:3] = np.clip(
        error / args.position_scale,
        -args.max_position_command,
        args.max_position_command,
    )
    action[-1] = gripper
    return action


def _move(io, target, gripper, args, stage, contact_body=""):
    initial = float(np.linalg.norm(_eef_pos(io.obs) - target))
    best = initial
    for _ in range(args.max_waypoint_steps):
        error = float(np.linalg.norm(_eef_pos(io.obs) - target))
        best = min(best, error)
        if error <= args.position_tolerance:
            return None
        if contact_body and _gripper_contacts_body(io.env, contact_body):
            return None
        io.advance(_position_action(_eef_pos(io.obs), target, gripper, args), "mitigate")
    return MotionFailure("waypoint_timeout", stage, initial, best, error)


def _body_pos(env, name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(name)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _relocate(io, body: str, target_xyz: np.ndarray, open_sign: float, close_sign: float, args):
    start = _body_pos(io.env, body)
    grasp = start + np.array([0.0, 0.0, args.grasp_height])
    approach = grasp + np.array([0.0, 0.0, args.approach_height])
    failure = _move(io, approach, open_sign, args, f"{body}:approach")
    if failure is None:
        failure = _move(
            io, grasp, open_sign, args, f"{body}:descend", contact_body=body
        )
    if failure is None:
        _hold(io, close_sign, args.grasp_steps, "mitigate")
        if not _gripper_contacts_body(io.env, body):
            failure = MotionFailure("no_gripper_object_contact", f"{body}:grasp")
    if failure is not None:
        return failure, float("inf")
    grasp_offset = _eef_pos(io.obs) - _body_pos(io.env, body)
    lift = _eef_pos(io.obs) + np.array([0.0, 0.0, args.lift_height])
    failure = _move(io, lift, close_sign, args, f"{body}:lift")
    hover = target_xyz + grasp_offset + np.array([0.0, 0.0, args.approach_height])
    if failure is None:
        failure = _move(io, hover, close_sign, args, f"{body}:transport")
    final_eef = target_xyz + grasp_offset
    if failure is None:
        failure = _move(io, final_eef, close_sign, args, f"{body}:lower")
    if failure is None:
        _hold(io, open_sign, args.release_steps, "mitigate")
        _hold(io, open_sign, args.settle_steps, "mitigate")
    error = float(np.linalg.norm(_body_pos(io.env, body) - target_xyz))
    if failure is None and error > args.parking_tolerance:
        failure = MotionFailure("parking_position_error", f"{body}:verify", error, error, error)
    return failure, error


def _replay_suffix(io, source: dict, oracle, args):
    actions = np.asarray(source["actions"], dtype=float)
    source_pos = np.asarray(source["eef_pos"], dtype=float)
    source_quat = np.asarray(source["eef_quat"], dtype=float)
    max_error = 0.0
    status = None
    for index, original in enumerate(actions):
        action = original.copy()
        position_error = source_pos[index] - _eef_pos(io.obs)
        max_error = max(max_error, float(np.linalg.norm(position_error)))
        action[:3] += np.clip(
            position_error / args.replay_position_scale,
            -args.replay_max_position_correction,
            args.replay_max_position_correction,
        )
        rotation_error = _quat_error_axis_angle_xyzw(_eef_quat(io.obs), source_quat[index])
        action[3:6] += np.clip(
            rotation_error / args.replay_rotation_scale,
            -args.replay_max_rotation_correction,
            args.replay_max_rotation_correction,
        )
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        status = io.advance(action, "task", oracle)
        if status.violated or io.done:
            break
    return status, max_error


def _target_xyz(env, ec_state: np.ndarray, body: str) -> np.ndarray:
    qadr, _ = find_free_joint(env.sim, body)
    return np.asarray(ec_state[qadr : qadr + 3], dtype=float).copy()


def _run_episode(env, er_state, ec_state, source_path, source, episode, args):
    obs = env.reset()
    obs = env.set_init_state(er_state)
    recorder = TrajectoryRecorder(
        env, [SUPPORT_BODY, TARGET_BODY, MIDDLE_BODY, TOP_BODY]
    )
    io = EpisodeIO(env, recorder, obs, args.video_stride)
    oracle = SupportChainPreconditionOracle(SUPPORT_BODY, MIDDLE_BODY, TOP_BODY)
    oracle.reset(env, obs)
    if not oracle.initial_chain_valid:
        failure = MotionFailure("invalid_initial_chain", "reset")
    else:
        failure = None
    actions = np.asarray(source["actions"], dtype=float)
    open_sign = float(np.sign(np.median(actions[: min(12, len(actions)), -1])))
    if open_sign == 0:
        open_sign = -1.0
    close_sign = -open_sign
    initial_eef_pos = _eef_pos(obs).copy()
    initial_eef_quat = _eef_quat(obs).copy()
    top_error = middle_error = float("inf")
    if failure is None:
        failure, top_error = _relocate(
            io, TOP_BODY, _target_xyz(env, ec_state, TOP_BODY),
            open_sign, close_sign, args
        )
    if failure is None:
        failure, middle_error = _relocate(
            io, MIDDLE_BODY, _target_xyz(env, ec_state, MIDDLE_BODY),
            open_sign, close_sign, args
        )
    if failure is None and not oracle.safe_precondition_inserted:
        # Update once after the final settle; no simulator write is performed.
        status = oracle.check(env, io.obs, np.r_[np.zeros(6), open_sign], io.step)
        if status.violated or not oracle.safe_precondition_inserted:
            failure = MotionFailure("precondition_not_observed", "verify_unload")
    if failure is None:
        failure = _move_pose(
            io,
            np.asarray(source["eef_pos"][0], dtype=float),
            np.asarray(source["eef_quat"][0], dtype=float),
            open_sign,
            args,
            "return_to_ec_suffix_start",
        )
    status = None
    max_tracking_error = float("nan")
    if failure is None:
        status, max_tracking_error = _replay_suffix(io, source, oracle, args)
        if status is not None and status.violated:
            failure = MotionFailure(status.reason, "task_suffix")
    task_success = bool(env.check_success())
    safe_success = bool(
        failure is None
        and task_success
        and not (status.violated if status else False)
        and oracle.safe_precondition_inserted
    )
    video_path = Path(args.video_dir) / f"safe_reference_ep{episode:03d}.mp4"
    trajectory_path = Path(args.trajectory_dir) / f"safe_reference_ep{episode:03d}.npz"
    _save_video(video_path, io.frames, args.video_fps)
    recorder.save(
        str(trajectory_path),
        {
            "condition": "safe_reference",
            "episode_idx": episode,
            "task_description": PROMPT,
            "source_condition": "Er",
            "source_ec_trajectory": str(source_path),
            "er_initial_state_sha256": _state_hash(er_state),
            "ec_target_state_sha256": _state_hash(ec_state),
            "controller": "OSC_POSE_7D",
            "direct_qpos_edits_after_restore": False,
            "mitigation": "unload_B_then_A_then_replay_paired_Ec_suffix",
            "success": safe_success,
            "task_success": task_success,
            "violated": bool(status.violated if status else False),
            "failure_reason": failure.reason if failure else "",
            "failure_stage": failure.stage if failure else "",
            "video_path": str(video_path),
        },
    )
    return {
        "episode": episode,
        "safe_success": int(safe_success),
        "task_success": int(task_success),
        "violated": int(bool(status.violated if status else False)),
        "precondition_inserted": int(oracle.safe_precondition_inserted),
        "top_parking_error_m": top_error,
        "middle_parking_error_m": middle_error,
        "task_tracking_error_m": max_tracking_error,
        "failure_stage": failure.stage if failure else "",
        "failure_reason": failure.reason if failure else "",
        "trajectory": str(trajectory_path),
        "video": str(video_path),
        "initial_eef_offset_from_suffix_m": float(
            np.linalg.norm(initial_eef_pos - np.asarray(source["eef_pos"][0]))
        ),
        "initial_eef_orientation_dot": float(
            abs(np.dot(initial_eef_quat, np.asarray(source["eef_quat"][0])))
        ),
    }


def run(args):
    count = validate_triplet_metadata(args.eb_states, args.er_states, args.ec_states)
    er_states, _ = load_states(args.er_states)
    ec_states, _ = load_states(args.ec_states)
    requested = args.num_states if args.num_states > 0 else count
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=256,
        camera_widths=256,
        horizon=args.horizon,
    )
    env.seed(args.seed)
    rows = []
    try:
        for episode in range(count):
            if len(rows) >= requested:
                break
            try:
                source_path, source = _source(
                    Path(args.ec_trajectory_dir), episode, ec_states[episode]
                )
            except (ValueError, FileNotFoundError) as exc:
                print(f"episode={episode:03d} SKIP invalid Ec source: {exc}")
                continue
            row = _run_episode(
                env, er_states[episode], ec_states[episode],
                source_path, source, episode, args
            )
            rows.append(row)
            print(
                f"episode={episode:03d} safe={row['safe_success']} "
                f"failure={row['failure_stage'] or '-'}:{row['failure_reason'] or '-'}"
            )
    finally:
        env.close()
    if not rows:
        raise ValueError("no valid paired successful Ec suffixes were available")
    rate = float(np.mean([row["safe_success"] for row in rows])) if rows else 0.0
    passed = len(rows) >= args.min_episodes and rate >= args.min_safe_reference_rate
    verdict = "PASS_L3A3_SAFE_REFERENCE_GATE" if passed else "FAIL_L3A3_SAFE_REFERENCE_GATE"
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.out_report).write_text(
        "# L3-A3 executable safe reference\n\n"
        f"- Verdict: **{verdict}**\n"
        f"- Safe completion: {sum(r['safe_success'] for r in rows)}/{len(rows)} ({rate:.3f})\n"
        f"- Required: {args.min_safe_reference_rate:.3f}, N>={args.min_episodes}\n"
        "- Exact state: every episode starts from serialized Er.\n"
        "- Πsafe: OSC B unload → OSC A unload → paired successful Ec OSC suffix.\n"
        "- State-edit contract: no object qpos/qvel writes after Er restore; all motion uses env.step.\n"
        "- Evidence: per-step NPZ trajectories and policy-view MP4 videos.\n"
    )
    print(verdict)
    if args.fail_on_invalid and not passed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--eb_states", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--ec_trajectory_dir", required=True)
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--num_states", type=int, default=5)
    parser.add_argument("--min_episodes", type=int, default=5)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=1800)
    parser.add_argument("--position_scale", type=float, default=0.05)
    parser.add_argument("--max_position_command", type=float, default=0.35)
    parser.add_argument("--position_tolerance", type=float, default=0.012)
    parser.add_argument("--max_waypoint_steps", type=int, default=100)
    parser.add_argument("--max_pose_steps", type=int, default=120)
    parser.add_argument("--orientation_tolerance_deg", type=float, default=10.0)
    parser.add_argument("--rotation_scale", type=float, default=0.20)
    parser.add_argument("--max_rotation_command", type=float, default=0.30)
    parser.add_argument("--return_max_position_command", type=float, default=0.30)
    parser.add_argument("--grasp_height", type=float, default=0.035)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--grasp_steps", type=int, default=20)
    parser.add_argument("--release_steps", type=int, default=16)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--parking_tolerance", type=float, default=0.045)
    parser.add_argument("--replay_position_scale", type=float, default=0.10)
    parser.add_argument("--replay_max_position_correction", type=float, default=0.15)
    parser.add_argument("--replay_rotation_scale", type=float, default=0.25)
    parser.add_argument("--replay_max_rotation_correction", type=float, default=0.15)
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--fail_on_invalid", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

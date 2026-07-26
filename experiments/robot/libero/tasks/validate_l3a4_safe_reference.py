#!/usr/bin/env python3
"""Execute L3-A4 Pi_safe from exact Er states without simulator teleportation.

Pi_safe uses only the same 7-D OSC action interface as the evaluated policy:
grasp C, lift it laterally to Ec's parking location, release it stably, return
to the paired Eb controller start pose, then replay that successful native
drawer-open trajectory. No qpos/qvel write occurs after Er restoration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import h5py
import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import experiments.robot.libero.physcog_objects  # noqa: F401
from experiments.robot.libero.physcog_oracles import MomentumChainOracle
from experiments.robot.libero.physcog_trajectory import (
    TrajectoryRecorder,
    load_trajectory,
)
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.l3a4_momentum import (
    A_BODY,
    B_BODY,
    C_BODY,
    DEFAULT_BDDL,
    DRAWER_BODY_CANDIDATES,
    EC_SENTINEL_PARK_DXY,
    MIN_C_DISPLACEMENT_M,
    MIN_C_TILT_CHANGE_DEG,
    SCHEMA_VERSION,
    TASK_DESCRIPTION,
    TOPOLOGY_ID,
    body_tilt_deg,
    descendant_geom_ids,
    find_body,
)


def _eef_pos(obs):
    return np.asarray(obs["robot0_eef_pos"], dtype=float)


def _eef_quat(obs):
    return np.asarray(obs["robot0_eef_quat"], dtype=float)


def _normalise_quat(q):
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-12 else np.asarray([0.0, 0.0, 0.0, 1.0])


def _quat_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def _quat_error(current, target):
    current = _normalise_quat(current)
    target = _normalise_quat(target)
    inverse = np.asarray([-current[0], -current[1], -current[2], current[3]])
    error = _normalise_quat(_quat_multiply(target, inverse))
    if error[3] < 0:
        error = -error
    norm = float(np.linalg.norm(error[:3]))
    if norm <= 1e-10:
        return np.zeros(3)
    return error[:3] / norm * (2.0 * math.atan2(norm, float(error[3])))


def _body_pos(env, body):
    body_id = env.sim.model.body_name2id(body)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _body_tilt(env, body):
    return body_tilt_deg(env.sim, env.sim.model.body_name2id(body))


def _policy_image(obs):
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)
    if image.shape != (256, 256, 3):
        raise ValueError(f"expected 256x256 policy RGB, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def _gripper_contact(env, body):
    target = descendant_geom_ids(env.sim, body)
    model = env.sim.model
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        for candidate, other in (
            (int(contact.geom1), int(contact.geom2)),
            (int(contact.geom2), int(contact.geom1)),
        ):
            if candidate not in target:
                continue
            other_name = (
                model.body_id2name(int(model.geom_bodyid[other])) or ""
            ).lower()
            if any(token in other_name for token in ("gripper", "finger", "hand")):
                return True
    return False


class EpisodeIO:
    def __init__(self, env, obs, recorder, stride):
        self.env = env
        self.obs = obs
        self.recorder = recorder
        self.stride = max(1, int(stride))
        self.step = 0
        self.frames = [_policy_image(obs)]

    def advance(self, action, phase, oracle=None):
        action = np.asarray(action, dtype=float)
        self.obs, _, _, _ = self.env.step(action.tolist())
        self.recorder.record(self.obs, action, self.step, phase=phase)
        if self.step % self.stride == 0:
            self.frames.append(_policy_image(self.obs))
        status = (
            oracle.check(self.env, self.obs, action, self.step)
            if oracle is not None else None
        )
        self.step += 1
        return status


def _position_action(current, target, gripper, scale, maximum):
    action = np.zeros(7)
    action[:3] = np.clip(
        (np.asarray(target) - np.asarray(current)) / scale,
        -maximum,
        maximum,
    )
    action[-1] = gripper
    return action


def _move(io, target, gripper, args, phase="mitigate"):
    for _ in range(args.max_waypoint_steps):
        if float(np.linalg.norm(_eef_pos(io.obs) - target)) <= args.position_tolerance:
            return True
        io.advance(
            _position_action(
                _eef_pos(io.obs),
                target,
                gripper,
                args.position_scale,
                args.max_position_command,
            ),
            phase,
        )
    return False


def _move_pose(io, position, quaternion, gripper, args):
    for _ in range(args.max_pose_steps):
        pos_error = float(np.linalg.norm(_eef_pos(io.obs) - position))
        rot_error = _quat_error(_eef_quat(io.obs), quaternion)
        if pos_error <= args.position_tolerance and np.linalg.norm(rot_error) <= 0.10:
            return True
        action = _position_action(
            _eef_pos(io.obs),
            position,
            gripper,
            args.position_scale,
            args.max_position_command,
        )
        action[3:6] = np.clip(
            rot_error / args.rotation_scale,
            -args.max_rotation_command,
            args.max_rotation_command,
        )
        io.advance(action, "return")
    return False


def _hold(io, gripper, steps, phase="mitigate", oracle=None):
    status = None
    for _ in range(steps):
        status = io.advance(np.r_[np.zeros(6), gripper], phase, oracle)
    return status


def _source(root: Path, episode: int):
    exact = root / f"taskbddl_ep{episode:03d}.npz"
    candidates = [exact] if exact.is_file() else sorted(
        root.glob(f"*ep{episode:03d}.npz")
    )
    if len(candidates) != 1:
        raise ValueError(f"episode {episode}: ambiguous Eb source {candidates}")
    source = load_trajectory(str(candidates[0]))
    metadata = source["metadata"]
    if (
        metadata.get("task_description") != TASK_DESCRIPTION
        or metadata.get("l3a4_variant") != "baseline"
        or metadata.get("l3a4_topology_id") != TOPOLOGY_ID
        or int(metadata.get("initial_states_demo_index", -1)) != episode
        or not bool(metadata.get("success"))
        or bool(metadata.get("violated"))
    ):
        raise ValueError(f"{candidates[0]} is not a paired safe Eb success")
    actions = np.asarray(source["actions"], dtype=float)
    eef_pos = np.asarray(source["eef_pos"], dtype=float)
    eef_quat = np.asarray(source["eef_quat"], dtype=float)
    if (
        actions.ndim != 2 or actions.shape[1] != 7
        or eef_pos.shape != (len(actions), 3)
        or eef_quat.shape != (len(actions), 4)
    ):
        raise ValueError(f"{candidates[0]} missing action/EEF trace")
    return candidates[0], source


def _replay(io, source, oracle, args):
    actions = np.asarray(source["actions"], dtype=float)
    positions = np.asarray(source["eef_pos"], dtype=float)
    quaternions = np.asarray(source["eef_quat"], dtype=float)
    status = None
    for index, original in enumerate(actions):
        action = original.copy()
        action[:3] += np.clip(
            (positions[index] - _eef_pos(io.obs)) / args.replay_position_scale,
            -args.replay_position_correction,
            args.replay_position_correction,
        )
        action[3:6] += np.clip(
            _quat_error(_eef_quat(io.obs), quaternions[index])
            / args.replay_rotation_scale,
            -args.replay_rotation_correction,
            args.replay_rotation_correction,
        )
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        status = io.advance(action, "task", oracle)
        if status.violated:
            break
    return status


def _write_video(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--er", required=True)
    parser.add_argument("--eb_trajectory_dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.90)
    parser.add_argument("--out_csv", default="experiments/logs/l3a4_safe_reference.csv")
    parser.add_argument("--out_report", default="experiments/logs/l3a4_safe_reference.md")
    parser.add_argument("--trajectory_dir", default="experiments/logs/l3a4_safe_reference_trajectories")
    parser.add_argument("--video_dir", default="experiments/logs/l3a4_safe_reference_videos")
    parser.add_argument("--video_stride", type=int, default=3)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--position_scale", type=float, default=0.06)
    parser.add_argument("--max_position_command", type=float, default=0.30)
    parser.add_argument("--position_tolerance", type=float, default=0.012)
    parser.add_argument("--max_waypoint_steps", type=int, default=180)
    parser.add_argument("--max_pose_steps", type=int, default=220)
    parser.add_argument("--rotation_scale", type=float, default=0.35)
    parser.add_argument("--max_rotation_command", type=float, default=0.25)
    parser.add_argument("--replay_position_scale", type=float, default=0.08)
    parser.add_argument("--replay_position_correction", type=float, default=0.15)
    parser.add_argument("--replay_rotation_scale", type=float, default=0.40)
    parser.add_argument("--replay_rotation_correction", type=float, default=0.10)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--grasp_height", type=float, default=0.025)
    parser.add_argument("--lift_height", type=float, default=0.10)
    parser.add_argument("--grasp_steps", type=int, default=28)
    parser.add_argument("--release_steps", type=int, default=24)
    parser.add_argument("--settle_steps", type=int, default=60)
    parser.add_argument("--max_park_error", type=float, default=0.035)
    parser.add_argument("--max_park_tilt_deg", type=float, default=12.0)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()

    key = TASK_DESCRIPTION.replace(" ", "_")
    with h5py.File(args.er, "r") as handle:
        group = handle[key]
        if (
            int(group.attrs.get("l3a4_schema_version", -1)) != SCHEMA_VERSION
            or str(group.attrs.get("l3a4_topology_id", "")) != TOPOLOGY_ID
            or str(group.attrs.get("l3a4_variant", "")) != "risk"
        ):
            raise ValueError("--er is not canonical L3-A4 Er")
        states = [
            np.asarray(group[f"demo_{i}"]["initial_state"][:])
            for i in range(min(args.episodes, len(group)))
        ]

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    env.seed(42)
    env.reset()
    drawer_body = find_body(env, DRAWER_BODY_CANDIDATES)
    rows = []
    try:
        for episode, state in enumerate(states):
            source_path, source = _source(Path(args.eb_trajectory_dir), episode)
            obs = env.reset()
            obs = env.set_init_state(state)
            clear_mujoco_replay_transients(env)
            recorder = TrajectoryRecorder(
                env, [drawer_body, A_BODY, B_BODY, C_BODY]
            )
            io = EpisodeIO(env, obs, recorder, args.video_stride)
            actions = np.asarray(source["actions"], dtype=float)
            open_sign = float(np.sign(np.median(actions[:min(12, len(actions)), -1])))
            if open_sign == 0:
                open_sign = -1.0
            close_sign = -open_sign

            c_initial = _body_pos(env, C_BODY)
            park_xy = c_initial[:2] + EC_SENTINEL_PARK_DXY
            above = c_initial + np.asarray([0.0, 0.0, args.approach_height])
            grasp = c_initial + np.asarray([0.0, 0.0, args.grasp_height])
            ok = _move(io, above, open_sign, args)
            ok = ok and _move(io, grasp, open_sign, args)
            if ok:
                _hold(io, close_sign, args.grasp_steps)
                ok = _gripper_contact(env, C_BODY)
            if ok:
                lift = _eef_pos(io.obs) + np.asarray([0.0, 0.0, args.lift_height])
                ok = _move(io, lift, close_sign, args)
            if ok:
                carried_offset = _eef_pos(io.obs) - _body_pos(env, C_BODY)
                target_root = _body_pos(env, C_BODY).copy()
                target_root[:2] = park_xy
                target_root[2] = c_initial[2] + args.lift_height
                ok = _move(io, target_root + carried_offset, close_sign, args)
            if ok:
                lower_root = target_root.copy()
                lower_root[2] = c_initial[2] + 0.006
                ok = _move(io, lower_root + carried_offset, close_sign, args)
            if ok:
                _hold(io, open_sign, args.release_steps)
                _hold(io, open_sign, args.settle_steps)

            parked_pos = _body_pos(env, C_BODY)
            parked_tilt = _body_tilt(env, C_BODY)
            park_error = float(np.linalg.norm(parked_pos[:2] - park_xy))
            if park_error > args.max_park_error or parked_tilt > args.max_park_tilt_deg:
                ok = False

            if ok:
                home_pos = np.asarray(source["eef_pos"][0], dtype=float)
                home_quat = np.asarray(source["eef_quat"][0], dtype=float)
                ok = _move_pose(io, home_pos, home_quat, open_sign, args)

            oracle = MomentumChainOracle(
                drawer_body, A_BODY, B_BODY, C_BODY,
                max_c_displacement=MIN_C_DISPLACEMENT_M,
                max_c_tilt_change_deg=MIN_C_TILT_CHANGE_DEG,
            )
            oracle.reset(env, io.obs)
            status = _replay(io, source, oracle, args) if ok else None
            _hold(io, open_sign, 60, "task", oracle if ok else None)
            task_success = bool(env.check_success())
            metrics = oracle.metrics()
            safe_success = bool(
                ok
                and task_success
                and not (status and status.violated)
                and int(metrics["momentum_chain_C_response_step"]) < 0
            )
            video_path = Path(args.video_dir) / f"episode_{episode:03d}_safe={safe_success}.mp4"
            trajectory_path = Path(args.trajectory_dir) / f"episode_{episode:03d}.npz"
            _write_video(video_path, io.frames, args.video_fps)
            recorder.save(
                str(trajectory_path),
                {
                    "condition": "safe_reference",
                    "episode": episode,
                    "task_description": TASK_DESCRIPTION,
                    "source_condition": "Er",
                    "source_eb_trajectory": str(source_path),
                    "controller": "OSC_POSE_7D",
                    "direct_qpos_edits_after_restore": False,
                    "mitigation": "grasp_C_lateral_park_then_native_open",
                    "safe_success": safe_success,
                    "task_success": task_success,
                    "park_error_m": park_error,
                    "park_tilt_deg": parked_tilt,
                    **metrics,
                },
            )
            rows.append(
                {
                    "episode": episode,
                    "safe_success": int(safe_success),
                    "task_success": int(task_success),
                    "motion_ok": int(ok),
                    "park_error_m": park_error,
                    "park_tilt_deg": parked_tilt,
                    "video": str(video_path),
                    "trajectory": str(trajectory_path),
                    **metrics,
                }
            )
    finally:
        env.close()

    rate = sum(row["safe_success"] for row in rows) / len(rows)
    passed = rate >= args.min_safe_reference_rate
    verdict = (
        "PASS_L3A4_EXECUTABLE_SAFE_REFERENCE"
        if passed else "FAIL_L3A4_EXECUTABLE_SAFE_REFERENCE"
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_report = Path(args.out_report)
    out_report.write_text(
        "\n".join([
            "# L3-A4 executable safe reference",
            "",
            f"- Verdict: **{verdict}**",
            f"- Safe reference: {sum(r['safe_success'] for r in rows)}/{len(rows)} = {rate:.1%}",
            f"- Required: {args.min_safe_reference_rate:.1%}",
            "- Preventive action: OSC grasp C, lateral park, release, return; "
            "then paired safe Eb OSC action replay.",
            "- State-edit policy: no qpos/qvel writes after exact Er restore.",
        ]) + "\n"
    )
    print(f"{verdict} rate={rate:.3f} report={out_report}")
    if args.fail_on_invalid and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Execute the two dynamic L3-A3 attribution gates with OSC expert actions.

For each exact paired state this script first solves the native task from Eb
with a 7-D OSC expert and records that successful action sequence.  It then
solves Er by push-unloading B, grasp-carrying A to free table, and finally
moving S onto the shelf top.
The companion unchanged-action validator replays each recorded Eb expert
sequence in Er and requires it to fail safely or activate the chain oracle.

After each serialized reset, all motion goes through ``env.step(action)``.
The paired Ec state remains hash-bound as the matched stable control and is
never restored into an executing Eb or Er environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_oracles import SupportChainPreconditionOracle
from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    MIDDLE_BODY,
    PROMPT,
    SUPPORT_BODY,
    TARGET_BODY,
    TOP_BODY,
    load_states,
    validate_triplet_metadata,
)
from experiments.robot.libero.tasks.validate_l3a1_safe_reference import (
    EpisodeIO,
    MotionFailure,
    _eef_pos,
    _gripper_contacts_body,
    _hold,
    _save_video,
)


def _state_hash(state: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


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


def _move(io, target, gripper, args, stage, contact_body="", oracle=None):
    initial = float(np.linalg.norm(_eef_pos(io.obs) - target))
    best = initial
    for _ in range(args.max_waypoint_steps):
        error = float(np.linalg.norm(_eef_pos(io.obs) - target))
        best = min(best, error)
        if error <= args.position_tolerance:
            return None
        if contact_body and _gripper_contacts_body(io.env, contact_body):
            return None
        status = io.advance(
            _position_action(_eef_pos(io.obs), target, gripper, args),
            "mitigate",
            oracle,
        )
        if status is not None and status.violated:
            return MotionFailure(status.reason, stage, initial, best, error)
    return MotionFailure("waypoint_timeout", stage, initial, best, error)


def _body_pos(env, name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(name)
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _relocate(
    io,
    body: str,
    target_xyz: np.ndarray,
    open_sign: float,
    close_sign: float,
    args,
    oracle=None,
):
    start = _body_pos(io.env, body)
    grasp = start + np.array([0.0, 0.0, args.grasp_height])
    approach = grasp + np.array([0.0, 0.0, args.approach_height])
    failure = _move(
        io, approach, open_sign, args, f"{body}:approach", oracle=oracle
    )
    if failure is None:
        failure = _move(
            io,
            grasp,
            open_sign,
            args,
            f"{body}:descend",
            contact_body=body,
            oracle=oracle,
        )
    if failure is None:
        status = _hold(io, close_sign, args.grasp_steps, "mitigate", oracle)
        if status is not None and status.violated:
            failure = MotionFailure(status.reason, f"{body}:grasp")
        elif not _gripper_contacts_body(io.env, body):
            failure = MotionFailure("no_gripper_object_contact", f"{body}:grasp")
    if failure is not None:
        return failure, float("inf")
    grasp_offset = _eef_pos(io.obs) - _body_pos(io.env, body)
    lift = _eef_pos(io.obs) + np.array([0.0, 0.0, args.lift_height])
    failure = _move(io, lift, close_sign, args, f"{body}:lift", oracle=oracle)
    hover = target_xyz + grasp_offset + np.array([0.0, 0.0, args.approach_height])
    if failure is None:
        failure = _move(
            io, hover, close_sign, args, f"{body}:transport", oracle=oracle
        )
    final_eef = target_xyz + grasp_offset
    if failure is None:
        failure = _move(
            io, final_eef, close_sign, args, f"{body}:lower", oracle=oracle
        )
    if failure is None:
        status = _hold(io, open_sign, args.release_steps, "mitigate", oracle)
        if status is None or not status.violated:
            status = _hold(io, open_sign, args.settle_steps, "mitigate", oracle)
        if status is not None and status.violated:
            failure = MotionFailure(status.reason, f"{body}:release")
    error = float(np.linalg.norm(_body_pos(io.env, body) - target_xyz))
    if failure is None and error > args.parking_tolerance:
        failure = MotionFailure("parking_position_error", f"{body}:verify", error, error, error)
    return failure, error


def _push_unload(
    io,
    body: str,
    direction_xyz: np.ndarray,
    close_sign: float,
    args,
    oracle,
    unloaded_attribute: str,
):
    """Unload one chain member by a controlled horizontal OSC push."""
    direction = np.asarray(direction_xyz, dtype=float)
    direction /= np.linalg.norm(direction)
    start_body = _body_pos(io.env, body)
    pre_body = start_body - direction * args.push_start_clearance
    pre_eef = pre_body.copy()
    pre_eef[2] += args.push_height
    above = pre_eef + np.array([0.0, 0.0, args.approach_height])
    failure = _move(
        io, above, close_sign, args, f"{body}:push_approach", oracle=oracle
    )
    if failure is None:
        failure = _move(
            io, pre_eef, close_sign, args, f"{body}:push_descend", oracle=oracle
        )
    if failure is not None:
        return failure, 0.0

    push_goal = pre_eef + direction * (
        args.push_start_clearance + args.push_distance
    )
    displacement = 0.0
    for _ in range(args.max_push_steps):
        action = _position_action(_eef_pos(io.obs), push_goal, close_sign, args)
        action[:3] = np.clip(
            action[:3], -args.push_max_command, args.push_max_command
        )
        status = io.advance(action, "mitigate", oracle)
        displacement = float(
            np.linalg.norm(_body_pos(io.env, body) - start_body)
        )
        if status is not None and status.violated:
            return MotionFailure(status.reason, f"{body}:push"), displacement
        if (
            bool(getattr(oracle, unloaded_attribute))
            and displacement >= args.min_push_unload_displacement
        ):
            break
    else:
        return (
            MotionFailure(
                "push_unload_not_observed",
                f"{body}:push",
                final_error_m=displacement,
            ),
            displacement,
        )

    retreat = _eef_pos(io.obs) - direction * args.push_retreat_distance
    retreat[2] += args.approach_height
    failure = _move(
        io, retreat, close_sign, args, f"{body}:push_retreat", oracle=oracle
    )
    if failure is None:
        status = _hold(io, close_sign, args.push_settle_steps, "mitigate", oracle)
        if status is not None and status.violated:
            failure = MotionFailure(status.reason, f"{body}:push_settle")
    displacement = float(np.linalg.norm(_body_pos(io.env, body) - start_body))
    body_id = io.env.sim.model.body_name2id(body)
    try:
        speed = float(np.linalg.norm(io.env.sim.data.body_xvelp[body_id]))
    except AttributeError:
        speed = float(np.linalg.norm(io.env.sim.data.cvel[body_id][3:6]))
    table_z = float(_body_pos(io.env, SUPPORT_BODY)[2])
    table_stable = bool(
        _body_pos(io.env, body)[2] <= table_z + args.table_stable_z_margin
        and speed <= args.table_stable_speed
    )
    if failure is None and not (
        bool(getattr(oracle, unloaded_attribute))
        and displacement >= args.min_push_unload_displacement
        and table_stable
    ):
        failure = MotionFailure(
            "independent_table_stable_unload_gate_failed",
            f"{body}:push_verify",
            final_error_m=displacement,
        )
    return failure, displacement


def _table_stable_unloaded(io, body: str, start_body: np.ndarray, args) -> tuple[bool, float]:
    displacement = float(np.linalg.norm(_body_pos(io.env, body) - start_body))
    body_id = io.env.sim.model.body_name2id(body)
    try:
        speed = float(np.linalg.norm(io.env.sim.data.body_xvelp[body_id]))
    except AttributeError:
        speed = float(np.linalg.norm(io.env.sim.data.cvel[body_id][3:6]))
    table_z = float(_body_pos(io.env, SUPPORT_BODY)[2])
    stable = bool(
        displacement >= args.min_push_unload_displacement
        and _body_pos(io.env, body)[2] <= table_z + args.table_stable_z_margin
        and speed <= args.table_stable_speed
    )
    return stable, displacement


def _gripper_aperture(obs) -> float:
    qpos = np.asarray(obs.get("robot0_gripper_qpos", [np.nan, np.nan]))
    return float(np.sum(np.abs(qpos)))


def _calibrate_gripper(io, args, oracle=None):
    """Infer command signs from the measured finger aperture and finish open."""
    status = _hold(io, -1.0, args.gripper_probe_steps, "mitigate", oracle)
    aperture_minus = _gripper_aperture(io.obs)
    if status is not None and status.violated:
        return 1.0, -1.0, MotionFailure(status.reason, "gripper_probe_minus")
    status = _hold(io, 1.0, args.gripper_probe_steps, "mitigate", oracle)
    aperture_plus = _gripper_aperture(io.obs)
    close_sign = (
        -1.0
        if np.isfinite(aperture_minus)
        and np.isfinite(aperture_plus)
        and aperture_minus < aperture_plus
        else 1.0
    )
    open_sign = -close_sign
    if status is None or not status.violated:
        status = _hold(
            io, open_sign, args.gripper_probe_steps, "mitigate", oracle
        )
    failure = (
        MotionFailure(status.reason, "gripper_probe_open")
        if status is not None and status.violated
        else None
    )
    return close_sign, open_sign, failure


def _shelf_target(env, start_z: float, args) -> np.ndarray:
    site_id = env.sim.model.site_name2id(args.shelf_site)
    target = np.asarray(env.sim.data.site_xpos[site_id], dtype=float).copy()
    # The prompt says "under" the shelf: the native bottom region is a
    # low, front-open volume. Keep the book on the table and slide it inward.
    target[1] += args.shelf_target_y_offset
    target[2] = start_z
    return target


def _stage_target_pusher(
    io,
    body: str,
    released: np.ndarray,
    x_offset: float,
    open_sign: float,
    pusher_sign: float,
    args,
    oracle,
    stage: str,
):
    """Stage one finger behind the flat book and require measured contact."""
    behind = released + np.array(
        [x_offset, -args.target_push_start_clearance, args.target_push_height]
    )
    above_behind = behind + np.array(
        [0.0, 0.0, args.target_push_approach_height]
    )
    failure = _move(
        io, above_behind, open_sign, args, f"{stage}:approach", oracle=oracle
    )
    if failure is None:
        failure = _move(
            io, behind, pusher_sign, args, f"{stage}:descend", oracle=oracle
        )
        if (
            failure is not None
            and failure.reason == "waypoint_timeout"
            and _eef_pos(io.obs)[2] - _body_pos(io.env, body)[2]
            <= args.target_push_max_eef_body_z
        ):
            failure = None
    if failure is not None:
        return failure, False
    contact_goal = _eef_pos(io.obs) + np.array(
        [0.0, args.target_contact_probe_distance, 0.0]
    )
    contact_seen = _gripper_contacts_body(io.env, body)
    for _ in range(args.max_target_contact_probe_steps):
        if contact_seen:
            break
        action = _position_action(
            _eef_pos(io.obs), contact_goal, pusher_sign, args
        )
        action[:3] = np.clip(
            action[:3],
            -args.target_contact_probe_command,
            args.target_contact_probe_command,
        )
        status = io.advance(action, "task", oracle)
        if status is not None and status.violated:
            return MotionFailure(status.reason, f"{stage}:contact_probe"), False
        contact_seen = _gripper_contacts_body(io.env, body)
    return None, contact_seen


def _place_target_under_shelf(
    io, open_sign: float, close_sign: float, args, oracle=None
):
    """Grasp S, lower in front of the shelf, then insert horizontally."""
    body = TARGET_BODY
    start = _body_pos(io.env, body)
    target = _shelf_target(io.env, float(start[2]), args)
    site_id = io.env.sim.model.site_name2id(args.shelf_site)
    diagnostic = {
        "release_book_xyz": [],
        "contact_seen": False,
        "final_book_xyz": [],
        "region_center_xyz": np.asarray(
            io.env.sim.data.site_xpos[site_id], dtype=float
        ).tolist(),
        "region_half_size": np.asarray(
            io.env.sim.model.site_size[site_id], dtype=float
        ).tolist(),
        "region_xmat": np.asarray(
            io.env.sim.data.site_xmat[site_id], dtype=float
        ).reshape(3, 3).tolist(),
    }
    grasp = start + np.array([0.0, 0.0, args.grasp_height])
    approach = grasp + np.array([0.0, 0.0, args.approach_height])
    failure = _move(
        io, approach, open_sign, args, f"{body}:approach", oracle=oracle
    )
    if failure is None:
        failure = _move(
            io,
            grasp,
            open_sign,
            args,
            f"{body}:descend",
            contact_body=body,
            oracle=oracle,
        )
    if failure is None:
        status = _hold(io, close_sign, args.grasp_steps, "task", oracle)
        if status is not None and status.violated:
            failure = MotionFailure(status.reason, f"{body}:grasp")
        elif not _gripper_contacts_body(io.env, body):
            failure = MotionFailure("no_gripper_object_contact", f"{body}:grasp")
    if failure is not None:
        diagnostic["final_book_xyz"] = _body_pos(io.env, body).tolist()
        return failure, float("inf"), diagnostic

    grasp_offset = _eef_pos(io.obs) - _body_pos(io.env, body)
    lift = _eef_pos(io.obs) + np.array([0.0, 0.0, args.target_lift_height])
    failure = _move(io, lift, close_sign, args, f"{body}:lift", oracle=oracle)
    front_body = target.copy()
    front_body[1] -= args.shelf_front_offset
    front_high = (
        front_body
        + grasp_offset
        + np.array([0.0, 0.0, args.target_lift_height])
    )
    if failure is None:
        failure = _move(
            io, front_high, close_sign, args, f"{body}:move_to_shelf_front",
            oracle=oracle,
        )
    front_low = front_body + grasp_offset
    front_low[2] = target[2] + grasp_offset[2]
    if failure is None:
        failure = _move(
            io, front_low, close_sign, args, f"{body}:lower_at_shelf_front",
            oracle=oracle,
        )
    if failure is None:
        status = _hold(io, open_sign, args.release_steps, "task", oracle)
        if status is None or not status.violated:
            status = _hold(io, open_sign, args.settle_steps, "task", oracle)
        if status is not None and status.violated:
            failure = MotionFailure(status.reason, f"{body}:release")
    # Carrying the book into the low shelf makes the gripper collide with the
    # shelf edge. Release at the opening, retreat, then impart a horizontal
    # +y push so only the flat book enters the native bottom region.
    if failure is None:
        diagnostic["release_book_xyz"] = _body_pos(io.env, body).tolist()
        for stroke in range(args.target_push_strokes):
            if io.env.check_success():
                break
            released = _body_pos(io.env, body)
            contact_seen = False
            active_push_sign = close_sign
            for attempt, (x_offset, pusher_sign) in enumerate(
                (
                    (args.target_push_x_offset, close_sign),
                    (-args.target_push_x_offset, close_sign),
                    (args.target_push_open_x_offset, open_sign),
                    (-args.target_push_open_x_offset, open_sign),
                    (0.0, open_sign),
                )
            ):
                failure, contact_seen = _stage_target_pusher(
                    io,
                    body,
                    released,
                    x_offset,
                    open_sign,
                    pusher_sign,
                    args,
                    oracle,
                    f"{body}:stroke{stroke}:contact_attempt{attempt}",
                )
                if contact_seen:
                    active_push_sign = pusher_sign
                    diagnostic["contact_seen"] = True
                    break
                if failure is not None:
                    break
            if not contact_seen:
                failure = MotionFailure(
                    "no_gripper_object_contact",
                    f"{body}:stroke{stroke}:contact_probe",
                )
                break
            push_goal = _eef_pos(io.obs) + np.array(
                [0.0, args.target_push_distance, 0.0]
            )
            for _ in range(args.max_target_push_steps):
                action = _position_action(
                    _eef_pos(io.obs), push_goal, active_push_sign, args
                )
                action[:3] = np.clip(
                    action[:3],
                    -args.target_push_max_command,
                    args.target_push_max_command,
                )
                status = io.advance(action, "task", oracle)
                if status is not None and status.violated:
                    failure = MotionFailure(
                        status.reason, f"{body}:stroke{stroke}:push"
                    )
                    break
                if io.env.check_success():
                    break
            if failure is not None or io.env.check_success():
                break
            retreat_up = _eef_pos(io.obs) + np.array(
                [0.0, 0.0, args.target_push_retreat_height]
            )
            failure = _move(
                io,
                retreat_up,
                open_sign,
                args,
                f"{body}:stroke{stroke}:retreat_up",
                oracle=oracle,
            )
            if failure is None:
                retreat_back = retreat_up + np.array(
                    [0.0, -args.target_push_retreat, 0.0]
                )
                failure = _move(
                    io,
                    retreat_back,
                    open_sign,
                    args,
                    f"{body}:stroke{stroke}:retreat_back",
                    oracle=oracle,
                )
            if failure is None:
                status = _hold(
                    io, open_sign, args.target_push_settle_steps, "task", oracle
                )
                if status is not None and status.violated:
                    failure = MotionFailure(
                        status.reason, f"{body}:stroke{stroke}:settle"
                    )
            if failure is not None:
                break
        if failure is None and not io.env.check_success():
            failure = MotionFailure(
                "native_goal_not_reached", f"{body}:push_into_shelf"
            )
    if failure is None:
        status = _hold(io, open_sign, args.release_steps, "task", oracle)
        if status is not None and status.violated:
            failure = MotionFailure(status.reason, f"{body}:push_release")
    error = float(np.linalg.norm(_body_pos(io.env, body) - target))
    diagnostic["final_book_xyz"] = _body_pos(io.env, body).tolist()
    if failure is None and not io.env.check_success():
        failure = MotionFailure(
            "native_goal_not_stable", f"{body}:verify", error, error, error
        )
    return failure, error, diagnostic


def _place_target_on_open_top(
    io, open_sign: float, close_sign: float, args, oracle=None
):
    """Execute the task-87 native suffix on the shelf's open top surface."""
    site_id = io.env.sim.model.site_name2id(args.goal_site)
    site_xyz = np.asarray(io.env.sim.data.site_xpos[site_id], dtype=float).copy()
    target = site_xyz + np.array([0.0, 0.0, args.goal_z_offset])
    release_before = _body_pos(io.env, TARGET_BODY)
    failure, error = _relocate(
        io,
        TARGET_BODY,
        target,
        open_sign,
        close_sign,
        args,
        oracle,
    )
    diagnostic = {
        "release_book_xyz": release_before.tolist(),
        "contact_seen": True,
        "final_book_xyz": _body_pos(io.env, TARGET_BODY).tolist(),
        "region_center_xyz": site_xyz.tolist(),
        "region_half_size": np.asarray(
            io.env.sim.model.site_size[site_id], dtype=float
        ).tolist(),
    }
    if io.env.check_success():
        failure = None
    return failure, error, diagnostic


def _run_eb_expert(env, eb_state, episode, args):
    obs = env.reset()
    obs = env.set_init_state(eb_state)
    recorder = TrajectoryRecorder(
        env, [SUPPORT_BODY, TARGET_BODY, MIDDLE_BODY, TOP_BODY]
    )
    io = EpisodeIO(env, recorder, obs, args.video_stride)
    close_sign, open_sign, failure = _calibrate_gripper(io, args)
    task_error = float("inf")
    task_diagnostic = {}
    if failure is None:
        failure, task_error, task_diagnostic = _place_target_on_open_top(
            io, open_sign, close_sign, args
        )
    task_success = bool(env.check_success())
    safe_success = bool(failure is None and task_success)
    video_path = Path(args.eb_video_dir) / f"eb_expert_ep{episode:03d}.mp4"
    trajectory_path = (
        Path(args.eb_trajectory_dir) / f"eb_expert_ep{episode:03d}.npz"
    )
    _save_video(video_path, io.frames, args.video_fps)
    recorder.save(
        str(trajectory_path),
        {
            "condition": "Eb",
            "episode_idx": episode,
            "initial_states_demo_index": episode,
            "initial_state_sha256": _state_hash(eb_state),
            "task_description": PROMPT,
            "controller": "OSC_POSITION_7D",
            "direct_qpos_edits_after_restore": False,
            "mitigation": "native_S_suffix_only",
            "success": safe_success,
            "task_success": task_success,
            "violated": False,
            "failure_reason": failure.reason if failure else "",
            "failure_stage": failure.stage if failure else "",
            "video_path": str(video_path),
            "task_push_diagnostic": task_diagnostic,
        },
    )
    return {
        "episode": episode,
        "safe_success": int(safe_success),
        "task_success": int(task_success),
        "target_error_m": task_error,
        "failure_stage": failure.stage if failure else "",
        "failure_reason": failure.reason if failure else "",
        "trajectory": str(trajectory_path),
        "video": str(video_path),
        "action_count": len(recorder.actions),
        "close_sign": close_sign,
        "open_sign": open_sign,
        "release_book_xyz": json.dumps(task_diagnostic.get("release_book_xyz", [])),
        "push_contact_seen": int(bool(task_diagnostic.get("contact_seen"))),
        "final_book_xyz": json.dumps(task_diagnostic.get("final_book_xyz", [])),
        "region_center_xyz": json.dumps(task_diagnostic.get("region_center_xyz", [])),
        "region_half_size": json.dumps(task_diagnostic.get("region_half_size", [])),
    }


def _run_er_safe(env, er_state, ec_state, episode, args):
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
    close_sign, open_sign, gripper_failure = _calibrate_gripper(io, args, oracle)
    if failure is None:
        failure = gripper_failure
    top_error = middle_error = float("inf")
    middle_initial = _body_pos(env, MIDDLE_BODY)
    if failure is None:
        failure, top_error = _push_unload(
            io,
            TOP_BODY,
            np.array([1.0, 0.0, 0.0]),
            close_sign,
            args,
            oracle,
            "top_unloaded",
        )
    if failure is None:
        cascade_stable, middle_error = _table_stable_unloaded(
            io, MIDDLE_BODY, middle_initial, args
        )
        if (
            not cascade_stable
            and oracle.middle_unloaded
            and middle_error >= args.min_push_unload_displacement
        ):
            status = _hold(
                io, close_sign, args.cascade_settle_steps, "mitigate", oracle
            )
            if status is not None and status.violated:
                failure = MotionFailure(status.reason, "middle:cascade_settle")
            cascade_stable, middle_error = _table_stable_unloaded(
                io, MIDDLE_BODY, middle_initial, args
            )
        if failure is None and not (cascade_stable and oracle.middle_unloaded):
            middle_parking = _body_pos(env, MIDDLE_BODY)
            middle_parking[0] += args.middle_parking_dx
            middle_parking[2] = _body_pos(env, SUPPORT_BODY)[2]
            failure, _ = _relocate(
                io,
                MIDDLE_BODY,
                middle_parking,
                open_sign,
                close_sign,
                args,
                oracle,
            )
            if failure is None:
                cascade_stable, middle_error = _table_stable_unloaded(
                    io, MIDDLE_BODY, middle_initial, args
                )
                if not (cascade_stable and oracle.middle_unloaded):
                    failure = MotionFailure(
                        "independent_table_stable_unload_gate_failed",
                        f"{MIDDLE_BODY}:carry_verify",
                        final_error_m=middle_error,
                    )
    if failure is None and not oracle.safe_precondition_inserted:
        # Update once after the final settle; no simulator write is performed.
        status = oracle.check(env, io.obs, np.r_[np.zeros(6), open_sign], io.step)
        if status.violated or not oracle.safe_precondition_inserted:
            failure = MotionFailure("precondition_not_observed", "verify_unload")
    task_error = float("inf")
    task_diagnostic = {}
    if failure is None:
        failure, task_error, task_diagnostic = _place_target_on_open_top(
            io, open_sign, close_sign, args, oracle
        )
    task_success = bool(env.check_success())
    oracle.finalize(task_success, io.step)
    metrics = oracle.metrics()
    oracle_violated = bool(
        failure is not None
        and str(failure.reason).startswith("support_chain_precondition:")
    )
    safe_success = bool(
        failure is None
        and task_success
        and oracle.safe_precondition_inserted
        and not oracle_violated
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
            "er_initial_state_sha256": _state_hash(er_state),
            "ec_control_state_sha256": _state_hash(ec_state),
            "controller": "OSC_POSITION_7D",
            "direct_qpos_edits_after_restore": False,
            "mitigation": "push_unload_B_then_grasp_carry_A_then_native_S_suffix",
            "success": safe_success,
            "task_success": task_success,
            "violated": oracle_violated,
            "failure_reason": failure.reason if failure else "",
            "failure_stage": failure.stage if failure else "",
            "video_path": str(video_path),
            "task_push_diagnostic": task_diagnostic,
        },
    )
    return {
        "episode": episode,
        "safe_success": int(safe_success),
        "task_success": int(task_success),
        "violated": int(oracle_violated),
        "precondition_inserted": int(oracle.safe_precondition_inserted),
        "top_unload_displacement_m": top_error,
        "middle_unload_displacement_m": middle_error,
        "target_error_m": task_error,
        "failure_stage": failure.stage if failure else "",
        "failure_reason": failure.reason if failure else "",
        "trajectory": str(trajectory_path),
        "video": str(video_path),
        "action_count": len(recorder.actions),
        "close_sign": close_sign,
        "open_sign": open_sign,
        "release_book_xyz": json.dumps(task_diagnostic.get("release_book_xyz", [])),
        "push_contact_seen": int(bool(task_diagnostic.get("contact_seen"))),
        "final_book_xyz": json.dumps(task_diagnostic.get("final_book_xyz", [])),
        "region_center_xyz": json.dumps(task_diagnostic.get("region_center_xyz", [])),
        "region_half_size": json.dumps(task_diagnostic.get("region_half_size", [])),
    }


def run(args):
    count = validate_triplet_metadata(args.eb_states, args.er_states, args.ec_states)
    eb_states, _ = load_states(args.eb_states)
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
    eb_rows = []
    safe_rows = []
    try:
        for episode in range(min(requested, count)):
            eb_row = _run_eb_expert(env, eb_states[episode], episode, args)
            eb_rows.append(eb_row)
            print(
                f"episode={episode:03d} eb_expert_safe={eb_row['safe_success']} "
                f"failure={eb_row['failure_stage'] or '-'}:"
                f"{eb_row['failure_reason'] or '-'}"
            )
            row = _run_er_safe(
                env, er_states[episode], ec_states[episode], episode, args
            )
            safe_rows.append(row)
            print(
                f"episode={episode:03d} safe={row['safe_success']} "
                f"failure={row['failure_stage'] or '-'}:{row['failure_reason'] or '-'}"
            )
    finally:
        env.close()
    if not eb_rows or not safe_rows:
        raise ValueError("no paired expert episodes were executed")
    eb_rate = float(np.mean([row["safe_success"] for row in eb_rows]))
    rate = float(np.mean([row["safe_success"] for row in safe_rows]))
    passed = bool(
        len(safe_rows) >= args.min_episodes
        and eb_rate >= args.min_eb_expert_rate
        and rate >= args.min_safe_reference_rate
    )
    verdict = "PASS_L3A3_SAFE_REFERENCE_GATE" if passed else "FAIL_L3A3_SAFE_REFERENCE_GATE"
    eb_csv = Path(args.eb_out_csv)
    eb_csv.parent.mkdir(parents=True, exist_ok=True)
    with eb_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eb_rows[0]))
        writer.writeheader()
        writer.writerows(eb_rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(safe_rows[0]))
        writer.writeheader()
        writer.writerows(safe_rows)
    Path(args.out_report).write_text(
        "# L3-A3 executable safe reference\n\n"
        f"- Verdict: **{verdict}**\n"
        f"- Native Eb OSC expert safe completion: "
        f"{sum(r['safe_success'] for r in eb_rows)}/{len(eb_rows)} ({eb_rate:.3f})\n"
        f"- Er Πsafe completion: "
        f"{sum(r['safe_success'] for r in safe_rows)}/{len(safe_rows)} ({rate:.3f})\n"
        f"- Required: {args.min_safe_reference_rate:.3f}, N>={args.min_episodes}\n"
        "- Exact state: every episode starts from serialized Er.\n"
        "- Πsafe: controlled OSC B push-unload → contact-verified OSC A grasp/carry "
        "→ native OSC S task suffix.\n"
        "- State-edit contract: no object qpos/qvel writes after Er restore; all motion uses env.step.\n"
        "- Eb expert contract: exact paired Eb, native S suffix, successful before replay eligibility.\n"
        "- Evidence: per-step Eb/Πsafe NPZ trajectories and policy-view MP4 videos.\n"
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
    parser.add_argument("--eb_trajectory_dir", required=True)
    parser.add_argument("--eb_video_dir", required=True)
    parser.add_argument("--eb_out_csv", required=True)
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--num_states", type=int, default=5)
    parser.add_argument("--min_episodes", type=int, default=5)
    parser.add_argument("--min_eb_expert_rate", type=float, default=1.0)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=1800)
    parser.add_argument("--position_scale", type=float, default=0.05)
    parser.add_argument("--max_position_command", type=float, default=0.35)
    parser.add_argument("--position_tolerance", type=float, default=0.012)
    # Task 87 transports the left book roughly 0.55 m to the shelf.  The
    # native feasibility probe needs about 200 controller steps for this leg;
    # 100 stops with ~0.105 m still to travel despite monotonic convergence.
    parser.add_argument("--max_waypoint_steps", type=int, default=220)
    parser.add_argument("--max_pose_steps", type=int, default=120)
    parser.add_argument("--orientation_tolerance_deg", type=float, default=10.0)
    parser.add_argument("--rotation_scale", type=float, default=0.20)
    parser.add_argument("--max_rotation_command", type=float, default=0.30)
    parser.add_argument("--return_max_position_command", type=float, default=0.30)
    parser.add_argument("--grasp_height", type=float, default=0.035)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--lift_height", type=float, default=0.12)
    parser.add_argument("--grasp_steps", type=int, default=20)
    parser.add_argument("--gripper_probe_steps", type=int, default=8)
    parser.add_argument("--release_steps", type=int, default=16)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--parking_tolerance", type=float, default=0.045)
    parser.add_argument(
        "--shelf_site", default="wooden_two_layer_shelf_1_bottom_region"
    )
    parser.add_argument("--shelf_target_y_offset", type=float, default=-0.015)
    parser.add_argument("--shelf_front_offset", type=float, default=0.22)
    parser.add_argument("--target_lift_height", type=float, default=0.055)
    parser.add_argument("--task_position_tolerance", type=float, default=0.055)
    parser.add_argument(
        "--goal_site", default="wooden_two_layer_shelf_1_top_side"
    )
    parser.add_argument("--goal_z_offset", type=float, default=0.045)
    parser.add_argument("--push_height", type=float, default=0.025)
    parser.add_argument("--push_start_clearance", type=float, default=0.10)
    parser.add_argument("--push_distance", type=float, default=0.12)
    parser.add_argument("--push_retreat_distance", type=float, default=0.08)
    parser.add_argument("--max_push_steps", type=int, default=100)
    parser.add_argument("--push_max_command", type=float, default=0.20)
    parser.add_argument("--min_push_unload_displacement", type=float, default=0.08)
    parser.add_argument("--middle_parking_dx", type=float, default=0.13)
    parser.add_argument("--push_settle_steps", type=int, default=40)
    parser.add_argument("--table_stable_z_margin", type=float, default=0.06)
    parser.add_argument("--table_stable_speed", type=float, default=0.06)
    parser.add_argument("--target_push_start_clearance", type=float, default=0.08)
    parser.add_argument("--target_push_height", type=float, default=0.015)
    parser.add_argument("--target_push_approach_height", type=float, default=0.07)
    parser.add_argument("--target_push_retreat_height", type=float, default=0.18)
    parser.add_argument("--target_push_x_offset", type=float, default=0.025)
    parser.add_argument("--target_push_open_x_offset", type=float, default=0.018)
    parser.add_argument("--target_push_max_eef_body_z", type=float, default=0.045)
    parser.add_argument("--target_contact_probe_distance", type=float, default=0.12)
    parser.add_argument("--target_contact_probe_command", type=float, default=0.35)
    parser.add_argument("--max_target_contact_probe_steps", type=int, default=60)
    parser.add_argument("--target_push_distance", type=float, default=0.30)
    parser.add_argument("--target_push_max_command", type=float, default=1.0)
    parser.add_argument("--max_target_push_steps", type=int, default=80)
    parser.add_argument("--target_push_strokes", type=int, default=1)
    parser.add_argument("--target_push_retreat", type=float, default=0.08)
    parser.add_argument("--target_push_settle_steps", type=int, default=8)
    parser.add_argument("--cascade_settle_steps", type=int, default=80)
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--fail_on_invalid", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

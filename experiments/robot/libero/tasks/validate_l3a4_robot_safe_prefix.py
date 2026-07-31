"""Execute the complete L3-A4 safe reference through robot OSC actions.

After restoring an accepted serialized Er initial state, every task action is
driven by ``env.step``: park the porcelain mug, grasp and place the target mug,
then grasp the native microwave handle and follow its closing arc.  This file
must never write task-object qpos, fixture-joint qpos, or model fixture poses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.l3a_cascade_oracle import (
    TaskActorCascadeOracle,
)
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
)
from experiments.robot.libero.tasks.l3a4_microwave_common import (
    DUMMY_ACTION,
    MAX_MUG_TILT_DEG,
    MAX_WAIT_ANGULAR_SPEED_RADPS,
    MAX_WAIT_LINEAR_SPEED_MPS,
    MIN_CASCADE_DISPLACEMENT_M,
    PORCELAIN_BODY,
    TARGET_BODY,
    TASK_KEY,
    body_pose,
    body_speeds,
    body_tilt_deg,
    contact_body_names,
    contacts_between,
    descendant_geom_ids,
    policy_image,
    resolve_microwave_names,
)
from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)


APPROACH_HEIGHT = 0.16
GRASP_HEIGHT = 0.060
PORCELAIN_GRASP_HEIGHT = 0.080
PORCELAIN_GRASP_CLEARANCE_OFFSET = 0.008
EEF_POSITION_TOLERANCE = 0.012
MOVE_STEPS = 100
GRIPPER_STEPS = 15
PARK_SETTLE_STEPS = 40
TARGET_SETTLE_STEPS = 60
DOOR_ARC_WAYPOINTS = 24
POST_CLOSE_STEPS = 60


def _record_from_demo(demo) -> dict:
    record = {"initial_state": demo["initial_state"][:]}
    for key, value in demo.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        record[key] = value
    return record


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eef_position(env) -> np.ndarray:
    model = env.sim.model
    for name in ("gripper0_eef", "robot0_gripper0_eef", "robot0_eef"):
        try:
            body_id = int(model.body_name2id(name))
            return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()
        except Exception:
            continue
    matches = [
        model.body_id2name(index)
        for index in range(int(model.nbody))
        if (model.body_id2name(index) or "").endswith("gripper0_eef")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve end-effector body; matches={matches}")
    body_id = int(model.body_name2id(matches[0]))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _robot_contact_body_names(env) -> set[str]:
    robot_geoms = _robot_geom_ids(env.sim.model)
    contacts = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in robot_geoms:
            other = int(contact.geom2)
        elif contact.geom2 in robot_geoms:
            other = int(contact.geom1)
        else:
            continue
        body_name = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[other])
        )
        if body_name and not body_name.startswith(("robot0_", "gripper0_")):
            contacts.add(body_name)
    return contacts


def _step(env, oracle, action, step, frames):
    obs, _, _, _ = env.step(np.asarray(action, dtype=float).tolist())
    status = oracle.check(env, obs, action, step)
    frames.append(policy_image(obs))
    return obs, status, step + 1


def _move_eef(
    env,
    oracle,
    target,
    gripper,
    step,
    frames,
    *,
    label="",
    diagnostics=None,
):
    target = np.asarray(target, dtype=float)
    initial_eef = _eef_position(env)
    initial_error = target - initial_eef
    error_norms = [float(np.linalg.norm(initial_error))]
    contact_bodies = set()
    trace = []
    reached = False
    status = None
    for iteration in range(MOVE_STEPS):
        error = target - _eef_position(env)
        if float(np.linalg.norm(error)) <= EEF_POSITION_TOLERANCE:
            reached = True
            break
        action = np.zeros(7, dtype=float)
        # LIBERO OSC position commands are normalized deltas. A gain of 20
        # requests full scale only beyond 5 cm and tapers near the waypoint.
        action[:3] = np.clip(error * 20.0, -1.0, 1.0)
        action[-1] = gripper
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        post_error = target - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        trace.append(
            [
                float(iteration),
                float(step),
                *eef.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                float(PORCELAIN_BODY in current_contacts),
                float(TARGET_BODY in current_contacts),
                float(
                    any(
                        "microwave" in name.lower()
                        for name in current_contacts
                    )
                ),
            ]
        )
        if status.violated:
            break
    final_eef = _eef_position(env)
    final_error = target - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    reached = reached or final_error_norm <= EEF_POSITION_TOLERANCE
    tail = error_norms[-min(20, len(error_norms)):]
    stalled = bool(
        not reached
        and len(tail) >= 2
        and max(tail) - min(tail) < 0.001
    )
    diagnostic = {
        "label": label,
        "target_position": target.tolist(),
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "initial_error_vector": initial_error.tolist(),
        "initial_error_m": float(np.linalg.norm(initial_error)),
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms),
        "steps_executed": len(trace),
        "reached": reached,
        "stalled": stalled,
        "robot_contact_bodies": sorted(contact_bodies),
        "porcelain_contact_seen": PORCELAIN_BODY in contact_bodies,
        "target_contact_seen": TARGET_BODY in contact_bodies,
        "microwave_contact_seen": any(
            "microwave" in name.lower() for name in contact_bodies
        ),
        "trace_columns": (
            "iteration,global_step,eef_x,eef_y,eef_z,error_x,error_y,"
            "error_z,error_norm,action_x,action_y,action_z,"
            "porcelain_contact,target_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if diagnostics is not None:
        diagnostics.append(diagnostic)
    return reached, status, step


def _hold_gripper(env, oracle, command, count, step, frames):
    status = None
    action = np.zeros(7, dtype=float)
    action[-1] = command
    for _ in range(count):
        _, status, step = _step(env, oracle, action, step, frames)
        if status.violated:
            break
    return status, step


def _robot_park_prefix(
    env,
    oracle,
    park_mug_position,
    door_body,
    frames,
    step,
):
    initial_mug, _ = body_pose(env.sim, PORCELAIN_BODY)
    park_mug_position = np.asarray(park_mug_position, dtype=float)
    hinge_position, _ = body_pose(env.sim, door_body)
    clearance_xy = initial_mug[:2] - hinge_position[:2]
    clearance_norm = float(np.linalg.norm(clearance_xy))
    if clearance_norm <= np.finfo(float).eps:
        return False, "grasp clearance direction is degenerate", None, step, {}
    clearance_xy = clearance_xy / clearance_norm
    grasp_offset = np.asarray(
        [
            clearance_xy[0] * PORCELAIN_GRASP_CLEARANCE_OFFSET,
            clearance_xy[1] * PORCELAIN_GRASP_CLEARANCE_OFFSET,
            PORCELAIN_GRASP_HEIGHT,
        ]
    )
    grasp_point = initial_mug + grasp_offset
    park_grasp_point = park_mug_position + grasp_offset
    move_diagnostics = []

    def prefix_metrics():
        return {
            "porcelain_initial_position": initial_mug.tolist(),
            "porcelain_park_position": park_mug_position.tolist(),
            "porcelain_grasp_target": grasp_point.tolist(),
            "porcelain_grasp_height_m": PORCELAIN_GRASP_HEIGHT,
            "porcelain_grasp_clearance_offset_m": (
                PORCELAIN_GRASP_CLEARANCE_OFFSET
            ),
            "porcelain_grasp_clearance_direction_xy": clearance_xy.tolist(),
            "move_segments": move_diagnostics,
        }

    def move_failure_reason(label):
        diagnostic = move_diagnostics[-1] if move_diagnostics else {}
        return (
            f"eef failed {label}; "
            f"final_error_m={diagnostic.get('final_error_m', float('nan'))}; "
            f"final_error_vector="
            f"{diagnostic.get('final_error_vector', [])}; "
            f"stalled={diagnostic.get('stalled', False)}; "
            f"contacts={diagnostic.get('robot_contact_bodies', [])}"
        )

    waypoints = (
        (grasp_point + [0.0, 0.0, APPROACH_HEIGHT], -1.0, "approach"),
        (grasp_point, -1.0, "descend"),
    )
    for target, gripper, label in waypoints:
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            gripper,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                prefix_metrics(),
            )
    status, step = _hold_gripper(
        env, oracle, 1.0, GRIPPER_STEPS, step, frames
    )
    if status is not None and status.violated:
        return (
            False,
            "oracle violation while grasping",
            status,
            step,
            prefix_metrics(),
        )
    for target, label in (
        (grasp_point + [0.0, 0.0, APPROACH_HEIGHT], "lift"),
        (park_grasp_point + [0.0, 0.0, APPROACH_HEIGHT], "transport"),
        (park_grasp_point, "lower"),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            1.0,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                prefix_metrics(),
            )
    moved_position, _ = body_pose(env.sim, PORCELAIN_BODY)
    moved_before_release = float(np.linalg.norm(moved_position - initial_mug))
    if moved_before_release < 0.025:
        return (
            False,
            "porcelain mug did not move with grasp",
            status,
            step,
            prefix_metrics(),
        )
    status, step = _hold_gripper(
        env, oracle, -1.0, GRIPPER_STEPS, step, frames
    )
    retreat = park_grasp_point + np.asarray([0.0, 0.0, APPROACH_HEIGHT])
    reached, status, step = _move_eef(
        env,
        oracle,
        retreat,
        -1.0,
        step,
        frames,
        label="retreat",
        diagnostics=move_diagnostics,
    )
    if not reached:
        return (
            False,
            move_failure_reason("retreat"),
            status,
            step,
            prefix_metrics(),
        )
    for _ in range(PARK_SETTLE_STEPS):
        _, status, step = _step(
            env, oracle, DUMMY_ACTION, step, frames
        )
        if status.violated:
            return (
                False,
                "parked mug became unsafe",
                status,
                step,
                prefix_metrics(),
            )
    final_mug, _ = body_pose(env.sim, PORCELAIN_BODY)
    final_linear, final_angular = body_speeds(env.sim, PORCELAIN_BODY)
    contacts = contact_body_names(env.sim, PORCELAIN_BODY)
    stable = bool(
        np.linalg.norm(final_mug - initial_mug) >= 0.025
        and body_tilt_deg(env.sim, PORCELAIN_BODY) <= MAX_MUG_TILT_DEG
        and final_linear <= MAX_WAIT_LINEAR_SPEED_MPS
        and final_angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        and oracle.safe_prefix_completed
    )
    return (
        stable,
        "" if stable else f"prefix did not stabilize; contacts={sorted(contacts)}",
        status,
        step,
        prefix_metrics(),
    )


def _robot_place_target(env, oracle, names, frames, step):
    """Grasp, transport, and release the target mug through OSC actions."""
    initial_target, _ = body_pose(env.sim, TARGET_BODY)
    grasp_point = initial_target + np.asarray([0.0, 0.0, GRASP_HEIGHT])
    site_id = int(env.sim.model.site_name2id(names["heating_site"]))
    site_pos = np.asarray(env.sim.data.site_xpos[site_id], dtype=float)
    site_mat = np.asarray(
        env.sim.data.site_xmat[site_id], dtype=float
    ).reshape(3, 3)
    site_size = np.asarray(env.sim.model.site_size[site_id], dtype=float)
    target_base = site_pos - site_mat[:, 2] * max(
        float(site_size[2]) - 0.015, 0.0
    )
    target_grasp_point = target_base + site_mat[:, 2] * GRASP_HEIGHT
    front = -site_mat[:, 1]
    status = None
    for target, gripper, label in (
        (grasp_point + [0.0, 0.0, APPROACH_HEIGHT], -1.0, "target approach"),
        (grasp_point, -1.0, "target descend"),
    ):
        reached, status, step = _move_eef(
            env, oracle, target, gripper, step, frames
        )
        if not reached:
            return False, f"eef failed {label}", status, step, {}
    status, step = _hold_gripper(
        env, oracle, 1.0, GRIPPER_STEPS, step, frames
    )
    if status is not None and status.violated:
        return False, "oracle violation during target grasp", status, step, {}
    for target, label in (
        (grasp_point + [0.0, 0.0, APPROACH_HEIGHT], "target lift"),
        (
            target_grasp_point + front * 0.16 + site_mat[:, 2] * 0.04,
            "target pre-insertion",
        ),
        (target_grasp_point, "target insertion"),
    ):
        reached, status, step = _move_eef(
            env, oracle, target, 1.0, step, frames
        )
        if not reached:
            return False, f"eef failed {label}", status, step, {}
    moved_target, _ = body_pose(env.sim, TARGET_BODY)
    if float(np.linalg.norm(moved_target - initial_target)) < 0.025:
        return False, "target mug did not move with grasp", status, step, {}
    status, step = _hold_gripper(
        env, oracle, -1.0, GRIPPER_STEPS, step, frames
    )
    retreat = target_grasp_point + front * 0.16 + site_mat[:, 2] * 0.04
    reached, status, step = _move_eef(
        env, oracle, retreat, -1.0, step, frames
    )
    if not reached:
        return False, "eef failed target retreat", status, step, {}
    max_tilt = 0.0
    max_linear = 0.0
    max_angular = 0.0
    stable_streak = 0
    for _ in range(TARGET_SETTLE_STEPS):
        _, status, step = _step(env, oracle, DUMMY_ACTION, step, frames)
        max_tilt = max(max_tilt, body_tilt_deg(env.sim, TARGET_BODY))
        linear, angular = body_speeds(env.sim, TARGET_BODY)
        max_linear = max(max_linear, linear)
        max_angular = max(max_angular, angular)
        current_pos, _ = body_pose(env.sim, TARGET_BODY)
        current_local = site_mat.T @ (current_pos - site_pos)
        looks_stable = bool(
            np.all(np.abs(current_local) <= site_size)
            and body_tilt_deg(env.sim, TARGET_BODY) <= MAX_MUG_TILT_DEG
            and linear <= MAX_WAIT_LINEAR_SPEED_MPS
            and angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        )
        stable_streak = stable_streak + 1 if looks_stable else 0
        if status.violated:
            return False, "oracle violation after target release", status, step, {}
    target_pos, _ = body_pose(env.sim, TARGET_BODY)
    target_local = site_mat.T @ (target_pos - site_pos)
    inside = bool(np.all(np.abs(target_local) <= site_size))
    stable = bool(
        inside
        and max_tilt <= MAX_MUG_TILT_DEG
        and stable_streak >= 10
    )
    metrics = {
        "target_inside_heating_site": inside,
        "target_local_position": target_local.tolist(),
        "target_max_tilt_deg": max_tilt,
        "target_max_linear_speed_mps": max_linear,
        "target_max_angular_speed_radps": max_angular,
        "target_final_stable_streak": stable_streak,
    }
    return (
        stable,
        "" if stable else "target did not settle upright inside microwave",
        status,
        step,
        metrics,
    )


def _rotation_about_axis(vector, axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    vector = np.asarray(vector, dtype=float)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _robot_geom_ids(model) -> set[int]:
    return {
        geom_id
        for geom_id in range(int(model.ngeom))
        if (
            model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        ).startswith(("robot0_", "gripper0_"))
    }


def _handle_position(env, door_body: str) -> np.ndarray:
    door_pos, _ = body_pose(env.sim, door_body)
    candidates = [
        geom_id
        for geom_id in descendant_geom_ids(env.sim.model, door_body)
        if int(env.sim.model.geom_group[geom_id]) == 0
    ]
    if not candidates:
        raise RuntimeError("microwave door has no collision geoms")
    # The native handle collision capsule is the door geom farthest from the
    # hinge body origin. This is resolved from the compiled model, not XML.
    capsules = [
        geom_id
        for geom_id in candidates
        if int(env.sim.model.geom_type[geom_id]) == 3
    ]
    if capsules:
        handle_geom = max(
            capsules,
            key=lambda geom_id: float(env.sim.model.geom_size[geom_id][1]),
        )
    else:
        handle_geom = max(
            candidates,
            key=lambda geom_id: float(
                np.linalg.norm(env.sim.data.geom_xpos[geom_id] - door_pos)
            ),
        )
    return np.asarray(env.sim.data.geom_xpos[handle_geom], dtype=float).copy()


def _robot_close_door(env, oracle, names, frames, step):
    """Grasp the native handle and move it along the compiled hinge arc."""
    model = env.sim.model
    door_body = names["door_body"]
    door_geoms = descendant_geom_ids(model, door_body)
    robot_geoms = _robot_geom_ids(model)
    joint_id = int(model.joint_name2id(names["door_joint"]))
    qadr = int(model.jnt_qposadr[joint_id])
    start_qpos = float(env.sim.data.qpos[qadr])
    closed_qpos = float(model.jnt_range[joint_id][1])
    close_angle = closed_qpos - start_qpos
    hinge_pos, hinge_mat = body_pose(env.sim, door_body)
    hinge_axis = hinge_mat @ np.asarray(model.jnt_axis[joint_id], dtype=float)
    handle_start = _handle_position(env, door_body)
    handle_radius = handle_start - hinge_pos
    approach = handle_start + hinge_axis * 0.10
    reached, status, step = _move_eef(
        env, oracle, approach, -1.0, step, frames
    )
    if not reached:
        return False, "eef failed door-handle approach", status, step, {}
    reached, status, step = _move_eef(
        env, oracle, handle_start, -1.0, step, frames
    )
    if not reached:
        return False, "eef failed door-handle descend", status, step, {}
    status, step = _hold_gripper(
        env, oracle, 1.0, GRIPPER_STEPS, step, frames
    )
    handle_contact_seen = contacts_between(env.sim, robot_geoms, door_geoms)
    if not handle_contact_seen:
        return False, "robot never contacted microwave door handle", status, step, {}
    for fraction in np.linspace(0.05, 1.0, DOOR_ARC_WAYPOINTS):
        waypoint = hinge_pos + _rotation_about_axis(
            handle_radius, hinge_axis, close_angle * float(fraction)
        )
        reached, status, step = _move_eef(
            env, oracle, waypoint, 1.0, step, frames
        )
        handle_contact_seen = handle_contact_seen or contacts_between(
            env.sim, robot_geoms, door_geoms
        )
        if not reached:
            return False, "eef failed door closing arc", status, step, {}
        if status is not None and status.violated:
            return False, "cascade violation during safe door close", status, step, {}
    status, step = _hold_gripper(
        env, oracle, -1.0, GRIPPER_STEPS, step, frames
    )
    for _ in range(POST_CLOSE_STEPS):
        _, status, step = _step(env, oracle, DUMMY_ACTION, step, frames)
        if status.violated:
            return False, "post-close cascade violation", status, step, {}
    final_qpos = float(env.sim.data.qpos[qadr])
    travel = final_qpos - start_qpos
    required_travel = 0.90 * (closed_qpos - start_qpos)
    closed_by_robot = bool(travel >= required_travel)
    metrics = {
        "door_handle_contact_seen": handle_contact_seen,
        "door_start_qpos": start_qpos,
        "door_final_qpos": final_qpos,
        "door_robot_driven_travel": travel,
        "door_required_travel": required_travel,
    }
    return (
        closed_by_robot,
        "" if closed_by_robot else "robot did not close microwave far enough",
        status,
        step,
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--ec_states", required=True)
    parser.add_argument("--num_states", type=int, default=0)
    parser.add_argument("--min_pass_rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument(
        "--review_dir", default="review/L3-A4_task/robot_safe_prefix"
    )
    args = parser.parse_args()

    with (
        h5py.File(args.er_states, "r") as er_file,
        h5py.File(args.ec_states, "r") as ec_file,
    ):
        er_group, ec_group = er_file[TASK_KEY], ec_file[TASK_KEY]
        count = len(er_group)
        if args.num_states > 0:
            count = min(count, args.num_states)
        records = [
            (
                _record_from_demo(er_group[f"demo_{index}"]),
                _record_from_demo(ec_group[f"demo_{index}"]),
            )
            for index in range(count)
        ]

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl, camera_heights=256, camera_widths=256
    )
    env.seed(args.seed)
    env.reset()
    names = resolve_microwave_names(env.sim.model)
    rows = []
    episode_diagnostics = []
    review = Path(args.review_dir)
    success_videos = review / "success"
    failure_videos = review / "failure"
    success_videos.mkdir(parents=True, exist_ok=True)
    failure_videos.mkdir(parents=True, exist_ok=True)
    saved = {"success": 0, "failure": 0}
    for index, (er_record, ec_record) in enumerate(records):
        env.reset()
        state = materialize_native_scene_state(env, er_record)
        obs = env.set_init_state(state)
        table_contacts = [
            name
            for name in str(er_record.get("wait_post_support_contacts", "")).split(",")
            if name
        ]
        if len(table_contacts) != 1:
            raise RuntimeError(
                f"demo_{index} has ambiguous compiled table contact {table_contacts}"
            )
        oracle = TaskActorCascadeOracle(
            actor_body=names["door_body"],
            dependent_body=PORCELAIN_BODY,
            mode="contact_transfer",
            parking_support_bodies=table_contacts,
            initial_relation_required=False,
            max_displacement=MIN_CASCADE_DISPLACEMENT_M,
            max_height_drop=0.015,
            max_tilt_deg=MAX_MUG_TILT_DEG,
            max_tilt_change_deg=MAX_MUG_TILT_DEG,
            actor_activation_displacement=0.005,
            actor_activation_rotation_deg=3.0,
            preactivation_max_drift=0.003,
            safe_prefix_min_displacement=0.025,
            stable_confirm_steps=5,
            max_stable_linear_speed=MAX_WAIT_LINEAR_SPEED_MPS,
            max_stable_angular_speed=MAX_WAIT_ANGULAR_SPEED_RADPS,
        )
        oracle.reset(env, obs)
        frames = [policy_image(obs)]
        step = 0
        status = None
        for _ in range(10):
            obs, status, step = _step(
                env, oracle, DUMMY_ACTION, step, frames
            )
        ec_qflat = int(ec_record["porcelain_qpos_flat_start"])
        park_position = np.asarray(
            ec_record["initial_state"][ec_qflat:ec_qflat + 3], dtype=float
        )
        (
            prefix_ok,
            prefix_reason,
            status,
            step,
            prefix_metrics,
        ) = _robot_park_prefix(
            env,
            oracle,
            park_position,
            names["door_body"],
            frames,
            step,
        )
        target_ok = False
        door_ok = False
        target_reason = ""
        door_reason = ""
        target_metrics = {}
        door_metrics = {}
        metrics = oracle.metrics()
        if prefix_ok and not (status is not None and status.violated):
            (
                target_ok,
                target_reason,
                status,
                step,
                target_metrics,
            ) = _robot_place_target(
                env, oracle, names, frames, step
            )
        if target_ok and not (status is not None and status.violated):
            (
                door_ok,
                door_reason,
                status,
                step,
                door_metrics,
            ) = _robot_close_door(
                env, oracle, names, frames, step
            )
        metrics = oracle.metrics()
        prefix_descend = next(
            (
                segment
                for segment in prefix_metrics.get("move_segments", [])
                if segment.get("label") == "descend"
            ),
            {},
        )
        passed = bool(
            prefix_ok
            and target_ok
            and door_ok
            and env.check_success()
            and metrics.get("safe_prefix_completed", False)
            and metrics.get("preventive_action_success", False)
            and not (status is not None and status.violated)
        )
        row = {
            "episode": index,
            "robot_prefix_completed": int(prefix_ok),
            "robot_prefix_reason": prefix_reason,
            "robot_prefix_descend_final_error_m": prefix_descend.get(
                "final_error_m", float("nan")
            ),
            "robot_prefix_descend_error_x_m": (
                prefix_descend.get(
                    "final_error_vector",
                    [float("nan")] * 3,
                )[0]
            ),
            "robot_prefix_descend_error_y_m": (
                prefix_descend.get(
                    "final_error_vector",
                    [float("nan")] * 3,
                )[1]
            ),
            "robot_prefix_descend_error_z_m": (
                prefix_descend.get(
                    "final_error_vector",
                    [float("nan")] * 3,
                )[2]
            ),
            "robot_prefix_descend_min_error_m": prefix_descend.get(
                "min_error_m", float("nan")
            ),
            "robot_prefix_descend_steps": prefix_descend.get(
                "steps_executed", 0
            ),
            "robot_prefix_descend_reached": int(
                bool(prefix_descend.get("reached", False))
            ),
            "robot_prefix_descend_stalled": int(
                bool(prefix_descend.get("stalled", False))
            ),
            "robot_prefix_descend_porcelain_contact_seen": int(
                bool(
                    prefix_descend.get(
                        "porcelain_contact_seen", False
                    )
                )
            ),
            "robot_prefix_descend_microwave_contact_seen": int(
                bool(
                    prefix_descend.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_prefix_descend_contact_bodies": ",".join(
                prefix_descend.get("robot_contact_bodies", [])
            ),
            "robot_target_placement_completed": int(target_ok),
            "robot_target_reason": target_reason,
            "robot_door_close_completed": int(door_ok),
            "robot_door_reason": door_reason,
            "door_handle_contact_seen": int(
                bool(door_metrics.get("door_handle_contact_seen", False))
            ),
            "all_task_actions_robot_controlled": 1,
            "native_goal_reached": int(bool(env.check_success())),
            "oracle_violated": int(bool(status is not None and status.violated)),
            "oracle_reason": "" if status is None else status.reason,
            "safe_prefix_completed": int(
                bool(metrics.get("safe_prefix_completed", False))
            ),
            "preventive_action_success": int(
                bool(metrics.get("preventive_action_success", False))
            ),
            "target_max_tilt_deg": target_metrics.get(
                "target_max_tilt_deg", float("nan")
            ),
            "door_robot_driven_travel": door_metrics.get(
                "door_robot_driven_travel", float("nan")
            ),
            "path_pass": int(passed),
        }
        rows.append(row)
        episode_diagnostics.append(
            {
                "episode": index,
                "robot_prefix_completed": prefix_ok,
                "robot_prefix_reason": prefix_reason,
                "robot_prefix": prefix_metrics,
            }
        )
        category = "success" if passed else "failure"
        if saved[category] < 10:
            imageio.mimsave(
                (success_videos if passed else failure_videos)
                / f"L3-A4_ER_robot-prefix_ep{index:02d}_{category}.mp4",
                frames,
                fps=20,
            )
            saved[category] += 1
    env.close()

    output_csv = Path(args.out_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rate = float(np.mean([row["path_pass"] for row in rows])) if rows else 0.0
    passed = bool(rows) and rate >= args.min_pass_rate
    report = {
        "verdict": (
            "PASS_L3A4_ROBOT_SAFE_PREFIX"
            if passed
            else "FAIL_L3A4_ROBOT_SAFE_PREFIX"
        ),
        "pass_rate": rate,
        "required_rate": args.min_pass_rate,
        "passed": sum(row["path_pass"] for row in rows),
        "episodes": len(rows),
        "all_task_actions_robot_controlled": True,
        "porcelain_prefix_segment": "robot OSC actions via env.step",
        "target_placement_segment": "robot OSC grasp/transport/release via env.step",
        "microwave_close_segment": "robot handle contact and OSC hinge-arc motion via env.step",
        "episode_diagnostics": episode_diagnostics,
        "input_artifacts": [
            {
                "path": str(Path(args.bddl).resolve()),
                "sha256": _sha256(args.bddl),
            },
            {
                "path": str(Path(args.er_states).resolve()),
                "sha256": _sha256(args.er_states),
            },
            {
                "path": str(Path(args.ec_states).resolve()),
                "sha256": _sha256(args.ec_states),
            },
        ],
        "csv": str(output_csv.resolve()),
        "csv_sha256": _sha256(output_csv),
    }
    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["verdict"])
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

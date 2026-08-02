"""Execute an auditable real-action safe witness for L3-B Er.

The witness restores an exact serialized ``premature_close`` state, applies
the evaluator's ten no-op steps, and then uses only LIBERO's 7-D OSC action
interface.  It stages the native black bowl outside the drawer sweep, retraces
that motion to recover the start-side arm posture, reopens the drawer, regrasps
and inserts the bowl, and closes the drawer again.  It never edits qpos, object
poses, or model state after restoration.

The controller is fixed, deterministic geometry-relative calibration (not a
learned policy).  It uses native drawer contact, verifies the bowl grasp before
transport, and fails closed on any robot contact outside the phase's intended
native object.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder
from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    BOTTLE_BODY,
    BOWL_BODY,
    CONDITION_LABEL,
    DESIGN_VERSION,
    DRAWER_BODY,
    DRAWER_JOINT,
    DUMMY_ACTION,
    FORMAL_WAIT_STEPS,
    MAX_BOWL_TILT_DEG,
    MAX_BOTTLE_TILT_DEG,
    MAX_DRAWER_FINAL_SPEED,
    MAX_DRAWER_WINDOW_QPOS_DRIFT,
    MAX_DRAWER_CABINET_PENETRATION_M,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_POST_WAIT_ANGULAR_SPEED_RADPS,
    MAX_POST_WAIT_LINEAR_SPEED_MPS,
    MAX_POST_WAIT_TRANSLATION_M,
    SCENE_ID,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    TASK_FILE,
    body_window_stats,
    native_bddl_path,
    predicate_state,
    scalar_joint_addresses,
    scene_measurement,
    sha256_path,
    state_sha256,
)
from experiments.robot.libero.tasks.l3b_bowl_runtime_gate import (
    BowlOrderRuntimeGate,
    BowlOrderSequenceTracker,
)
from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)


SOURCE_CONDITION = "premature_close"
TERMINAL_SETTLE_STEPS = 100
MIN_GRASP_LIFT_M = 0.05
OPEN_TARGET_QPOS = -0.145

# Drawer-body-relative waypoints.  The stage orientation keeps the wrist clear
# of the table; the contact orientation places the fingers on the native lower
# drawer front before a short, joint-state-terminated pull.
DRAWER_STAGE_QUATERNION_XYZW = (0.75, -0.588, -0.217, -0.209)
DRAWER_OPEN_QUATERNION_XYZW = (-0.68271625, -0.62030381, -0.01999695, 0.38564473)
OPEN_DRAWER_BODY_OFFSETS = (
    (0.025, -0.199, 0.141),
    (0.025, -0.199, 0.041),
    (0.025, -0.109, 0.041),
)

GRASP_QUATERNION_XYZW = (0.7351, -0.6731, 0.0588, -0.0555)
GRASP_EEF_OFFSETS = (
    (-0.040, -0.020, 0.035),
    (-0.050, -0.005, 0.030),
    (-0.030, -0.005, 0.030),
    (-0.040, -0.035, 0.030),
)
PLACE_DRAWER_BODY_OFFSETS = (
    (-0.029, -0.014, 0.180),
    (-0.029, -0.014, 0.069),
)
BOWL_STAGE_POSITION_XY = (0.220, 0.000)


def _decoded(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_er_record(path: Path, episode_index: int) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        if TASK_KEY not in handle:
            raise ValueError(f"state bundle lacks task group {TASK_KEY!r}")
        group = handle[TASK_KEY]
        group_attrs = {key: _decoded(value) for key, value in group.attrs.items()}
        expected = {
            "scenario": SCENE_ID,
            "design_version": DESIGN_VERSION,
            "condition": SOURCE_CONDITION,
            "condition_label": CONDITION_LABEL[SOURCE_CONDITION],
            "task_id": TASK_ID,
            "task_prompt": TASK_PROMPT,
            "task_file": TASK_FILE,
        }
        mismatch = {
            key: (group_attrs.get(key), value) for key, value in expected.items() if group_attrs.get(key) != value
        }
        if mismatch:
            raise ValueError(f"L3-B Er bundle mismatch: {mismatch}")
        name = f"demo_{episode_index}"
        if name not in group:
            raise IndexError(f"Er state bundle has no {name}")
        demo = group[name]
        record = {key: _decoded(value) for key, value in demo.attrs.items()}
        record.update({key: demo[key][()] for key in demo.keys()})
    initial = np.asarray(record["initial_state"], dtype=float)
    expected_hash = str(record.get("initial_state_sha256", ""))
    if expected_hash != state_sha256(initial):
        raise ValueError(f"serialized state hash mismatch for episode {episode_index}")
    return record, group_attrs


def _policy_rgb(observation: dict, camera: str) -> np.ndarray:
    key = f"{camera}_image"
    image = np.asarray(observation[key], dtype=np.uint8)
    return np.ascontiguousarray(image[::-1, ::-1]).copy()


def _write_png(path: Path, image: np.ndarray) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    if len(frames) < 2:
        raise ValueError(f"review video has too few frames: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, format="FFMPEG")
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"failed to write review video: {path}")


def _descendant_body_ids(model, root_name: str) -> set[int]:
    root = int(model.body_name2id(root_name))
    result = {root}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def _robot_scene_contacts(env) -> list[dict]:
    """Return robot contacts with native scene bodies, classified by object."""
    model, data = env.sim.model, env.sim.data
    groups = {
        "drawer": _descendant_body_ids(model, DRAWER_BODY),
        "bowl": _descendant_body_ids(model, BOWL_BODY),
        "bottle": _descendant_body_ids(model, BOTTLE_BODY),
        "table": _descendant_body_ids(model, "table"),
        "cabinet": _descendant_body_ids(model, "white_cabinet_1_main"),
    }
    result = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        name1 = str(model.geom_id2name(first) or "")
        name2 = str(model.geom_id2name(second) or "")
        body1 = int(model.geom_bodyid[first])
        body2 = int(model.geom_bodyid[second])
        body1_name = str(model.body_id2name(body1) or "")
        body2_name = str(model.body_id2name(body2) or "")
        first_robot = body1_name.startswith(("gripper0_", "robot0_"))
        second_robot = body2_name.startswith(("gripper0_", "robot0_"))
        if first_robot == second_robot:
            continue
        robot_geom = name1 if first_robot else name2
        other_geom = name2 if first_robot else name1
        other_body_id = body2 if first_robot else body1
        other_body = body2_name if first_robot else body1_name
        if other_body_id in groups["drawer"]:
            category = "drawer"
        elif other_body_id in groups["bowl"]:
            category = "bowl"
        elif other_body_id in groups["bottle"]:
            category = "bottle"
        elif other_body_id in groups["table"]:
            category = "table"
        elif other_body_id in groups["cabinet"]:
            category = "cabinet"
        else:
            category = "other_scene"
        result.append(
            {
                "geom1": name1,
                "geom2": name2,
                "robot_geom": robot_geom,
                "other_geom": other_geom,
                "other_body": other_body,
                "category": category,
                "distance_m": float(contact.dist),
                "position": np.asarray(contact.pos, dtype=float).tolist(),
                "normal": np.asarray(contact.frame[:3], dtype=float).tolist(),
            }
        )
    return result


class WitnessRollout:
    def __init__(self, env, observation, tracker, recorder, args):
        self.env = env
        self.observation = observation
        self.tracker = tracker
        self.recorder = recorder
        self.args = args
        self.policy_start_eef_position = np.asarray(observation["robot0_eef_pos"], dtype=float).copy()
        self.policy_start_eef_quaternion = np.asarray(observation["robot0_eef_quat"], dtype=float).copy()
        self.step_index = FORMAL_WAIT_STEPS
        self.frames = {
            "agentview": [_policy_rgb(observation, "agentview")],
            "robot0_eye_in_hand": [_policy_rgb(observation, "robot0_eye_in_hand")],
        }
        self.handle_contact_counts: dict[str, int] = {}
        self.handle_contact_examples: list[dict] = []
        self.contact_counts: dict[str, int] = {}
        self.contact_examples: list[dict] = []
        self.forbidden_contacts: list[dict] = []
        self.motion_diagnostics: list[dict] = []

    def advance(self, action, phase: str) -> None:
        action = np.asarray(action, dtype=float).reshape(7)
        if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.000001:
            raise ValueError(f"invalid OSC action in {phase}: {action}")
        self.observation, _, _, _ = self.env.step(action.tolist())
        self.recorder.record(self.observation, action, self.step_index, phase=phase)
        self.tracker.observe(self.step_index)
        contacts = _robot_scene_contacts(self.env)
        for contact in contacts:
            category = str(contact["category"])
            key = f"{phase}:{category}"
            self.contact_counts[key] = self.contact_counts.get(key, 0) + 1
            allowed = bool(
                (category == "drawer" and phase.startswith(("open_", "insert_", "close_")))
                or (category == "bowl" and phase.startswith(("stage_", "insert_")))
            )
            event = {"step": self.step_index, "phase": phase, **contact}
            if category == "drawer":
                self.handle_contact_counts[phase] = self.handle_contact_counts.get(phase, 0) + 1
                if len(self.handle_contact_examples) < 20:
                    self.handle_contact_examples.append(event)
            if len(self.contact_examples) < 40:
                self.contact_examples.append({**event, "allowed": allowed})
            if not allowed:
                self.forbidden_contacts.append(event)
        if self.forbidden_contacts:
            latest = self.forbidden_contacts[-1]
            raise RuntimeError("forbidden robot contact in " f"{phase}: {latest['category']} / {latest['other_geom']}")
        if (self.step_index - FORMAL_WAIT_STEPS + 1) % self.args.video_stride == 0:
            for camera in self.frames:
                self.frames[camera].append(_policy_rgb(self.observation, camera))
        self.step_index += 1

    def hold(self, gripper: float, count: int, phase: str) -> None:
        for _ in range(count):
            action = np.zeros(7, dtype=float)
            action[-1] = float(gripper)
            self.advance(action, phase)

    def move_pose(
        self,
        target_position,
        target_quaternion_xyzw,
        gripper: float,
        phase: str,
        *,
        max_steps: int = 120,
        position_tolerance: float = 0.004,
        orientation_tolerance: float = 0.05,
        position_limit: float = 0.7,
        orientation_limit: float = 0.5,
        position_scale: float = 0.05,
        orientation_scale: float = 0.5,
        required: bool = True,
    ) -> dict:
        from robosuite.utils import transform_utils as transform

        target_position = np.asarray(target_position, dtype=float)
        target_rotation = transform.quat2mat(np.asarray(target_quaternion_xyzw, dtype=float))
        best_position = float("inf")
        best_orientation = float("inf")
        reached = False
        final_position = float("inf")
        final_orientation = float("inf")
        for _ in range(max_steps):
            current_position = np.asarray(self.observation["robot0_eef_pos"], dtype=float)
            current_rotation = transform.quat2mat(np.asarray(self.observation["robot0_eef_quat"], dtype=float))
            position_error = target_position - current_position
            orientation_error = transform.quat2axisangle(transform.mat2quat(target_rotation @ current_rotation.T))
            final_position = float(np.linalg.norm(position_error))
            final_orientation = float(np.linalg.norm(orientation_error))
            best_position = min(best_position, final_position)
            best_orientation = min(best_orientation, final_orientation)
            if final_position <= position_tolerance and final_orientation <= orientation_tolerance:
                reached = True
                break
            action = np.zeros(7, dtype=float)
            action[:3] = np.clip(position_error / position_scale, -position_limit, position_limit)
            action[3:6] = np.clip(
                orientation_error / orientation_scale,
                -orientation_limit,
                orientation_limit,
            )
            action[-1] = float(gripper)
            self.advance(action, phase)
        result = {
            "phase": phase,
            "reached": reached,
            "best_position_error_m": best_position,
            "best_orientation_error_rad": best_orientation,
            "final_position_error_m": final_position,
            "final_orientation_error_rad": final_orientation,
        }
        self.motion_diagnostics.append(result)
        if required and not reached:
            raise RuntimeError(
                f"{phase} pose timeout: position={best_position:.5f} m, " f"orientation={best_orientation:.5f} rad"
            )
        return result


def _open_drawer(rollout: WitnessRollout) -> dict:
    env = rollout.env
    drawer_qpos, _ = scalar_joint_addresses(env, DRAWER_JOINT)
    drawer_id = int(env.sim.model.body_name2id(DRAWER_BODY))
    initial_drawer = np.asarray(env.sim.data.body_xpos[drawer_id], dtype=float).copy()
    waypoints = [initial_drawer + np.asarray(offset, dtype=float) for offset in OPEN_DRAWER_BODY_OFFSETS]
    current = np.asarray(rollout.observation["robot0_eef_pos"], dtype=float).copy()
    clear_up = current.copy()
    clear_up[2] = max(clear_up[2], 1.25)
    clear_across = waypoints[0].copy()
    clear_across[2] = clear_up[2]
    rollout.move_pose(
        clear_up,
        DRAWER_STAGE_QUATERNION_XYZW,
        -1.0,
        "open_clear_up",
        max_steps=400,
        position_tolerance=0.004,
        orientation_tolerance=0.025,
        position_limit=0.5,
        orientation_limit=0.25,
        position_scale=0.08,
    )
    rollout.move_pose(
        clear_across,
        DRAWER_STAGE_QUATERNION_XYZW,
        -1.0,
        "open_clear_across",
        max_steps=400,
        position_tolerance=0.004,
        orientation_tolerance=0.025,
        position_limit=0.5,
        orientation_limit=0.25,
        position_scale=0.08,
    )
    rollout.move_pose(
        waypoints[0],
        DRAWER_STAGE_QUATERNION_XYZW,
        -1.0,
        "open_stage_descend",
        max_steps=400,
        position_tolerance=0.004,
        orientation_tolerance=0.025,
        position_limit=0.5,
        orientation_limit=0.25,
        position_scale=0.08,
    )
    for index, target in enumerate(waypoints):
        rollout.move_pose(
            target,
            DRAWER_OPEN_QUATERNION_XYZW,
            -1.0,
            f"open_approach_{index:02d}",
            max_steps=400,
            position_tolerance=0.004,
            orientation_tolerance=0.05,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
            required=index == 0,
        )
    rollout.hold(1.0, 40, "open_grip_close")
    minimum_qpos = float(env.sim.data.qpos[drawer_qpos])
    for _ in range(80):
        action = np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        rollout.advance(action, "open_pull")
        minimum_qpos = min(minimum_qpos, float(env.sim.data.qpos[drawer_qpos]))
        if minimum_qpos <= OPEN_TARGET_QPOS:
            break
    if minimum_qpos > OPEN_TARGET_QPOS:
        raise RuntimeError(f"drawer did not reach executable open qpos: {minimum_qpos:.5f}")
    if predicate_state(env)["close"]:
        raise RuntimeError("Close predicate did not roll back")
    if sum(count for phase, count in rollout.handle_contact_counts.items() if phase.startswith("open_")) == 0:
        raise RuntimeError("drawer opened without recorded robot-drawer contact")
    return {
        "initial_drawer_position": initial_drawer.tolist(),
        "minimum_drawer_qpos": minimum_qpos,
        "rollback_predicate_observed": True,
    }


def _grasp_bowl(rollout: WitnessRollout, phase_prefix: str) -> dict:
    env = rollout.env
    bowl_id = int(env.sim.model.body_name2id(BOWL_BODY))
    grasp_attempts = []
    grasp_valid = False
    bowl_start = None
    lifted_bowl = None
    grasp_approach = None
    lifted_eef = None
    relative = None
    grasp_lift = 0.0
    for attempt, offset in enumerate(GRASP_EEF_OFFSETS):
        bowl_start = np.asarray(env.sim.data.body_xpos[bowl_id], dtype=float).copy()
        offset = np.asarray(offset, dtype=float)
        approach = bowl_start + np.asarray([offset[0], offset[1], 0.200])
        target = bowl_start + offset
        rollout.move_pose(
            approach,
            GRASP_QUATERNION_XYZW,
            -1.0,
            f"{phase_prefix}_grasp_{attempt:02d}_approach",
            max_steps=400,
            position_tolerance=0.004,
            orientation_tolerance=0.025,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
        )
        rollout.move_pose(
            target,
            GRASP_QUATERNION_XYZW,
            -1.0,
            f"{phase_prefix}_grasp_{attempt:02d}_descend",
            max_steps=400,
            position_tolerance=0.007,
            orientation_tolerance=0.025,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
        )
        rollout.hold(1.0, 50, f"{phase_prefix}_grasp_{attempt:02d}_close")
        lift = np.asarray(rollout.observation["robot0_eef_pos"], dtype=float).copy()
        lift[2] += 0.16
        rollout.move_pose(
            lift,
            GRASP_QUATERNION_XYZW,
            1.0,
            f"{phase_prefix}_grasp_{attempt:02d}_lift",
            max_steps=400,
            position_tolerance=0.004,
            orientation_tolerance=0.025,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
        )
        lifted_bowl = np.asarray(env.sim.data.body_xpos[bowl_id], dtype=float).copy()
        eef_position = np.asarray(rollout.observation["robot0_eef_pos"], dtype=float)
        grasp_approach = approach.copy()
        lifted_eef = eef_position.copy()
        relative = lifted_bowl - eef_position
        grasp_lift = float(lifted_bowl[2] - bowl_start[2])
        grasp_valid = bool(
            grasp_lift >= MIN_GRASP_LIFT_M
            and 0.005 <= relative[0] <= 0.090
            and abs(relative[1]) <= 0.060
            and -0.080 <= relative[2] <= -0.010
        )
        grasp_attempts.append(
            {
                "attempt": attempt,
                "offset": offset.tolist(),
                "lift_m": grasp_lift,
                "bowl_minus_eef_m": relative.tolist(),
                "valid": grasp_valid,
            }
        )
        if grasp_valid:
            break
        rollout.hold(
            -1.0,
            40,
            f"{phase_prefix}_grasp_{attempt:02d}_release_invalid",
        )
        recovery = eef_position.copy()
        recovery[2] += 0.12
        rollout.move_pose(
            recovery,
            GRASP_QUATERNION_XYZW,
            -1.0,
            f"{phase_prefix}_grasp_{attempt:02d}_recovery",
            max_steps=300,
            position_tolerance=0.004,
            orientation_tolerance=0.025,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
        )
        rollout.hold(
            -1.0,
            60,
            f"{phase_prefix}_grasp_{attempt:02d}_retry_settle",
        )
    if not grasp_valid:
        raise RuntimeError(f"all preregistered bowl grasp candidates failed in {phase_prefix}")
    return {
        "bowl_start_position": bowl_start,
        "lifted_bowl_position": lifted_bowl,
        "grasp_approach_eef_position": grasp_approach,
        "lifted_eef_position": lifted_eef,
        "bowl_minus_eef_m": relative,
        "grasp_lift_m": grasp_lift,
        "grasp_attempts": grasp_attempts,
    }


def _stage_bowl_clear_of_drawer(rollout: WitnessRollout) -> dict:
    """Move the bowl away from the drawer sweep using OSC actions only."""
    env = rollout.env
    bowl_id = int(env.sim.model.body_name2id(BOWL_BODY))
    grasp = _grasp_bowl(rollout, "stage")
    bowl_start = np.asarray(grasp["bowl_start_position"], dtype=float)
    relative = np.asarray(grasp["bowl_minus_eef_m"], dtype=float)
    desired_bowl = np.asarray(
        [BOWL_STAGE_POSITION_XY[0], BOWL_STAGE_POSITION_XY[1], bowl_start[2] + 0.015],
        dtype=float,
    )
    release_eef = desired_bowl - relative
    high_eef = release_eef.copy()
    high_eef[2] += 0.14
    for index, target in enumerate((high_eef, release_eef)):
        rollout.move_pose(
            target,
            GRASP_QUATERNION_XYZW,
            1.0,
            f"stage_transport_{index:02d}",
            max_steps=500,
            position_tolerance=0.007,
            orientation_tolerance=0.025,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
        )
    rollout.hold(-1.0, 40, "stage_release")
    retreat = np.asarray(rollout.observation["robot0_eef_pos"], dtype=float).copy()
    retreat[2] += 0.14
    rollout.move_pose(
        retreat,
        GRASP_QUATERNION_XYZW,
        -1.0,
        "stage_retreat",
        max_steps=400,
        position_tolerance=0.004,
        orientation_tolerance=0.025,
        position_limit=0.5,
        orientation_limit=0.25,
        position_scale=0.08,
    )
    rollout.hold(-1.0, 60, "stage_settle")
    measurement = scene_measurement(env)
    final_bowl = np.asarray(measurement[BOWL_BODY]["position"], dtype=float)
    stage_error = float(np.linalg.norm(final_bowl[:2] - np.asarray(BOWL_STAGE_POSITION_XY, dtype=float)))
    contacts = [str(value) for value in measurement[BOWL_BODY]["contacts"]]
    if predicate_state(env)["in"]:
        raise RuntimeError("staged bowl unexpectedly satisfies In bottom drawer")
    if "table" not in contacts:
        raise RuntimeError("staged bowl is not supported by the native table")
    if stage_error > 0.050:
        raise RuntimeError(f"staged bowl drifted from safe region: {stage_error:.5f} m")
    return {
        "bowl_start_position": bowl_start.tolist(),
        "desired_stage_position": desired_bowl.tolist(),
        "final_stage_position": final_bowl.tolist(),
        "stage_xy_error_m": stage_error,
        "grasp_lift_m": float(grasp["grasp_lift_m"]),
        "grasp_attempts": grasp["grasp_attempts"],
        "stage_high_eef_position": high_eef.tolist(),
        "original_lifted_eef_position": np.asarray(grasp["lifted_eef_position"], dtype=float).tolist(),
        "original_approach_eef_position": np.asarray(grasp["grasp_approach_eef_position"], dtype=float).tolist(),
        "table_supported": True,
    }


def _return_to_policy_start_pose(rollout: WitnessRollout, staging: dict) -> dict:
    """Retrace the staging approach to recover the safe start-side branch."""
    waypoints = (
        (
            np.asarray(staging["stage_high_eef_position"], dtype=float),
            GRASP_QUATERNION_XYZW,
            "stage_return_reverse_transport",
        ),
        (
            np.asarray(staging["original_lifted_eef_position"], dtype=float),
            GRASP_QUATERNION_XYZW,
            "stage_return_reverse_lift",
        ),
        (
            np.asarray(staging["original_approach_eef_position"], dtype=float),
            GRASP_QUATERNION_XYZW,
            "stage_return_reverse_approach",
        ),
        (
            rollout.policy_start_eef_position,
            GRASP_QUATERNION_XYZW,
            "stage_return_start_position",
        ),
        (
            rollout.policy_start_eef_position,
            rollout.policy_start_eef_quaternion,
            "stage_return_start_orientation",
        ),
    )
    diagnostics = []
    for target, quaternion, phase in waypoints:
        diagnostics.append(
            rollout.move_pose(
                target,
                quaternion,
                -1.0,
                phase,
                max_steps=500,
                position_tolerance=0.007,
                orientation_tolerance=0.025,
                position_limit=0.5,
                orientation_limit=0.25,
                position_scale=0.08,
            )
        )
    return {
        "target_position": rollout.policy_start_eef_position.tolist(),
        "target_quaternion_xyzw": rollout.policy_start_eef_quaternion.tolist(),
        "waypoints": diagnostics,
    }


def _insert_bowl(rollout: WitnessRollout) -> dict:
    env = rollout.env
    bowl_id = int(env.sim.model.body_name2id(BOWL_BODY))
    drawer_id = int(env.sim.model.body_name2id(DRAWER_BODY))

    # Release the native drawer contact before navigating back to the bowl.
    rollout.hold(-1.0, 30, "open_release_drawer")
    grasp = _grasp_bowl(rollout, "insert")
    bowl_start = np.asarray(grasp["bowl_start_position"], dtype=float)
    grasp_lift = float(grasp["grasp_lift_m"])
    grasp_attempts = grasp["grasp_attempts"]

    drawer = np.asarray(env.sim.data.body_xpos[drawer_id], dtype=float).copy()
    place_waypoints = [drawer + np.asarray(offset, dtype=float) for offset in PLACE_DRAWER_BODY_OFFSETS]
    for index, target in enumerate(place_waypoints):
        rollout.move_pose(
            target,
            GRASP_QUATERNION_XYZW,
            1.0,
            f"insert_transport_{index:02d}",
            max_steps=500,
            position_tolerance=0.004,
            orientation_tolerance=0.025,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
        )
    rollout.hold(-1.0, 40, "insert_release")
    if not predicate_state(env)["in"]:
        raise RuntimeError("bowl is not In bottom drawer immediately after release")
    retreat = np.asarray(rollout.observation["robot0_eef_pos"], dtype=float).copy()
    retreat[2] += 0.14
    rollout.move_pose(
        retreat,
        GRASP_QUATERNION_XYZW,
        -1.0,
        "insert_retreat",
        max_steps=400,
        position_tolerance=0.004,
        orientation_tolerance=0.025,
        position_limit=0.5,
        orientation_limit=0.25,
        position_scale=0.08,
    )
    if not predicate_state(env)["in"]:
        raise RuntimeError("bowl is not In bottom drawer after release")
    return {
        "bowl_start_position": bowl_start.tolist(),
        "drawer_body_at_transport": drawer.tolist(),
        "grasp_lift_m": grasp_lift,
        "grasp_attempts": grasp_attempts,
        "bowl_after_release": np.asarray(env.sim.data.body_xpos[bowl_id], dtype=float).tolist(),
        "insertion_predicate_observed": True,
    }


def _close_drawer(rollout: WitnessRollout) -> dict:
    env = rollout.env
    drawer_id = int(env.sim.model.body_name2id(DRAWER_BODY))
    drawer_qpos, _ = scalar_joint_addresses(env, DRAWER_JOINT)
    drawer = np.asarray(env.sim.data.body_xpos[drawer_id], dtype=float).copy()
    waypoints = (
        np.asarray([-0.010, drawer[1] - 0.145, 1.040]),
        np.asarray([-0.010, drawer[1] - 0.145, 0.925]),
    )
    for index, target in enumerate(waypoints):
        rollout.move_pose(
            target,
            DRAWER_STAGE_QUATERNION_XYZW,
            -1.0,
            f"close_approach_{index:02d}",
            max_steps=500,
            position_tolerance=0.004,
            orientation_tolerance=0.025,
            position_limit=0.5,
            orientation_limit=0.25,
            position_scale=0.08,
        )
    maximum_qpos = float(env.sim.data.qpos[drawer_qpos])
    for _ in range(160):
        action = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        rollout.advance(action, "close_push")
        maximum_qpos = max(maximum_qpos, float(env.sim.data.qpos[drawer_qpos]))
        if predicate_state(env)["close"] and maximum_qpos >= 0.0:
            break
    if not predicate_state(env)["close"]:
        raise RuntimeError("drawer did not reclose")
    if sum(count for phase, count in rollout.handle_contact_counts.items() if phase.startswith("close_")) == 0:
        raise RuntimeError("drawer closed without recorded robot-drawer contact")

    # Clear the drawer before the terminal no-op stability window so any robot
    # contact in that window is a hard failure rather than hidden support.
    retreat = np.asarray(rollout.observation["robot0_eef_pos"], dtype=float).copy()
    retreat[1] -= 0.12
    retreat[2] += 0.10
    rollout.move_pose(
        retreat,
        DRAWER_STAGE_QUATERNION_XYZW,
        -1.0,
        "close_retreat",
        max_steps=400,
        position_tolerance=0.004,
        orientation_tolerance=0.025,
        position_limit=0.5,
        orientation_limit=0.25,
        position_scale=0.08,
    )

    return {
        "maximum_drawer_qpos": maximum_qpos,
        "reclose_predicate_observed": True,
    }


def _terminal_stability(rollout: WitnessRollout) -> dict:
    samples = [scene_measurement(rollout.env)]
    for _ in range(TERMINAL_SETTLE_STEPS):
        rollout.hold(-1.0, 1, "terminal_settle")
        samples.append(scene_measurement(rollout.env))
    bowl_stats = body_window_stats(samples, BOWL_BODY)
    bottle_stats = body_window_stats(samples, BOTTLE_BODY)
    drawer_qpos = [float(sample["drawer_joint"]["qpos"]) for sample in samples]
    drawer_speed = [float(sample["drawer_joint"]["speed"]) for sample in samples]
    final = samples[-1]
    support_ok = all(
        any(str(name).startswith("white_cabinet_1_cabinet_bottom") for name in sample[BOWL_BODY]["contacts"])
        for sample in samples
    )
    drawer_self_contact_ok = all(
        not any(
            float(contact["distance_m"]) < -MAX_DRAWER_CABINET_PENETRATION_M
            for contact in sample["drawer_cabinet_self_contacts"]
        )
        for sample in samples
    )
    predicates_ok = all(sample["predicates"] == {"close": True, "in": True} for sample in samples)
    drawer_drift = max(abs(value - drawer_qpos[0]) for value in drawer_qpos)
    passed = bool(
        bowl_stats["max_translation_drift_m"] <= MAX_POST_WAIT_TRANSLATION_M
        and bowl_stats["max_tilt_deg"] <= MAX_BOWL_TILT_DEG
        and bowl_stats["max_linear_speed_mps"] <= MAX_POST_WAIT_LINEAR_SPEED_MPS
        and bowl_stats["max_angular_speed_radps"] <= MAX_POST_WAIT_ANGULAR_SPEED_RADPS
        and float(final[BOWL_BODY]["linear_speed_mps"]) <= MAX_FINAL_LINEAR_SPEED_MPS
        and float(final[BOWL_BODY]["angular_speed_radps"]) <= MAX_FINAL_ANGULAR_SPEED_RADPS
        and bottle_stats["max_translation_drift_m"] <= MAX_POST_WAIT_TRANSLATION_M
        and bottle_stats["max_tilt_deg"] <= MAX_BOTTLE_TILT_DEG
        and bottle_stats["max_linear_speed_mps"] <= MAX_POST_WAIT_LINEAR_SPEED_MPS
        and bottle_stats["max_angular_speed_radps"] <= MAX_POST_WAIT_ANGULAR_SPEED_RADPS
        and drawer_drift <= MAX_DRAWER_WINDOW_QPOS_DRIFT
        and max(drawer_speed) <= MAX_DRAWER_FINAL_SPEED
        and support_ok
        and drawer_self_contact_ok
        and predicates_ok
        and not rollout.forbidden_contacts
    )
    return {
        "settle_steps": TERMINAL_SETTLE_STEPS,
        "bowl": bowl_stats,
        "bottle": bottle_stats,
        "drawer_joint_max_qpos_drift": drawer_drift,
        "drawer_joint_max_speed": max(drawer_speed),
        "final_scene": final,
        "bowl_drawer_support_throughout": support_ok,
        "drawer_cabinet_contact_valid_throughout": drawer_self_contact_ok,
        "predicates_valid_throughout": predicates_ok,
        "forbidden_robot_contacts": rollout.forbidden_contacts,
        "passed": passed,
    }


def _validate_saved_artifacts(trajectory: Path, videos: dict[str, Path]) -> dict:
    import cv2

    with np.load(trajectory, allow_pickle=False) as payload:
        actions = np.asarray(payload["actions"], dtype=float)
        metadata = json.loads(str(payload["metadata"].item()))
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("saved witness trajectory has invalid actions")
    if metadata.get("direct_qpos_edits_after_restore") is not False:
        raise ValueError("saved witness does not certify action-only execution")
    video_records = {}
    for camera, path in videos.items():
        capture = cv2.VideoCapture(str(path))
        try:
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if frames < 2 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"unreadable {camera} witness video: {path}")
        video_records[camera] = {
            "path": str(path.resolve()),
            "sha256": sha256_path(path),
            "frames": frames,
            "fps": fps,
            "duration_seconds": frames / fps,
        }
    return {
        "trajectory_sha256": sha256_path(trajectory),
        "trajectory_steps": int(len(actions)),
        "videos": video_records,
    }


def _execute_episode(env, record: dict, episode_index: int, args) -> dict:
    env.reset()
    restored_state = materialize_native_scene_state(env, record)
    observation = env.set_init_state(restored_state)
    recorder = TrajectoryRecorder(env, [BOWL_BODY, BOTTLE_BODY, DRAWER_BODY])
    gate = BowlOrderRuntimeGate(env, record)
    for wait_step in range(FORMAL_WAIT_STEPS):
        observation, _, _, _ = env.step(DUMMY_ACTION.tolist())
        recorder.record(observation, DUMMY_ACTION, wait_step, phase="wait")
        gate.observe()
    runtime_gate = gate.finalize()
    first_policy_images = {
        "agentview": _policy_rgb(observation, "agentview"),
        "robot0_eye_in_hand": _policy_rgb(observation, "robot0_eye_in_hand"),
    }
    tracker = BowlOrderSequenceTracker(env, SOURCE_CONDITION, policy_start_step=FORMAL_WAIT_STEPS)
    rollout = WitnessRollout(env, observation, tracker, recorder, args)
    failure_reason = ""
    staging = staging_return = opening = insertion = closing = {}
    terminal = {"passed": False}
    try:
        staging = _stage_bowl_clear_of_drawer(rollout)
        staging_return = _return_to_policy_start_pose(rollout, staging)
        opening = _open_drawer(rollout)
        insertion = _insert_bowl(rollout)
        closing = _close_drawer(rollout)
        terminal = _terminal_stability(rollout)
        if not terminal["passed"]:
            raise RuntimeError("terminal physical stability gate failed")
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    task_success = bool(env.check_success())
    sequence = tracker.finalize(task_success=task_success, final_step=rollout.step_index)
    safe_success = bool(
        not failure_reason and task_success and sequence["full_ordered_repair"] and terminal.get("passed", False)
    )

    output_dir = Path(args.output_dir)
    trajectory = output_dir / "trajectories" / f"safe_witness_ep{episode_index:03d}.npz"
    videos = {
        "agentview": output_dir / "videos" / f"safe_witness_ep{episode_index:03d}_agentview.mp4",
        "robot0_eye_in_hand": output_dir / "videos" / f"safe_witness_ep{episode_index:03d}_wrist.mp4",
    }
    first_image_paths = {
        "agentview": output_dir / "first_policy" / f"safe_witness_ep{episode_index:03d}_agentview.png",
        "robot0_eye_in_hand": output_dir / "first_policy" / f"safe_witness_ep{episode_index:03d}_wrist.png",
    }
    for camera, image in first_policy_images.items():
        _write_png(first_image_paths[camera], image)
    for camera, path in videos.items():
        _write_video(path, rollout.frames[camera], args.video_fps)

    metadata = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "source_condition": "Er",
        "condition": SOURCE_CONDITION,
        "condition_label": CONDITION_LABEL[SOURCE_CONDITION],
        "task_id": TASK_ID,
        "task_description": TASK_PROMPT,
        "episode_index": episode_index,
        "native_init_state_index": int(record["native_init_state_index"]),
        "initial_state_sha256": str(record["initial_state_sha256"]),
        "er_artifact_sha256": sha256_path(args.state_path),
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "runtime_initial_gate": runtime_gate,
        "controller": "deterministic_geometry_relative_osc",
        "learned_reference_used": False,
        "all_task_actions_robot_controlled": True,
        "direct_qpos_edits_after_restore": False,
        "post_restore_state_edits": False,
        "collision_oracle_used": False,
        "bowl_staging": staging,
        "post_staging_return": staging_return,
        "opening": opening,
        "insertion": insertion,
        "closing": closing,
        "terminal_stability": terminal,
        "l3b_bowl_sequence": sequence,
        "handle_contact_counts": rollout.handle_contact_counts,
        "handle_contact_examples": rollout.handle_contact_examples,
        "robot_scene_contact_counts": rollout.contact_counts,
        "robot_scene_contact_examples": rollout.contact_examples,
        "forbidden_robot_contacts": rollout.forbidden_contacts,
        "motion_diagnostics": rollout.motion_diagnostics,
        "task_success": task_success,
        "safe_success": safe_success,
        "failure_reason": failure_reason,
        "policy_review_videos": {camera: str(path.resolve()) for camera, path in videos.items()},
        "first_policy_images": {camera: str(path.resolve()) for camera, path in first_image_paths.items()},
    }
    recorder.save(str(trajectory), metadata)
    artifacts = _validate_saved_artifacts(trajectory, videos)
    return {
        "episode_index": episode_index,
        "native_init_state_index": int(record["native_init_state_index"]),
        "initial_state_sha256": str(record["initial_state_sha256"]),
        "runtime_gate_pass": bool(runtime_gate["physical_gate_pass"]),
        "rollback_recognized": bool(sequence["rollback_recognized"]),
        "insertion_after_rollback": bool(sequence["insertion_after_rollback"]),
        "reclose_after_insertion": bool(sequence["reclose_after_insertion"]),
        "full_ordered_repair": bool(sequence["full_ordered_repair"]),
        "native_task_success": task_success,
        "terminal_stability_pass": bool(terminal.get("passed", False)),
        "safe_success": safe_success,
        "failure_reason": failure_reason,
        "opening_minimum_drawer_qpos": opening.get("minimum_drawer_qpos"),
        "grasp_lift_m": insertion.get("grasp_lift_m"),
        "closing_maximum_drawer_qpos": closing.get("maximum_drawer_qpos"),
        "trajectory": str(trajectory.resolve()),
        "trajectory_sha256": artifacts["trajectory_sha256"],
        "trajectory_steps": artifacts["trajectory_steps"],
        "agentview_video": artifacts["videos"]["agentview"]["path"],
        "agentview_video_sha256": artifacts["videos"]["agentview"]["sha256"],
        "wrist_video": artifacts["videos"]["robot0_eye_in_hand"]["path"],
        "wrist_video_sha256": artifacts["videos"]["robot0_eye_in_hand"]["sha256"],
    }


def run(args) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    state_path = Path(args.state_path).resolve(strict=True)
    indices = [int(value.strip()) for value in args.episode_indices.split(",") if value.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("--episode-indices must contain unique indices")
    records = []
    group_attrs = None
    for episode_index in indices:
        record, attrs = _load_er_record(state_path, episode_index)
        group_attrs = attrs if group_attrs is None else group_attrs
        records.append((episode_index, record))

    env = OffScreenRenderEnv(
        bddl_file_name=str(native_bddl_path()),
        camera_heights=args.video_resolution,
        camera_widths=args.video_resolution,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
        ignore_done=True,
        horizon=4000,
    )
    env.seed(args.seed)
    rows = []
    try:
        for episode_index, record in records:
            row = _execute_episode(env, record, episode_index, args)
            rows.append(row)
            print(
                f"episode={episode_index:03d} safe={row['safe_success']} "
                f"rollback={row['rollback_recognized']} "
                f"insert={row['insertion_after_rollback']} "
                f"reclose={row['reclose_after_insertion']}"
            )
    finally:
        env.close()

    success_rate = float(np.mean([row["safe_success"] for row in rows]))
    verdict = (
        "PASS_L3B_EXECUTABLE_SAFE_WITNESS"
        if success_rate >= args.min_success_rate
        else "FAIL_L3B_EXECUTABLE_SAFE_WITNESS"
    )
    report = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "verdict": verdict,
        "source_condition": "Er",
        "task_prompt": TASK_PROMPT,
        "episode_indices": indices,
        "count": len(rows),
        "safe_success_count": int(sum(row["safe_success"] for row in rows)),
        "safe_success_rate": success_rate,
        "required_success_rate": args.min_success_rate,
        "er_state_bundle": str(state_path),
        "er_state_bundle_sha256": sha256_path(state_path),
        "group_contract": group_attrs,
        "motion_interface": "real 7-D OSC actions through env.step",
        "controller": "deterministic_geometry_relative_osc",
        "learned_reference_used": False,
        "direct_qpos_edits_after_restore": False,
        "collision_oracle_used": False,
        "episodes": rows,
    }
    output_json = Path(args.out_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_csv = Path(args.out_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_markdown = Path(args.out_report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(
        "\n".join(
            [
                "# L3-B executable safe witness",
                "",
                f"- Verdict: **{verdict}**",
                f"- Exact Er episodes: {len(rows)}",
                f"- Full ordered repair: {sum(row['safe_success'] for row in rows)}/{len(rows)}",
                f"- Success rate: {success_rate:.3f} (required {args.min_success_rate:.3f})",
                "- Required trace: `Close True→False`, `In False→True`, `Close False→True`.",
                "- Route: action-only bowl staging, reopen, regrasp, insert, and reclose.",
                "- Execution: deterministic 7-D OSC actions through `env.step` only.",
                "- Learned reference: false.",
                "- Post-restore qpos/object edits: false.",
                "- Evidence: per-episode trajectory plus agent-view and wrist-view MP4s.",
                f"- Er SHA-256: `{report['er_state_bundle_sha256']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(verdict)
    if args.fail_on_invalid and verdict != "PASS_L3B_EXECUTABLE_SAFE_WITNESS":
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-path",
        default=str(Path(__file__).with_name("l3b_bowl_er_states.hdf5")),
    )
    parser.add_argument("--episode-indices", default="0,1,2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-stride", type=int, default=3)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="review/L3-B_bowl_order_task/safe_witness",
    )
    parser.add_argument(
        "--out-json",
        default="review/L3-B_bowl_order_task/L3-B_executable_safe_witness.json",
    )
    parser.add_argument(
        "--out-csv",
        default="review/L3-B_bowl_order_task/L3-B_executable_safe_witness.csv",
    )
    parser.add_argument(
        "--out-report",
        default="review/L3-B_bowl_order_task/L3-B_executable_safe_witness.md",
    )
    parser.add_argument("--fail-on-invalid", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

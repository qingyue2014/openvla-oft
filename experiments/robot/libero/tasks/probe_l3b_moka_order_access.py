"""Real-action access probe for the native two-moka-pot LIBERO task.

The probe starts from two partial-goal states that share one preregistered final
arrangement:

* ``far_first``: the far pot is already on the stove; place the near pot.
* ``near_first``: the near pot is already on the stove; place the far pot.

Only native moka-pot poses and velocities are changed while constructing the
partial states.  The remaining placement is performed through LIBERO's 7-D
OSC action interface.  This is a falsification probe: safe success in both
orders rejects the simple "near-first blocks far" hypothesis.  A one-sided
failure is only a candidate effect, not proof, because alternate safe routes
must still be tested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder
from experiments.robot.libero.tasks.l3b_moka_runtime_gate import (
    MokaOrderRuntimeGate,
)
from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)
from experiments.robot.libero.tasks.l3b_order_candidates import (
    CANDIDATES,
    DUMMY_ACTION,
    FORMAL_WAIT_STEPS,
    SUITE,
    _refresh_observation,
    _trusted_native_states,
    body_measurement,
    native_bddl_path,
    static_preflight,
)
from experiments.robot.libero.tasks.probe_l3b_moka_pair_fit import (
    COOK_SITE,
    MAX_FINAL_ANGULAR_SPEED_RADPS,
    MAX_FINAL_LINEAR_SPEED_MPS,
    MAX_TILT_DEG,
    POT_BODIES,
    _free_joint_addresses,
    _set_free_body_pose,
    yaw_quaternion_wxyz,
)


GRASP_OFFSETS_XY = (
    (0.0, 0.0),
    (0.01, 0.0),
    (-0.01, 0.0),
    (0.0, 0.01),
    (0.0, -0.01),
)
TERMINAL_STABILITY_STEPS = 30
TERMINAL_MAX_DRIFT_M = 0.003


def _terminal_stability(rollout, body_name: str) -> dict:
    positions = np.asarray(
        rollout.recorder.body_pos[body_name], dtype=float
    )
    quaternions = np.asarray(
        rollout.recorder.body_quat[body_name], dtype=float
    )
    phases = np.asarray(rollout.recorder.phases).astype(str)
    indices = np.flatnonzero(phases == "settle")
    if len(indices) < TERMINAL_STABILITY_STEPS:
        return {
            "sample_count": int(len(indices)),
            "max_translation_drift_m": None,
            "max_tilt_deg": None,
            "max_linear_speed_mps": None,
            "max_angular_speed_radps": None,
            "stove_support_all_samples": False,
            "passed": False,
        }
    terminal_positions = positions[indices]
    q = quaternions[indices]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rzz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    tilt = np.degrees(np.arccos(np.clip(rzz, -1.0, 1.0)))
    drift = float(
        np.max(
            np.linalg.norm(
                terminal_positions - terminal_positions[0],
                axis=1,
            )
        )
    )
    maximum_tilt = float(np.max(tilt))
    measurements = [
        rollout.body_measurements[body_name][index] for index in indices
    ]
    maximum_linear_speed = float(
        max(item["linear_speed_mps"] for item in measurements)
    )
    maximum_angular_speed = float(
        max(item["angular_speed_radps"] for item in measurements)
    )
    stove_support_all_samples = all(
        any(
            str(name).startswith("flat_stove_1_")
            for name in item["contacts"]
        )
        for item in measurements
    )
    return {
        "sample_count": int(len(indices)),
        "max_translation_drift_m": drift,
        "max_tilt_deg": maximum_tilt,
        "max_linear_speed_mps": maximum_linear_speed,
        "max_angular_speed_radps": maximum_angular_speed,
        "stove_support_all_samples": stove_support_all_samples,
        "passed": (
            drift <= TERMINAL_MAX_DRIFT_M
            and maximum_tilt <= MAX_TILT_DEG
            and maximum_linear_speed <= MAX_FINAL_LINEAR_SPEED_MPS
            and maximum_angular_speed <= MAX_FINAL_ANGULAR_SPEED_RADPS
            and stove_support_all_samples
        ),
    }


def _policy_rgb(observation) -> np.ndarray:
    image = np.asarray(observation["agentview_image"])
    return np.ascontiguousarray(image[::-1, ::-1]).copy()


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2

    if not frames:
        raise ValueError("cannot write an empty review video")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"review video was not written: {path}")


def _runtime_task():
    from libero.libero import benchmark, get_libero_path

    candidate = CANDIDATES["moka"]
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(candidate.task_id)
    if task.bddl_file != candidate.task_file or task.language != candidate.prompt:
        raise ValueError("runtime native moka task does not match the lock")
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if bddl_path.resolve() != native_bddl_path(candidate).resolve():
        raise ValueError("runtime BDDL is not the locked native BDDL")
    return candidate, suite, task, bddl_path


def _load_zero_yaw_final_layout(path: Path) -> dict[str, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_STABLE_NATIVE_MOKA_PAIR_FOUND":
        raise ValueError("moka pair-fit report did not pass")
    for record in report.get("accepted", []):
        if all(
            abs(float(record["yaw_deg"][body_name])) < 1e-9
            for body_name in POT_BODIES
        ):
            return {
                body_name: np.asarray(
                    record["stability"][body_name]["first_policy"]["position"],
                    dtype=float,
                )
                for body_name in POT_BODIES
            }
    raise ValueError("pair-fit report has no accepted zero-yaw layout")


def _state_with_first_placement(
    env,
    native_state: np.ndarray,
    *,
    placed_body: str,
    placed_position: np.ndarray,
    settle_steps: int,
) -> tuple[np.ndarray, dict]:
    env.reset()
    env.set_init_state(native_state)
    # Normalize both pot yaws so that the two partial states lead to the exact
    # same preregistered final arrangement under a zero-rotation OSC transport.
    for body_name in POT_BODIES:
        current = body_measurement(env, body_name)
        position = np.asarray(current["position"], dtype=float)
        _set_free_body_pose(
            env,
            body_name,
            position,
            yaw_quaternion_wxyz(0.0),
        )
    spawn = np.asarray(placed_position, dtype=float).copy()
    spawn[2] += 0.03
    _set_free_body_pose(
        env,
        placed_body,
        spawn,
        yaw_quaternion_wxyz(0.0),
    )
    env.sim.forward()
    for _ in range(settle_steps):
        env.step(DUMMY_ACTION)
    state = np.asarray(env.sim.get_state().flatten(), dtype=float)
    placed = body_measurement(env, placed_body)
    other_body = POT_BODIES[1] if placed_body == POT_BODIES[0] else POT_BODIES[0]
    other = body_measurement(env, other_body)
    if placed["tilt_deg"] > MAX_TILT_DEG:
        raise ValueError(f"{placed_body} tipped while constructing partial state")
    if not any(
        str(name).startswith("flat_stove_1_") for name in placed["contacts"]
    ):
        raise ValueError(f"{placed_body} lacks stove support in partial state")
    if any(str(name).startswith("moka_pot_") for name in placed["contacts"]):
        raise ValueError(f"{placed_body} contacts the remaining pot")
    return state, {"placed": placed, "remaining": other}


class AccessRollout:
    def __init__(
        self,
        env,
        observation,
        *,
        placed_body: str,
        moving_body: str,
        args,
    ):
        self.env = env
        self.observation = observation
        self.placed_body = placed_body
        self.moving_body = moving_body
        self.args = args
        self.frames = [_policy_rgb(observation)]
        self.steps = 0
        self.forbidden_contacts: list[dict] = []
        self.body_measurements = {
            placed_body: [],
            moving_body: [],
        }
        self.recorder = TrajectoryRecorder(
            env,
            [*POT_BODIES, "flat_stove_1_main"],
        )

    def _observe_forbidden_contacts(self, phase: str) -> None:
        placed = body_measurement(self.env, self.placed_body)
        moving = body_measurement(self.env, self.moving_body)
        self.body_measurements[self.placed_body].append(placed)
        self.body_measurements[self.moving_body].append(moving)
        contacts = []
        for name in placed["contacts"]:
            text = str(name)
            if text.startswith(("robot0_", "gripper0_", self.moving_body[:-4])):
                contacts.append(f"{self.placed_body}::{text}")
        for name in moving["contacts"]:
            text = str(name)
            if text.startswith(self.placed_body[:-4]):
                contacts.append(f"{self.moving_body}::{text}")
        if contacts:
            self.forbidden_contacts.append(
                {
                    "step": self.steps,
                    "phase": phase,
                    "contacts": sorted(set(contacts)),
                }
            )

    def advance(self, action: np.ndarray, phase: str) -> None:
        self.observation, _, _, _ = self.env.step(action.tolist())
        self.recorder.record(
            self.observation,
            action,
            self.steps,
            phase=phase,
        )
        self._observe_forbidden_contacts(phase)
        if self.steps % self.args.video_stride == 0:
            self.frames.append(_policy_rgb(self.observation))
        self.steps += 1

    def hold(self, gripper: float, count: int, phase: str) -> None:
        for _ in range(count):
            action = np.zeros(7, dtype=float)
            action[-1] = gripper
            self.advance(action, phase)

    def rotate_yaw(
        self,
        gripper: float,
        command: float,
        count: int,
    ) -> None:
        for _ in range(count):
            action = np.zeros(7, dtype=float)
            action[5] = float(command)
            action[-1] = gripper
            self.advance(action, "orient_gripper")

    def orient_to(
        self,
        target_quaternion_xyzw,
        gripper: float,
        *,
        tolerance_rad: float,
        max_steps: int,
        command_limit: float,
    ) -> float:
        from robosuite.utils import transform_utils as transform

        target = transform.quat2mat(
            np.asarray(target_quaternion_xyzw, dtype=float)
        )
        best = float("inf")
        for _ in range(max_steps):
            current = transform.quat2mat(
                np.asarray(
                    self.observation["robot0_eef_quat"], dtype=float
                )
            )
            error_matrix = target @ current.T
            error_quaternion = transform.mat2quat(error_matrix)
            axis_angle = transform.quat2axisangle(error_quaternion)
            error = float(np.linalg.norm(axis_angle))
            best = min(best, error)
            if error <= tolerance_rad:
                return best
            action = np.zeros(7, dtype=float)
            # OSC_POSE maps a unit rotation command to 0.5 rad.
            action[3:6] = np.clip(
                axis_angle / 0.5,
                -command_limit,
                command_limit,
            )
            action[-1] = gripper
            self.advance(action, "orient_gripper")
        raise RuntimeError(
            "OSC gripper orientation timeout "
            f"best_error_rad={best:.5f}"
        )

    def seat_grasp(self, target: np.ndarray) -> None:
        for _ in range(self.args.grasp_seat_steps):
            current = np.asarray(
                self.observation["robot0_eef_pos"], dtype=float
            )
            action = np.zeros(7, dtype=float)
            action[:3] = np.clip(
                (np.asarray(target) - current) / self.args.action_scale,
                -self.args.grasp_seat_max_command,
                self.args.grasp_seat_max_command,
            )
            action[-1] = 1.0
            self.advance(action, "grasp_seat")

    def seat_grasp_relative(
        self,
        body_name: str,
        offset: np.ndarray,
    ) -> None:
        for _ in range(self.args.grasp_seat_steps):
            body_position = np.asarray(
                body_measurement(self.env, body_name)["position"],
                dtype=float,
            )
            target = body_position + np.asarray(offset, dtype=float)
            current = np.asarray(
                self.observation["robot0_eef_pos"], dtype=float
            )
            action = np.zeros(7, dtype=float)
            action[:3] = np.clip(
                (target - current) / self.args.action_scale,
                -self.args.grasp_seat_max_command,
                self.args.grasp_seat_max_command,
            )
            action[-1] = 1.0
            self.advance(action, "grasp_seat")

    def move(
        self,
        target: np.ndarray,
        gripper: float,
        phase: str,
        *,
        max_steps: int | None = None,
        tolerance: float | None = None,
        command_limit: float = 1.0,
    ) -> tuple[bool, float]:
        max_steps = self.args.max_waypoint_steps if max_steps is None else max_steps
        tolerance = (
            self.args.position_tolerance if tolerance is None else tolerance
        )
        best = float("inf")
        for _ in range(max_steps):
            current = np.asarray(
                self.observation["robot0_eef_pos"], dtype=float
            )
            error = float(np.linalg.norm(np.asarray(target) - current))
            best = min(best, error)
            if error <= tolerance:
                return True, best
            action = np.zeros(7, dtype=float)
            action[:3] = np.clip(
                (np.asarray(target) - current) / self.args.action_scale,
                -command_limit,
                command_limit,
            )
            action[-1] = gripper
            self.advance(action, phase)
        return False, best

    def move_pose(
        self,
        target_position: np.ndarray,
        target_quaternion_xyzw: np.ndarray,
        gripper: float,
        phase: str,
        *,
        max_steps: int,
        position_tolerance: float,
        orientation_tolerance_rad: float,
        position_command_limit: float,
        orientation_command_limit: float,
    ) -> tuple[bool, dict]:
        """Track one absolute OSC pose waypoint with a real 7-D action."""
        from robosuite.utils import transform_utils as transform

        target_position = np.asarray(target_position, dtype=float)
        target_rotation = transform.quat2mat(
            np.asarray(target_quaternion_xyzw, dtype=float)
        )
        best_position = float("inf")
        best_orientation = float("inf")
        final_position = float("inf")
        final_orientation = float("inf")
        for _ in range(max_steps):
            current_position = np.asarray(
                self.observation["robot0_eef_pos"], dtype=float
            )
            current_rotation = transform.quat2mat(
                np.asarray(
                    self.observation["robot0_eef_quat"], dtype=float
                )
            )
            position_delta = target_position - current_position
            orientation_delta = transform.quat2axisangle(
                transform.mat2quat(target_rotation @ current_rotation.T)
            )
            final_position = float(np.linalg.norm(position_delta))
            final_orientation = float(np.linalg.norm(orientation_delta))
            best_position = min(best_position, final_position)
            best_orientation = min(best_orientation, final_orientation)
            if (
                final_position <= position_tolerance
                and final_orientation <= orientation_tolerance_rad
            ):
                return True, {
                    "best_position_error_m": best_position,
                    "best_orientation_error_rad": best_orientation,
                    "final_position_error_m": final_position,
                    "final_orientation_error_rad": final_orientation,
                }
            action = np.zeros(7, dtype=float)
            action[:3] = np.clip(
                position_delta / self.args.action_scale,
                -position_command_limit,
                position_command_limit,
            )
            # OSC_POSE maps a unit rotation command to 0.5 rad.
            action[3:6] = np.clip(
                orientation_delta / 0.5,
                -orientation_command_limit,
                orientation_command_limit,
            )
            action[-1] = gripper
            self.advance(action, phase)
        return False, {
            "best_position_error_m": best_position,
            "best_orientation_error_rad": best_orientation,
            "final_position_error_m": final_position,
            "final_orientation_error_rad": final_orientation,
        }


def _complete_remaining_placement(
    env,
    partial_state: np.ndarray,
    *,
    order: str,
    placed_body: str,
    moving_body: str,
    target_position: np.ndarray,
    grasp_offset_xy: np.ndarray,
    args,
    state_record: dict | None = None,
) -> tuple[dict, list[np.ndarray], TrajectoryRecorder]:
    observation = env.reset()
    restored_state = (
        materialize_native_scene_state(env, state_record)
        if state_record is not None
        else partial_state
    )
    observation = env.set_init_state(restored_state)
    runtime_gate = (
        MokaOrderRuntimeGate(env, state_record)
        if state_record is not None
        else None
    )
    for _ in range(FORMAL_WAIT_STEPS):
        observation, _, _, _ = env.step(DUMMY_ACTION)
        if runtime_gate is not None:
            runtime_gate.observe()
    runtime_metrics = (
        runtime_gate.finalize() if runtime_gate is not None else None
    )
    rollout = AccessRollout(
        env,
        observation,
        placed_body=placed_body,
        moving_body=moving_body,
        args=args,
    )
    failure_reason = ""
    failure_stage = ""

    moving_start = np.asarray(
        body_measurement(env, moving_body)["position"], dtype=float
    )
    placed_start = np.asarray(
        body_measurement(env, placed_body)["position"], dtype=float
    )
    close_start_offset_xy = np.asarray(
        getattr(args, "grasp_close_start_offset_xy", grasp_offset_xy),
        dtype=float,
    )
    reference_waypoints = getattr(args, "grasp_pose_waypoints", None)
    reference_results = []
    if reference_waypoints:
        first = reference_waypoints[0]
        first_target = moving_start + np.asarray(
            first["offset_xyz"], dtype=float
        )
        reached, best = rollout.move(
            first_target,
            -1.0,
            "reference_approach",
            max_steps=getattr(args, "reference_approach_max_steps", 240),
            tolerance=getattr(
                args, "reference_approach_position_tolerance", 0.0015
            ),
            command_limit=getattr(
                args, "reference_approach_command_limit", 0.8
            ),
        )
        if not reached:
            failure_reason = f"waypoint_timeout_best_{best:.4f}"
            failure_stage = "reference_approach"
        if not failure_reason:
            rollout.orient_to(
                first["quaternion_xyzw"],
                -1.0,
                tolerance_rad=getattr(
                    args, "reference_initial_orientation_tolerance_rad", 0.003
                ),
                max_steps=getattr(
                    args, "reference_initial_orientation_max_steps", 300
                ),
                command_limit=getattr(
                    args, "reference_initial_orientation_command_limit", 0.15
                ),
            )
            reached, best = rollout.move(
                first_target,
                -1.0,
                "reference_reseat",
                max_steps=getattr(args, "reference_reseat_max_steps", 160),
                tolerance=getattr(
                    args, "reference_reseat_position_tolerance", 0.001
                ),
                command_limit=getattr(
                    args, "reference_reseat_command_limit", 0.4
                ),
            )
            if not reached:
                failure_reason = f"waypoint_timeout_best_{best:.4f}"
                failure_stage = "reference_reseat"
        for waypoint_index, waypoint in enumerate(reference_waypoints[1:], 1):
            if failure_reason:
                break
            gripper = float(waypoint["gripper"])
            phase = (
                "reference_sweep_open"
                if gripper < 0.0
                else "reference_close_lift"
            )
            reached, metrics = rollout.move_pose(
                moving_start
                + np.asarray(waypoint["offset_xyz"], dtype=float),
                np.asarray(waypoint["quaternion_xyzw"], dtype=float),
                gripper,
                phase,
                max_steps=getattr(args, "reference_waypoint_max_steps", 100),
                position_tolerance=getattr(
                    args, "reference_position_tolerance", 0.003
                ),
                orientation_tolerance_rad=getattr(
                    args, "reference_orientation_tolerance_rad", 0.012
                ),
                position_command_limit=getattr(
                    args, "reference_position_command_limit", 0.35
                ),
                orientation_command_limit=getattr(
                    args, "reference_orientation_command_limit", 0.20
                ),
            )
            reference_results.append(
                {
                    "waypoint_index": waypoint_index,
                    "source_step": waypoint.get("source_step"),
                    "phase": phase,
                    "reached": reached,
                    **metrics,
                }
            )
            if not reached:
                failure_reason = (
                    "reference_waypoint_timeout_"
                    f"position_{metrics['best_position_error_m']:.4f}_"
                    f"orientation_{metrics['best_orientation_error_rad']:.4f}"
                )
                failure_stage = phase
    else:
        above = moving_start + np.asarray(
            [
                close_start_offset_xy[0],
                close_start_offset_xy[1],
                args.approach_height,
            ],
            dtype=float,
        )
        grasp = moving_start + np.asarray(
            [
                close_start_offset_xy[0],
                close_start_offset_xy[1],
                args.grasp_height,
            ],
            dtype=float,
        )
        seated_grasp = moving_start + np.asarray(
            [grasp_offset_xy[0], grasp_offset_xy[1], args.grasp_height],
            dtype=float,
        )
        for stage, target in (("approach", above), ("descend", grasp)):
            reached, best = rollout.move(target, -1.0, stage)
            contact_seated = bool(
                stage == "descend" and best <= args.grasp_contact_tolerance
            )
            if not reached and not contact_seated:
                failure_reason = f"waypoint_timeout_best_{best:.4f}"
                failure_stage = stage
                break
            if (
                stage == "approach"
                and not failure_reason
                and getattr(args, "grasp_yaw_steps", 0) > 0
            ):
                rollout.rotate_yaw(
                    -1.0,
                    getattr(args, "grasp_yaw_command", 0.0),
                    args.grasp_yaw_steps,
                )
            target_quaternion = getattr(
                args, "grasp_target_quaternion", None
            )
            if (
                stage == "approach"
                and not failure_reason
                and target_quaternion is not None
            ):
                rollout.orient_to(
                    target_quaternion,
                    -1.0,
                    tolerance_rad=getattr(
                        args, "orientation_tolerance_rad", 0.04
                    ),
                    max_steps=getattr(args, "orientation_max_steps", 80),
                    command_limit=getattr(
                        args, "orientation_command_limit", 0.35
                    ),
                )

        if not failure_reason:
            if getattr(args, "grasp_seat_follow_body", False):
                rollout.seat_grasp_relative(
                    moving_body,
                    np.asarray(
                        [
                            grasp_offset_xy[0],
                            grasp_offset_xy[1],
                            args.grasp_height,
                        ],
                        dtype=float,
                    ),
                )
            else:
                rollout.seat_grasp(seated_grasp)
            rollout.hold(1.0, args.grasp_steps, "grasp")

    if not failure_reason:
        grasped_offset = np.asarray(
            rollout.observation["robot0_eef_pos"], dtype=float
        ) - np.asarray(
            body_measurement(env, moving_body)["position"], dtype=float
        )
        if reference_waypoints:
            reached = True
            best = 0.0
        else:
            lift_target = np.asarray(
                rollout.observation["robot0_eef_pos"], dtype=float
            ).copy()
            lift_target[2] += args.lift_height
            reached, best = rollout.move(
                lift_target,
                1.0,
                "lift",
                command_limit=getattr(args, "lift_max_command", 1.0),
            )
        lifted = float(
            body_measurement(env, moving_body)["position"][2] - moving_start[2]
        )
        maximum_lift = max(
            (
                float(position[2]) - moving_start[2]
                for position in rollout.recorder.body_pos[moving_body]
            ),
            default=lifted,
        )
        if not reached:
            failure_reason = f"waypoint_timeout_best_{best:.4f}"
            failure_stage = "lift"
        elif lifted < args.minimum_lift:
            failure_reason = f"grasp_failed_lift_{lifted:.4f}"
            failure_stage = "lift"
    else:
        grasped_offset = np.zeros(3, dtype=float)
        lifted = 0.0
        maximum_lift = 0.0

    if not failure_reason:
        desired = np.asarray(target_position, dtype=float)
        transit_body = np.asarray(
            body_measurement(env, moving_body)["position"], dtype=float
        )
        transit_body[2] = max(transit_body[2], desired[2]) + args.transport_height
        targets = (
            ("raise", transit_body + grasped_offset),
            (
                "translate",
                np.asarray([desired[0], desired[1], transit_body[2]])
                + grasped_offset,
            ),
            (
                "preplace",
                desired
                + np.asarray([0.0, 0.0, args.release_clearance])
                + grasped_offset,
            ),
        )
        for stage, target in targets:
            reached, best = rollout.move(
                target,
                1.0,
                stage,
                max_steps=args.transport_max_waypoint_steps,
            )
            if not reached:
                failure_reason = f"waypoint_timeout_best_{best:.4f}"
                failure_stage = stage
                break

    if not failure_reason:
        rollout.hold(-1.0, args.release_steps, "release")
        withdraw_distance = float(
            getattr(args, "withdraw_distance", 0.0)
        )
        if withdraw_distance > 0.0:
            direction = np.asarray(grasped_offset[:2], dtype=float)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-9:
                failure_reason = "withdraw_direction_undefined"
                failure_stage = "withdraw"
            else:
                withdraw = np.asarray(
                    rollout.observation["robot0_eef_pos"], dtype=float
                ).copy()
                withdraw[:2] += (
                    direction / direction_norm * withdraw_distance
                )
                withdraw[2] += float(
                    getattr(args, "withdraw_height", 0.0)
                )
                reached, best = rollout.move(
                    withdraw,
                    -1.0,
                    "withdraw",
                    max_steps=getattr(
                        args,
                        "withdraw_max_steps",
                        args.transport_max_waypoint_steps,
                    ),
                    command_limit=getattr(
                        args, "withdraw_command_limit", 0.6
                    ),
                )
                if not reached:
                    failure_reason = f"waypoint_timeout_best_{best:.4f}"
                    failure_stage = "withdraw"
        retreat = np.asarray(
            rollout.observation["robot0_eef_pos"], dtype=float
        ).copy()
        retreat[2] += args.retreat_height
        reached, best = rollout.move(retreat, -1.0, "retreat")
        if not reached:
            failure_reason = f"waypoint_timeout_best_{best:.4f}"
            failure_stage = "retreat"
        rollout.hold(-1.0, args.final_settle_steps, "settle")

    final_placed = body_measurement(env, placed_body)
    final_moving = body_measurement(env, moving_body)
    placed_displacement = float(
        np.linalg.norm(
            np.asarray(final_placed["position"], dtype=float) - placed_start
        )
    )
    target_xy_error = float(
        np.linalg.norm(
            np.asarray(final_moving["position"][:2], dtype=float)
            - np.asarray(target_position[:2], dtype=float)
        )
    )
    terminal = {
        body_name: _terminal_stability(rollout, body_name)
        for body_name in (placed_body, moving_body)
    }
    task_success = bool(env.check_success())
    stable = all(
        measurement["tilt_deg"] <= MAX_TILT_DEG
        and measurement["linear_speed_mps"] is not None
        and measurement["linear_speed_mps"] <= MAX_FINAL_LINEAR_SPEED_MPS
        and measurement["angular_speed_radps"] is not None
        and measurement["angular_speed_radps"]
        <= MAX_FINAL_ANGULAR_SPEED_RADPS
        and any(
            str(name).startswith("flat_stove_1_")
            for name in measurement["contacts"]
        )
        for measurement in (final_placed, final_moving)
    ) and all(record["passed"] for record in terminal.values())
    final_robot_contact = any(
        str(name).startswith(("robot0_", "gripper0_"))
        for name in final_moving["contacts"]
    )
    maximum_preplaced_displacement = getattr(
        args, "maximum_preplaced_displacement", None
    )
    maximum_target_xy_error = getattr(
        args, "maximum_target_xy_error", None
    )
    preserved_preplaced = bool(
        maximum_preplaced_displacement is None
        or placed_displacement <= maximum_preplaced_displacement
    )
    target_reached = bool(
        maximum_target_xy_error is None
        or target_xy_error <= maximum_target_xy_error
    )
    safe_success = bool(
        not failure_reason
        and not rollout.forbidden_contacts
        and task_success
        and stable
        and not final_robot_contact
        and preserved_preplaced
        and target_reached
    )
    if not failure_reason and rollout.forbidden_contacts:
        failure_reason = "forbidden_order_contact"
        failure_stage = rollout.forbidden_contacts[0]["phase"]
    elif not failure_reason and not task_success:
        failure_reason = "native_goal_false"
        failure_stage = "final"
    elif not failure_reason and not stable:
        failure_reason = "unstable_final_state"
        failure_stage = "final"
    elif not failure_reason and final_robot_contact:
        failure_reason = "final_robot_object_contact"
        failure_stage = "final"
    elif not failure_reason and not preserved_preplaced:
        failure_reason = "preplaced_object_displaced"
        failure_stage = "final"
    elif not failure_reason and not target_reached:
        failure_reason = "target_slot_missed"
        failure_stage = "final"
    result = {
        "order": order,
        "placed_body": placed_body,
        "moving_body": moving_body,
        "grasp_offset_xy": grasp_offset_xy.tolist(),
        "grasp_close_start_offset_xy": close_start_offset_xy.tolist(),
        "grasp_yaw_steps": int(getattr(args, "grasp_yaw_steps", 0)),
        "grasp_yaw_command": float(
            getattr(args, "grasp_yaw_command", 0.0)
        ),
        "grasp_target_quaternion_xyzw": (
            None
            if getattr(args, "grasp_target_quaternion", None) is None
            else np.asarray(
                args.grasp_target_quaternion, dtype=float
            ).tolist()
        ),
        "grasp_reference_label": getattr(
            args, "grasp_reference_label", None
        ),
        "grasp_reference_waypoint_results": reference_results,
        "grasp_lift_m": lifted,
        "maximum_grasp_lift_m": maximum_lift,
        "preplaced_body_displacement_m": placed_displacement,
        "maximum_preplaced_body_displacement_m": (
            maximum_preplaced_displacement
        ),
        "target_xy_error_m": target_xy_error,
        "maximum_target_xy_error_m": maximum_target_xy_error,
        "final_robot_object_contact": final_robot_contact,
        "steps": rollout.steps,
        "final_eef_position": np.asarray(
            rollout.observation["robot0_eef_pos"], dtype=float
        ).tolist(),
        "final_eef_quaternion": np.asarray(
            rollout.observation["robot0_eef_quat"], dtype=float
        ).tolist(),
        "final_gripper_qpos": np.asarray(
            rollout.observation["robot0_gripper_qpos"], dtype=float
        ).tolist(),
        "final_finger_positions": {
            name: np.asarray(
                env.sim.data.body_xpos[env.sim.model.body_name2id(name)],
                dtype=float,
            ).tolist()
            for name in ("gripper0_leftfinger", "gripper0_rightfinger")
        },
        "task_success": task_success,
        "stable_final": stable,
        "safe_success": safe_success,
        "failure_reason": failure_reason,
        "failure_stage": failure_stage,
        "forbidden_contacts": rollout.forbidden_contacts,
        "runtime_initial_gate": runtime_metrics,
        "terminal_stability": terminal,
        "final": {
            placed_body: final_placed,
            moving_body: final_moving,
        },
    }
    return result, rollout.frames, rollout.recorder


def run_probe(args) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    candidate, suite, task, bddl_path = _runtime_task()
    native_states = _trusted_native_states(suite, task)
    final_layout = _load_zero_yaw_final_layout(Path(args.pair_fit_report))
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(0)
    results = {}
    try:
        order_specs = {
            "far_first": {
                "placed_body": POT_BODIES[0],
                "moving_body": POT_BODIES[1],
            },
            "near_first": {
                "placed_body": POT_BODIES[1],
                "moving_body": POT_BODIES[0],
            },
        }
        for order, spec in order_specs.items():
            partial_state, partial_measurements = _state_with_first_placement(
                env,
                native_states[0],
                placed_body=spec["placed_body"],
                placed_position=final_layout[spec["placed_body"]],
                settle_steps=args.construction_settle_steps,
            )
            attempts = []
            for attempt_index, offset in enumerate(GRASP_OFFSETS_XY):
                result, frames, _ = _complete_remaining_placement(
                    env,
                    partial_state,
                    order=order,
                    placed_body=spec["placed_body"],
                    moving_body=spec["moving_body"],
                    target_position=final_layout[spec["moving_body"]],
                    grasp_offset_xy=np.asarray(offset, dtype=float),
                    args=args,
                )
                result["attempt_index"] = attempt_index
                attempts.append(result)
                if result["safe_success"]:
                    video_path = (
                        Path(args.output_dir)
                        / "success"
                        / f"{order}_safe_success.mp4"
                    )
                    _write_video(video_path, frames, args.video_fps)
                    result["review_video"] = str(video_path.resolve())
                    break
            results[order] = {
                "partial_state": partial_measurements,
                "attempts": attempts,
                "safe_success": any(item["safe_success"] for item in attempts),
            }
    finally:
        env.close()

    far_success = results["far_first"]["safe_success"]
    near_success = results["near_first"]["safe_success"]
    if far_success and near_success:
        verdict = "REJECT_SIMPLE_ORDER_CONSTRAINT_BOTH_ORDERS_SAFE"
    elif far_success and not near_success:
        verdict = "POTENTIAL_ORDER_EFFECT_NEEDS_ALTERNATE_ROUTE_TESTS"
    elif not far_success and near_success:
        verdict = "REJECT_HYPOTHESIZED_DIRECTION_NEAR_FIRST_NOT_WORSE"
    else:
        verdict = "INCONCLUSIVE_OSC_CONTROLLER_FAILED_BOTH_ORDERS"
    return {
        **static_preflight(candidate),
        "diagnostic_only": True,
        "formal_authorized": False,
        "pair_fit_report": str(Path(args.pair_fit_report).resolve()),
        "same_final_layout": {
            body_name: position.tolist()
            for body_name, position in final_layout.items()
        },
        "intervention": {
            "changed_bodies": list(POT_BODIES),
            "changed_fields": ["free_joint_qpos", "free_joint_qvel"],
            "asset_inventory_changed": False,
            "prompt_changed": False,
            "bddl_changed": False,
        },
        "results": results,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe real-action order access for the native moka task"
    )
    parser.add_argument(
        "--pair-fit-report",
        default=(
            "review/L3-B_order_candidates_task/moka_pair_fit/"
            "moka_pair_fit.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="review/L3-B_order_candidates_task/moka_order_access",
    )
    parser.add_argument("--construction-settle-steps", type=int, default=80)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.006)
    parser.add_argument("--max-waypoint-steps", type=int, default=160)
    parser.add_argument("--transport-max-waypoint-steps", type=int, default=220)
    parser.add_argument("--approach-height", type=float, default=0.18)
    parser.add_argument("--grasp-height", type=float, default=0.035)
    parser.add_argument("--grasp-contact-tolerance", type=float, default=0.045)
    parser.add_argument("--grasp-seat-steps", type=int, default=20)
    parser.add_argument("--grasp-seat-max-command", type=float, default=0.2)
    parser.add_argument("--grasp-steps", type=int, default=15)
    parser.add_argument("--grasp-yaw-steps", type=int, default=0)
    parser.add_argument("--grasp-yaw-command", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.13)
    parser.add_argument("--lift-max-command", type=float, default=1.0)
    parser.add_argument("--minimum-lift", type=float, default=0.04)
    parser.add_argument("--transport-height", type=float, default=0.10)
    parser.add_argument("--release-clearance", type=float, default=0.008)
    parser.add_argument("--release-steps", type=int, default=15)
    parser.add_argument("--retreat-height", type=float, default=0.12)
    parser.add_argument("--final-settle-steps", type=int, default=30)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_probe(args)
    report_path = output_dir / "moka_order_access.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{report['verdict']} report={report_path}")


if __name__ == "__main__":
    main()

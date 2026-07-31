"""Generate the L3-A3 safe reference through LIBERO's 7-D OSC interface.

The rollout starts from a serialized Er state.  After the evaluator-parity
reset/wait, every manipulation action is issued with ``env.step``: grasp and
park the native wine bottle on the native table, then push the native plate
into the task's native stove-front region.  The script fails closed unless the
shared causal oracle, native task predicate, and final bottle stability gates
all pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.l3a_cascade_oracle import TaskActorCascadeOracle
from experiments.robot.libero.physcog_trajectory import TrajectoryRecorder
from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    BOTTLE_BODY,
    FORMAL_WAIT_STEPS,
    GOAL_SITE,
    PLATE_BODY,
    SAFE_PREFIX_MIN_DISPLACEMENT_M,
    SCENE_ID,
    TABLE_BODY,
    TASK_KEY,
    TASK_PROMPT,
    body_pose,
    body_tilt_deg,
    body_velocity,
    bodies_contact,
)
from experiments.robot.libero.tasks.validate_l3a3_state_bundle import (
    artifact_binding,
)


def _position_action(current, target, gripper, scale):
    action = np.zeros(7, dtype=float)
    action[:3] = np.clip(
        (np.asarray(target) - np.asarray(current)) / scale, -1.0, 1.0
    )
    action[-1] = float(gripper)
    return action


def _select_reachable_trailing_contact(
    plate_xy, push_direction_xy, eef_xy, backoff
):
    """Choose a reachable trailing EEF line that the push moves inward from.

    A point exactly opposite the diagonal goal direction is geometrically
    natural but unnecessarily couples both workspace axes.  In this native
    task that point moves the Franka toward its difficult +X reach limit.
    Cardinal candidates preserve a positive inward component of the requested
    push.  ``backoff`` locates the EEF centre inside the native plate footprint;
    the fingers, not the EEF origin, establish the physical contact.
    """
    plate_xy = np.asarray(plate_xy, dtype=float)
    direction = np.asarray(push_direction_xy, dtype=float)
    eef_xy = np.asarray(eef_xy, dtype=float)
    norm = float(np.linalg.norm(direction))
    if plate_xy.shape != (2,) or eef_xy.shape != (2,) or direction.shape != (2,):
        raise ValueError("plate, EEF, and push coordinates must be 2-D")
    if norm <= 1e-9 or not np.isfinite(norm):
        raise ValueError("push direction must be finite and nonzero")
    if backoff <= 0:
        raise ValueError("plate contact backoff must be positive")
    direction = direction / norm
    offsets = (
        np.array([backoff, 0.0]),
        np.array([-backoff, 0.0]),
        np.array([0.0, backoff]),
        np.array([0.0, -backoff]),
    )
    # Moving along ``direction`` must point from the pusher into the plate:
    # dot(direction, plate - contact) > 0.
    trailing = [
        offset for offset in offsets if float(np.dot(direction, -offset)) > 1e-6
    ]
    if not trailing:
        raise ValueError("no cardinal trailing plate contact candidate")
    offset = min(
        trailing,
        key=lambda value: float(
            np.linalg.norm((plate_xy + value) - eef_xy)
        ),
    )
    return plate_xy + offset


def _live_plate_tracking_target(
    live_plate, goal, confirmed_contact_offset, push_increment
):
    """Anchor one push step to the live plate and confirmed contact pose."""
    live_plate = np.asarray(live_plate, dtype=float)
    goal = np.asarray(goal, dtype=float)
    contact_offset = np.asarray(confirmed_contact_offset, dtype=float)
    if live_plate.shape != (3,) or goal.shape != (3,):
        raise ValueError("live plate and goal coordinates must be 3-D")
    if contact_offset.shape != (3,):
        raise ValueError("confirmed contact offset must be 3-D")
    if not np.all(np.isfinite(live_plate)):
        raise ValueError("live plate coordinates must be finite")
    if not np.all(np.isfinite(goal)) or not np.all(np.isfinite(contact_offset)):
        raise ValueError("goal and confirmed contact offset must be finite")
    if push_increment <= 0:
        raise ValueError("push increment must be positive")
    direction_xy = goal[:2] - live_plate[:2]
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1e-9 or not np.isfinite(norm):
        raise ValueError("live plate-to-goal direction must be finite and nonzero")
    direction_xy /= norm
    target = live_plate + contact_offset
    target[:2] += direction_xy * push_increment
    return target, direction_xy


def _gate_live_contact_offset_xy(
    explicit_contact_anchor,
    live_contact_offset,
    push_direction_xy,
    maximum_xy_drift,
):
    """Use a live XY offset only while it remains near the trailing anchor."""
    anchor = np.asarray(explicit_contact_anchor, dtype=float)
    live = np.asarray(live_contact_offset, dtype=float)
    direction = np.asarray(push_direction_xy, dtype=float)
    if anchor.shape != (3,) or live.shape != (3,):
        raise ValueError("anchor and live contact offsets must be 3-D")
    if direction.shape != (2,):
        raise ValueError("push direction must be 2-D")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(live)):
        raise ValueError("anchor and live contact offsets must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9 or not np.isfinite(direction_norm):
        raise ValueError("push direction must be finite and nonzero")
    if maximum_xy_drift <= 0 or not np.isfinite(maximum_xy_drift):
        raise ValueError("maximum live contact XY drift must be positive")
    direction /= direction_norm
    xy_drift = float(np.linalg.norm(live[:2] - anchor[:2]))
    anchor_trailing_projection = float(-np.dot(anchor[:2], direction))
    live_trailing_projection = float(-np.dot(live[:2], direction))
    trailing_side = live_trailing_projection > 0.0
    within_anchor_gate = xy_drift <= float(maximum_xy_drift)
    accepted = trailing_side and within_anchor_gate
    selected = anchor.copy()
    if accepted:
        selected[:2] = live[:2]
    if not trailing_side:
        reason = "live_offset_not_on_trailing_side"
    elif not within_anchor_gate:
        reason = "live_offset_exceeds_explicit_anchor_gate"
    else:
        reason = "bounded_live_offset_accepted"
    diagnostics = {
        "accepted": accepted,
        "reason": reason,
        "explicit_anchor_offset": anchor.tolist(),
        "live_contact_offset": live.tolist(),
        "selected_contact_offset": selected.tolist(),
        "push_direction_xy": direction.tolist(),
        "xy_drift_from_explicit_anchor_m": xy_drift,
        "maximum_xy_drift_m": float(maximum_xy_drift),
        "anchor_trailing_projection_m": anchor_trailing_projection,
        "live_trailing_projection_m": live_trailing_projection,
        "live_offset_on_trailing_side": trailing_side,
    }
    return selected, diagnostics


def _contact_progress_saturation_evidence(
    robot_contact_steps, incremental_progress, minimum_progress
):
    """Accept push timeout only when contact produced non-noise progress."""
    if minimum_progress <= 0:
        raise ValueError("minimum saturation progress must be positive")
    if robot_contact_steps < 1:
        return None
    if (
        not np.isfinite(incremental_progress)
        or incremental_progress <= minimum_progress
    ):
        return None
    return {
        "status": "contact_progress_saturated",
        "acceptance_reason": (
            "controller budget saturated under real robot-plate contact "
            "with positive plate progress above numerical noise"
        ),
        "robot_contact_steps": int(robot_contact_steps),
        "incremental_plate_progress_m": float(incremental_progress),
        "minimum_progress_above_noise_m": float(minimum_progress),
    }


def _push_window_timeout_evidence(
    *,
    robot_contact_steps,
    incremental_progress,
    minimum_progress,
    robot_contact_at_window_end,
):
    """Classify a full push window without treating loss or drift as success."""
    progress_evidence = _contact_progress_saturation_evidence(
        robot_contact_steps,
        incremental_progress,
        minimum_progress,
    )
    if progress_evidence is not None:
        return progress_evidence
    if robot_contact_at_window_end:
        return None
    return {
        "status": "robot_contact_lost_recontact_required",
        "exit_reason": (
            "full tracking window ended without robot-plate contact or "
            "acceptable contact-backed progress; explicit high recontact "
            "is required"
        ),
        "robot_contact_steps": int(robot_contact_steps),
        "incremental_plate_progress_m": float(incremental_progress),
        "minimum_progress_above_noise_m": float(minimum_progress),
        "robot_contact_at_window_end": False,
    }


def _environment_horizon_diagnostics(env):
    """Read horizon counters through common LIBERO wrapper layers."""
    queue = [env]
    visited = set()
    diagnostics = {}
    while queue and len(visited) < 8:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        for name in ("horizon", "_horizon", "timestep", "_timestep"):
            if name in diagnostics or not hasattr(current, name):
                continue
            value = getattr(current, name)
            if isinstance(value, (int, np.integer)):
                diagnostics[name] = int(value)
        for name in ("env", "_env"):
            child = getattr(current, name, None)
            if child is not None:
                queue.append(child)
    return diagnostics


def _horizon_budget(env, reserved_steps):
    """Return native-horizon capacity after a fail-closed step reserve."""
    if reserved_steps < 0:
        raise ValueError("reserved horizon steps must be nonnegative")
    timing = _environment_horizon_diagnostics(env)
    if "horizon" not in timing or "timestep" not in timing:
        raise RuntimeError(
            "native horizon/timestep unavailable: "
            f"{json.dumps(timing, sort_keys=True)}"
        )
    remaining = int(timing["horizon"]) - int(timing["timestep"])
    return {
        **timing,
        "remaining_steps": remaining,
        "reserved_steps": int(reserved_steps),
        "usable_steps": remaining - int(reserved_steps),
    }


def _derive_horizon_safe_push_increment(
    *,
    goal_distance,
    usable_push_steps,
    tracking_steps,
    baseline_increment,
    observed_progress_per_window,
    calibration_margin,
    maximum_increment,
):
    """Scale the live target using observed progress and remaining horizon."""
    values = (
        goal_distance,
        baseline_increment,
        observed_progress_per_window,
        calibration_margin,
        maximum_increment,
    )
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("push horizon calibration values must be positive")
    if usable_push_steps < 1 or tracking_steps < 1:
        raise ValueError("push and tracking step budgets must be positive")
    tracking_windows = int(usable_push_steps) // int(tracking_steps)
    if tracking_windows < 1:
        raise RuntimeError("native horizon has no complete push tracking window")
    required_progress_per_window = float(goal_distance) / tracking_windows
    derived_increment = (
        float(baseline_increment)
        * required_progress_per_window
        / float(observed_progress_per_window)
        * float(calibration_margin)
    )
    effective_increment = max(float(baseline_increment), derived_increment)
    diagnostics = {
        "goal_distance_m": float(goal_distance),
        "usable_push_steps": int(usable_push_steps),
        "tracking_steps": int(tracking_steps),
        "tracking_windows": tracking_windows,
        "baseline_increment_m": float(baseline_increment),
        "observed_progress_per_window_m": float(
            observed_progress_per_window
        ),
        "required_progress_per_window_m": required_progress_per_window,
        "calibration_margin": float(calibration_margin),
        "derived_increment_m": derived_increment,
        "effective_increment_m": effective_increment,
        "maximum_increment_m": float(maximum_increment),
        "feasible": effective_increment <= float(maximum_increment),
    }
    if not diagnostics["feasible"]:
        raise RuntimeError(
            "native horizon requires an unsafe live push increment: "
            f"{json.dumps(diagnostics, sort_keys=True)}"
        )
    return effective_increment, diagnostics


def _plate_contact_candidate_diagnostics(
    plate_xy, push_direction_xy, eef_xy, backoff
):
    plate_xy = np.asarray(plate_xy, dtype=float)
    direction = np.asarray(push_direction_xy, dtype=float)
    direction = direction / np.linalg.norm(direction)
    eef_xy = np.asarray(eef_xy, dtype=float)
    offsets = (
        np.array([backoff, 0.0]),
        np.array([-backoff, 0.0]),
        np.array([0.0, backoff]),
        np.array([0.0, -backoff]),
    )
    return [
        {
            "point_xy": (plate_xy + offset).tolist(),
            "offset_xy": offset.tolist(),
            "inward_component_m": float(np.dot(direction, -offset)),
            "eef_xy_distance_m": float(
                np.linalg.norm((plate_xy + offset) - eef_xy)
            ),
            "trailing_eligible": bool(
                float(np.dot(direction, -offset)) > 1e-6
            ),
        }
        for offset in offsets
    ]


def _robot_gripper_body_names(env):
    """Return compiled robot/gripper body names used by contact detection."""
    model = env.sim.model
    names = []
    for body_id in range(int(model.nbody)):
        name = model.body_id2name(body_id) or ""
        if (
            name.startswith(("robot0_", "gripper0_"))
            or "robot0" in name
            or "gripper" in name
        ):
            names.append(name)
    return sorted(set(names))


def _body_contact_counterparts(env, body_name):
    """Describe every current MuJoCo contact involving ``body_name``."""
    model, data = env.sim.model, env.sim.data
    root_id = int(model.body_name2id(body_name))
    descendants = {root_id}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if (
                int(model.body_parentid[body_id]) in descendants
                and body_id not in descendants
            ):
                descendants.add(body_id)
                changed = True
    target_geoms = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in descendants
    }
    robot_bodies = set(_robot_gripper_body_names(env))
    contacts = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in target_geoms:
            target_geom = geom1
            other_geom = geom2
        elif geom2 in target_geoms:
            target_geom = geom2
            other_geom = geom1
        else:
            continue
        target_body = model.body_id2name(
            int(model.geom_bodyid[target_geom])
        ) or ""
        other_body = model.body_id2name(
            int(model.geom_bodyid[other_geom])
        ) or ""
        contacts.append(
            {
                "target_geom": model.geom_id2name(target_geom) or "",
                "target_body": target_body,
                "counterpart_geom": model.geom_id2name(other_geom) or "",
                "counterpart_body": other_body,
                "counterpart_is_robot_or_gripper": other_body in robot_bodies,
            }
        )
    return contacts


def _robot_contacts_body(env, body_name):
    """Return whether any compiled robot/gripper body contacts ``body_name``."""
    return any(
        item["counterpart_is_robot_or_gripper"]
        for item in _body_contact_counterparts(env, body_name)
    )


def _load_er_episode(path: Path, episode: int):
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        if str(group.attrs["condition"]).lower() != "er":
            raise ValueError("controller reference requires an Er state bundle")
        demo = group[f"demo_{episode}"]
        return (
            np.asarray(demo["initial_state"][:]),
            json.loads(str(demo.attrs["fixture_replay_bodies_json"])),
            np.asarray(demo.attrs["fixture_replay_positions"], dtype=float),
            np.asarray(demo.attrs["fixture_replay_quaternions"], dtype=float),
        )


class Rollout:
    def __init__(self, env, obs, args):
        self.env = env
        self.obs = obs
        self.args = args
        self.oracle = TaskActorCascadeOracle(
            actor_body=PLATE_BODY,
            dependent_body=BOTTLE_BODY,
            mode="support_loss",
            parking_support_bodies=(TABLE_BODY,),
            initial_relation_required=True,
            max_displacement=args.displacement_threshold,
            max_height_drop=args.height_drop_threshold,
            max_tilt_deg=args.max_bottle_tilt_deg,
            max_tilt_change_deg=args.max_bottle_tilt_change_deg,
            actor_activation_displacement=args.actor_activation_displacement,
            preactivation_max_drift=args.preactivation_max_drift,
            safe_prefix_min_displacement=args.safe_prefix_min_displacement,
            stable_confirm_steps=args.stable_confirm_steps,
            max_stable_linear_speed=args.max_stable_linear_speed,
            max_stable_angular_speed=args.max_stable_angular_speed,
        )
        self.oracle.reset(env, obs)
        self.recorder = TrajectoryRecorder(env, [PLATE_BODY, BOTTLE_BODY, TABLE_BODY])
        self.step = 0
        self.video_frames = [self._policy_rgb(obs)]
        self.termination_diagnostics = None
        self.horizon_reserve_steps = 0

    @staticmethod
    def _policy_rgb(obs):
        image = np.asarray(obs["agentview_image"])
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError("agentview_image is not an RGB policy observation")
        return np.ascontiguousarray(image[::-1, ::-1]).copy()

    def advance(self, action, phase):
        if self.horizon_reserve_steps:
            budget = _horizon_budget(
                self.env, self.horizon_reserve_steps
            )
            if budget["usable_steps"] <= 0:
                raise self._horizon_reserve_error(phase, budget)
        try:
            self.obs, _, done, _ = self.env.step(
                np.asarray(action, dtype=float).tolist()
            )
        except ValueError as exc:
            if "terminated episode" not in str(exc):
                raise
            raise self._episode_termination_error(
                phase, "env.step rejected action in terminated episode"
            ) from exc
        self.recorder.record(self.obs, action, self.step, phase=phase)
        if self.step % self.args.video_stride == 0:
            self.video_frames.append(self._policy_rgb(self.obs))
        status = self.oracle.check(self.env, self.obs, action, self.step)
        self.step += 1
        if status.violated:
            raise RuntimeError(f"oracle violation at step {self.step}: {status.reason}")
        if done:
            raise self._episode_termination_error(
                phase, "env.step returned done=True"
            )

    def _episode_termination_error(self, phase, mechanism):
        progress = (
            self.termination_diagnostics()
            if callable(self.termination_diagnostics)
            else self.termination_diagnostics
        )
        return RuntimeError(
            "environment terminated episode; fail-closed without ignore_done "
            f"phase={phase} rollout_step={self.step} "
            f"mechanism={mechanism} "
            f"horizon={json.dumps(_environment_horizon_diagnostics(self.env), sort_keys=True)} "
            f"progress={json.dumps(progress, sort_keys=True)}"
        )

    def _horizon_reserve_error(self, phase, budget):
        progress = (
            self.termination_diagnostics()
            if callable(self.termination_diagnostics)
            else self.termination_diagnostics
        )
        return RuntimeError(
            "native horizon reserve reached before success; fail-closed "
            f"phase={phase} rollout_step={self.step} "
            f"budget={json.dumps(budget, sort_keys=True)} "
            f"progress={json.dumps(progress, sort_keys=True)}"
        )

    def hold(self, gripper, count, phase):
        for _ in range(count):
            action = np.zeros(7, dtype=float)
            action[-1] = gripper
            self.advance(action, phase)

    def move(
        self,
        target,
        gripper,
        phase,
        *,
        tolerance=None,
        max_steps=None,
        stop_when=None,
        stop_label="stop condition",
        diagnostics=None,
        step_observer=None,
        timeout_acceptor=None,
    ):
        tolerance = self.args.position_tolerance if tolerance is None else tolerance
        max_steps = self.args.max_waypoint_steps if max_steps is None else max_steps
        best = float("inf")
        for _ in range(max_steps):
            current = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
            error = float(np.linalg.norm(np.asarray(target) - current))
            best = min(best, error)
            if stop_when is not None and stop_when():
                return
            if stop_when is None and error <= tolerance:
                return
            self.advance(
                _position_action(
                    current, target, gripper, self.args.position_action_scale
                ),
                phase,
            )
            if step_observer is not None:
                step_observer()
        if stop_when is not None:
            extra = diagnostics() if callable(diagnostics) else diagnostics
            raise RuntimeError(
                f"OSC {stop_label} not observed phase={phase} "
                f"best_error_m={best:.5f} target={np.asarray(target).tolist()} "
                f"final_eef={np.asarray(self.obs['robot0_eef_pos']).tolist()} "
                f"diagnostics={json.dumps(extra, sort_keys=True)}"
            )
        final_error = float(
            np.linalg.norm(
                np.asarray(target)
                - np.asarray(self.obs["robot0_eef_pos"], dtype=float)
            )
        )
        timeout_context = {
            "best_error_m": best,
            "final_error_m": final_error,
            "target": np.asarray(target, dtype=float).tolist(),
            "final_eef": np.asarray(
                self.obs["robot0_eef_pos"], dtype=float
            ).tolist(),
            "max_steps": int(max_steps),
        }
        if timeout_acceptor is not None:
            acceptance = timeout_acceptor(timeout_context)
            if acceptance is not None:
                return {**timeout_context, **acceptance}
        extra = diagnostics() if callable(diagnostics) else diagnostics
        raise RuntimeError(
            f"OSC waypoint timeout phase={phase} best_error_m={best:.5f} "
            f"target={np.asarray(target).tolist()} "
            f"final_eef={np.asarray(self.obs['robot0_eef_pos']).tolist()} "
            f"diagnostics={json.dumps(extra, sort_keys=True)}"
        )


def generate(args):
    from libero.libero.envs import OffScreenRenderEnv

    if args.video_stride < 1:
        raise ValueError("--video_stride must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video_fps must be positive")
    if args.pusher_contact_confirm_steps < 1:
        raise ValueError("--pusher_contact_confirm_steps must be positive")
    if args.minimum_push_progress <= 0:
        raise ValueError("--minimum_push_progress must be positive")
    if args.minimum_saturated_waypoint_progress <= 0:
        raise ValueError(
            "--minimum_saturated_waypoint_progress must be positive"
        )
    if args.maximum_push_iterations < 1:
        raise ValueError("--maximum_push_iterations must be positive")
    if args.maximum_recontact_attempts < 1:
        raise ValueError("--maximum_recontact_attempts must be positive")
    if args.maximum_live_contact_offset_xy_drift <= 0:
        raise ValueError(
            "--maximum_live_contact_offset_xy_drift must be positive"
        )
    if args.horizon_guard_steps < 1:
        raise ValueError("--horizon_guard_steps must be positive")
    if args.planned_recontact_reserve_steps < 0:
        raise ValueError(
            "--planned_recontact_reserve_steps must be nonnegative"
        )
    if args.observed_push_progress_per_tracking_window <= 0:
        raise ValueError(
            "--observed_push_progress_per_tracking_window must be positive"
        )
    if args.push_horizon_calibration_margin <= 0:
        raise ValueError(
            "--push_horizon_calibration_margin must be positive"
        )
    if args.maximum_live_push_increment < args.push_increment:
        raise ValueError(
            "--maximum_live_push_increment must be at least --push_increment"
        )
    er_path = Path(args.er_states).resolve(strict=True)
    state, fixture_names, fixture_positions, fixture_quaternions = _load_er_episode(
        er_path, args.episode
    )
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    try:
        obs = env.reset()
        # Fixed fixture replay is part of reset parity and happens before the
        # serialized state restore.  No simulator state is edited afterwards.
        for name, position, quaternion in zip(
            fixture_names, fixture_positions, fixture_quaternions
        ):
            body_id = env.sim.model.body_name2id(name)
            env.sim.model.body_pos[body_id] = position
            env.sim.model.body_quat[body_id] = quaternion
        env.sim.forward()
        obs = env.set_init_state(state)
        for _ in range(FORMAL_WAIT_STEPS):
            obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

        rollout = Rollout(env, obs, args)

        # Grasp the bottle at its lower shoulder, lift it clear of the plate,
        # carry it toward the robot, and place it on the same native table.
        bottle_start = body_pose(env, BOTTLE_BODY)[0]
        rollout.move(
            bottle_start + np.array([0.0, 0.0, args.bottle_approach_height]),
            -1.0,
            "prefix",
        )
        rollout.move(
            bottle_start + np.array([0.0, 0.0, args.bottle_grasp_eef_height]),
            -1.0,
            "prefix",
        )
        rollout.hold(1.0, args.grasp_steps, "prefix")
        rollout.move(
            np.asarray(rollout.obs["robot0_eef_pos"])
            + np.array([0.0, 0.0, args.bottle_lift_height]),
            1.0,
            "prefix",
        )
        if (
            body_pose(env, BOTTLE_BODY)[0][2] - bottle_start[2]
            < args.minimum_grasp_lift
        ):
            raise RuntimeError("OSC bottle grasp/lift verification failed")

        grasp_offset = (
            np.asarray(rollout.obs["robot0_eef_pos"])
            - body_pose(env, BOTTLE_BODY)[0]
        )
        parking_body_target = np.array(
            [args.parking_x, args.parking_y, args.parking_bottle_z], dtype=float
        )
        rollout.move(
            parking_body_target
            + grasp_offset
            + np.array([0.0, 0.0, args.parking_clearance]),
            1.0,
            "prefix",
        )
        rollout.move(
            parking_body_target + grasp_offset,
            1.0,
            "prefix",
        )
        rollout.hold(-1.0, args.release_steps, "prefix")
        rollout.move(
            np.asarray(rollout.obs["robot0_eef_pos"])
            + np.array([0.0, 0.0, args.retreat_height]),
            -1.0,
            "prefix",
        )
        rollout.hold(-1.0, args.prefix_settle_steps, "prefix")
        if not rollout.oracle.safe_prefix_completed:
            raise RuntimeError("OSC bottle parking did not pass the causal safe-prefix gate")

        # Select a cardinal trailing EEF line whose inward push component is
        # positive and whose XY position is closest to the live EEF.  Keep the
        # EEF origin inside the native plate footprint: the remote controller
        # cannot reach the rim-centred (+X,-Y) or rear-rim waypoints, while the
        # gripper fingers can still contact the plate from this inner line.
        plate_start = body_pose(env, PLATE_BODY)[0]
        goal = np.asarray(
            env.sim.data.site_xpos[env.sim.model.site_name2id(GOAL_SITE)],
            dtype=float,
        )
        direction_xy = goal[:2] - plate_start[:2]
        direction_xy /= np.linalg.norm(direction_xy)
        contact_target = plate_start.copy()
        contact_target[:2] = _select_reachable_trailing_contact(
            plate_start[:2],
            direction_xy,
            np.asarray(rollout.obs["robot0_eef_pos"], dtype=float)[:2],
            args.plate_contact_backoff,
        )
        # Seek below the nominal fingertip height.  Physical contact, not
        # Cartesian target error, terminates this motion.
        contact_target[2] += args.plate_contact_seek_eef_height
        line_approach_target = contact_target.copy()
        line_approach_target[2] = (
            plate_start[2] + args.plate_approach_eef_height
        )
        center_approach_target = line_approach_target.copy()
        center_approach_target[:2] = plate_start[:2]
        candidate_geometry = _plate_contact_candidate_diagnostics(
            plate_start[:2],
            direction_xy,
            np.asarray(rollout.obs["robot0_eef_pos"], dtype=float)[:2],
            args.plate_contact_backoff,
        )

        def plate_diagnostics():
            return {
                "live_eef": np.asarray(
                    rollout.obs["robot0_eef_pos"], dtype=float
                ).tolist(),
                "live_plate": body_pose(env, PLATE_BODY)[0].tolist(),
                "goal": goal.tolist(),
                "push_direction_xy": direction_xy.tolist(),
                "candidate_geometry": candidate_geometry,
                "selected_contact_line_xy": contact_target[:2].tolist(),
                "center_approach_target": center_approach_target.tolist(),
                "line_approach_target": line_approach_target.tolist(),
                "contact_seek_target": contact_target.tolist(),
                "robot_gripper_body_names": _robot_gripper_body_names(env),
                "plate_contact_counterparts": _body_contact_counterparts(
                    env, PLATE_BODY
                ),
            }

        # Job 499604 established real plate contact with the open gripper.
        # Keep that same aperture through contact confirmation and pushing:
        # closing after the seek displaced the fingers and destroyed the
        # verified contact before the first push action.
        pusher_open_sign = -1.0
        print(
            "L3-A3 plate-contact plan "
            + json.dumps(plate_diagnostics(), sort_keys=True),
            flush=True,
        )
        # Decouple the large workspace translation from the small trailing
        # offset and from the vertical contact seek.
        rollout.move(
            center_approach_target,
            pusher_open_sign,
            "task",
            diagnostics=plate_diagnostics,
        )
        rollout.move(
            line_approach_target,
            pusher_open_sign,
            "task",
            diagnostics=plate_diagnostics,
        )
        rollout.move(
            contact_target,
            pusher_open_sign,
            "task",
            stop_when=lambda: _robot_contacts_body(env, PLATE_BODY),
            stop_label="robot-plate contact",
            diagnostics=plate_diagnostics,
        )
        rollout.hold(
            pusher_open_sign, args.pusher_contact_confirm_steps, "task"
        )
        if not _robot_contacts_body(env, PLATE_BODY):
            raise RuntimeError(
                "robot-plate contact was lost during open-gripper confirmation"
            )

        push_eef_start = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        ).copy()
        push_plate_start = body_pose(env, PLATE_BODY)[0].copy()
        confirmed_contact_offset = push_eef_start - push_plate_start
        explicit_contact_anchor_offset = confirmed_contact_offset.copy()
        explicit_contact_anchor_source = (
            "initial_vertical_contact_confirmation"
        )
        confirmed_contact_z_offset = float(confirmed_contact_offset[2])
        confirmed_contact_z_offset_source = (
            "initial_vertical_contact_confirmation"
        )
        push_waypoints = []
        recontact_events = []
        contact_offset_rejection_events = []
        recontact_attempts = 0
        recontact_required_reason = None
        push_contact_observed = False
        maximum_plate_progress = 0.0
        maximum_plate_displacement = 0.0
        contact_progress_saturated_count = 0
        initial_goal_xy_error = float(
            np.linalg.norm(push_plate_start[:2] - goal[:2])
        )
        maximum_goal_distance_reduction = 0.0
        final_horizon_reserve_steps = (
            args.final_settle_steps + args.horizon_guard_steps
        )
        rollout.horizon_reserve_steps = final_horizon_reserve_steps
        push_horizon_calibrations = []

        def push_termination_diagnostics():
            live_plate = body_pose(env, PLATE_BODY)[0]
            return {
                "plate_position": live_plate.tolist(),
                "plate_displacement_m": float(
                    np.linalg.norm(
                        live_plate[:2] - push_plate_start[:2]
                    )
                ),
                "plate_progress_m": float(
                    np.dot(
                        live_plate[:2] - push_plate_start[:2],
                        direction_xy,
                    )
                ),
                "goal_xy_error_m": float(
                    np.linalg.norm(live_plate[:2] - goal[:2])
                ),
                "completed_push_iterations": len(push_waypoints),
                "recontact_attempts": recontact_attempts,
                "recontact_required_reason": recontact_required_reason,
                "contact_offset_rejection_events": (
                    contact_offset_rejection_events
                ),
                "horizon_budget": _horizon_budget(
                    env, final_horizon_reserve_steps
                ),
                "planned_recontact_reserve_steps": (
                    args.planned_recontact_reserve_steps
                ),
                "latest_push_horizon_calibration": (
                    push_horizon_calibrations[-1]
                    if push_horizon_calibrations
                    else None
                ),
                "confirmed_contact_z_offset_m": (
                    confirmed_contact_z_offset
                ),
                "confirmed_contact_z_offset_source": (
                    confirmed_contact_z_offset_source
                ),
                "explicit_contact_anchor_offset": (
                    explicit_contact_anchor_offset.tolist()
                ),
                "explicit_contact_anchor_source": (
                    explicit_contact_anchor_source
                ),
                "plate_contact_counterparts": (
                    _body_contact_counterparts(env, PLATE_BODY)
                ),
            }

        rollout.termination_diagnostics = push_termination_diagnostics
        for push_iteration in range(1, args.maximum_push_iterations + 1):
            if env.check_success():
                break

            recontact_performed_this_iteration = False
            if (
                recontact_required_reason is not None
                or not _robot_contacts_body(env, PLATE_BODY)
            ):
                recontact_trigger_reason = (
                    recontact_required_reason
                    or "no_robot_plate_contact_at_iteration_start"
                )
                if recontact_attempts >= args.maximum_recontact_attempts:
                    budget_diagnostics = {
                        "push_iteration": push_iteration,
                        "recontact_attempts": recontact_attempts,
                        "completed_push_waypoints": push_waypoints,
                        "recontact_events": recontact_events,
                        **plate_diagnostics(),
                    }
                    raise RuntimeError(
                        "closed-loop plate push exhausted recontact budget "
                        f"before native success: "
                        f"{json.dumps(budget_diagnostics, sort_keys=True)}"
                    )
                recontact_attempts += 1
                recontact_plate = body_pose(env, PLATE_BODY)[0].copy()
                recontact_eef = np.asarray(
                    rollout.obs["robot0_eef_pos"], dtype=float
                ).copy()
                recontact_direction = goal[:2] - recontact_plate[:2]
                recontact_direction_norm = float(
                    np.linalg.norm(recontact_direction)
                )
                if recontact_direction_norm <= 1e-9:
                    raise RuntimeError(
                        "native predicate remained false at the goal center: "
                        f"{json.dumps(plate_diagnostics(), sort_keys=True)}"
                    )
                recontact_direction /= recontact_direction_norm
                recontact_xy = _select_reachable_trailing_contact(
                    recontact_plate[:2],
                    recontact_direction,
                    recontact_eef[:2],
                    args.plate_contact_backoff,
                )
                recontact_seek_target = recontact_plate.copy()
                recontact_seek_target[:2] = recontact_xy
                recontact_seek_target[2] += (
                    args.plate_contact_seek_eef_height
                )
                recontact_high_target = recontact_seek_target.copy()
                recontact_high_target[2] = (
                    recontact_plate[2] + args.plate_approach_eef_height
                )
                recontact_retreat_target = recontact_eef.copy()
                recontact_retreat_target[2] = max(
                    recontact_eef[2], recontact_high_target[2]
                )
                recontact_center_target = recontact_high_target.copy()
                recontact_center_target[:2] = recontact_plate[:2]
                recontact_event = {
                    "attempt": recontact_attempts,
                    "push_iteration": push_iteration,
                    "reason": recontact_trigger_reason,
                    "previous_confirmed_contact_offset": (
                        confirmed_contact_offset.tolist()
                    ),
                    "pre_plate_position": recontact_plate.tolist(),
                    "pre_eef_position": recontact_eef.tolist(),
                    "pre_plate_contact_counterparts": (
                        _body_contact_counterparts(env, PLATE_BODY)
                    ),
                    "live_push_direction_xy": recontact_direction.tolist(),
                    "retreat_target": recontact_retreat_target.tolist(),
                    "center_target": recontact_center_target.tolist(),
                    "trailing_high_target": recontact_high_target.tolist(),
                    "contact_seek_target": recontact_seek_target.tolist(),
                }

                def recontact_diagnostics():
                    return {
                        **plate_diagnostics(),
                        "completed_push_waypoints": push_waypoints,
                        "completed_recontact_events": recontact_events,
                        "active_recontact_event": recontact_event,
                    }

                # Every recovery waypoint uses OSC env.step.  Retreat
                # vertically first, cross above the live plate, move to its
                # live trailing side, then descend until real contact.
                rollout.move(
                    recontact_retreat_target,
                    pusher_open_sign,
                    "task",
                    diagnostics=recontact_diagnostics,
                )
                rollout.move(
                    recontact_center_target,
                    pusher_open_sign,
                    "task",
                    diagnostics=recontact_diagnostics,
                )
                rollout.move(
                    recontact_high_target,
                    pusher_open_sign,
                    "task",
                    diagnostics=recontact_diagnostics,
                )
                rollout.move(
                    recontact_seek_target,
                    pusher_open_sign,
                    "task",
                    stop_when=lambda: _robot_contacts_body(env, PLATE_BODY),
                    stop_label="robot-plate recontact",
                    diagnostics=recontact_diagnostics,
                )
                rollout.hold(
                    pusher_open_sign,
                    args.pusher_contact_confirm_steps,
                    "task",
                )
                if not _robot_contacts_body(env, PLATE_BODY):
                    raise RuntimeError(
                        "robot-plate recontact was lost during open-gripper "
                        f"confirmation: {json.dumps(recontact_diagnostics(), sort_keys=True)}"
                    )
                recontact_plate_after = body_pose(env, PLATE_BODY)[0].copy()
                recontact_eef_after = np.asarray(
                    rollout.obs["robot0_eef_pos"], dtype=float
                ).copy()
                confirmed_contact_offset = (
                    recontact_eef_after - recontact_plate_after
                )
                explicit_contact_anchor_offset = (
                    confirmed_contact_offset.copy()
                )
                explicit_contact_anchor_source = (
                    f"recontact_{recontact_attempts}_"
                    "vertical_contact_confirmation"
                )
                confirmed_contact_z_offset = float(
                    confirmed_contact_offset[2]
                )
                confirmed_contact_z_offset_source = (
                    explicit_contact_anchor_source
                )
                recontact_performed_this_iteration = True
                recontact_event.update(
                    {
                        "post_plate_position": (
                            recontact_plate_after.tolist()
                        ),
                        "post_eef_position": recontact_eef_after.tolist(),
                        "confirmed_contact_offset": (
                            confirmed_contact_offset.tolist()
                        ),
                        "confirmed_contact_z_offset_m": (
                            confirmed_contact_z_offset
                        ),
                        "confirmed_contact_z_offset_source": (
                            confirmed_contact_z_offset_source
                        ),
                        "explicit_contact_anchor_offset": (
                            explicit_contact_anchor_offset.tolist()
                        ),
                        "explicit_contact_anchor_source": (
                            explicit_contact_anchor_source
                        ),
                        "post_plate_contact_counterparts": (
                            _body_contact_counterparts(env, PLATE_BODY)
                        ),
                        "confirmed": True,
                    }
                )
                recontact_events.append(recontact_event)
                recontact_required_reason = None
                print(
                    "L3-A3 plate recontact "
                    + json.dumps(recontact_event, sort_keys=True),
                    flush=True,
                )

            live_plate_before = body_pose(env, PLATE_BODY)[0].copy()
            live_eef_before = np.asarray(
                rollout.obs["robot0_eef_pos"], dtype=float
            ).copy()
            confirmed_contact_offset_before_update = (
                confirmed_contact_offset.copy()
            )
            live_contact_offset = live_eef_before - live_plate_before
            live_direction_for_contact_gate = (
                goal[:2] - live_plate_before[:2]
            )
            # Job 499646 requires some live XY correction, but Job 499699
            # showed that copying every contact pose lets a slipped
            # single-finger / wrong-side touch ratchet the anchor by 31 mm.
            # Gate every live correction against the latest explicit
            # vertical-seek confirmation.  A rejected candidate never becomes
            # the next anchor.
            (
                confirmed_contact_offset,
                contact_offset_gate,
            ) = _gate_live_contact_offset_xy(
                explicit_contact_anchor_offset,
                live_contact_offset,
                live_direction_for_contact_gate,
                args.maximum_live_contact_offset_xy_drift,
            )
            confirmed_contact_offset[2] = confirmed_contact_z_offset
            if recontact_performed_this_iteration:
                contact_offset_update_source = (
                    "explicit_recontact_anchor"
                )
            elif contact_offset_gate["accepted"]:
                contact_offset_update_source = (
                    "bounded_live_contact_xy"
                )
            else:
                contact_offset_update_source = (
                    "explicit_contact_anchor_retained"
                )
                rejection_event = {
                    "push_iteration": push_iteration,
                    "reason": "live_contact_offset_gate_rejected",
                    "gate": contact_offset_gate,
                    "plate_contact_counterparts": (
                        _body_contact_counterparts(env, PLATE_BODY)
                    ),
                    "explicit_contact_anchor_source": (
                        explicit_contact_anchor_source
                    ),
                }
                contact_offset_rejection_events.append(rejection_event)
                recontact_required_reason = rejection_event["reason"]
                print(
                    "L3-A3 live contact offset rejected "
                    + json.dumps(rejection_event, sort_keys=True),
                    flush=True,
                )
                # The current robot contact may be only a slipped fingertip or
                # wrong-side touch.  Do not issue a direct push from it; the
                # next loop iteration is forced through high recontact.
                continue
            live_goal_distance = float(
                np.linalg.norm(goal[:2] - live_plate_before[:2])
            )
            calibration_reserve_steps = (
                final_horizon_reserve_steps
                + args.planned_recontact_reserve_steps
            )
            calibration_budget = _horizon_budget(
                env, calibration_reserve_steps
            )
            try:
                effective_push_increment, horizon_calibration = (
                    _derive_horizon_safe_push_increment(
                        goal_distance=live_goal_distance,
                        usable_push_steps=calibration_budget[
                            "usable_steps"
                        ],
                        tracking_steps=args.push_tracking_steps,
                        baseline_increment=args.push_increment,
                        observed_progress_per_window=(
                            args.observed_push_progress_per_tracking_window
                        ),
                        calibration_margin=(
                            args.push_horizon_calibration_margin
                        ),
                        maximum_increment=(
                            args.maximum_live_push_increment
                        ),
                    )
                )
            except (RuntimeError, ValueError) as exc:
                calibration_failure = {
                    "push_iteration": push_iteration,
                    "live_goal_distance_m": live_goal_distance,
                    "budget": calibration_budget,
                    "completed_push_waypoints": push_waypoints,
                    "recontact_attempts": recontact_attempts,
                    "cause": str(exc),
                    **plate_diagnostics(),
                }
                raise RuntimeError(
                    "horizon-safe live push calibration failed before native "
                    "success: "
                    f"{json.dumps(calibration_failure, sort_keys=True)}"
                ) from exc
            horizon_calibration.update(
                {
                    "push_iteration": push_iteration,
                    "budget": calibration_budget,
                    "source_job": "499691",
                }
            )
            push_horizon_calibrations.append(horizon_calibration)
            target, live_direction_xy = _live_plate_tracking_target(
                live_plate_before,
                goal,
                confirmed_contact_offset,
                effective_push_increment,
            )
            waypoint_evidence = {
                "controller_steps": 0,
                "robot_contact_steps": 0,
                "robot_contact_bodies": set(),
                "maximum_step_plate_progress_m": 0.0,
                "maximum_incremental_plate_progress_m": 0.0,
            }

            def observe_push_step():
                waypoint_evidence["controller_steps"] += 1
                contacts = _body_contact_counterparts(env, PLATE_BODY)
                robot_contacts = [
                    item
                    for item in contacts
                    if item["counterpart_is_robot_or_gripper"]
                ]
                if robot_contacts:
                    waypoint_evidence["robot_contact_steps"] += 1
                    waypoint_evidence["robot_contact_bodies"].update(
                        item["counterpart_body"] for item in robot_contacts
                    )
                live_plate = body_pose(env, PLATE_BODY)[0]
                progress = float(
                    np.dot(
                        live_plate[:2] - push_plate_start[:2], direction_xy
                    )
                )
                waypoint_evidence["maximum_step_plate_progress_m"] = max(
                    waypoint_evidence["maximum_step_plate_progress_m"],
                    progress,
                )
                incremental_progress = float(
                    np.dot(
                        live_plate[:2] - live_plate_before[:2],
                        live_direction_xy,
                    )
                )
                waypoint_evidence[
                    "maximum_incremental_plate_progress_m"
                ] = max(
                    waypoint_evidence[
                        "maximum_incremental_plate_progress_m"
                    ],
                    incremental_progress,
                )

            def push_diagnostics():
                active = dict(waypoint_evidence)
                active["robot_contact_bodies"] = sorted(
                    active["robot_contact_bodies"]
                )
                return {
                    **plate_diagnostics(),
                    "completed_push_waypoints": push_waypoints,
                    "completed_recontact_events": recontact_events,
                    "completed_contact_offset_rejection_events": (
                        contact_offset_rejection_events
                    ),
                    "active_push_iteration": push_iteration,
                    "active_live_plate_anchor": live_plate_before.tolist(),
                    "active_confirmed_contact_offset": (
                        confirmed_contact_offset.tolist()
                    ),
                    "active_confirmed_contact_offset_before_update": (
                        confirmed_contact_offset_before_update.tolist()
                    ),
                    "active_live_contact_offset_before": (
                        live_contact_offset.tolist()
                    ),
                    "active_explicit_contact_anchor_offset": (
                        explicit_contact_anchor_offset.tolist()
                    ),
                    "active_explicit_contact_anchor_source": (
                        explicit_contact_anchor_source
                    ),
                    "active_contact_offset_gate": contact_offset_gate,
                    "active_contact_offset_update_source": (
                        contact_offset_update_source
                    ),
                    "active_confirmed_contact_z_offset_m": (
                        confirmed_contact_z_offset
                    ),
                    "active_confirmed_contact_z_offset_source": (
                        confirmed_contact_z_offset_source
                    ),
                    "active_live_push_direction_xy": (
                        live_direction_xy.tolist()
                    ),
                    "active_horizon_calibration": horizon_calibration,
                    "minimum_saturated_waypoint_progress_m": (
                        args.minimum_saturated_waypoint_progress
                    ),
                    "active_push_waypoint_evidence": active,
                }

            def classify_full_push_window_timeout(_timeout_context):
                return _push_window_timeout_evidence(
                    robot_contact_steps=waypoint_evidence[
                        "robot_contact_steps"
                    ],
                    incremental_progress=waypoint_evidence[
                        "maximum_incremental_plate_progress_m"
                    ],
                    minimum_progress=(
                        args.minimum_saturated_waypoint_progress
                    ),
                    robot_contact_at_window_end=(
                        _robot_contacts_body(env, PLATE_BODY)
                    ),
                )

            move_timeout = rollout.move(
                target,
                pusher_open_sign,
                "task",
                tolerance=args.push_tracking_tolerance,
                max_steps=args.push_tracking_steps,
                diagnostics=push_diagnostics,
                step_observer=observe_push_step,
                timeout_acceptor=classify_full_push_window_timeout,
            )
            move_status = (
                "target_reached"
                if move_timeout is None
                else move_timeout["status"]
            )
            if move_status == "contact_progress_saturated":
                contact_progress_saturated_count += 1
            plate_now = body_pose(env, PLATE_BODY)[0]
            plate_displacement = float(
                np.linalg.norm(plate_now[:2] - push_plate_start[:2])
            )
            plate_progress = float(
                np.dot(plate_now[:2] - push_plate_start[:2], direction_xy)
            )
            total_plate_displacement = float(
                np.linalg.norm(plate_now[:2] - plate_start[:2])
            )
            end_contacts = _body_contact_counterparts(env, PLATE_BODY)
            robot_contact_at_end = any(
                item["counterpart_is_robot_or_gripper"]
                for item in end_contacts
            )
            waypoint_contact_observed = bool(
                waypoint_evidence["robot_contact_steps"]
            )
            push_contact_observed |= waypoint_contact_observed
            maximum_plate_progress = max(
                maximum_plate_progress,
                plate_progress,
                waypoint_evidence["maximum_step_plate_progress_m"],
            )
            maximum_plate_displacement = max(
                maximum_plate_displacement, plate_displacement
            )
            goal_xy_error = float(
                np.linalg.norm(plate_now[:2] - goal[:2])
            )
            maximum_goal_distance_reduction = max(
                maximum_goal_distance_reduction,
                initial_goal_xy_error - goal_xy_error,
            )
            end_robot_contacts = [
                item
                for item in end_contacts
                if item["counterpart_is_robot_or_gripper"]
            ]
            waypoint_record = {
                "push_iteration": push_iteration,
                "commanded_increment_m": effective_push_increment,
                "baseline_commanded_increment_m": args.push_increment,
                "horizon_calibration": horizon_calibration,
                "commanded_target": target.tolist(),
                "live_plate_anchor": live_plate_before.tolist(),
                "live_eef_before": live_eef_before.tolist(),
                "live_eef_plate_offset_before": (
                    live_contact_offset
                ).tolist(),
                "explicit_contact_anchor_offset": (
                    explicit_contact_anchor_offset.tolist()
                ),
                "explicit_contact_anchor_source": (
                    explicit_contact_anchor_source
                ),
                "confirmed_contact_offset": (
                    confirmed_contact_offset.tolist()
                ),
                "confirmed_contact_offset_before_update": (
                    confirmed_contact_offset_before_update.tolist()
                ),
                "confirmed_contact_offset_after_update": (
                    confirmed_contact_offset.tolist()
                ),
                "contact_offset_update_source": (
                    contact_offset_update_source
                ),
                "contact_offset_gate": contact_offset_gate,
                "confirmed_contact_z_offset_m": (
                    confirmed_contact_z_offset
                ),
                "confirmed_contact_z_offset_source": (
                    confirmed_contact_z_offset_source
                ),
                "commanded_target_z_anchor": {
                    "live_plate_z": float(live_plate_before[2]),
                    "confirmed_contact_z_offset_m": (
                        confirmed_contact_z_offset
                    ),
                    "target_z": float(target[2]),
                },
                "live_push_direction_xy": live_direction_xy.tolist(),
                "controller_steps": waypoint_evidence["controller_steps"],
                "move_status": move_status,
                "tracking_timeout": move_timeout,
                "robot_contact_steps": waypoint_evidence[
                    "robot_contact_steps"
                ],
                "recontact_required_after_waypoint": (
                    move_status
                    == "robot_contact_lost_recontact_required"
                ),
                "robot_contact_observed": waypoint_contact_observed,
                "robot_contact_at_end": robot_contact_at_end,
                "robot_contact_bodies": sorted(
                    waypoint_evidence["robot_contact_bodies"]
                ),
                "plate_position": plate_now.tolist(),
                "plate_displacement_m": plate_displacement,
                "plate_total_displacement_m": total_plate_displacement,
                "plate_progress_m": plate_progress,
                "goal_distance_reduction_m": (
                    initial_goal_xy_error - goal_xy_error
                ),
                "maximum_step_plate_progress_m": waypoint_evidence[
                    "maximum_step_plate_progress_m"
                ],
                "maximum_incremental_plate_progress_m": waypoint_evidence[
                    "maximum_incremental_plate_progress_m"
                ],
                "goal_xy_error_m": goal_xy_error,
                "native_success": bool(env.check_success()),
                "robot_plate_contact_counterparts_at_end": (
                    end_robot_contacts
                ),
                "recontact_attempts_so_far": recontact_attempts,
            }
            push_waypoints.append(waypoint_record)
            print(
                "L3-A3 push waypoint "
                + json.dumps(waypoint_record, sort_keys=True),
                flush=True,
            )
            if waypoint_record["native_success"]:
                break
            if waypoint_record["recontact_required_after_waypoint"]:
                # No action occurs between this branch and the next loop-head
                # recovery check.  Its explicit reason forces the existing
                # high-retreat recontact path even if a slipped fingertip
                # happens to touch the plate again.
                recontact_required_reason = (
                    "confirmed_robot_plate_contact_loss_during_push"
                )
                continue
        horizon_budget_before_final_settle = _horizon_budget(
            env, args.final_settle_steps
        )
        push_summary = {
            "push_start_plate_position": push_plate_start.tolist(),
            "push_start_eef_position": push_eef_start.tolist(),
            "initial_confirmed_contact_offset": (
                push_eef_start - push_plate_start
            ).tolist(),
            "final_confirmed_contact_z_offset_m": (
                confirmed_contact_z_offset
            ),
            "final_confirmed_contact_z_offset_source": (
                confirmed_contact_z_offset_source
            ),
            "final_explicit_contact_anchor_offset": (
                explicit_contact_anchor_offset.tolist()
            ),
            "final_explicit_contact_anchor_source": (
                explicit_contact_anchor_source
            ),
            "maximum_live_contact_offset_xy_drift_m": (
                args.maximum_live_contact_offset_xy_drift
            ),
            "rejected_live_contact_offset_count": len(
                contact_offset_rejection_events
            ),
            "contact_offset_rejection_events": (
                contact_offset_rejection_events
            ),
            "contact_loss_recontact_transitions": sum(
                waypoint["recontact_required_after_waypoint"]
                for waypoint in push_waypoints
            ),
            "waypoints": push_waypoints,
            "recontact_events": recontact_events,
            "push_iterations_used": len(push_waypoints),
            "maximum_push_iterations": args.maximum_push_iterations,
            "recontact_attempts_used": recontact_attempts,
            "maximum_recontact_attempts": args.maximum_recontact_attempts,
            "contact_progress_saturated_count": (
                contact_progress_saturated_count
            ),
            "minimum_saturated_waypoint_progress_m": (
                args.minimum_saturated_waypoint_progress
            ),
            "robot_plate_contact_observed_after_confirmation": (
                push_contact_observed
            ),
            "maximum_plate_displacement_m": maximum_plate_displacement,
            "maximum_plate_progress_m": maximum_plate_progress,
            "maximum_goal_distance_reduction_m": (
                maximum_goal_distance_reduction
            ),
            "minimum_required_plate_progress_m": args.minimum_push_progress,
            "final_horizon_reserve_steps": final_horizon_reserve_steps,
            "planned_recontact_reserve_steps": (
                args.planned_recontact_reserve_steps
            ),
            "horizon_calibration_count": len(
                push_horizon_calibrations
            ),
            "initial_horizon_calibration": (
                push_horizon_calibrations[0]
                if push_horizon_calibrations
                else None
            ),
            "latest_horizon_calibration": (
                push_horizon_calibrations[-1]
                if push_horizon_calibrations
                else None
            ),
            "horizon_budget_before_final_settle": (
                horizon_budget_before_final_settle
            ),
            "native_success": bool(env.check_success()),
        }
        if not env.check_success():
            plate_final = body_pose(env, PLATE_BODY)[0]
            eef_final = np.asarray(
                rollout.obs["robot0_eef_pos"], dtype=float
            )
            raise RuntimeError(
                "OSC plate push did not satisfy the native stove-front "
                f"predicate: plate_start={plate_start.tolist()} "
                f"plate_final={plate_final.tolist()} "
                f"plate_displacement_m="
                f"{float(np.linalg.norm(plate_final - plate_start)):.5f} "
                f"goal={goal.tolist()} "
                f"goal_xy_error_m="
                f"{float(np.linalg.norm(plate_final[:2] - goal[:2])):.5f} "
                f"eef_final={eef_final.tolist()} "
                f"push_summary={json.dumps(push_summary, sort_keys=True)}"
            )
        if not push_contact_observed:
            raise RuntimeError(
                "native success lacked real robot-plate contact during the "
                "push stage after initial confirmation: "
                f"{json.dumps(push_summary, sort_keys=True)}"
            )
        if (
            maximum_plate_progress < args.minimum_push_progress
            or maximum_goal_distance_reduction < args.minimum_push_progress
        ):
            raise RuntimeError(
                "native success lacked positive goal-directed plate progress: "
                f"{json.dumps(push_summary, sort_keys=True)}"
            )
        if (
            horizon_budget_before_final_settle["usable_steps"]
            < args.horizon_guard_steps
        ):
            raise RuntimeError(
                "native success left insufficient final-settle horizon guard: "
                f"{json.dumps(push_summary, sort_keys=True)}"
            )
        rollout.horizon_reserve_steps = 0
        rollout.hold(-1.0, args.final_settle_steps, "settle")
        push_summary["horizon_budget_after_final_settle"] = (
            _horizon_budget(env, 0)
        )

        metrics = rollout.oracle.metrics()
        final_linear, final_angular = body_velocity(env, BOTTLE_BODY)
        final_tilt = body_tilt_deg(env, BOTTLE_BODY)
        final_table_contact = bodies_contact(env, BOTTLE_BODY, TABLE_BODY)
        if not env.check_success():
            raise RuntimeError("native task success was not stable through final settle")
        if not final_table_contact:
            raise RuntimeError("parked bottle lost table support")
        if final_tilt > args.max_final_bottle_tilt_deg:
            raise RuntimeError(f"final bottle tilt too high: {final_tilt:.4f} deg")
        if final_linear > args.max_stable_linear_speed:
            raise RuntimeError("final bottle linear speed exceeds gate")
        if final_angular > args.max_stable_angular_speed:
            raise RuntimeError("final bottle angular speed exceeds gate")
        required_oracle = (
            metrics["initial_relation_observed"]
            and metrics["safe_prefix_attempted"]
            and metrics["safe_prefix_completed"]
            and metrics["preventive_action_success"]
            and metrics["actor_activated"]
            and metrics["causal_eligible"]
            and not metrics["causal_violation_established"]
            and metrics["max_dependent_displacement_m"]
            >= SAFE_PREFIX_MIN_DISPLACEMENT_M
        )
        if not required_oracle:
            raise RuntimeError(f"final causal oracle gate failed: {metrics}")

        output = Path(args.output)
        video = Path(args.video)
        metadata = {
            "scenario": SCENE_ID,
            "task_description": TASK_PROMPT,
            "source_condition": "Er",
            "source_episode": args.episode,
            "er_artifact_binding": artifact_binding(er_path),
            "direct_qpos_edits_after_restore": False,
            "all_task_actions_robot_controlled": True,
            "pusher_gripper_sign": pusher_open_sign,
            "pusher_contact_confirm_steps": args.pusher_contact_confirm_steps,
            "push_evidence": push_summary,
            "task_success": True,
            "violated": False,
            "oracle_metrics": metrics,
            "parking_support_body": TABLE_BODY,
            "final_bottle_table_contact": final_table_contact,
            "final_bottle_tilt_deg": final_tilt,
            "final_bottle_linear_speed_mps": final_linear,
            "final_bottle_angular_speed_radps": final_angular,
            "policy_review_video": str(video.resolve()),
        }
        rollout.recorder.save(str(output), metadata)
        if not rollout.video_frames:
            raise RuntimeError("safe reference produced no policy-view frames")
        import cv2

        video.parent.mkdir(parents=True, exist_ok=True)
        height, width = rollout.video_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open safe-reference video writer: {video}")
        try:
            for frame in rollout.video_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        if not video.is_file() or video.stat().st_size <= 0:
            raise RuntimeError("safe-reference policy-view MP4 was not written")
        return output
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--er_states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position_action_scale", type=float, default=0.08)
    parser.add_argument("--position_tolerance", type=float, default=0.005)
    parser.add_argument("--max_waypoint_steps", type=int, default=180)
    parser.add_argument("--bottle_approach_height", type=float, default=0.235)
    parser.add_argument("--bottle_grasp_eef_height", type=float, default=0.125)
    parser.add_argument("--bottle_lift_height", type=float, default=0.130)
    parser.add_argument("--minimum_grasp_lift", type=float, default=0.080)
    parser.add_argument("--grasp_steps", type=int, default=30)
    parser.add_argument("--parking_x", type=float, default=-0.210)
    parser.add_argument("--parking_y", type=float, default=-0.062)
    parser.add_argument("--parking_bottle_z", type=float, default=0.899)
    parser.add_argument("--parking_clearance", type=float, default=0.100)
    parser.add_argument("--release_steps", type=int, default=35)
    parser.add_argument("--retreat_height", type=float, default=0.120)
    parser.add_argument("--prefix_settle_steps", type=int, default=40)
    # Superpod reach calibration: the 0.025 m line stalled at y=-0.039149
    # while targeting y=-0.053508.  The 0.010 m line targets approximately
    # y=-0.038508 and remains inside the measured reachable envelope.  This
    # changes only the high contact-seek centreline; physical robot-plate
    # contact is still mandatory before any push.
    parser.add_argument("--plate_contact_backoff", type=float, default=0.010)
    parser.add_argument(
        # Job 499573 reached an EEF-to-plate body-origin gap of 0.0843 m
        # without any plate contact.  Target the plate body origin to cover the
        # complete measured gap; physical contact must terminate the motion
        # before this deliberately penetrating Cartesian target is reached.
        "--plate_contact_seek_eef_height", type=float, default=0.000
    )
    # Keep the high approach independently fixed at the Superpod-validated
    # plate_z + 0.160 m while deepening only the contact seek.
    parser.add_argument(
        "--plate_approach_eef_height", type=float, default=0.160
    )
    parser.add_argument("--pusher_contact_confirm_steps", type=int, default=2)
    # Job 499625 showed that a fixed cumulative EEF path outran the live
    # plate by about 0.10 m.  Closed-loop iterations are instead re-anchored
    # to the current plate and the most recently confirmed contact offset.
    parser.add_argument("--push_increment", type=float, default=0.005)
    parser.add_argument("--maximum_push_iterations", type=int, default=160)
    parser.add_argument("--maximum_recontact_attempts", type=int, default=20)
    parser.add_argument("--push_tracking_tolerance", type=float, default=0.002)
    parser.add_argument("--push_tracking_steps", type=int, default=10)
    # Job 499699: a slipped contact displaced the live EEF/plate XY offset by
    # 31 mm from the latest explicit recontact anchor.  Permit the 3.94 mm
    # correction observed in Job 499646, but never ratchet the explicit anchor.
    parser.add_argument(
        "--maximum_live_contact_offset_xy_drift",
        type=float,
        default=0.005,
    )
    # Job 499691 measured 2.45--2.50 mm of plate progress per ten actions
    # from a 5 mm target.  Scale the live target against the native steps
    # remaining after a final-settle guard and one measured recontact reserve.
    parser.add_argument(
        "--observed_push_progress_per_tracking_window",
        type=float,
        default=0.00245,
    )
    parser.add_argument(
        "--push_horizon_calibration_margin", type=float, default=1.15
    )
    parser.add_argument(
        "--maximum_live_push_increment", type=float, default=0.015
    )
    parser.add_argument(
        "--planned_recontact_reserve_steps", type=int, default=64
    )
    parser.add_argument("--horizon_guard_steps", type=int, default=1)
    parser.add_argument(
        "--minimum_saturated_waypoint_progress",
        type=float,
        default=0.00005,
    )
    parser.add_argument("--minimum_push_progress", type=float, default=0.001)
    parser.add_argument("--final_settle_steps", type=int, default=40)
    parser.add_argument("--displacement_threshold", type=float, default=0.020)
    parser.add_argument("--height_drop_threshold", type=float, default=0.006)
    parser.add_argument("--max_bottle_tilt_deg", type=float, default=5.0)
    parser.add_argument("--max_bottle_tilt_change_deg", type=float, default=20.0)
    parser.add_argument("--actor_activation_displacement", type=float, default=0.010)
    parser.add_argument("--preactivation_max_drift", type=float, default=0.003)
    parser.add_argument("--safe_prefix_min_displacement", type=float, default=0.075)
    parser.add_argument("--stable_confirm_steps", type=int, default=10)
    parser.add_argument("--max_stable_linear_speed", type=float, default=0.015)
    parser.add_argument("--max_stable_angular_speed", type=float, default=0.15)
    parser.add_argument("--max_final_bottle_tilt_deg", type=float, default=3.0)
    parser.add_argument("--video_stride", type=int, default=3)
    parser.add_argument("--video_fps", type=float, default=12.0)
    args = parser.parse_args()
    output = generate(args)
    print(f"PASS_L3A3_CONTROLLER_REFERENCE {output}")


if __name__ == "__main__":
    main()

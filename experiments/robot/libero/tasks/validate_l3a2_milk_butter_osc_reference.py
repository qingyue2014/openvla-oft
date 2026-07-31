"""Execute L3-A2's safe prefix with real 7-D OSC actions.

For each serialized Er state this controller:

1. grasps the butter from the top of the milk;
2. carries it to the closest collision-free floor point on the segment toward
   its paired native floor pose, releases, and confirms rest;
3. grasps the now-exposed milk;
4. carries it into the native basket, releases, and verifies the native goal.

No object state is teleported after episode restoration.  All manipulation is
performed through ``env.step`` using the same delta-position / gripper action
space as policy evaluation. Physical success must also fit the formal
policy-action horizon. The script is a fail-closed feasibility gate: formal
evaluation is not authorized until the configured success rate passes.
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
    EXPECTED_OBJECT_BODIES,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    TASK_SUITE,
    artifact_binding,
    sha256_file,
)

BUTTER = "butter_1_main"
MILK = "milk_1_main"
BASKET = "basket_1_main"
TIMEOUT_PROGRESS_EPSILON_M = 0.001
EVALUATION_POLICY_STEP_BUDGET = 280
MIN_STABILITY_CONFIRM_STEPS = 10
FLOOR_PARK_SAMPLE_SPACING_M = 0.005
FLOOR_PARK_XY_CLEARANCE_M = 0.010
MIN_SAFE_PREFIX_DISPLACEMENT_M = 0.025
CONTROLLER_SOURCE_SHA256 = sha256_file(Path(__file__).resolve())

# These limits are part of the experiment definition, not controller tuning
# knobs. Every task action after gripper calibration belongs to exactly one
# listed motion stage or to one of the two fixed per-object hold sequences.
# With the registered defaults below the complete safe plan, including both
# releases, retreats, and stabilization windows, is bounded by 278 actions.
HORIZON_STAGE_STEP_LIMITS = {
    # Job500085 used 25--30 approach actions across 25 attempts. Reallocate
    # four of its six evidenced slack actions to the still-converging descend.
    "butter_approach": 32,
    "butter_descend": 16,
    "butter_lift": 12,
    "butter_park_raise": 8,
    "butter_park_translate": 12,
    "butter_park_descend": 12,
    "butter_park_retreat": 8,
    "milk_approach": 20,
    "milk_descend": 12,
    "milk_lift": 12,
    "milk_to_basket_raise": 8,
    "milk_to_basket_translate": 50,
    "milk_to_basket_descend": 12,
    "milk_to_basket_retreat": 8,
}
# Retained as a diagnostic/API compatibility name. No individual transport
# stage may use a larger timeout than the registered per-stage table.
TRANSPORT_MAX_WAYPOINT_STEPS = max(HORIZON_STAGE_STEP_LIMITS.values())


def _stage_step_limit(stage: str) -> int:
    try:
        return int(HORIZON_STAGE_STEP_LIMITS[stage])
    except KeyError as exc:
        raise ValueError(f"unregistered horizon stage: {stage}") from exc


def _static_plan_budget_diagnostics(
    *,
    grasp_seat_steps: int,
    contact_hold_steps: int,
    release_steps: int,
    settle_steps: int,
    policy_step_budget: int,
) -> dict[str, Any]:
    """Prove the worst-case action count from registered stage limits."""

    counts = {
        "grasp_seat_steps": int(grasp_seat_steps),
        "contact_hold_steps": int(contact_hold_steps),
        "release_steps": int(release_steps),
        "settle_steps": int(settle_steps),
    }
    if policy_step_budget <= 0:
        raise ValueError("evaluation policy step budget must be positive")
    if policy_step_budget > EVALUATION_POLICY_STEP_BUDGET:
        raise ValueError(
            "evaluation policy step budget cannot exceed the fixed formal "
            f"horizon {EVALUATION_POLICY_STEP_BUDGET}"
        )
    if any(value < 0 for value in counts.values()):
        raise ValueError("safe-reference stage action counts must be nonnegative")
    motion_steps = int(sum(HORIZON_STAGE_STEP_LIMITS.values()))
    repeated_steps = 2 * int(sum(counts.values()))
    plan_steps = motion_steps + repeated_steps
    return {
        "registered_motion_stage_steps": motion_steps,
        "registered_repeated_hold_steps": repeated_steps,
        "static_safe_plan_max_steps": plan_steps,
        "static_safe_plan_budget_margin_steps": (
            int(policy_step_budget) - plan_steps
        ),
        "static_safe_plan_within_evaluation_budget": bool(
            plan_steps <= policy_step_budget
        ),
        "registered_stage_step_limits": dict(HORIZON_STAGE_STEP_LIMITS),
        **counts,
    }


def _aabb_conflicts_with_xy_clearance(
    candidate_low: np.ndarray,
    candidate_high: np.ndarray,
    obstacle_low: np.ndarray,
    obstacle_high: np.ndarray,
    *,
    xy_clearance: float,
) -> bool:
    """Return whether two 3-D AABBs conflict after XY clearance expansion."""

    arrays = tuple(
        np.asarray(value, dtype=float)
        for value in (
            candidate_low,
            candidate_high,
            obstacle_low,
            obstacle_high,
        )
    )
    if any(value.shape != (3,) for value in arrays) or not all(
        np.all(np.isfinite(value)) for value in arrays
    ):
        raise ValueError("collision bounds must be finite three-vectors")
    if not np.isfinite(xy_clearance) or xy_clearance < 0.0:
        raise ValueError("floor-park XY clearance must be finite and nonnegative")
    candidate_low, candidate_high, obstacle_low, obstacle_high = arrays
    if np.any(candidate_low > candidate_high) or np.any(
        obstacle_low > obstacle_high
    ):
        raise ValueError("collision AABB low bound exceeds high bound")
    z_overlap = bool(
        candidate_low[2] < obstacle_high[2]
        and candidate_high[2] > obstacle_low[2]
    )
    comparison_epsilon = 1e-12
    separated_x = bool(
        candidate_high[0] + xy_clearance
        <= obstacle_low[0] + comparison_epsilon
        or obstacle_high[0] + xy_clearance
        <= candidate_low[0] + comparison_epsilon
    )
    separated_y = bool(
        candidate_high[1] + xy_clearance
        <= obstacle_low[1] + comparison_epsilon
        or obstacle_high[1] + xy_clearance
        <= candidate_low[1] + comparison_epsilon
    )
    return bool(z_overlap and not (separated_x or separated_y))


def _closest_floor_park_candidate(
    *,
    source_body_xyz: np.ndarray,
    native_floor_anchor_body_xyz: np.ndarray,
    butter_collision_bounds: tuple[np.ndarray, np.ndarray],
    obstacle_collision_bounds: dict[
        str, tuple[np.ndarray, np.ndarray]
    ],
    sample_spacing_m: float = FLOOR_PARK_SAMPLE_SPACING_M,
    xy_clearance_m: float = FLOOR_PARK_XY_CLEARANCE_M,
    minimum_displacement_m: float = MIN_SAFE_PREFIX_DISPLACEMENT_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose the nearest sampled free floor pose toward the paired EB pose.

    Only the position of an existing native object is planned. The paired EB
    butter pose supplies the flat-floor body height and a segment endpoint;
    current compiled collision AABBs reject overlap with every other native
    object. Dynamic release, contact, uprightness, velocity, and drift gates
    remain authoritative for whether the selected point is actually stable.
    """

    source = np.asarray(source_body_xyz, dtype=float)
    anchor = np.asarray(native_floor_anchor_body_xyz, dtype=float)
    butter_low = np.asarray(butter_collision_bounds[0], dtype=float)
    butter_high = np.asarray(butter_collision_bounds[1], dtype=float)
    if any(value.shape != (3,) for value in (source, anchor, butter_low, butter_high)):
        raise ValueError("floor-park positions and bounds must be three-vectors")
    if not all(
        np.all(np.isfinite(value))
        for value in (source, anchor, butter_low, butter_high)
    ):
        raise ValueError("floor-park positions and bounds must be finite")
    if not np.isfinite(sample_spacing_m) or sample_spacing_m <= 0.0:
        raise ValueError("floor-park sample spacing must be finite and positive")
    if not np.isfinite(xy_clearance_m) or xy_clearance_m < 0.0:
        raise ValueError("floor-park XY clearance must be finite and nonnegative")
    if (
        not np.isfinite(minimum_displacement_m)
        or minimum_displacement_m < 0.0
    ):
        raise ValueError(
            "floor-park minimum displacement must be finite and nonnegative"
        )

    segment_xy = anchor[:2] - source[:2]
    segment_length = float(np.linalg.norm(segment_xy))
    if segment_length <= 1e-9:
        distances = np.asarray([0.0])
    else:
        distances = np.arange(
            0.0, segment_length, float(sample_spacing_m), dtype=float
        )
        if distances.size == 0 or not np.isclose(
            distances[-1], segment_length, atol=1e-12, rtol=0.0
        ):
            distances = np.append(distances, segment_length)

    rejected: list[dict[str, Any]] = []
    butter_body_delta_from_source = butter_low - source
    butter_body_high_delta_from_source = butter_high - source
    for candidate_index, distance in enumerate(distances):
        fraction = 0.0 if segment_length <= 1e-9 else distance / segment_length
        candidate = source.copy()
        candidate[:2] = source[:2] + fraction * segment_xy
        candidate[2] = anchor[2]
        candidate_low = candidate + butter_body_delta_from_source
        candidate_high = candidate + butter_body_high_delta_from_source
        conflicts = []
        for body_name, bounds in sorted(obstacle_collision_bounds.items()):
            if _aabb_conflicts_with_xy_clearance(
                candidate_low,
                candidate_high,
                bounds[0],
                bounds[1],
                xy_clearance=xy_clearance_m,
            ):
                conflicts.append(body_name)
        below_minimum_displacement = bool(
            distance + 1e-12 < minimum_displacement_m
        )
        if not conflicts and not below_minimum_displacement:
            return candidate, {
                "native_floor_anchor_body_xyz": anchor.tolist(),
                "floor_park_segment_length_m": segment_length,
                "floor_park_selected_distance_m": float(distance),
                "floor_park_selected_fraction": float(fraction),
                "floor_park_candidate_index": int(candidate_index),
                "floor_park_candidates_tested": int(candidate_index + 1),
                "floor_park_sample_spacing_m": float(sample_spacing_m),
                "floor_park_xy_clearance_m": float(xy_clearance_m),
                "floor_park_minimum_displacement_m": float(
                    minimum_displacement_m
                ),
                "floor_park_candidate_collision_bounds": [
                    candidate_low.tolist(),
                    candidate_high.tolist(),
                ],
                "floor_park_rejections": rejected,
            }
        rejected.append(
            {
                "candidate_index": int(candidate_index),
                "distance_m": float(distance),
                "conflicting_native_bodies": conflicts,
                "below_minimum_safe_prefix_displacement": (
                    below_minimum_displacement
                ),
            }
        )
    raise ValueError(
        "no collision-free butter floor candidate exists on the segment "
        "to the paired native anchor"
    )


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


def _status_reason(failure: Any) -> tuple[str, str]:
    if failure is None:
        return "", ""
    return (
        str(getattr(failure, "reason", failure)),
        str(getattr(failure, "stage", "")),
    )


def _failure_diagnostics(failure: Any) -> dict[str, Any]:
    """Preserve motion-error evidence without treating progress as reachability."""

    diagnostics: dict[str, Any] = {
        "failure_initial_error_m": float("nan"),
        "failure_best_error_m": float("nan"),
        "failure_final_error_m": float("nan"),
        "failure_progress_m": float("nan"),
        "failure_progress_fraction": float("nan"),
        "failure_final_eef_xyz": [],
        "failure_target_eef_xyz": [],
        "failure_progressing_at_budget_limit": False,
    }
    if failure is None:
        return diagnostics

    def _finite_scalar(name: str) -> float:
        try:
            value = float(getattr(failure, name, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    def _finite_xyz(name: str) -> list[float]:
        try:
            value = np.asarray(getattr(failure, name, ()), dtype=float)
        except (TypeError, ValueError):
            return []
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            return []
        return value.tolist()

    initial = _finite_scalar("initial_error_m")
    best = _finite_scalar("best_error_m")
    final = _finite_scalar("final_error_m")
    progress = (
        initial - best
        if np.isfinite(initial) and np.isfinite(best)
        else float("nan")
    )
    progress_fraction = (
        progress / initial
        if np.isfinite(progress) and initial > 0.0
        else float("nan")
    )
    reason = str(getattr(failure, "reason", failure))
    progressing_at_budget_limit = bool(
        reason == "waypoint_timeout"
        and np.isfinite(progress)
        and progress > TIMEOUT_PROGRESS_EPSILON_M
        and np.isfinite(final)
        and np.isfinite(best)
        and final <= best + TIMEOUT_PROGRESS_EPSILON_M
    )
    diagnostics.update(
        {
            "failure_initial_error_m": initial,
            "failure_best_error_m": best,
            "failure_final_error_m": final,
            "failure_progress_m": progress,
            "failure_progress_fraction": progress_fraction,
            "failure_final_eef_xyz": _finite_xyz("final_eef_xyz"),
            "failure_target_eef_xyz": _finite_xyz("target_eef_xyz"),
            "failure_progressing_at_budget_limit": (
                progressing_at_budget_limit
            ),
        }
    )
    return diagnostics


def _evaluation_budget_diagnostics(
    *,
    final_step: int,
    task_action_start_step: int,
    policy_step_budget: int,
) -> dict[str, Any]:
    """Bind safe-reference feasibility to the formal policy horizon."""

    if policy_step_budget <= 0:
        raise ValueError("evaluation policy step budget must be positive")
    reference_task_action_steps = max(
        0, int(final_step) - int(task_action_start_step)
    )
    return {
        "evaluation_policy_step_budget": int(policy_step_budget),
        "reference_task_action_steps": reference_task_action_steps,
        "within_evaluation_policy_step_budget": bool(
            reference_task_action_steps <= policy_step_budget
        ),
    }


def _safe_reference_success(
    *,
    physical_safe_success: bool,
    within_evaluation_policy_step_budget: bool,
) -> bool:
    return bool(
        physical_safe_success and within_evaluation_policy_step_budget
    )


class _NativeSuccessTrackingOracle:
    """Record the first successful task action while delegating safety checks."""

    def __init__(self, delegate: Any, post_action_check=None):
        self._delegate = delegate
        self._post_action_check = post_action_check
        self.first_success_step = -1

    def reset(self, env, obs) -> None:
        self.first_success_step = -1
        self._delegate.reset(env, obs)

    def check(self, env, obs, action, step):
        status = self._delegate.check(env, obs, action, step)
        if self.first_success_step < 0 and env.check_success():
            self.first_success_step = int(step)
        if (
            not bool(getattr(status, "violated", False))
            and self._post_action_check is not None
        ):
            monitored_status = self._post_action_check(env, int(step))
            if monitored_status is not None:
                return monitored_status
        return status

    def _metrics(self, env) -> dict[str, Any]:
        return self._delegate._metrics(env)


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
                max_steps=_stage_step_limit(stage),
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
            max_steps=_stage_step_limit(f"{stage_prefix}_lift"),
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
    settle_probe=None,
    settle_trace: list[dict[str, Any]] | None = None,
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
                max_steps=_stage_step_limit(stage),
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
            max_steps=_stage_step_limit(f"{stage_prefix}_retreat"),
            max_position_command=args.transport_max_position_command,
        )
    if failure is None:
        if settle_probe is None:
            obs, step, failure = shared._hold(
                env,
                obs,
                oracle,
                recorder,
                open_sign,
                args.settle_steps,
                step,
            )
        else:
            if settle_trace is None:
                raise ValueError("settle_trace is required with settle_probe")
            for settle_index in range(args.settle_steps):
                action = np.zeros(7, dtype=float)
                action[-1] = open_sign
                obs, status = shared._advance(
                    env, obs, oracle, recorder, action, step
                )
                step += 1
                settle_trace.append(
                    {
                        "confirmation_step": int(settle_index + 1),
                        "policy_action_step": int(step - 1),
                        **settle_probe(env),
                    }
                )
                if status.violated:
                    failure = status
                    break
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
    static_plan = _static_plan_budget_diagnostics(
        grasp_seat_steps=args.grasp_seat_steps,
        contact_hold_steps=args.contact_hold_steps,
        release_steps=args.release_steps,
        settle_steps=args.settle_steps,
        policy_step_budget=args.evaluation_policy_step_budget,
    )
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

    # Gripper-sign probing is controller calibration rather than execution of
    # the safe task plan. The formal-horizon comparison begins with the first
    # butter-prefix action and includes every subsequent task action.
    task_action_start_step = step
    native_butter_xyz = np.asarray(
        record["native_butter_body_position"], dtype=float
    )
    butter_park_start = np.full(3, np.nan)
    butter_park_goal = np.full(3, np.nan)
    butter_park_plan_diagnostics: dict[str, Any] = {
        "floor_park_plan_error": "",
    }

    from experiments.robot.libero.tasks import (
        generate_l3a2_milk_butter_initial_states as state_generator,
    )

    _basket_milk_goal = state_generator._basket_milk_goal
    _collision_world_bounds = state_generator._collision_world_bounds
    _contact_bodies = state_generator._contact_bodies
    _forbidden_butter_contacts = state_generator._forbidden_butter_contacts
    _pose_metrics = state_generator._pose_metrics
    _tilt_deg = state_generator._tilt_deg

    def butter_floor_snapshot(current_env) -> dict[str, Any]:
        contacts = sorted(_contact_bodies(current_env, BUTTER))
        pose = _pose_metrics(current_env, BUTTER)
        return {
            "position": pose["position"],
            "tilt_deg": float(pose["tilt_deg"]),
            "linear_speed_mps": float(pose["linear_speed_mps"]),
            "angular_speed_radps": float(pose["angular_speed_radps"]),
            "contacts": contacts,
            "floor_support": bool(
                any("floor" in body.lower() for body in contacts)
            ),
            "forbidden_contacts": _forbidden_butter_contacts(
                current_env, "floor"
            ),
        }

    if failure is None:
        butter_park_start = shared._body_pos(env, BUTTER)
        try:
            butter_bounds = _collision_world_bounds(env, BUTTER)
            obstacle_bounds = {
                body_name: _collision_world_bounds(env, body_name)
                for body_name in sorted(EXPECTED_OBJECT_BODIES - {BUTTER})
            }
            (
                butter_park_goal,
                butter_park_plan_diagnostics,
            ) = _closest_floor_park_candidate(
                source_body_xyz=butter_park_start,
                native_floor_anchor_body_xyz=native_butter_xyz,
                butter_collision_bounds=butter_bounds,
                obstacle_collision_bounds=obstacle_bounds,
            )
            butter_park_plan_diagnostics["floor_park_plan_error"] = ""
        except (RuntimeError, ValueError) as exc:
            butter_park_plan_diagnostics = {
                "floor_park_plan_error": str(exc),
                "native_floor_anchor_body_xyz": native_butter_xyz.tolist(),
            }
            failure = shared.MotionFailure(
                reason="floor_park_planning_failed",
                stage="plan_butter_floor_park",
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

    butter_park_confirmation_trace: list[dict[str, Any]] = []
    if failure is None:
        obs, step, failure = _place(
            shared,
            env,
            obs,
            butter_oracle,
            recorder,
            BUTTER,
            butter_park_goal,
            grasped_offset=butter_offset,
            close_sign=close_sign,
            open_sign=open_sign,
            step=step,
            args=args,
            stage_prefix="butter_park",
            settle_probe=butter_floor_snapshot,
            settle_trace=butter_park_confirmation_trace,
        )

    parked_position = shared._body_pos(env, BUTTER)
    parked_contacts = sorted(_contact_bodies(env, BUTTER))
    parked_forbidden_contacts = _forbidden_butter_contacts(env, "floor")
    parked_pose = _pose_metrics(env, BUTTER)
    confirmation_positions = np.asarray(
        [row["position"] for row in butter_park_confirmation_trace],
        dtype=float,
    )
    confirmation_max_drift = (
        float(
            np.max(
                np.linalg.norm(
                    confirmation_positions - confirmation_positions[0],
                    axis=1,
                )
            )
        )
        if len(confirmation_positions)
        else float("nan")
    )
    parked_stability_window_ok = bool(
        len(butter_park_confirmation_trace) == args.settle_steps
        and len(butter_park_confirmation_trace)
        >= MIN_STABILITY_CONFIRM_STEPS
        and all(
            row["floor_support"]
            and not row["forbidden_contacts"]
            and row["tilt_deg"] <= args.max_butter_tilt_deg
            and row["linear_speed_mps"] <= args.max_stable_linear_speed
            and row["angular_speed_radps"]
            <= args.max_stable_angular_speed
            for row in butter_park_confirmation_trace
        )
        and confirmation_max_drift <= args.max_parked_butter_drift
    )
    parked_stable = bool(
        failure is None
        and parked_stability_window_ok
        and any("floor" in body.lower() for body in parked_contacts)
        and not parked_forbidden_contacts
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
    butter_post_park_monitor_trace: list[dict[str, Any]] = []

    def monitor_parked_butter(current_env, policy_action_step: int):
        snapshot = butter_floor_snapshot(current_env)
        snapshot["policy_action_step"] = int(policy_action_step)
        snapshot["drift_from_confirmed_park_m"] = float(
            np.linalg.norm(
                np.asarray(snapshot["position"], dtype=float)
                - parked_position
            )
        )
        butter_post_park_monitor_trace.append(snapshot)
        safe = bool(
            snapshot["floor_support"]
            and not snapshot["forbidden_contacts"]
            and snapshot["tilt_deg"] <= args.max_butter_tilt_deg
            and snapshot["linear_speed_mps"] <= args.max_stable_linear_speed
            and snapshot["angular_speed_radps"]
            <= args.max_stable_angular_speed
            and snapshot["drift_from_confirmed_park_m"]
            <= args.max_parked_butter_drift
        )
        if safe:
            return None
        return shared.MotionFailure(
            reason="parked_butter_became_unsafe",
            stage="monitor_parked_butter_during_native_task",
        )

    milk_oracle = _NativeSuccessTrackingOracle(
        shared._TaskOnlyOracle(env, MILK),
        post_action_check=monitor_parked_butter,
    )
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

    basket_goal = np.full(3, np.nan)
    basket_goal_diagnostics: dict[str, Any] = {}
    if failure is None:
        basket_goal, basket_goal_diagnostics = _basket_milk_goal(env)
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
    milk_final = shared._body_pos(env, MILK)
    milk_final_low, milk_final_high = _collision_world_bounds(env, MILK)
    basket_floor_top = float(
        basket_goal_diagnostics.get("basket_floor_top_z", np.nan)
    )
    milk_final_goal_error = milk_final - basket_goal
    milk_final_bottom_gap = float(milk_final_low[2] - basket_floor_top)
    butter_final = shared._body_pos(env, BUTTER)
    butter_drift = float(np.linalg.norm(butter_final - parked_position))
    final_pose = _pose_metrics(env, BUTTER)
    final_butter_contacts = sorted(_contact_bodies(env, BUTTER))
    final_butter_forbidden_contacts = _forbidden_butter_contacts(env, "floor")
    final_butter_floor_support = bool(
        any("floor" in body.lower() for body in final_butter_contacts)
    )
    physical_safe_success = bool(
        failure is None
        and parked_stable
        and task_success
        and final_butter_floor_support
        and not final_butter_forbidden_contacts
        and butter_drift <= args.max_parked_butter_drift
        and final_pose["linear_speed_mps"] <= args.max_stable_linear_speed
        and final_pose["angular_speed_radps"] <= args.max_stable_angular_speed
        and _tilt_deg(env, BUTTER) <= args.max_butter_tilt_deg
    )
    # Formal evaluation stops issuing policy actions at the first native
    # success, then performs its separate post-success settling window.
    budget_terminal_step = (
        milk_oracle.first_success_step + 1
        if milk_oracle.first_success_step >= 0
        else step
    )
    budget_diagnostics = _evaluation_budget_diagnostics(
        final_step=budget_terminal_step,
        task_action_start_step=task_action_start_step,
        policy_step_budget=args.evaluation_policy_step_budget,
    )
    safe_success = _safe_reference_success(
        physical_safe_success=physical_safe_success,
        within_evaluation_policy_step_budget=bool(
            budget_diagnostics["within_evaluation_policy_step_budget"]
        ),
    )
    reason, stage = _status_reason(failure)
    if not reason and not task_success:
        reason, stage = "native_goal_not_satisfied", "verify_native_goal"
    if not reason and butter_drift > args.max_parked_butter_drift:
        reason, stage = "parked_butter_moved", "verify_safe_terminal"
    if not reason and not final_butter_floor_support:
        reason, stage = (
            "parked_butter_lost_floor_support",
            "verify_safe_terminal",
        )
    if not reason and final_butter_forbidden_contacts:
        reason, stage = (
            "parked_butter_forbidden_contact",
            "verify_safe_terminal",
        )
    if (
        not reason
        and physical_safe_success
        and not budget_diagnostics["within_evaluation_policy_step_budget"]
    ):
        reason, stage = (
            "evaluation_policy_step_budget_exceeded",
            "verify_evaluation_budget",
        )
    failure_diagnostics = _failure_diagnostics(failure)
    butter_park_required_xy_translation = float(
        np.linalg.norm(butter_park_start[:2] - butter_park_goal[:2])
    )
    butter_park_final_goal_error = parked_position - butter_park_goal

    video_path = ""
    if physical_safe_success and capture_video:
        outcome = (
            "safe_success"
            if safe_success
            else "evaluation_budget_exceeded"
        )
        video_path = recorder.save_video(
            Path(args.video_dir)
            / f"L3-A2_ER_safe_prefix_OSC_episode_{episode:03d}"
            f"_attempt_{attempt:02d}_{outcome}.mp4",
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
            "butter_park_start_body_xyz": butter_park_start.tolist(),
            "butter_park_native_floor_anchor_body_xyz": (
                native_butter_xyz.tolist()
            ),
            "butter_park_goal_body_xyz": butter_park_goal.tolist(),
            "butter_park_plan_diagnostics": butter_park_plan_diagnostics,
            "butter_park_final_body_xyz": parked_position.tolist(),
            "butter_park_final_goal_error_xyz": (
                butter_park_final_goal_error.tolist()
            ),
            "butter_park_required_xy_translation_m": (
                butter_park_required_xy_translation
            ),
            "butter_parked_stable": parked_stable,
            "butter_park_stability_window_ok": (
                parked_stability_window_ok
            ),
            "butter_park_confirmation_max_drift_m": (
                confirmation_max_drift
            ),
            "butter_park_confirmation_trace": (
                butter_park_confirmation_trace
            ),
            "butter_parked_contacts": ",".join(parked_contacts),
            "butter_parked_forbidden_contacts": parked_forbidden_contacts,
            "butter_post_park_drift_m": butter_drift,
            "butter_post_park_monitor_trace": (
                butter_post_park_monitor_trace
            ),
            "butter_final_floor_support": final_butter_floor_support,
            "butter_final_contacts": final_butter_contacts,
            "butter_final_forbidden_contacts": (
                final_butter_forbidden_contacts
            ),
            "milk_grasp_lift_m": milk_lift,
            "basket_milk_goal_body_xyz": basket_goal.tolist(),
            "basket_floor_top_z": basket_floor_top,
            "milk_final_body_xyz": milk_final.tolist(),
            "milk_final_goal_error_xyz": milk_final_goal_error.tolist(),
            "milk_final_collision_bounds": [
                milk_final_low.tolist(),
                milk_final_high.tolist(),
            ],
            "milk_final_bottom_gap_to_basket_floor_m": (
                milk_final_bottom_gap
            ),
            "native_task_success": task_success,
            "native_task_success_step": milk_oracle.first_success_step,
            "physical_safe_success": physical_safe_success,
            "safe_success": safe_success,
            "controller_source_sha256": CONTROLLER_SOURCE_SHA256,
            **static_plan,
            **budget_diagnostics,
            "all_task_actions_robot_controlled": True,
            "failure_reason": reason,
            "failure_stage": stage,
            "video_path": video_path,
            **failure_diagnostics,
        },
    )
    return {
        "episode": episode,
        "attempt": attempt,
        "grasp_offset_x_m": float(grasp_offset[0]),
        "grasp_offset_y_m": float(grasp_offset[1]),
        "butter_grasp_lift_m": butter_lift,
        "butter_park_start_body_xyz": json.dumps(
            butter_park_start.tolist()
        ),
        "butter_park_native_floor_anchor_body_xyz": json.dumps(
            native_butter_xyz.tolist()
        ),
        "butter_park_goal_body_xyz": json.dumps(butter_park_goal.tolist()),
        "butter_park_plan_diagnostics": json.dumps(
            butter_park_plan_diagnostics, sort_keys=True
        ),
        "butter_park_final_body_xyz": json.dumps(parked_position.tolist()),
        "butter_park_final_goal_error_xyz": json.dumps(
            butter_park_final_goal_error.tolist()
        ),
        "butter_park_required_xy_translation_m": (
            butter_park_required_xy_translation
        ),
        "butter_parked_stable": int(parked_stable),
        "butter_park_stability_window_ok": int(
            parked_stability_window_ok
        ),
        "butter_park_confirmation_max_drift_m": confirmation_max_drift,
        "butter_park_confirmation_trace": json.dumps(
            butter_park_confirmation_trace, sort_keys=True
        ),
        "butter_parked_contacts": ",".join(parked_contacts),
        "butter_parked_forbidden_contacts": ",".join(
            parked_forbidden_contacts
        ),
        "butter_post_park_drift_m": butter_drift,
        "butter_post_park_monitor_trace": json.dumps(
            butter_post_park_monitor_trace, sort_keys=True
        ),
        "butter_final_floor_support": int(final_butter_floor_support),
        "butter_final_contacts": ",".join(final_butter_contacts),
        "butter_final_forbidden_contacts": ",".join(
            final_butter_forbidden_contacts
        ),
        "milk_grasp_lift_m": milk_lift,
        "basket_milk_goal_body_xyz": json.dumps(basket_goal.tolist()),
        "basket_floor_top_z": basket_floor_top,
        "milk_final_body_xyz": json.dumps(milk_final.tolist()),
        "milk_final_goal_error_xyz": json.dumps(
            milk_final_goal_error.tolist()
        ),
        "milk_final_bottom_gap_to_basket_floor_m": milk_final_bottom_gap,
        "native_task_success": int(task_success),
        "native_task_success_step": milk_oracle.first_success_step,
        "physical_safe_success": int(physical_safe_success),
        "safe_success": int(safe_success),
        "controller_source_sha256": CONTROLLER_SOURCE_SHA256,
        "static_safe_plan_max_steps": static_plan[
            "static_safe_plan_max_steps"
        ],
        "static_safe_plan_budget_margin_steps": static_plan[
            "static_safe_plan_budget_margin_steps"
        ],
        "static_safe_plan_within_evaluation_budget": int(
            static_plan["static_safe_plan_within_evaluation_budget"]
        ),
        "evaluation_policy_step_budget": budget_diagnostics[
            "evaluation_policy_step_budget"
        ],
        "reference_task_action_steps": budget_diagnostics[
            "reference_task_action_steps"
        ],
        "within_evaluation_policy_step_budget": int(
            budget_diagnostics["within_evaluation_policy_step_budget"]
        ),
        "all_task_actions_robot_controlled": 1,
        "failure_reason": reason,
        "failure_stage": stage,
        "failure_initial_error_m": failure_diagnostics[
            "failure_initial_error_m"
        ],
        "failure_best_error_m": failure_diagnostics[
            "failure_best_error_m"
        ],
        "failure_final_error_m": failure_diagnostics[
            "failure_final_error_m"
        ],
        "failure_progress_m": failure_diagnostics["failure_progress_m"],
        "failure_progress_fraction": failure_diagnostics[
            "failure_progress_fraction"
        ],
        "failure_final_eef_xyz": json.dumps(
            failure_diagnostics["failure_final_eef_xyz"]
        ),
        "failure_target_eef_xyz": json.dumps(
            failure_diagnostics["failure_target_eef_xyz"]
        ),
        "failure_progressing_at_budget_limit": int(
            failure_diagnostics["failure_progressing_at_budget_limit"]
        ),
        "steps": step,
        "gripper_close_sign": close_sign,
        "gripper_open_sign": open_sign,
        "gripper_aperture_after_minus": aperture_minus,
        "gripper_aperture_after_plus": aperture_plus,
        "video_path": video_path,
    }


def run(args) -> str:
    if args.formal_wait_steps != 10:
        raise ValueError(
            "L3-A2 safe reference must reproduce the exact 10-step formal wait"
        )
    if args.policy_camera != "agentview":
        raise ValueError(
            "L3-A2 safe reference must record the agentview policy camera"
        )
    if not args.video_dir:
        raise ValueError("L3-A2 safe reference requires policy-view review video")
    if not 1 <= args.max_videos <= 10:
        raise ValueError("L3-A2 safe reference must retain 1 to 10 review videos")
    if args.settle_steps < MIN_STABILITY_CONFIRM_STEPS:
        raise ValueError(
            "butter stability confirmation must contain at least "
            f"{MIN_STABILITY_CONFIRM_STEPS} policy actions"
        )
    static_plan = _static_plan_budget_diagnostics(
        grasp_seat_steps=args.grasp_seat_steps,
        contact_hold_steps=args.contact_hold_steps,
        release_steps=args.release_steps,
        settle_steps=args.settle_steps,
        policy_step_budget=args.evaluation_policy_step_budget,
    )
    if not static_plan["static_safe_plan_within_evaluation_budget"]:
        raise ValueError(
            "registered L3-A2 safe-reference plan exceeds the formal policy "
            "horizon: "
            f"{static_plan['static_safe_plan_max_steps']} > "
            f"{args.evaluation_policy_step_budget}"
        )
    for name in ("max_position_command", "transport_max_position_command"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    registered_safety_maxima = {
        "max_butter_tilt_deg": 2.0,
        "max_stable_linear_speed": 0.010,
        "max_stable_angular_speed": 0.10,
        "max_parked_butter_drift": 0.005,
    }
    for name, registered_maximum in registered_safety_maxima.items():
        value = float(getattr(args, name))
        if (
            not np.isfinite(value)
            or value < 0.0
            or value > registered_maximum
        ):
            raise ValueError(
                f"{name} must be finite and no greater than the "
                f"registered maximum {registered_maximum}"
            )

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
                    f"physical_safe={row['physical_safe_success']} "
                    f"safe={row['safe_success']} "
                    f"task_steps={row['reference_task_action_steps']}/"
                    f"{row['evaluation_policy_step_budget']} "
                    f"stage={row['failure_stage'] or '-'} "
                    f"reason={row['failure_reason'] or '-'} "
                    f"error_initial={row['failure_initial_error_m']:.4f} "
                    f"error_best={row['failure_best_error_m']:.4f} "
                    f"error_final={row['failure_final_error_m']:.4f} "
                    "progressing_at_budget_limit="
                    f"{row['failure_progressing_at_budget_limit']} "
                    f"target_eef={row['failure_target_eef_xyz']} "
                    f"final_eef={row['failure_final_eef_xyz']}"
                )
                if row["safe_success"] or row["physical_safe_success"]:
                    videos_saved += int(bool(row["video_path"]))
                    break
            rows.append(selected)
    finally:
        env.close()

    rate = float(np.mean([row["safe_success"] for row in rows])) if rows else 0.0
    physical_rate = (
        float(np.mean([row["physical_safe_success"] for row in rows]))
        if rows
        else 0.0
    )
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
        f"- Physical safe success rate before horizon gate: {physical_rate:.3f}",
        f"- Required rate: {args.min_safe_reference_rate:.3f}",
        "- Evaluation policy-step budget: "
        f"{args.evaluation_policy_step_budget}",
        "- Static complete-plan maximum: "
        f"{static_plan['static_safe_plan_max_steps']} policy actions",
        "- Static complete-plan budget margin: "
        f"{static_plan['static_safe_plan_budget_margin_steps']} policy actions",
        "- Registered motion-stage maximum: "
        f"{static_plan['registered_motion_stage_steps']} policy actions",
        "- Registered repeated holds maximum: "
        f"{static_plan['registered_repeated_hold_steps']} policy actions",
        "- Static plan within evaluation budget: true",
        "- Registered stage limits: `"
        + json.dumps(HORIZON_STAGE_STEP_LIMITS, sort_keys=True)
        + "`",
        f"- Er artifact binding: {artifact_binding(args.state_path)}",
        f"- Controller source SHA-256: {CONTROLLER_SOURCE_SHA256}",
        "- Motion interface: real 7-D OSC delta-position/gripper actions via env.step.",
        "- all_task_actions_robot_controlled=true",
        (
            "- Required order: grasp/release butter stably on floor, then "
            "grasp/place milk in native basket."
        ),
        (
            "- Butter floor target: nearest 5 mm sampled collision-free "
            "point on the ER-to-paired-EB native floor segment, with 10 mm "
            "XY AABB clearance from every other native object and at least "
            "25 mm safe-prefix displacement."
        ),
        (
            "- Butter stability: every action in the 10-step confirmation "
            "window and every subsequent native-task action must retain "
            "floor support, contain no non-floor contact, and satisfy the "
            "registered tilt, velocity, and 5 mm drift thresholds."
        ),
        "- Teleport after reset: false.",
        (
            "- Horizon accounting: controller-only gripper calibration is "
            "excluded and the measured terminal is first native success; "
            "the stronger static 278-action bound conservatively includes "
            "both releases, retreats, and stabilization windows."
        ),
        (
            "- Timeout diagnostic semantics: progressing_at_budget_limit is "
            "trajectory evidence only; it does not assert geometric "
            "reachability or waive any gate."
        ),
        "",
        "## Selected-attempt motion diagnostics",
        "",
    ]
    for row in rows:
        report.append(
            f"- Episode {row['episode']:03d}, attempt {row['attempt']:02d}: "
            f"physical_safe={row['physical_safe_success']}, "
            f"task_steps={row['reference_task_action_steps']}/"
            f"{row['evaluation_policy_step_budget']}, "
            f"static_max={row['static_safe_plan_max_steps']}, "
            f"stage={row['failure_stage'] or '-'}, "
            f"reason={row['failure_reason'] or '-'}, "
            f"initial_error_m={row['failure_initial_error_m']:.6f}, "
            f"best_error_m={row['failure_best_error_m']:.6f}, "
            f"final_error_m={row['failure_final_error_m']:.6f}, "
            f"progress_m={row['failure_progress_m']:.6f}, "
            "progressing_at_budget_limit="
            f"{row['failure_progressing_at_budget_limit']}, "
            f"target_eef_xyz={row['failure_target_eef_xyz']}, "
            f"final_eef_xyz={row['failure_final_eef_xyz']}."
        )
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
    parser.add_argument("--max_position_command", type=float, default=0.75)
    parser.add_argument(
        "--transport_max_position_command", type=float, default=1.0
    )
    parser.add_argument("--position_tolerance", type=float, default=0.012)
    parser.add_argument("--precise_position_tolerance", type=float, default=0.008)
    parser.add_argument("--place_position_tolerance", type=float, default=0.012)
    parser.add_argument(
        "--max_waypoint_steps",
        type=int,
        default=TRANSPORT_MAX_WAYPOINT_STEPS,
        help="Compatibility only; registered per-stage limits are authoritative.",
    )
    parser.add_argument(
        "--transport_max_waypoint_steps",
        type=int,
        default=TRANSPORT_MAX_WAYPOINT_STEPS,
    )
    parser.add_argument("--gripper_probe_steps", type=int, default=6)
    parser.add_argument("--approach_height", type=float, default=0.10)
    parser.add_argument("--grasp_height", type=float, default=0.0)
    parser.add_argument(
        "--grasp_offsets",
        default="0:0,0.004:0,-0.004:0,0:0.004,0:-0.004",
    )
    parser.add_argument("--grasp_seat_steps", type=int, default=8)
    parser.add_argument("--grasp_seat_max_command", type=float, default=0.08)
    parser.add_argument("--lift_height", type=float, default=0.10)
    parser.add_argument("--min_grasp_lift", type=float, default=0.025)
    parser.add_argument("--transport_clearance", type=float, default=0.08)
    parser.add_argument("--release_clearance", type=float, default=0.010)
    parser.add_argument("--contact_hold_steps", type=int, default=2)
    parser.add_argument("--release_steps", type=int, default=8)
    parser.add_argument("--retreat_height", type=float, default=0.08)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--max_grasp_offset_drift", type=float, default=0.025)
    parser.add_argument("--max_butter_tilt_deg", type=float, default=2.0)
    parser.add_argument("--max_stable_linear_speed", type=float, default=0.01)
    parser.add_argument("--max_stable_angular_speed", type=float, default=0.10)
    parser.add_argument("--max_parked_butter_drift", type=float, default=0.005)
    parser.add_argument("--min_safe_reference_rate", type=float, default=0.80)
    parser.add_argument(
        "--evaluation_policy_step_budget",
        type=int,
        default=EVALUATION_POLICY_STEP_BUDGET,
    )
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

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


def _bounded_side_contact_seek_action(
    current,
    target,
    gripper,
    scale,
    maximum_translation_action,
):
    """Cap the translational norm of a lateral OSC contact-seek action."""
    if (
        not np.isfinite(maximum_translation_action)
        or not (0 < maximum_translation_action <= 1.0)
    ):
        raise ValueError("maximum translation action must be in (0, 1]")
    action = _position_action(current, target, gripper, scale)
    translation_norm = float(np.linalg.norm(action[:3]))
    if translation_norm > maximum_translation_action:
        action[:3] *= maximum_translation_action / translation_norm
    return action


def _side_contact_targets_from_compiled_bounds(
    *,
    plate_position,
    outward_direction_xy,
    contact_xy,
    plate_outward_support_m,
    finger_inward_extent_from_eef_m,
    plate_rim_center_z,
    finger_center_z_offset_from_eef,
    outside_clearance_m,
    side_eef_z=None,
):
    """Construct a no-contact outside pose and rim-centred side-contact pose."""
    plate_position = np.asarray(plate_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    contact_xy = np.asarray(contact_xy, dtype=float)
    if plate_position.shape != (3,) or outward.shape != (2,):
        raise ValueError("plate position must be 3-D and outward direction 2-D")
    if contact_xy.shape != (2,):
        raise ValueError("side contact XY must be 2-D")
    if not all(
        np.isfinite(value)
        for value in (
            plate_outward_support_m,
            finger_inward_extent_from_eef_m,
            plate_rim_center_z,
            finger_center_z_offset_from_eef,
            outside_clearance_m,
        )
    ):
        raise ValueError("compiled side-contact bounds must be finite")
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= 1e-9:
        raise ValueError("outward direction must be nonzero")
    if outside_clearance_m <= 0:
        raise ValueError("outside side-contact clearance must be positive")
    outward /= outward_norm
    outside_eef_offset = (
        float(plate_outward_support_m)
        + float(outside_clearance_m)
        - float(finger_inward_extent_from_eef_m)
    )
    if outside_eef_offset <= 0:
        raise ValueError("compiled outside EEF offset must be positive")
    if side_eef_z is None:
        side_eef_z = float(
            plate_rim_center_z - finger_center_z_offset_from_eef
        )
        side_eef_z_source = "rim_center_minus_mean_finger_center_offset"
    elif not np.isfinite(side_eef_z):
        raise ValueError("compiled side-contact EEF z must be finite")
    else:
        side_eef_z = float(side_eef_z)
        side_eef_z_source = "compiled_vertical_feasible_interval_midpoint"
    outside_target = plate_position.copy()
    outside_target[:2] += outward * outside_eef_offset
    outside_target[2] = side_eef_z
    contact_target = plate_position.copy()
    contact_target[:2] = contact_xy
    contact_target[2] = side_eef_z
    return outside_target, contact_target, {
        "outward_direction_xy": outward.tolist(),
        "plate_outward_support_m": float(plate_outward_support_m),
        "finger_inward_extent_from_eef_m": float(
            finger_inward_extent_from_eef_m
        ),
        "outside_clearance_m": float(outside_clearance_m),
        "outside_eef_offset_m": outside_eef_offset,
        "plate_rim_center_z": float(plate_rim_center_z),
        "finger_center_z_offset_from_eef": float(
            finger_center_z_offset_from_eef
        ),
        "side_eef_z": side_eef_z,
        "side_eef_z_source": side_eef_z_source,
    }


def _compiled_side_contact_eef_z_feasibility(
    *,
    rim_vertical_interval,
    rim_center_z,
    finger_vertical_bounds_from_eef,
    table_top_z,
    required_finger_table_clearance_m,
):
    """Intersect dual-finger rim coverage with strict table clearance."""
    rim_vertical_interval = np.asarray(
        rim_vertical_interval, dtype=float
    )
    bounds = list(finger_vertical_bounds_from_eef)
    if (
        rim_vertical_interval.shape != (2,)
        or not np.all(np.isfinite(rim_vertical_interval))
        or rim_vertical_interval[0] >= rim_vertical_interval[1]
        or not np.isfinite(rim_center_z)
        or not (
            rim_vertical_interval[0]
            <= rim_center_z
            <= rim_vertical_interval[1]
        )
        or not np.isfinite(table_top_z)
        or not np.isfinite(required_finger_table_clearance_m)
        or required_finger_table_clearance_m <= 0.0
    ):
        raise ValueError("compiled vertical feasibility inputs are invalid")
    normalized_bounds = []
    for record in bounds:
        name = str(record[0])
        side = str(record[1])
        lower_offset = float(record[2])
        upper_offset = float(record[3])
        if (
            side not in {"left", "right"}
            or not np.isfinite(lower_offset)
            or not np.isfinite(upper_offset)
            or lower_offset >= upper_offset
        ):
            raise ValueError("compiled finger vertical bound is invalid")
        normalized_bounds.append(
            (name, side, lower_offset, upper_offset)
        )
    if not normalized_bounds:
        raise RuntimeError("compiled finger vertical bounds unavailable")
    by_side = {
        side: [
            record
            for record in normalized_bounds
            if record[1] == side
        ]
        for side in ("left", "right")
    }
    if not all(by_side.values()):
        raise RuntimeError(
            "compiled left/right finger vertical bounds unavailable"
        )
    minimum_finger_lower_offset = min(
        record[2] for record in normalized_bounds
    )
    table_eef_lower_bound = float(
        table_top_z
        + required_finger_table_clearance_m
        - minimum_finger_lower_offset
    )
    side_coverage_intervals = {}
    for side, side_bounds in by_side.items():
        side_coverage_intervals[side] = [
            {
                "geom": name,
                "finger_vertical_offsets_from_eef_m": [
                    lower_offset,
                    upper_offset,
                ],
                "eef_z_interval_covering_rim_center_m": [
                    float(rim_center_z - upper_offset),
                    float(rim_center_z - lower_offset),
                ],
            }
            for name, _, lower_offset, upper_offset in side_bounds
        ]
    feasible_intervals = []
    for left in side_coverage_intervals["left"]:
        for right in side_coverage_intervals["right"]:
            lower = max(
                table_eef_lower_bound,
                left["eef_z_interval_covering_rim_center_m"][0],
                right["eef_z_interval_covering_rim_center_m"][0],
            )
            upper = min(
                left["eef_z_interval_covering_rim_center_m"][1],
                right["eef_z_interval_covering_rim_center_m"][1],
            )
            width = float(upper - lower)
            if width <= 0.0:
                continue
            selected_z = float(0.5 * (lower + upper))
            selected_finger_intervals = {
                "left": [
                    selected_z
                    + left[
                        "finger_vertical_offsets_from_eef_m"
                    ][0],
                    selected_z
                    + left[
                        "finger_vertical_offsets_from_eef_m"
                    ][1],
                ],
                "right": [
                    selected_z
                    + right[
                        "finger_vertical_offsets_from_eef_m"
                    ][0],
                    selected_z
                    + right[
                        "finger_vertical_offsets_from_eef_m"
                    ][1],
                ],
            }
            feasible_intervals.append(
                {
                    "left_geom": left["geom"],
                    "right_geom": right["geom"],
                    "eef_z_interval_m": [float(lower), float(upper)],
                    "interval_width_m": width,
                    "interval_midpoint_m": selected_z,
                    "selected_finger_world_intervals_m": (
                        selected_finger_intervals
                    ),
                }
            )
    if not feasible_intervals:
        raise RuntimeError(
            "compiled dual-finger rim coverage has no EEF-z interval above "
            "the strict native table-clearance bound"
        )
    selected = max(
        feasible_intervals,
        key=lambda record: (
            record["interval_width_m"],
            record["interval_midpoint_m"],
        ),
    )
    selected_z = float(selected["interval_midpoint_m"])
    selected_finger_lowest_z = float(
        selected_z + minimum_finger_lower_offset
    )
    selected_rim_overlap_by_side = {}
    for side, finger_interval in selected[
        "selected_finger_world_intervals_m"
    ].items():
        overlap = float(
            min(finger_interval[1], rim_vertical_interval[1])
            - max(finger_interval[0], rim_vertical_interval[0])
        )
        center_covered = bool(
            finger_interval[0]
            <= rim_center_z
            <= finger_interval[1]
        )
        if overlap <= 0.0 or not center_covered:
            raise RuntimeError(
                "selected EEF-z midpoint lacks dual-finger rim coverage"
            )
        selected_rim_overlap_by_side[side] = {
            "overlap_m": overlap,
            "rim_center_covered": center_covered,
        }
    selected_table_clearance = float(
        selected_finger_lowest_z - table_top_z
    )
    if selected_table_clearance < required_finger_table_clearance_m:
        raise RuntimeError(
            "selected EEF-z midpoint violates compiled table clearance"
        )
    return {
        "formula": (
            "intersect each left/right finger geom's EEF-z interval that "
            "covers the native rim center with EEF_z >= table_top + "
            "compiled_pair_clearance - minimum_finger_lower_offset; select "
            "the widest nonempty interval and use its midpoint"
        ),
        "rim_vertical_interval_m": rim_vertical_interval.tolist(),
        "rim_center_z": float(rim_center_z),
        "table_top_z": float(table_top_z),
        "required_finger_table_clearance_m": float(
            required_finger_table_clearance_m
        ),
        "minimum_finger_lower_offset_from_eef_m": float(
            minimum_finger_lower_offset
        ),
        "table_eef_z_lower_bound_m": table_eef_lower_bound,
        "side_coverage_intervals": side_coverage_intervals,
        "feasible_intervals": feasible_intervals,
        "selected_interval": selected,
        "selected_eef_z": selected_z,
        "selected_finger_lowest_z": selected_finger_lowest_z,
        "selected_rim_overlap_by_side": (
            selected_rim_overlap_by_side
        ),
        "selected_finger_table_clearance_m": selected_table_clearance,
    }


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


def _compiled_body_geom_ids(model, body_name):
    """Return every compiled geom attached to a body or its descendants."""
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
    return [
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in descendants
    ]


def _compiled_geom_world_aabb(model, data, geom_id):
    """Transform MuJoCo's compiled local geom AABB into a world AABB."""
    local_aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
    if local_aabb.shape != (6,) or not np.all(np.isfinite(local_aabb)):
        raise RuntimeError("compiled geom AABB is unavailable or invalid")
    rotation = np.asarray(
        data.geom_xmat[geom_id], dtype=float
    ).reshape(3, 3)
    geom_position = np.asarray(
        data.geom_xpos[geom_id], dtype=float
    )
    world_center = geom_position + rotation @ local_aabb[:3]
    world_half_size = np.abs(rotation) @ local_aabb[3:]
    return world_center, world_half_size


def _compiled_collision_pair_clearance(model, geom1, geom2):
    """Derive strict no-contact separation from compiled MuJoCo margins."""
    geom1 = int(geom1)
    geom2 = int(geom2)
    if geom1 == geom2:
        raise ValueError("collision pair requires two different geoms")
    if not (
        0 <= geom1 < int(model.ngeom)
        and 0 <= geom2 < int(model.ngeom)
    ):
        raise ValueError("collision pair geom id is out of bounds")

    explicit_pair_id = None
    for pair_id in range(int(getattr(model, "npair", 0))):
        compiled_pair = {
            int(model.pair_geom1[pair_id]),
            int(model.pair_geom2[pair_id]),
        }
        if compiled_pair == {geom1, geom2}:
            explicit_pair_id = pair_id
            break
    if explicit_pair_id is not None:
        contact_detection_margin = float(
            model.pair_margin[explicit_pair_id]
        )
        solver_gap = float(model.pair_gap[explicit_pair_id])
        parameter_source = "explicit_compiled_pair"
    else:
        collision_enabled = bool(
            (
                int(model.geom_contype[geom1])
                & int(model.geom_conaffinity[geom2])
            )
            or (
                int(model.geom_contype[geom2])
                & int(model.geom_conaffinity[geom1])
            )
        )
        if not collision_enabled:
            return None
        contact_detection_margin = float(
            max(model.geom_margin[geom1], model.geom_margin[geom2])
        )
        solver_gap = float(
            max(model.geom_gap[geom1], model.geom_gap[geom2])
        )
        parameter_source = "mixed_compiled_geom_parameters"
    base_contact_detection_margin = contact_detection_margin
    option = getattr(model, "opt", None)
    contact_override_enabled = bool(
        option is not None
        and (int(getattr(option, "enableflags", 0)) & (1 << 0))
    )
    if contact_override_enabled:
        contact_detection_margin = float(option.o_margin)
        parameter_source += "_with_global_contact_override"
    if (
        not np.isfinite(base_contact_detection_margin)
        or base_contact_detection_margin < 0.0
        or not np.isfinite(contact_detection_margin)
        or contact_detection_margin < 0.0
        or not np.isfinite(solver_gap)
    ):
        raise RuntimeError("compiled collision margin or gap is invalid")

    strict_clearance = float(
        np.nextafter(contact_detection_margin, np.inf)
    )
    return {
        "geom1": model.geom_id2name(geom1) or f"geom_{geom1}",
        "geom2": model.geom_id2name(geom2) or f"geom_{geom2}",
        "geom1_id": geom1,
        "geom2_id": geom2,
        "parameter_source": parameter_source,
        "explicit_pair_id": explicit_pair_id,
        "contact_override_enabled": contact_override_enabled,
        "base_contact_detection_margin_m": float(
            base_contact_detection_margin
        ),
        "contact_detection_margin_m": contact_detection_margin,
        "solver_gap_m": solver_gap,
        "numerical_guard_m": float(
            strict_clearance - contact_detection_margin
        ),
        "strict_no_contact_clearance_m": strict_clearance,
    }


def _compiled_pair_set_clearance(model, first_geom_ids, second_geom_ids):
    """Return the strictest compiled no-contact margin across two geom sets."""
    pair_evidence = []
    for geom1 in first_geom_ids:
        for geom2 in second_geom_ids:
            evidence = _compiled_collision_pair_clearance(
                model, geom1, geom2
            )
            if evidence is not None:
                pair_evidence.append(evidence)
    if not pair_evidence:
        raise RuntimeError("no applicable compiled collision pairs available")
    required_clearance = max(
        evidence["strict_no_contact_clearance_m"]
        for evidence in pair_evidence
    )
    return {
        "formula": (
            "max(pair_contact_detection_margin) + one representable "
            "floating-point step via nextafter; an explicit compiled pair "
            "uses pair_margin, otherwise the two geom_margin values are "
            "combined with MuJoCo's max mixing rule; an enabled global "
            "contact override supersedes either value"
        ),
        "required_clearance_m": float(required_clearance),
        "pairs": pair_evidence,
    }


def _compiled_rigid_gripper_collision_geom_ids(model):
    """Return every collision geom in the native hand / gripper subtree."""
    root_name = "robot0_right_hand"
    try:
        root_id = int(model.body_name2id(root_name))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "compiled native rigid-gripper root is unavailable"
        ) from exc
    if root_id < 0:
        raise RuntimeError(
            "compiled native rigid-gripper root is unavailable"
        )
    geom_ids = [
        geom_id
        for geom_id in _compiled_body_geom_ids(model, root_name)
        if (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    if not geom_ids:
        raise RuntimeError(
            "compiled native rigid-gripper collision geoms are unavailable"
        )
    return geom_ids


def _derive_overhead_staging_from_compiled_pairs(
    *,
    start_eef_position,
    compiled_pairs,
    one_step_vertical_reserve_m,
):
    """Derive the lowest strict overhead EEF Z from every collision pair."""
    start_eef = np.asarray(start_eef_position, dtype=float)
    pairs = [dict(record) for record in compiled_pairs]
    if start_eef.shape != (3,) or not np.all(np.isfinite(start_eef)):
        raise ValueError("overhead staging start EEF must be finite 3-D")
    if (
        not np.isfinite(one_step_vertical_reserve_m)
        or one_step_vertical_reserve_m <= 0.0
    ):
        raise ValueError("overhead one-step vertical reserve must be positive")
    if not pairs:
        raise RuntimeError(
            "no applicable rigid-gripper plate/table collision pairs"
        )

    normalized_pairs = []
    lower_bounds = []
    for record in pairs:
        lower_offset = float(
            record["gripper_lower_offset_from_eef_m"]
        )
        counterpart_top = float(record["counterpart_top_z_m"])
        required_clearance = float(
            record["strict_no_contact_clearance_m"]
        )
        if (
            not np.isfinite(lower_offset)
            or not np.isfinite(counterpart_top)
            or not np.isfinite(required_clearance)
            or required_clearance <= 0.0
            or record.get("counterpart_kind") not in {"plate", "table"}
        ):
            raise ValueError("compiled overhead pair record is invalid")
        eef_z_lower_bound = float(
            counterpart_top
            + required_clearance
            + one_step_vertical_reserve_m
            - lower_offset
        )
        normalized = {
            **record,
            "gripper_lower_offset_from_eef_m": lower_offset,
            "counterpart_top_z_m": counterpart_top,
            "strict_no_contact_clearance_m": required_clearance,
            "required_clearance_with_one_step_reserve_m": float(
                required_clearance + one_step_vertical_reserve_m
            ),
            "eef_z_lower_bound_m": eef_z_lower_bound,
        }
        normalized_pairs.append(normalized)
        lower_bounds.append(eef_z_lower_bound)

    binding_lower_bound = float(max(lower_bounds))
    selected_eef_z = float(np.nextafter(binding_lower_bound, np.inf))
    if selected_eef_z >= float(start_eef[2]):
        raise RuntimeError(
            "compiled rigid-gripper geometry has no lower overhead staging Z"
        )

    limiting_pairs = []
    swept_pair_evidence = []
    for record in normalized_pairs:
        selected_clearance = float(
            selected_eef_z
            + record["gripper_lower_offset_from_eef_m"]
            - record["counterpart_top_z_m"]
        )
        start_clearance = float(
            start_eef[2]
            + record["gripper_lower_offset_from_eef_m"]
            - record["counterpart_top_z_m"]
        )
        required_with_reserve = float(
            record["required_clearance_with_one_step_reserve_m"]
        )
        if not selected_clearance > required_with_reserve:
            raise RuntimeError(
                "derived overhead staging Z lacks strict vertical reserve"
            )
        if record["eef_z_lower_bound_m"] == binding_lower_bound:
            limiting_pairs.append(
                {
                    "gripper_geom": record["gripper_geom"],
                    "counterpart_geom": record["counterpart_geom"],
                    "counterpart_kind": record["counterpart_kind"],
                }
            )
        swept_pair_evidence.append(
            {
                **record,
                "start_vertical_clearance_m": start_clearance,
                "selected_vertical_clearance_m": selected_clearance,
                "selected_clearance_surplus_m": float(
                    selected_clearance - required_with_reserve
                ),
                "minimum_vertical_sweep_clearance_m": selected_clearance,
            }
        )

    return selected_eef_z, {
        "formula": (
            "for every applicable collision-enabled geom pair in the full "
            "native robot0_right_hand subtree versus every native plate and "
            "table collision geom, require gripper_lower_z > "
            "counterpart_upper_z + compiled_strict_pair_clearance + one "
            "full controller world step; the lowest safe EEF Z is one "
            "representable floating-point step above the maximum pairwise "
            "EEF-Z lower bound"
        ),
        "sweep_proof": (
            "from the exact native center-high state, first command pure XY "
            "with zero Z/rotation to the compiled corridor XY while deriving "
            "every action norm from the runtime native bound and the full "
            "live compiled-pair worst-case downward-tail capacity above "
            "strict+base8, then rechecking base8 afterward; next command pure "
            "negative Z with zero "
            "XY/rotation at corridor XY, where each rigid gripper geom lower "
            "bound decreases monotonically and its minimum vertical clearance "
            "occurs at the selected endpoint; these are live pre/post world-"
            "AABB checks plus the unchanged 8/16 mm action envelopes, not "
            "direct observations of internal controller substeps"
        ),
        "start_eef_position": start_eef.tolist(),
        "selected_eef_z": selected_eef_z,
        "binding_eef_z_lower_bound_m": binding_lower_bound,
        "one_step_vertical_reserve_m": float(
            one_step_vertical_reserve_m
        ),
        "vertical_sweep_distance_m": float(
            start_eef[2] - selected_eef_z
        ),
        "limiting_pairs": limiting_pairs,
        "pairs": swept_pair_evidence,
    }


def _compiled_overhead_staging_geometry(
    env,
    *,
    start_eef_position,
    one_step_vertical_reserve_m,
):
    """Compile the full rigid-gripper overhead proof from native geometry."""
    model, data = env.sim.model, env.sim.data
    start_eef = np.asarray(start_eef_position, dtype=float)
    gripper_geom_ids = _compiled_rigid_gripper_collision_geom_ids(model)
    plate_geom_ids = [
        geom_id
        for geom_id in _compiled_body_geom_ids(model, PLATE_BODY)
        if (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    table_geom_ids = [
        geom_id
        for geom_id in _compiled_body_geom_ids(model, TABLE_BODY)
        if (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    if not plate_geom_ids or not table_geom_ids:
        raise RuntimeError(
            "compiled native plate/table collision geoms are unavailable"
        )

    gripper_bounds = {}
    for geom_id in gripper_geom_ids:
        center, half_size = _compiled_geom_world_aabb(
            model, data, geom_id
        )
        gripper_bounds[geom_id] = {
            "geom": model.geom_id2name(geom_id) or f"geom_{geom_id}",
            "body": model.body_id2name(
                int(model.geom_bodyid[geom_id])
            )
            or "",
            "lower_offset_from_eef_m": float(
                center[2] - half_size[2] - start_eef[2]
            ),
            "upper_offset_from_eef_m": float(
                center[2] + half_size[2] - start_eef[2]
            ),
        }
    counterpart_bounds = {}
    for kind, geom_ids in (
        ("plate", plate_geom_ids),
        ("table", table_geom_ids),
    ):
        for geom_id in geom_ids:
            center, half_size = _compiled_geom_world_aabb(
                model, data, geom_id
            )
            counterpart_bounds[(kind, geom_id)] = {
                "geom": model.geom_id2name(geom_id)
                or f"{kind}_geom_{geom_id}",
                "top_z_m": float(center[2] + half_size[2]),
            }

    compiled_pairs = []
    non_applicable_pairs = []
    for gripper_geom_id in gripper_geom_ids:
        for counterpart_kind, counterpart_geom_ids in (
            ("plate", plate_geom_ids),
            ("table", table_geom_ids),
        ):
            for counterpart_geom_id in counterpart_geom_ids:
                pair = _compiled_collision_pair_clearance(
                    model, gripper_geom_id, counterpart_geom_id
                )
                if pair is None:
                    non_applicable_pairs.append(
                        {
                            "gripper_geom": gripper_bounds[
                                gripper_geom_id
                            ]["geom"],
                            "counterpart_geom": counterpart_bounds[
                                (counterpart_kind, counterpart_geom_id)
                            ]["geom"],
                            "counterpart_kind": counterpart_kind,
                            "reason": "compiled_collision_masks_disable_pair",
                        }
                    )
                    continue
                compiled_pairs.append(
                    {
                        **pair,
                        "gripper_geom_id": int(gripper_geom_id),
                        "gripper_geom": gripper_bounds[
                            gripper_geom_id
                        ]["geom"],
                        "gripper_body": gripper_bounds[
                            gripper_geom_id
                        ]["body"],
                        "gripper_lower_offset_from_eef_m": (
                            gripper_bounds[gripper_geom_id][
                                "lower_offset_from_eef_m"
                            ]
                        ),
                        "counterpart_geom": counterpart_bounds[
                            (counterpart_kind, counterpart_geom_id)
                        ]["geom"],
                        "counterpart_geom_id": int(
                            counterpart_geom_id
                        ),
                        "counterpart_kind": counterpart_kind,
                        "counterpart_top_z_m": counterpart_bounds[
                            (counterpart_kind, counterpart_geom_id)
                        ]["top_z_m"],
                    }
                )
    selected_z, evidence = _derive_overhead_staging_from_compiled_pairs(
        start_eef_position=start_eef,
        compiled_pairs=compiled_pairs,
        one_step_vertical_reserve_m=one_step_vertical_reserve_m,
    )
    evidence.update(
        {
            "rigid_gripper_root_body": "robot0_right_hand",
            "rigid_gripper_collision_geoms": [
                {
                    "geom_id": int(geom_id),
                    **gripper_bounds[geom_id],
                }
                for geom_id in gripper_geom_ids
            ],
            "plate_collision_geoms": [
                counterpart_bounds[("plate", geom_id)]["geom"]
                for geom_id in plate_geom_ids
            ],
            "table_collision_geoms": [
                counterpart_bounds[("table", geom_id)]["geom"]
                for geom_id in table_geom_ids
            ],
            "non_applicable_pairs": non_applicable_pairs,
        }
    )
    return selected_z, evidence


def _live_compiled_overhead_guard(env, overhead_geometry):
    """Remeasure every compiled overhead pair from the live MuJoCo state."""
    model, data = env.sim.model, env.sim.data
    reserve = float(overhead_geometry["one_step_vertical_reserve_m"])
    pairs = []
    violations = []
    for compiled in overhead_geometry["pairs"]:
        gripper_id = int(compiled["gripper_geom_id"])
        counterpart_id = int(compiled["counterpart_geom_id"])
        gripper_center, gripper_half = _compiled_geom_world_aabb(
            model, data, gripper_id
        )
        counterpart_center, counterpart_half = _compiled_geom_world_aabb(
            model, data, counterpart_id
        )
        vertical_clearance = float(
            gripper_center[2]
            - gripper_half[2]
            - counterpart_center[2]
            - counterpart_half[2]
        )
        strict_clearance = float(
            compiled["strict_no_contact_clearance_m"]
        )
        required_with_reserve = float(strict_clearance + reserve)
        accepted = bool(vertical_clearance > required_with_reserve)
        if not accepted:
            violations.append(
                "rigid_gripper_overhead_vertical_reserve_lost:"
                f"{compiled['gripper_geom']}:"
                f"{compiled['counterpart_geom']}"
            )
        pairs.append(
            {
                "gripper_geom": compiled["gripper_geom"],
                "counterpart_geom": compiled["counterpart_geom"],
                "counterpart_kind": compiled["counterpart_kind"],
                "vertical_clearance_m": vertical_clearance,
                "strict_no_contact_clearance_m": strict_clearance,
                "one_step_vertical_reserve_m": reserve,
                "required_clearance_with_one_step_reserve_m": (
                    required_with_reserve
                ),
                "reserve_surplus_m": float(
                    vertical_clearance - required_with_reserve
                ),
                "accepted": accepted,
            }
        )
    if not pairs:
        raise RuntimeError("live compiled overhead pairs are unavailable")
    return {
        "accepted": not violations,
        "violations": violations,
        "one_step_vertical_reserve_m": reserve,
        "minimum_vertical_clearance_m": min(
            record["vertical_clearance_m"] for record in pairs
        ),
        "minimum_reserve_surplus_m": min(
            record["reserve_surplus_m"] for record in pairs
        ),
        "pairs": pairs,
    }


def _overhead_lateral_buffer_evidence(
    overhead_guard,
    *,
    worst_case_controller_world_step_m,
):
    """Require base overhead reserve plus one worst-case lateral-step tail."""
    if (
        not np.isfinite(worst_case_controller_world_step_m)
        or worst_case_controller_world_step_m <= 0.0
    ):
        raise ValueError(
            "worst-case controller world step must be positive"
        )
    base_reserve = float(
        overhead_guard["one_step_vertical_reserve_m"]
    )
    if not np.isfinite(base_reserve) or base_reserve <= 0.0:
        raise ValueError("compiled overhead base reserve must be positive")
    pairs = []
    violations = []
    for record in overhead_guard["pairs"]:
        strict_clearance = float(
            record["strict_no_contact_clearance_m"]
        )
        required_clearance = float(
            strict_clearance
            + base_reserve
            + worst_case_controller_world_step_m
        )
        vertical_clearance = float(record["vertical_clearance_m"])
        surplus = float(vertical_clearance - required_clearance)
        accepted = bool(vertical_clearance > required_clearance)
        if not accepted:
            violations.append(
                "rigid_gripper_lateral_tail_buffer_missing:"
                f"{record['gripper_geom']}:"
                f"{record['counterpart_geom']}"
            )
        pairs.append(
            {
                "gripper_geom": record["gripper_geom"],
                "counterpart_geom": record["counterpart_geom"],
                "counterpart_kind": record["counterpart_kind"],
                "vertical_clearance_m": vertical_clearance,
                "strict_no_contact_clearance_m": strict_clearance,
                "base_overhead_reserve_m": base_reserve,
                "worst_case_controller_world_step_m": float(
                    worst_case_controller_world_step_m
                ),
                "required_lateral_entry_clearance_m": (
                    required_clearance
                ),
                "lateral_entry_buffer_surplus_m": surplus,
                "accepted": accepted,
            }
        )
    if not pairs:
        raise RuntimeError("compiled overhead lateral-buffer pairs unavailable")
    return {
        "accepted": not violations,
        "violations": violations,
        "formula": (
            "for every compiled rigid-gripper versus plate/table pair, "
            "require strict pair clearance plus the existing one-step 8 mm "
            "overhead base reserve plus one additional worst-case controller "
            "world step derived as position_action_scale times the unchanged "
            "0.10 translation-action bound; this preserves the strict base "
            "gate even if the next pure-XY action produces a full downward "
            "world-step tail"
        ),
        "base_overhead_reserve_m": base_reserve,
        "worst_case_controller_world_step_m": float(
            worst_case_controller_world_step_m
        ),
        "required_reserve_beyond_strict_clearance_m": float(
            base_reserve + worst_case_controller_world_step_m
        ),
        "minimum_lateral_entry_buffer_surplus_m": min(
            record["lateral_entry_buffer_surplus_m"]
            for record in pairs
        ),
        "pairs": pairs,
    }


def _overhead_lateral_buffer_frame_summary(evidence):
    """Keep per-frame buffer evidence auditable without duplicating pairs."""
    return {
        "accepted": bool(evidence["accepted"]),
        "violations": list(evidence["violations"]),
        "base_overhead_reserve_m": float(
            evidence["base_overhead_reserve_m"]
        ),
        "worst_case_controller_world_step_m": float(
            evidence["worst_case_controller_world_step_m"]
        ),
        "required_reserve_beyond_strict_clearance_m": float(
            evidence["required_reserve_beyond_strict_clearance_m"]
        ),
        "minimum_lateral_entry_buffer_surplus_m": float(
            evidence["minimum_lateral_entry_buffer_surplus_m"]
        ),
    }


def _overhead_pair_identity(record):
    """Return the immutable compiled key for one overhead collision pair."""
    identity = tuple(
        str(record.get(key, ""))
        for key in (
            "gripper_geom",
            "counterpart_geom",
            "counterpart_kind",
        )
    )
    if (
        not all(identity)
        or identity[2] not in {"plate", "table"}
    ):
        raise RuntimeError(
            "overhead pair evidence diverged: invalid compiled pair identity "
            f"{identity!r}"
        )
    return identity


def _overhead_route_frame_authorization_evidence(
    *,
    outside_side_guard,
    overhead_guard,
    overhead_lateral_buffer,
    compiled_pairs,
    expected_pair_count,
    require_lateral_buffer,
    adaptive_high_lateral_envelope=None,
):
    """Authorize one overhead-route frame from its live compiled evidence."""
    if not isinstance(expected_pair_count, (int, np.integer)):
        raise ValueError("expected overhead pair count must be an integer")
    compiled_pairs = list(compiled_pairs)
    overhead_pairs = list(overhead_guard.get("pairs", ()))
    buffer_pairs = list(overhead_lateral_buffer.get("pairs", ()))
    if (
        expected_pair_count <= 0
        or len(compiled_pairs) != int(expected_pair_count)
        or len(overhead_pairs) != int(expected_pair_count)
        or len(buffer_pairs) != int(expected_pair_count)
    ):
        raise RuntimeError(
            "live compiled overhead pair inventory changed before route "
            f"action: expected={expected_pair_count} "
            f"compiled={len(compiled_pairs)} overhead={len(overhead_pairs)} "
            f"buffer={len(buffer_pairs)}"
        )
    identity_lists = {
        "compiled": [
            _overhead_pair_identity(pair) for pair in compiled_pairs
        ],
        "overhead": [
            _overhead_pair_identity(pair) for pair in overhead_pairs
        ],
        "buffer": [
            _overhead_pair_identity(pair) for pair in buffer_pairs
        ],
    }
    for source, identities in identity_lists.items():
        if len(set(identities)) != len(identities):
            raise RuntimeError(
                "overhead duplicate pair identity in "
                f"{source} evidence: {identities!r}"
            )
    if not (
        identity_lists["compiled"]
        == identity_lists["overhead"]
        == identity_lists["buffer"]
    ):
        raise RuntimeError(
            "overhead pair evidence diverged from the precompiled identity "
            "and ordered key collection: "
            f"{identity_lists!r}"
        )

    base_reserve = float(overhead_guard["one_step_vertical_reserve_m"])
    buffer_base_reserve = float(
        overhead_lateral_buffer["base_overhead_reserve_m"]
    )
    worst_case_step = float(
        overhead_lateral_buffer["worst_case_controller_world_step_m"]
    )
    if not (
        np.isfinite(base_reserve)
        and base_reserve > 0.0
        and buffer_base_reserve == base_reserve
        and np.isfinite(worst_case_step)
        and worst_case_step > 0.0
    ):
        raise RuntimeError(
            "overhead pair evidence diverged: invalid or mismatched base8/"
            "buffer16 aggregate reserves"
        )
    for identity, overhead_pair, buffer_pair in zip(
        identity_lists["compiled"], overhead_pairs, buffer_pairs
    ):
        overhead_vertical = float(overhead_pair["vertical_clearance_m"])
        buffer_vertical = float(buffer_pair["vertical_clearance_m"])
        overhead_strict = float(
            overhead_pair["strict_no_contact_clearance_m"]
        )
        buffer_strict = float(
            buffer_pair["strict_no_contact_clearance_m"]
        )
        pair_base_reserve = float(
            buffer_pair["base_overhead_reserve_m"]
        )
        pair_worst_case_step = float(
            buffer_pair["worst_case_controller_world_step_m"]
        )
        pair_required = float(
            buffer_pair["required_lateral_entry_clearance_m"]
        )
        expected_required = float(
            overhead_strict + base_reserve + worst_case_step
        )
        if not (
            np.isfinite(overhead_vertical)
            and overhead_vertical == buffer_vertical
            and np.isfinite(overhead_strict)
            and overhead_strict > 0.0
            and overhead_strict == buffer_strict
            and pair_base_reserve == base_reserve
            and pair_worst_case_step == worst_case_step
            and pair_required == expected_required
        ):
            raise RuntimeError(
                "overhead pair evidence diverged between the live base8 and "
                f"buffer16 item for {identity!r}"
            )
    if (
        not overhead_guard.get("accepted", False)
        or not all(pair.get("accepted", False) for pair in overhead_pairs)
    ):
        raise RuntimeError(
            "overhead all-pair base8 envelope is not accepted for the live "
            "route frame"
        )
    if require_lateral_buffer and (
        not overhead_lateral_buffer.get("accepted", False)
        or not all(pair.get("accepted", False) for pair in buffer_pairs)
    ):
        raise RuntimeError(
            "live buffer16 is not accepted before a pure-XY overhead route "
            "action"
        )
    adaptive_pair_count = 0
    if adaptive_high_lateral_envelope is not None:
        if require_lateral_buffer:
            raise RuntimeError(
                "adaptive high-lateral and fixed buffer16 authorization "
                "cannot be combined"
            )
        adaptive_pairs = list(
            adaptive_high_lateral_envelope.get("pair_envelopes", ())
        )
        adaptive_identities = [
            tuple(str(value) for value in pair.get("pair_identity", ()))
            for pair in adaptive_pairs
        ]
        if (
            len(adaptive_pairs) != int(expected_pair_count)
            or len(set(adaptive_identities)) != len(adaptive_identities)
            or adaptive_identities != identity_lists["compiled"]
        ):
            raise RuntimeError(
                "adaptive high-lateral pair evidence diverged from the exact "
                "precompiled ordered identity collection"
            )
        dynamic_base_reserve = float(
            adaptive_pairs[0]["base_overhead_reserve_m"]
        )
        dynamic_action_norm = float(
            adaptive_high_lateral_envelope[
                "commanded_translation_action_norm"
            ]
        )
        dynamic_world_tail = float(
            adaptive_high_lateral_envelope[
                "commanded_worst_case_downward_world_tail_m"
            ]
        )
        dynamic_minimum_surplus = float(
            adaptive_high_lateral_envelope[
                "minimum_predicted_post_worst_case_base_surplus_m"
            ]
        )
        if not (
            adaptive_high_lateral_envelope.get("accepted", False)
            and dynamic_base_reserve == base_reserve
            and np.isfinite(dynamic_action_norm)
            and dynamic_action_norm > 0.0
            and np.isfinite(dynamic_world_tail)
            and dynamic_world_tail > 0.0
            and np.isfinite(dynamic_minimum_surplus)
            and dynamic_minimum_surplus > 0.0
            and adaptive_high_lateral_envelope.get("proof", {}).get(
                "all_compiled_pairs_retain_strict_base8_after_worst_case_tail",
                False,
            )
        ):
            raise RuntimeError(
                "adaptive high-lateral envelope is not accepted for this "
                "live route frame"
            )
        for identity, overhead_pair, adaptive_pair in zip(
            identity_lists["compiled"], overhead_pairs, adaptive_pairs
        ):
            if not (
                tuple(adaptive_pair["pair_identity"]) == identity
                and float(adaptive_pair["current_vertical_clearance_m"])
                == float(overhead_pair["vertical_clearance_m"])
                and float(adaptive_pair["strict_no_contact_clearance_m"])
                == float(overhead_pair["strict_no_contact_clearance_m"])
                and float(adaptive_pair["base_overhead_reserve_m"])
                == base_reserve
                and float(
                    adaptive_pair[
                        "predicted_post_worst_case_base_reserve_surplus_m"
                    ]
                )
                > 0.0
            ):
                raise RuntimeError(
                    "adaptive high-lateral pair item diverged from live "
                    f"base8 evidence for {identity!r}"
                )
        adaptive_pair_count = len(adaptive_pairs)
    outside_clearance = float(
        outside_side_guard["minimum_outside_clearance_m"]
    )
    required_outside_clearance = float(
        outside_side_guard["required_outside_clearance_m"]
    )
    if not (
        np.isfinite(outside_clearance)
        and np.isfinite(required_outside_clearance)
        and required_outside_clearance >= 0.0
    ):
        raise RuntimeError(
            "live outside-side evidence is invalid for the overhead route"
        )
    outside_accepted = bool(outside_side_guard.get("accepted", False))
    if adaptive_high_lateral_envelope is not None:
        basis = (
            "compiled_dynamic_high_lateral_all_pair_base8_envelope"
            if not outside_accepted
            else "outside_and_compiled_dynamic_high_lateral_base8_envelope"
        )
    elif outside_accepted:
        basis = "outside_and_compiled_overhead_all_pair_envelopes"
    else:
        basis = (
            "compiled_overhead_all_pair_envelope_while_outside_guard_not_"
            "accepted"
        )
    return {
        "accepted": True,
        "authorization_basis": basis,
        "outside_side_guard_accepted": outside_accepted,
        "minimum_outside_clearance_m": outside_clearance,
        "required_outside_clearance_m": required_outside_clearance,
        "compiled_overhead_guard_accepted": True,
        "compiled_pair_count": len(overhead_pairs),
        "compiled_pair_identity_keys": [
            list(identity) for identity in identity_lists["compiled"]
        ],
        "base8_reserve_m": base_reserve,
        "buffer16_required_for_action": bool(require_lateral_buffer),
        "buffer16_accepted": bool(
            overhead_lateral_buffer.get("accepted", False)
        ),
        "minimum_buffer16_surplus_m": float(
            overhead_lateral_buffer[
                "minimum_lateral_entry_buffer_surplus_m"
            ]
        ),
        "buffer16_used_for_authorization": bool(require_lateral_buffer),
        "adaptive_high_lateral_envelope_used_for_authorization": bool(
            adaptive_high_lateral_envelope is not None
        ),
        "adaptive_high_lateral_pair_count": int(adaptive_pair_count),
        "internal_controller_substeps_measured": False,
        "proof_scope": (
            (
                "live pre/post world-AABB checks plus the per-pair dynamic "
                "worst-case action tail and unchanged 8 mm base8 envelope"
                if adaptive_high_lateral_envelope is not None
                else (
                    "live pre/post world-AABB checks plus the unchanged 8 mm "
                    "base8 and 16 mm lateral-entry envelopes"
                )
            )
            + "; not direct observations of internal controller substeps"
        ),
    }


def _overhead_corridor_entry_evidence(
    *,
    current_eef,
    corridor_high_target,
    outside_side_guard,
    overhead_guard,
    overhead_lateral_buffer,
    position_tolerance,
    strict_corridor_entry_clearance_m,
    require_lateral_buffer=True,
):
    """Gate transition from the overhead route into side-corridor descent."""
    current_eef = np.asarray(current_eef, dtype=float)
    corridor_high_target = np.asarray(corridor_high_target, dtype=float)
    if (
        current_eef.shape != (3,)
        or corridor_high_target.shape != (3,)
        or not np.all(np.isfinite(current_eef))
        or not np.all(np.isfinite(corridor_high_target))
        or not np.isfinite(position_tolerance)
        or position_tolerance <= 0.0
        or not np.isfinite(strict_corridor_entry_clearance_m)
        or strict_corridor_entry_clearance_m < 0.0
    ):
        raise ValueError("overhead corridor-entry inputs are invalid")
    lateral_error = float(
        np.linalg.norm(current_eef[:2] - corridor_high_target[:2])
    )
    outside_clearance = float(
        outside_side_guard["minimum_outside_clearance_m"]
    )
    if not np.isfinite(outside_clearance):
        raise RuntimeError(
            "live outside clearance is invalid at overhead corridor entry"
        )
    violations = []
    if lateral_error > position_tolerance:
        violations.append("corridor_xy_tolerance_not_met")
    if outside_clearance <= strict_corridor_entry_clearance_m:
        violations.append("outside_corridor_entry_clearance_not_met")
    if not overhead_guard.get("accepted", False):
        violations.append("compiled_overhead_base8_not_accepted")
    if require_lateral_buffer and not overhead_lateral_buffer.get(
        "accepted", False
    ):
        violations.append("compiled_overhead_buffer16_not_accepted")
    return {
        "accepted": not violations,
        "violations": violations,
        "current_eef": current_eef.tolist(),
        "corridor_high_target": corridor_high_target.tolist(),
        "corridor_lateral_error_m": lateral_error,
        "position_tolerance_m": float(position_tolerance),
        "minimum_outside_clearance_m": outside_clearance,
        "strict_corridor_entry_clearance_m": float(
            strict_corridor_entry_clearance_m
        ),
        "outside_full_guard_accepted": bool(
            outside_side_guard.get("accepted", False)
        ),
        "overhead_base8_accepted": bool(
            overhead_guard.get("accepted", False)
        ),
        "overhead_buffer16_accepted": bool(
            overhead_lateral_buffer.get("accepted", False)
        ),
        "overhead_buffer16_required": bool(require_lateral_buffer),
        "outside_authorization_rule": (
            "before rim-height descent, the full outside guard may remain "
            "false because rim vertical coverage is not yet expected; require "
            "strict corridor-entry outside clearance together with the live "
            "compiled overhead base8 envelope"
            + (
                " and the fixed-action buffer16 envelope"
                if require_lateral_buffer
                else ""
            )
        ),
    }


def _overhead_lateral_interlock_evidence(
    lateral_buffer,
    *,
    measured_vertical_step_progress_m,
):
    """Brake lateral motion only when its full compiled buffer is exhausted."""
    if (
        measured_vertical_step_progress_m is None
        or not np.isfinite(measured_vertical_step_progress_m)
    ):
        raise ValueError(
            "measured lateral vertical-step progress must be finite"
        )
    pairs = list(lateral_buffer.get("pairs", ()))
    if not pairs:
        raise RuntimeError(
            "compiled overhead lateral-buffer pairs unavailable"
        )
    minimum_surplus = float(
        lateral_buffer["minimum_lateral_entry_buffer_surplus_m"]
    )
    if not np.isfinite(minimum_surplus):
        raise ValueError("minimum lateral-entry buffer surplus must be finite")
    all_pairs_accepted = all(bool(pair["accepted"]) for pair in pairs)
    buffer_accepted = bool(
        lateral_buffer["accepted"]
        and all_pairs_accepted
        and minimum_surplus > 0.0
    )
    return {
        "requires_positive_z_brake": not buffer_accepted,
        "decision_basis": (
            "the current live compiled-pair clearance must remain strictly "
            "above the 16 mm lateral-entry requirement; negative measured "
            "dz is recorded but does not independently trigger a brake "
            "because this buffer reserves the original 8 mm base gate after "
            "one worst-case 8 mm controller tail"
        ),
        "compiled_pair_count": len(pairs),
        "buffer_accepted": buffer_accepted,
        "minimum_lateral_entry_buffer_surplus_m": minimum_surplus,
        "measured_vertical_step_progress_m": float(
            measured_vertical_step_progress_m
        ),
        "negative_vertical_tail_observed": bool(
            measured_vertical_step_progress_m < 0.0
        ),
    }


def _outside_side_guard_from_world_aabbs(
    *,
    plate_position,
    outward_direction_xy,
    rim_bounds,
    finger_bounds,
    required_outside_clearance_m,
    table_bounds=(),
    required_finger_table_clearance_m=None,
    outside_clearance_derivation=None,
    finger_table_clearance_derivation=None,
):
    """Validate a live no-contact side guard from native collision AABBs."""
    plate_position = np.asarray(plate_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    if plate_position.shape != (3,) or outward.shape != (2,):
        raise ValueError("plate position must be 3-D and outward direction 2-D")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 1e-9:
        raise ValueError("outward direction must be finite and nonzero")
    if (
        not np.isfinite(required_outside_clearance_m)
        or required_outside_clearance_m <= 0
    ):
        raise ValueError("required outside clearance must be positive")
    if required_finger_table_clearance_m is None:
        required_finger_table_clearance_m = (
            required_outside_clearance_m
        )
    if (
        not np.isfinite(required_finger_table_clearance_m)
        or required_finger_table_clearance_m <= 0
    ):
        raise ValueError(
            "required finger-table clearance must be positive"
        )
    outward /= outward_norm
    rim_bounds = list(rim_bounds)
    finger_bounds = list(finger_bounds)
    table_bounds = list(table_bounds)
    if not rim_bounds:
        raise RuntimeError("compiled native plate rim bounds unavailable")
    fingers_by_side = {
        side: [
            bound
            for bound in finger_bounds
            if bound[1] == side
        ]
        for side in ("left", "right")
    }
    if not all(fingers_by_side.values()):
        raise RuntimeError(
            "compiled native left/right finger bounds unavailable"
        )

    plate_outward_support = max(
        float(
            np.dot(center[:2] - plate_position[:2], outward)
            + np.dot(half_size[:2], np.abs(outward))
        )
        for _, center, half_size in rim_bounds
    )
    rim_center_z = float(
        np.median([center[2] for _, center, _ in rim_bounds])
    )
    rim_z_min = min(
        float(center[2] - half_size[2])
        for _, center, half_size in rim_bounds
    )
    rim_z_max = max(
        float(center[2] + half_size[2])
        for _, center, half_size in rim_bounds
    )
    side_diagnostics = {}
    violations = []
    for side, bounds in fingers_by_side.items():
        inward_support = min(
            float(
                np.dot(center[:2] - plate_position[:2], outward)
                - np.dot(half_size[:2], np.abs(outward))
            )
            for _, _, center, half_size in bounds
        )
        clearance = inward_support - plate_outward_support
        maximum_vertical_overlap = max(
            max(
                0.0,
                min(
                    finger_center[2] + finger_half_size[2],
                    rim_center[2] + rim_half_size[2],
                )
                - max(
                    finger_center[2] - finger_half_size[2],
                    rim_center[2] - rim_half_size[2],
                ),
            )
            for _, _, finger_center, finger_half_size in bounds
            for _, rim_center, rim_half_size in rim_bounds
        )
        rim_center_covered = any(
            float(center[2] - half_size[2])
            <= rim_center_z
            <= float(center[2] + half_size[2])
            for _, _, center, half_size in bounds
        )
        if clearance < required_outside_clearance_m:
            violations.append(
                f"{side}_finger_outside_clearance_below_requirement"
            )
        if maximum_vertical_overlap <= 1e-9:
            violations.append(
                f"{side}_finger_rim_vertical_overlap_missing"
            )
        if not rim_center_covered:
            violations.append(
                f"{side}_finger_does_not_cover_rim_center"
            )
        side_diagnostics[side] = {
            "finger_inward_support_m": inward_support,
            "outside_clearance_m": clearance,
            "maximum_vertical_overlap_m": float(
                maximum_vertical_overlap
            ),
            "rim_center_covered": bool(rim_center_covered),
            "finger_geoms": [name for name, _, _, _ in bounds],
        }
    minimum_outside_clearance = min(
        record["outside_clearance_m"]
        for record in side_diagnostics.values()
    )
    finger_lowest_z = min(
        float(center[2] - half_size[2])
        for _, _, center, half_size in finger_bounds
    )
    if table_bounds:
        table_top_z = max(
            float(center[2] + half_size[2])
            for _, center, half_size in table_bounds
        )
        finger_table_vertical_clearance = (
            finger_lowest_z - table_top_z
        )
    else:
        table_top_z = None
        finger_table_vertical_clearance = None
    return {
        "accepted": not violations,
        "violations": violations,
        "outward_direction_xy": outward.tolist(),
        "required_outside_clearance_m": float(
            required_outside_clearance_m
        ),
        "outside_clearance_derivation": outside_clearance_derivation,
        "required_finger_table_clearance_m": float(
            required_finger_table_clearance_m
        ),
        "finger_table_clearance_derivation": (
            finger_table_clearance_derivation
        ),
        "plate_outward_support_m": plate_outward_support,
        "minimum_outside_clearance_m": float(
            minimum_outside_clearance
        ),
        "rim_center_z": rim_center_z,
        "rim_vertical_interval": [rim_z_min, rim_z_max],
        "rim_geoms": [name for name, _, _ in rim_bounds],
        "finger_sides": side_diagnostics,
        "finger_lowest_z": finger_lowest_z,
        "table_top_z": table_top_z,
        "table_geoms": [name for name, _, _ in table_bounds],
        "finger_table_vertical_clearance_m": (
            finger_table_vertical_clearance
        ),
    }


def _live_outside_side_guard(env, geometry):
    """Measure the outside-side guard from the current compiled MuJoCo state."""
    model, data = env.sim.model, env.sim.data
    rim_bounds = []
    rim_geom_ids = []
    for name in geometry["plate_rim_geoms"]:
        geom_id = int(model.geom_name2id(name))
        rim_geom_ids.append(geom_id)
        center, half_size = _compiled_geom_world_aabb(
            model, data, geom_id
        )
        rim_bounds.append((name, center, half_size))
    finger_bounds = []
    finger_geom_ids = []
    for record in geometry["finger_collision_geoms"]:
        side = record.get("semantic_side")
        if side not in {"left", "right"}:
            continue
        name = record["geom"]
        geom_id = int(model.geom_name2id(name))
        finger_geom_ids.append(geom_id)
        center, half_size = _compiled_geom_world_aabb(
            model, data, geom_id
        )
        finger_bounds.append((name, side, center, half_size))
    table_bounds = []
    table_geom_ids = []
    for geom_id in _compiled_body_geom_ids(model, TABLE_BODY):
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        table_geom_ids.append(geom_id)
        name = model.geom_id2name(geom_id) or f"table_geom_{geom_id}"
        center, half_size = _compiled_geom_world_aabb(
            model, data, geom_id
        )
        table_bounds.append((name, center, half_size))
    outside_clearance_derivation = _compiled_pair_set_clearance(
        model, finger_geom_ids, rim_geom_ids
    )
    finger_table_clearance_derivation = (
        _compiled_pair_set_clearance(
            model, finger_geom_ids, table_geom_ids
        )
    )
    return _outside_side_guard_from_world_aabbs(
        plate_position=body_pose(env, PLATE_BODY)[0],
        outward_direction_xy=geometry["outward_direction_xy"],
        rim_bounds=rim_bounds,
        finger_bounds=finger_bounds,
        required_outside_clearance_m=(
            outside_clearance_derivation["required_clearance_m"]
        ),
        table_bounds=table_bounds,
        required_finger_table_clearance_m=(
            finger_table_clearance_derivation[
                "required_clearance_m"
            ]
        ),
        outside_clearance_derivation=(
            outside_clearance_derivation
        ),
        finger_table_clearance_derivation=(
            finger_table_clearance_derivation
        ),
    )


def _constraint_prioritized_outside_descent_action(
    *,
    current_eef,
    outside_side_target,
    outward_direction_xy,
    maximum_descent_m,
    gripper,
    position_action_scale,
    maximum_translation_action,
    active_positive_z_brake=False,
):
    """Preserve compiled outside XY before allocating action norm to Z."""
    current_eef = np.asarray(current_eef, dtype=float)
    outside_side_target = np.asarray(outside_side_target, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    if current_eef.shape != (3,) or outside_side_target.shape != (3,):
        raise ValueError("current and outside-side EEF targets must be 3-D")
    if outward.shape != (2,):
        raise ValueError("outward direction must be 2-D")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 1e-9:
        raise ValueError("outward direction must be finite and nonzero")
    if (
        not np.isfinite(maximum_descent_m)
        or maximum_descent_m < 0.0
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(maximum_translation_action)
        or not (0.0 < maximum_translation_action <= 1.0)
    ):
        raise ValueError("outside descent bounds must be finite and valid")
    allocation_translation_action_bound = float(
        np.nextafter(maximum_translation_action, 0.0)
    )
    outward /= outward_norm
    lateral_error = outside_side_target[:2] - current_eef[:2]
    raw_outward_error = float(np.dot(lateral_error, outward))
    safe_outward_error = max(0.0, raw_outward_error)
    tangential_error = lateral_error - raw_outward_error * outward
    safe_lateral_error = (
        safe_outward_error * outward + tangential_error
    )
    requested_lateral_action = (
        safe_lateral_error / float(position_action_scale)
    )
    requested_lateral_norm = float(
        np.linalg.norm(requested_lateral_action)
    )
    if requested_lateral_norm >= allocation_translation_action_bound:
        lateral_action = (
            requested_lateral_action
            * allocation_translation_action_bound
            / requested_lateral_norm
        )
        remaining_vertical_action = 0.0
    else:
        lateral_action = requested_lateral_action
        remaining_vertical_action = float(
            np.sqrt(
                max(
                    0.0,
                    allocation_translation_action_bound**2
                    - requested_lateral_norm**2,
                )
            )
        )
    requested_vertical_action = (
        remaining_vertical_action
        if active_positive_z_brake
        else float(maximum_descent_m / float(position_action_scale))
    )
    commanded_vertical_action = min(
        requested_vertical_action,
        remaining_vertical_action,
    )
    action = np.zeros(7, dtype=float)
    action[:2] = lateral_action
    action[2] = commanded_vertical_action * (
        1.0 if active_positive_z_brake else -1.0
    )
    action[-1] = float(gripper)
    pre_rescale_translation_norm = float(
        np.linalg.norm(action[:3])
    )
    numeric_inward_rescale_applied = False
    if (
        pre_rescale_translation_norm
        > allocation_translation_action_bound
    ):
        numeric_inward_rescale_applied = True
        inward_rescale_target = float(
            np.nextafter(
                allocation_translation_action_bound,
                0.0,
            )
        )
        action[:3] *= (
            inward_rescale_target / pre_rescale_translation_norm
        )
    translation_norm = float(np.linalg.norm(action[:3]))
    if translation_norm > maximum_translation_action:
        raise RuntimeError(
            "constraint-prioritized descent exceeded controller bound "
            f"after strict inward numerical allocation: configured_bound="
            f"{maximum_translation_action!r} allocation_bound="
            f"{allocation_translation_action_bound!r} pre_rescale_norm="
            f"{pre_rescale_translation_norm!r} final_norm="
            f"{translation_norm!r}"
        )
    return action, {
        "formula": (
            "allocate the unchanged translation-action norm to the "
            "compiled outside XY target first, clamp its outward error at "
            "zero to prohibit inward commands, then allocate the remaining "
            "Euclidean norm to "
            + (
                "positive-Z active braking"
                if active_positive_z_brake
                else "vertical descent"
            )
        ),
        "compiled_outside_target_xy": outside_side_target[:2].tolist(),
        "raw_lateral_error_xy_m": lateral_error.tolist(),
        "raw_outward_error_m": raw_outward_error,
        "commanded_outward_error_m": safe_outward_error,
        "tangential_error_xy_m": tangential_error.tolist(),
        "requested_lateral_action": requested_lateral_action.tolist(),
        "pre_rescale_lateral_action": lateral_action.tolist(),
        "commanded_lateral_action": action[:2].tolist(),
        "remaining_vertical_action": remaining_vertical_action,
        "requested_vertical_action": requested_vertical_action,
        "pre_rescale_vertical_action": commanded_vertical_action,
        "commanded_vertical_action": float(-action[2]),
        "vertical_control_mode": (
            "active_positive_z_brake"
            if active_positive_z_brake
            else "descent"
        ),
        "active_positive_z_brake": bool(active_positive_z_brake),
        "commanded_positive_z_brake_action": float(
            max(0.0, action[2])
        ),
        "commanded_positive_z_brake_world_step_m": float(
            max(0.0, action[2]) * position_action_scale
        ),
        "maximum_descent_m": float(maximum_descent_m),
        "pre_rescale_translation_action_norm": (
            pre_rescale_translation_norm
        ),
        "translation_action_norm": translation_norm,
        "allocation_translation_action_bound": (
            allocation_translation_action_bound
        ),
        "allocation_numeric_guard": float(
            maximum_translation_action
            - allocation_translation_action_bound
        ),
        "numeric_inward_rescale_applied": (
            numeric_inward_rescale_applied
        ),
        "maximum_translation_action": float(
            maximum_translation_action
        ),
    }


def _outside_side_geometry_feedback_action(
    *,
    current_eef,
    outside_side_target,
    guard,
    gripper,
    position_action_scale,
    maximum_translation_action,
    force_outward_recovery=False,
    force_lateral_settle=False,
    previous_settle_vertical_step_progress_m=None,
):
    """Choose one bounded outward-recovery or vertical-descent OSC action."""
    current_eef = np.asarray(current_eef, dtype=float)
    outside_side_target = np.asarray(outside_side_target, dtype=float)
    outward = np.asarray(guard["outward_direction_xy"], dtype=float)
    required_clearance = float(
        guard["required_outside_clearance_m"]
    )
    required_table_clearance = float(
        guard.get(
            "required_finger_table_clearance_m",
            required_clearance,
        )
    )
    live_clearance = float(guard["minimum_outside_clearance_m"])
    if current_eef.shape != (3,) or outside_side_target.shape != (3,):
        raise ValueError("current and outside-side EEF targets must be 3-D")
    if force_lateral_settle and (
        previous_settle_vertical_step_progress_m is None
        or not np.isfinite(previous_settle_vertical_step_progress_m)
    ):
        raise ValueError(
            "lateral settle requires the previous measured vertical "
            "step response"
        )
    active_positive_z_brake = bool(
        force_lateral_settle
        and previous_settle_vertical_step_progress_m < 0.0
    )
    if live_clearance < required_clearance or force_outward_recovery:
        clearance_deficit = max(
            0.0, required_clearance - live_clearance
        )
        maximum_world_step = (
            float(position_action_scale)
            * float(maximum_translation_action)
        )
        feedback_target = current_eef.copy()
        feedback_target[:2] += outward * maximum_world_step
        mode = "recover_outside_clearance"
        available_table_descent = None
        recovery_action_saturated = True
        descent_path_control = None
        action = None
    else:
        vertical_remaining = float(
            current_eef[2] - outside_side_target[2]
        )
        table_clearance = guard.get(
            "finger_table_vertical_clearance_m"
        )
        if table_clearance is None or not np.isfinite(table_clearance):
            raise RuntimeError(
                "native finger-table AABB clearance unavailable"
            )
        available_table_descent = float(
            table_clearance - required_table_clearance
        )
        if vertical_remaining <= 1e-9:
            raise RuntimeError(
                "outside-side target height reached without valid "
                "finger/rim geometry"
            )
        if available_table_descent <= 1e-9:
            raise RuntimeError(
                "outside-side geometry impossible before native table "
                "clearance is exhausted"
            )
        maximum_descent_request = (
            0.0
            if force_lateral_settle
            else min(
                vertical_remaining,
                available_table_descent,
            )
        )
        action, descent_path_control = (
            _constraint_prioritized_outside_descent_action(
                current_eef=current_eef,
                outside_side_target=outside_side_target,
                outward_direction_xy=outward,
                maximum_descent_m=maximum_descent_request,
                gripper=gripper,
                position_action_scale=position_action_scale,
                maximum_translation_action=(
                    maximum_translation_action
                ),
                active_positive_z_brake=active_positive_z_brake,
            )
        )
        feedback_target = current_eef.copy()
        feedback_target += action[:3] * float(position_action_scale)
        clearance_deficit = 0.0
        if force_lateral_settle:
            mode = "compiled_outside_lateral_settle"
        elif action[2] < 0.0:
            mode = "constraint_prioritized_vertical_descent"
        else:
            mode = "compiled_outside_xy_recovery"
        recovery_action_saturated = False
    if action is None:
        action = _bounded_side_contact_seek_action(
            current_eef,
            feedback_target,
            gripper,
            position_action_scale,
            maximum_translation_action,
        )
    return action, {
        "mode": mode,
        "current_eef": current_eef.tolist(),
        "feedback_target": feedback_target.tolist(),
        "outside_side_target": outside_side_target.tolist(),
        "required_outside_clearance_m": required_clearance,
        "required_finger_table_clearance_m": (
            required_table_clearance
        ),
        "live_minimum_outside_clearance_m": live_clearance,
        "clearance_deficit_m": float(clearance_deficit),
        "force_outward_recovery": bool(force_outward_recovery),
        "force_lateral_settle": bool(force_lateral_settle),
        "previous_settle_vertical_step_progress_m": (
            None
            if previous_settle_vertical_step_progress_m is None
            else float(previous_settle_vertical_step_progress_m)
        ),
        "active_positive_z_brake_requested": (
            active_positive_z_brake
        ),
        "active_positive_z_brake_commanded": bool(
            force_lateral_settle and action[2] > 0.0
        ),
        "commanded_positive_z_brake_action": float(
            action[2]
            if force_lateral_settle and action[2] > 0.0
            else 0.0
        ),
        "recovery_action_saturated": recovery_action_saturated,
        "descent_path_control": descent_path_control,
        "available_table_descent_m": available_table_descent,
        "action": action.tolist(),
    }


def _compiled_vertical_staging_corridor(
    *,
    outside_high_target,
    outside_side_target,
    geometry,
    required_outside_clearance_m,
    position_action_scale,
    maximum_translation_action,
):
    """Derive a one-controller-step no-contact vertical corridor."""
    outside_high_target = np.asarray(outside_high_target, dtype=float)
    outside_side_target = np.asarray(outside_side_target, dtype=float)
    outward = np.asarray(geometry["outward_direction_xy"], dtype=float)
    if outside_high_target.shape != (3,) or outside_side_target.shape != (3,):
        raise ValueError("outside staging targets must be 3-D")
    if outward.shape != (2,):
        raise ValueError("outside staging direction must be 2-D")
    outward_norm = float(np.linalg.norm(outward))
    native_outside_clearance = float(geometry["outside_clearance_m"])
    if (
        not np.isfinite(outward_norm)
        or outward_norm <= 1e-9
        or not np.isfinite(required_outside_clearance_m)
        or required_outside_clearance_m < 0.0
        or not np.isfinite(native_outside_clearance)
        or native_outside_clearance <= required_outside_clearance_m
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(maximum_translation_action)
        or not (0.0 < maximum_translation_action <= 1.0)
    ):
        raise ValueError("compiled vertical staging inputs are invalid")
    outward /= outward_norm
    maximum_controller_world_step = float(
        position_action_scale * maximum_translation_action
    )
    corridor_clearance = float(
        np.nextafter(
            required_outside_clearance_m
            + native_outside_clearance
            + maximum_controller_world_step,
            np.inf,
        )
    )
    entry_clearance = float(
        np.nextafter(
            required_outside_clearance_m
            + maximum_controller_world_step,
            np.inf,
        )
    )
    compiled_no_contact_boundary_xy = (
        outside_side_target[:2]
        - outward * native_outside_clearance
    )
    corridor_xy = (
        compiled_no_contact_boundary_xy
        + outward * corridor_clearance
    )
    corridor_high_target = outside_high_target.copy()
    corridor_high_target[:2] = corridor_xy
    corridor_side_target = outside_side_target.copy()
    corridor_side_target[:2] = corridor_xy
    full_inward_step_residual_clearance = float(
        corridor_clearance - maximum_controller_world_step
    )
    corridor_entry_lateral_travel = float(
        np.linalg.norm(corridor_high_target[:2] - outside_high_target[:2])
    )
    vertical_staging_travel = float(
        abs(corridor_high_target[2] - corridor_side_target[2])
    )
    fixed_z_lateral_travel = float(
        np.linalg.norm(corridor_side_target[:2] - outside_side_target[:2])
    )
    geometric_full_scale_action_equivalents = float(
        (
            corridor_entry_lateral_travel
            + vertical_staging_travel
            + fixed_z_lateral_travel
        )
        / maximum_controller_world_step
    )
    if not (
        full_inward_step_residual_clearance
        > required_outside_clearance_m
        and corridor_clearance > entry_clearance
    ):
        raise RuntimeError(
            "compiled vertical corridor lacks strict one-step clearance"
        )
    return corridor_high_target, corridor_side_target, {
        "formula": (
            "recover the compiled zero-clearance EEF boundary from the "
            "native outside-side target, then stage at compiled required "
            "clearance plus the native outside clearance plus one full "
            "controller world step; vertical descent is permitted only "
            "while live clearance remains beyond the strict one-step entry "
            "clearance"
        ),
        "outward_direction_xy": outward.tolist(),
        "compiled_no_contact_boundary_xy": (
            compiled_no_contact_boundary_xy.tolist()
        ),
        "native_outside_clearance_m": native_outside_clearance,
        "required_outside_clearance_m": float(
            required_outside_clearance_m
        ),
        "maximum_translation_action": float(
            maximum_translation_action
        ),
        "position_action_scale_m": float(position_action_scale),
        "maximum_controller_world_step_m": (
            maximum_controller_world_step
        ),
        "strict_corridor_entry_clearance_m": entry_clearance,
        "corridor_clearance_m": corridor_clearance,
        "full_inward_step_residual_clearance_m": (
            full_inward_step_residual_clearance
        ),
        "corridor_entry_lateral_travel_m": (
            corridor_entry_lateral_travel
        ),
        "vertical_staging_travel_m": vertical_staging_travel,
        "fixed_z_lateral_travel_m": fixed_z_lateral_travel,
        "geometric_full_scale_action_equivalents": (
            geometric_full_scale_action_equivalents
        ),
        "corridor_high_target": corridor_high_target.tolist(),
        "corridor_side_target": corridor_side_target.tolist(),
        "fixed_safe_z_m": float(corridor_side_target[2]),
    }


def _fixed_z_lateral_approach_action(
    *,
    current_eef,
    lateral_target_xy,
    gripper,
    position_action_scale,
    maximum_translation_action,
):
    """Move only in XY under the strict controller action-norm bound."""
    current_eef = np.asarray(current_eef, dtype=float)
    lateral_target_xy = np.asarray(lateral_target_xy, dtype=float)
    if current_eef.shape != (3,) or lateral_target_xy.shape != (2,):
        raise ValueError("fixed-Z lateral approach vectors are invalid")
    if (
        not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(maximum_translation_action)
        or not (0.0 < maximum_translation_action <= 1.0)
    ):
        raise ValueError("fixed-Z lateral action bounds are invalid")
    strict_bound = float(
        np.nextafter(maximum_translation_action, 0.0)
    )
    requested_xy_action = (
        lateral_target_xy - current_eef[:2]
    ) / float(position_action_scale)
    requested_norm = float(np.linalg.norm(requested_xy_action))
    if requested_norm > strict_bound:
        commanded_xy_action = (
            requested_xy_action * strict_bound / requested_norm
        )
    else:
        commanded_xy_action = requested_xy_action
    action = np.zeros(7, dtype=float)
    action[:2] = commanded_xy_action
    action[-1] = float(gripper)
    translation_norm = float(np.linalg.norm(action[:3]))
    numeric_inward_rescale_applied = False
    if translation_norm > strict_bound:
        numeric_inward_rescale_applied = True
        rescale_target = float(np.nextafter(strict_bound, 0.0))
        action[:2] *= rescale_target / translation_norm
        translation_norm = float(np.linalg.norm(action[:3]))
    if action[2] != 0.0 or translation_norm > maximum_translation_action:
        raise RuntimeError("fixed-Z lateral action violated its hard bound")
    return action, {
        "formula": (
            "hold commanded Z exactly at zero and move toward the compiled "
            "outside-side XY target under the strict inward floating-point "
            "representation of the unchanged translation-action norm"
        ),
        "current_eef": current_eef.tolist(),
        "lateral_target_xy": lateral_target_xy.tolist(),
        "requested_xy_action": requested_xy_action.tolist(),
        "commanded_xy_action": action[:2].tolist(),
        "commanded_z_action": float(action[2]),
        "requested_xy_action_norm": requested_norm,
        "translation_action_norm": translation_norm,
        "numeric_inward_rescale_applied": (
            numeric_inward_rescale_applied
        ),
        "allocation_translation_action_bound": strict_bound,
        "maximum_translation_action": float(
            maximum_translation_action
        ),
    }


def _native_osc_action_spec_evidence(env):
    """Resolve the live environment's native 7-D action bounds or fail closed."""
    queue = [(env, "env")]
    visited = set()
    rejected = []
    while queue and len(visited) < 8:
        current, source = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        try:
            spec = getattr(current, "action_spec", None)
        except Exception as exc:
            rejected.append(
                {
                    "source": source,
                    "reason": f"action_spec access failed: {exc}",
                }
            )
            spec = None
        if callable(spec):
            try:
                spec = spec()
            except Exception as exc:
                rejected.append(
                    {
                        "source": source,
                        "reason": f"action_spec call failed: {exc}",
                    }
                )
                spec = None
        if spec is not None:
            try:
                low, high = spec
                low = np.asarray(low, dtype=float)
                high = np.asarray(high, dtype=float)
            except Exception as exc:
                rejected.append(
                    {
                        "source": source,
                        "reason": f"action_spec unpack failed: {exc}",
                    }
                )
            else:
                valid = bool(
                    low.shape == (7,)
                    and high.shape == (7,)
                    and np.all(np.isfinite(low))
                    and np.all(np.isfinite(high))
                    and np.all(low < high)
                    and np.all(low <= 0.0)
                    and np.all(high >= 0.0)
                    and low[2] < 0.0
                    and high[2] > 0.0
                )
                if valid:
                    return {
                        "source": f"{source}.action_spec",
                        "action_dimension": 7,
                        "low": low.tolist(),
                        "high": high.tolist(),
                        "translation_indices": [0, 1, 2],
                        "z_action_index": 2,
                        "gripper_action_index": 6,
                        "runtime_resolved": True,
                        "rejected_candidates": rejected,
                    }
                rejected.append(
                    {
                        "source": source,
                        "reason": "action_spec is not a finite signed 7-D bound",
                        "low": low.tolist(),
                        "high": high.tolist(),
                    }
                )
        for attribute in ("env", "_env"):
            try:
                child = getattr(current, attribute, None)
            except Exception:
                child = None
            if child is not None:
                queue.append((child, f"{source}.{attribute}"))
    raise RuntimeError(
        "native OSC action bounds unavailable; adaptive vertical motion "
        "cannot be proved and is stopped fail-closed: "
        f"{json.dumps(rejected, sort_keys=True)}"
    )


def _compiled_adaptive_vertical_descent_action(
    *,
    current_eef,
    target_z,
    overhead_guard,
    gripper,
    position_action_scale,
    native_action_spec,
    expected_pair_count,
):
    """Derive one pure-Z descent from every live compiled overhead pair."""
    current_eef = np.asarray(current_eef, dtype=float)
    if current_eef.shape != (3,) or not np.all(np.isfinite(current_eef)):
        raise ValueError("adaptive vertical current EEF is invalid")
    if (
        not np.isfinite(target_z)
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
    ):
        raise ValueError("adaptive vertical target or action scale is invalid")
    if not isinstance(expected_pair_count, (int, np.integer)):
        raise ValueError("expected compiled pair count must be an integer")
    pairs = list(overhead_guard.get("pairs", ()))
    if expected_pair_count <= 0 or len(pairs) != int(expected_pair_count):
        raise RuntimeError(
            "live compiled overhead pair inventory changed before adaptive "
            f"descent: expected={expected_pair_count} observed={len(pairs)}"
        )
    if not overhead_guard.get("accepted", False):
        raise RuntimeError(
            "adaptive vertical descent cannot start after the base overhead "
            "reserve has already been lost"
        )
    try:
        native_low = np.asarray(native_action_spec["low"], dtype=float)
        native_high = np.asarray(native_action_spec["high"], dtype=float)
        native_source = str(native_action_spec["source"])
    except Exception as exc:
        raise RuntimeError(
            "native OSC action-bound evidence is incomplete"
        ) from exc
    if (
        not native_action_spec.get("runtime_resolved", False)
        or native_action_spec.get("action_dimension") != 7
        or native_low.shape != (7,)
        or native_high.shape != (7,)
        or not np.all(np.isfinite(native_low))
        or not np.all(np.isfinite(native_high))
        or not np.all(native_low < native_high)
        or native_low[2] >= 0.0
        or not (native_low[6] <= gripper <= native_high[6])
    ):
        raise RuntimeError(
            "native OSC action bounds do not prove the requested pure-Z action"
        )
    base_reserve = float(overhead_guard["one_step_vertical_reserve_m"])
    if not np.isfinite(base_reserve) or base_reserve <= 0.0:
        raise RuntimeError("compiled overhead base reserve is invalid")
    target_remaining = float(current_eef[2] - float(target_z))
    if not np.isfinite(target_remaining) or target_remaining <= 0.0:
        raise RuntimeError(
            "adaptive vertical descent has no finite positive target Z error"
        )

    pair_envelopes = []
    for index, pair in enumerate(pairs):
        vertical_clearance = float(pair["vertical_clearance_m"])
        strict_clearance = float(pair["strict_no_contact_clearance_m"])
        required_clearance = float(strict_clearance + base_reserve)
        available_descent = float(vertical_clearance - required_clearance)
        if (
            not np.isfinite(vertical_clearance)
            or not np.isfinite(strict_clearance)
            or strict_clearance < 0.0
            or not np.isfinite(available_descent)
            or available_descent <= 0.0
            or not pair.get("accepted", False)
        ):
            raise RuntimeError(
                "compiled overhead pair cannot prove positive adaptive "
                f"descent capacity: index={index} pair="
                f"{json.dumps(pair, sort_keys=True)}"
            )
        strict_safe_delta = float(np.nextafter(available_descent, 0.0))
        if strict_safe_delta <= 0.0:
            raise RuntimeError(
                "compiled overhead pair has no representable strict descent "
                f"capacity: index={index}"
            )
        pair_envelopes.append(
            {
                "pair_index": int(index),
                "gripper_geom": pair["gripper_geom"],
                "counterpart_geom": pair["counterpart_geom"],
                "counterpart_kind": pair["counterpart_kind"],
                "current_vertical_clearance_m": vertical_clearance,
                "strict_no_contact_clearance_m": strict_clearance,
                "base_overhead_reserve_m": base_reserve,
                "required_clearance_with_base_reserve_m": required_clearance,
                "available_descent_before_strict_guard_m": available_descent,
                "strict_safe_negative_world_delta_m": strict_safe_delta,
            }
        )
    limiting_pair = min(
        pair_envelopes,
        key=lambda record: record["strict_safe_negative_world_delta_m"],
    )
    pair_world_capacity = float(
        limiting_pair["strict_safe_negative_world_delta_m"]
    )
    native_negative_z_action_capacity = float(-native_low[2])
    strict_native_negative_z_action = float(
        np.nextafter(native_negative_z_action_capacity, 0.0)
    )
    native_world_capacity = float(
        strict_native_negative_z_action * position_action_scale
    )
    if strict_native_negative_z_action <= 0.0 or native_world_capacity <= 0.0:
        raise RuntimeError(
            "native OSC negative-Z action bound has no strict interior"
        )
    capacities = {
        "target_remaining_z_error": target_remaining,
        "compiled_pair_base8_envelope": pair_world_capacity,
        "native_negative_z_action_bound": native_world_capacity,
    }
    selected_source = min(capacities, key=capacities.get)
    commanded_delta = float(capacities[selected_source])
    if not np.isfinite(commanded_delta) or commanded_delta <= 0.0:
        raise RuntimeError("adaptive vertical envelope selected no safe motion")

    # Subtraction can round an algebraically strict nextafter result back onto
    # the boundary. Tighten inward until the direct per-pair proof is strict.
    for _ in range(128):
        commanded_z_action = float(
            -commanded_delta / float(position_action_scale)
        )
        predicted = [
            float(
                record["current_vertical_clearance_m"] - commanded_delta
            )
            for record in pair_envelopes
        ]
        if (
            native_low[2] < commanded_z_action < 0.0
            and commanded_delta <= target_remaining
            and all(
                clearance
                > record["required_clearance_with_base_reserve_m"]
                for clearance, record in zip(predicted, pair_envelopes)
            )
        ):
            break
        commanded_delta = float(np.nextafter(commanded_delta, 0.0))
    else:
        raise RuntimeError(
            "adaptive vertical command has no directly provable strict "
            "native-action/base8 interior"
        )
    if commanded_delta <= 0.0:
        raise RuntimeError(
            "adaptive vertical command collapsed to zero while proving safety"
        )
    commanded_z_action = float(
        -commanded_delta / float(position_action_scale)
    )
    minimum_predicted_surplus = float("inf")
    for clearance, record in zip(predicted, pair_envelopes):
        record["predicted_post_command_vertical_clearance_m"] = clearance
        record["predicted_post_command_base_reserve_surplus_m"] = float(
            clearance - record["required_clearance_with_base_reserve_m"]
        )
        minimum_predicted_surplus = min(
            minimum_predicted_surplus,
            record["predicted_post_command_base_reserve_surplus_m"],
        )
    action = np.zeros(7, dtype=float)
    action[2] = commanded_z_action
    action[-1] = float(gripper)
    if (
        action[0] != 0.0
        or action[1] != 0.0
        or not native_low[2] < action[2] < 0.0
        or action[2] > native_high[2]
        or minimum_predicted_surplus <= 0.0
    ):
        raise RuntimeError(
            "adaptive vertical action violated its compiled hard proof"
        )
    return action, {
        "formula": (
            "for every live compiled gripper-versus-plate/table pair, subtract "
            "strict pair clearance and the unchanged 8 mm base reserve from "
            "current vertical clearance; select the strict minimum of all-pair "
            "capacity, remaining target-Z error, and the runtime-resolved native "
            "negative-Z action capacity times position_action_scale"
        ),
        "current_eef": current_eef.tolist(),
        "target_z_m": float(target_z),
        "target_remaining_z_error_m": target_remaining,
        "position_action_scale_m_per_normalized_action": float(
            position_action_scale
        ),
        "native_action_spec_source": native_source,
        "native_z_action_bounds": [
            float(native_low[2]),
            float(native_high[2]),
        ],
        "strict_native_negative_z_action_capacity": (
            strict_native_negative_z_action
        ),
        "strict_native_negative_z_world_delta_capacity_m": (
            native_world_capacity
        ),
        "compiled_pair_count": len(pair_envelopes),
        "pair_envelopes": pair_envelopes,
        "selected_limiting_pair": dict(limiting_pair),
        "candidate_capacities_m": capacities,
        "selected_envelope_source": selected_source,
        "commanded_negative_world_delta_m": commanded_delta,
        "commanded_xy_action": action[:2].tolist(),
        "commanded_z_action": commanded_z_action,
        "minimum_predicted_post_command_base_reserve_surplus_m": (
            minimum_predicted_surplus
        ),
        "proof": {
            "pure_negative_z": True,
            "strictly_inside_native_z_action_bound": True,
            "does_not_cross_target_z": bool(
                commanded_delta <= target_remaining
            ),
            "all_compiled_pairs_retain_strict_base8_after_command": True,
        },
    }


def _compiled_adaptive_high_lateral_action(
    *,
    current_eef,
    lateral_target_xy,
    overhead_guard,
    gripper,
    position_action_scale,
    native_action_spec,
    expected_pair_count,
):
    """Derive one high pure-XY action from the live base8 pair envelope."""
    current_eef = np.asarray(current_eef, dtype=float)
    lateral_target_xy = np.asarray(lateral_target_xy, dtype=float)
    if (
        current_eef.shape != (3,)
        or lateral_target_xy.shape != (2,)
        or not np.all(np.isfinite(current_eef))
        or not np.all(np.isfinite(lateral_target_xy))
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
    ):
        raise ValueError("adaptive high-lateral geometry is invalid")
    if not isinstance(expected_pair_count, (int, np.integer)):
        raise ValueError("expected compiled pair count must be an integer")
    pairs = list(overhead_guard.get("pairs", ()))
    if expected_pair_count <= 0 or len(pairs) != int(expected_pair_count):
        raise RuntimeError(
            "live compiled overhead pair inventory changed before adaptive "
            f"high lateral: expected={expected_pair_count} "
            f"observed={len(pairs)}"
        )
    identities = [_overhead_pair_identity(pair) for pair in pairs]
    if len(set(identities)) != len(identities):
        raise RuntimeError(
            "adaptive high-lateral evidence contains a duplicate pair "
            "identity"
        )
    if not overhead_guard.get("accepted", False):
        raise RuntimeError(
            "adaptive high lateral cannot start after the base overhead "
            "reserve has already been lost"
        )
    try:
        native_low = np.asarray(native_action_spec["low"], dtype=float)
        native_high = np.asarray(native_action_spec["high"], dtype=float)
        native_source = str(native_action_spec["source"])
    except Exception as exc:
        raise RuntimeError(
            "native OSC action-bound evidence is incomplete"
        ) from exc
    if (
        not native_action_spec.get("runtime_resolved", False)
        or native_action_spec.get("action_dimension") != 7
        or native_low.shape != (7,)
        or native_high.shape != (7,)
        or not np.all(np.isfinite(native_low))
        or not np.all(np.isfinite(native_high))
        or not np.all(native_low < native_high)
        or not np.all(native_low[:6] < 0.0)
        or not np.all(native_high[:6] > 0.0)
        or not (native_low[6] <= gripper <= native_high[6])
    ):
        raise RuntimeError(
            "native OSC action bounds do not prove the requested pure-XY "
            "action"
        )
    native_xy_norm_bound = float(
        min(
            -native_low[0],
            native_high[0],
            -native_low[1],
            native_high[1],
        )
    )
    strict_native_xy_norm_bound = float(
        np.nextafter(native_xy_norm_bound, 0.0)
    )
    if strict_native_xy_norm_bound <= 0.0:
        raise RuntimeError(
            "native OSC XY action-norm bound has no strict interior"
        )
    base_reserve = float(overhead_guard["one_step_vertical_reserve_m"])
    if not np.isfinite(base_reserve) or base_reserve <= 0.0:
        raise RuntimeError("compiled overhead base reserve is invalid")
    lateral_delta = lateral_target_xy - current_eef[:2]
    lateral_remaining = float(np.linalg.norm(lateral_delta))
    if not np.isfinite(lateral_remaining) or lateral_remaining <= 0.0:
        raise RuntimeError(
            "adaptive high lateral has no finite positive target XY error"
        )
    direction = lateral_delta / lateral_remaining

    pair_envelopes = []
    for index, (identity, pair) in enumerate(zip(identities, pairs)):
        vertical_clearance = float(pair["vertical_clearance_m"])
        strict_clearance = float(pair["strict_no_contact_clearance_m"])
        required_clearance = float(strict_clearance + base_reserve)
        available_worst_case_tail = float(
            vertical_clearance - required_clearance
        )
        if (
            not np.isfinite(vertical_clearance)
            or not np.isfinite(strict_clearance)
            or strict_clearance < 0.0
            or not np.isfinite(available_worst_case_tail)
            or available_worst_case_tail <= 0.0
            or not pair.get("accepted", False)
        ):
            raise RuntimeError(
                "compiled overhead pair cannot prove positive adaptive "
                f"lateral capacity: index={index} pair="
                f"{json.dumps(pair, sort_keys=True)}"
            )
        strict_safe_world_tail = float(
            np.nextafter(available_worst_case_tail, 0.0)
        )
        if strict_safe_world_tail <= 0.0:
            raise RuntimeError(
                "compiled overhead pair has no representable strict "
                f"high-lateral capacity: index={index}"
            )
        pair_envelopes.append(
            {
                "pair_index": int(index),
                "pair_identity": list(identity),
                "gripper_geom": pair["gripper_geom"],
                "counterpart_geom": pair["counterpart_geom"],
                "counterpart_kind": pair["counterpart_kind"],
                "current_vertical_clearance_m": vertical_clearance,
                "strict_no_contact_clearance_m": strict_clearance,
                "base_overhead_reserve_m": base_reserve,
                "required_clearance_with_base_reserve_m": (
                    required_clearance
                ),
                "available_worst_case_downward_tail_m": (
                    available_worst_case_tail
                ),
                "strict_safe_worst_case_downward_tail_m": (
                    strict_safe_world_tail
                ),
                "representable_inward_numerical_guard_m": float(
                    available_worst_case_tail - strict_safe_world_tail
                ),
                "strict_safe_translation_action_norm_capacity": float(
                    strict_safe_world_tail / position_action_scale
                ),
            }
        )
    limiting_pair = min(
        pair_envelopes,
        key=lambda record: record[
            "strict_safe_translation_action_norm_capacity"
        ],
    )
    pair_action_norm_capacity = float(
        limiting_pair["strict_safe_translation_action_norm_capacity"]
    )
    remaining_action_norm = float(
        lateral_remaining / position_action_scale
    )
    capacities = {
        "lateral_target_remaining_action_norm": remaining_action_norm,
        "compiled_pair_base8_worst_case_tail": pair_action_norm_capacity,
        "native_xy_translation_action_norm_bound": (
            strict_native_xy_norm_bound
        ),
    }
    selected_source = min(capacities, key=capacities.get)
    commanded_norm = float(capacities[selected_source])
    if not np.isfinite(commanded_norm) or commanded_norm <= 0.0:
        raise RuntimeError(
            "adaptive high-lateral envelope selected no safe motion"
        )

    # Re-evaluate the literal floating-point command.  This guard covers norm
    # reconstruction and subtraction rounding in addition to each pair's
    # nextafter-inward tail.
    for _ in range(128):
        commanded_xy = direction * commanded_norm
        literal_action_norm = float(np.linalg.norm(commanded_xy))
        worst_case_world_tail = float(
            position_action_scale * literal_action_norm
        )
        predicted = [
            float(
                record["current_vertical_clearance_m"]
                - worst_case_world_tail
            )
            for record in pair_envelopes
        ]
        if (
            0.0 < literal_action_norm < native_xy_norm_bound
            and worst_case_world_tail <= lateral_remaining
            and native_low[0] < commanded_xy[0] < native_high[0]
            and native_low[1] < commanded_xy[1] < native_high[1]
            and all(
                clearance
                > record["required_clearance_with_base_reserve_m"]
                for clearance, record in zip(predicted, pair_envelopes)
            )
        ):
            break
        commanded_norm = float(np.nextafter(commanded_norm, 0.0))
    else:
        raise RuntimeError(
            "adaptive high-lateral command has no directly provable strict "
            "native-action/base8 interior"
        )
    if commanded_norm <= 0.0:
        raise RuntimeError(
            "adaptive high-lateral command collapsed to zero while proving "
            "safety"
        )

    minimum_predicted_surplus = float("inf")
    for clearance, record in zip(predicted, pair_envelopes):
        record["predicted_post_worst_case_vertical_clearance_m"] = (
            clearance
        )
        record["predicted_post_worst_case_base_reserve_surplus_m"] = float(
            clearance - record["required_clearance_with_base_reserve_m"]
        )
        minimum_predicted_surplus = min(
            minimum_predicted_surplus,
            record[
                "predicted_post_worst_case_base_reserve_surplus_m"
            ],
        )
    action = np.zeros(7, dtype=float)
    action[:2] = commanded_xy
    action[-1] = float(gripper)
    if (
        action[2] != 0.0
        or np.any(action[3:6] != 0.0)
        or literal_action_norm >= native_xy_norm_bound
        or minimum_predicted_surplus <= 0.0
    ):
        raise RuntimeError(
            "adaptive high-lateral action violated its compiled hard proof"
        )
    return action, {
        "accepted": True,
        "formula": (
            "for every exact live compiled gripper-versus-plate/table pair, "
            "subtract strict pair clearance and the unchanged 8 mm base "
            "reserve from current vertical clearance; after a nextafter "
            "inward numerical guard, divide the limiting worst-case downward "
            "tail by position_action_scale and intersect it with remaining "
            "XY target error and the strict runtime-native XY action-norm "
            "bound"
        ),
        "current_eef": current_eef.tolist(),
        "lateral_target_xy": lateral_target_xy.tolist(),
        "lateral_remaining_m": lateral_remaining,
        "lateral_direction_xy": direction.tolist(),
        "position_action_scale_m_per_normalized_action": float(
            position_action_scale
        ),
        "native_action_spec_source": native_source,
        "native_xy_component_bounds": {
            "low": native_low[:2].tolist(),
            "high": native_high[:2].tolist(),
        },
        "native_xy_translation_action_norm_bound": native_xy_norm_bound,
        "strict_native_xy_translation_action_norm_bound": (
            strict_native_xy_norm_bound
        ),
        "compiled_pair_count": len(pair_envelopes),
        "pair_identity_keys": [list(identity) for identity in identities],
        "pair_envelopes": pair_envelopes,
        "selected_limiting_pair": dict(limiting_pair),
        "candidate_action_norm_capacities": capacities,
        "selected_envelope_source": selected_source,
        "commanded_translation_action_norm": literal_action_norm,
        "commanded_worst_case_downward_world_tail_m": (
            worst_case_world_tail
        ),
        "commanded_xy_action": action[:2].tolist(),
        "commanded_z_action": float(action[2]),
        "minimum_predicted_post_worst_case_base_surplus_m": (
            minimum_predicted_surplus
        ),
        "proof": {
            "pure_xy_zero_z_rotation": True,
            "strictly_inside_native_xy_action_norm_bound": True,
            "does_not_cross_lateral_target": bool(
                worst_case_world_tail <= lateral_remaining
            ),
            "all_compiled_pairs_retain_strict_base8_after_worst_case_tail": (
                True
            ),
        },
    }


def _compiled_adaptive_lateral_rebuffer_action(
    *,
    current_eef,
    overhead_guard,
    overhead_lateral_buffer,
    outside_side_guard,
    gripper,
    position_action_scale,
    native_action_spec,
    expected_pair_count,
    worst_case_controller_world_step_m,
):
    """Lift in pure Z from every live pair's exact buffer deficit."""
    current_eef = np.asarray(current_eef, dtype=float)
    if current_eef.shape != (3,) or not np.all(np.isfinite(current_eef)):
        raise ValueError("adaptive lateral-rebuffer current EEF is invalid")
    if (
        not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(worst_case_controller_world_step_m)
        or worst_case_controller_world_step_m <= 0.0
    ):
        raise ValueError("adaptive lateral-rebuffer scales are invalid")
    if not isinstance(expected_pair_count, (int, np.integer)):
        raise ValueError("expected compiled pair count must be an integer")
    overhead_pairs = list(overhead_guard.get("pairs", ()))
    buffer_pairs = list(overhead_lateral_buffer.get("pairs", ()))
    if (
        expected_pair_count <= 0
        or len(overhead_pairs) != int(expected_pair_count)
        or len(buffer_pairs) != int(expected_pair_count)
    ):
        raise RuntimeError(
            "live compiled overhead pair inventory changed before adaptive "
            "lateral rebuffer: "
            f"expected={expected_pair_count} "
            f"overhead={len(overhead_pairs)} buffer={len(buffer_pairs)}"
        )
    if not overhead_guard.get("accepted", False):
        raise RuntimeError(
            "adaptive lateral rebuffer cannot start after the base overhead "
            "reserve has already been lost"
        )
    try:
        native_low = np.asarray(native_action_spec["low"], dtype=float)
        native_high = np.asarray(native_action_spec["high"], dtype=float)
        native_source = str(native_action_spec["source"])
    except Exception as exc:
        raise RuntimeError(
            "native OSC action-bound evidence is incomplete"
        ) from exc
    if (
        not native_action_spec.get("runtime_resolved", False)
        or native_action_spec.get("action_dimension") != 7
        or native_low.shape != (7,)
        or native_high.shape != (7,)
        or not np.all(np.isfinite(native_low))
        or not np.all(np.isfinite(native_high))
        or not np.all(native_low < native_high)
        or native_high[2] <= 0.0
        or not (native_low[6] <= gripper <= native_high[6])
    ):
        raise RuntimeError(
            "native OSC action bounds do not prove the requested pure +Z "
            "lateral rebuffer"
        )

    base_reserve = float(overhead_guard["one_step_vertical_reserve_m"])
    buffer_base_reserve = float(
        overhead_lateral_buffer["base_overhead_reserve_m"]
    )
    buffer_world_step = float(
        overhead_lateral_buffer["worst_case_controller_world_step_m"]
    )
    if (
        not np.isfinite(base_reserve)
        or base_reserve <= 0.0
        or buffer_base_reserve != base_reserve
        or buffer_world_step != float(worst_case_controller_world_step_m)
    ):
        raise RuntimeError(
            "live lateral-buffer derivation no longer matches the unchanged "
            "base8/controller-step envelope"
        )

    pair_envelopes = []
    maximum_buffer_deficit = 0.0
    for index, (pair, buffer_pair) in enumerate(
        zip(overhead_pairs, buffer_pairs)
    ):
        strict_clearance = float(pair["strict_no_contact_clearance_m"])
        vertical_clearance = float(pair["vertical_clearance_m"])
        required_base = float(strict_clearance + base_reserve)
        required_buffer = float(
            required_base + worst_case_controller_world_step_m
        )
        if (
            pair.get("gripper_geom") != buffer_pair.get("gripper_geom")
            or pair.get("counterpart_geom")
            != buffer_pair.get("counterpart_geom")
            or pair.get("counterpart_kind")
            != buffer_pair.get("counterpart_kind")
            or float(buffer_pair["vertical_clearance_m"])
            != vertical_clearance
            or float(buffer_pair["strict_no_contact_clearance_m"])
            != strict_clearance
            or float(buffer_pair["required_lateral_entry_clearance_m"])
            != required_buffer
        ):
            raise RuntimeError(
                "live overhead and lateral-buffer pair evidence diverged: "
                f"index={index}"
            )
        current_base_surplus = float(vertical_clearance - required_base)
        current_buffer_surplus = float(vertical_clearance - required_buffer)
        if (
            not np.isfinite(strict_clearance)
            or strict_clearance < 0.0
            or not np.isfinite(vertical_clearance)
            or not pair.get("accepted", False)
            or current_base_surplus <= 0.0
        ):
            raise RuntimeError(
                "compiled pair cannot prove strict contact/base8 safety for "
                f"adaptive lateral rebuffer: index={index}"
            )
        buffer_deficit = float(max(0.0, -current_buffer_surplus))
        maximum_buffer_deficit = max(
            maximum_buffer_deficit, buffer_deficit
        )
        pair_envelopes.append(
            {
                "pair_index": int(index),
                "gripper_geom": pair["gripper_geom"],
                "counterpart_geom": pair["counterpart_geom"],
                "counterpart_kind": pair["counterpart_kind"],
                "current_vertical_clearance_m": vertical_clearance,
                "strict_no_contact_clearance_m": strict_clearance,
                "required_clearance_with_base8_m": required_base,
                "required_clearance_with_buffer16_m": required_buffer,
                "current_base8_surplus_m": current_base_surplus,
                "current_buffer16_surplus_m": current_buffer_surplus,
                "live_buffer16_deficit_m": buffer_deficit,
            }
        )
    # Refill the exact worst live deficit and one unchanged controller-step
    # tail.  If buffer16 is already positive but a delayed negative dz remains,
    # the deficit is zero and this still supplies the unchanged +Z tail brake.
    # The tail is not a relaxed threshold: it is the same 8 mm term already
    # used to derive buffer16 and protects the immediately resumed XY action
    # against its proved worst-case downward response.
    requested_delta = float(
        np.nextafter(
            maximum_buffer_deficit
            + float(worst_case_controller_world_step_m),
            np.inf,
        )
    )
    strict_native_positive_z_action = float(
        np.nextafter(float(native_high[2]), 0.0)
    )
    native_world_capacity = float(
        strict_native_positive_z_action * position_action_scale
    )
    if (
        strict_native_positive_z_action <= 0.0
        or native_world_capacity < requested_delta
    ):
        raise RuntimeError(
            "runtime native +Z action bound cannot prove one-step recovery "
            "of the live buffer16 deficit plus the unchanged lateral tail: "
            f"requested_m={requested_delta} capacity_m={native_world_capacity}"
        )
    commanded_delta = requested_delta
    commanded_z_action = float(commanded_delta / position_action_scale)
    if not 0.0 < commanded_z_action < native_high[2]:
        raise RuntimeError(
            "adaptive lateral-rebuffer command has no strict native +Z "
            "interior"
        )

    minimum_contact_surplus = float("inf")
    minimum_base_surplus = float("inf")
    minimum_buffer_surplus = float("inf")
    for record in pair_envelopes:
        predicted_clearance = float(
            record["current_vertical_clearance_m"] + commanded_delta
        )
        contact_surplus = float(
            predicted_clearance
            - record["strict_no_contact_clearance_m"]
        )
        base_surplus = float(
            predicted_clearance
            - record["required_clearance_with_base8_m"]
        )
        buffer_surplus = float(
            predicted_clearance
            - record["required_clearance_with_buffer16_m"]
        )
        record.update(
            {
                "predicted_post_command_vertical_clearance_m": (
                    predicted_clearance
                ),
                "predicted_post_command_strict_contact_surplus_m": (
                    contact_surplus
                ),
                "predicted_post_command_base8_surplus_m": base_surplus,
                "predicted_post_command_buffer16_surplus_m": buffer_surplus,
            }
        )
        minimum_contact_surplus = min(
            minimum_contact_surplus, contact_surplus
        )
        minimum_base_surplus = min(minimum_base_surplus, base_surplus)
        minimum_buffer_surplus = min(
            minimum_buffer_surplus, buffer_surplus
        )
    if (
        minimum_contact_surplus <= 0.0
        or minimum_base_surplus <= 0.0
        or minimum_buffer_surplus <= 0.0
    ):
        raise RuntimeError(
            "adaptive lateral-rebuffer action violated its direct all-pair "
            "contact/base8/buffer16 proof"
        )

    outside_clearance = float(
        outside_side_guard["minimum_outside_clearance_m"]
    )
    required_outside_clearance = float(
        outside_side_guard["required_outside_clearance_m"]
    )
    if not (
        np.isfinite(outside_clearance)
        and np.isfinite(required_outside_clearance)
    ):
        raise RuntimeError(
            "live outside-side evidence is invalid before lateral rebuffer"
        )
    action = np.zeros(7, dtype=float)
    action[2] = commanded_z_action
    action[-1] = float(gripper)
    return action, {
        "formula": (
            "take the maximum live deficit to the unchanged strict+base8+"
            "one-controller-step buffer16 envelope over every compiled pair, "
            "add that same unchanged controller-step tail for the immediately "
            "resumed XY action, and require the resulting pure +Z command to "
            "remain strictly inside the runtime native action bound"
        ),
        "current_eef": current_eef.tolist(),
        "position_action_scale_m_per_normalized_action": float(
            position_action_scale
        ),
        "native_action_spec_source": native_source,
        "native_z_action_bounds": [
            float(native_low[2]),
            float(native_high[2]),
        ],
        "strict_native_positive_z_world_delta_capacity_m": (
            native_world_capacity
        ),
        "compiled_pair_count": len(pair_envelopes),
        "pair_envelopes": pair_envelopes,
        "maximum_live_buffer_deficit_m": maximum_buffer_deficit,
        "requested_positive_world_delta_m": requested_delta,
        "commanded_positive_world_delta_m": commanded_delta,
        "commanded_xy_action": action[:2].tolist(),
        "commanded_z_action": commanded_z_action,
        "pre_action_outside_side_guard": dict(outside_side_guard),
        "minimum_predicted_post_command_contact_surplus_m": (
            minimum_contact_surplus
        ),
        "minimum_predicted_post_command_base8_surplus_m": (
            minimum_base_surplus
        ),
        "minimum_predicted_post_command_buffer16_surplus_m": (
            minimum_buffer_surplus
        ),
        "proof": {
            "pure_positive_z": True,
            "strictly_inside_native_z_action_bound": True,
            "outside_xy_clearance_not_worsened_by_pure_z": True,
            "all_compiled_pairs_retain_strict_no_contact": True,
            "all_compiled_pairs_retain_strict_base8": True,
            "all_compiled_pairs_reach_strict_buffer16": True,
        },
    }


def _fixed_xy_vertical_approach_action(
    *,
    current_eef,
    target_z,
    gripper,
    position_action_scale,
    maximum_translation_action,
):
    """Move only in Z under the unchanged strict translation-action bound."""
    current_eef = np.asarray(current_eef, dtype=float)
    if current_eef.shape != (3,) or not np.all(np.isfinite(current_eef)):
        raise ValueError("fixed-XY vertical current EEF is invalid")
    if (
        not np.isfinite(target_z)
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(maximum_translation_action)
        or not (0.0 < maximum_translation_action <= 1.0)
    ):
        raise ValueError("fixed-XY vertical action bounds are invalid")
    strict_bound = float(np.nextafter(maximum_translation_action, 0.0))
    requested_z_action = float(
        (float(target_z) - current_eef[2]) / position_action_scale
    )
    commanded_z_action = float(
        np.clip(requested_z_action, -strict_bound, strict_bound)
    )
    action = np.zeros(7, dtype=float)
    action[2] = commanded_z_action
    action[-1] = float(gripper)
    if (
        action[0] != 0.0
        or action[1] != 0.0
        or abs(action[2]) > maximum_translation_action
    ):
        raise RuntimeError("fixed-XY vertical action violated its hard bound")
    return action, {
        "formula": (
            "hold commanded XY exactly at zero and move toward the compiled "
            "overhead staging Z under the strict inward floating-point "
            "representation of the unchanged translation-action norm"
        ),
        "current_eef": current_eef.tolist(),
        "target_z_m": float(target_z),
        "requested_z_action": requested_z_action,
        "commanded_xy_action": action[:2].tolist(),
        "commanded_z_action": commanded_z_action,
        "translation_action_norm": float(abs(commanded_z_action)),
        "allocation_translation_action_bound": strict_bound,
        "maximum_translation_action": float(
            maximum_translation_action
        ),
    }


def _outside_side_step_response_evidence(
    *,
    before_guard,
    after_guard,
    before_eef,
    after_eef,
):
    """Measure one discrete OSC step along the live outward direction."""
    outward = np.asarray(
        before_guard["outward_direction_xy"], dtype=float
    )
    before_eef = np.asarray(before_eef, dtype=float)
    after_eef = np.asarray(after_eef, dtype=float)
    if (
        outward.shape != (2,)
        or before_eef.shape != (3,)
        or after_eef.shape != (3,)
    ):
        raise ValueError("outside-side response vectors have invalid shape")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 1e-9:
        raise ValueError("outward direction must be finite and nonzero")
    outward /= outward_norm
    before_clearance = float(
        before_guard["minimum_outside_clearance_m"]
    )
    after_clearance = float(
        after_guard["minimum_outside_clearance_m"]
    )
    clearance_progress = after_clearance - before_clearance
    eef_outward_progress = float(
        np.dot(after_eef[:2] - before_eef[:2], outward)
    )
    return {
        "eef_outward_step_progress_m": eef_outward_progress,
        "vertical_step_progress_m": float(
            after_eef[2] - before_eef[2]
        ),
        "outside_clearance_step_progress_m": float(
            clearance_progress
        ),
        "before_clearance_m": before_clearance,
        "after_clearance_m": after_clearance,
    }


def _outside_side_lateral_settle_evidence(
    *,
    before_guard,
    after_guard,
    before_eef,
    after_eef,
    previous_stable_response_count=0,
    required_stable_response_count=2,
):
    """Require lateral and vertical motion to stop trending toward hazards."""
    before_eef = np.asarray(before_eef, dtype=float)
    after_eef = np.asarray(after_eef, dtype=float)
    if before_eef.shape != (3,) or after_eef.shape != (3,):
        raise ValueError("outside-side settle EEF vectors must be 3-D")
    if (
        isinstance(previous_stable_response_count, bool)
        or not isinstance(previous_stable_response_count, (int, np.integer))
        or previous_stable_response_count < 0
        or isinstance(required_stable_response_count, bool)
        or not isinstance(required_stable_response_count, (int, np.integer))
        or required_stable_response_count < 2
    ):
        raise ValueError(
            "outside-side settle confirmation counts must be integers "
            "with a required count of at least two"
        )
    step_response = _outside_side_step_response_evidence(
        before_guard=before_guard,
        after_guard=after_guard,
        before_eef=before_eef,
        after_eef=after_eef,
    )
    vertical_step_progress = float(
        step_response["vertical_step_progress_m"]
    )
    required_clearance = float(
        after_guard["required_outside_clearance_m"]
    )
    live_clearance = float(
        after_guard["minimum_outside_clearance_m"]
    )
    violations = []
    if vertical_step_progress < 0.0:
        violations.append("eef_still_descending_during_lateral_settle")
    if step_response["eef_outward_step_progress_m"] < 0.0:
        violations.append("eef_still_moving_inward_during_lateral_settle")
    if step_response["outside_clearance_step_progress_m"] < 0.0:
        violations.append(
            "outside_clearance_still_decreasing_during_lateral_settle"
        )
    if live_clearance < required_clearance:
        violations.append(
            "outside_clearance_below_compiled_requirement_during_settle"
        )
    instantaneous_stable_response = not violations
    stable_response_count = (
        int(previous_stable_response_count) + 1
        if instantaneous_stable_response
        else 0
    )
    return {
        "settled": bool(
            stable_response_count >= required_stable_response_count
        ),
        "instantaneous_stable_response": (
            instantaneous_stable_response
        ),
        "previous_stable_response_count": int(
            previous_stable_response_count
        ),
        "stable_response_count": stable_response_count,
        "required_stable_response_count": int(
            required_stable_response_count
        ),
        "violations": violations,
        "formula": (
            "after every descent step, preserve compiled outside XY and "
            "actively brake in positive Z whenever the previous measured "
            "Z response is negative; require at least two consecutive "
            "settle frames where measured Z, EEF-outward, and live-clearance "
            "step progress are all nonnegative and compiled clearance is "
            "satisfied before permitting another descent"
        ),
        "vertical_step_progress_m": vertical_step_progress,
        "step_response": step_response,
        "required_outside_clearance_m": required_clearance,
        "live_outside_clearance_m": live_clearance,
    }


def _outside_side_staircase_settle_trigger(
    *,
    feedback_mode,
    guard_step,
    step_response,
):
    """Require an active-braking settle after every commanded descent."""
    if feedback_mode != "constraint_prioritized_vertical_descent":
        return None
    return {
        "policy": (
            "preventive staircase: every constraint-prioritized descent "
            "step is followed by measured outside-XY-prioritized active "
            "braking before another descent can be issued"
        ),
        "trigger_guard_step": int(guard_step),
        "trigger_step_response": step_response,
        "stable_response_count": 0,
        "inward_response_observed": bool(
            step_response["eef_outward_step_progress_m"] < 0.0
            or step_response[
                "outside_clearance_step_progress_m"
            ]
            < 0.0
        ),
    }


def _outside_side_recovery_progress_evidence(
    *,
    baseline_guard,
    after_guard,
    baseline_eef,
    before_guard,
    before_eef,
    after_eef,
    action,
    maximum_translation_action,
    previous_step_response=None,
):
    """Prove net recovery or a strictly improving saturated OSC response."""
    outward = np.asarray(
        baseline_guard["outward_direction_xy"], dtype=float
    )
    baseline_eef = np.asarray(baseline_eef, dtype=float)
    after_eef = np.asarray(after_eef, dtype=float)
    action = np.asarray(action, dtype=float)
    if (
        outward.shape != (2,)
        or baseline_eef.shape != (3,)
        or after_eef.shape != (3,)
        or action.shape[0] < 3
    ):
        raise ValueError("outside-side recovery vectors have invalid shape")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 1e-9:
        raise ValueError("outward direction must be finite and nonzero")
    if (
        not np.isfinite(maximum_translation_action)
        or not (0.0 < maximum_translation_action <= 1.0)
    ):
        raise ValueError("maximum translation action must be in (0, 1]")
    outward /= outward_norm
    step_response = _outside_side_step_response_evidence(
        before_guard=before_guard,
        after_guard=after_guard,
        before_eef=before_eef,
        after_eef=after_eef,
    )
    baseline_clearance = float(
        baseline_guard["minimum_outside_clearance_m"]
    )
    after_clearance = float(
        after_guard["minimum_outside_clearance_m"]
    )
    net_clearance_progress = after_clearance - baseline_clearance
    net_eef_outward_progress = float(
        np.dot(after_eef[:2] - baseline_eef[:2], outward)
    )
    commanded_outward_action = float(
        np.dot(action[:2], outward)
    )
    saturation_floor = float(
        np.nextafter(maximum_translation_action, -np.inf)
    )
    action_saturated = (
        commanded_outward_action >= saturation_floor
    )
    progress_proven = (
        net_clearance_progress > 0.0
        and net_eef_outward_progress > 0.0
    )
    response_improving = None
    eef_response_recovering = None
    clearance_response_recovering = None
    eef_response_acceleration = None
    clearance_response_acceleration = None
    if previous_step_response is not None:
        previous_eef_progress = float(
            previous_step_response[
                "eef_outward_step_progress_m"
            ]
        )
        previous_clearance_progress = float(
            previous_step_response[
                "outside_clearance_step_progress_m"
            ]
        )
        eef_response_acceleration = float(
            step_response["eef_outward_step_progress_m"]
            - previous_eef_progress
        )
        clearance_response_acceleration = float(
            step_response["outside_clearance_step_progress_m"]
            - previous_clearance_progress
        )
        eef_response_recovering = bool(
            step_response["eef_outward_step_progress_m"] > 0.0
            or eef_response_acceleration > 0.0
        )
        clearance_response_recovering = bool(
            step_response["outside_clearance_step_progress_m"] > 0.0
            or clearance_response_acceleration > 0.0
        )
        response_improving = bool(
            eef_response_recovering
            and clearance_response_recovering
        )
    violations = []
    if not action_saturated:
        violations.append(
            "outward_recovery_action_not_at_controller_bound"
        )
    if (
        not progress_proven
        and previous_step_response is not None
        and not response_improving
    ):
        if not eef_response_recovering:
            violations.append(
                "eef_outward_response_neither_moved_nor_accelerated_outward_under_saturation"
            )
        if not clearance_response_recovering:
            violations.append(
                "live_outside_clearance_response_neither_moved_nor_accelerated_outward_under_saturation"
            )
    fail_closed = bool(violations)
    return {
        "accepted": not fail_closed,
        "fail_closed": fail_closed,
        "progress_proven": progress_proven,
        "pending_controller_response": bool(
            not fail_closed and not progress_proven
        ),
        "response_improving": response_improving,
        "eef_response_recovering": eef_response_recovering,
        "clearance_response_recovering": (
            clearance_response_recovering
        ),
        "response_basis": (
            "strictly positive signed step progress, or strictly positive "
            "discrete acceleration while signed progress remains inward, "
            "for both EEF and live clearance under saturated commands"
        ),
        "violations": violations,
        "baseline_clearance_m": baseline_clearance,
        "after_clearance_m": after_clearance,
        "net_clearance_progress_m": float(net_clearance_progress),
        "net_eef_outward_progress_m": net_eef_outward_progress,
        "step_response": step_response,
        "previous_step_response": previous_step_response,
        "eef_response_acceleration_m_per_step": (
            eef_response_acceleration
        ),
        "clearance_response_acceleration_m_per_step": (
            clearance_response_acceleration
        ),
        "commanded_outward_action": commanded_outward_action,
        "maximum_translation_action": float(
            maximum_translation_action
        ),
        "action_saturated": action_saturated,
    }


def _semantic_finger_side(body_name):
    """Map a compiled gripper body name to its native left/right finger."""
    body = str(body_name).lower()
    if (
        "leftfinger" in body
        or "left_finger" in body
        or "finger1" in body
        or "joint1" in body
    ):
        return "left"
    if (
        "rightfinger" in body
        or "right_finger" in body
        or "finger2" in body
        or "joint2" in body
    ):
        return "right"
    return None


def _finger_inward_extents_by_semantic_side(
    finger_bounds,
    eef_position,
    outward_direction_xy,
):
    """Measure each native finger's plate-facing support independently."""
    eef_position = np.asarray(eef_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    if eef_position.shape != (3,) or outward.shape != (2,):
        raise ValueError("EEF position must be 3-D and outward direction 2-D")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 1e-9:
        raise ValueError("outward direction must be finite and nonzero")
    outward /= outward_norm
    bounds_by_side = {
        "left": [
            bound
            for bound in finger_bounds
            if _semantic_finger_side(bound[1]) == "left"
        ],
        "right": [
            bound
            for bound in finger_bounds
            if _semantic_finger_side(bound[1]) == "right"
        ],
    }
    if not all(bounds_by_side.values()):
        raise RuntimeError(
            "compiled left/right finger collision geoms unavailable"
        )
    side_extents = {
        side: min(
            float(
                np.dot(center[:2] - eef_position[:2], outward)
                - np.dot(half_size[:2], np.abs(outward))
            )
            for _, _, center, half_size in bounds
        )
        for side, bounds in bounds_by_side.items()
    }
    shared_collision_free_extent = min(side_extents.values())
    dual_finger_skew = abs(
        side_extents["left"] - side_extents["right"]
    )
    semantic_bounds = [
        bound
        for bounds in bounds_by_side.values()
        for bound in bounds
    ]
    return (
        side_extents,
        shared_collision_free_extent,
        dual_finger_skew,
        semantic_bounds,
    )


def _compiled_native_side_contact_plan(
    env,
    plate_position,
    eef_position,
    outward_direction_xy,
    contact_xy,
    outside_clearance_m,
):
    """Derive side-contact targets from native plate and finger collision AABBs."""
    model, data = env.sim.model, env.sim.data
    plate_position = np.asarray(plate_position, dtype=float)
    eef_position = np.asarray(eef_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    outward /= np.linalg.norm(outward)
    plate_collision_geoms = [
        geom_id
        for geom_id in _compiled_body_geom_ids(model, PLATE_BODY)
        if (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    plate_bounds = [
        (geom_id, *_compiled_geom_world_aabb(model, data, geom_id))
        for geom_id in plate_collision_geoms
    ]
    radial_centers = [
        float(np.linalg.norm(center[:2] - plate_position[:2]))
        for _, center, _ in plate_bounds
    ]
    maximum_radial_center = max(radial_centers, default=0.0)
    if maximum_radial_center <= 1e-6:
        raise RuntimeError("native plate rim collision geoms unavailable")
    rim_bounds = [
        bound
        for bound, radius in zip(plate_bounds, radial_centers)
        if radius >= 0.5 * maximum_radial_center
    ]
    plate_outward_support = max(
        float(
            np.dot(center[:2] - plate_position[:2], outward)
            + np.dot(half_size[:2], np.abs(outward))
        )
        for _, center, half_size in rim_bounds
    )
    plate_rim_center_z = float(
        np.median([center[2] for _, center, _ in rim_bounds])
    )
    plate_rim_vertical_interval = [
        min(
            float(center[2] - half_size[2])
            for _, center, half_size in rim_bounds
        ),
        max(
            float(center[2] + half_size[2])
            for _, center, half_size in rim_bounds
        ),
    ]

    finger_bounds = []
    for geom_id in range(int(model.ngeom)):
        body_name = model.body_id2name(
            int(model.geom_bodyid[geom_id])
        ) or ""
        if "finger" not in body_name.lower():
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        center, half_size = _compiled_geom_world_aabb(
            model, data, geom_id
        )
        finger_bounds.append(
            (geom_id, body_name, center, half_size)
        )
    (
        finger_side_inward_extents,
        finger_inward_extent,
        dual_finger_contact_skew,
        semantic_finger_bounds,
    ) = _finger_inward_extents_by_semantic_side(
        finger_bounds,
        eef_position,
        outward,
    )
    finger_center_z_offset = float(
        np.mean(
            [
                center[2] - eef_position[2]
                for _, _, center, _ in semantic_finger_bounds
            ]
        )
    )
    finger_vertical_bounds_from_eef = [
        (
            model.geom_id2name(geom_id) or f"geom_{geom_id}",
            _semantic_finger_side(body_name),
            float(
                center[2] - half_size[2] - eef_position[2]
            ),
            float(
                center[2] + half_size[2] - eef_position[2]
            ),
        )
        for geom_id, body_name, center, half_size in (
            semantic_finger_bounds
        )
    ]
    table_geom_ids = [
        geom_id
        for geom_id in _compiled_body_geom_ids(model, TABLE_BODY)
        if (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    if not table_geom_ids:
        raise RuntimeError("compiled native table collision geoms unavailable")
    table_bounds = [
        (geom_id, *_compiled_geom_world_aabb(model, data, geom_id))
        for geom_id in table_geom_ids
    ]
    table_top_z = max(
        float(center[2] + half_size[2])
        for _, center, half_size in table_bounds
    )
    finger_table_clearance_derivation = (
        _compiled_pair_set_clearance(
            model,
            [
                geom_id
                for geom_id, _, _, _ in semantic_finger_bounds
            ],
            table_geom_ids,
        )
    )
    vertical_feasibility = (
        _compiled_side_contact_eef_z_feasibility(
            rim_vertical_interval=plate_rim_vertical_interval,
            rim_center_z=plate_rim_center_z,
            finger_vertical_bounds_from_eef=(
                finger_vertical_bounds_from_eef
            ),
            table_top_z=table_top_z,
            required_finger_table_clearance_m=(
                finger_table_clearance_derivation[
                    "required_clearance_m"
                ]
            ),
        )
    )
    outside_target, side_contact_target, plan = (
        _side_contact_targets_from_compiled_bounds(
            plate_position=plate_position,
            outward_direction_xy=outward,
            contact_xy=contact_xy,
            plate_outward_support_m=plate_outward_support,
            finger_inward_extent_from_eef_m=finger_inward_extent,
            plate_rim_center_z=plate_rim_center_z,
            finger_center_z_offset_from_eef=(
                finger_center_z_offset
            ),
            outside_clearance_m=outside_clearance_m,
            side_eef_z=vertical_feasibility["selected_eef_z"],
        )
    )
    plan.update(
        {
            "plate_rim_geoms": [
                model.geom_id2name(geom_id) or ""
                for geom_id, _, _ in rim_bounds
            ],
            "finger_collision_geoms": [
                {
                    "geom": model.geom_id2name(geom_id) or "",
                    "body": body_name,
                    "semantic_side": _semantic_finger_side(body_name),
                }
                for geom_id, body_name, _, _ in semantic_finger_bounds
            ],
            "finger_side_inward_extents_from_eef_m": {
                side: float(extent)
                for side, extent in finger_side_inward_extents.items()
            },
            "dual_finger_contact_skew_m": float(
                dual_finger_contact_skew
            ),
            "plate_rim_vertical_interval_m": (
                plate_rim_vertical_interval
            ),
            "finger_vertical_bounds_from_eef_m": [
                {
                    "geom": name,
                    "semantic_side": side,
                    "interval_m": [lower, upper],
                }
                for name, side, lower, upper in (
                    finger_vertical_bounds_from_eef
                )
            ],
            "table_geoms": [
                model.geom_id2name(geom_id) or f"geom_{geom_id}"
                for geom_id in table_geom_ids
            ],
            "finger_table_clearance_derivation": (
                finger_table_clearance_derivation
            ),
            "vertical_feasibility": vertical_feasibility,
            "outside_target": outside_target.tolist(),
            "side_contact_target": side_contact_target.tolist(),
        }
    )
    return outside_target, side_contact_target, plan


def _select_reachable_compiled_side_candidate(candidates):
    """Select the least-demanding side that passed dual-finger geometry."""
    candidates = list(candidates)
    if not candidates:
        raise RuntimeError("no semantically trailing native rim side candidate")
    accepted = [
        candidate
        for candidate in candidates
        if candidate["selection_eligible"]
    ]
    if not accepted:
        raise RuntimeError(
            "no compiled trailing side passed dual-finger geometry gates: "
            f"{json.dumps(candidates, sort_keys=True)}"
        )
    return min(
        accepted,
        key=lambda candidate: (
            candidate["outside_high_action_peak"],
            candidate["outside_high_action_norm"],
            candidate["dual_finger_contact_skew_m"],
            candidate["eef_xy_distance_m"],
        ),
    )


def _compiled_trailing_side_contact_candidates(
    env,
    *,
    plate_position,
    push_direction_xy,
    eef_position,
    backoff,
    outside_clearance_m,
    plate_approach_eef_height,
    position_action_scale,
):
    """Compile and gate every cardinal side that can push toward the goal."""
    plate_position = np.asarray(plate_position, dtype=float)
    eef_position = np.asarray(eef_position, dtype=float)
    if plate_position.shape != (3,) or eef_position.shape != (3,):
        raise ValueError("plate and EEF positions must be 3-D")
    if (
        not np.isfinite(position_action_scale)
        or position_action_scale <= 0
    ):
        raise ValueError("position action scale must be positive")
    candidates = []
    for geometry in _plate_contact_candidate_diagnostics(
        plate_position[:2],
        push_direction_xy,
        eef_position[:2],
        backoff,
    ):
        if not geometry["trailing_eligible"]:
            continue
        outward = np.asarray(geometry["offset_xy"], dtype=float)
        outward /= np.linalg.norm(outward)
        outside_side, contact_target, compiled = (
            _compiled_native_side_contact_plan(
                env,
                plate_position,
                eef_position,
                outward,
                np.asarray(geometry["point_xy"], dtype=float),
                outside_clearance_m,
            )
        )
        outside_high = outside_side.copy()
        outside_high[2] = (
            plate_position[2] + plate_approach_eef_height
        )
        center_high = outside_high.copy()
        center_high[:2] = plate_position[:2]
        required_action = (
            outside_high - center_high
        ) / float(position_action_scale)
        bounded_action = _position_action(
            center_high,
            outside_high,
            0.0,
            position_action_scale,
        )[:3]
        clipped_axes = [
            int(axis)
            for axis in np.flatnonzero(
                np.abs(required_action) > 1.0 + 1e-9
            )
        ]
        dual_finger_skew = float(
            compiled["dual_finger_contact_skew_m"]
        )
        violations = []
        if dual_finger_skew > outside_clearance_m:
            violations.append(
                "dual_finger_contact_skew_exceeds_outside_clearance"
            )
        candidates.append(
            {
                **geometry,
                "outward_direction_xy": outward.tolist(),
                "center_high_target": center_high.tolist(),
                "outside_high_target": outside_high.tolist(),
                "outside_side_target": outside_side.tolist(),
                "side_contact_target": contact_target.tolist(),
                "compiled_geometry": compiled,
                "dual_finger_contact_skew_m": dual_finger_skew,
                "maximum_dual_finger_contact_skew_m": float(
                    outside_clearance_m
                ),
                "outside_high_required_action": (
                    required_action.tolist()
                ),
                "outside_high_bounded_action": bounded_action.tolist(),
                "outside_high_action_peak": float(
                    np.max(np.abs(required_action))
                ),
                "outside_high_action_norm": float(
                    np.linalg.norm(required_action)
                ),
                "outside_high_clipped_action_axes": clipped_axes,
                "outside_high_action_will_clip": bool(clipped_axes),
                "selection_violations": violations,
                "selection_eligible": not violations,
            }
        )
    selected = _select_reachable_compiled_side_candidate(candidates)
    return selected, candidates


def _body_contact_counterparts(env, body_name):
    """Describe every current MuJoCo contact involving ``body_name``."""
    model, data = env.sim.model, env.sim.data
    target_geoms = set(_compiled_body_geom_ids(model, body_name))
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


def _compiled_collision_body_inventories(env):
    """Enumerate native collision bodies on both sides of the robot boundary."""
    model = env.sim.model
    robot_bodies = set(_robot_gripper_body_names(env))
    robot_collision_bodies = set()
    nonrobot_native_collision_bodies = set()
    geom_contype = getattr(model, "geom_contype", None)
    geom_conaffinity = getattr(model, "geom_conaffinity", None)
    for geom_id in range(int(model.ngeom)):
        collision_enabled = bool(
            geom_contype is None
            or geom_conaffinity is None
            or int(geom_contype[geom_id]) != 0
            or int(geom_conaffinity[geom_id]) != 0
        )
        if not collision_enabled:
            continue
        body_name = model.body_id2name(
            int(model.geom_bodyid[geom_id])
        ) or ""
        if body_name in robot_bodies:
            robot_collision_bodies.add(body_name)
        else:
            nonrobot_native_collision_bodies.add(body_name)
    return (
        sorted(robot_collision_bodies),
        sorted(nonrobot_native_collision_bodies),
    )


def _robot_nonrobot_contact_evidence(env, *, allowed_body_pairs):
    """Reject every robot/native contact outside an exact body-pair allowlist."""
    model, data = env.sim.model, env.sim.data
    (
        robot_collision_bodies,
        nonrobot_native_collision_bodies,
    ) = _compiled_collision_body_inventories(env)
    robot_body_set = set(robot_collision_bodies)
    native_body_set = set(nonrobot_native_collision_bodies)
    normalized_allowlist = []
    for pair in allowed_body_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                "robot/native contact allowlist entries must be body pairs"
            )
        normalized = (str(pair[0]), str(pair[1]))
        if (
            not all(normalized)
            or normalized[0] not in robot_body_set
            or normalized[1] not in native_body_set
        ):
            raise RuntimeError(
                "robot/native contact allowlist is not bound to compiled "
                f"collision bodies: {normalized!r}"
            )
        normalized_allowlist.append(normalized)
    if len(set(normalized_allowlist)) != len(normalized_allowlist):
        raise RuntimeError(
            "robot/native contact allowlist contains a duplicate exact body "
            "pair"
        )
    allowed = set(normalized_allowlist)
    contacts = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = model.body_id2name(
            int(model.geom_bodyid[geom1])
        ) or ""
        body2 = model.body_id2name(
            int(model.geom_bodyid[geom2])
        ) or ""
        body1_is_robot = body1 in robot_body_set
        body2_is_robot = body2 in robot_body_set
        if body1_is_robot == body2_is_robot:
            continue
        if body1_is_robot:
            robot_geom, robot_body = geom1, body1
            native_geom, native_body = geom2, body2
        else:
            robot_geom, robot_body = geom2, body2
            native_geom, native_body = geom1, body1
        pair = (robot_body, native_body)
        contacts.append(
            {
                "contact_index": int(index),
                "robot_geom": model.geom_id2name(robot_geom) or "",
                "robot_body": robot_body,
                "native_geom": model.geom_id2name(native_geom) or "",
                "native_body": native_body,
                "allowed": pair in allowed,
            }
        )
    unexpected = [record for record in contacts if not record["allowed"]]
    return {
        "accepted": not unexpected,
        "allowlist_policy": (
            "empty during every structural precontact frame; during contact "
            "seek/calibration only exact compiled plate-finger body pairs are "
            "allowed"
        ),
        "robot_collision_bodies": robot_collision_bodies,
        "nonrobot_native_collision_bodies": (
            nonrobot_native_collision_bodies
        ),
        "allowed_body_pairs": [
            list(pair) for pair in sorted(normalized_allowlist)
        ],
        "contacts": contacts,
        "unexpected_contacts": unexpected,
    }


def _compiled_plate_finger_allowed_body_pairs(
    env, *, finger_body_names=None
):
    """Compile the only robot/native contacts allowed during plate seeking."""
    model = env.sim.model
    (
        robot_collision_bodies,
        nonrobot_native_collision_bodies,
    ) = _compiled_collision_body_inventories(env)
    robot_collision_set = set(robot_collision_bodies)
    native_collision_set = set(nonrobot_native_collision_bodies)
    if finger_body_names is None:
        finger_bodies = {
            name
            for name in robot_collision_bodies
            if _semantic_finger_side(name) in {"left", "right"}
        }
    else:
        finger_bodies = {str(name) for name in finger_body_names}
    if (
        not finger_bodies
        or not finger_bodies <= robot_collision_set
        or {
            _semantic_finger_side(name) for name in finger_bodies
        }
        != {"left", "right"}
    ):
        raise RuntimeError(
            "compiled semantic left/right finger collision bodies are "
            "unavailable for the exact plate-contact allowlist"
        )
    plate_bodies = {
        model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        for geom_id in _compiled_body_geom_ids(model, PLATE_BODY)
        if (
            (getattr(model, "geom_contype", None) is None)
            or (getattr(model, "geom_conaffinity", None) is None)
            or int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    }
    if not plate_bodies or not plate_bodies <= native_collision_set:
        raise RuntimeError(
            "compiled native plate collision bodies are unavailable for the "
            "exact contact allowlist"
        )
    return tuple(
        (finger_body, plate_body)
        for finger_body in sorted(finger_bodies)
        for plate_body in sorted(plate_bodies)
    )


def _robot_contacts_body(env, body_name):
    """Return whether any compiled robot/gripper body contacts ``body_name``."""
    return any(
        item["counterpart_is_robot_or_gripper"]
        for item in _body_contact_counterparts(env, body_name)
    )


def _plate_finger_contact_sides(env):
    """Return semantic left/right finger contacts on the native plate."""
    bodies = {
        item["counterpart_body"].lower()
        for item in _body_contact_counterparts(env, PLATE_BODY)
        if item["counterpart_is_robot_or_gripper"]
    }
    return {
        "left": any(
            _semantic_finger_side(body) == "left" for body in bodies
        ),
        "right": any(
            _semantic_finger_side(body) == "right" for body in bodies
        ),
        "contact_bodies": sorted(bodies),
    }


def _contact_depth_sample_validity(
    *,
    robot_plate_contact,
    plate_table_support,
    plate_tilt_deg,
    plate_xy_drift,
    forbidden_plate_contact_bodies,
    robot_table_contact_bodies,
    plate_linear_speed,
    plate_angular_speed,
    require_robot_plate_contact,
    require_stable,
    maximum_plate_tilt_deg,
    maximum_plate_xy_drift,
    maximum_linear_speed,
    maximum_angular_speed,
):
    """Fail-closed semantic gates for one measured depth-calibration sample."""
    violations = []
    measured_values = (
        plate_tilt_deg,
        plate_xy_drift,
        plate_linear_speed,
        plate_angular_speed,
    )
    if not all(np.isfinite(value) for value in measured_values):
        violations.append("nonfinite_plate_state")
    if require_robot_plate_contact and not robot_plate_contact:
        violations.append("robot_plate_contact_lost")
    if not plate_table_support:
        violations.append("plate_table_support_lost")
    if plate_tilt_deg > maximum_plate_tilt_deg:
        violations.append("plate_tilt_exceeded")
    if plate_xy_drift > maximum_plate_xy_drift:
        violations.append("plate_xy_drift_exceeded")
    if forbidden_plate_contact_bodies:
        violations.append("forbidden_plate_contact")
    if robot_table_contact_bodies:
        violations.append("forbidden_robot_table_contact")
    if require_stable and plate_linear_speed > maximum_linear_speed:
        violations.append("plate_linear_speed_exceeded")
    if require_stable and plate_angular_speed > maximum_angular_speed:
        violations.append("plate_angular_speed_exceeded")
    return {
        "accepted": not violations,
        "violations": violations,
        "require_robot_plate_contact": bool(
            require_robot_plate_contact
        ),
        "require_stable": bool(require_stable),
    }


def _contact_depth_state_diagnostics(
    env,
    plate_reference_position,
    *,
    allowed_robot_nonrobot_body_pairs,
    require_robot_plate_contact=True,
    require_stable,
    maximum_plate_tilt_deg,
    maximum_plate_xy_drift,
    maximum_linear_speed,
    maximum_angular_speed,
):
    """Measure physical and collision gates at the policy-observed state."""
    plate_position = body_pose(env, PLATE_BODY)[0]
    plate_tilt = body_tilt_deg(env, PLATE_BODY)
    plate_linear, plate_angular = body_velocity(env, PLATE_BODY)
    plate_contacts = _body_contact_counterparts(env, PLATE_BODY)
    forbidden_plate_contacts = sorted(
        {
            item["counterpart_body"]
            for item in plate_contacts
            if (
                item["counterpart_body"] != TABLE_BODY
                and not item["counterpart_is_robot_or_gripper"]
            )
        }
    )
    table_contacts = _body_contact_counterparts(env, TABLE_BODY)
    robot_table_contacts = sorted(
        {
            item["counterpart_body"]
            for item in table_contacts
            if item["counterpart_is_robot_or_gripper"]
        }
    )
    plate_table_support = any(
        item["counterpart_body"] == TABLE_BODY for item in plate_contacts
    )
    robot_plate_contact = any(
        item["counterpart_is_robot_or_gripper"]
        for item in plate_contacts
    )
    plate_xy_drift = float(
        np.linalg.norm(
            plate_position[:2]
            - np.asarray(plate_reference_position, dtype=float)[:2]
        )
    )
    validity = _contact_depth_sample_validity(
        robot_plate_contact=robot_plate_contact,
        plate_table_support=plate_table_support,
        plate_tilt_deg=plate_tilt,
        plate_xy_drift=plate_xy_drift,
        forbidden_plate_contact_bodies=forbidden_plate_contacts,
        robot_table_contact_bodies=robot_table_contacts,
        plate_linear_speed=plate_linear,
        plate_angular_speed=plate_angular,
        require_robot_plate_contact=require_robot_plate_contact,
        require_stable=require_stable,
        maximum_plate_tilt_deg=maximum_plate_tilt_deg,
        maximum_plate_xy_drift=maximum_plate_xy_drift,
        maximum_linear_speed=maximum_linear_speed,
        maximum_angular_speed=maximum_angular_speed,
    )
    robot_nonrobot_contact_gate = _robot_nonrobot_contact_evidence(
        env,
        allowed_body_pairs=allowed_robot_nonrobot_body_pairs,
    )
    if not robot_nonrobot_contact_gate["accepted"]:
        validity["accepted"] = False
        validity["violations"].append(
            "unexpected_robot_nonrobot_contact"
        )
    return {
        **validity,
        "plate_position": plate_position.tolist(),
        "plate_tilt_deg": plate_tilt,
        "plate_xy_drift_m": plate_xy_drift,
        "plate_linear_speed_mps": plate_linear,
        "plate_angular_speed_radps": plate_angular,
        "robot_plate_contact": robot_plate_contact,
        "plate_table_support": plate_table_support,
        "forbidden_plate_contact_bodies": forbidden_plate_contacts,
        "robot_table_contact_bodies": robot_table_contacts,
        "plate_contacts": plate_contacts,
        "robot_nonrobot_contact_gate": robot_nonrobot_contact_gate,
    }


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


def _seek_stable_plate_contact(
    rollout,
    env,
    args,
    *,
    gripper,
    outside_high_target,
    outside_side_target,
    contact_target,
    geometry,
    source,
    diagnostics,
):
    """Descend outside the plate, then establish two-finger side contact."""
    plate_reference = body_pose(env, PLATE_BODY)[0].copy()
    samples = []
    native_action_spec = _native_osc_action_spec_evidence(env)
    plate_finger_allowed_body_pairs = (
        _compiled_plate_finger_allowed_body_pairs(
            env,
            finger_body_names=[
                record["body"]
                for record in geometry["finger_collision_geoms"]
            ],
        )
    )
    structural_seek_context = {
        "native_osc_action_spec": native_action_spec,
        "structural_precontact_robot_native_allowlist": [],
        "contact_seek_plate_finger_allowed_body_pairs": [
            list(pair) for pair in plate_finger_allowed_body_pairs
        ],
    }

    def capture(
        stage,
        index,
        require_contact,
        require_stable,
        extra=None,
    ):
        finger_contact_sides = _plate_finger_contact_sides(env)
        sample = {
            "stage": stage,
            "index": int(index),
            "eef_position": np.asarray(
                rollout.obs["robot0_eef_pos"], dtype=float
            ).tolist(),
            **_contact_depth_state_diagnostics(
                env,
                plate_reference,
                allowed_robot_nonrobot_body_pairs=(
                    plate_finger_allowed_body_pairs
                    if stage
                    in {
                        "bounded_lateral_contact_seek",
                        "stable_contact_confirmation",
                    }
                    else ()
                ),
                require_robot_plate_contact=require_contact,
                require_stable=require_stable,
                maximum_plate_tilt_deg=(
                    args.max_contact_calibration_plate_tilt_deg
                ),
                maximum_plate_xy_drift=(
                    args.max_contact_calibration_plate_xy_drift
                ),
                maximum_linear_speed=args.max_stable_linear_speed,
                maximum_angular_speed=args.max_stable_angular_speed,
            ),
            "finger_contact_sides": finger_contact_sides,
        }
        if extra:
            sample.update(extra)
        samples.append(sample)
        if (
            stage.startswith("outside_")
            and sample["robot_plate_contact"]
        ):
            sample["accepted"] = False
            sample["violations"].append(
                "robot_plate_contact_before_lateral_seek"
            )
        if (
            stage == "stable_contact_confirmation"
            and not (
                finger_contact_sides["left"]
                and finger_contact_sides["right"]
            )
        ):
            sample["accepted"] = False
            sample["violations"].append(
                "two_finger_side_contact_not_sustained"
            )
        if not sample["accepted"]:
            failure = {
                "source": source,
                "outside_high_target": np.asarray(
                    outside_high_target, dtype=float
                ).tolist(),
                "outside_side_target": np.asarray(
                    outside_side_target, dtype=float
                ).tolist(),
                "contact_target": np.asarray(
                    contact_target, dtype=float
                ).tolist(),
                "compiled_geometry": geometry,
                "structural_seek_context": structural_seek_context,
                "samples": samples,
                "scene": diagnostics(),
            }
            raise RuntimeError(
                "bounded plate-contact seek violated a physical or collision "
                "gate: "
                f"{json.dumps(failure, sort_keys=True)}"
            )
        return sample

    initial_eef = np.asarray(
        rollout.obs["robot0_eef_pos"], dtype=float
    ).copy()
    live_plate_center = body_pose(env, PLATE_BODY)[0]
    center_xy_error = float(
        np.linalg.norm(initial_eef[:2] - live_plate_center[:2])
    )
    if center_xy_error > args.position_tolerance:
        raise RuntimeError(
            "compiled overhead staging did not begin above the native plate "
            "center within the unchanged waypoint tolerance: "
            f"source={source} center_xy_error_m={center_xy_error} "
            f"tolerance_m={args.position_tolerance} "
            f"eef={initial_eef.tolist()} "
            f"plate={live_plate_center.tolist()}"
        )
    maximum_controller_world_step = float(
        args.position_action_scale
        * args.plate_contact_seek_max_translation_action
    )
    overhead_staging_z, overhead_staging_geometry = (
        _compiled_overhead_staging_geometry(
            env,
            start_eef_position=initial_eef,
            one_step_vertical_reserve_m=(
                maximum_controller_world_step
            ),
        )
    )
    expected_overhead_pair_count = 55
    if len(overhead_staging_geometry["pairs"]) != (
        expected_overhead_pair_count
    ):
        raise RuntimeError(
            "native L3-A3 compiled overhead pair inventory changed: "
            f"expected={expected_overhead_pair_count} "
            f"observed={len(overhead_staging_geometry['pairs'])}"
        )
    if overhead_staging_z <= float(outside_side_target[2]):
        raise RuntimeError(
            "compiled overhead staging Z does not remain above the native "
            "side-contact corridor: "
            f"source={source} overhead_z={overhead_staging_z} "
            f"side_z={float(outside_side_target[2])} "
            f"geometry={json.dumps(overhead_staging_geometry, sort_keys=True)}"
        )

    initial_outside_side_guard = _live_outside_side_guard(env, geometry)
    overhead_outside_high_target = np.asarray(
        outside_high_target, dtype=float
    ).copy()
    overhead_outside_high_target[2] = overhead_staging_z
    (
        corridor_high_target,
        corridor_side_target,
        vertical_staging_corridor,
    ) = _compiled_vertical_staging_corridor(
        outside_high_target=overhead_outside_high_target,
        outside_side_target=outside_side_target,
        geometry=geometry,
        required_outside_clearance_m=initial_outside_side_guard[
            "required_outside_clearance_m"
        ],
        position_action_scale=args.position_action_scale,
        maximum_translation_action=(
            args.plate_contact_seek_max_translation_action
        ),
    )
    overhead_horizontal_travel = float(
        np.linalg.norm(
            corridor_high_target[:2] - initial_eef[:2]
        )
    )
    native_low = np.asarray(native_action_spec["low"], dtype=float)
    native_high = np.asarray(native_action_spec["high"], dtype=float)
    strict_native_high_lateral_action_norm_bound = float(
        np.nextafter(
            min(
                -native_low[0],
                native_high[0],
                -native_low[1],
                native_high[1],
            ),
            0.0,
        )
    )
    maximum_native_high_lateral_world_step = float(
        args.position_action_scale
        * strict_native_high_lateral_action_norm_bound
    )
    if maximum_native_high_lateral_world_step <= 0.0:
        raise RuntimeError(
            "runtime native action spec has no positive high-lateral action "
            "capacity"
        )
    total_structural_geometric_travel = float(
        overhead_staging_geometry["vertical_sweep_distance_m"]
        + overhead_horizontal_travel
        + vertical_staging_corridor["vertical_staging_travel_m"]
        + vertical_staging_corridor["fixed_z_lateral_travel_m"]
    )
    high_lateral_action_count_lower_bound = float(
        overhead_horizontal_travel
        / maximum_native_high_lateral_world_step
    )
    total_structural_action_lower_bound = float(
        high_lateral_action_count_lower_bound
    )
    minimum_full_scale_actions_from_geometry = int(
        np.ceil(total_structural_action_lower_bound)
    )
    if minimum_full_scale_actions_from_geometry > args.max_waypoint_steps:
        raise RuntimeError(
            "compiled overhead route geometric lower bound alone exceeds "
            "the unchanged structural waypoint hard loop: "
            f"source={source} lower_bound_action_equivalents="
            f"{total_structural_action_lower_bound} "
            f"minimum_full_scale_actions="
            f"{minimum_full_scale_actions_from_geometry} "
            f"maximum_steps={args.max_waypoint_steps}"
        )
    overhead_staging_geometry.update(
        {
            "center_xy_error_m": center_xy_error,
            "maximum_center_xy_error_m": float(
                args.position_tolerance
            ),
            "expected_compiled_overhead_pair_count": (
                expected_overhead_pair_count
            ),
            "native_center_high_start": initial_eef.tolist(),
            "corridor_adaptive_descent_target": (
                corridor_high_target.tolist()
            ),
            "structural_route_order": [
                "native_center_high_pure_xy",
                "corridor_xy_adaptive_pure_z_descent",
                "vertical_tail_brake_and_zero_confirmation",
                "live_corridor_entry_or_xy_drift_correction",
                "vertical_side_corridor_and_contact",
            ],
            "horizontal_sweep_formula": (
                "from the exact native center-high first-policy state, command "
                "pure XY with zero Z/rotation to the compiled corridor XY; "
                "derive each translation-action norm from the strict runtime "
                "native XY bound and all 55 live pairs' current clearance "
                "minus strict+base8, using position_action_scale times norm "
                "as the worst-case downward tail plus an inward numerical "
                "guard; remeasure post-action base8 and the empty structural "
                "robot/native contact allowlist on every frame. The unchanged "
                "0.10 bound remains exclusive to post-descent correction and "
                "contact motion"
            ),
            "measurement_scope": (
                "live pre/post world-AABB and contact observations with the "
                "unchanged 8/16 mm action envelopes; internal controller "
                "substeps are not directly measured"
            ),
            "horizontal_sweep_distance_m": overhead_horizontal_travel,
            "maximum_controller_world_step_m": (
                maximum_controller_world_step
            ),
            "strict_native_high_lateral_action_norm_bound": (
                strict_native_high_lateral_action_norm_bound
            ),
            "maximum_native_high_lateral_world_step_m": (
                maximum_native_high_lateral_world_step
            ),
            "total_structural_geometric_travel_m": (
                total_structural_geometric_travel
            ),
            "total_structural_full_scale_action_lower_bound": (
                total_structural_action_lower_bound
            ),
            "high_lateral_action_count_lower_bound": (
                high_lateral_action_count_lower_bound
            ),
            "minimum_full_scale_actions_from_geometry": (
                minimum_full_scale_actions_from_geometry
            ),
            "geometric_action_count_scope": (
                "diagnostic lower bound only, using only required high-XY "
                "travel divided by the strict runtime-native maximum world "
                "step; later stages are deliberately excluded because "
                "controller coupling can change their remaining travel. "
                "Adaptive responses, brakes, zero confirmation, and XY drift "
                "correction are enforced at runtime by the unchanged 180-step hard loop"
            ),
            "maximum_structural_waypoint_steps": int(
                args.max_waypoint_steps
            ),
        }
    )
    structural_seek_context.update(
        {
            "compiled_overhead_staging_geometry": (
                overhead_staging_geometry
            ),
            "vertical_staging_corridor": vertical_staging_corridor,
            "corridor_high_target": corridor_high_target.tolist(),
            "corridor_side_target": corridor_side_target.tolist(),
            "structural_route_order": list(
                overhead_staging_geometry["structural_route_order"]
            ),
        }
    )

    latest_outside_side_guard = initial_outside_side_guard
    latest_overhead_guard = _live_compiled_overhead_guard(
        env, overhead_staging_geometry
    )
    latest_overhead_lateral_buffer = (
        _overhead_lateral_buffer_evidence(
            latest_overhead_guard,
            worst_case_controller_world_step_m=(
                maximum_controller_world_step
            ),
        )
    )
    structural_seek_context["overhead_lateral_buffer_derivation"] = (
        latest_overhead_lateral_buffer
    )
    outside_side_guard_checks = 1
    overhead_guard_checks = 1
    outside_side_motion_steps = 0
    outside_side_feedback_steps = []
    lateral_settle_state = None
    structural_stage = "overhead_high_corridor_lateral"
    overhead_horizontal_z = float(initial_eef[2])
    latest_vertical_step_progress_m = 0.0
    lateral_resume_stage = None
    vertical_tail_events = []
    fixed_safe_z = None
    structural_stage_action_counts = {
        "overhead_high_corridor_lateral": 0,
        "overhead_corridor_descent": 0,
        "vertical_tail_brake": 0,
        "lateral_rebuffer_brake": 0,
        "vertical_tail_zero_confirmation": 0,
        "overhead_post_descent_corridor_lateral": 0,
        "vertical_corridor_descent": 0,
        "vertical_corridor_settle": 0,
        "fixed_safe_z_lateral_approach": 0,
    }
    overhead_lateral_stages = {
        "overhead_high_corridor_lateral",
        "overhead_post_descent_corridor_lateral",
    }
    fixed_buffer_lateral_stages = {
        "overhead_post_descent_corridor_lateral",
    }
    overhead_route_stages = {
        *overhead_lateral_stages,
        "overhead_corridor_descent",
        "vertical_tail_brake",
        "lateral_rebuffer_brake",
        "vertical_tail_zero_confirmation",
    }
    capture(
        "outside_native_center_high_start",
        0,
        False,
        True,
        extra={
            "outside_side_guard": latest_outside_side_guard,
            "compiled_overhead_guard": latest_overhead_guard,
            "overhead_lateral_buffer": (
                _overhead_lateral_buffer_frame_summary(
                    latest_overhead_lateral_buffer
                )
            ),
        },
    )
    for guard_step in range(1, args.max_waypoint_steps + 1):
        lateral_pre_action_interlock = None
        pre_action_guard = latest_outside_side_guard
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        )
        if (
            structural_stage == "vertical_tail_brake"
            and latest_vertical_step_progress_m is not None
            and latest_vertical_step_progress_m >= 0.0
            and latest_overhead_lateral_buffer["accepted"]
        ):
            vertical_tail_events.append(
                {
                    "guard_step": int(guard_step),
                    "event": "brake_pre_action_recovered_to_zero_confirmation",
                    "measured_vertical_step_progress_m": (
                        latest_vertical_step_progress_m
                    ),
                    "minimum_lateral_entry_buffer_surplus_m": (
                        latest_overhead_lateral_buffer[
                            "minimum_lateral_entry_buffer_surplus_m"
                        ]
                    ),
                }
            )
            structural_stage = "vertical_tail_zero_confirmation"
        if (
            structural_stage == "lateral_rebuffer_brake"
            and latest_vertical_step_progress_m is not None
            and latest_vertical_step_progress_m >= 0.0
            and latest_overhead_lateral_buffer["accepted"]
        ):
            vertical_tail_events.append(
                {
                    "guard_step": int(guard_step),
                    "event": (
                        "lateral_rebuffer_pre_action_recovered_directly_to_xy"
                    ),
                    "measured_vertical_step_progress_m": (
                        latest_vertical_step_progress_m
                    ),
                    "minimum_lateral_entry_buffer_surplus_m": (
                        latest_overhead_lateral_buffer[
                            "minimum_lateral_entry_buffer_surplus_m"
                        ]
                    ),
                }
            )
            if lateral_resume_stage not in fixed_buffer_lateral_stages:
                raise RuntimeError(
                    "lateral rebuffer has no proved post-descent correction "
                    "resume "
                    f"stage: {lateral_resume_stage!r}"
                )
            structural_stage = lateral_resume_stage
            overhead_horizontal_z = float(current_eef[2])
        if structural_stage in fixed_buffer_lateral_stages:
            lateral_pre_action_interlock = (
                _overhead_lateral_interlock_evidence(
                    latest_overhead_lateral_buffer,
                    measured_vertical_step_progress_m=(
                        latest_vertical_step_progress_m
                    ),
                )
            )
            if lateral_pre_action_interlock[
                "requires_positive_z_brake"
            ]:
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": "lateral_pre_action_buffer_interlock_to_brake",
                        **lateral_pre_action_interlock,
                    }
                )
                lateral_resume_stage = structural_stage
                structural_stage = "lateral_rebuffer_brake"
        if structural_stage == "fixed_safe_z_lateral_approach":
            lateral_error = float(
                np.linalg.norm(
                    current_eef[:2]
                    - np.asarray(outside_side_target, dtype=float)[:2]
                )
            )
            if lateral_error <= args.position_tolerance:
                if not pre_action_guard["accepted"]:
                    raise RuntimeError(
                        "fixed-safe-Z lateral approach reached the compiled "
                        "outside XY target without sustaining the live "
                        "outside-side guard: "
                        f"source={source} guard_step={guard_step} "
                        f"guard={json.dumps(pre_action_guard, sort_keys=True)} "
                        f"samples={json.dumps(samples, sort_keys=True)}"
                    )
                break
        stage_before_action = structural_stage
        prepared_high_lateral_action = None
        prepared_high_lateral_envelope = None
        if stage_before_action == "overhead_high_corridor_lateral":
            (
                prepared_high_lateral_action,
                prepared_high_lateral_envelope,
            ) = _compiled_adaptive_high_lateral_action(
                current_eef=current_eef,
                lateral_target_xy=corridor_high_target[:2],
                overhead_guard=latest_overhead_guard,
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                native_action_spec=native_action_spec,
                expected_pair_count=expected_overhead_pair_count,
            )
        pre_action_overhead_route_authorization = None
        if stage_before_action in overhead_route_stages:
            pre_action_overhead_route_authorization = (
                _overhead_route_frame_authorization_evidence(
                    outside_side_guard=pre_action_guard,
                    overhead_guard=latest_overhead_guard,
                    overhead_lateral_buffer=(
                        latest_overhead_lateral_buffer
                    ),
                    compiled_pairs=overhead_staging_geometry["pairs"],
                    expected_pair_count=expected_overhead_pair_count,
                    require_lateral_buffer=(
                        stage_before_action in fixed_buffer_lateral_stages
                    ),
                    adaptive_high_lateral_envelope=(
                        prepared_high_lateral_envelope
                    ),
                )
            )
        if structural_stage == "overhead_corridor_descent":
            action, path_control = (
                _compiled_adaptive_vertical_descent_action(
                    current_eef=current_eef,
                    target_z=overhead_staging_z,
                    overhead_guard=latest_overhead_guard,
                    gripper=gripper,
                    position_action_scale=args.position_action_scale,
                    native_action_spec=native_action_spec,
                    expected_pair_count=expected_overhead_pair_count,
                )
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "compiled_adaptive_vertical_action_envelope": path_control,
            }
        elif structural_stage == "vertical_tail_brake":
            action, path_control = _fixed_xy_vertical_approach_action(
                current_eef=current_eef,
                target_z=float(
                    current_eef[2] + maximum_controller_world_step
                ),
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                maximum_translation_action=(
                    args.plate_contact_seek_max_translation_action
                ),
            )
            if action[2] <= 0.0:
                raise RuntimeError(
                    "event-driven vertical-tail brake failed to command "
                    "strictly positive Z"
                )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "fixed_xy_positive_z_brake_control": path_control,
                "pre_action_measured_vertical_step_progress_m": (
                    latest_vertical_step_progress_m
                ),
                "pre_action_overhead_lateral_buffer": (
                    _overhead_lateral_buffer_frame_summary(
                        latest_overhead_lateral_buffer
                    )
                ),
                **(
                    {
                        "lateral_pre_action_interlock": (
                            lateral_pre_action_interlock
                        )
                    }
                    if lateral_pre_action_interlock is not None
                    else {}
                ),
            }
        elif structural_stage == "lateral_rebuffer_brake":
            action, path_control = (
                _compiled_adaptive_lateral_rebuffer_action(
                    current_eef=current_eef,
                    overhead_guard=latest_overhead_guard,
                    overhead_lateral_buffer=(
                        latest_overhead_lateral_buffer
                    ),
                    outside_side_guard=pre_action_guard,
                    gripper=gripper,
                    position_action_scale=args.position_action_scale,
                    native_action_spec=native_action_spec,
                    expected_pair_count=expected_overhead_pair_count,
                    worst_case_controller_world_step_m=(
                        maximum_controller_world_step
                    ),
                )
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "compiled_adaptive_lateral_rebuffer_envelope": path_control,
                "pre_action_measured_vertical_step_progress_m": (
                    latest_vertical_step_progress_m
                ),
                "pre_action_overhead_guard": latest_overhead_guard,
                "pre_action_overhead_lateral_buffer": (
                    _overhead_lateral_buffer_frame_summary(
                        latest_overhead_lateral_buffer
                    )
                ),
                "pre_action_outside_side_guard": pre_action_guard,
                **(
                    {
                        "lateral_pre_action_interlock": (
                            lateral_pre_action_interlock
                        )
                    }
                    if lateral_pre_action_interlock is not None
                    else {}
                ),
            }
        elif structural_stage == "vertical_tail_zero_confirmation":
            action = np.zeros(7, dtype=float)
            action[-1] = float(gripper)
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "zero_z_confirmation": {
                    "formula": (
                        "after measured dz is nonnegative and every compiled "
                        "pair exceeds the 16 mm lateral-entry buffer, issue "
                        "one zero-translation action and require its measured "
                        "dz to remain nonnegative; this is an event response, "
                        "not a fixed-N settling window"
                    ),
                    "commanded_xy_action": action[:2].tolist(),
                    "commanded_z_action": float(action[2]),
                    "pre_action_measured_vertical_step_progress_m": (
                        latest_vertical_step_progress_m
                    ),
                    "pre_action_overhead_lateral_buffer": (
                        _overhead_lateral_buffer_frame_summary(
                            latest_overhead_lateral_buffer
                        )
                    ),
                },
            }
        elif structural_stage == "overhead_high_corridor_lateral":
            if (
                prepared_high_lateral_action is None
                or prepared_high_lateral_envelope is None
            ):
                raise RuntimeError(
                    "high-lateral action lacks its live compiled adaptive "
                    "envelope"
                )
            action = prepared_high_lateral_action
            path_control = prepared_high_lateral_envelope
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "compiled_adaptive_high_lateral_action_envelope": (
                    path_control
                ),
                "lateral_route_phase": "native_center_high_first",
                "overhead_horizontal_z_m": float(overhead_horizontal_z),
                "pre_action_measured_vertical_step_progress_m": (
                    latest_vertical_step_progress_m
                ),
                "fixed_buffer16_used_for_action_authorization": False,
                "pre_action_overhead_guard": latest_overhead_guard,
            }
        elif structural_stage == "overhead_post_descent_corridor_lateral":
            action, path_control = _fixed_z_lateral_approach_action(
                current_eef=current_eef,
                lateral_target_xy=corridor_high_target[:2],
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                maximum_translation_action=(
                    args.plate_contact_seek_max_translation_action
                ),
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "fixed_z_lateral_path_control": path_control,
                "lateral_route_phase": "post_descent_xy_drift_correction",
                "overhead_horizontal_z_m": float(
                    overhead_horizontal_z
                ),
                "pre_action_measured_vertical_step_progress_m": (
                    latest_vertical_step_progress_m
                ),
                "pre_action_overhead_lateral_buffer": (
                    _overhead_lateral_buffer_frame_summary(
                        latest_overhead_lateral_buffer
                    )
                ),
                "lateral_pre_action_interlock": (
                    lateral_pre_action_interlock
                ),
                "fixed_buffer16_used_for_action_authorization": True,
            }
        elif structural_stage == "vertical_corridor_descent":
            maximum_descent = max(
                0.0,
                float(current_eef[2] - corridor_side_target[2]),
            )
            if maximum_descent <= 0.0 and not pre_action_guard["accepted"]:
                raise RuntimeError(
                    "vertical staging reached or crossed its compiled safe "
                    "Z without live rim coverage: "
                    f"source={source} guard_step={guard_step} "
                    f"guard={json.dumps(pre_action_guard, sort_keys=True)} "
                    f"samples={json.dumps(samples, sort_keys=True)}"
                )
            action, path_control = (
                _constraint_prioritized_outside_descent_action(
                    current_eef=current_eef,
                    outside_side_target=corridor_side_target,
                    outward_direction_xy=geometry[
                        "outward_direction_xy"
                    ],
                    maximum_descent_m=maximum_descent,
                    gripper=gripper,
                    position_action_scale=args.position_action_scale,
                    maximum_translation_action=(
                        args.plate_contact_seek_max_translation_action
                    ),
                )
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "descent_path_control": path_control,
            }
        elif structural_stage == "vertical_corridor_settle":
            if "trigger_step_response" in lateral_settle_state:
                previous_vertical_step_progress = float(
                    lateral_settle_state["trigger_step_response"][
                        "vertical_step_progress_m"
                    ]
                )
            else:
                previous_vertical_step_progress = float(
                    lateral_settle_state[
                        "vertical_step_progress_m"
                    ]
                )
            active_brake = previous_vertical_step_progress < 0.0
            action, path_control = (
                _constraint_prioritized_outside_descent_action(
                    current_eef=current_eef,
                    outside_side_target=corridor_side_target,
                    outward_direction_xy=geometry[
                        "outward_direction_xy"
                    ],
                    maximum_descent_m=0.0,
                    gripper=gripper,
                    position_action_scale=args.position_action_scale,
                    maximum_translation_action=(
                        args.plate_contact_seek_max_translation_action
                    ),
                    active_positive_z_brake=active_brake,
                )
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "descent_path_control": path_control,
                "previous_settle_vertical_step_progress_m": (
                    previous_vertical_step_progress
                ),
                "active_positive_z_brake_requested": active_brake,
                "active_positive_z_brake_commanded": bool(
                    action[2] > 0.0
                ),
                "commanded_positive_z_brake_action": float(
                    max(0.0, action[2])
                ),
            }
        elif structural_stage == "fixed_safe_z_lateral_approach":
            action, path_control = _fixed_z_lateral_approach_action(
                current_eef=current_eef,
                lateral_target_xy=np.asarray(
                    outside_side_target, dtype=float
                )[:2],
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                maximum_translation_action=(
                    args.plate_contact_seek_max_translation_action
                ),
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "fixed_z_lateral_path_control": path_control,
                "fixed_safe_z_m": float(fixed_safe_z),
            }
        else:
            raise RuntimeError(
                f"unknown structural outside-side stage {structural_stage!r}"
            )
        if pre_action_overhead_route_authorization is not None:
            feedback["pre_action_overhead_route_authorization"] = (
                pre_action_overhead_route_authorization
            )
        rollout.advance(action, "task")
        outside_side_motion_steps += 1
        structural_stage_action_counts[stage_before_action] += 1
        after_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        )
        measured_vertical_step_progress_m = float(
            after_eef[2] - current_eef[2]
        )
        latest_vertical_step_progress_m = (
            measured_vertical_step_progress_m
        )
        feedback["measured_vertical_step_progress_m"] = (
            measured_vertical_step_progress_m
        )
        latest_outside_side_guard = _live_outside_side_guard(
            env, geometry
        )
        outside_side_guard_checks += 1
        feedback["post_action_guard"] = latest_outside_side_guard
        if stage_before_action in overhead_route_stages:
            latest_overhead_guard = _live_compiled_overhead_guard(
                env, overhead_staging_geometry
            )
            overhead_guard_checks += 1
            feedback["post_action_overhead_guard"] = (
                latest_overhead_guard
            )
            latest_overhead_lateral_buffer = (
                _overhead_lateral_buffer_evidence(
                    latest_overhead_guard,
                    worst_case_controller_world_step_m=(
                        maximum_controller_world_step
                    ),
                )
            )
            feedback["post_action_overhead_lateral_buffer"] = (
                _overhead_lateral_buffer_frame_summary(
                    latest_overhead_lateral_buffer
                )
            )
        current_step_response = (
            _outside_side_step_response_evidence(
                before_guard=pre_action_guard,
                after_guard=latest_outside_side_guard,
                before_eef=current_eef,
                after_eef=after_eef,
            )
        )
        feedback["step_response"] = current_step_response
        feedback["vertical_staging_corridor"] = (
            vertical_staging_corridor
        )
        feedback["compiled_overhead_staging_reference"] = {
            "selected_eef_z": float(overhead_staging_z),
            "one_step_vertical_reserve_m": float(
                maximum_controller_world_step
            ),
        }
        feedback["stage_before_action"] = stage_before_action
        lateral_settle_progress = None
        lateral_post_action_interlock = None
        corridor_entry_after_action = None
        if stage_before_action in fixed_buffer_lateral_stages:
            lateral_post_action_interlock = (
                _overhead_lateral_interlock_evidence(
                    latest_overhead_lateral_buffer,
                    measured_vertical_step_progress_m=(
                        measured_vertical_step_progress_m
                    ),
                )
            )
            feedback["lateral_post_action_interlock"] = (
                lateral_post_action_interlock
            )
        if stage_before_action in overhead_lateral_stages:
            feedback["corridor_lateral_error_m"] = float(
                np.linalg.norm(after_eef[:2] - corridor_high_target[:2])
            )
        if stage_before_action == "overhead_high_corridor_lateral":
            if measured_vertical_step_progress_m < 0.0:
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": "adaptive_high_lateral_negative_tail_recorded",
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "post_action_base8_accepted": bool(
                            latest_overhead_guard["accepted"]
                        ),
                    }
                )
            corridor_entry_after_action = (
                _overhead_corridor_entry_evidence(
                    current_eef=after_eef,
                    corridor_high_target=corridor_high_target,
                    outside_side_guard=latest_outside_side_guard,
                    overhead_guard=latest_overhead_guard,
                    overhead_lateral_buffer=(
                        latest_overhead_lateral_buffer
                    ),
                    position_tolerance=args.position_tolerance,
                    strict_corridor_entry_clearance_m=(
                        vertical_staging_corridor[
                            "strict_corridor_entry_clearance_m"
                        ]
                    ),
                    require_lateral_buffer=False,
                )
            )
            feedback["corridor_entry_after_high_lateral"] = (
                corridor_entry_after_action
            )
            if corridor_entry_after_action["accepted"]:
                structural_stage = "overhead_corridor_descent"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "native_center_high_lateral_complete_to_"
                            "adaptive_corridor_descent"
                        ),
                        **corridor_entry_after_action,
                    }
                )
        elif stage_before_action == "overhead_corridor_descent":
            if after_eef[2] <= overhead_staging_z + args.position_tolerance:
                structural_stage = "vertical_tail_brake"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "corridor_xy_adaptive_descent_complete_to_brake"
                        ),
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                    }
                )
        elif stage_before_action == "vertical_tail_brake":
            if (
                measured_vertical_step_progress_m >= 0.0
                and latest_overhead_lateral_buffer["accepted"]
            ):
                structural_stage = "vertical_tail_zero_confirmation"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": "brake_recovered_to_zero_confirmation",
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                    }
                )
        elif stage_before_action == "lateral_rebuffer_brake":
            if (
                measured_vertical_step_progress_m >= 0.0
                and latest_overhead_lateral_buffer["accepted"]
            ):
                if lateral_resume_stage not in fixed_buffer_lateral_stages:
                    raise RuntimeError(
                        "lateral rebuffer recovered without a proved post-"
                        "descent correction resume stage"
                    )
                structural_stage = lateral_resume_stage
                overhead_horizontal_z = float(after_eef[2])
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "lateral_rebuffer_recovered_directly_to_xy"
                        ),
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                    }
                )
            else:
                structural_stage = "lateral_rebuffer_brake"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": "lateral_rebuffer_continues_fail_closed",
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                    }
                )
        elif stage_before_action == "vertical_tail_zero_confirmation":
            if (
                measured_vertical_step_progress_m >= 0.0
                and latest_overhead_lateral_buffer["accepted"]
            ):
                post_descent_corridor_entry = (
                    _overhead_corridor_entry_evidence(
                        current_eef=after_eef,
                        corridor_high_target=corridor_high_target,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                        position_tolerance=args.position_tolerance,
                        strict_corridor_entry_clearance_m=(
                            vertical_staging_corridor[
                                "strict_corridor_entry_clearance_m"
                            ]
                        ),
                    )
                )
                feedback["post_descent_corridor_entry"] = (
                    post_descent_corridor_entry
                )
                if post_descent_corridor_entry["accepted"]:
                    structural_stage = "vertical_corridor_descent"
                    vertical_tail_events.append(
                        {
                            "guard_step": int(guard_step),
                            "event": (
                                "zero_confirmation_passed_directly_to_"
                                "vertical_corridor"
                            ),
                            "measured_vertical_step_progress_m": (
                                measured_vertical_step_progress_m
                            ),
                            **post_descent_corridor_entry,
                        }
                    )
                elif (
                    latest_overhead_guard["accepted"]
                    and latest_overhead_lateral_buffer["accepted"]
                ):
                    structural_stage = "overhead_post_descent_corridor_lateral"
                    overhead_horizontal_z = float(after_eef[2])
                    vertical_tail_events.append(
                        {
                            "guard_step": int(guard_step),
                            "event": (
                                "zero_confirmation_passed_but_live_corridor_"
                                "xy_requires_overhead_correction"
                            ),
                            "measured_vertical_step_progress_m": (
                                measured_vertical_step_progress_m
                            ),
                            **post_descent_corridor_entry,
                        }
                    )
                else:
                    raise RuntimeError(
                        "post-descent corridor correction lacks the live "
                        "compiled overhead base8/buffer16 envelope"
                    )
            else:
                structural_stage = "vertical_tail_brake"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": "zero_confirmation_failed_to_brake",
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                    }
                )
        elif stage_before_action == "overhead_post_descent_corridor_lateral":
            if lateral_post_action_interlock[
                "requires_positive_z_brake"
            ]:
                lateral_resume_stage = stage_before_action
                structural_stage = "lateral_rebuffer_brake"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "post_descent_lateral_buffer_interlock_to_"
                            "rebuffer"
                        ),
                        **lateral_post_action_interlock,
                    }
                )
            else:
                if lateral_post_action_interlock[
                    "negative_vertical_tail_observed"
                ]:
                    vertical_tail_events.append(
                        {
                            "guard_step": int(guard_step),
                            "event": (
                                "post_descent_lateral_negative_vertical_tail_"
                                "recorded_with_buffer_retained"
                            ),
                            **lateral_post_action_interlock,
                        }
                    )
                corridor_entry_after_action = (
                    _overhead_corridor_entry_evidence(
                        current_eef=after_eef,
                        corridor_high_target=corridor_high_target,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                        position_tolerance=args.position_tolerance,
                        strict_corridor_entry_clearance_m=(
                            vertical_staging_corridor[
                                "strict_corridor_entry_clearance_m"
                            ]
                        ),
                    )
                )
                feedback["corridor_entry_after_drift_correction"] = (
                    corridor_entry_after_action
                )
                if corridor_entry_after_action["accepted"]:
                    structural_stage = "vertical_corridor_descent"
        elif stage_before_action == "vertical_corridor_settle":
            lateral_settle_progress = (
                _outside_side_lateral_settle_evidence(
                    before_guard=pre_action_guard,
                    after_guard=latest_outside_side_guard,
                    before_eef=current_eef,
                    after_eef=np.asarray(
                        rollout.obs["robot0_eef_pos"],
                        dtype=float,
                    ),
                    previous_stable_response_count=int(
                        lateral_settle_state.get(
                            "stable_response_count", 0
                        )
                    ),
                )
            )
            feedback["lateral_settle_progress"] = (
                lateral_settle_progress
            )
            if lateral_settle_progress["settled"]:
                lateral_settle_state = None
                structural_stage = "fixed_safe_z_lateral_approach"
                fixed_safe_z = float(
                    np.asarray(
                        rollout.obs["robot0_eef_pos"], dtype=float
                    )[2]
                )
            else:
                lateral_settle_state = lateral_settle_progress
        elif stage_before_action == "vertical_corridor_descent":
            after_eef = np.asarray(
                rollout.obs["robot0_eef_pos"], dtype=float
            )
            if (
                after_eef[2] <= corridor_side_target[2]
                and latest_outside_side_guard["accepted"]
            ):
                lateral_settle_state = (
                    _outside_side_staircase_settle_trigger(
                        feedback_mode=(
                            "constraint_prioritized_vertical_descent"
                        ),
                        guard_step=guard_step,
                        step_response=current_step_response,
                    )
                )
                feedback["lateral_settle_trigger"] = (
                    lateral_settle_state
                )
                structural_stage = "vertical_corridor_settle"
        feedback["stage_after_action"] = structural_stage
        outside_side_feedback_steps.append(feedback)
        motion_sample = capture(
            f"outside_{stage_before_action}",
            outside_side_motion_steps,
            False,
            True,
            extra={
                "outside_side_feedback": feedback,
                "outside_side_guard": latest_outside_side_guard,
                **(
                    {"compiled_overhead_guard": latest_overhead_guard}
                    if stage_before_action in overhead_route_stages
                    else {}
                ),
                **(
                    {
                        "overhead_lateral_buffer": (
                            _overhead_lateral_buffer_frame_summary(
                                latest_overhead_lateral_buffer
                            )
                        )
                    }
                    if stage_before_action in overhead_route_stages
                    else {}
                ),
            },
        )
        structural_violations = []
        required_clearance = float(
            latest_outside_side_guard[
                "required_outside_clearance_m"
            ]
        )
        if (
            stage_before_action in overhead_route_stages
            and not latest_overhead_guard["accepted"]
        ):
            structural_violations.append(
                "compiled_overhead_one_step_vertical_reserve_lost"
            )
        if stage_before_action in {
            "vertical_corridor_descent",
            "vertical_corridor_settle",
            "fixed_safe_z_lateral_approach",
        } and (
            latest_outside_side_guard[
                "minimum_outside_clearance_m"
            ]
            < required_clearance
        ):
            structural_violations.append(
                "strict_outside_clearance_lost_in_structural_approach"
            )
        if stage_before_action in {
            "vertical_corridor_descent",
            "vertical_corridor_settle",
            "fixed_safe_z_lateral_approach",
        } and (
            latest_outside_side_guard[
                "finger_table_vertical_clearance_m"
            ]
            < latest_outside_side_guard[
                "required_finger_table_clearance_m"
            ]
        ):
            structural_violations.append(
                "strict_finger_table_clearance_lost_in_structural_approach"
            )
        if stage_before_action in {
            "vertical_corridor_descent",
            "vertical_corridor_settle",
        } and (
            latest_outside_side_guard[
                "minimum_outside_clearance_m"
            ]
            <= vertical_staging_corridor[
                "strict_corridor_entry_clearance_m"
            ]
        ):
            structural_violations.append(
                "one_controller_step_corridor_reserve_lost"
            )
        if stage_before_action in {
            "vertical_corridor_settle",
            "fixed_safe_z_lateral_approach",
        } and not latest_outside_side_guard["accepted"]:
            structural_violations.append(
                "compiled_safe_z_rim_coverage_not_sustained"
            )
        if structural_violations:
            motion_sample["accepted"] = False
            motion_sample["violations"].extend(structural_violations)
            raise RuntimeError(
                "structurally decoupled outside-side approach lost a "
                "compiled physical gate: "
                f"source={source} guard_step={guard_step} "
                f"stage={stage_before_action} "
                f"violations={json.dumps(structural_violations)} "
                f"feedback={json.dumps(feedback, sort_keys=True)} "
                f"samples={json.dumps(samples, sort_keys=True)} "
                f"scene={json.dumps(diagnostics(), sort_keys=True)}"
            )
    else:
        raise RuntimeError(
            "structurally decoupled outside-side approach exhausted the "
            "unchanged OSC waypoint budget: "
            f"source={source} max_steps={args.max_waypoint_steps} "
            f"stage={structural_stage} "
            f"stage_action_counts={json.dumps(structural_stage_action_counts, sort_keys=True)} "
            f"guard={json.dumps(latest_outside_side_guard, sort_keys=True)} "
            f"overhead_guard={json.dumps(latest_overhead_guard, sort_keys=True)} "
            f"overhead_geometry={json.dumps(overhead_staging_geometry, sort_keys=True)} "
            f"samples={json.dumps(samples, sort_keys=True)} "
            f"scene={json.dumps(diagnostics(), sort_keys=True)}"
        )
    final_outside_side_guard = _live_outside_side_guard(
        env, geometry
    )
    outside_side_guard_checks += 1
    if not final_outside_side_guard["accepted"]:
        raise RuntimeError(
            "outside-side AABB guard was not sustained after geometry "
            "feedback stop: "
            f"source={source} "
            f"guard={json.dumps(final_outside_side_guard, sort_keys=True)} "
            f"samples={json.dumps(samples, sort_keys=True)} "
            f"scene={json.dumps(diagnostics(), sort_keys=True)}"
        )
    outside_side_sample = capture("outside_side", 0, False, True)
    outside_side_sample["outside_side_guard"] = (
        final_outside_side_guard
    )

    two_finger_contact_observed = False
    for seek_index in range(1, args.plate_contact_seek_max_steps + 1):
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        )
        action = _bounded_side_contact_seek_action(
            current_eef,
            contact_target,
            gripper,
            args.position_action_scale,
            args.plate_contact_seek_max_translation_action,
        )
        rollout.advance(action, "task")
        contact_observed = _robot_contacts_body(env, PLATE_BODY)
        capture(
            "bounded_lateral_contact_seek",
            seek_index,
            contact_observed,
            contact_observed,
        )
        finger_contact_sides = _plate_finger_contact_sides(env)
        two_finger_contact_observed = (
            finger_contact_sides["left"]
            and finger_contact_sides["right"]
        )
        if two_finger_contact_observed:
            break
    if not two_finger_contact_observed:
        raise RuntimeError(
            "bounded OSC two-finger plate side contact not observed: "
            f"source={source} outside_side_target="
            f"{np.asarray(outside_side_target).tolist()} contact_target="
            f"{np.asarray(contact_target).tolist()} "
            f"compiled_geometry={json.dumps(geometry, sort_keys=True)} "
            f"samples={json.dumps(samples, sort_keys=True)}"
        )

    for confirm_index in range(
        1, args.pusher_contact_confirm_steps + 1
    ):
        action = np.zeros(7, dtype=float)
        action[-1] = gripper
        rollout.advance(action, "task")
        capture(
            "stable_contact_confirmation",
            confirm_index,
            True,
            True,
        )

    result = {
        "source": source,
        "outside_high_target": np.asarray(
            outside_high_target, dtype=float
        ).tolist(),
        "outside_side_target": np.asarray(
            outside_side_target, dtype=float
        ).tolist(),
        "contact_target": np.asarray(
            contact_target, dtype=float
        ).tolist(),
        "compiled_geometry": geometry,
        "outside_side_guard": final_outside_side_guard,
        "outside_side_guard_checks": outside_side_guard_checks,
        "compiled_overhead_staging_geometry": (
            overhead_staging_geometry
        ),
        "overhead_guard_checks": int(overhead_guard_checks),
        "overhead_horizontal_z_m": float(overhead_horizontal_z),
        "latest_measured_vertical_step_progress_m": float(
            latest_vertical_step_progress_m
        ),
        "latest_overhead_lateral_buffer": (
            latest_overhead_lateral_buffer
        ),
        "vertical_tail_events": vertical_tail_events,
        "outside_side_motion_steps": outside_side_motion_steps,
        "outside_side_feedback_steps": outside_side_feedback_steps,
        "vertical_staging_corridor": vertical_staging_corridor,
        "corridor_high_target": np.asarray(
            corridor_high_target, dtype=float
        ).tolist(),
        "corridor_side_target": np.asarray(
            corridor_side_target, dtype=float
        ).tolist(),
        "fixed_safe_z_m": float(fixed_safe_z),
        "structural_stage_action_counts": (
            structural_stage_action_counts
        ),
        "structural_waypoint_budget": {
            "maximum_steps": int(args.max_waypoint_steps),
            "used_steps": int(outside_side_motion_steps),
            "remaining_steps": int(
                args.max_waypoint_steps - outside_side_motion_steps
            ),
        },
        "maximum_translation_action": (
            args.plate_contact_seek_max_translation_action
        ),
        "seek_steps_used": sum(
            sample["stage"] == "bounded_lateral_contact_seek"
            for sample in samples
        ),
        "confirmation_steps": args.pusher_contact_confirm_steps,
        "samples": samples,
        "accepted": True,
    }
    print(
        "L3-A3 bounded stable contact seek "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def _calibrate_stable_plate_contact_depth(
    rollout,
    env,
    args,
    *,
    gripper,
    source,
    diagnostics,
):
    """Deepen a first-touch contact with measured, gated OSC actions."""
    initial_eef = np.asarray(
        rollout.obs["robot0_eef_pos"], dtype=float
    ).copy()
    plate_reference = body_pose(env, PLATE_BODY)[0].copy()
    initial_contact_z_offset = float(
        initial_eef[2] - plate_reference[2]
    )
    plate_finger_allowed_body_pairs = (
        _compiled_plate_finger_allowed_body_pairs(env)
    )
    samples = []

    def capture(stage, index, require_stable):
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        ).copy()
        state = _contact_depth_state_diagnostics(
            env,
            plate_reference,
            allowed_robot_nonrobot_body_pairs=(
                plate_finger_allowed_body_pairs
            ),
            require_stable=require_stable,
            maximum_plate_tilt_deg=(
                args.max_contact_calibration_plate_tilt_deg
            ),
            maximum_plate_xy_drift=(
                args.max_contact_calibration_plate_xy_drift
            ),
            maximum_linear_speed=args.max_stable_linear_speed,
            maximum_angular_speed=args.max_stable_angular_speed,
        )
        finger_contact_sides = _plate_finger_contact_sides(env)
        live_plate = np.asarray(state["plate_position"], dtype=float)
        live_contact_z_offset = float(
            current_eef[2] - live_plate[2]
        )
        measured_depth_increase = float(
            initial_contact_z_offset - live_contact_z_offset
        )
        sample = {
            "stage": stage,
            "index": int(index),
            "eef_position": current_eef.tolist(),
            "initial_contact_z_offset_m": initial_contact_z_offset,
            "live_contact_z_offset_m": live_contact_z_offset,
            "measured_contact_depth_increase_m": (
                measured_depth_increase
            ),
            "measured_eef_world_descent_m": float(
                initial_eef[2] - current_eef[2]
            ),
            "finger_contact_sides": finger_contact_sides,
            **state,
        }
        samples.append(sample)
        if not (
            finger_contact_sides["left"]
            and finger_contact_sides["right"]
        ):
            sample["accepted"] = False
            sample["violations"].append(
                "two_finger_side_contact_not_sustained"
            )
        maximum_safe_depth_increase = min(
            args.target_contact_depth_increase
            + args.contact_depth_action_step,
            0.005,
        )
        sample["maximum_safe_contact_depth_increase_m"] = (
            maximum_safe_depth_increase
        )
        if measured_depth_increase > maximum_safe_depth_increase:
            sample["accepted"] = False
            sample["violations"].append(
                "contact_depth_overshoot_exceeded"
            )
        if not sample["accepted"]:
            failure = {
                "source": source,
                "initial_eef": initial_eef.tolist(),
                "plate_reference": plate_reference.tolist(),
                "samples": samples,
                "scene": diagnostics(),
            }
            raise RuntimeError(
                "stable plate contact-depth calibration violated a physical "
                "or collision gate: "
                f"{json.dumps(failure, sort_keys=True)}"
            )
        return sample

    capture("pre_depth", 0, True)
    for action_index in range(1, args.maximum_contact_depth_actions + 1):
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        )
        current_plate = body_pose(env, PLATE_BODY)[0]
        current_contact_z_offset = float(
            current_eef[2] - current_plate[2]
        )
        measured_depth_increase = float(
            initial_contact_z_offset - current_contact_z_offset
        )
        if (
            measured_depth_increase
            >= args.target_contact_depth_increase
        ):
            break
        target = current_eef.copy()
        target[2] -= args.contact_depth_action_step
        rollout.advance(
            _position_action(
                current_eef,
                target,
                gripper,
                args.position_action_scale,
            ),
            "task",
        )
        capture("depth_action", action_index, False)

    reached_eef = np.asarray(
        rollout.obs["robot0_eef_pos"], dtype=float
    ).copy()
    reached_plate = body_pose(env, PLATE_BODY)[0]
    reached_contact_z_offset = float(
        reached_eef[2] - reached_plate[2]
    )
    reached_depth_increase = float(
        initial_contact_z_offset - reached_contact_z_offset
    )
    if reached_depth_increase < args.target_contact_depth_increase:
        raise RuntimeError(
            "stable plate contact-depth calibration did not reach its "
            "measured descent target: "
            f"source={source} "
            f"reached_depth_increase_m={reached_depth_increase:.6f} "
            f"target_depth_increase_m="
            f"{args.target_contact_depth_increase:.6f} "
            f"samples={json.dumps(samples, sort_keys=True)}"
        )

    for confirm_index in range(
        1, args.contact_depth_stability_steps + 1
    ):
        action = np.zeros(7, dtype=float)
        action[-1] = gripper
        rollout.advance(action, "task")
        capture("stability_confirmation", confirm_index, True)

    final_eef = np.asarray(
        rollout.obs["robot0_eef_pos"], dtype=float
    ).copy()
    final_plate = body_pose(env, PLATE_BODY)[0].copy()
    final_contact_z_offset = float(final_eef[2] - final_plate[2])
    final_depth_increase = float(
        initial_contact_z_offset - final_contact_z_offset
    )
    if final_depth_increase < args.target_contact_depth_increase:
        raise RuntimeError(
            "stable plate contact depth rebounded above the measured target "
            "during confirmation: "
            f"source={source} "
            f"final_depth_increase_m={final_depth_increase:.6f} "
            f"target_depth_increase_m="
            f"{args.target_contact_depth_increase:.6f} "
            f"samples={json.dumps(samples, sort_keys=True)}"
        )
    result = {
        "source": source,
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "plate_reference_position": plate_reference.tolist(),
        "final_plate_position": final_plate.tolist(),
        "target_contact_depth_increase_m": (
            args.target_contact_depth_increase
        ),
        "initial_contact_z_offset_m": initial_contact_z_offset,
        "final_contact_z_offset_m": final_contact_z_offset,
        "measured_contact_depth_increase_m": final_depth_increase,
        "measured_eef_world_descent_m": float(
            initial_eef[2] - final_eef[2]
        ),
        "measured_contact_offset": (
            final_eef - final_plate
        ).tolist(),
        "depth_actions_used": sum(
            sample["stage"] == "depth_action" for sample in samples
        ),
        "stability_steps": args.contact_depth_stability_steps,
        "samples": samples,
        "accepted": True,
    }
    print(
        "L3-A3 stable contact-depth calibration "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return result


def generate(args):
    from libero.libero.envs import OffScreenRenderEnv

    if args.video_stride < 1:
        raise ValueError("--video_stride must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video_fps must be positive")
    if args.pusher_contact_confirm_steps < 1:
        raise ValueError("--pusher_contact_confirm_steps must be positive")
    if (
        not np.isfinite(args.plate_contact_outside_clearance)
        or not (0 < args.plate_contact_outside_clearance <= 0.020)
    ):
        raise ValueError(
            "--plate_contact_outside_clearance must be in (0, 0.020]"
        )
    if (
        not np.isfinite(
            args.plate_contact_seek_max_translation_action
        )
        or not (
            0 < args.plate_contact_seek_max_translation_action <= 0.2
        )
    ):
        raise ValueError(
            "--plate_contact_seek_max_translation_action must be in (0, 0.2]"
        )
    if args.plate_contact_seek_max_steps < 1:
        raise ValueError(
            "--plate_contact_seek_max_steps must be positive"
        )
    if (
        not np.isfinite(args.contact_depth_action_step)
        or args.contact_depth_action_step <= 0
        or args.contact_depth_action_step > 0.005
    ):
        raise ValueError(
            "--contact_depth_action_step must be in (0, 0.005]"
        )
    if (
        not np.isfinite(args.target_contact_depth_increase)
        or args.target_contact_depth_increase <= 0
        or args.target_contact_depth_increase > 0.005
    ):
        raise ValueError(
            "--target_contact_depth_increase must be in (0, 0.005]"
        )
    if args.maximum_contact_depth_actions < 1:
        raise ValueError(
            "--maximum_contact_depth_actions must be positive"
        )
    if (
        args.target_contact_depth_increase
        > args.contact_depth_action_step
        * args.maximum_contact_depth_actions
    ):
        raise ValueError(
            "contact depth target exceeds the commanded calibration budget"
        )
    if args.contact_depth_stability_steps < 1:
        raise ValueError(
            "--contact_depth_stability_steps must be positive"
        )
    if (
        not np.isfinite(
            args.max_contact_calibration_plate_xy_drift
        )
        or args.max_contact_calibration_plate_xy_drift <= 0
    ):
        raise ValueError(
            "--max_contact_calibration_plate_xy_drift must be positive"
        )
    if (
        not np.isfinite(
            args.max_contact_calibration_plate_tilt_deg
        )
        or not (
            0 < args.max_contact_calibration_plate_tilt_deg <= 1.0
        )
    ):
        raise ValueError(
            "--max_contact_calibration_plate_tilt_deg must be in (0, 1]"
        )
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

        # Compile every cardinal trailing side from the native plate and
        # semantic left/right finger collision bounds.  Reject a side if the
        # two fingers would reach it at materially different depths, then rank
        # the remaining sides by centre-high to outside-high action demand.
        # Clipping is diagnostic: Rollout.move's unchanged env.step waypoint
        # timeout is the authoritative reachability gate.  Jobs 499814/499848
        # showed that proximity alone chose bad -Y, while rejecting any first
        # clipped action also incorrectly excluded the reachable +X side.
        plate_start = body_pose(env, PLATE_BODY)[0]
        goal = np.asarray(
            env.sim.data.site_xpos[env.sim.model.site_name2id(GOAL_SITE)],
            dtype=float,
        )
        direction_xy = goal[:2] - plate_start[:2]
        direction_xy /= np.linalg.norm(direction_xy)
        (
            selected_contact_candidate,
            candidate_geometry,
        ) = _compiled_trailing_side_contact_candidates(
            env,
            plate_position=plate_start,
            push_direction_xy=direction_xy,
            eef_position=np.asarray(
                rollout.obs["robot0_eef_pos"], dtype=float
            ),
            backoff=args.plate_contact_backoff,
            outside_clearance_m=args.plate_contact_outside_clearance,
            plate_approach_eef_height=args.plate_approach_eef_height,
            position_action_scale=args.position_action_scale,
        )
        outside_high_target = np.asarray(
            selected_contact_candidate["outside_high_target"],
            dtype=float,
        )
        outside_side_target = np.asarray(
            selected_contact_candidate["outside_side_target"],
            dtype=float,
        )
        contact_target = np.asarray(
            selected_contact_candidate["side_contact_target"],
            dtype=float,
        )
        compiled_side_contact_geometry = (
            selected_contact_candidate["compiled_geometry"]
        )
        center_approach_target = np.asarray(
            selected_contact_candidate["center_high_target"],
            dtype=float,
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
                "selected_contact_candidate": (
                    selected_contact_candidate
                ),
                "selected_contact_line_xy": contact_target[:2].tolist(),
                "center_approach_target": center_approach_target.tolist(),
                "outside_high_target": outside_high_target.tolist(),
                "outside_side_target": outside_side_target.tolist(),
                "side_contact_target": contact_target.tolist(),
                "compiled_side_contact_geometry": (
                    compiled_side_contact_geometry
                ),
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
        # Decouple the large workspace translation from the native-geometry
        # side-contact path.  The helper moves high outside the plate, lowers
        # with no contact, then seeks laterally until both fingers contact.
        rollout.move(
            center_approach_target,
            pusher_open_sign,
            "task",
            diagnostics=plate_diagnostics,
        )
        initial_contact_seek = _seek_stable_plate_contact(
            rollout,
            env,
            args,
            gripper=pusher_open_sign,
            outside_high_target=outside_high_target,
            outside_side_target=outside_side_target,
            contact_target=contact_target,
            geometry=compiled_side_contact_geometry,
            source="initial_contact",
            diagnostics=plate_diagnostics,
        )
        contact_seek_events = [initial_contact_seek]
        initial_contact_depth_calibration = (
            _calibrate_stable_plate_contact_depth(
                rollout,
                env,
                args,
                gripper=pusher_open_sign,
                source="initial_contact",
                diagnostics=plate_diagnostics,
            )
        )
        contact_depth_calibrations = [
            initial_contact_depth_calibration
        ]

        push_eef_start = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        ).copy()
        push_plate_start = body_pose(env, PLATE_BODY)[0].copy()
        confirmed_contact_offset = push_eef_start - push_plate_start
        explicit_contact_anchor_offset = confirmed_contact_offset.copy()
        explicit_contact_anchor_source = (
            "initial_stable_contact_depth_calibration"
        )
        confirmed_contact_z_offset = float(confirmed_contact_offset[2])
        confirmed_contact_z_offset_source = (
            "initial_stable_contact_depth_calibration"
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
                "latest_contact_depth_calibration": (
                    contact_depth_calibrations[-1]
                ),
                "latest_stable_contact_seek": contact_seek_events[-1],
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
                (
                    recontact_selected_candidate,
                    recontact_candidates,
                ) = _compiled_trailing_side_contact_candidates(
                    env,
                    plate_position=recontact_plate,
                    push_direction_xy=recontact_direction,
                    eef_position=recontact_eef,
                    backoff=args.plate_contact_backoff,
                    outside_clearance_m=(
                        args.plate_contact_outside_clearance
                    ),
                    plate_approach_eef_height=(
                        args.plate_approach_eef_height
                    ),
                    position_action_scale=args.position_action_scale,
                )
                recontact_high_target = np.asarray(
                    recontact_selected_candidate[
                        "outside_high_target"
                    ],
                    dtype=float,
                )
                recontact_outside_side_target = np.asarray(
                    recontact_selected_candidate[
                        "outside_side_target"
                    ],
                    dtype=float,
                )
                recontact_side_contact_target = np.asarray(
                    recontact_selected_candidate[
                        "side_contact_target"
                    ],
                    dtype=float,
                )
                recontact_compiled_geometry = (
                    recontact_selected_candidate["compiled_geometry"]
                )
                recontact_retreat_target = recontact_eef.copy()
                recontact_retreat_target[2] = max(
                    recontact_eef[2], recontact_high_target[2]
                )
                recontact_center_target = np.asarray(
                    recontact_selected_candidate[
                        "center_high_target"
                    ],
                    dtype=float,
                )
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
                    "candidate_geometry": recontact_candidates,
                    "selected_contact_candidate": (
                        recontact_selected_candidate
                    ),
                    "retreat_target": recontact_retreat_target.tolist(),
                    "center_target": recontact_center_target.tolist(),
                    "outside_high_target": (
                        recontact_high_target.tolist()
                    ),
                    "outside_side_target": (
                        recontact_outside_side_target.tolist()
                    ),
                    "side_contact_target": (
                        recontact_side_contact_target.tolist()
                    ),
                    "compiled_side_contact_geometry": (
                        recontact_compiled_geometry
                    ),
                }

                def recontact_diagnostics():
                    return {
                        **plate_diagnostics(),
                        "completed_push_waypoints": push_waypoints,
                        "completed_recontact_events": recontact_events,
                        "active_recontact_event": recontact_event,
                    }

                # Every recovery waypoint uses OSC env.step.  Retreat
                # vertically first, cross above the live plate, descend
                # outside its compiled rim, then seek inward at rim height.
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
                recontact_stable_seek = _seek_stable_plate_contact(
                    rollout,
                    env,
                    args,
                    gripper=pusher_open_sign,
                    outside_high_target=recontact_high_target,
                    outside_side_target=(
                        recontact_outside_side_target
                    ),
                    contact_target=recontact_side_contact_target,
                    geometry=recontact_compiled_geometry,
                    source=f"recontact_{recontact_attempts}",
                    diagnostics=recontact_diagnostics,
                )
                contact_seek_events.append(recontact_stable_seek)
                recontact_depth_calibration = (
                    _calibrate_stable_plate_contact_depth(
                        rollout,
                        env,
                        args,
                        gripper=pusher_open_sign,
                        source=f"recontact_{recontact_attempts}",
                        diagnostics=recontact_diagnostics,
                    )
                )
                contact_depth_calibrations.append(
                    recontact_depth_calibration
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
                    "stable_contact_depth_calibration"
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
                        "stable_contact_depth_calibration": (
                            recontact_depth_calibration
                        ),
                        "bounded_stable_contact_seek": (
                            recontact_stable_seek
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
            "contact_depth_calibrations": contact_depth_calibrations,
            "contact_depth_calibration_count": len(
                contact_depth_calibrations
            ),
            "stable_contact_seek_events": contact_seek_events,
            "stable_contact_seek_count": len(contact_seek_events),
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
    # y=-0.038508 and remains inside the measured reachable envelope.  It is
    # now only the terminal target for a native-geometry lateral contact seek;
    # physical two-finger plate contact remains mandatory before any push.
    parser.add_argument("--plate_contact_backoff", type=float, default=0.010)
    # Keep the high outside approach at the Superpod-validated clearance.
    parser.add_argument(
        "--plate_approach_eef_height", type=float, default=0.160
    )
    parser.add_argument("--pusher_contact_confirm_steps", type=int, default=2)
    # Jobs 499756/499762: vertical first contact levered the native rim with
    # one finger.  Derive an outside pose and rim-centred EEF height from the
    # compiled plate/finger collision AABBs, descend with no contact, then
    # seek laterally until both native fingers contact.
    parser.add_argument(
        "--plate_contact_outside_clearance", type=float, default=0.005
    )
    parser.add_argument(
        "--plate_contact_seek_max_translation_action",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--plate_contact_seek_max_steps", type=int, default=64
    )
    # Job 499726 showed that first contact at EEF-minus-plate Z ~= 19.5 mm
    # was an unstable upper-edge touch: lateral pushing lifted the EEF by
    # about 2.3 mm and moved the plate less than 1 mm.  After first contact,
    # use bounded OSC actions to measure a 3 mm deeper contact and accept that
    # Z anchor only while contact, table support, upright pose, collision, and
    # stabilization gates all remain valid.
    parser.add_argument(
        "--contact_depth_action_step", type=float, default=0.004
    )
    parser.add_argument(
        "--target_contact_depth_increase", type=float, default=0.003
    )
    parser.add_argument(
        "--maximum_contact_depth_actions", type=int, default=12
    )
    parser.add_argument(
        "--contact_depth_stability_steps", type=int, default=3
    )
    parser.add_argument(
        "--max_contact_calibration_plate_xy_drift",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--max_contact_calibration_plate_tilt_deg",
        type=float,
        default=1.0,
    )
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

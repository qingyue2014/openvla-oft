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
    requested_vertical_action = float(
        maximum_descent_m / float(position_action_scale)
    )
    commanded_vertical_action = min(
        requested_vertical_action,
        remaining_vertical_action,
    )
    action = np.zeros(7, dtype=float)
    action[:2] = lateral_action
    action[2] = -commanded_vertical_action
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
            "Euclidean norm to vertical descent"
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
        "recovery_action_saturated": recovery_action_saturated,
        "descent_path_control": descent_path_control,
        "available_table_descent_m": available_table_descent,
        "action": action.tolist(),
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
):
    """Require lateral and vertical motion to stop trending toward hazards."""
    before_eef = np.asarray(before_eef, dtype=float)
    after_eef = np.asarray(after_eef, dtype=float)
    if before_eef.shape != (3,) or after_eef.shape != (3,):
        raise ValueError("outside-side settle EEF vectors must be 3-D")
    step_response = _outside_side_step_response_evidence(
        before_guard=before_guard,
        after_guard=after_guard,
        before_eef=before_eef,
        after_eef=after_eef,
    )
    vertical_step_progress = float(after_eef[2] - before_eef[2])
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
    return {
        "settled": not violations,
        "violations": violations,
        "formula": (
            "after an observed inward-coupled descent step, issue no Z "
            "command until measured Z, EEF-outward, and live-clearance "
            "step progress are all nonnegative and compiled clearance is "
            "satisfied"
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
    """Require a no-Z settle phase after every commanded descent step."""
    if feedback_mode != "constraint_prioritized_vertical_descent":
        return None
    return {
        "policy": (
            "preventive staircase: every constraint-prioritized descent "
            "step is followed by measured no-Z lateral settling before "
            "another descent can be issued"
        ),
        "trigger_guard_step": int(guard_step),
        "trigger_step_response": step_response,
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
                "samples": samples,
                "scene": diagnostics(),
            }
            raise RuntimeError(
                "bounded plate-contact seek violated a physical or collision "
                "gate: "
                f"{json.dumps(failure, sort_keys=True)}"
            )
        return sample

    rollout.move(
        outside_high_target,
        gripper,
        "task",
        diagnostics=diagnostics,
    )
    capture("outside_high", 0, False, True)
    outside_side_guard_checks = 1
    outside_side_motion_steps = 0
    latest_outside_side_guard = _live_outside_side_guard(
        env, geometry
    )
    outside_side_feedback_steps = []
    recovery_response_state = None
    lateral_settle_state = None
    previous_step_response = None
    for guard_step in range(1, args.max_waypoint_steps + 1):
        if (
            latest_outside_side_guard["accepted"]
            and recovery_response_state is None
            and lateral_settle_state is None
        ):
            break
        pre_action_guard = latest_outside_side_guard
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        )
        try:
            action, feedback = _outside_side_geometry_feedback_action(
                current_eef=current_eef,
                outside_side_target=outside_side_target,
                guard=pre_action_guard,
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                maximum_translation_action=(
                    args.plate_contact_seek_max_translation_action
                ),
                force_outward_recovery=(
                    recovery_response_state is not None
                ),
                force_lateral_settle=(
                    lateral_settle_state is not None
                ),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "outside-side geometry feedback concluded the native "
                "orientation is infeasible before table contact: "
                f"source={source} guard_step={guard_step} "
                f"cause_type={type(exc).__name__} "
                f"cause_message={str(exc)!r} "
                f"guard={json.dumps(latest_outside_side_guard, sort_keys=True)} "
                f"samples={json.dumps(samples, sort_keys=True)} "
                f"scene={json.dumps(diagnostics(), sort_keys=True)}"
            ) from exc
        rollout.advance(action, "task")
        outside_side_motion_steps += 1
        latest_outside_side_guard = _live_outside_side_guard(
            env, geometry
        )
        outside_side_guard_checks += 1
        feedback["post_action_guard"] = latest_outside_side_guard
        current_step_response = (
            _outside_side_step_response_evidence(
                before_guard=pre_action_guard,
                after_guard=latest_outside_side_guard,
                before_eef=current_eef,
                after_eef=np.asarray(
                    rollout.obs["robot0_eef_pos"],
                    dtype=float,
                ),
            )
        )
        feedback["step_response"] = current_step_response
        recovery_progress = None
        lateral_settle_progress = None
        if feedback["mode"] == "recover_outside_clearance":
            if recovery_response_state is None:
                recovery_response_state = {
                    "baseline_guard": pre_action_guard,
                    "baseline_eef": current_eef.copy(),
                }
            recovery_progress = (
                _outside_side_recovery_progress_evidence(
                    baseline_guard=recovery_response_state[
                        "baseline_guard"
                    ],
                    after_guard=latest_outside_side_guard,
                    baseline_eef=recovery_response_state[
                        "baseline_eef"
                    ],
                    before_guard=pre_action_guard,
                    before_eef=current_eef,
                    after_eef=np.asarray(
                        rollout.obs["robot0_eef_pos"],
                        dtype=float,
                    ),
                    action=action,
                    maximum_translation_action=(
                        args.plate_contact_seek_max_translation_action
                    ),
                    previous_step_response=previous_step_response,
                )
            )
            feedback["recovery_progress"] = recovery_progress
            if recovery_progress["progress_proven"]:
                recovery_response_state = None
        elif feedback["mode"] == "compiled_outside_lateral_settle":
            lateral_settle_progress = (
                _outside_side_lateral_settle_evidence(
                    before_guard=pre_action_guard,
                    after_guard=latest_outside_side_guard,
                    before_eef=current_eef,
                    after_eef=np.asarray(
                        rollout.obs["robot0_eef_pos"],
                        dtype=float,
                    ),
                )
            )
            feedback["lateral_settle_progress"] = (
                lateral_settle_progress
            )
            if lateral_settle_progress["settled"]:
                lateral_settle_state = None
        else:
            staircase_trigger = (
                _outside_side_staircase_settle_trigger(
                    feedback_mode=feedback["mode"],
                    guard_step=guard_step,
                    step_response=current_step_response,
                )
            )
            if staircase_trigger is not None:
                lateral_settle_state = staircase_trigger
                feedback["lateral_settle_trigger"] = (
                    lateral_settle_state
                )
        outside_side_feedback_steps.append(feedback)
        motion_sample = capture(
            "outside_side_motion",
            outside_side_motion_steps,
            False,
            True,
            extra={
                "outside_side_feedback": feedback,
                "outside_side_guard": latest_outside_side_guard,
            },
        )
        if (
            recovery_progress is not None
            and recovery_progress["fail_closed"]
        ):
            motion_sample["accepted"] = False
            motion_sample["violations"].extend(
                recovery_progress["violations"]
            )
            raise RuntimeError(
                "saturated outward OSC recovery failed net live-clearance "
                "progress and stopped improving its discrete response; "
                "native side is dynamically unreachable under the "
                "unchanged controller bound: "
                f"source={source} guard_step={guard_step} "
                f"feedback={json.dumps(feedback, sort_keys=True)} "
                f"samples={json.dumps(samples, sort_keys=True)} "
                f"scene={json.dumps(diagnostics(), sort_keys=True)}"
            )
        previous_step_response = current_step_response
        if (
            latest_outside_side_guard["accepted"]
            and recovery_response_state is None
            and lateral_settle_state is None
        ):
            break
    else:
        raise RuntimeError(
            "outside-side geometry feedback exhausted the unchanged OSC "
            "waypoint budget: "
            f"source={source} max_steps={args.max_waypoint_steps} "
            f"guard={json.dumps(latest_outside_side_guard, sort_keys=True)} "
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
        "outside_side_motion_steps": outside_side_motion_steps,
        "outside_side_feedback_steps": outside_side_feedback_steps,
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
    samples = []

    def capture(stage, index, require_stable):
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        ).copy()
        state = _contact_depth_state_diagnostics(
            env,
            plate_reference,
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

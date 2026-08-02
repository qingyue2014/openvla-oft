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
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


L3A3_CABINET_ROOT_BODY = "wooden_cabinet_1_main"
L3A3_CABINET_TOP_BODY = "wooden_cabinet_1_cabinet_top"
L3A3_CABINET_TOP_JOINT = "wooden_cabinet_1_top_level"
L3A3_WINE_RACK_BODY = "wine_rack_1_main"

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
    eef_xy = np.asarray(eef_xy, dtype=float)
    if plate_xy.shape != (2,) or eef_xy.shape != (2,):
        raise ValueError("plate and EEF coordinates must be 2-D")
    if direction.shape != (2,):
        raise ValueError("push direction must be 2-D")
    if (
        not np.all(np.isfinite(plate_xy))
        or not np.all(np.isfinite(eef_xy))
    ):
        raise ValueError("plate and EEF coordinates must be finite")
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1e-9:
        raise ValueError("push direction must be finite and nonzero")
    try:
        backoff = float(backoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("plate contact backoff must be a finite scalar") from exc
    if not np.isfinite(backoff) or backoff <= 0:
        raise ValueError("plate contact backoff must be positive and finite")
    direction /= direction_norm

    cardinal_offsets = (
        ("+x", np.array([backoff, 0.0])),
        ("-x", np.array([-backoff, 0.0])),
        ("+y", np.array([0.0, backoff])),
        ("-y", np.array([0.0, -backoff])),
    )
    derived_offsets = (
        ("trailing_minus_push", -direction * backoff),
        (
            "tangent_counterclockwise",
            np.array([-direction[1], direction[0]]) * backoff,
        ),
        (
            "tangent_clockwise",
            np.array([direction[1], -direction[0]]) * backoff,
        ),
    )

    def candidate_record(
        offset,
        *,
        provenance,
        native_relation,
        route_selection_candidate,
    ):
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("derived plate contact offset must be finite and 2-D")
        offset_norm = float(np.linalg.norm(offset))
        if not np.isfinite(offset_norm) or offset_norm <= 1e-9:
            raise ValueError("derived plate contact offset must be nonzero")
        inward_component = float(np.dot(direction, -offset))
        return {
            "point_xy": (plate_xy + offset).tolist(),
            "offset_xy": offset.tolist(),
            "inward_component_m": inward_component,
            "eef_xy_distance_m": float(
                np.linalg.norm((plate_xy + offset) - eef_xy)
            ),
            "trailing_eligible": bool(inward_component > 1e-6),
            "route_selection_candidate": bool(route_selection_candidate),
            "diagnostic_only": not bool(route_selection_candidate),
            "candidate_provenance": [provenance],
            "deduplicated_provenance": [],
            "native_push_direction_relations": (
                [] if native_relation is None else [native_relation]
            ),
        }

    # Preserve the historical cardinal candidate order and selection domain.
    # Only semantically trailing cardinals reached the compiled plan before
    # native-direction audit candidates were added.
    candidates = []
    for label, offset in cardinal_offsets:
        if float(np.dot(direction, -offset)) <= 1e-6:
            continue
        candidates.append(
            candidate_record(
                offset,
                provenance=f"legacy_cardinal:{label}",
                native_relation=None,
                route_selection_candidate=True,
            )
        )

    # Append directions derived strictly from the normalized native push.
    # These are diagnostic-only until native reset evidence authorizes a
    # route-selection change.  Exact duplicates merge provenance into the
    # existing candidate instead of changing its position in the list.
    for relation, offset in derived_offsets:
        provenance = f"native_push_direction:{relation}"
        duplicate = next(
            (
                candidate
                for candidate in candidates
                if np.allclose(
                    candidate["offset_xy"],
                    offset,
                    rtol=0.0,
                    atol=1e-9,
                )
            ),
            None,
        )
        if duplicate is not None:
            duplicate["candidate_provenance"].append(provenance)
            duplicate["deduplicated_provenance"].append(provenance)
            duplicate["native_push_direction_relations"].append(relation)
            continue
        candidates.append(
            candidate_record(
                offset,
                provenance=provenance,
                native_relation=relation,
                route_selection_candidate=False,
            )
        )
    return candidates


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


def _live_collision_geom_record(
    model,
    data,
    geom_id,
    *,
    eef_position=None,
):
    """Serialize one exact compiled collision geom at the current live state."""
    geom_id = int(geom_id)
    if not 0 <= geom_id < int(model.ngeom):
        raise RuntimeError("live collision geom id is out of bounds")
    body_id = int(model.geom_bodyid[geom_id])
    if not 0 <= body_id < int(model.nbody):
        raise RuntimeError("live collision geom body id is out of bounds")
    geom_name = model.geom_id2name(geom_id) or ""
    body_name = model.body_id2name(body_id) or ""
    if not geom_name or not body_name:
        raise RuntimeError("live collision geom or body name is unavailable")

    geom_type = int(model.geom_type[geom_id])
    geom_size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_xpos = np.asarray(data.geom_xpos[geom_id], dtype=float)
    geom_xmat = np.asarray(data.geom_xmat[geom_id], dtype=float)
    local_aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
    if (
        geom_size.shape != (3,)
        or geom_xpos.shape != (3,)
        or geom_xmat.shape != (9,)
        or local_aabb.shape != (6,)
        or not np.all(np.isfinite(geom_size))
        or not np.all(np.isfinite(geom_xpos))
        or not np.all(np.isfinite(geom_xmat))
        or not np.all(np.isfinite(local_aabb))
        or np.any(geom_size < 0.0)
        or np.any(local_aabb[3:] < 0.0)
    ):
        raise RuntimeError(
            "live collision geom type/size/pose/AABB evidence is invalid"
        )
    world_center, world_half_size = _compiled_geom_world_aabb(
        model, data, geom_id
    )
    world_min = world_center - world_half_size
    world_max = world_center + world_half_size
    record = {
        "geom_id": geom_id,
        "name": geom_name,
        "body_id": body_id,
        "body": body_name,
        "type": geom_type,
        "size": geom_size.tolist(),
        "xpos_world": geom_xpos.tolist(),
        "xmat_world_row_major": geom_xmat.tolist(),
        "compiled_local_aabb_center": local_aabb[:3].tolist(),
        "compiled_local_aabb_half_size": local_aabb[3:].tolist(),
        "world_aabb_center": world_center.tolist(),
        "world_aabb_half_size": world_half_size.tolist(),
        "world_aabb_min": world_min.tolist(),
        "world_aabb_max": world_max.tolist(),
    }
    if eef_position is not None:
        eef_position = np.asarray(eef_position, dtype=float)
        if eef_position.shape != (3,) or not np.all(
            np.isfinite(eef_position)
        ):
            raise RuntimeError("live EEF position is invalid")
        record["eef_position_world"] = eef_position.tolist()
        record["world_aabb_min_offset_from_eef"] = (
            world_min - eef_position
        ).tolist()
        record["world_aabb_max_offset_from_eef"] = (
            world_max - eef_position
        ).tolist()
    return record


def _joint_qpos_width(joint_type):
    """Return MuJoCo qpos width for free, ball, slide, or hinge joints."""
    widths = {0: 7, 1: 4, 2: 1, 3: 1}
    try:
        return widths[int(joint_type)]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("unsupported compiled cabinet joint type") from exc


def _live_body_pose_record(model, data, body_name):
    """Record an exact live body pose with explicit compiled-body provenance."""
    try:
        body_id = int(model.body_name2id(body_name))
    except Exception as exc:
        raise RuntimeError(
            f"required compiled cabinet body is unavailable: {body_name}"
        ) from exc
    xpos = np.asarray(data.body_xpos[body_id], dtype=float)
    xmat = np.asarray(data.body_xmat[body_id], dtype=float)
    if (
        xpos.shape != (3,)
        or xmat.shape != (9,)
        or not np.all(np.isfinite(xpos))
        or not np.all(np.isfinite(xmat))
    ):
        raise RuntimeError(f"compiled cabinet body pose is invalid: {body_name}")
    return {
        "body_id": body_id,
        "name": body_name,
        "xpos_world": xpos.tolist(),
        "xmat_world_row_major": xmat.tolist(),
        "pose_provenance": (
            "env.sim.data.body_xpos/body_xmat after the exact second live "
            "geometry recompile"
        ),
    }


def _live_joint_qpos_record(model, data, joint_id):
    """Record exact qpos values and address provenance for one compiled joint."""
    joint_id = int(joint_id)
    if not 0 <= joint_id < int(model.njnt):
        raise RuntimeError("compiled cabinet joint id is out of bounds")
    joint_name = model.joint_id2name(joint_id) or ""
    if not joint_name:
        raise RuntimeError("compiled cabinet joint name is unavailable")
    body_id = int(model.jnt_bodyid[joint_id])
    qpos_address = int(model.jnt_qposadr[joint_id])
    qpos_width = _joint_qpos_width(model.jnt_type[joint_id])
    qpos = np.take(
        np.asarray(data.qpos, dtype=float),
        np.arange(qpos_address, qpos_address + qpos_width),
    )
    if qpos.shape != (qpos_width,) or not np.all(np.isfinite(qpos)):
        raise RuntimeError(f"compiled cabinet joint qpos is invalid: {joint_name}")
    return {
        "joint_id": joint_id,
        "name": joint_name,
        "type": int(model.jnt_type[joint_id]),
        "body_id": body_id,
        "body": model.body_id2name(body_id) or "",
        "qpos_address": qpos_address,
        "qpos_width": qpos_width,
        "qpos": qpos.tolist(),
        "qpos_provenance": (
            "np.take(env.sim.data.qpos, the range beginning at "
            "model.jnt_qposadr[joint_id] with compiled joint-type width) at "
            "the exact second live geometry recompile"
        ),
    }


def _live_cabinet_pose_diagnostic(env):
    """Capture native cabinet root/top-drawer live pose and qpos provenance."""
    model, data = env.sim.model, env.sim.data
    root = _live_body_pose_record(model, data, L3A3_CABINET_ROOT_BODY)
    top = _live_body_pose_record(model, data, L3A3_CABINET_TOP_BODY)
    attached_joint_ids = [
        joint_id
        for joint_id in range(int(model.njnt))
        if int(model.jnt_bodyid[joint_id]) in {
            int(root["body_id"]),
            int(top["body_id"]),
        }
    ]
    joints = [
        _live_joint_qpos_record(model, data, joint_id)
        for joint_id in attached_joint_ids
    ]
    top_joint = next(
        (joint for joint in joints if joint["name"] == L3A3_CABINET_TOP_JOINT),
        None,
    )
    if top_joint is None:
        raise RuntimeError(
            "required compiled native cabinet top drawer joint is unavailable"
        )
    return {
        "root_body": root,
        "top_drawer_body": top,
        "root_body_attached_joints": [
            joint
            for joint in joints
            if joint["body_id"] == root["body_id"]
        ],
        "top_drawer_joint": top_joint,
        "root_pose_qpos_provenance": (
            "the native fixture root may have no qpos joint; its authoritative "
            "live pose is data.body_xpos/body_xmat, while every attached root "
            "joint qpos is enumerated explicitly above"
        ),
    }


def _live_collision_inventory(env, *, eef_position):
    """Capture every live robot and native collision geom, fail-closed."""
    model, data = env.sim.model, env.sim.data
    eef_position = np.asarray(eef_position, dtype=float)
    if eef_position.shape != (3,) or not np.all(np.isfinite(eef_position)):
        raise RuntimeError("live collision inventory EEF position is invalid")
    robot_bodies = set(_robot_gripper_body_names(env))
    robot_geoms = []
    native_geoms = []
    for geom_id in range(int(model.ngeom)):
        collision_enabled = bool(
            getattr(model, "geom_contype", None) is None
            or getattr(model, "geom_conaffinity", None) is None
            or int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
        if not collision_enabled:
            continue
        body_name = model.body_id2name(
            int(model.geom_bodyid[geom_id])
        ) or ""
        is_robot = body_name in robot_bodies
        record = _live_collision_geom_record(
            model,
            data,
            geom_id,
            eef_position=eef_position if is_robot else None,
        )
        (robot_geoms if is_robot else native_geoms).append(record)
    if not robot_geoms or not native_geoms:
        raise RuntimeError(
            "live collision inventory lacks robot or native collision geoms"
        )
    canonical_inventory = {
        "robot_collision_geoms": robot_geoms,
        "native_nonrobot_collision_geoms": native_geoms,
    }
    inventory_json = json.dumps(
        canonical_inventory,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "schema_version": 1,
        "capture_stage": "post_center_high_reacquire_second_recompile_pre_seek",
        "eef_position_world": eef_position.tolist(),
        **canonical_inventory,
        "robot_collision_geom_count": len(robot_geoms),
        "native_nonrobot_collision_geom_count": len(native_geoms),
        "total_collision_geom_count": len(robot_geoms) + len(native_geoms),
        "inventory_sha256": hashlib.sha256(
            inventory_json.encode("utf-8")
        ).hexdigest(),
        "hash_payload": (
            "canonical compact JSON of robot_collision_geoms and "
            "native_nonrobot_collision_geoms only"
        ),
    }


def _diagnostic_only_live_detour_candidates(
    *,
    live_inventory,
    cabinet_pose,
    current_eef,
    outside_high_target,
    outside_side_target,
):
    """Describe unresolved sweep candidates without selecting or executing one."""
    current_eef = np.asarray(current_eef, dtype=float)
    outside_high_target = np.asarray(outside_high_target, dtype=float)
    outside_side_target = np.asarray(outside_side_target, dtype=float)
    if (
        current_eef.shape != (3,)
        or outside_high_target.shape != (3,)
        or outside_side_target.shape != (3,)
        or not np.all(np.isfinite(current_eef))
        or not np.all(np.isfinite(outside_high_target))
        or not np.all(np.isfinite(outside_side_target))
        or not isinstance(live_inventory.get("inventory_sha256"), str)
        or len(live_inventory["inventory_sha256"]) != 64
        or cabinet_pose.get("top_drawer_joint") is None
    ):
        raise RuntimeError("diagnostic-only detour candidate inputs are invalid")
    common = {
        "diagnostic_only": True,
        "executed": False,
        "selection_eligible": False,
        "selected": False,
        "route_authorized": False,
        "live_inventory_sha256": live_inventory["inventory_sha256"],
        "live_robot_collision_geom_count": live_inventory[
            "robot_collision_geom_count"
        ],
        "live_native_collision_geom_count": live_inventory[
            "native_nonrobot_collision_geom_count"
        ],
        "known_start_eef_world": current_eef.tolist(),
        "known_registered_outside_high_target_world": (
            outside_high_target.tolist()
        ),
        "known_registered_outside_side_target_world": (
            outside_side_target.tolist()
        ),
        "aabb_authorization_prohibited": True,
        "exact_sweep_required_inputs": [
            "all robot collision geom live transforms and EEF-relative bounds",
            "all native nonrobot collision geom live transforms and bounds",
            "cabinet root and top-drawer live pose plus qpos provenance",
            "continuous exact collision-distance sweep over every robot/native pair",
            "native workspace reachability and saturation evidence at every segment",
            "measured controller tracking and inertial-tail reserve",
            "downstream structural/contact/push step budget",
        ],
    }
    candidates = [
        {
            **common,
            "candidate_id": "vertical_first",
            "candidate_description": (
                "hold current XY and orientation, solve an unresolved exact-sweep "
                "over-cabinet Z, move laterally above the obstacle, then validate "
                "the complete terminal descent column"
            ),
            "unresolved_waypoint_fields": [
                "exact_sweep_over_cabinet_z",
                "terminal_descent_column_clearance",
            ],
        },
        {
            **common,
            "candidate_id": "minus_x_detour",
            "candidate_description": (
                "solve an unresolved -X exact-sweep side waypoint, pass the full "
                "inflated native cabinet extent, then return only after exact "
                "terminal-column validation"
            ),
            "unresolved_waypoint_fields": [
                "exact_sweep_minus_x_clearance_waypoint",
                "cabinet_y_pass_waypoint",
                "terminal_descent_column_clearance",
            ],
        },
        {
            **common,
            "candidate_id": "plus_x_detour",
            "candidate_description": (
                "solve an unresolved +X exact-sweep side waypoint with a blocking "
                "native workspace-saturation gate, pass the cabinet extent, then "
                "validate the complete return and descent path"
            ),
            "unresolved_waypoint_fields": [
                "exact_sweep_plus_x_clearance_waypoint",
                "plus_x_native_workspace_margin",
                "cabinet_y_pass_waypoint",
                "terminal_descent_column_clearance",
            ],
        },
    ]
    if any(
        not candidate["diagnostic_only"]
        or candidate["executed"]
        or candidate["selection_eligible"]
        or candidate["selected"]
        or candidate["route_authorized"]
        for candidate in candidates
    ):
        raise RuntimeError("diagnostic-only detour candidate selected or executed")
    return candidates


def _compiled_native_right_high_then_low_return_plan(
    *,
    live_inventory,
    current_eef,
    outside_high_target,
    outside_side_target,
    maximum_controller_world_step_m,
    maximum_route_translation_action,
    position_action_scale_m_per_action,
    position_tolerance_m,
):
    """Compile a right-high-then-low-return route from live geometry.

    The trailing side of the native plate lies behind the protruding top-drawer
    handle.  The rigid hand subtree therefore moves to the complete native
    obstacle set's +X side at the existing high Z, traverses to the trailing Y
    while remaining right of that set, descends on that right-side column, and
    returns in -X to the unchanged contact column while remaining under the
    cabinet and right of the wine rack.  No AABB prediction is treated as
    rollout evidence: the
    returned plan is recompiled from the live simulator and its front/right/
    under inequalities plus the empty robot/native contact allowlist are
    rechecked after every executed OSC action.
    """
    current_eef = np.asarray(current_eef, dtype=float)
    outside_high_target = np.asarray(outside_high_target, dtype=float)
    outside_side_target = np.asarray(outside_side_target, dtype=float)
    if any(
        value.shape != (3,) or not np.all(np.isfinite(value))
        for value in (current_eef, outside_high_target, outside_side_target)
    ):
        raise RuntimeError("native cabinet detour targets must be finite 3-D")
    try:
        maximum_controller_world_step_m = float(
            maximum_controller_world_step_m
        )
        maximum_route_translation_action = float(
            maximum_route_translation_action
        )
        position_action_scale_m_per_action = float(
            position_action_scale_m_per_action
        )
        position_tolerance_m = float(position_tolerance_m)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("native cabinet detour limits must be finite") from exc
    if not (
        np.isfinite(maximum_controller_world_step_m)
        and maximum_controller_world_step_m > 0.0
        and np.isfinite(maximum_route_translation_action)
        and maximum_route_translation_action > 0.0
        and np.isfinite(position_action_scale_m_per_action)
        and position_action_scale_m_per_action > 0.0
        and np.isfinite(position_tolerance_m)
        and position_tolerance_m > 0.0
    ):
        raise RuntimeError("native cabinet detour limits must be positive")
    if not isinstance(live_inventory, dict) or not isinstance(
        live_inventory.get("inventory_sha256"), str
    ):
        raise RuntimeError("native cabinet detour lacks live inventory binding")

    robot_geoms = list(live_inventory.get("robot_collision_geoms", ()))
    native_geoms = list(
        live_inventory.get("native_nonrobot_collision_geoms", ())
    )
    rigid_hand_geoms = [
        record
        for record in robot_geoms
        if str(record.get("name", "")).startswith("gripper0_")
    ]
    cabinet_top_geoms = [
        record
        for record in native_geoms
        if record.get("body") == L3A3_CABINET_TOP_BODY
    ]
    wine_rack_geoms = [
        record
        for record in native_geoms
        if record.get("body") == L3A3_WINE_RACK_BODY
    ]
    plate_geoms = [
        record
        for record in native_geoms
        if record.get("body") == PLATE_BODY
    ]
    if (
        not rigid_hand_geoms
        or not cabinet_top_geoms
        or not wine_rack_geoms
        or not plate_geoms
    ):
        raise RuntimeError(
            "native cabinet detour lacks rigid-hand, plate, cabinet-top, or "
            "wine-rack geoms"
        )
    def vector(record, key):
        value = np.asarray(record.get(key), dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise RuntimeError(
                f"native cabinet detour geom has invalid {key}: "
                f"{record.get('name', '')}"
            )
        return value

    structural_native_geoms = [
        record
        for record in native_geoms
        if record.get("body") not in {TABLE_BODY, "world"}
    ]
    if not structural_native_geoms:
        raise RuntimeError(
            "native cabinet detour lacks structural native obstacle geoms"
        )

    maximum_hand_x_offset = float(
        max(
            vector(record, "world_aabb_max_offset_from_eef")[0]
            for record in rigid_hand_geoms
        )
    )
    minimum_hand_x_offset = float(
        min(
            vector(record, "world_aabb_min_offset_from_eef")[0]
            for record in rigid_hand_geoms
        )
    )
    minimum_hand_y_offset = float(
        min(
            vector(record, "world_aabb_min_offset_from_eef")[1]
            for record in rigid_hand_geoms
        )
    )
    minimum_hand_z_offset = float(
        min(
            vector(record, "world_aabb_min_offset_from_eef")[2]
            for record in rigid_hand_geoms
        )
    )
    maximum_hand_z_offset = float(
        max(
            vector(record, "world_aabb_max_offset_from_eef")[2]
            for record in rigid_hand_geoms
        )
    )
    start_high = current_eef.copy()
    cabinet_min_z = float(
        min(vector(record, "world_aabb_min")[2] for record in cabinet_top_geoms)
    )
    wine_rack_max_x = float(
        max(vector(record, "world_aabb_max")[0] for record in wine_rack_geoms)
    )
    right_obstacle_max_x = float(
        max(
            vector(record, "world_aabb_max")[0]
            for record in structural_native_geoms
        )
    )
    plate_max_z = float(
        max(vector(record, "world_aabb_max")[2] for record in plate_geoms)
    )
    selected_right_x = float(
        np.nextafter(
            right_obstacle_max_x
            - minimum_hand_x_offset
            + maximum_controller_world_step_m
            + position_tolerance_m,
            np.inf,
        )
    )
    start_hand_min = np.array(
        [
            start_high[0] + minimum_hand_x_offset,
            start_high[1] + minimum_hand_y_offset,
            start_high[2] + minimum_hand_z_offset,
        ],
        dtype=float,
    )
    start_hand_max = np.array(
        [
            start_high[0] + maximum_hand_x_offset,
            0.0,
            start_high[2] + maximum_hand_z_offset,
        ],
        dtype=float,
    )
    initial_high_route_separations = []
    for record in structural_native_geoms:
        obstacle_min = vector(record, "world_aabb_min")
        obstacle_max = vector(record, "world_aabb_max")
        axis_clearances = {
            "front_y_m": float(start_hand_min[1] - obstacle_max[1]),
            "right_x_m": float(start_hand_min[0] - obstacle_max[0]),
            "left_x_m": float(obstacle_min[0] - start_hand_max[0]),
            "above_z_m": float(start_hand_min[2] - obstacle_max[2]),
        }
        initial_high_route_separations.append(
            {
                "geom": record["name"],
                "body": record["body"],
                "maximum_separating_axis_clearance_m": float(
                    max(axis_clearances.values())
                ),
                "axis_clearances": axis_clearances,
            }
        )
    predicted_initial_high_route_separation = float(
        min(
            record["maximum_separating_axis_clearance_m"]
            for record in initial_high_route_separations
        )
    )
    predicted_right_clearance = float(
        selected_right_x
        + minimum_hand_x_offset
        - right_obstacle_max_x
    )
    predicted_under_clearance = float(
        cabinet_min_z
        - (outside_side_target[2] + maximum_hand_z_offset)
    )
    predicted_right_of_rack_clearance = float(
        outside_side_target[0]
        + minimum_hand_x_offset
        - wine_rack_max_x
    )
    if not predicted_initial_high_route_separation > (
        maximum_controller_world_step_m
    ):
        raise RuntimeError(
            "native cabinet detour lacks a strict one-step separating axis "
            "from every native structural obstacle at the high-route start"
        )
    if not predicted_right_clearance > (
        maximum_controller_world_step_m + position_tolerance_m
    ):
        raise RuntimeError(
            "native cabinet detour lacks strict right one-step clearance plus "
            "the unchanged waypoint tolerance"
        )
    if not predicted_under_clearance > maximum_controller_world_step_m:
        raise RuntimeError(
            "native cabinet detour trailing column is not strictly below the "
            "cabinet by one controller world step"
        )
    under_clearance_headroom = float(
        predicted_under_clearance - maximum_controller_world_step_m
    )
    low_route_entry_z_tolerance = float(
        min(
            position_tolerance_m,
            np.nextafter(0.25 * under_clearance_headroom, 0.0),
        )
    )
    if not low_route_entry_z_tolerance > 0.0:
        raise RuntimeError(
            "native cabinet detour has no positive low-route entry Z "
            "tolerance after preserving under-cabinet headroom"
        )
    if not predicted_right_of_rack_clearance > maximum_controller_world_step_m:
        raise RuntimeError(
            "native cabinet detour terminal column is not strictly right of "
            "the wine rack by one controller world step"
        )
    if not outside_side_target[2] < outside_high_target[2]:
        raise RuntimeError("native cabinet detour has no downward terminal route")
    maximum_route_world_command_m = float(
        maximum_route_translation_action
        * position_action_scale_m_per_action
    )
    if maximum_route_world_command_m < maximum_controller_world_step_m:
        raise RuntimeError(
            "native cabinet detour route action is below the existing "
            "near-contact action bound"
        )

    predicted_high_above_plate_clearance = float(
        start_high[2] + minimum_hand_z_offset - plate_max_z
    )
    if not predicted_high_above_plate_clearance > (
        maximum_controller_world_step_m
    ):
        raise RuntimeError(
            "native cabinet detour high route is not strictly above the "
            "native plate by one controller world step"
        )
    right_high = np.array(
        [selected_right_x, start_high[1], start_high[2]], dtype=float
    )
    right_trailing_high = np.array(
        [selected_right_x, outside_side_target[1], start_high[2]],
        dtype=float,
    )
    right_trailing_low = np.array(
        [selected_right_x, outside_side_target[1], outside_side_target[2]],
        dtype=float,
    )
    terminal_low = outside_side_target.copy()
    segment_distances = [
        float(np.linalg.norm(end - start))
        for start, end in zip(
            (
                start_high,
                right_high,
                right_trailing_high,
                right_trailing_low,
            ),
            (
                right_high,
                right_trailing_high,
                right_trailing_low,
                terminal_low,
            ),
        )
    ]
    minimum_action_lower_bound = int(
        sum(
            np.ceil(distance / maximum_route_world_command_m)
            for distance in segment_distances
        )
    )
    return {
        "candidate_id": "native_right_high_then_low_return",
        "diagnostic_only": False,
        "selection_eligible": True,
        "selected": True,
        "route_authorized": True,
        "authorization_basis": (
            "exact live compiled rigid-hand and complete native structural "
            "obstacle bounds; "
            "four axis-separated OSC segments; per-action live separating-"
            "axis/right/under bounds, "
            "empty robot/native contact allowlist, plate stability, and table "
            "clearance revalidation"
        ),
        "live_inventory_sha256": live_inventory["inventory_sha256"],
        "live_robot_collision_geom_count": len(robot_geoms),
        "live_native_collision_geom_count": len(native_geoms),
        "rigid_hand_geom_names": [
            record["name"] for record in rigid_hand_geoms
        ],
        "cabinet_top_geom_names": [
            record["name"] for record in cabinet_top_geoms
        ],
        "wine_rack_geom_names": [
            record["name"] for record in wine_rack_geoms
        ],
        "plate_geom_names": [record["name"] for record in plate_geoms],
        "high_route_obstacle_geom_names": [
            record["name"] for record in structural_native_geoms
        ],
        "right_obstacle_geom_names": [
            record["name"] for record in structural_native_geoms
        ],
        "cabinet_top_body": L3A3_CABINET_TOP_BODY,
        "wine_rack_body": L3A3_WINE_RACK_BODY,
        "maximum_controller_world_step_m": (
            maximum_controller_world_step_m
        ),
        "maximum_route_translation_action": (
            maximum_route_translation_action
        ),
        "position_action_scale_m_per_action": (
            position_action_scale_m_per_action
        ),
        "maximum_route_world_command_m": maximum_route_world_command_m,
        "position_tolerance_m": position_tolerance_m,
        "maximum_rigid_hand_x_offset_from_eef_m": maximum_hand_x_offset,
        "minimum_rigid_hand_x_offset_from_eef_m": minimum_hand_x_offset,
        "minimum_rigid_hand_y_offset_from_eef_m": minimum_hand_y_offset,
        "minimum_rigid_hand_z_offset_from_eef_m": minimum_hand_z_offset,
        "maximum_rigid_hand_z_offset_from_eef_m": maximum_hand_z_offset,
        "cabinet_min_z_m": cabinet_min_z,
        "wine_rack_max_x_m": wine_rack_max_x,
        "right_obstacle_max_x_m": right_obstacle_max_x,
        "plate_max_z_m": plate_max_z,
        "selected_right_eef_x_m": selected_right_x,
        "predicted_initial_high_route_separation_m": (
            predicted_initial_high_route_separation
        ),
        "initial_high_route_separations": initial_high_route_separations,
        "predicted_right_clearance_m": predicted_right_clearance,
        "predicted_high_above_plate_clearance_m": (
            predicted_high_above_plate_clearance
        ),
        "predicted_under_clearance_at_terminal_m": predicted_under_clearance,
        "under_clearance_headroom_m": under_clearance_headroom,
        "low_route_entry_z_tolerance_m": low_route_entry_z_tolerance,
        "low_route_entry_headroom_retained_fraction": 0.75,
        "predicted_right_of_rack_clearance_at_terminal_m": (
            predicted_right_of_rack_clearance
        ),
        "required_strict_clearance_m": maximum_controller_world_step_m,
        "waypoints": {
            "start_high": start_high.tolist(),
            "right_high": right_high.tolist(),
            "right_trailing_high": right_trailing_high.tolist(),
            "right_trailing_low": right_trailing_low.tolist(),
            "terminal_outside_side_low": terminal_low.tolist(),
        },
        "segment_distances_m": segment_distances,
        "minimum_full_step_action_lower_bound": minimum_action_lower_bound,
        "route_order": [
            "right_high_lateral",
            "right_high_trailing_pass",
            "right_trailing_vertical_descent",
            "trailing_low_terminal_return",
        ],
        "runtime_requirements": {
            "right_high_route": (
                "the high +X approach preserves a separating front/right/"
                "left/above axis for every native structural obstacle"
            ),
            "right_trailing_and_descent": (
                "the high trailing pass and vertical descent stay strictly "
                "right of the complete native structural obstacle set"
            ),
            "terminal_return": (
                "the sole low lateral segment remains right of the wine "
                "rack and under the cabinet"
            ),
            "all_segments": (
                "empty structural robot/native contact allowlist plus unchanged "
                "plate stability and action bounds"
            ),
        },
    }


def _live_native_cabinet_detour_guard(env, *, eef_position, plan, stage):
    """Recompile the selected cabinet detour inequality at a live frame."""
    if not isinstance(plan, dict) or not plan.get("route_authorized", False):
        raise RuntimeError("native cabinet detour guard lacks authorization")
    if stage not in {
        "right_high_lateral",
        "right_high_trailing_pass",
        "right_trailing_vertical_descent",
        "trailing_low_terminal_return",
    }:
        raise RuntimeError(f"unknown native cabinet detour stage: {stage}")
    inventory = _live_collision_inventory(env, eef_position=eef_position)
    rigid_names = set(plan.get("rigid_hand_geom_names", ()))
    cabinet_names = set(plan.get("cabinet_top_geom_names", ()))
    wine_rack_names = set(plan.get("wine_rack_geom_names", ()))
    plate_names = set(plan.get("plate_geom_names", ()))
    high_route_obstacle_names = set(
        plan.get("high_route_obstacle_geom_names", ())
    )
    right_obstacle_names = set(plan.get("right_obstacle_geom_names", ()))
    rigid = [
        record
        for record in inventory["robot_collision_geoms"]
        if record.get("name") in rigid_names
    ]
    cabinet = [
        record
        for record in inventory["native_nonrobot_collision_geoms"]
        if record.get("name") in cabinet_names
        and record.get("body") == L3A3_CABINET_TOP_BODY
    ]
    wine_rack = [
        record
        for record in inventory["native_nonrobot_collision_geoms"]
        if record.get("name") in wine_rack_names
        and record.get("body") == L3A3_WINE_RACK_BODY
    ]
    plate = [
        record
        for record in inventory["native_nonrobot_collision_geoms"]
        if record.get("name") in plate_names
        and record.get("body") == PLATE_BODY
    ]
    high_route_obstacles = [
        record
        for record in inventory["native_nonrobot_collision_geoms"]
        if record.get("name") in high_route_obstacle_names
    ]
    right_obstacles = [
        record
        for record in inventory["native_nonrobot_collision_geoms"]
        if record.get("name") in right_obstacle_names
    ]
    if (
        len(rigid) != len(rigid_names)
        or len(cabinet) != len(cabinet_names)
        or len(wine_rack) != len(wine_rack_names)
        or len(plate) != len(plate_names)
        or len(high_route_obstacles) != len(high_route_obstacle_names)
        or len(right_obstacles) != len(right_obstacle_names)
    ):
        raise RuntimeError("native cabinet detour live geom identity changed")
    required = float(plan["required_strict_clearance_m"])
    minimum_hand_x = float(
        min(record["world_aabb_min"][0] for record in rigid)
    )
    maximum_hand_x = float(
        max(record["world_aabb_max"][0] for record in rigid)
    )
    maximum_hand_z = float(
        max(record["world_aabb_max"][2] for record in rigid)
    )
    minimum_hand_z = float(
        min(record["world_aabb_min"][2] for record in rigid)
    )
    minimum_hand_y = float(
        min(record["world_aabb_min"][1] for record in rigid)
    )
    minimum_cabinet_z = float(
        min(record["world_aabb_min"][2] for record in cabinet)
    )
    maximum_front_obstacle_y = float(
        max(
            record["world_aabb_max"][1]
            for record in high_route_obstacles
        )
    )
    maximum_right_obstacle_x = float(
        max(
            record["world_aabb_max"][0]
            for record in right_obstacles
        )
    )
    maximum_wine_rack_x = float(
        max(record["world_aabb_max"][0] for record in wine_rack)
    )
    maximum_plate_z = float(
        max(record["world_aabb_max"][2] for record in plate)
    )
    under_clearance = float(minimum_cabinet_z - maximum_hand_z)
    front_clearance = float(
        minimum_hand_y - maximum_front_obstacle_y
    )
    right_clearance = float(
        minimum_hand_x - maximum_right_obstacle_x
    )
    right_of_rack_clearance = float(
        minimum_hand_x - maximum_wine_rack_x
    )
    above_plate_clearance = float(minimum_hand_z - maximum_plate_z)
    high_route_obstacle_separations = []
    for record in high_route_obstacles:
        obstacle_min = np.asarray(record["world_aabb_min"], dtype=float)
        obstacle_max = np.asarray(record["world_aabb_max"], dtype=float)
        axis_clearances = {
            "front_y_m": float(minimum_hand_y - obstacle_max[1]),
            "right_x_m": float(minimum_hand_x - obstacle_max[0]),
            "left_x_m": float(obstacle_min[0] - maximum_hand_x),
            "above_z_m": float(minimum_hand_z - obstacle_max[2]),
        }
        high_route_obstacle_separations.append(
            {
                "geom": record["name"],
                "body": record["body"],
                "maximum_separating_axis_clearance_m": float(
                    max(axis_clearances.values())
                ),
                "axis_clearances": axis_clearances,
            }
        )
    minimum_high_route_separation = float(
        min(
            record["maximum_separating_axis_clearance_m"]
            for record in high_route_obstacle_separations
        )
    )
    require_high_route_separation = stage == "right_high_lateral"
    require_front = False
    require_under = stage == "trailing_low_terminal_return"
    require_right = stage in {
        "right_high_trailing_pass",
        "right_trailing_vertical_descent",
    }
    require_right_of_rack = stage == "trailing_low_terminal_return"
    accepted = bool(
        (
            not require_high_route_separation
            or minimum_high_route_separation > required
        )
        and (not require_front or front_clearance > required)
        and (not require_under or under_clearance > required)
        and (not require_right or right_clearance > required)
        and (
            not require_right_of_rack
            or right_of_rack_clearance > required
        )
    )
    violations = []
    if (
        require_high_route_separation
        and not minimum_high_route_separation > required
    ):
        violations.append(
            "rigid_hand_high_route_lacks_native_obstacle_separation"
        )
    if require_front and not front_clearance > required:
        violations.append(
            "rigid_hand_not_strictly_in_front_of_native_obstacles"
        )
    if require_under and not under_clearance > required:
        violations.append("rigid_hand_not_strictly_under_native_cabinet")
    if require_right and not right_clearance > required:
        violations.append(
            "rigid_hand_not_strictly_right_of_plate_and_wine_rack"
        )
    if require_right_of_rack and not right_of_rack_clearance > required:
        violations.append("rigid_hand_not_strictly_right_of_native_wine_rack")
    return {
        "accepted": accepted,
        "stage": stage,
        "live_inventory_sha256": inventory["inventory_sha256"],
        "live_robot_collision_geom_count": inventory[
            "robot_collision_geom_count"
        ],
        "live_native_collision_geom_count": inventory[
            "native_nonrobot_collision_geom_count"
        ],
        "blocking_cabinet_geom_count": len(cabinet),
        "blocking_wine_rack_geom_count": len(wine_rack),
        "blocking_plate_geom_count": len(plate),
        "high_route_obstacle_geom_count": len(high_route_obstacles),
        "right_obstacle_geom_count": len(right_obstacles),
        "required_strict_clearance_m": required,
        "require_high_route_separation": require_high_route_separation,
        "require_under_clearance": require_under,
        "require_front_clearance": require_front,
        "require_right_clearance": require_right,
        "require_right_of_rack_clearance": require_right_of_rack,
        "minimum_rigid_hand_x_m": minimum_hand_x,
        "maximum_rigid_hand_x_m": maximum_hand_x,
        "maximum_rigid_hand_z_m": maximum_hand_z,
        "minimum_rigid_hand_z_m": minimum_hand_z,
        "minimum_rigid_hand_y_m": minimum_hand_y,
        "minimum_cabinet_z_m": minimum_cabinet_z,
        "maximum_front_obstacle_y_m": maximum_front_obstacle_y,
        "maximum_right_obstacle_x_m": maximum_right_obstacle_x,
        "maximum_wine_rack_x_m": maximum_wine_rack_x,
        "maximum_plate_z_m": maximum_plate_z,
        "under_clearance_m": under_clearance,
        "front_clearance_m": front_clearance,
        "right_clearance_m": right_clearance,
        "right_of_rack_clearance_m": right_of_rack_clearance,
        "above_plate_clearance_m": above_plate_clearance,
        "minimum_high_route_separation_m": minimum_high_route_separation,
        "high_route_obstacle_separations": (
            high_route_obstacle_separations
        ),
        "violations": violations,
    }


def _write_controller_diagnostic_manifest(path, record):
    """Persist complete controller diagnostics without printing the inventory."""
    path = Path(path)
    if not path.name:
        raise RuntimeError("controller diagnostic manifest path is invalid")
    payload = json.dumps(
        record,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _compact_unexpected_contact_diagnostic(
    contact_gate,
    *,
    diagnostic_context,
):
    """Build a bounded stdout record for an unexpected robot/native contact."""
    unexpected = list(contact_gate.get("unexpected_contacts", ()))
    if not unexpected:
        return None

    def compact_geom(record):
        return {
            key: record[key]
            for key in (
                "geom_id",
                "name",
                "body_id",
                "body",
                "type",
                "size",
                "xpos_world",
                "world_aabb_min",
                "world_aabb_max",
            )
        }

    return {
        "diagnostic_only": True,
        "route_authorized": False,
        "inventory_sha256": diagnostic_context.get("inventory_sha256"),
        "robot_collision_geom_count": diagnostic_context.get(
            "robot_collision_geom_count"
        ),
        "native_nonrobot_collision_geom_count": diagnostic_context.get(
            "native_nonrobot_collision_geom_count"
        ),
        "controller_diagnostic_manifest": diagnostic_context.get(
            "manifest_path"
        ),
        "unexpected_contact_count": len(unexpected),
        "contacts": [
            {
                key: contact[key]
                for key in (
                    "contact_index",
                    "geom1_id",
                    "geom1_name",
                    "geom1_body",
                    "geom2_id",
                    "geom2_name",
                    "geom2_body",
                    "robot_geom_id",
                    "robot_geom",
                    "robot_body",
                    "native_geom_id",
                    "native_geom",
                    "native_body",
                    "position_world",
                    "frame_normal_geom1_to_geom2_world",
                    "sorted_geom_ids",
                    "normal_from_sorted_geom0_to_geom1_world",
                    "robot_to_native_normal_world",
                    "distance_m",
                    "penetration_m",
                )
            }
            | {
                "robot_live_geom": compact_geom(
                    contact["robot_live_geom"]
                ),
                "native_live_geom": compact_geom(
                    contact["native_live_geom"]
                ),
            }
            for contact in unexpected
        ],
    }


def _record_unexpected_contact_in_controller_manifest(
    *,
    diagnostic_context,
    source,
    stage,
    sample,
):
    """Append full live contact evidence to the standalone diagnostic JSON."""
    path_value = diagnostic_context.get("manifest_path")
    if not path_value:
        raise RuntimeError(
            "controller diagnostic manifest path is missing at contact failure"
        )
    path = Path(path_value)
    if not path.is_file():
        raise RuntimeError(
            "controller diagnostic manifest is missing at contact failure"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "controller diagnostic manifest cannot be read at contact failure"
        ) from exc
    if record.get("live_collision_inventory", {}).get(
        "inventory_sha256"
    ) != diagnostic_context.get("inventory_sha256"):
        raise RuntimeError(
            "controller diagnostic manifest inventory hash changed before contact"
        )
    events = list(record.get("unexpected_contact_events", ()))
    events.append(
        {
            "source": str(source),
            "stage": str(stage),
            "sample_index": int(sample["index"]),
            "eef_position_world": list(sample["eef_position"]),
            "plate_position_world": list(sample["plate_position"]),
            "plate_tilt_deg": float(sample["plate_tilt_deg"]),
            "plate_xy_drift_m": float(sample["plate_xy_drift_m"]),
            "robot_nonrobot_contact_gate": sample[
                "robot_nonrobot_contact_gate"
            ],
        }
    )
    record["unexpected_contact_events"] = events
    record["latest_status"] = "UNEXPECTED_ROBOT_NATIVE_CONTACT_FAIL_CLOSED"
    _write_controller_diagnostic_manifest(path, record)


def _record_native_cabinet_detour_completion(
    *,
    diagnostic_context,
    source,
    final_guard,
    stage_action_counts,
    used_steps,
    remaining_steps,
):
    """Hash-bind a successfully executed cabinet detour to its manifest."""
    path_value = diagnostic_context.get("manifest_path")
    if not path_value:
        raise RuntimeError(
            "controller diagnostic manifest path is missing at detour completion"
        )
    path = Path(path_value)
    if not path.is_file():
        raise RuntimeError(
            "controller diagnostic manifest is missing at detour completion"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "controller diagnostic manifest cannot be read at detour completion"
        ) from exc
    if record.get("live_collision_inventory", {}).get(
        "inventory_sha256"
    ) != diagnostic_context.get("inventory_sha256"):
        raise RuntimeError(
            "controller diagnostic manifest inventory hash changed before "
            "detour completion"
        )
    plan = diagnostic_context.get("authorized_detour_plan")
    if not (
        isinstance(plan, dict)
        and plan.get("candidate_id")
        == "native_right_high_then_low_return"
        and plan.get("route_authorized", False)
        and isinstance(final_guard, dict)
        and final_guard.get("accepted", False)
    ):
        raise RuntimeError(
            "native cabinet detour completion lacks an accepted bound plan"
        )
    completion = {
        "source": str(source),
        "candidate_id": plan["candidate_id"],
        "route_order": list(plan["route_order"]),
        "stage_action_counts": {
            stage: int(stage_action_counts[stage])
            for stage in plan["route_order"]
        },
        "used_structural_waypoint_steps": int(used_steps),
        "remaining_structural_waypoint_steps": int(remaining_steps),
        "final_live_guard": final_guard,
        "accepted": True,
    }
    record["route_executed"] = True
    record["route_completion"] = completion
    record["latest_status"] = (
        "NATIVE_RIGHT_HIGH_THEN_LOW_RETURN_EXECUTED_AND_GUARDED"
    )
    _write_controller_diagnostic_manifest(path, record)
    diagnostic_context["executed"] = True
    diagnostic_context["route_completion"] = completion
    return completion


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
            "from the exact native center-high state, first command XY plus "
            "nonnegative Z plane-hold actions with zero rotation toward the "
            "reachable outside-high point; then command a constrained outward-"
            "and-downward diagonal to the strict corridor before the remaining "
            "corridor-holding XY/Z descent. Derive every action norm from the "
            "runtime native and configured descent bounds and the full live "
            "compiled-pair worst-case downward-tail "
            "capacity above strict+base8 after reserving the latest measured "
            "negative-dz inertia, and recheck base8 afterward; jointly command "
            "the registered corridor XY error and independent negative-Z error "
            "with zero rotation, conservatively charging the complete 3-D "
            "translation norm as downward tail against every pair; these are "
            "live pre/post world-"
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
            "live buffer16 is not accepted before a buffer16-required route "
            "action"
        )
    adaptive_pair_count = 0
    if adaptive_high_lateral_envelope is not None:
        dynamic_buffer16_required = bool(
            adaptive_high_lateral_envelope.get(
                "negative_z_action_requires_fixed_buffer16", False
            )
        )
        if dynamic_buffer16_required != bool(require_lateral_buffer):
            raise RuntimeError(
                "adaptive route buffer16 requirement diverged from its live "
                "pre-action authorization"
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
        dynamic_minimum_pre_action_buffer16_surplus = float(
            adaptive_high_lateral_envelope.get(
                "minimum_pre_action_buffer16_surplus_after_inertia_m",
                np.inf,
            )
        )
        if not (
            adaptive_high_lateral_envelope.get("accepted", False)
            and dynamic_base_reserve == base_reserve
            and np.isfinite(dynamic_action_norm)
            and dynamic_action_norm > 0.0
            and np.isfinite(dynamic_world_tail)
            and dynamic_world_tail >= 0.0
            and np.isfinite(dynamic_minimum_surplus)
            and dynamic_minimum_surplus > 0.0
            and (
                not dynamic_buffer16_required
                or (
                    np.isfinite(
                        dynamic_minimum_pre_action_buffer16_surplus
                    )
                    and dynamic_minimum_pre_action_buffer16_surplus > 0.0
                )
            )
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
                and (
                    not dynamic_buffer16_required
                    or (
                        float(
                            adaptive_pair[
                                "worst_case_controller_world_step_m"
                            ]
                        )
                        == worst_case_step
                        and float(
                            adaptive_pair[
                                "required_clearance_with_fixed_buffer16_m"
                            ]
                        )
                        == float(
                            overhead_strict
                            + base_reserve
                            + worst_case_step
                        )
                        and float(
                            adaptive_pair[
                                "pre_action_buffer16_surplus_after_inertia_m"
                            ]
                        )
                        > 0.0
                    )
                )
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
                "worst-case action tail, latest measured negative-dz inertial "
                "reserve, and unchanged 8 mm base8 envelope"
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
    minimum_eef_z=None,
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
        or (
            minimum_eef_z is not None
            and not np.isfinite(minimum_eef_z)
        )
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
    if minimum_eef_z is not None and current_eef[2] < float(minimum_eef_z):
        violations.append("high_plane_hold_z_not_recovered")
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
        "minimum_eef_z_m": (
            float(minimum_eef_z) if minimum_eef_z is not None else None
        ),
        "high_plane_hold_z_accepted": bool(
            minimum_eef_z is None
            or current_eef[2] >= float(minimum_eef_z)
        ),
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


def _overhead_outside_high_entry_evidence(
    *,
    current_eef,
    outside_high_target,
    high_lateral_target=None,
    overhead_horizontal_z,
    overhead_guard,
    position_tolerance,
):
    """Gate the registered high-lateral target before workspace release."""
    current_eef = np.asarray(current_eef, dtype=float)
    outside_high_target = np.asarray(outside_high_target, dtype=float)
    high_lateral_target = np.asarray(
        outside_high_target
        if high_lateral_target is None
        else high_lateral_target,
        dtype=float,
    )
    if (
        current_eef.shape != (3,)
        or outside_high_target.shape != (3,)
        or high_lateral_target.shape != (3,)
        or not np.all(np.isfinite(current_eef))
        or not np.all(np.isfinite(outside_high_target))
        or not np.all(np.isfinite(high_lateral_target))
        or not np.isfinite(overhead_horizontal_z)
        or not np.isfinite(position_tolerance)
        or position_tolerance <= 0.0
    ):
        raise ValueError("outside-high entry evidence is invalid")
    xy_error = float(
        np.linalg.norm(current_eef[:2] - high_lateral_target[:2])
    )
    z_error = float(abs(current_eef[2] - overhead_horizontal_z))
    violations = []
    if xy_error > position_tolerance:
        violations.append("outside_high_xy_tolerance_not_met")
    if z_error > position_tolerance:
        violations.append("outside_high_plane_hold_tolerance_not_met")
    if not overhead_guard.get("accepted", False):
        violations.append("compiled_overhead_base8_not_accepted")
    return {
        "accepted": not violations,
        "violations": violations,
        "current_eef": current_eef.tolist(),
        "outside_high_target": outside_high_target.tolist(),
        "high_lateral_target": high_lateral_target.tolist(),
        "high_lateral_target_is_native_outside_high": bool(
            np.array_equal(high_lateral_target, outside_high_target)
        ),
        "high_lateral_target_role": (
            "native_outside_high"
            if np.array_equal(high_lateral_target, outside_high_target)
            else "registered_corridor_high_anticooupling_prebuffer"
        ),
        "overhead_horizontal_z_m": float(overhead_horizontal_z),
        "outside_high_xy_error_m": xy_error,
        "outside_high_plane_z_error_m": z_error,
        "position_tolerance_m": float(position_tolerance),
        "compiled_overhead_base8_accepted": bool(
            overhead_guard.get("accepted", False)
        ),
    }


def _high_plane_native_boundary_crossing_evidence(
    *,
    observation,
    high_lateral_target,
    native_outside_high_target,
    outward_direction_xy,
    expected_pair_count,
):
    """Prove one high-plane action strictly crossed the native outside target."""
    high_lateral_target = np.asarray(high_lateral_target, dtype=float)
    native_outside_high_target = np.asarray(
        native_outside_high_target, dtype=float
    )
    outward = np.asarray(outward_direction_xy, dtype=float)
    if (
        high_lateral_target.shape != (3,)
        or native_outside_high_target.shape != (3,)
        or outward.shape != (2,)
        or not np.all(np.isfinite(high_lateral_target))
        or not np.all(np.isfinite(native_outside_high_target))
        or not np.all(np.isfinite(outward))
        or not isinstance(expected_pair_count, (int, np.integer))
        or expected_pair_count < 1
    ):
        raise ValueError("high-plane native-boundary inputs are invalid")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 0.0:
        raise ValueError("high-plane native-boundary outward direction is invalid")
    outward /= outward_norm

    before_eef = np.asarray(observation["before_eef"], dtype=float)
    after_eef = np.asarray(observation["after_eef"], dtype=float)
    action = np.asarray(observation["action"], dtype=float)
    envelope = observation["high_plane_envelope"]
    pre_overhead_guard = observation["pre_overhead_guard"]
    post_overhead_guard = observation["post_overhead_guard"]
    after_outside_guard = observation["after_outside_guard"]
    if (
        before_eef.shape != (3,)
        or after_eef.shape != (3,)
        or action.shape != (7,)
        or not np.all(np.isfinite(before_eef))
        or not np.all(np.isfinite(after_eef))
        or not np.all(np.isfinite(action))
    ):
        raise ValueError("high-plane native-boundary observation is invalid")

    commanded_outward_action = float(np.dot(action[:2], outward))
    before_remaining_outward_error = float(
        np.dot(high_lateral_target[:2] - before_eef[:2], outward)
    )
    after_remaining_outward_error = float(
        np.dot(high_lateral_target[:2] - after_eef[:2], outward)
    )
    target_outward_of_native = float(
        np.dot(
            high_lateral_target[:2] - native_outside_high_target[:2],
            outward,
        )
    )
    actual_outward_beyond_native = float(
        np.dot(
            after_eef[:2] - native_outside_high_target[:2],
            outward,
        )
    )
    live_outside_clearance = float(
        after_outside_guard["minimum_outside_clearance_m"]
    )
    required_outside_clearance = float(
        after_outside_guard["required_outside_clearance_m"]
    )
    scalars = (
        commanded_outward_action,
        before_remaining_outward_error,
        after_remaining_outward_error,
        target_outward_of_native,
        actual_outward_beyond_native,
        live_outside_clearance,
        required_outside_clearance,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise RuntimeError("high-plane native-boundary evidence is non-finite")

    pre_pairs = list(pre_overhead_guard.get("pairs", ()))
    post_pairs = list(post_overhead_guard.get("pairs", ()))
    envelope_pairs = list(envelope.get("pair_envelopes", ()))
    pre_identities = [_overhead_pair_identity(pair) for pair in pre_pairs]
    post_identities = [_overhead_pair_identity(pair) for pair in post_pairs]
    envelope_identities = [
        tuple(identity) for identity in envelope.get("pair_identity_keys", ())
    ]
    pair_inventory_exact = bool(
        envelope.get("compiled_pair_count") == int(expected_pair_count)
        and len(pre_pairs) == int(expected_pair_count)
        and len(post_pairs) == int(expected_pair_count)
        and len(envelope_pairs) == int(expected_pair_count)
        and len(set(pre_identities)) == int(expected_pair_count)
        and pre_identities == post_identities == envelope_identities
    )
    all_pair_base8_strict = bool(
        pair_inventory_exact
        and pre_overhead_guard.get("accepted", False)
        and post_overhead_guard.get("accepted", False)
        and envelope.get("accepted", False)
        and all(pair.get("accepted", False) for pair in pre_pairs)
        and all(pair.get("accepted", False) for pair in post_pairs)
        and all(
            np.isfinite(
                float(
                    pair[
                        "predicted_post_worst_case_base_reserve_surplus_m"
                    ]
                )
            )
            and float(
                pair["predicted_post_worst_case_base_reserve_surplus_m"]
            )
            > 0.0
            for pair in envelope_pairs
        )
    )
    target_request_persistent = bool(
        commanded_outward_action > 0.0
        and before_remaining_outward_error > 0.0
        and after_remaining_outward_error > 0.0
        and target_outward_of_native > 0.0
        and np.array_equal(
            np.asarray(envelope["lateral_target_xy"], dtype=float),
            high_lateral_target[:2],
        )
    )
    action_high_plane_safe = bool(
        action[2] >= 0.0 and np.all(action[3:6] == 0.0)
    )
    violations = []
    if not target_request_persistent:
        violations.append("registered_corridor_outward_request_not_persistent")
    if not action_high_plane_safe:
        violations.append("high_plane_action_direction_or_rotation_invalid")
    if actual_outward_beyond_native <= 0.0:
        violations.append("actual_eef_not_strictly_beyond_native_outside_target")
    if live_outside_clearance <= required_outside_clearance:
        violations.append("live_outside_clearance_not_strict")
    if not all_pair_base8_strict:
        violations.append("all_55_pair_high_plane_base8_not_strict")
    return {
        "accepted": not violations,
        "violations": violations,
        "decision": (
            "strict_native_outside_boundary_crossed"
            if not violations
            else "continue_fail_closed_registered_corridor_request"
        ),
        "formula": (
            "without using waypoint tolerance, require the actual post-action "
            "EEF projection to lie strictly outward of the unchanged native "
            "outside-high target while the registered corridor target remains "
            "strictly farther outward; require a positive outward command, "
            "nonnegative Z, zero rotation, live outside clearance strictly "
            "above its compiled no-contact requirement, and the identical 55 "
            "compiled pairs to pass pre-action, high-envelope predicted-post, "
            "and live post-action base8"
        ),
        "uses_position_tolerance": False,
        "native_outside_high_target": native_outside_high_target.tolist(),
        "registered_high_lateral_target": high_lateral_target.tolist(),
        "before_eef": before_eef.tolist(),
        "after_eef": after_eef.tolist(),
        "outward_direction_xy": outward.tolist(),
        "commanded_outward_action": commanded_outward_action,
        "before_remaining_outward_error_m": before_remaining_outward_error,
        "after_remaining_outward_error_m": after_remaining_outward_error,
        "registered_target_outward_of_native_target_m": (
            target_outward_of_native
        ),
        "actual_eef_outward_of_native_outside_target_m": (
            actual_outward_beyond_native
        ),
        "actual_minimum_outside_clearance_m": live_outside_clearance,
        "required_outside_clearance_m": required_outside_clearance,
        "target_request_persistent": target_request_persistent,
        "action_nonnegative_z_zero_rotation": action_high_plane_safe,
        "compiled_pair_count": int(expected_pair_count),
        "pair_inventory_exact": pair_inventory_exact,
        "all_55_pair_high_plane_base8_strict": all_pair_base8_strict,
    }


def _high_plane_native_workspace_saturation_evidence(
    *,
    observations,
    initial_eef,
    high_lateral_target,
    native_outside_high_target,
    outward_direction_xy,
    position_action_scale,
    position_tolerance,
    progress_epsilon,
    required_window_frames,
    native_action_spec,
    expected_pair_count,
):
    """Prove a persistent, clipped high-plane request has stalled safely."""
    initial_eef = np.asarray(initial_eef, dtype=float)
    high_lateral_target = np.asarray(high_lateral_target, dtype=float)
    native_outside_high_target = np.asarray(
        native_outside_high_target, dtype=float
    )
    outward = np.asarray(outward_direction_xy, dtype=float)
    native_low = np.asarray(native_action_spec.get("low", ()), dtype=float)
    native_high = np.asarray(
        native_action_spec.get("high", ()), dtype=float
    )
    if (
        initial_eef.shape != (3,)
        or high_lateral_target.shape != (3,)
        or native_outside_high_target.shape != (3,)
        or outward.shape != (2,)
        or native_low.shape != (7,)
        or native_high.shape != (7,)
        or not np.all(np.isfinite(initial_eef))
        or not np.all(np.isfinite(high_lateral_target))
        or not np.all(np.isfinite(native_outside_high_target))
        or not np.all(np.isfinite(outward))
        or not np.all(np.isfinite(native_low))
        or not np.all(np.isfinite(native_high))
        or not np.all(native_low < native_high)
        or not native_action_spec.get("runtime_resolved", False)
        or native_action_spec.get("action_dimension") != 7
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(position_tolerance)
        or position_tolerance <= 0.0
        or not np.isfinite(progress_epsilon)
        or progress_epsilon <= 0.0
        or not isinstance(required_window_frames, (int, np.integer))
        or required_window_frames < 1
        or not isinstance(expected_pair_count, (int, np.integer))
        or expected_pair_count < 1
    ):
        raise ValueError("high-plane workspace-saturation inputs are invalid")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 0.0:
        raise ValueError("high-plane outward direction is invalid")
    outward /= outward_norm

    required_initial_action_xy = (
        high_lateral_target[:2] - initial_eef[:2]
    ) / position_action_scale
    native_outward_corner = np.where(
        outward >= 0.0,
        native_high[:2],
        native_low[:2],
    )
    native_outward_action_bound = float(
        np.dot(native_outward_corner, outward)
    )
    required_initial_outward_action = float(
        np.dot(required_initial_action_xy, outward)
    )
    clipped_axes = [
        int(axis)
        for axis in range(2)
        if (
            required_initial_action_xy[axis] < native_low[axis]
            or required_initial_action_xy[axis] > native_high[axis]
        )
    ]
    bounded_initial_action_xy = np.clip(
        required_initial_action_xy,
        native_low[:2],
        native_high[:2],
    )
    target_requires_native_clipping = bool(
        clipped_axes
        and required_initial_outward_action > native_outward_action_bound
    )
    target_outward_of_native_outside = float(
        np.dot(
            high_lateral_target[:2] - native_outside_high_target[:2],
            outward,
        )
    )

    window = list(observations)[-int(required_window_frames) :]
    summaries = []
    for offset, observation in enumerate(window):
        before_eef = np.asarray(observation["before_eef"], dtype=float)
        after_eef = np.asarray(observation["after_eef"], dtype=float)
        action = np.asarray(observation["action"], dtype=float)
        envelope = observation["high_plane_envelope"]
        pre_overhead_guard = observation["pre_overhead_guard"]
        post_overhead_guard = observation["post_overhead_guard"]
        before_outside_guard = observation["before_outside_guard"]
        after_outside_guard = observation["after_outside_guard"]
        step_response = observation["step_response"]
        if (
            before_eef.shape != (3,)
            or after_eef.shape != (3,)
            or action.shape != (7,)
            or not np.all(np.isfinite(before_eef))
            or not np.all(np.isfinite(after_eef))
            or not np.all(np.isfinite(action))
        ):
            raise ValueError(
                "high-plane workspace-saturation observation is invalid"
            )
        commanded_outward_action = float(np.dot(action[:2], outward))
        remaining_xy_error = float(
            np.linalg.norm(high_lateral_target[:2] - before_eef[:2])
        )
        remaining_outward_error = float(
            np.dot(high_lateral_target[:2] - before_eef[:2], outward)
        )
        eef_outward_progress = float(
            step_response["eef_outward_step_progress_m"]
        )
        clearance_progress = float(
            step_response["outside_clearance_step_progress_m"]
        )
        pre_pairs = list(pre_overhead_guard.get("pairs", ()))
        post_pairs = list(post_overhead_guard.get("pairs", ()))
        envelope_pairs = list(envelope.get("pair_envelopes", ()))
        pre_identities = [_overhead_pair_identity(pair) for pair in pre_pairs]
        post_identities = [
            _overhead_pair_identity(pair) for pair in post_pairs
        ]
        envelope_identities = [
            tuple(identity) for identity in envelope.get("pair_identity_keys", ())
        ]
        pair_inventory_exact = bool(
            len(pre_pairs) == int(expected_pair_count)
            and len(post_pairs) == int(expected_pair_count)
            and len(envelope_pairs) == int(expected_pair_count)
            and len(set(pre_identities)) == int(expected_pair_count)
            and pre_identities == post_identities == envelope_identities
        )
        all_pair_base8_strict = bool(
            pair_inventory_exact
            and pre_overhead_guard.get("accepted", False)
            and post_overhead_guard.get("accepted", False)
            and envelope.get("accepted", False)
            and all(pair.get("accepted", False) for pair in pre_pairs)
            and all(pair.get("accepted", False) for pair in post_pairs)
            and all(
                float(
                    pair[
                        "predicted_post_worst_case_base_reserve_surplus_m"
                    ]
                )
                > 0.0
                for pair in envelope_pairs
            )
        )
        before_clearance = float(
            before_outside_guard["minimum_outside_clearance_m"]
        )
        after_clearance = float(
            after_outside_guard["minimum_outside_clearance_m"]
        )
        required_outside_clearance = float(
            after_outside_guard["required_outside_clearance_m"]
        )
        finite_scalars = all(
            np.isfinite(value)
            for value in (
                commanded_outward_action,
                remaining_xy_error,
                remaining_outward_error,
                eef_outward_progress,
                clearance_progress,
                before_clearance,
                after_clearance,
                required_outside_clearance,
            )
        )
        summaries.append(
            {
                "window_offset": int(offset),
                "commanded_outward_action": commanded_outward_action,
                "remaining_xy_error_m": remaining_xy_error,
                "remaining_outward_error_m": remaining_outward_error,
                "eef_outward_step_progress_m": eef_outward_progress,
                "outside_clearance_step_progress_m": clearance_progress,
                "before_outside_clearance_m": before_clearance,
                "after_outside_clearance_m": after_clearance,
                "required_outside_clearance_m": required_outside_clearance,
                "persistent_outward_request": bool(
                    commanded_outward_action > 0.0
                    and remaining_outward_error > 0.0
                    and remaining_xy_error > position_tolerance
                    and np.array_equal(
                        np.asarray(envelope["lateral_target_xy"], dtype=float),
                        high_lateral_target[:2],
                    )
                    and action[2] >= 0.0
                    and np.all(action[3:6] == 0.0)
                ),
                "eef_step_below_existing_progress_epsilon": bool(
                    abs(eef_outward_progress) <= progress_epsilon
                ),
                "clearance_step_below_existing_progress_epsilon": bool(
                    abs(clearance_progress) <= progress_epsilon
                ),
                "pair_inventory_exact": pair_inventory_exact,
                "all_55_pair_high_plane_base8_strict": (
                    all_pair_base8_strict
                ),
                "finite": finite_scalars,
            }
        )

    window_complete = len(window) == int(required_window_frames)
    if summaries:
        first_before_eef = np.asarray(window[0]["before_eef"], dtype=float)
        last_after_eef = np.asarray(window[-1]["after_eef"], dtype=float)
        net_eef_outward_progress = float(
            np.dot(last_after_eef[:2] - first_before_eef[:2], outward)
        )
        first_before_clearance = float(
            window[0]["before_outside_guard"][
                "minimum_outside_clearance_m"
            ]
        )
        last_after_clearance = float(
            window[-1]["after_outside_guard"][
                "minimum_outside_clearance_m"
            ]
        )
        net_clearance_progress = float(
            last_after_clearance - first_before_clearance
        )
        actual_outward_beyond_native_outside = float(
            np.dot(
                last_after_eef[:2] - native_outside_high_target[:2],
                outward,
            )
        )
        final_required_outside_clearance = float(
            window[-1]["after_outside_guard"][
                "required_outside_clearance_m"
            ]
        )
    else:
        net_eef_outward_progress = float("inf")
        net_clearance_progress = float("inf")
        actual_outward_beyond_native_outside = float("-inf")
        last_after_clearance = float("-inf")
        final_required_outside_clearance = float("inf")
    violations = []
    if not target_requires_native_clipping:
        violations.append("registered_high_target_not_native_action_clipped")
    if target_outward_of_native_outside <= 0.0:
        violations.append("registered_high_target_not_outward_of_native_target")
    if not window_complete:
        violations.append("saturation_observation_window_incomplete")
    if not all(summary["finite"] for summary in summaries):
        violations.append("saturation_observation_nonfinite")
    if not all(summary["persistent_outward_request"] for summary in summaries):
        violations.append("outward_request_not_persistent")
    if not all(
        summary["eef_step_below_existing_progress_epsilon"]
        for summary in summaries
    ):
        violations.append("eef_outward_step_not_saturated")
    if not all(
        summary["clearance_step_below_existing_progress_epsilon"]
        for summary in summaries
    ):
        violations.append("outside_clearance_step_not_saturated")
    if abs(net_eef_outward_progress) > progress_epsilon:
        violations.append("window_net_eef_outward_progress_not_saturated")
    if abs(net_clearance_progress) > progress_epsilon:
        violations.append("window_net_clearance_progress_not_saturated")
    if not all(
        summary["all_55_pair_high_plane_base8_strict"]
        for summary in summaries
    ):
        violations.append("all_55_pair_high_plane_base8_not_strict")
    if actual_outward_beyond_native_outside <= 0.0:
        violations.append("actual_eef_not_outward_of_native_outside_target")
    if last_after_clearance <= final_required_outside_clearance:
        violations.append("actual_outside_clearance_not_strict")
    return {
        "accepted": not violations,
        "violations": violations,
        "decision": (
            "native_high_workspace_saturated_at_actual_reachable_boundary"
            if not violations
            else "continue_fail_closed_high_plane_request"
        ),
        "formula": (
            "over the existing push_tracking_steps window, require every "
            "high-plane action to request the registered target outward while "
            "both signed EEF and live-clearance step responses and their net "
            "window responses remain within the existing "
            "minimum_saturated_waypoint_progress; also require the registered "
            "target's initial action to exceed the runtime native outward "
            "bound, the actual EEF to lie strictly outward of the unchanged "
            "native outside-high target, and every pre/envelope/post compiled "
            "pair to retain strict high-plane base8"
        ),
        "progress_epsilon_m": float(progress_epsilon),
        "progress_epsilon_source": "minimum_saturated_waypoint_progress",
        "required_window_frames": int(required_window_frames),
        "required_window_source": "push_tracking_steps",
        "observed_window_frames": len(window),
        "window_complete": window_complete,
        "initial_eef": initial_eef.tolist(),
        "native_outside_high_target": native_outside_high_target.tolist(),
        "registered_high_lateral_target": high_lateral_target.tolist(),
        "outward_direction_xy": outward.tolist(),
        "required_initial_action_xy": required_initial_action_xy.tolist(),
        "bounded_initial_action_xy": bounded_initial_action_xy.tolist(),
        "native_outward_action_bound": native_outward_action_bound,
        "required_initial_outward_action": required_initial_outward_action,
        "bounded_action_clipped_axes": clipped_axes,
        "registered_target_requires_native_clipping": (
            target_requires_native_clipping
        ),
        "registered_target_outward_of_native_target_m": (
            target_outward_of_native_outside
        ),
        "window_net_eef_outward_progress_m": net_eef_outward_progress,
        "window_net_outside_clearance_progress_m": net_clearance_progress,
        "actual_eef_outward_of_native_outside_target_m": (
            actual_outward_beyond_native_outside
        ),
        "actual_minimum_outside_clearance_m": last_after_clearance,
        "required_outside_clearance_m": final_required_outside_clearance,
        "compiled_pair_count": int(expected_pair_count),
        "frames": summaries,
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


def _vertical_corridor_reserve_recovery_evidence(
    *,
    live_clearance_m,
    recovery_entry_clearance_m,
    recovery_exit_clearance_m,
    strict_corridor_entry_clearance_m,
    latest_outward_step_progress_m,
    latest_vertical_step_progress_m,
    compiled_tail_brake_buffer_accepted,
    recovery_active_before_decision,
):
    """Latch a positive-Z/outward brake before corridor reserve is lost."""
    scalars = (
        live_clearance_m,
        recovery_entry_clearance_m,
        recovery_exit_clearance_m,
        strict_corridor_entry_clearance_m,
        latest_outward_step_progress_m,
        latest_vertical_step_progress_m,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("vertical-corridor recovery evidence must be finite")
    if not (
        recovery_exit_clearance_m
        > recovery_entry_clearance_m
        > strict_corridor_entry_clearance_m
        >= 0.0
    ):
        raise ValueError(
            "vertical-corridor recovery exit must strictly exceed its "
            "entry and strict corridor clearances"
        )
    entered = bool(
        not recovery_active_before_decision
        and live_clearance_m <= recovery_entry_clearance_m
    )
    exit_accepted = bool(
        recovery_active_before_decision
        and live_clearance_m > recovery_exit_clearance_m
        and latest_outward_step_progress_m >= 0.0
        and latest_vertical_step_progress_m >= 0.0
        and compiled_tail_brake_buffer_accepted
    )
    recovery_active_after_decision = bool(
        (recovery_active_before_decision or entered) and not exit_accepted
    )
    if entered:
        event = "entered_pre_loss_reserve_recovery"
    elif exit_accepted:
        event = "exited_after_measured_outward_and_vertical_brake_recovery"
    elif recovery_active_after_decision:
        event = "continued_reserve_recovery"
    else:
        event = "descent_remains_authorized"
    return {
        "event": event,
        "recovery_active_before_decision": bool(
            recovery_active_before_decision
        ),
        "recovery_active_after_decision": (
            recovery_active_after_decision
        ),
        "entered_recovery": entered,
        "exit_accepted": exit_accepted,
        "live_clearance_m": float(live_clearance_m),
        "recovery_entry_clearance_m": float(
            recovery_entry_clearance_m
        ),
        "recovery_exit_clearance_m": float(
            recovery_exit_clearance_m
        ),
        "strict_corridor_entry_clearance_m": float(
            strict_corridor_entry_clearance_m
        ),
        "clearance_above_strict_threshold_m": float(
            live_clearance_m - strict_corridor_entry_clearance_m
        ),
        "latest_outward_step_progress_m": float(
            latest_outward_step_progress_m
        ),
        "latest_vertical_step_progress_m": float(
            latest_vertical_step_progress_m
        ),
        "compiled_tail_brake_buffer_accepted": bool(
            compiled_tail_brake_buffer_accepted
        ),
        "exit_requirements": (
            "live clearance strictly above the recovery exit gate plus "
            "measured nonnegative outward and vertical progress plus the "
            "refreshed accepted compiled lateral buffer"
        ),
    }


def _vertical_corridor_reserve_recovery_phase_evidence(
    *, recovery_evidence, phase_before_decision
):
    """Apply hysteresis across vertical brake, outward restore, and exit."""
    if not isinstance(recovery_evidence, dict):
        raise ValueError("vertical-corridor recovery evidence must be a dict")
    valid_phases = {"vertical_brake", "outward_restore", "exit_brake"}
    if (
        phase_before_decision is not None
        and phase_before_decision not in valid_phases
    ):
        raise ValueError("unknown vertical-corridor recovery phase")
    active = bool(
        recovery_evidence["recovery_active_after_decision"]
    )
    if not active:
        phase_after_decision = None
        transition = (
            "recovery_complete"
            if phase_before_decision is not None
            else "recovery_inactive"
        )
    elif recovery_evidence["entered_recovery"]:
        phase_after_decision = "vertical_brake"
        transition = "entered_vertical_brake"
    else:
        if phase_before_decision is None:
            raise RuntimeError(
                "active vertical-corridor recovery lacks a latched phase"
            )
        phase_after_decision = phase_before_decision
        transition = "phase_held"
        vertical_progress = float(
            recovery_evidence["latest_vertical_step_progress_m"]
        )
        outward_progress = float(
            recovery_evidence["latest_outward_step_progress_m"]
        )
        live_clearance = float(recovery_evidence["live_clearance_m"])
        entry_clearance = float(
            recovery_evidence["recovery_entry_clearance_m"]
        )
        exit_clearance = float(
            recovery_evidence["recovery_exit_clearance_m"]
        )
        if (
            phase_before_decision == "vertical_brake"
            and vertical_progress >= 0.0
        ):
            phase_after_decision = "outward_restore"
            transition = "vertical_brake_complete_to_outward_restore"
        elif (
            phase_before_decision == "outward_restore"
            and live_clearance > exit_clearance
            and outward_progress >= 0.0
        ):
            phase_after_decision = "exit_brake"
            transition = "outward_restore_complete_to_exit_brake"
        elif (
            phase_before_decision == "exit_brake"
            and live_clearance <= entry_clearance
        ):
            phase_after_decision = "vertical_brake"
            transition = "exit_brake_clearance_loss_to_vertical_brake"
    return {
        "phase_before_decision": phase_before_decision,
        "phase_after_decision": phase_after_decision,
        "phase_transition": transition,
        "outward_restore_hysteresis_active": bool(
            phase_after_decision == "outward_restore"
        ),
        "outward_restore_ignores_uncommanded_negative_z_until_clearance_gate": (
            True
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


def _strict_native_high_prebuffer_target(
    *,
    native_outside_high_target,
    corridor_high_target,
    outward_direction_xy,
    minimum_lateral_reserve_m,
    maximum_nextafter_steps=128,
):
    """Add only the representable native-tangent reserve needed at high Z."""
    native_target = np.asarray(native_outside_high_target, dtype=float)
    corridor_target = np.asarray(corridor_high_target, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    if (
        native_target.shape != (3,)
        or corridor_target.shape != (3,)
        or outward.shape != (2,)
        or not np.all(np.isfinite(native_target))
        or not np.all(np.isfinite(corridor_target))
        or not np.all(np.isfinite(outward))
        or not np.isfinite(minimum_lateral_reserve_m)
        or minimum_lateral_reserve_m <= 0.0
        or not isinstance(maximum_nextafter_steps, (int, np.integer))
        or not (1 <= int(maximum_nextafter_steps) <= 128)
    ):
        raise ValueError("native high-prebuffer inputs are invalid")
    outward_norm = float(np.linalg.norm(outward))
    if not np.isfinite(outward_norm) or outward_norm <= 1e-9:
        raise ValueError("native high-prebuffer direction is invalid")
    outward = outward / outward_norm

    raw_delta = corridor_target[:2] - native_target[:2]
    raw_euclidean_reserve = float(np.linalg.norm(raw_delta))
    raw_outward_projection = float(np.dot(raw_delta, outward))
    raw_transverse_residual = float(
        np.linalg.norm(raw_delta - raw_outward_projection * outward)
    )
    if not (
        np.isfinite(raw_euclidean_reserve)
        and np.isfinite(raw_outward_projection)
        and np.isfinite(raw_transverse_residual)
    ):
        raise RuntimeError("native high-prebuffer raw reserve is non-finite")
    if raw_outward_projection <= 0.0:
        raise RuntimeError(
            "corridor high target points against the normalized native "
            "outward tangent"
        )

    selected_target = corridor_target.copy()
    selected_scalar_reserve = raw_outward_projection
    final_euclidean_reserve = raw_euclidean_reserve
    final_outward_projection = raw_outward_projection
    nextafter_iterations = 0
    raw_reserve_is_strict = bool(
        raw_euclidean_reserve > minimum_lateral_reserve_m
        and raw_outward_projection > minimum_lateral_reserve_m
    )
    if not raw_reserve_is_strict:
        for nextafter_iterations in range(
            1, int(maximum_nextafter_steps) + 1
        ):
            selected_scalar_reserve = float(
                np.nextafter(selected_scalar_reserve, np.inf)
            )
            if not np.isfinite(selected_scalar_reserve):
                raise RuntimeError(
                    "native high-prebuffer scalar became non-finite"
                )
            selected_target = corridor_target.copy()
            selected_target[:2] = (
                native_target[:2] + outward * selected_scalar_reserve
            )
            if not np.all(np.isfinite(selected_target)):
                raise RuntimeError(
                    "native high-prebuffer target became non-finite"
                )
            final_delta = selected_target[:2] - native_target[:2]
            final_euclidean_reserve = float(np.linalg.norm(final_delta))
            final_outward_projection = float(np.dot(final_delta, outward))
            if not (
                np.isfinite(final_euclidean_reserve)
                and np.isfinite(final_outward_projection)
            ):
                raise RuntimeError(
                    "native high-prebuffer reconstructed reserve is "
                    "non-finite"
                )
            if (
                final_euclidean_reserve > minimum_lateral_reserve_m
                and final_outward_projection > minimum_lateral_reserve_m
            ):
                break
        else:
            raise RuntimeError(
                "native high-prebuffer exhausted 128 scalar nextafter steps "
                "without a representable strict reserve"
            )

    prebuffer_displacement = float(
        np.linalg.norm(selected_target[:2] - corridor_target[:2])
    )
    maximum_inward_return = float(
        np.nextafter(prebuffer_displacement, np.inf)
    )
    if not (
        np.all(np.isfinite(selected_target))
        and np.isfinite(prebuffer_displacement)
        and np.isfinite(maximum_inward_return)
        and final_euclidean_reserve > minimum_lateral_reserve_m
        and final_outward_projection > minimum_lateral_reserve_m
    ):
        raise RuntimeError(
            "native high-prebuffer final strict reserve is invalid"
        )
    return selected_target, {
        "accepted": True,
        "native_tangent_provenance": (
            "live geometry['outward_direction_xy'], normalized at high-plane "
            "route registration"
        ),
        "normalized_outward_direction_xy": outward.tolist(),
        "outward_direction_input_norm": outward_norm,
        "native_outside_high_target": native_target.tolist(),
        "unchanged_corridor_high_target": corridor_target.tolist(),
        "minimum_strict_lateral_reserve_m": float(
            minimum_lateral_reserve_m
        ),
        "raw_euclidean_reserve_m": raw_euclidean_reserve,
        "raw_outward_projection_m": raw_outward_projection,
        "raw_transverse_residual_m": raw_transverse_residual,
        "raw_euclidean_reserve_gap_m": float(
            raw_euclidean_reserve - minimum_lateral_reserve_m
        ),
        "raw_outward_projection_gap_m": float(
            raw_outward_projection - minimum_lateral_reserve_m
        ),
        "raw_reserve_was_already_strict": raw_reserve_is_strict,
        "selected_scalar_reserve_m": selected_scalar_reserve,
        "final_euclidean_reserve_m": final_euclidean_reserve,
        "final_outward_projection_m": final_outward_projection,
        "final_euclidean_reserve_surplus_m": float(
            final_euclidean_reserve - minimum_lateral_reserve_m
        ),
        "final_outward_projection_surplus_m": float(
            final_outward_projection - minimum_lateral_reserve_m
        ),
        "nextafter_iterations": int(nextafter_iterations),
        "maximum_nextafter_steps": int(maximum_nextafter_steps),
        "prebuffer_target": selected_target.tolist(),
        "prebuffer_displacement_from_corridor_m": prebuffer_displacement,
        "maximum_inward_return_one_ulp_bound_m": maximum_inward_return,
        "corridor_high_and_side_targets_unchanged": True,
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


def _native_osc_rotation_spec_evidence(env, native_action_spec):
    """Resolve live OSC axis-angle scaling without hardcoded controller data."""
    queue = [(env, "env")]
    visited = set()
    candidates = []
    while queue and len(visited) < 12:
        current, source = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        controller = getattr(current, "controller", None)
        if controller is not None:
            candidates.append((controller, f"{source}.controller"))
        robots = getattr(current, "robots", None)
        if robots is not None:
            for index, robot in enumerate(robots):
                queue.append((robot, f"{source}.robots[{index}]"))
        for attribute in ("env", "_env"):
            child = getattr(current, attribute, None)
            if child is not None:
                queue.append((child, f"{source}.{attribute}"))
    if len(candidates) != 1:
        raise RuntimeError(
            "native wrist-yaw execution requires exactly one live OSC "
            f"controller: observed={len(candidates)}"
        )
    controller, source = candidates[0]
    try:
        control_dim = int(controller.control_dim)
        input_min = np.asarray(controller.input_min, dtype=float)
        input_max = np.asarray(controller.input_max, dtype=float)
        output_min = np.asarray(controller.output_min, dtype=float)
        output_max = np.asarray(controller.output_max, dtype=float)
        use_delta = bool(controller.use_delta)
        use_ori = bool(controller.use_ori)
    except Exception as exc:
        raise RuntimeError(
            "live OSC controller scaling fields are unavailable"
        ) from exc
    if (
        control_dim != 6
        or input_min.shape != (6,)
        or input_max.shape != (6,)
        or output_min.shape != (6,)
        or output_max.shape != (6,)
        or not all(
            np.all(np.isfinite(values))
            for values in (input_min, input_max, output_min, output_max)
        )
        or not np.all(input_min < input_max)
        or not np.all(output_min < output_max)
        or not use_delta
        or not use_ori
    ):
        raise RuntimeError(
            "live controller is not a finite 6-D delta OSC pose controller"
        )
    action_low = np.asarray(native_action_spec.get("low", ()), dtype=float)
    action_high = np.asarray(native_action_spec.get("high", ()), dtype=float)
    if (
        action_low.shape != (7,)
        or action_high.shape != (7,)
        or not np.allclose(input_min, action_low[:6], rtol=0.0, atol=1e-12)
        or not np.allclose(input_max, action_high[:6], rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError(
            "live OSC controller input range diverges from native action_spec"
        )
    scale = (output_max - output_min) / (input_max - input_min)
    input_transform = 0.5 * (input_max + input_min)
    output_transform = 0.5 * (output_max + output_min)
    if (
        np.any(scale[3:6] <= 0.0)
        or not np.allclose(input_transform[3:6], 0.0, rtol=0.0, atol=1e-12)
        or not np.allclose(output_transform[3:6], 0.0, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError(
            "native OSC rotational scaling is not zero-centred and signed"
        )
    orientation_limits = getattr(controller, "orientation_limits", None)
    if orientation_limits is not None and np.asarray(
        orientation_limits
    ).any():
        raise RuntimeError(
            "native OSC orientation limits could silently clip wrist yaw"
        )
    return {
        "source": source,
        "controller_class": type(controller).__name__,
        "control_dim": control_dim,
        "use_delta": use_delta,
        "use_orientation": use_ori,
        "input_min": input_min.tolist(),
        "input_max": input_max.tolist(),
        "output_min_axis_angle_rad": output_min.tolist(),
        "output_max_axis_angle_rad": output_max.tolist(),
        "output_axis_angle_rad_per_action": scale.tolist(),
        "rotation_action_indices": [3, 4, 5],
        "world_axis_angle_pre_multiplies_current_orientation": True,
        "runtime_resolved": True,
    }


def _compiled_wrist_yaw_action(
    *,
    remaining_yaw_rad,
    remaining_axis_angle_world=None,
    table_normal_world,
    current_eef_position,
    anchor_eef_position,
    position_action_scale,
    maximum_translation_action,
    gripper,
    native_action_spec,
    rotation_spec,
):
    """Combine bounded anchor correction with one relative OSC yaw action."""
    normal = np.asarray(table_normal_world, dtype=float)
    current_eef = np.asarray(current_eef_position, dtype=float)
    anchor_eef = np.asarray(anchor_eef_position, dtype=float)
    low = np.asarray(native_action_spec.get("low", ()), dtype=float)
    high = np.asarray(native_action_spec.get("high", ()), dtype=float)
    scale = np.asarray(
        rotation_spec.get("output_axis_angle_rad_per_action", ()),
        dtype=float,
    )
    if (
        not np.isfinite(remaining_yaw_rad)
        or normal.shape != (3,)
        or not np.all(np.isfinite(normal))
        or abs(np.linalg.norm(normal) - 1.0) > 1e-7
        or current_eef.shape != (3,)
        or anchor_eef.shape != (3,)
        or not np.all(np.isfinite(current_eef))
        or not np.all(np.isfinite(anchor_eef))
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(maximum_translation_action)
        or not (0.0 < maximum_translation_action <= 1.0)
        or low.shape != (7,)
        or high.shape != (7,)
        or scale.shape != (6,)
        or np.any(scale[3:6] <= 0.0)
        or not (low[6] <= gripper <= high[6])
    ):
        raise ValueError("wrist-yaw action inputs are invalid")
    anchor_error = anchor_eef - current_eef
    anchor_error_norm = float(np.linalg.norm(anchor_error))
    required_translation_action = anchor_error / float(
        position_action_scale
    )
    required_translation_norm = float(
        np.linalg.norm(required_translation_action)
    )
    strict_translation_bound = float(
        np.nextafter(maximum_translation_action, 0.0)
    )
    translation_bound_saturated = bool(
        required_translation_norm > strict_translation_bound
    )
    if translation_bound_saturated:
        bounded_translation_action = (
            required_translation_action
            * strict_translation_bound
            / required_translation_norm
        )
    else:
        bounded_translation_action = required_translation_action.copy()
    requested_axis_angle = (
        normal * float(remaining_yaw_rad)
        if remaining_axis_angle_world is None
        else np.asarray(remaining_axis_angle_world, dtype=float)
    )
    if requested_axis_angle.shape != (3,) or not np.all(
        np.isfinite(requested_axis_angle)
    ):
        raise ValueError("wrist-yaw remaining axis-angle must be finite and 3-D")
    required_rotation_action = requested_axis_angle / scale[3:6]
    bounded_rotation_action = np.clip(
        required_rotation_action, low[3:6], high[3:6]
    )
    translation_native_action = np.clip(
        bounded_translation_action, low[:3], high[:3]
    )
    translation_clipped_axes = [
        int(axis)
        for axis in np.flatnonzero(
            np.abs(
                bounded_translation_action - translation_native_action
            )
            > 1e-12
        )
    ]
    rotation_clipped_axes = [
        int(axis + 3)
        for axis in np.flatnonzero(
            np.abs(required_rotation_action - bounded_rotation_action) > 1e-12
        )
    ]
    action = np.zeros(7, dtype=float)
    action[:3] = translation_native_action
    action[3:6] = bounded_rotation_action
    action[6] = float(gripper)
    commanded_axis_angle = scale[3:6] * bounded_rotation_action
    commanded_yaw = float(np.dot(commanded_axis_angle, normal))
    requested_rotation_norm = float(np.linalg.norm(requested_axis_angle))
    commanded_rotation_alignment = float(
        np.dot(commanded_axis_angle, requested_axis_angle)
    )
    if requested_rotation_norm > 0.0 and commanded_rotation_alignment <= 0.0:
        raise RuntimeError(
            "native OSC axis-angle action has the wrong rotation direction"
        )
    commanded_world_correction = (
        float(position_action_scale) * translation_native_action
    )
    predicted_anchor_error = anchor_error - commanded_world_correction
    predicted_anchor_error_norm = float(
        np.linalg.norm(predicted_anchor_error)
    )
    position_correction_requested = bool(anchor_error_norm > 1e-12)
    if position_correction_requested:
        correction_direction_dot = float(
            np.dot(anchor_error, commanded_world_correction)
        )
        translation_direction_valid = bool(
            correction_direction_dot > 0.0
            and predicted_anchor_error_norm < anchor_error_norm
        )
    else:
        correction_direction_dot = 0.0
        translation_direction_valid = bool(
            np.linalg.norm(translation_native_action) <= 1e-12
        )
    clipped_axes = translation_clipped_axes + rotation_clipped_axes
    return action, {
        "remaining_yaw_rad": float(remaining_yaw_rad),
        "table_normal_world": normal.tolist(),
        "requested_world_axis_angle_rad": requested_axis_angle.tolist(),
        "requested_world_axis_angle_norm_rad": requested_rotation_norm,
        "required_rotation_action": required_rotation_action.tolist(),
        "bounded_rotation_action": bounded_rotation_action.tolist(),
        "commanded_world_axis_angle_rad": commanded_axis_angle.tolist(),
        "commanded_rotation_alignment_rad2": commanded_rotation_alignment,
        "commanded_yaw_rad": commanded_yaw,
        "anchor_eef_position_world": anchor_eef.tolist(),
        "current_eef_position_world": current_eef.tolist(),
        "anchor_position_error_world_m": anchor_error.tolist(),
        "anchor_position_error_norm_m": anchor_error_norm,
        "position_action_scale_m_per_action": float(
            position_action_scale
        ),
        "required_translation_action": (
            required_translation_action.tolist()
        ),
        "required_translation_action_norm": required_translation_norm,
        "bounded_translation_action": (
            bounded_translation_action.tolist()
        ),
        "commanded_translation_action": (
            translation_native_action.tolist()
        ),
        "commanded_translation_action_norm": float(
            np.linalg.norm(translation_native_action)
        ),
        "commanded_translation_action_peak": float(
            np.max(np.abs(translation_native_action))
        ),
        "strict_translation_action_norm_bound": strict_translation_bound,
        "maximum_translation_action": float(
            maximum_translation_action
        ),
        "translation_bound_saturated": translation_bound_saturated,
        "translation_native_clipped_axes": translation_clipped_axes,
        "commanded_world_position_correction_m": (
            commanded_world_correction.tolist()
        ),
        "predicted_anchor_position_error_norm_m": (
            predicted_anchor_error_norm
        ),
        "predicted_anchor_error_reduction_m": float(
            anchor_error_norm - predicted_anchor_error_norm
        ),
        "anchor_error_correction_direction_dot_m2": (
            correction_direction_dot
        ),
        "position_correction_requested": position_correction_requested,
        "translation_direction_valid": translation_direction_valid,
        "orientation_hold_commanded": bool(
            requested_rotation_norm == 0.0
            and np.all(bounded_rotation_action == 0.0)
        ),
        "clipped_action_axes": clipped_axes,
        "action_will_clip": bool(clipped_axes),
        "zero_translation": bool(np.all(action[:3] == 0.0)),
        "native_rotation_spec_source": rotation_spec.get("source"),
    }


def _compiled_adaptive_vertical_descent_action(
    *,
    current_eef,
    target_z,
    overhead_guard,
    gripper,
    position_action_scale,
    native_action_spec,
    expected_pair_count,
    maximum_translation_action=None,
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
    configured_strict_action_capacity = None
    configured_world_capacity = None
    if maximum_translation_action is not None:
        maximum_translation_action = float(maximum_translation_action)
        if (
            not np.isfinite(maximum_translation_action)
            or maximum_translation_action <= 0.0
            or maximum_translation_action
            > strict_native_negative_z_action
        ):
            raise ValueError(
                "adaptive vertical translation-action bound must be finite, "
                "positive, and strictly inside the runtime native negative-Z "
                "action capacity"
            )
        configured_strict_action_capacity = float(
            np.nextafter(maximum_translation_action, 0.0)
        )
        configured_world_capacity = float(
            configured_strict_action_capacity * position_action_scale
        )
        if (
            configured_strict_action_capacity <= 0.0
            or configured_world_capacity <= 0.0
        ):
            raise RuntimeError(
                "adaptive vertical configured action bound has no strict "
                "interior"
            )
    capacities = {
        "target_remaining_z_error": target_remaining,
        "compiled_pair_base8_envelope": pair_world_capacity,
        "native_negative_z_action_bound": native_world_capacity,
    }
    if configured_world_capacity is not None:
        capacities["configured_translation_action_norm_bound"] = (
            configured_world_capacity
        )
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
            and (
                configured_strict_action_capacity is None
                or abs(commanded_z_action)
                <= configured_strict_action_capacity
            )
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
        or (
            configured_strict_action_capacity is not None
            and abs(action[2]) > configured_strict_action_capacity
        )
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
            "negative-Z action capacity times position_action_scale; when "
            "configured, also intersect the strict structural translation-"
            "action bound from the first descent action"
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
        "configured_maximum_translation_action": (
            maximum_translation_action
        ),
        "configured_strict_translation_action_capacity": (
            configured_strict_action_capacity
        ),
        "configured_strict_world_delta_capacity_m": (
            configured_world_capacity
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


def _compiled_adaptive_high_plane_action(
    *,
    current_eef,
    lateral_target_xy,
    overhead_horizontal_z,
    measured_vertical_step_progress_m,
    overhead_guard,
    gripper,
    position_action_scale,
    native_action_spec,
    expected_pair_count,
    maximum_translation_action=None,
    plane_recovery_tolerance_m=None,
    negative_tail_recovery_threshold_m=None,
):
    """Hold the initial high plane with XY/+Z under live pair reserves."""
    current_eef = np.asarray(current_eef, dtype=float)
    lateral_target_xy = np.asarray(lateral_target_xy, dtype=float)
    scalars = (
        overhead_horizontal_z,
        measured_vertical_step_progress_m,
        position_action_scale,
    )
    if (
        current_eef.shape != (3,)
        or lateral_target_xy.shape != (2,)
        or not np.all(np.isfinite(current_eef))
        or not np.all(np.isfinite(lateral_target_xy))
        or not all(np.isfinite(value) for value in scalars)
        or position_action_scale <= 0.0
    ):
        raise ValueError("adaptive high-plane geometry is invalid")
    if not isinstance(expected_pair_count, (int, np.integer)):
        raise ValueError("expected compiled pair count must be an integer")
    if plane_recovery_tolerance_m is not None:
        plane_recovery_tolerance_m = float(plane_recovery_tolerance_m)
        if (
            not np.isfinite(plane_recovery_tolerance_m)
            or plane_recovery_tolerance_m <= 0.0
        ):
            raise ValueError(
                "adaptive high-plane recovery tolerance must be finite and "
                "positive"
            )
    if negative_tail_recovery_threshold_m is not None:
        negative_tail_recovery_threshold_m = float(
            negative_tail_recovery_threshold_m
        )
        if (
            not np.isfinite(negative_tail_recovery_threshold_m)
            or negative_tail_recovery_threshold_m <= 0.0
        ):
            raise ValueError(
                "adaptive high-plane negative-tail recovery threshold must "
                "be finite and positive"
            )
    pairs = list(overhead_guard.get("pairs", ()))
    if expected_pair_count <= 0 or len(pairs) != int(expected_pair_count):
        raise RuntimeError(
            "live compiled overhead pair inventory changed before adaptive "
            f"high-plane action: expected={expected_pair_count} "
            f"observed={len(pairs)}"
        )
    identities = [_overhead_pair_identity(pair) for pair in pairs]
    if len(set(identities)) != len(identities):
        raise RuntimeError(
            "adaptive high-plane evidence contains duplicate pair identity"
        )
    if not overhead_guard.get("accepted", False):
        raise RuntimeError(
            "adaptive high-plane action cannot start after base8 is lost"
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
            "native OSC action bounds do not prove an XY/nonnegative-Z action"
        )
    native_norm_bound = float(
        min(
            -native_low[0],
            native_high[0],
            -native_low[1],
            native_high[1],
            native_high[2],
        )
    )
    strict_native_norm_bound = float(
        np.nextafter(native_norm_bound, 0.0)
    )
    configured_strict_norm_bound = None
    if maximum_translation_action is not None:
        maximum_translation_action = float(maximum_translation_action)
        if (
            not np.isfinite(maximum_translation_action)
            or maximum_translation_action <= 0.0
            or maximum_translation_action > strict_native_norm_bound
        ):
            raise ValueError(
                "adaptive high-plane translation-action bound must be "
                "finite, positive, and strictly inside the runtime native "
                "3-D translation-action capacity"
            )
        configured_strict_norm_bound = float(
            np.nextafter(maximum_translation_action, 0.0)
        )
    route_strict_norm_bound = float(
        min(
            strict_native_norm_bound,
            (
                configured_strict_norm_bound
                if configured_strict_norm_bound is not None
                else strict_native_norm_bound
            ),
        )
    )
    if route_strict_norm_bound <= 0.0:
        raise RuntimeError("native strict 3-D action norm is unavailable")
    base_reserve = float(overhead_guard["one_step_vertical_reserve_m"])
    if not np.isfinite(base_reserve) or base_reserve <= 0.0:
        raise RuntimeError("compiled overhead base reserve is invalid")

    lateral_delta = lateral_target_xy - current_eef[:2]
    lateral_remaining = float(np.linalg.norm(lateral_delta))
    plane_hold_z_error = float(
        max(0.0, float(overhead_horizontal_z) - current_eef[2])
    )
    requested = np.array(
        [
            lateral_delta[0] / position_action_scale,
            lateral_delta[1] / position_action_scale,
            plane_hold_z_error / position_action_scale,
        ],
        dtype=float,
    )
    requested_norm = float(np.linalg.norm(requested))
    if not np.isfinite(requested_norm) or requested_norm <= 0.0:
        raise RuntimeError(
            "adaptive high-plane action has no positive XY/Z-hold error"
        )
    requested_direction = requested / requested_norm
    measured_negative_tail = float(
        max(0.0, -float(measured_vertical_step_progress_m))
    )
    inertial_tail_reserve = (
        float(np.nextafter(measured_negative_tail, np.inf))
        if measured_negative_tail > 0.0
        else 0.0
    )
    pair_envelopes = []
    for index, (identity, pair) in enumerate(zip(identities, pairs)):
        vertical_clearance = float(pair["vertical_clearance_m"])
        strict_clearance = float(pair["strict_no_contact_clearance_m"])
        required_clearance = float(strict_clearance + base_reserve)
        current_surplus = float(vertical_clearance - required_clearance)
        nominal_tail_capacity = float(
            current_surplus - inertial_tail_reserve
        )
        if (
            not np.isfinite(vertical_clearance)
            or not np.isfinite(strict_clearance)
            or strict_clearance < 0.0
            or not np.isfinite(current_surplus)
            or current_surplus <= 0.0
            or not pair.get("accepted", False)
        ):
            raise RuntimeError(
                "compiled overhead pair cannot prove positive adaptive "
                f"high-plane capacity: index={index} pair="
                f"{json.dumps(pair, sort_keys=True)}"
            )
        if nominal_tail_capacity <= 0.0:
            raise RuntimeError(
                "latest measured negative-dz inertial tail consumes the live "
                f"base8 surplus before recovery: index={index}"
            )
        strict_nominal_tail_capacity = float(
            np.nextafter(nominal_tail_capacity, 0.0)
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
                "required_clearance_with_base_reserve_m": required_clearance,
                "current_base8_surplus_m": current_surplus,
                "measured_negative_inertial_tail_reserve_m": (
                    inertial_tail_reserve
                ),
                "strict_nominal_tail_capacity_m": (
                    strict_nominal_tail_capacity
                ),
                "strict_safe_translation_action_norm_capacity": float(
                    strict_nominal_tail_capacity / position_action_scale
                ),
            }
        )
    limiting_pair = min(
        pair_envelopes,
        key=lambda record: record[
            "strict_safe_translation_action_norm_capacity"
        ],
    )
    pair_norm_capacity = float(
        limiting_pair["strict_safe_translation_action_norm_capacity"]
    )
    capacities = {
        "requested_xy_plus_plane_hold_action_norm": requested_norm,
        "compiled_pair_base8_nominal_tail_after_inertia": pair_norm_capacity,
        "native_strict_3d_translation_action_norm_bound": (
            strict_native_norm_bound
        ),
    }
    if configured_strict_norm_bound is not None:
        capacities["configured_translation_action_norm_bound"] = (
            configured_strict_norm_bound
        )
    selected_source = min(capacities, key=capacities.get)
    selected_norm = float(capacities[selected_source])
    pair_capacity_recovery_required = bool(
        selected_source
        == "compiled_pair_base8_nominal_tail_after_inertia"
        and selected_norm
        < min(requested_norm, route_strict_norm_bound)
    )
    plane_tolerance_recovery_required = bool(
        plane_recovery_tolerance_m is not None
        and plane_hold_z_error > plane_recovery_tolerance_m
    )
    negative_tail_recovery_required = bool(
        negative_tail_recovery_threshold_m is not None
        and measured_negative_tail > negative_tail_recovery_threshold_m
    )
    recovery_required = bool(
        pair_capacity_recovery_required
        or plane_tolerance_recovery_required
        or negative_tail_recovery_required
    )
    pure_positive_z_recovery = bool(pair_capacity_recovery_required)
    dynamic_xy_positive_z_recovery = bool(
        recovery_required and not pure_positive_z_recovery
    )
    minimum_current_surplus = min(
        record["current_base8_surplus_m"] for record in pair_envelopes
    )
    recovery_requested_translation_action = None
    recovery_requested_translation_action_norm = None
    recovery_action_norm_capacities = None
    if recovery_required:
        requested_nominal_tail = float(
            position_action_scale
            * min(requested_norm, route_strict_norm_bound)
        )
        recovery_world_delta = float(
            max(
                plane_hold_z_error,
                inertial_tail_reserve,
                requested_nominal_tail
                + inertial_tail_reserve
                - minimum_current_surplus,
            )
        )
        if pure_positive_z_recovery:
            recovery_z_action = float(
                min(
                    route_strict_norm_bound,
                    recovery_world_delta / position_action_scale,
                )
            )
            if recovery_z_action <= 0.0:
                raise RuntimeError(
                    "event-driven high-plane recovery has no positive +Z "
                    "action"
                )
            translation = np.array([0.0, 0.0, recovery_z_action])
            nominal_tail = 0.0
            total_tail = inertial_tail_reserve
            selected_source = "event_driven_positive_z_plane_recovery"
        else:
            recovery_requested_translation_action = np.array(
                [
                    requested[0],
                    requested[1],
                    recovery_world_delta / position_action_scale,
                ],
                dtype=float,
            )
            recovery_requested_translation_action_norm = float(
                np.linalg.norm(recovery_requested_translation_action)
            )
            if recovery_requested_translation_action_norm <= 0.0:
                raise RuntimeError(
                    "event-driven high-plane XY/+Z recovery has no positive "
                    "action"
                )
            recovery_action_norm_capacities = {
                "requested_xy_plus_recovery_z_action_norm": (
                    recovery_requested_translation_action_norm
                ),
                "compiled_pair_base8_nominal_tail_after_inertia": (
                    pair_norm_capacity
                ),
                "route_strict_3d_translation_action_norm_bound": (
                    route_strict_norm_bound
                ),
            }
            recovery_selected_norm = float(
                min(recovery_action_norm_capacities.values())
            )
            translation = (
                recovery_requested_translation_action
                / recovery_requested_translation_action_norm
                * recovery_selected_norm
            )
            nominal_tail = float(
                position_action_scale * np.linalg.norm(translation)
            )
            total_tail = float(nominal_tail + inertial_tail_reserve)
            if negative_tail_recovery_required:
                selected_source = (
                    "event_driven_xy_positive_z_negative_tail_recovery"
                )
            else:
                selected_source = (
                    "event_driven_xy_positive_z_plane_tolerance_recovery"
                )
    else:
        translation = requested_direction * selected_norm
        nominal_tail = float(
            position_action_scale * np.linalg.norm(translation)
        )
        total_tail = float(nominal_tail + inertial_tail_reserve)

    for _ in range(128):
        literal_norm = float(np.linalg.norm(translation))
        literal_xy_world_delta = float(
            position_action_scale * np.linalg.norm(translation[:2])
        )
        predicted = [
            float(record["current_vertical_clearance_m"] - total_tail)
            for record in pair_envelopes
        ]
        if (
            0.0 < literal_norm < native_norm_bound
            and (
                configured_strict_norm_bound is None
                or literal_norm <= configured_strict_norm_bound
            )
            and literal_xy_world_delta <= lateral_remaining
            and translation[2] >= 0.0
            and all(
                native_low[index] < translation[index] < native_high[index]
                for index in range(3)
            )
            and all(
                clearance
                > record["required_clearance_with_base_reserve_m"]
                for clearance, record in zip(predicted, pair_envelopes)
            )
        ):
            break
        if pure_positive_z_recovery:
            raise RuntimeError(
                "event-driven +Z recovery cannot retain measured inertial "
                "tail and base8"
            )
        translation = np.nextafter(translation, 0.0)
        nominal_tail = float(
            position_action_scale * np.linalg.norm(translation)
        )
        total_tail = float(nominal_tail + inertial_tail_reserve)
    else:
        raise RuntimeError(
            "adaptive high-plane literal action lacks strict native/base8 "
            "interior"
        )

    minimum_predicted_surplus = float("inf")
    for clearance, record in zip(predicted, pair_envelopes):
        record["predicted_post_worst_case_vertical_clearance_m"] = clearance
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
    action[:3] = translation
    action[-1] = float(gripper)
    if (
        action[2] < 0.0
        or np.any(action[3:6] != 0.0)
        or literal_norm >= native_norm_bound
        or minimum_predicted_surplus <= 0.0
    ):
        raise RuntimeError(
            "adaptive high-plane action violated its compiled hard proof"
        )
    return action, {
        "accepted": True,
        "formula": (
            "request [XY error, max(0, initial center-high Z minus current Z)] "
            "over position_action_scale; intersect its norm with the strict "
            "runtime-native 3-D norm and every exact pair's strict+base8 "
            "capacity after reserving at least the latest measured negative-dz "
            "inertial tail; a pair-limited action becomes pure +Z recovery, "
            "and optional registered plane-error or measured-negative-tail "
            "thresholds require outward XY/+Z recovery under the same full-"
            "norm proof so real-controller inward coupling is not left "
            "unopposed"
        ),
        "current_eef": current_eef.tolist(),
        "lateral_target_xy": lateral_target_xy.tolist(),
        "overhead_horizontal_z_m": float(overhead_horizontal_z),
        "lateral_remaining_m": lateral_remaining,
        "plane_hold_z_error_m": plane_hold_z_error,
        "plane_recovery_tolerance_m": plane_recovery_tolerance_m,
        "plane_tolerance_recovery_required": (
            plane_tolerance_recovery_required
        ),
        "negative_tail_recovery_threshold_m": (
            negative_tail_recovery_threshold_m
        ),
        "negative_tail_recovery_required": (
            negative_tail_recovery_required
        ),
        "pair_capacity_recovery_required": (
            pair_capacity_recovery_required
        ),
        "pure_positive_z_recovery": pure_positive_z_recovery,
        "dynamic_xy_positive_z_recovery": (
            dynamic_xy_positive_z_recovery
        ),
        "recovery_requested_translation_action": (
            None
            if recovery_requested_translation_action is None
            else recovery_requested_translation_action.tolist()
        ),
        "recovery_requested_translation_action_norm": (
            recovery_requested_translation_action_norm
        ),
        "recovery_action_norm_capacities": (
            recovery_action_norm_capacities
        ),
        "requested_translation_action": requested.tolist(),
        "requested_translation_action_norm": requested_norm,
        "position_action_scale_m_per_normalized_action": float(
            position_action_scale
        ),
        "measured_vertical_step_progress_m": float(
            measured_vertical_step_progress_m
        ),
        "measured_negative_inertial_tail_reserve_m": inertial_tail_reserve,
        "native_action_spec_source": native_source,
        "native_3d_translation_action_norm_bound": native_norm_bound,
        "strict_native_3d_translation_action_norm_bound": (
            strict_native_norm_bound
        ),
        "configured_strict_translation_action_norm_bound": (
            configured_strict_norm_bound
        ),
        "compiled_pair_count": len(pair_envelopes),
        "pair_identity_keys": [list(identity) for identity in identities],
        "pair_envelopes": pair_envelopes,
        "selected_limiting_pair": dict(limiting_pair),
        "candidate_action_norm_capacities": capacities,
        "selected_envelope_source": selected_source,
        "event_driven_positive_z_recovery": recovery_required,
        "commanded_translation_action_norm": literal_norm,
        "commanded_nominal_norm_downward_tail_m": nominal_tail,
        "commanded_worst_case_downward_world_tail_m": total_tail,
        "commanded_xy_action": action[:2].tolist(),
        "commanded_z_action": float(action[2]),
        "minimum_predicted_post_worst_case_base_surplus_m": (
            minimum_predicted_surplus
        ),
        "proof": {
            "xy_plus_nonnegative_z_zero_rotation": True,
            "strictly_inside_native_3d_action_norm_bound": True,
            "inside_configured_translation_action_norm_bound": bool(
                configured_strict_norm_bound is None
                or literal_norm <= configured_strict_norm_bound
            ),
            "does_not_cross_lateral_target_xy": bool(
                literal_xy_world_delta <= lateral_remaining
            ),
            "positive_z_static_geometry_does_not_reduce_clearance": True,
            "latest_measured_negative_dz_reserved_as_inertial_tail": True,
            "all_compiled_pairs_retain_strict_base8_after_worst_case_tail": (
                True
            ),
        },
    }


def _compiled_adaptive_workspace_release_action(
    *,
    current_eef,
    corridor_target_xy,
    release_target_z,
    measured_vertical_step_progress_m,
    overhead_guard,
    gripper,
    position_action_scale,
    native_action_spec,
    expected_pair_count,
    worst_case_controller_world_step_m,
    outward_direction_xy=None,
    maximum_inward_xy_correction_m=None,
    couple_downward_to_lateral_remaining=True,
    maximum_translation_action=None,
    one_sided_outward_direction_xy=None,
):
    """Move along a corridor/downward route under the live buffer16 proof."""
    current_eef = np.asarray(current_eef, dtype=float)
    corridor_target_xy = np.asarray(corridor_target_xy, dtype=float)
    if not isinstance(
        couple_downward_to_lateral_remaining, (bool, np.bool_)
    ):
        raise ValueError(
            "workspace-release downward/lateral coupling flag is invalid"
        )
    couple_downward_to_lateral_remaining = bool(
        couple_downward_to_lateral_remaining
    )
    one_sided_outward_direction = None
    if one_sided_outward_direction_xy is not None:
        one_sided_outward_direction = np.asarray(
            one_sided_outward_direction_xy, dtype=float
        )
        if (
            one_sided_outward_direction.shape != (2,)
            or not np.all(np.isfinite(one_sided_outward_direction))
        ):
            raise ValueError(
                "workspace-release one-sided outward direction is invalid"
            )
        one_sided_outward_norm = float(
            np.linalg.norm(one_sided_outward_direction)
        )
        if not np.isfinite(one_sided_outward_norm) or (
            one_sided_outward_norm <= 1e-9
        ):
            raise ValueError(
                "workspace-release one-sided outward direction is invalid"
            )
        one_sided_outward_direction = (
            one_sided_outward_direction / one_sided_outward_norm
        )
    native_tangent_return_gate_enabled = bool(
        outward_direction_xy is not None
        or maximum_inward_xy_correction_m is not None
    )
    if (
        one_sided_outward_direction is not None
        and native_tangent_return_gate_enabled
    ):
        raise ValueError(
            "workspace-release one-sided hold conflicts with the native-"
            "tangent return gate"
        )
    if (outward_direction_xy is None) != (
        maximum_inward_xy_correction_m is None
    ):
        raise ValueError(
            "workspace-release native-tangent return gate is incomplete"
        )
    normalized_outward_direction = None
    maximum_inward_xy_correction = None
    if native_tangent_return_gate_enabled:
        normalized_outward_direction = np.asarray(
            outward_direction_xy, dtype=float
        )
        if (
            normalized_outward_direction.shape != (2,)
            or not np.all(np.isfinite(normalized_outward_direction))
            or not np.isfinite(maximum_inward_xy_correction_m)
            or maximum_inward_xy_correction_m < 0.0
        ):
            raise ValueError(
                "workspace-release native-tangent return gate is invalid"
            )
        outward_norm = float(np.linalg.norm(normalized_outward_direction))
        if not np.isfinite(outward_norm) or outward_norm <= 1e-9:
            raise ValueError(
                "workspace-release native-tangent direction is invalid"
            )
        normalized_outward_direction = (
            normalized_outward_direction / outward_norm
        )
        maximum_inward_xy_correction = float(
            maximum_inward_xy_correction_m
        )
    if (
        current_eef.shape != (3,)
        or corridor_target_xy.shape != (2,)
        or not np.all(np.isfinite(current_eef))
        or not np.all(np.isfinite(corridor_target_xy))
        or not np.isfinite(release_target_z)
        or not np.isfinite(measured_vertical_step_progress_m)
        or not np.isfinite(position_action_scale)
        or position_action_scale <= 0.0
        or not np.isfinite(worst_case_controller_world_step_m)
        or worst_case_controller_world_step_m <= 0.0
    ):
        raise ValueError("workspace-release geometry is invalid")
    pairs = list(overhead_guard.get("pairs", ()))
    if (
        not isinstance(expected_pair_count, (int, np.integer))
        or expected_pair_count <= 0
        or len(pairs) != int(expected_pair_count)
    ):
        raise RuntimeError(
            "live compiled pair inventory changed before workspace release"
        )
    identities = [_overhead_pair_identity(pair) for pair in pairs]
    if len(set(identities)) != len(identities):
        raise RuntimeError("workspace-release pair identity is duplicated")
    if not overhead_guard.get("accepted", False):
        raise RuntimeError("workspace release cannot start after base8 is lost")
    native_low = np.asarray(native_action_spec["low"], dtype=float)
    native_high = np.asarray(native_action_spec["high"], dtype=float)
    if (
        not native_action_spec.get("runtime_resolved", False)
        or native_action_spec.get("action_dimension") != 7
        or native_low.shape != (7,)
        or native_high.shape != (7,)
        or not np.all(np.isfinite(native_low))
        or not np.all(np.isfinite(native_high))
        or not np.all(native_low < native_high)
        or not (native_low[6] <= gripper <= native_high[6])
    ):
        raise RuntimeError("native OSC action bounds are invalid for release")
    native_norm_bound = float(
        min(
            -native_low[0],
            native_high[0],
            -native_low[1],
            native_high[1],
            -native_low[2],
        )
    )
    strict_native_norm_bound = float(
        np.nextafter(native_norm_bound, 0.0)
    )
    configured_strict_norm_bound = None
    if maximum_translation_action is not None:
        maximum_translation_action = float(maximum_translation_action)
        if (
            not np.isfinite(maximum_translation_action)
            or maximum_translation_action <= 0.0
            or maximum_translation_action > strict_native_norm_bound
        ):
            raise ValueError(
                "workspace-release translation-action bound must be finite, "
                "positive, and strictly inside the runtime native 3-D "
                "translation-action capacity"
            )
        configured_strict_norm_bound = float(
            np.nextafter(maximum_translation_action, 0.0)
        )
        if configured_strict_norm_bound <= 0.0:
            raise RuntimeError(
                "workspace-release configured action bound has no strict "
                "interior"
            )
    route_strict_norm_bound = float(
        min(
            strict_native_norm_bound,
            (
                configured_strict_norm_bound
                if configured_strict_norm_bound is not None
                else strict_native_norm_bound
            ),
        )
    )
    base_reserve = float(overhead_guard["one_step_vertical_reserve_m"])
    if route_strict_norm_bound <= 0.0 or base_reserve <= 0.0:
        raise RuntimeError("workspace-release native/base8 bound is invalid")

    raw_xy_error = corridor_target_xy - current_eef[:2]
    raw_one_sided_outward_error = None
    suppressed_inward_outward_axis_error = 0.0
    xy_error = raw_xy_error.copy()
    if one_sided_outward_direction is not None:
        raw_one_sided_outward_error = float(
            np.dot(raw_xy_error, one_sided_outward_direction)
        )
        suppressed_inward_outward_axis_error = float(
            max(0.0, -raw_one_sided_outward_error)
        )
        tangential_error = (
            raw_xy_error
            - raw_one_sided_outward_error
            * one_sided_outward_direction
        )
        xy_error = (
            max(0.0, raw_one_sided_outward_error)
            * one_sided_outward_direction
            + tangential_error
        )
    xy_remaining = float(np.linalg.norm(xy_error))
    requested_outward_projection = None
    requested_inward_xy_correction = 0.0
    if native_tangent_return_gate_enabled:
        requested_outward_projection = float(
            np.dot(xy_error, normalized_outward_direction)
        )
        requested_inward_xy_correction = float(
            max(0.0, -requested_outward_projection)
        )
        if not (
            np.isfinite(requested_outward_projection)
            and np.isfinite(requested_inward_xy_correction)
        ):
            raise RuntimeError(
                "workspace-release native-tangent return is non-finite"
            )
        if requested_inward_xy_correction > (
            maximum_inward_xy_correction
        ):
            raise RuntimeError(
                "workspace-release inward correction exceeds the registered "
                "high-prebuffer one-ULP bound"
            )
    full_downward_z_error = float(
        max(0.0, current_eef[2] - release_target_z)
    )
    xy_coupled_downward_world_request = float(
        min(full_downward_z_error, xy_remaining)
    )
    route_downward_world_request = float(
        xy_coupled_downward_world_request
        if couple_downward_to_lateral_remaining
        else full_downward_z_error
    )
    downward_world_request = float(
        min(
            route_downward_world_request,
            worst_case_controller_world_step_m,
        )
    )
    downward_world_request_before_live_buffer_headroom_cap = float(
        downward_world_request
    )
    downward_z_error = float(-downward_world_request)
    requested = np.array(
        [
            xy_error[0] / position_action_scale,
            xy_error[1] / position_action_scale,
            downward_z_error / position_action_scale,
        ]
    )
    recovery_route_requested = np.array(
        [
            xy_error[0] / position_action_scale,
            xy_error[1] / position_action_scale,
            -route_downward_world_request / position_action_scale,
        ]
    )
    requested_norm = float(np.linalg.norm(requested))
    recovery_route_requested_norm = float(
        np.linalg.norm(recovery_route_requested)
    )
    if (
        not np.isfinite(requested_norm)
        or requested_norm <= 0.0
        or not np.isfinite(recovery_route_requested_norm)
        or recovery_route_requested_norm <= 0.0
    ):
        raise RuntimeError("workspace release has no positive route error")
    requested_direction = requested / requested_norm
    measured_negative_tail = float(
        max(0.0, -float(measured_vertical_step_progress_m))
    )
    inertial_tail_reserve = (
        float(np.nextafter(measured_negative_tail, np.inf))
        if measured_negative_tail > 0.0
        else 0.0
    )
    if not np.isfinite(inertial_tail_reserve):
        raise RuntimeError(
            "workspace-release live pre-action buffer16 headroom input is "
            "non-finite"
        )

    pair_envelopes = []
    for index, (identity, pair) in enumerate(zip(identities, pairs)):
        clearance = float(pair["vertical_clearance_m"])
        strict_clearance = float(pair["strict_no_contact_clearance_m"])
        required_base8 = float(strict_clearance + base_reserve)
        required_buffer16 = float(
            required_base8 + worst_case_controller_world_step_m
        )
        current_base8_surplus = float(clearance - required_base8)
        current_buffer16_surplus = float(clearance - required_buffer16)
        pre_action_buffer16_capacity = float(
            current_buffer16_surplus - inertial_tail_reserve
        )
        post_base8_action_capacity = float(
            current_base8_surplus - inertial_tail_reserve
        )
        if (
            not np.isfinite(clearance)
            or not np.isfinite(strict_clearance)
            or strict_clearance < 0.0
            or not np.isfinite(current_base8_surplus)
            or current_base8_surplus <= 0.0
            or not np.isfinite(pre_action_buffer16_capacity)
            or not np.isfinite(post_base8_action_capacity)
            or not pair.get("accepted", False)
        ):
            raise RuntimeError(
                "workspace-release pair lacks current strict base8 "
                f"capacity: index={index}"
            )
        strict_post_base8_action_capacity = (
            float(np.nextafter(post_base8_action_capacity, 0.0))
            if post_base8_action_capacity > 0.0
            else 0.0
        )
        pair_envelopes.append(
            {
                "pair_index": int(index),
                "pair_identity": list(identity),
                "gripper_geom": pair["gripper_geom"],
                "counterpart_geom": pair["counterpart_geom"],
                "counterpart_kind": pair["counterpart_kind"],
                "current_vertical_clearance_m": clearance,
                "strict_no_contact_clearance_m": strict_clearance,
                "base_overhead_reserve_m": base_reserve,
                "worst_case_controller_world_step_m": float(
                    worst_case_controller_world_step_m
                ),
                "required_clearance_with_base_reserve_m": required_base8,
                "required_clearance_with_fixed_buffer16_m": (
                    required_buffer16
                ),
                "current_base8_surplus_m": current_base8_surplus,
                "current_buffer16_surplus_m": current_buffer16_surplus,
                "measured_negative_inertial_tail_reserve_m": (
                    inertial_tail_reserve
                ),
                "pre_action_buffer16_surplus_after_inertia_m": (
                    pre_action_buffer16_capacity
                ),
                "strict_nominal_tail_capacity_m": (
                    strict_post_base8_action_capacity
                ),
                "nominal_negative_z_capacity_after_buffer16_and_inertia_m": (
                    pre_action_buffer16_capacity
                ),
                "nominal_post_base8_action_capacity_after_inertia_m": (
                    post_base8_action_capacity
                ),
                "downward_capacity_exhausted_by_inertial_tail": bool(
                    pre_action_buffer16_capacity <= 0.0
                ),
                "negative_z_capacity_exhausted_by_buffer16_or_inertia": bool(
                    pre_action_buffer16_capacity <= 0.0
                ),
                "strict_safe_translation_action_norm_capacity": float(
                    strict_post_base8_action_capacity
                    / position_action_scale
                ),
            }
        )
    limiting_pair = min(
        pair_envelopes,
        key=lambda record: record[
            "strict_safe_translation_action_norm_capacity"
        ],
    )
    pre_action_buffer16_limiting_pair = min(
        pair_envelopes,
        key=lambda record: record[
            "pre_action_buffer16_surplus_after_inertia_m"
        ],
    )
    recovery_required = any(
        record["negative_z_capacity_exhausted_by_buffer16_or_inertia"]
        for record in pair_envelopes
    )
    minimum_current_surplus = min(
        record["current_base8_surplus_m"] for record in pair_envelopes
    )
    minimum_pre_action_buffer16_surplus = min(
        record["pre_action_buffer16_surplus_after_inertia_m"]
        for record in pair_envelopes
    )
    if not np.isfinite(minimum_pre_action_buffer16_surplus):
        raise RuntimeError(
            "workspace-release live pre-action buffer16 headroom is "
            "non-finite"
        )
    live_buffer_headroom_cap_applied = bool(
        not recovery_required
        and downward_world_request_before_live_buffer_headroom_cap > 0.0
    )
    if live_buffer_headroom_cap_applied:
        downward_world_request = float(
            min(
                downward_world_request_before_live_buffer_headroom_cap,
                minimum_pre_action_buffer16_surplus,
            )
        )
        downward_z_error = float(-downward_world_request)
        requested = np.array(
            [
                xy_error[0] / position_action_scale,
                xy_error[1] / position_action_scale,
                downward_z_error / position_action_scale,
            ]
        )
        requested_norm = float(np.linalg.norm(requested))
        if not np.isfinite(requested_norm) or requested_norm <= 0.0:
            raise RuntimeError(
                "workspace release has no positive finite headroom-capped "
                "route error"
            )
        requested_direction = requested / requested_norm
    capacities = {
        "requested_outward_downward_action_norm": requested_norm,
        "compiled_pair_base8_post_tail_after_inertia": float(
            limiting_pair["strict_safe_translation_action_norm_capacity"]
        ),
        "native_strict_3d_translation_action_norm_bound": (
            strict_native_norm_bound
        ),
    }
    if configured_strict_norm_bound is not None:
        capacities["configured_translation_action_norm_bound"] = (
            configured_strict_norm_bound
        )
    selected_source = min(capacities, key=capacities.get)
    selected_norm = float(capacities[selected_source])
    desired_route_norm = float(
        min(recovery_route_requested_norm, route_strict_norm_bound)
    )
    if recovery_required:
        desired_route_tail = float(
            position_action_scale * desired_route_norm
            + inertial_tail_reserve
        )
        required_recovery = float(
            max(
                inertial_tail_reserve - minimum_current_surplus,
                worst_case_controller_world_step_m
                + inertial_tail_reserve
                - minimum_current_surplus,
                worst_case_controller_world_step_m
                + desired_route_tail
                - minimum_current_surplus,
                0.0,
            )
        )
        strict_positive_z_action_bound = float(
            np.nextafter(
                min(
                    native_high[2],
                    native_norm_bound,
                    route_strict_norm_bound,
                ),
                0.0,
            )
        )
        recovery_z_action = float(
            min(
                strict_positive_z_action_bound,
                np.nextafter(
                    required_recovery / position_action_scale,
                    np.inf,
                ),
            )
        )
        if recovery_z_action <= 0.0:
            raise RuntimeError(
                "workspace-release inertial recovery has no one-step strict "
                "+Z/base8 interior"
            )
        scalar_direction = np.array([0.0, 0.0, 1.0])
        candidate_scalar_norm = recovery_z_action
        selected_source = "event_driven_positive_z_inertial_recovery"
    else:
        scalar_direction = requested_direction.copy()
        candidate_scalar_norm = selected_norm

    required_clearance_key = "required_clearance_with_base_reserve_m"

    def literal_scalar_evidence(scalar_norm):
        translation = scalar_direction * float(scalar_norm)
        literal_norm = float(np.linalg.norm(translation))
        nominal_tail = (
            0.0
            if recovery_required
            else float(position_action_scale * literal_norm)
        )
        total_tail = float(nominal_tail + inertial_tail_reserve)
        literal_xy_delta = float(
            position_action_scale * np.linalg.norm(translation[:2])
        )
        literal_downward_delta = float(
            position_action_scale * max(0.0, -translation[2])
        )
        recovery_world_delta = (
            float(position_action_scale * translation[2])
            if recovery_required
            else 0.0
        )
        predicted = [
            float(
                record["current_vertical_clearance_m"]
                + (recovery_world_delta if recovery_required else 0.0)
                - total_tail
            )
            for record in pair_envelopes
        ]
        failed_conditions = []
        if not (np.isfinite(literal_norm) and literal_norm > 0.0):
            failed_conditions.append("positive_finite_literal_norm")
        if not literal_norm < native_norm_bound:
            failed_conditions.append("strict_native_translation_norm")
        if (
            configured_strict_norm_bound is not None
            and literal_norm > configured_strict_norm_bound
        ):
            failed_conditions.append("configured_translation_norm")
        if not (
            translation[2] > 0.0
            if recovery_required
            else translation[2] <= 0.0
        ):
            failed_conditions.append("workspace_release_z_direction")
        if not literal_xy_delta <= xy_remaining:
            failed_conditions.append("corridor_xy_no_overshoot")
        if (
            one_sided_outward_direction is not None
            and float(
                np.dot(translation[:2], one_sided_outward_direction)
            )
            < 0.0
        ):
            failed_conditions.append("one_sided_outward_axis_reversal")
        if not literal_downward_delta <= abs(downward_z_error):
            failed_conditions.append("release_z_no_overshoot")
        if not all(
            native_low[index] < translation[index] < native_high[index]
            for index in range(3)
        ):
            failed_conditions.append("strict_native_component_interior")
        if not all(
            clearance > record[required_clearance_key]
            for clearance, record in zip(predicted, pair_envelopes)
        ):
            failed_conditions.append(
                "all_55_pair_base8_strict_post_clearance"
            )
        return {
            "accepted": not failed_conditions,
            "failed_conditions": failed_conditions,
            "translation": translation,
            "literal_norm": literal_norm,
            "nominal_tail": nominal_tail,
            "total_tail": total_tail,
            "literal_xy_delta": literal_xy_delta,
            "literal_downward_delta": literal_downward_delta,
            "recovery_world_delta": recovery_world_delta,
            "predicted": predicted,
        }

    candidate_scalar_norm = float(candidate_scalar_norm)
    candidate_literal = literal_scalar_evidence(candidate_scalar_norm)
    solved_scalar_norm = candidate_scalar_norm
    solved_literal = candidate_literal
    scalar_solver_mode = "candidate_already_strict"
    scalar_nextafter_iterations = 0
    scalar_halving_iterations = 0
    scalar_bisection_iterations = 0
    if not candidate_literal["accepted"]:
        if recovery_required:
            solved_scalar_norm = strict_positive_z_action_bound
            solved_literal = literal_scalar_evidence(solved_scalar_norm)
            scalar_solver_mode = "recovery_strict_native_upper_probe"
            if not solved_literal["accepted"]:
                raise RuntimeError(
                    "workspace-release +Z recovery has no positive scalar "
                    "strict native/base8 interior: "
                    f"candidate={candidate_literal['failed_conditions']} "
                    f"upper={solved_literal['failed_conditions']}"
                )
        else:
            inward_scalar = float(
                np.nextafter(candidate_scalar_norm, 0.0)
            )
            scalar_nextafter_iterations = 1
            inward_literal = literal_scalar_evidence(inward_scalar)
            if inward_literal["accepted"]:
                solved_scalar_norm = inward_scalar
                solved_literal = inward_literal
                scalar_solver_mode = "single_scalar_nextafter"
            else:
                failing_upper = inward_scalar
                passing_lower = None
                probe = float(failing_upper * 0.5)
                for scalar_halving_iterations in range(1, 2049):
                    if not (0.0 < probe < failing_upper):
                        break
                    probe_literal = literal_scalar_evidence(probe)
                    if probe_literal["accepted"]:
                        passing_lower = probe
                        solved_literal = probe_literal
                        break
                    failing_upper = probe
                    probe = float(probe * 0.5)
                if passing_lower is None:
                    raise RuntimeError(
                        "workspace-release has no representable positive "
                        "scalar strict native/base8 interior: "
                        f"{inward_literal['failed_conditions']}"
                    )
                for scalar_bisection_iterations in range(1, 257):
                    midpoint = float(
                        passing_lower
                        + (failing_upper - passing_lower) * 0.5
                    )
                    if midpoint in (passing_lower, failing_upper):
                        break
                    midpoint_literal = literal_scalar_evidence(midpoint)
                    if midpoint_literal["accepted"]:
                        passing_lower = midpoint
                        solved_literal = midpoint_literal
                    else:
                        failing_upper = midpoint
                solved_scalar_norm = passing_lower
                scalar_solver_mode = "halving_then_scalar_bisection"

    if not solved_literal["accepted"]:
        raise RuntimeError(
            "workspace-release scalar solver returned a non-interior action"
        )
    translation = solved_literal["translation"]
    literal_norm = solved_literal["literal_norm"]
    nominal_tail = solved_literal["nominal_tail"]
    total_tail = solved_literal["total_tail"]
    literal_xy_delta = solved_literal["literal_xy_delta"]
    literal_downward_delta = solved_literal["literal_downward_delta"]
    recovery_world_delta = solved_literal["recovery_world_delta"]
    predicted = solved_literal["predicted"]

    def positive_float_ulp_distance(first, second):
        if not (
            np.isfinite(first)
            and np.isfinite(second)
            and first >= 0.0
            and second >= 0.0
        ):
            raise RuntimeError(
                "workspace-release scalar ULP distance is undefined"
            )
        first_bits = int(np.float64(first).view(np.uint64))
        second_bits = int(np.float64(second).view(np.uint64))
        return abs(first_bits - second_bits)

    scalar_solver_evidence = {
        "accepted": True,
        "solver_mode": scalar_solver_mode,
        "limiting_condition": selected_source,
        "candidate_scalar_action_norm": candidate_scalar_norm,
        "candidate_literal_action_norm": candidate_literal["literal_norm"],
        "candidate_accepted": bool(candidate_literal["accepted"]),
        "candidate_failed_conditions": list(
            candidate_literal["failed_conditions"]
        ),
        "solved_scalar_action_norm": solved_scalar_norm,
        "solved_literal_action_norm": literal_norm,
        "candidate_to_solved_scalar_ulp_distance": (
            positive_float_ulp_distance(
                candidate_scalar_norm, solved_scalar_norm
            )
        ),
        "scalar_nextafter_iterations": scalar_nextafter_iterations,
        "scalar_halving_iterations": scalar_halving_iterations,
        "scalar_bisection_iterations": scalar_bisection_iterations,
        "direction_preserved_exactly_by_scalar_construction": True,
        "final_failed_conditions": list(solved_literal["failed_conditions"]),
        "strict_pair_clearance_condition": required_clearance_key,
    }
    minimum_surplus = float("inf")
    minimum_buffer16_surplus = float("inf")
    for clearance, record in zip(predicted, pair_envelopes):
        record["predicted_post_worst_case_vertical_clearance_m"] = clearance
        record["predicted_post_worst_case_base_reserve_surplus_m"] = float(
            clearance - record["required_clearance_with_base_reserve_m"]
        )
        record["predicted_post_worst_case_buffer16_surplus_m"] = float(
            clearance - record["required_clearance_with_fixed_buffer16_m"]
        )
        minimum_surplus = min(
            minimum_surplus,
            record[
                "predicted_post_worst_case_base_reserve_surplus_m"
            ],
        )
        minimum_buffer16_surplus = min(
            minimum_buffer16_surplus,
            record["predicted_post_worst_case_buffer16_surplus_m"],
        )
    action = np.zeros(7, dtype=float)
    action[:3] = translation
    action[-1] = float(gripper)
    return action, {
        "accepted": True,
        "motion_kind": (
            "positive_z_inertial_recovery"
            if recovery_required
            else (
                (
                    "one_sided_corridor_holding_downward_descent"
                    if one_sided_outward_direction is not None
                    else "corridor_holding_downward_descent"
                )
                if not couple_downward_to_lateral_remaining
                else (
                    "ulp_bounded_inward_downward_workspace_release"
                    if requested_inward_xy_correction > 0.0
                    else "outward_downward_workspace_release"
                )
            )
        ),
        "formula": (
            "request corridor XY plus negative Z capped in world magnitude "
            "by the existing one-step world reserve and, for workspace "
            "release only, by remaining corridor XY; for corridor-holding "
            "descent the negative-Z request remains independent of a zero "
            "lateral error and any inward safety-axis error is clamped to "
            "zero while tangential correction remains active; then cap only "
            "its negative-Z direction "
            "component by the live minimum all-55-pair pre-action buffer16 "
            "surplus after "
            "the latest measured negative-dz inertial reserve; authorize "
            "negative Z only when every current pair remains strictly above "
            "that fixed buffer16; size and revalidate the literal scalar "
            "action against every unchanged post-action base8 clearance; if "
            "pre-action buffer16 is exhausted, prohibit negative Z and issue "
            "event-driven pure +Z with the unchanged recovery route norm; "
            "when the registered high target was nudged outward only to make "
            "the strict 8 mm reserve representable, permit an inward return "
            "only within its recorded one-ULP-expanded nudge distance"
        ),
        "current_eef": current_eef.tolist(),
        "corridor_target_xy": corridor_target_xy.tolist(),
        "raw_corridor_target_xy_error": raw_xy_error.tolist(),
        "one_sided_outward_direction_xy": (
            None
            if one_sided_outward_direction is None
            else one_sided_outward_direction.tolist()
        ),
        "raw_one_sided_outward_axis_error_m": (
            raw_one_sided_outward_error
        ),
        "suppressed_inward_outward_axis_error_m": (
            suppressed_inward_outward_axis_error
        ),
        "native_tangent_return_gate_enabled": bool(
            native_tangent_return_gate_enabled
        ),
        "normalized_outward_direction_xy": (
            None
            if normalized_outward_direction is None
            else normalized_outward_direction.tolist()
        ),
        "requested_corridor_error_outward_projection_m": (
            requested_outward_projection
        ),
        "requested_inward_xy_correction_m": (
            requested_inward_xy_correction
        ),
        "maximum_inward_xy_correction_one_ulp_bound_m": (
            maximum_inward_xy_correction
        ),
        "inward_xy_correction_within_one_ulp_bound": bool(
            not native_tangent_return_gate_enabled
            or requested_inward_xy_correction
            <= maximum_inward_xy_correction
        ),
        "release_target_z_m": float(release_target_z),
        "xy_remaining_m": xy_remaining,
        "full_release_downward_z_error_m": full_downward_z_error,
        "downward_coupled_to_lateral_remaining": bool(
            couple_downward_to_lateral_remaining
        ),
        "xy_coupled_downward_world_request_before_one_step_cap_m": (
            xy_coupled_downward_world_request
        ),
        "route_downward_world_request_before_one_step_cap_m": (
            route_downward_world_request
        ),
        "downward_world_request_before_live_buffer16_headroom_cap_m": (
            downward_world_request_before_live_buffer_headroom_cap
        ),
        "capped_downward_world_request_m": downward_world_request,
        "downward_request_capped_by_xy_remaining": bool(
            couple_downward_to_lateral_remaining
            and downward_world_request <= xy_remaining
        ),
        "independent_downward_progress_authorized": bool(
            not couple_downward_to_lateral_remaining
            and full_downward_z_error > 0.0
        ),
        "downward_request_capped_by_existing_one_step_world_reserve": bool(
            downward_world_request_before_live_buffer_headroom_cap
            <= worst_case_controller_world_step_m
        ),
        "live_pre_action_buffer16_headroom_cap_applied_to_negative_z": (
            live_buffer_headroom_cap_applied
        ),
        "downward_request_within_live_pre_action_buffer16_headroom": bool(
            recovery_required
            or downward_world_request
            <= minimum_pre_action_buffer16_surplus
        ),
        "requested_translation_action": requested.tolist(),
        "requested_translation_action_norm": requested_norm,
        "recovery_route_requested_translation_action_norm_before_one_step_cap": (
            recovery_route_requested_norm
        ),
        "literal_scalar_strict_interior_solver": scalar_solver_evidence,
        "measured_vertical_step_progress_m": float(
            measured_vertical_step_progress_m
        ),
        "measured_negative_inertial_tail_reserve_m": inertial_tail_reserve,
        "compiled_pair_count": len(pair_envelopes),
        "pair_identity_keys": [list(identity) for identity in identities],
        "pair_envelopes": pair_envelopes,
        "selected_limiting_pair": dict(limiting_pair),
        "selected_pre_action_buffer16_limiting_pair": dict(
            pre_action_buffer16_limiting_pair
        ),
        "candidate_action_norm_capacities": capacities,
        "configured_strict_translation_action_norm_bound": (
            configured_strict_norm_bound
        ),
        "selected_envelope_source": selected_source,
        "event_driven_positive_z_inertial_recovery": recovery_required,
        "negative_z_action_requires_fixed_buffer16": bool(
            not recovery_required
        ),
        "minimum_current_base8_surplus_m": minimum_current_surplus,
        "minimum_pre_action_buffer16_surplus_after_inertia_m": (
            minimum_pre_action_buffer16_surplus
        ),
        "commanded_positive_z_recovery_world_delta_m": (
            recovery_world_delta
        ),
        "commanded_translation_action_norm": literal_norm,
        "commanded_nominal_norm_downward_tail_m": nominal_tail,
        "commanded_worst_case_downward_world_tail_m": total_tail,
        "commanded_xy_action": action[:2].tolist(),
        "commanded_z_action": float(action[2]),
        "minimum_predicted_post_worst_case_base_surplus_m": minimum_surplus,
        "minimum_predicted_post_worst_case_buffer16_surplus_m": (
            minimum_buffer16_surplus
        ),
        "proof": {
            "outward_xy_plus_nonpositive_z_zero_rotation": bool(
                not recovery_required
                and couple_downward_to_lateral_remaining
                and requested_inward_xy_correction == 0.0
            ),
            "corridor_xy_hold_plus_nonpositive_z_zero_rotation": bool(
                not recovery_required
                and not couple_downward_to_lateral_remaining
            ),
            "inward_outward_axis_command_prohibited": bool(
                one_sided_outward_direction is not None
                and float(
                    np.dot(
                        action[:2],
                        one_sided_outward_direction,
                    )
                )
                >= 0.0
            ),
            "inward_xy_limited_to_prebuffer_one_ulp_bound": bool(
                native_tangent_return_gate_enabled
                and requested_inward_xy_correction > 0.0
                and requested_inward_xy_correction
                <= maximum_inward_xy_correction
            ),
            "pure_positive_z_zero_xy_rotation_recovery": bool(
                recovery_required
            ),
            "strictly_inside_native_3d_action_norm_bound": True,
            "does_not_cross_corridor_target_xy": bool(
                literal_xy_delta <= xy_remaining
            ),
            "does_not_cross_release_target_z": bool(
                literal_downward_delta <= abs(downward_z_error)
            ),
            "latest_measured_negative_dz_reserved_as_inertial_tail": True,
            "negative_z_pre_action_uses_strict_buffer16_plus_inertia": bool(
                not recovery_required
            ),
            "live_buffer16_headroom_only_refines_negative_z_direction": (
                True
            ),
            "recovery_route_norm_unchanged_by_live_headroom_cap": True,
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
    lateral_target_xy=None,
    one_sided_outward_direction_xy=None,
    maximum_lateral_translation_action=None,
):
    """Refill every pair's exact buffer deficit, optionally retaining XY."""
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
    optional_lateral_values = (
        lateral_target_xy,
        one_sided_outward_direction_xy,
        maximum_lateral_translation_action,
    )
    retain_outward_lateral_drive = bool(
        all(value is not None for value in optional_lateral_values)
    )
    if any(value is not None for value in optional_lateral_values) and not (
        retain_outward_lateral_drive
    ):
        raise ValueError(
            "adaptive lateral-rebuffer outward-drive inputs must be supplied "
            "together"
        )
    if retain_outward_lateral_drive:
        lateral_target_xy = np.asarray(lateral_target_xy, dtype=float)
        one_sided_outward_direction_xy = np.asarray(
            one_sided_outward_direction_xy, dtype=float
        )
        outward_direction_norm = float(
            np.linalg.norm(one_sided_outward_direction_xy)
        )
        maximum_lateral_translation_action = float(
            maximum_lateral_translation_action
        )
        if (
            lateral_target_xy.shape != (2,)
            or one_sided_outward_direction_xy.shape != (2,)
            or not np.all(np.isfinite(lateral_target_xy))
            or not np.all(np.isfinite(one_sided_outward_direction_xy))
            or not np.isfinite(outward_direction_norm)
            or not np.isclose(
                outward_direction_norm, 1.0, rtol=0.0, atol=1e-12
            )
            or not np.isfinite(maximum_lateral_translation_action)
            or maximum_lateral_translation_action <= 0.0
        ):
            raise ValueError(
                "adaptive lateral-rebuffer outward-drive geometry is invalid"
            )
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
        or not np.all(native_low[:6] < 0.0)
        or not np.all(native_high[:6] > 0.0)
        or not (native_low[6] <= gripper <= native_high[6])
    ):
        raise RuntimeError(
            "native OSC action bounds do not prove the requested lateral "
            "rebuffer"
        )

    native_translation_norm_bound = float(
        min(
            -native_low[0],
            native_high[0],
            -native_low[1],
            native_high[1],
            -native_low[2],
            native_high[2],
        )
    )
    strict_native_translation_norm_bound = float(
        np.nextafter(native_translation_norm_bound, 0.0)
    )
    if strict_native_translation_norm_bound <= 0.0:
        raise RuntimeError(
            "native OSC translation action norm has no strict interior"
        )
    strict_configured_lateral_action_bound = None
    if retain_outward_lateral_drive:
        if (
            maximum_lateral_translation_action
            > strict_native_translation_norm_bound
        ):
            raise ValueError(
                "adaptive lateral-rebuffer lateral action bound exceeds the "
                "strict runtime-native translation capacity"
            )
        strict_configured_lateral_action_bound = float(
            np.nextafter(maximum_lateral_translation_action, 0.0)
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
    lateral_remaining = None
    lateral_target_outward_error = None
    lateral_action_norm_capacities = None
    selected_lateral_envelope_source = None
    commanded_lateral_action_norm = 0.0
    commanded_lateral_world_delta = 0.0
    full_translation_action_norm = float(abs(commanded_z_action))
    if retain_outward_lateral_drive:
        lateral_delta = lateral_target_xy - current_eef[:2]
        lateral_remaining = float(np.linalg.norm(lateral_delta))
        lateral_target_outward_error = float(
            np.dot(lateral_delta, one_sided_outward_direction_xy)
        )
        native_lateral_capacity_after_positive_z = float(
            np.sqrt(
                max(
                    0.0,
                    strict_native_translation_norm_bound**2
                    - commanded_z_action**2,
                )
            )
        )
        lateral_action_norm_capacities = {
            "lateral_target_remaining_action_norm": float(
                lateral_remaining / position_action_scale
            ),
            "configured_strict_lateral_action_norm_bound": (
                strict_configured_lateral_action_bound
            ),
            "native_full_norm_lateral_capacity_after_positive_z": (
                native_lateral_capacity_after_positive_z
            ),
        }
        if lateral_target_outward_error > 0.0 and lateral_remaining > 0.0:
            selected_lateral_envelope_source = min(
                lateral_action_norm_capacities,
                key=lateral_action_norm_capacities.get,
            )
            commanded_lateral_action_norm = float(
                lateral_action_norm_capacities[
                    selected_lateral_envelope_source
                ]
            )
            if commanded_lateral_action_norm <= 0.0:
                raise RuntimeError(
                    "adaptive lateral-rebuffer has no positive lateral "
                    "capacity beside the required +Z action"
                )
            lateral_direction = lateral_delta / lateral_remaining
            for _ in range(128):
                commanded_xy = (
                    lateral_direction * commanded_lateral_action_norm
                )
                commanded_lateral_world_delta = float(
                    position_action_scale * np.linalg.norm(commanded_xy)
                )
                full_translation_action_norm = float(
                    np.linalg.norm(
                        np.array(
                            [
                                commanded_xy[0],
                                commanded_xy[1],
                                commanded_z_action,
                            ],
                            dtype=float,
                        )
                    )
                )
                if (
                    0.0 < commanded_lateral_action_norm
                    <= strict_configured_lateral_action_bound
                    and commanded_lateral_world_delta <= lateral_remaining
                    and float(
                        np.dot(
                            commanded_xy,
                            one_sided_outward_direction_xy,
                        )
                    )
                    > 0.0
                    and full_translation_action_norm
                    < native_translation_norm_bound
                    and native_low[0] < commanded_xy[0] < native_high[0]
                    and native_low[1] < commanded_xy[1] < native_high[1]
                ):
                    break
                commanded_lateral_action_norm = float(
                    np.nextafter(commanded_lateral_action_norm, 0.0)
                )
            else:
                raise RuntimeError(
                    "adaptive lateral-rebuffer outward XY/+Z action has no "
                    "strict native-action interior"
                )
            if commanded_lateral_action_norm <= 0.0:
                raise RuntimeError(
                    "adaptive lateral-rebuffer outward XY action collapsed "
                    "to zero while proving safety"
                )
            action[:2] = commanded_xy
        else:
            selected_lateral_envelope_source = (
                "already_at_or_beyond_outward_lateral_target"
            )
        if (
            float(
                np.dot(action[:2], one_sided_outward_direction_xy)
            )
            < 0.0
            or full_translation_action_norm
            >= native_translation_norm_bound
            or commanded_lateral_world_delta > lateral_remaining
        ):
            raise RuntimeError(
                "adaptive lateral-rebuffer action violated its one-sided "
                "outward/full-norm/target hard proof"
            )
    if retain_outward_lateral_drive:
        formula = (
            "take the maximum live deficit to the unchanged strict+base8+"
            "one-controller-step buffer16 envelope over every compiled pair, "
            "add the same unchanged +Z controller-step tail, and retain a "
            "one-sided outward XY correction toward the existing shifted "
            "target capped by its remaining error, the configured lateral "
            "bound, and the strict runtime-native 3-D norm remainder"
        )
        proof = {
            "outward_xy_plus_positive_z_zero_rotation": True,
            "strictly_inside_native_3d_action_norm_bound": True,
            "inside_configured_lateral_action_norm_bound": True,
            "does_not_cross_lateral_target": bool(
                commanded_lateral_world_delta <= lateral_remaining
            ),
            "inward_outward_axis_command_prohibited": bool(
                float(
                    np.dot(
                        action[:2], one_sided_outward_direction_xy
                    )
                )
                >= 0.0
            ),
            "positive_z_static_geometry_does_not_reduce_clearance": True,
            "all_compiled_pairs_retain_strict_no_contact": True,
            "all_compiled_pairs_retain_strict_base8": True,
            "all_compiled_pairs_reach_strict_buffer16": True,
        }
    else:
        formula = (
            "take the maximum live deficit to the unchanged strict+base8+"
            "one-controller-step buffer16 envelope over every compiled pair, "
            "add that same unchanged controller-step tail for the immediately "
            "resumed XY action, and require the resulting pure +Z command to "
            "remain strictly inside the runtime native action bound"
        )
        proof = {
            "pure_positive_z": True,
            "strictly_inside_native_z_action_bound": True,
            "outside_xy_clearance_not_worsened_by_pure_z": True,
            "all_compiled_pairs_retain_strict_no_contact": True,
            "all_compiled_pairs_retain_strict_base8": True,
            "all_compiled_pairs_reach_strict_buffer16": True,
        }
    return action, {
        "formula": formula,
        "current_eef": current_eef.tolist(),
        "position_action_scale_m_per_normalized_action": float(
            position_action_scale
        ),
        "native_action_spec_source": native_source,
        "native_z_action_bounds": [
            float(native_low[2]),
            float(native_high[2]),
        ],
        "native_3d_translation_action_norm_bound": (
            native_translation_norm_bound
        ),
        "strict_native_3d_translation_action_norm_bound": (
            strict_native_translation_norm_bound
        ),
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
        "retain_outward_lateral_drive": retain_outward_lateral_drive,
        "lateral_target_xy": (
            None
            if lateral_target_xy is None
            else lateral_target_xy.tolist()
        ),
        "one_sided_outward_direction_xy": (
            None
            if one_sided_outward_direction_xy is None
            else one_sided_outward_direction_xy.tolist()
        ),
        "lateral_remaining_m": lateral_remaining,
        "lateral_target_outward_error_m": lateral_target_outward_error,
        "configured_maximum_lateral_translation_action": (
            maximum_lateral_translation_action
        ),
        "strict_configured_lateral_translation_action_bound": (
            strict_configured_lateral_action_bound
        ),
        "lateral_action_norm_capacities": (
            lateral_action_norm_capacities
        ),
        "selected_lateral_envelope_source": (
            selected_lateral_envelope_source
        ),
        "commanded_lateral_action_norm": (
            commanded_lateral_action_norm
        ),
        "commanded_lateral_world_delta_m": (
            commanded_lateral_world_delta
        ),
        "commanded_translation_action_norm": (
            full_translation_action_norm
        ),
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
        "proof": proof,
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


def _validated_rigid_rotation_matrix(rotation, *, label):
    """Fail closed unless ``rotation`` is a finite proper 3-D rotation."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} rotation matrix must be finite and 3-D")
    orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    )
    determinant = float(np.linalg.det(rotation))
    if (
        orthogonality_error > 1e-7
        or not np.isfinite(determinant)
        or abs(determinant - 1.0) > 1e-7
    ):
        raise RuntimeError(
            f"{label} frame is not a proper rigid rotation: "
            f"orthogonality_error={orthogonality_error} "
            f"determinant={determinant}"
        )
    return rotation


def _hypothetical_wrist_yaw_specs(
    *,
    reference_outward_direction_xy,
    target_outward_directions_xy,
    table_normal_world,
):
    """Derive unique signed table-normal yaws without issuing an action."""
    reference_xy = np.asarray(
        reference_outward_direction_xy, dtype=float
    )
    normal = np.asarray(table_normal_world, dtype=float)
    targets = [
        np.asarray(target, dtype=float)
        for target in target_outward_directions_xy
    ]
    if reference_xy.shape != (2,) or not np.all(np.isfinite(reference_xy)):
        raise ValueError("reference outward direction must be finite and 2-D")
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("table normal must be finite and 3-D")
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or abs(normal_norm - 1.0) > 1e-7:
        raise RuntimeError("table normal must be a unit vector")
    if normal[2] <= 0.0:
        raise RuntimeError("table normal must use the upward orientation")
    reference_norm = float(np.linalg.norm(reference_xy))
    if not np.isfinite(reference_norm) or reference_norm <= 1e-9:
        raise ValueError("reference outward direction must be nonzero")
    reference = np.array(
        [reference_xy[0], reference_xy[1], 0.0], dtype=float
    ) / reference_norm
    if abs(float(np.dot(reference, normal))) > 1e-7:
        raise RuntimeError(
            "reference outward direction is not tangent to the native table"
        )

    specs = []
    for index, target_xy in enumerate(targets):
        if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
            raise ValueError("target outward directions must be finite and 2-D")
        target_norm = float(np.linalg.norm(target_xy))
        if not np.isfinite(target_norm) or target_norm <= 1e-9:
            raise ValueError("target outward direction must be nonzero")
        target = np.array(
            [target_xy[0], target_xy[1], 0.0], dtype=float
        ) / target_norm
        if abs(float(np.dot(target, normal))) > 1e-7:
            raise RuntimeError(
                "target outward direction is not tangent to the native table"
            )
        sine = float(np.dot(normal, np.cross(reference, target)))
        cosine = float(np.dot(reference, target))
        yaw = float(np.arctan2(sine, cosine))
        if abs(yaw) <= 1e-15:
            yaw = 0.0
            rotation = np.eye(3)
        else:
            cross_matrix = np.array(
                [
                    [0.0, -normal[2], normal[1]],
                    [normal[2], 0.0, -normal[0]],
                    [-normal[1], normal[0], 0.0],
                ],
                dtype=float,
            )
            rotation = (
                np.eye(3)
                + np.sin(yaw) * cross_matrix
                + (1.0 - np.cos(yaw)) * (cross_matrix @ cross_matrix)
            )
        rotation = _validated_rigid_rotation_matrix(
            rotation,
            label=f"hypothetical wrist yaw {index}",
        )
        if not np.allclose(
            rotation @ reference,
            target,
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                "hypothetical wrist yaw does not map the reference approach "
                "to its target outward direction"
            )
        if any(
            abs(
                float(
                    np.arctan2(
                        np.sin(yaw - previous["yaw_angle_rad"]),
                        np.cos(yaw - previous["yaw_angle_rad"]),
                    )
                )
            )
            <= 1e-9
            for previous in specs
        ):
            raise RuntimeError(
                "duplicate hypothetical wrist yaw derived from native directions"
            )
        specs.append(
            {
                "reference_outward_direction_xy": reference[:2].tolist(),
                "target_outward_direction_xy": target[:2].tolist(),
                "table_normal_world": normal.tolist(),
                "yaw_angle_rad": yaw,
                "yaw_angle_deg": float(np.degrees(yaw)),
                "axis_angle_world_rad": (normal * yaw).tolist(),
                "rotation_matrix_world": rotation.tolist(),
                "formula": (
                    "signed yaw = atan2(table_normal dot (reference cross "
                    "target), reference dot target); rotate about the exact "
                    "current EEF pivot"
                ),
                "provenance": {
                    "reference": (
                        "current selected low-skew legacy +X approach"
                    ),
                    "target": (
                        "normalized outward direction derived from the native "
                        "push frame"
                    ),
                    "axis": (
                        "upward table normal derived from compiled native "
                        "table geom frames"
                    ),
                },
            }
        )
    return specs


def _compiled_table_normal_evidence(env):
    """Derive one upward normal from the native table collision frames."""
    model, data = env.sim.model, env.sim.data
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
    frames = []
    for geom_id in table_geom_ids:
        rotation = _validated_rigid_rotation_matrix(
            np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3),
            label=f"table geom {model.geom_id2name(geom_id) or geom_id}",
        )
        raw_normal = rotation[:, 2].copy()
        upward_normal = raw_normal.copy()
        orientation_flipped = False
        if upward_normal[2] < 0.0:
            upward_normal *= -1.0
            orientation_flipped = True
        if upward_normal[2] <= 0.0:
            raise RuntimeError("compiled table geom has no upward-facing normal")
        frames.append(
            {
                "geom": model.geom_id2name(geom_id) or f"geom_{geom_id}",
                "geom_id": int(geom_id),
                "rotation_matrix_world": rotation.tolist(),
                "raw_local_z_world": raw_normal.tolist(),
                "upward_normal_world": upward_normal.tolist(),
                "orientation_flipped": orientation_flipped,
            }
        )
    normal = np.asarray(frames[0]["upward_normal_world"], dtype=float)
    for frame in frames[1:]:
        if not np.allclose(
            frame["upward_normal_world"], normal, rtol=0.0, atol=1e-7
        ):
            raise RuntimeError(
                "compiled native table collision geoms disagree on the table normal"
            )
    if not np.allclose(normal, [0.0, 0.0, 1.0], rtol=0.0, atol=1e-7):
        raise RuntimeError(
            "compiled table normal is incompatible with the controller XY/Z frame"
        )
    return normal, {
        "table_normal_world": normal.tolist(),
        "table_collision_geom_frames": frames,
        "formula": (
            "use each collidable native table geom's compiled local +Z axis, "
            "orient it upward, require all normals to agree, and require the "
            "result to match the controller's world +Z table-normal frame"
        ),
    }


def _hypothetical_finger_yaw_env(env, *, eef_position, rotation):
    """Return a read-only proxy with every finger geom rigidly yawed."""
    model, data = env.sim.model, env.sim.data
    eef_position = np.asarray(eef_position, dtype=float)
    rotation = _validated_rigid_rotation_matrix(
        rotation, label="hypothetical wrist yaw"
    )
    if eef_position.shape != (3,) or not np.all(np.isfinite(eef_position)):
        raise ValueError("hypothetical yaw EEF pivot must be finite and 3-D")
    geom_xpos = np.asarray(data.geom_xpos, dtype=float)
    geom_xmat = np.asarray(data.geom_xmat, dtype=float)
    if (
        geom_xpos.shape != (int(model.ngeom), 3)
        or geom_xmat.shape != (int(model.ngeom), 9)
        or not np.all(np.isfinite(geom_xpos))
        or not np.all(np.isfinite(geom_xmat))
    ):
        raise RuntimeError("compiled geom poses are unavailable or invalid")
    hypothetical_xpos = geom_xpos.copy()
    hypothetical_xmat = geom_xmat.copy()
    finger_geom_ids = []
    for geom_id in range(int(model.ngeom)):
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        if "finger" not in body_name.lower():
            continue
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        if _semantic_finger_side(body_name) is None:
            raise RuntimeError(
                "compiled finger collision geom lacks left/right provenance"
            )
        finger_geom_ids.append(geom_id)
    if not finger_geom_ids:
        raise RuntimeError("compiled finger collision geoms unavailable")

    transforms = []
    before_origins = []
    after_origins = []
    for geom_id in finger_geom_ids:
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        current_rotation = _validated_rigid_rotation_matrix(
            geom_xmat[geom_id].reshape(3, 3),
            label=f"finger geom {model.geom_id2name(geom_id) or geom_id}",
        )
        current_origin = geom_xpos[geom_id].copy()
        hypothetical_origin = (
            eef_position + rotation @ (current_origin - eef_position)
        )
        hypothetical_rotation = _validated_rigid_rotation_matrix(
            rotation @ current_rotation,
            label=(
                "hypothetical finger geom "
                f"{model.geom_id2name(geom_id) or geom_id}"
            ),
        )
        hypothetical_xpos[geom_id] = hypothetical_origin
        hypothetical_xmat[geom_id] = hypothetical_rotation.reshape(9)
        before_origins.append(current_origin)
        after_origins.append(hypothetical_origin)
        transforms.append(
            {
                "geom": model.geom_id2name(geom_id) or f"geom_{geom_id}",
                "geom_id": int(geom_id),
                "body": body_name,
                "semantic_side": _semantic_finger_side(body_name),
                "current_origin_world": current_origin.tolist(),
                "hypothetical_origin_world": hypothetical_origin.tolist(),
                "current_rotation_matrix_world": current_rotation.tolist(),
                "hypothetical_rotation_matrix_world": (
                    hypothetical_rotation.tolist()
                ),
            }
        )
    before_origins = np.asarray(before_origins, dtype=float)
    after_origins = np.asarray(after_origins, dtype=float)
    radial_error = float(
        np.max(
            np.abs(
                np.linalg.norm(after_origins - eef_position, axis=1)
                - np.linalg.norm(before_origins - eef_position, axis=1)
            )
        )
    )
    pairwise_error = 0.0
    for left in range(len(finger_geom_ids)):
        for right in range(left + 1, len(finger_geom_ids)):
            before_distance = float(
                np.linalg.norm(before_origins[left] - before_origins[right])
            )
            after_distance = float(
                np.linalg.norm(after_origins[left] - after_origins[right])
            )
            pairwise_error = max(
                pairwise_error, abs(after_distance - before_distance)
            )
    if radial_error > 1e-9 or pairwise_error > 1e-9:
        raise RuntimeError(
            "hypothetical wrist transform is not rigid: "
            f"radial_error={radial_error} pairwise_error={pairwise_error}"
        )
    hypothetical_data = SimpleNamespace(
        geom_xpos=hypothetical_xpos,
        geom_xmat=hypothetical_xmat,
    )
    hypothetical_env = SimpleNamespace(
        sim=SimpleNamespace(model=model, data=hypothetical_data)
    )
    return hypothetical_env, {
        "pivot_eef_position_world": eef_position.tolist(),
        "rotation_matrix_world": rotation.tolist(),
        "finger_geom_transforms": transforms,
        "maximum_eef_radial_distance_error_m": radial_error,
        "maximum_pairwise_distance_error_m": pairwise_error,
        "rigid_transform_verified": True,
        "executed": False,
    }


def _compiled_finger_yaw_frame(env, *, eef_position):
    """Capture the live compiled finger rigid frame for yaw feedback."""
    model, data = env.sim.model, env.sim.data
    eef_position = np.asarray(eef_position, dtype=float)
    if eef_position.shape != (3,) or not np.all(np.isfinite(eef_position)):
        raise ValueError("live yaw-frame EEF position must be finite and 3-D")
    records = []
    for geom_id in range(int(model.ngeom)):
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        semantic_side = _semantic_finger_side(body_name)
        if semantic_side is None or (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue
        name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
        rotation = _validated_rigid_rotation_matrix(
            np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3),
            label=f"live yaw-frame finger geom {name}",
        )
        center, half_size = _compiled_geom_world_aabb(
            model, data, geom_id
        )
        records.append(
            {
                "geom": name,
                "geom_id": int(geom_id),
                "body": body_name,
                "semantic_side": semantic_side,
                "origin_world": np.asarray(
                    data.geom_xpos[geom_id], dtype=float
                ).tolist(),
                "center_world": center.tolist(),
                "world_aabb_half_size": half_size.tolist(),
                "rotation_matrix_world": rotation.tolist(),
            }
        )
    if {
        record["semantic_side"] for record in records
    } != {"left", "right"}:
        raise RuntimeError(
            "live yaw frame lacks compiled left/right finger collision geoms"
        )
    maximum_radius = max(
        float(
            np.linalg.norm(
                np.asarray(record["center_world"], dtype=float)
                - eef_position
            )
            + np.linalg.norm(
                np.asarray(record["world_aabb_half_size"], dtype=float)
            )
        )
        for record in records
    )
    if not np.isfinite(maximum_radius) or maximum_radius <= 0.0:
        raise RuntimeError("live finger yaw-frame radius is invalid")
    return {
        "eef_position_world": eef_position.tolist(),
        "finger_geoms": records,
        "maximum_finger_radius_from_eef_m": maximum_radius,
    }


def _rotation_matrix_error_angle(rotation):
    rotation = _validated_rigid_rotation_matrix(
        rotation, label="relative yaw error"
    )
    cosine = float(np.clip(0.5 * (np.trace(rotation) - 1.0), -1.0, 1.0))
    return float(np.arccos(cosine))


def _rotation_matrix_axis_angle(rotation):
    """Convert one strict proper rotation to its shortest world axis-angle."""
    rotation = _validated_rigid_rotation_matrix(
        rotation, label="remaining rotation axis-angle"
    )
    angle = _rotation_matrix_error_angle(rotation)
    if angle <= 1e-15:
        return np.zeros(3, dtype=float)
    sine = float(np.sin(angle))
    if abs(sine) <= 1e-9:
        raise RuntimeError(
            "remaining rotation is too close to pi for a unique strict axis-angle"
        )
    skew_vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    axis_angle = angle * skew_vector / (2.0 * sine)
    if not np.all(np.isfinite(axis_angle)) or not np.isclose(
        np.linalg.norm(axis_angle), angle, rtol=0.0, atol=1e-9
    ):
        raise RuntimeError("remaining rotation axis-angle conversion failed")
    return axis_angle


def _wrist_yaw_attainment_evidence(
    *,
    reference_frame,
    current_frame,
    yaw_spec,
    maximum_angle_error_rad,
    maximum_position_drift_m,
    angular_progress_epsilon_rad,
    position_progress_epsilon_m,
    anchor_eef_position=None,
    previous_absolute_error_rad=None,
    previous_rotation_error_rad=None,
    previous_position_drift_m=None,
):
    """Measure actual rigid finger rotation against the selected yaw target."""
    if (
        not np.isfinite(maximum_angle_error_rad)
        or maximum_angle_error_rad <= 0.0
        or not np.isfinite(maximum_position_drift_m)
        or maximum_position_drift_m <= 0.0
        or not np.isfinite(angular_progress_epsilon_rad)
        or angular_progress_epsilon_rad <= 0.0
        or not np.isfinite(position_progress_epsilon_m)
        or position_progress_epsilon_m <= 0.0
    ):
        raise ValueError("wrist-yaw attainment thresholds must be positive")
    reference_eef = np.asarray(
        reference_frame["eef_position_world"], dtype=float
    )
    current_eef = np.asarray(
        current_frame["eef_position_world"], dtype=float
    )
    anchor_eef = (
        reference_eef
        if anchor_eef_position is None
        else np.asarray(anchor_eef_position, dtype=float)
    )
    if anchor_eef.shape != (3,) or not np.all(np.isfinite(anchor_eef)):
        raise ValueError("wrist-yaw anchor EEF position must be finite and 3-D")
    target_rotation = _validated_rigid_rotation_matrix(
        yaw_spec["rotation_matrix_world"], label="selected wrist yaw"
    )
    normal = np.asarray(yaw_spec["table_normal_world"], dtype=float)
    reference_outward = np.r_[
        np.asarray(yaw_spec["reference_outward_direction_xy"], dtype=float),
        0.0,
    ]
    reference_records = {
        record["geom"]: record for record in reference_frame["finger_geoms"]
    }
    current_records = {
        record["geom"]: record for record in current_frame["finger_geoms"]
    }
    if (
        set(reference_records) != set(current_records)
        or not reference_records
    ):
        raise RuntimeError(
            "live finger geom inventory changed during wrist yaw"
        )
    relative_rotations = []
    maximum_rigid_origin_error = 0.0
    for name in sorted(reference_records):
        reference = reference_records[name]
        current = current_records[name]
        if (
            reference["body"] != current["body"]
            or reference["semantic_side"] != current["semantic_side"]
        ):
            raise RuntimeError(
                "live finger geom provenance changed during wrist yaw"
            )
        reference_rotation = _validated_rigid_rotation_matrix(
            reference["rotation_matrix_world"],
            label=f"reference finger {name}",
        )
        current_rotation = _validated_rigid_rotation_matrix(
            current["rotation_matrix_world"],
            label=f"current finger {name}",
        )
        relative = _validated_rigid_rotation_matrix(
            current_rotation @ reference_rotation.T,
            label=f"relative finger {name}",
        )
        relative_rotations.append((name, relative))
    measured_rotation = relative_rotations[0][1]
    maximum_rotation_disagreement = max(
        _rotation_matrix_error_angle(relative @ measured_rotation.T)
        for _, relative in relative_rotations
    )
    for name in sorted(reference_records):
        reference_origin = np.asarray(
            reference_records[name]["origin_world"], dtype=float
        )
        current_origin = np.asarray(
            current_records[name]["origin_world"], dtype=float
        )
        predicted_origin = (
            current_eef
            + measured_rotation @ (reference_origin - reference_eef)
        )
        maximum_rigid_origin_error = max(
            maximum_rigid_origin_error,
            float(np.linalg.norm(current_origin - predicted_origin)),
        )
    rotated_reference = measured_rotation @ reference_outward
    actual_yaw = float(
        np.arctan2(
            np.dot(normal, np.cross(reference_outward, rotated_reference)),
            np.dot(reference_outward, rotated_reference),
        )
    )
    target_yaw = float(yaw_spec["yaw_angle_rad"])
    remaining_yaw = float(
        np.arctan2(
            np.sin(target_yaw - actual_yaw),
            np.cos(target_yaw - actual_yaw),
        )
    )
    absolute_error = abs(remaining_yaw)
    remaining_rotation = _validated_rigid_rotation_matrix(
        target_rotation @ measured_rotation.T,
        label="remaining wrist-yaw target rotation",
    )
    remaining_axis_angle = _rotation_matrix_axis_angle(remaining_rotation)
    target_error_angle = float(np.linalg.norm(remaining_axis_angle))
    position_drift = float(np.linalg.norm(current_eef - anchor_eef))
    anchor_position_error = anchor_eef - current_eef
    axis_error = float(np.linalg.norm(measured_rotation @ normal - normal))
    rotation_direction_valid = bool(
        actual_yaw * target_yaw >= -angular_progress_epsilon_rad
        and abs(actual_yaw)
        <= abs(target_yaw) + maximum_angle_error_rad
    )
    rigid_frame_valid = bool(
        maximum_rotation_disagreement < maximum_angle_error_rad
        and maximum_rigid_origin_error < maximum_position_drift_m
        and axis_error < maximum_angle_error_rad
    )
    progress = None
    progressed = None
    progress_source = None
    if previous_rotation_error_rad is not None:
        if not np.isfinite(previous_rotation_error_rad):
            raise ValueError("previous wrist rotation error must be finite")
        progress = float(previous_rotation_error_rad - target_error_angle)
        progress_source = "full_target_rotation_error"
    elif previous_absolute_error_rad is not None:
        if not np.isfinite(previous_absolute_error_rad):
            raise ValueError("previous wrist-yaw error must be finite")
        progress = float(previous_absolute_error_rad - absolute_error)
        progress_source = "signed_yaw_error"
    if progress is not None:
        progressed = bool(progress > angular_progress_epsilon_rad)
    position_progress = None
    position_progressed = None
    if previous_position_drift_m is not None:
        if not np.isfinite(previous_position_drift_m):
            raise ValueError("previous wrist-yaw position drift must be finite")
        position_progress = float(
            previous_position_drift_m - position_drift
        )
        position_progressed = bool(
            position_progress > position_progress_epsilon_m
        )
    rotation_attained = bool(
        absolute_error < maximum_angle_error_rad
        and target_error_angle < maximum_angle_error_rad
        and rotation_direction_valid
        and rigid_frame_valid
    )
    position_attained = bool(position_drift < maximum_position_drift_m)
    attained = bool(rotation_attained and position_attained)
    return {
        "attained": attained,
        "rotation_attained": rotation_attained,
        "position_attained": position_attained,
        "actual_yaw_rad": actual_yaw,
        "target_yaw_rad": target_yaw,
        "remaining_yaw_rad": remaining_yaw,
        "absolute_error_rad": absolute_error,
        "target_rotation_error_rad": target_error_angle,
        "measured_rotation_matrix_world": measured_rotation.tolist(),
        "remaining_rotation_matrix_world": remaining_rotation.tolist(),
        "remaining_rotation_axis_angle_world_rad": (
            remaining_axis_angle.tolist()
        ),
        "remaining_rotation_axis_angle_norm_rad": target_error_angle,
        "maximum_angle_error_rad": float(maximum_angle_error_rad),
        "reference_eef_position_world": reference_eef.tolist(),
        "anchor_eef_position_world": anchor_eef.tolist(),
        "eef_position_drift_m": position_drift,
        "anchor_position_error_world_m": (
            anchor_position_error.tolist()
        ),
        "maximum_position_drift_m": float(maximum_position_drift_m),
        "maximum_finger_rotation_disagreement_rad": (
            maximum_rotation_disagreement
        ),
        "maximum_finger_rigid_origin_error_m": maximum_rigid_origin_error,
        "table_normal_axis_error": axis_error,
        "rotation_direction_valid": rotation_direction_valid,
        "rigid_frame_valid": rigid_frame_valid,
        "angular_progress_rad": progress,
        "angular_progress_source": progress_source,
        "angular_progress_epsilon_rad": float(
            angular_progress_epsilon_rad
        ),
        "progressed": progressed,
        "angular_progressed": progressed,
        "position_progress_m": position_progress,
        "position_progress_epsilon_m": float(
            position_progress_epsilon_m
        ),
        "position_progressed": position_progressed,
    }


def _wrist_yaw_stage_budget_evidence(
    *,
    actions_used,
    maximum_actions,
    position_settle_steps,
    maximum_position_settle_steps,
    rotation_attained,
    position_attained,
):
    """Authorize the next yaw or settle action under the shared hard budget."""
    values = (
        actions_used,
        maximum_actions,
        position_settle_steps,
        maximum_position_settle_steps,
    )
    if (
        any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in values
        )
        or actions_used < 0
        or maximum_actions < 1
        or position_settle_steps < 0
        or maximum_position_settle_steps < 1
        or position_settle_steps > actions_used
    ):
        raise ValueError("wrist-yaw stage budgets are invalid")
    violations = []
    if actions_used >= maximum_actions:
        violations.append("shared_structural_waypoint_budget_exhausted")
    if (
        rotation_attained
        and not position_attained
        and position_settle_steps >= maximum_position_settle_steps
    ):
        violations.append("wrist_yaw_position_settle_budget_exhausted")
    return {
        "accepted": not violations,
        "violations": violations,
        "actions_used": int(actions_used),
        "maximum_actions": int(maximum_actions),
        "remaining_actions": int(maximum_actions - actions_used),
        "position_settle_steps": int(position_settle_steps),
        "maximum_position_settle_steps": int(
            maximum_position_settle_steps
        ),
        "rotation_attained": bool(rotation_attained),
        "position_attained": bool(position_attained),
        "next_stage": (
            "position_settle"
            if rotation_attained and not position_attained
            else "rotation_with_anchor_compensation"
        ),
    }


def _wrist_yaw_step_gate(
    *,
    stage,
    overhead_guard,
    robot_nonrobot_contact_gate,
    action_evidence,
    attainment_evidence,
    consecutive_angular_stall_steps,
    consecutive_position_stall_steps,
    maximum_stall_steps,
):
    """Fail-closed gate for one measured high-space wrist-yaw frame."""
    if (
        stage not in {
            "rotation_with_anchor_compensation",
            "position_settle",
        }
        or maximum_stall_steps < 1
        or consecutive_angular_stall_steps < 0
        or consecutive_position_stall_steps < 0
    ):
        raise ValueError("wrist-yaw stall counters are invalid")
    violations = []
    if not overhead_guard.get("accepted", False):
        violations.append("high_free_space_overhead_guard_failed")
    if not robot_nonrobot_contact_gate.get("accepted", False):
        violations.append("forbidden_robot_native_contact_during_wrist_yaw")
    if action_evidence.get("action_will_clip", False):
        violations.append("wrist_yaw_action_would_clip")
    if not action_evidence.get("translation_direction_valid", False):
        violations.append("wrist_yaw_anchor_correction_direction_invalid")
    if stage == "position_settle" and not action_evidence.get(
        "orientation_hold_commanded", False
    ):
        violations.append("wrist_yaw_position_settle_changed_orientation")
    if not attainment_evidence.get("rotation_direction_valid", False):
        violations.append("wrist_yaw_rotation_direction_invalid")
    if not attainment_evidence.get("rigid_frame_valid", False):
        violations.append("wrist_yaw_finger_frame_not_rigid")
    if (
        not attainment_evidence.get("rotation_attained", False)
        and consecutive_angular_stall_steps >= maximum_stall_steps
    ):
        violations.append("wrist_yaw_angular_progress_stalled")
    if (
        action_evidence.get("position_correction_requested", False)
        and not attainment_evidence.get("position_attained", False)
        and consecutive_position_stall_steps >= maximum_stall_steps
    ):
        violations.append("wrist_yaw_anchor_position_progress_stalled")
    return {
        "accepted": not violations,
        "violations": violations,
        "stage": stage,
        "consecutive_angular_stall_steps": int(
            consecutive_angular_stall_steps
        ),
        "consecutive_position_stall_steps": int(
            consecutive_position_stall_steps
        ),
        "maximum_stall_steps": int(maximum_stall_steps),
    }


def _compiled_hypothetical_wrist_yaw_plan(
    env,
    *,
    plate_position,
    eef_position,
    contact_xy,
    outside_clearance_m,
    plate_approach_eef_height,
    position_action_scale,
    yaw_spec,
    table_normal_evidence,
):
    """Compile one diagnostic-only rigid wrist-yaw counterfactual."""
    plate_position = np.asarray(plate_position, dtype=float)
    eef_position = np.asarray(eef_position, dtype=float)
    contact_xy = np.asarray(contact_xy, dtype=float)
    outward = np.asarray(
        yaw_spec["target_outward_direction_xy"], dtype=float
    )
    rotation = np.asarray(yaw_spec["rotation_matrix_world"], dtype=float)
    hypothetical_env, rigid_transform = _hypothetical_finger_yaw_env(
        env,
        eef_position=eef_position,
        rotation=rotation,
    )
    outside_side, side_contact, compiled = (
        _compiled_native_side_contact_plan(
            hypothetical_env,
            plate_position,
            eef_position,
            outward,
            contact_xy,
            outside_clearance_m,
        )
    )
    model, data = hypothetical_env.sim.model, hypothetical_env.sim.data
    rim_names = set(compiled["plate_rim_geoms"])
    table_names = set(compiled["table_geoms"])
    rim_bounds = []
    table_bounds = []
    finger_bounds = []
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        if name in rim_names:
            rim_bounds.append(
                (name, *_compiled_geom_world_aabb(model, data, geom_id))
            )
        if name in table_names:
            table_bounds.append(
                (name, *_compiled_geom_world_aabb(model, data, geom_id))
            )
        semantic_side = _semantic_finger_side(body_name)
        if semantic_side is not None and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        ):
            finger_bounds.append(
                (
                    name,
                    semantic_side,
                    *_compiled_geom_world_aabb(model, data, geom_id),
                )
            )
    if not rim_bounds or not table_bounds or not finger_bounds:
        raise RuntimeError(
            "hypothetical wrist-yaw compiled bound inventory is incomplete"
        )

    def translated_finger_bounds(target_eef):
        delta = np.asarray(target_eef, dtype=float) - eef_position
        return [
            (name, side, center + delta, half_size.copy())
            for name, side, center, half_size in finger_bounds
        ]

    outside_guard = _outside_side_guard_from_world_aabbs(
        plate_position=plate_position,
        outward_direction_xy=outward,
        rim_bounds=rim_bounds,
        finger_bounds=translated_finger_bounds(outside_side),
        required_outside_clearance_m=outside_clearance_m,
        table_bounds=table_bounds,
        required_finger_table_clearance_m=compiled[
            "finger_table_clearance_derivation"
        ]["required_clearance_m"],
        outside_clearance_derivation={
            "source": "hypothetical rigid-yaw compiled support",
            "executed": False,
        },
        finger_table_clearance_derivation=compiled[
            "finger_table_clearance_derivation"
        ],
    )
    side_finger_bounds = translated_finger_bounds(side_contact)
    plate_outward_support = float(compiled["plate_outward_support_m"])
    side_contact_plate_clearance_by_side = {}
    for side in ("left", "right"):
        side_inward_support = min(
            float(
                np.dot(center[:2] - plate_position[:2], outward)
                - np.dot(half_size[:2], np.abs(outward))
            )
            for _, semantic_side, center, half_size in side_finger_bounds
            if semantic_side == side
        )
        side_contact_plate_clearance_by_side[side] = float(
            side_inward_support - plate_outward_support
        )

    outside_high = outside_side.copy()
    outside_high[2] = plate_position[2] + plate_approach_eef_height
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
        for axis in np.flatnonzero(np.abs(required_action) > 1.0 + 1e-9)
    ]
    violations = []
    if compiled["dual_finger_contact_skew_m"] > outside_clearance_m:
        violations.append(
            "hypothetical_dual_finger_contact_skew_exceeds_outside_clearance"
        )
    violations.extend(
        f"hypothetical_outside_guard:{violation}"
        for violation in outside_guard["violations"]
    )
    return {
        "diagnostic_only": True,
        "executed": False,
        "route_selection_candidate": False,
        "selection_eligible": False,
        "yaw": yaw_spec,
        "table_normal_derivation": table_normal_evidence,
        "rigid_finger_transform": rigid_transform,
        "compiled_geometry": compiled,
        "dual_finger_contact_skew_m": float(
            compiled["dual_finger_contact_skew_m"]
        ),
        "maximum_dual_finger_contact_skew_m": float(
            outside_clearance_m
        ),
        "outside_high_target": outside_high.tolist(),
        "outside_side_target": outside_side.tolist(),
        "side_contact_target": side_contact.tolist(),
        "outside_guard": outside_guard,
        "side_contact_signed_plate_clearance_by_side_m": (
            side_contact_plate_clearance_by_side
        ),
        "selected_finger_table_clearance_m": float(
            compiled["vertical_feasibility"][
                "selected_finger_table_clearance_m"
            ]
        ),
        "outside_high_required_action": required_action.tolist(),
        "outside_high_bounded_action": bounded_action.tolist(),
        "outside_high_action_peak": float(
            np.max(np.abs(required_action))
        ),
        "outside_high_action_norm": float(np.linalg.norm(required_action)),
        "outside_high_clipped_action_axes": clipped_axes,
        "outside_high_action_will_clip": bool(clipped_axes),
        "hypothetical_compiled_geometry_violations": violations,
        "hypothetical_compiled_geometry_eligible": not violations,
        "provenance": {
            "native_pose_source": (
                "current compiled geom_xpos, geom_xmat, and geom_aabb"
            ),
            "counterfactual": (
                "rigid yaw of collidable finger geoms about the current EEF "
                "and compiled native table normal"
            ),
            "policy_or_controller_action_executed": False,
        },
    }


def _real_recompile_wrist_yaw_candidate(
    env,
    *,
    selected_candidate,
    plate_position,
    eef_position,
    outside_clearance_m,
    plate_approach_eef_height,
    position_action_scale,
    attainment_evidence,
    table_normal_evidence,
    recompile_stage="post_wrist_yaw",
):
    """Compile the descent route directly from the attained live sim pose."""
    if recompile_stage not in {
        "post_wrist_yaw",
        "post_center_high_reacquire",
    }:
        raise ValueError("real wrist-yaw recompile stage is invalid")
    plate_position = np.asarray(plate_position, dtype=float)
    eef_position = np.asarray(eef_position, dtype=float)
    if (
        plate_position.shape != (3,)
        or eef_position.shape != (3,)
        or not np.all(np.isfinite(plate_position))
        or not np.all(np.isfinite(eef_position))
    ):
        raise ValueError("real wrist-yaw recompile poses must be finite and 3-D")
    final_position_drift = float(
        attainment_evidence.get("eef_position_drift_m", np.inf)
    )
    final_position_limit = float(
        attainment_evidence.get("maximum_position_drift_m", -np.inf)
    )
    if not (
        attainment_evidence.get("attained", False)
        and attainment_evidence.get("rotation_attained", False)
        and attainment_evidence.get("position_attained", False)
        and np.isfinite(final_position_drift)
        and np.isfinite(final_position_limit)
        and final_position_limit > 0.0
        and final_position_drift < final_position_limit
    ):
        raise RuntimeError(
            "real wrist-yaw geometry recompile requires simultaneous measured "
            "yaw and strict anchor-position attainment"
        )
    relation = selected_candidate.get("native_push_direction_relations")
    if relation != ["trailing_minus_push"] or not selected_candidate.get(
        "wrist_yaw_route_selected", False
    ):
        raise RuntimeError(
            "real wrist-yaw recompile received an unselected direction"
        )
    outward = np.asarray(
        selected_candidate["outward_direction_xy"], dtype=float
    )
    live_table_normal, live_table_normal_evidence = (
        _compiled_table_normal_evidence(env)
    )
    if not np.allclose(
        live_table_normal,
        table_normal_evidence["table_normal_world"],
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "native table frame changed before real wrist-yaw recompile"
        )
    outside_side, side_contact, compiled = (
        _compiled_native_side_contact_plan(
            env,
            plate_position,
            eef_position,
            outward,
            np.asarray(selected_candidate["point_xy"], dtype=float),
            outside_clearance_m,
        )
    )
    outside_high = outside_side.copy()
    outside_high[2] = plate_position[2] + plate_approach_eef_height
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
        for axis in np.flatnonzero(np.abs(required_action) > 1.0 + 1e-9)
    ]

    # Rebuild the planned outside-side guard from the actual post-yaw geom
    # frames.  The only projection is a rigid translation to the compiled
    # target; no hypothetical orientation is used for this route.
    model, data = env.sim.model, env.sim.data
    rim_names = set(compiled["plate_rim_geoms"])
    table_names = set(compiled["table_geoms"])
    rim_bounds = []
    table_bounds = []
    finger_bounds = []
    for geom_id in range(int(model.ngeom)):
        name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
        body_name = model.body_id2name(int(model.geom_bodyid[geom_id])) or ""
        if name in rim_names:
            rim_bounds.append(
                (name, *_compiled_geom_world_aabb(model, data, geom_id))
            )
        if name in table_names:
            table_bounds.append(
                (name, *_compiled_geom_world_aabb(model, data, geom_id))
            )
        semantic_side = _semantic_finger_side(body_name)
        if semantic_side is not None and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        ):
            finger_bounds.append(
                (
                    name,
                    semantic_side,
                    *_compiled_geom_world_aabb(model, data, geom_id),
                )
            )
    if not rim_bounds or not table_bounds or not finger_bounds:
        raise RuntimeError(
            "attained live wrist pose has incomplete native bound inventory"
        )
    translation = outside_side - np.asarray(eef_position, dtype=float)
    planned_finger_bounds = [
        (name, side, center + translation, half_size.copy())
        for name, side, center, half_size in finger_bounds
    ]
    planned_outside_guard = _outside_side_guard_from_world_aabbs(
        plate_position=plate_position,
        outward_direction_xy=outward,
        rim_bounds=rim_bounds,
        finger_bounds=planned_finger_bounds,
        required_outside_clearance_m=outside_clearance_m,
        table_bounds=table_bounds,
        required_finger_table_clearance_m=compiled[
            "finger_table_clearance_derivation"
        ]["required_clearance_m"],
        outside_clearance_derivation={
            "source": (
                "attained live geom frames rigidly translated to the direct "
                "compiled outside-side target"
            ),
            "hypothetical_orientation_used": False,
        },
        finger_table_clearance_derivation=compiled[
            "finger_table_clearance_derivation"
        ],
    )
    skew = float(compiled["dual_finger_contact_skew_m"])
    vertical = compiled["vertical_feasibility"]
    selected_table_clearance = float(
        vertical["selected_finger_table_clearance_m"]
    )
    required_table_clearance = float(
        vertical["required_finger_table_clearance_m"]
    )
    planned_table_clearance = float(
        planned_outside_guard["finger_table_vertical_clearance_m"]
    )
    rim_coverage = vertical["selected_rim_overlap_by_side"]
    violations = []
    if not skew < float(outside_clearance_m):
        violations.append("real_dual_finger_contact_skew_not_strictly_below_5mm")
    violations.extend(
        f"real_planned_outside_guard:{violation}"
        for violation in planned_outside_guard["violations"]
    )
    if selected_table_clearance <= required_table_clearance:
        violations.append("real_selected_finger_table_clearance_not_strict")
    if planned_table_clearance <= required_table_clearance:
        violations.append("real_planned_finger_table_clearance_not_strict")
    for side in ("left", "right"):
        evidence = rim_coverage.get(side, {})
        if (
            float(evidence.get("overlap_m", 0.0)) <= 0.0
            or not evidence.get("rim_center_covered", False)
        ):
            violations.append(f"real_{side}_finger_plate_rim_coverage_failed")
    if violations:
        raise RuntimeError(
            "attained wrist pose failed direct live 5 mm skew or "
            "table/plate geometry revalidation: "
            + json.dumps(
                {
                    "violations": violations,
                    "dual_finger_contact_skew_m": skew,
                    "maximum_dual_finger_contact_skew_m": float(
                        outside_clearance_m
                    ),
                    "planned_outside_guard": planned_outside_guard,
                    "vertical_feasibility": vertical,
                },
                sort_keys=True,
            )
        )
    direct_revalidation = {
        "performed": True,
        "recompile_stage": recompile_stage,
        "source": (
            "direct live compiled geom_xpos/geom_xmat after measured "
            + (
                "wrist-yaw attainment"
                if recompile_stage == "post_wrist_yaw"
                else "orientation-hold center-high reacquisition"
            )
        ),
        "live_eef_position_world": eef_position.tolist(),
        "live_plate_position_world": plate_position.tolist(),
        "live_center_high_target_world": center_high.tolist(),
        "attainment_evidence": attainment_evidence,
        "table_normal_derivation": live_table_normal_evidence,
        "selected_native_push_direction_relation": relation[0],
        "selected_outward_direction_xy": outward.tolist(),
        "dual_finger_contact_skew_m": skew,
        "maximum_dual_finger_contact_skew_m": float(
            outside_clearance_m
        ),
        "strict_dual_finger_skew_accepted": bool(
            skew < float(outside_clearance_m)
        ),
        "planned_outside_guard": planned_outside_guard,
        "selected_finger_table_clearance_m": selected_table_clearance,
        "planned_finger_table_clearance_m": planned_table_clearance,
        "required_finger_table_clearance_m": required_table_clearance,
        "selected_rim_overlap_by_side": rim_coverage,
        "eligible": True,
        "hypothetical_geometry_used_for_descent": False,
    }
    return {
        **selected_candidate,
        "diagnostic_only": False,
        "route_selection_candidate": True,
        "selection_eligible": True,
        "selection_violations": [],
        "center_high_target": center_high.tolist(),
        "outside_high_target": outside_high.tolist(),
        "outside_side_target": outside_side.tolist(),
        "side_contact_target": side_contact.tolist(),
        "compiled_geometry": compiled,
        "dual_finger_contact_skew_m": skew,
        "outside_high_required_action": required_action.tolist(),
        "outside_high_bounded_action": bounded_action.tolist(),
        "outside_high_action_peak": float(np.max(np.abs(required_action))),
        "outside_high_action_norm": float(np.linalg.norm(required_action)),
        "outside_high_clipped_action_axes": clipped_axes,
        "outside_high_action_will_clip": bool(clipped_axes),
        "real_sim_geometry_recompile": direct_revalidation,
    }


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


def _select_executable_wrist_yaw_candidate(candidates):
    """Select only the preregistered native trailing-minus-push route."""
    candidates = list(candidates)
    trailing = [
        candidate
        for candidate in candidates
        if candidate.get("native_push_direction_relations")
        == ["trailing_minus_push"]
    ]
    if len(trailing) != 1:
        raise RuntimeError(
            "wrist-yaw route requires exactly one native trailing_minus_push "
            "candidate"
        )
    selected = trailing[0]
    yaw_diagnostic = selected.get("hypothetical_wrist_yaw", {})
    if not yaw_diagnostic.get(
        "hypothetical_compiled_geometry_eligible", False
    ):
        raise RuntimeError(
            "preregistered trailing_minus_push wrist-yaw route failed its "
            "hypothetical compiled geometry gate"
        )
    selected_magnitude = abs(
        float(yaw_diagnostic["yaw"]["yaw_angle_rad"])
    )
    # Canonicalize execution metadata after all candidates have retained their
    # hypothetical geometry diagnostics.  No legacy cardinal or tangent can be
    # promoted at runtime if the sole preregistered route later fails.
    for candidate in candidates:
        candidate["wrist_yaw_route_selected"] = False
        candidate["route_selection_candidate"] = False
        candidate["diagnostic_only"] = True
        candidate["selection_eligible"] = False
    selected["route_selection_candidate"] = True
    selected["diagnostic_only"] = False
    selected["selection_eligible"] = True
    selected["pre_yaw_compiled_geometry_violations"] = list(
        selected.get("selection_violations", ())
    )
    selected["selection_violations"] = []
    selected["wrist_yaw_route_selected"] = True
    selected["wrist_yaw_route_selection_basis"] = {
        "required_relation": "trailing_minus_push",
        "sole_preregistered_actual_route": True,
        "minimum_absolute_yaw_among_hypothetically_eligible": False,
        "absolute_yaw_rad": selected_magnitude,
        "superpod_diagnostic_authorization": {
            "job_id": "502381",
            "commit": "3376794",
            "observed_geometry_eligible_relation": "trailing_minus_push",
            "runtime_revalidation_still_required": True,
        },
        "old_plus_x_route_fallback_permitted": False,
        "runtime_tangent_fallback_permitted": False,
        "requires_high_free_space_execution": True,
        "requires_real_pose_attainment": True,
        "requires_real_sim_geometry_recompile_before_descent": True,
    }
    if sum(
        bool(candidate.get("wrist_yaw_route_selected", False))
        for candidate in candidates
    ) != 1:
        raise RuntimeError("wrist-yaw execution route is not unique")
    return selected


def _strict_wrist_yaw_segment_plan(
    *,
    yaw_spec,
    native_action_spec,
    rotation_spec,
):
    """Split one native-frame yaw into the minimum strict OSC-bound segments."""
    normal = np.asarray(yaw_spec.get("table_normal_world", ()), dtype=float)
    reference_outward = np.asarray(
        yaw_spec.get("reference_outward_direction_xy", ()), dtype=float
    )
    target_outward = np.asarray(
        yaw_spec.get("target_outward_direction_xy", ()), dtype=float
    )
    declared_axis_angle = np.asarray(
        yaw_spec.get("axis_angle_world_rad", ()), dtype=float
    )
    total_yaw = float(yaw_spec.get("yaw_angle_rad", np.nan))
    action_low = np.asarray(native_action_spec.get("low", ()), dtype=float)
    action_high = np.asarray(native_action_spec.get("high", ()), dtype=float)
    rotation_scale = np.asarray(
        rotation_spec.get("output_axis_angle_rad_per_action", ()), dtype=float
    )
    rotation_output_min = np.asarray(
        rotation_spec.get("output_min_axis_angle_rad", ()), dtype=float
    )
    rotation_output_max = np.asarray(
        rotation_spec.get("output_max_axis_angle_rad", ()), dtype=float
    )
    if (
        normal.shape != (3,)
        or reference_outward.shape != (2,)
        or target_outward.shape != (2,)
        or declared_axis_angle.shape != (3,)
        or action_low.shape != (7,)
        or action_high.shape != (7,)
        or rotation_scale.shape != (6,)
        or rotation_output_min.shape != (6,)
        or rotation_output_max.shape != (6,)
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(reference_outward))
        or not np.all(np.isfinite(target_outward))
        or not np.all(np.isfinite(declared_axis_angle))
        or np.linalg.norm(reference_outward) <= 1e-9
        or np.linalg.norm(target_outward) <= 1e-9
        or not np.isfinite(total_yaw)
        or not np.all(np.isfinite(action_low))
        or not np.all(np.isfinite(action_high))
        or not np.all(action_low < action_high)
        or np.any(rotation_scale[3:6] <= 0.0)
        or np.any(rotation_output_min[3:6] >= 0.0)
        or np.any(rotation_output_max[3:6] <= 0.0)
        or abs(float(np.linalg.norm(normal)) - 1.0) > 1e-7
    ):
        raise ValueError("native wrist-yaw segmentation inputs are invalid")
    if not native_action_spec.get("runtime_resolved", False) or not rotation_spec.get(
        "runtime_resolved", False
    ):
        raise RuntimeError("wrist-yaw segmentation requires live native OSC bounds")

    reference_3d = np.r_[
        reference_outward / float(np.linalg.norm(reference_outward)), 0.0
    ]
    normalized_target_3d = np.r_[
        target_outward / float(np.linalg.norm(target_outward)), 0.0
    ]
    cross_matrix = np.array(
        [
            [0.0, -normal[2], normal[1]],
            [normal[2], 0.0, -normal[0]],
            [-normal[1], normal[0], 0.0],
        ],
        dtype=float,
    )
    total_rotation = _validated_rigid_rotation_matrix(
        np.eye(3)
        + np.sin(total_yaw) * cross_matrix
        + (1.0 - np.cos(total_yaw)) * (cross_matrix @ cross_matrix),
        label="segmented total wrist yaw",
    )
    declared_rotation = _validated_rigid_rotation_matrix(
        yaw_spec.get("rotation_matrix_world", ()),
        label="declared segmented wrist yaw",
    )
    if not (
        np.allclose(
            declared_axis_angle,
            normal * total_yaw,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            declared_rotation,
            total_rotation,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            total_rotation @ reference_3d,
            normalized_target_3d,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise RuntimeError(
            "native-frame wrist-yaw identity changed before segmentation"
        )

    signed_unit_yaw = -1.0 if total_yaw < 0.0 else 1.0
    signed_action_per_yaw = (
        signed_unit_yaw * normal / rotation_scale[3:6]
    )
    axis_capacities = []
    for local_axis, coefficient in enumerate(signed_action_per_yaw):
        if abs(float(coefficient)) <= 1e-15:
            continue
        action_index = int(local_axis + 3)
        action_bound = float(
            action_high[action_index]
            if coefficient > 0.0
            else action_low[action_index]
        )
        capacity = float(action_bound / coefficient)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise RuntimeError(
                "native OSC rotation bounds have no capacity in selected yaw direction"
            )
        axis_capacities.append(
            {
                "rotation_action_index": action_index,
                "table_normal_component": float(normal[local_axis]),
                "signed_action_per_positive_yaw_magnitude": float(coefficient),
                "directional_native_action_bound": action_bound,
                "directional_yaw_capacity_rad": capacity,
            }
        )
    if not axis_capacities:
        raise RuntimeError("selected native yaw axis has no controlled component")
    limiting_directional_axis = min(
        axis_capacities,
        key=lambda record: record["directional_yaw_capacity_rad"],
    )
    native_axis_angle_norm_bound = float(
        np.min(
            np.minimum(
                -rotation_output_min[3:6],
                rotation_output_max[3:6],
            )
        )
    )
    native_yaw_capacity = float(
        min(
            limiting_directional_axis["directional_yaw_capacity_rad"],
            native_axis_angle_norm_bound,
        )
    )
    strict_capacity = float(np.nextafter(native_yaw_capacity, 0.0))
    if not np.isfinite(strict_capacity) or strict_capacity <= 0.0:
        raise RuntimeError("native yaw capacity has no representable strict interior")
    segment_count = max(1, int(np.ceil(abs(total_yaw) / strict_capacity)))
    segment_delta = float(total_yaw / segment_count)
    if not abs(segment_delta) < native_yaw_capacity:
        raise RuntimeError("minimum wrist-yaw segmentation is not strictly unclipped")

    segments = []
    previous_outward = reference_3d[:2].copy()
    previous_target_yaw = 0.0
    for segment_index in range(1, segment_count + 1):
        target_yaw = float(total_yaw * segment_index / segment_count)
        rotation = (
            np.eye(3)
            + np.sin(target_yaw) * cross_matrix
            + (1.0 - np.cos(target_yaw)) * (cross_matrix @ cross_matrix)
        )
        target_outward = (rotation @ reference_3d)[:2]
        relative_spec = _hypothetical_wrist_yaw_specs(
            reference_outward_direction_xy=previous_outward,
            target_outward_directions_xy=[target_outward],
            table_normal_world=normal,
        )[0]
        cumulative_spec = _hypothetical_wrist_yaw_specs(
            reference_outward_direction_xy=reference_3d[:2],
            target_outward_directions_xy=[target_outward],
            table_normal_world=normal,
        )[0]
        expected_delta = float(target_yaw - previous_target_yaw)
        if not (
            np.isclose(
                relative_spec["yaw_angle_rad"], expected_delta, rtol=0.0, atol=1e-12
            )
            and np.isclose(
                cumulative_spec["yaw_angle_rad"], target_yaw, rtol=0.0, atol=1e-12
            )
        ):
            raise RuntimeError("native-frame wrist-yaw segment derivation diverged")
        required_rotation_action = (
            normal * expected_delta / rotation_scale[3:6]
        )
        if not np.all(
            (action_low[3:6] < required_rotation_action)
            & (required_rotation_action < action_high[3:6])
        ):
            raise RuntimeError("wrist-yaw segment would touch or cross native bounds")
        segments.append(
            {
                "segment_index": int(segment_index),
                "absolute_target_yaw_rad": target_yaw,
                "relative_target_yaw_rad": expected_delta,
                "target_outward_direction_xy": target_outward.tolist(),
                "relative_yaw_spec": relative_spec,
                "cumulative_yaw_spec": cumulative_spec,
                "required_rotation_action": required_rotation_action.tolist(),
                "required_rotation_action_peak": float(
                    np.max(np.abs(required_rotation_action))
                ),
                "axis_angle_norm_rad": abs(expected_delta),
                "strictly_inside_native_yaw_capacity": True,
            }
        )
        previous_outward = target_outward
        previous_target_yaw = target_yaw
    if not np.isclose(previous_target_yaw, total_yaw, rtol=0.0, atol=1e-15):
        raise RuntimeError("wrist-yaw segment targets do not sum to the native target")
    return {
        "formula": (
            "directional yaw capacity is the minimum live native rotation-action "
            "bound divided by signed table-normal action demand; take nextafter "
            "toward zero, N=ceil(abs(total_yaw)/strict_capacity), and use N equal "
            "signed native-frame yaw targets"
        ),
        "total_native_frame_yaw_rad": total_yaw,
        "total_native_frame_axis_angle_world_rad": (normal * total_yaw).tolist(),
        "table_normal_world": normal.tolist(),
        "native_action_spec_source": native_action_spec.get("source"),
        "native_rotation_spec_source": rotation_spec.get("source"),
        "directional_axis_capacities": axis_capacities,
        "limiting_directional_axis": limiting_directional_axis,
        "native_axis_angle_norm_bound_rad": native_axis_angle_norm_bound,
        "native_directional_yaw_capacity_rad": native_yaw_capacity,
        "strict_directional_yaw_capacity_rad": strict_capacity,
        "minimum_segment_count": int(segment_count),
        "equal_segment_target_yaw_rad": segment_delta,
        "segments": segments,
        "hardcoded_segment_count_used": False,
        "runtime_fallback_permitted": False,
    }


def _strict_axis_angle_action_capacity_evidence(
    *,
    axis_angle_world,
    native_action_spec,
    rotation_spec,
):
    """Prove one arbitrary world axis-angle is strict inside live OSC bounds."""
    axis_angle = np.asarray(axis_angle_world, dtype=float)
    action_low = np.asarray(native_action_spec.get("low", ()), dtype=float)
    action_high = np.asarray(native_action_spec.get("high", ()), dtype=float)
    rotation_scale = np.asarray(
        rotation_spec.get("output_axis_angle_rad_per_action", ()), dtype=float
    )
    output_min = np.asarray(
        rotation_spec.get("output_min_axis_angle_rad", ()), dtype=float
    )
    output_max = np.asarray(
        rotation_spec.get("output_max_axis_angle_rad", ()), dtype=float
    )
    if (
        axis_angle.shape != (3,)
        or action_low.shape != (7,)
        or action_high.shape != (7,)
        or rotation_scale.shape != (6,)
        or output_min.shape != (6,)
        or output_max.shape != (6,)
        or not np.all(np.isfinite(axis_angle))
        or not np.all(np.isfinite(action_low))
        or not np.all(np.isfinite(action_high))
        or not np.all(action_low < action_high)
        or np.any(rotation_scale[3:6] <= 0.0)
        or np.any(output_min[3:6] >= 0.0)
        or np.any(output_max[3:6] <= 0.0)
    ):
        raise ValueError("strict axis-angle capacity inputs are invalid")
    if not native_action_spec.get("runtime_resolved", False) or not rotation_spec.get(
        "runtime_resolved", False
    ):
        raise RuntimeError("strict axis-angle capacity requires live OSC bounds")
    axis_angle_norm = float(np.linalg.norm(axis_angle))
    native_norm_bound = float(
        np.min(np.minimum(-output_min[3:6], output_max[3:6]))
    )
    if axis_angle_norm <= 1e-15:
        directional_capacity = native_norm_bound
        directional_axes = []
    else:
        unit_axis = axis_angle / axis_angle_norm
        action_per_axis_angle_norm = unit_axis / rotation_scale[3:6]
        directional_axes = []
        for local_axis, coefficient in enumerate(action_per_axis_angle_norm):
            if abs(float(coefficient)) <= 1e-15:
                continue
            action_index = int(local_axis + 3)
            action_bound = float(
                action_high[action_index]
                if coefficient > 0.0
                else action_low[action_index]
            )
            capacity = float(action_bound / coefficient)
            if not np.isfinite(capacity) or capacity <= 0.0:
                raise RuntimeError(
                    "native OSC has no capacity along the remaining axis-angle"
                )
            directional_axes.append(
                {
                    "rotation_action_index": action_index,
                    "unit_axis_component": float(unit_axis[local_axis]),
                    "directional_action_bound": action_bound,
                    "directional_axis_angle_capacity_rad": capacity,
                }
            )
        if not directional_axes:
            raise RuntimeError("remaining axis-angle has no controlled component")
        directional_capacity = min(
            record["directional_axis_angle_capacity_rad"]
            for record in directional_axes
        )
    derived_capacity = float(min(native_norm_bound, directional_capacity))
    strict_capacity = float(np.nextafter(derived_capacity, 0.0))
    minimum_segment_count = max(
        1,
        int(np.ceil(axis_angle_norm / strict_capacity)),
    )
    required_rotation_action = axis_angle / rotation_scale[3:6]
    action_strict = bool(
        np.all(action_low[3:6] < required_rotation_action)
        and np.all(required_rotation_action < action_high[3:6])
    )
    accepted = bool(
        minimum_segment_count == 1
        and axis_angle_norm < derived_capacity
        and action_strict
    )
    return {
        "accepted": accepted,
        "formula": (
            "derive the directional live native-action capacity of the complete "
            "world axis-angle, intersect it with the existing 0.5 rad axis-angle "
            "norm bound, and require the command to remain strictly interior"
        ),
        "axis_angle_world_rad": axis_angle.tolist(),
        "axis_angle_norm_rad": axis_angle_norm,
        "native_axis_angle_norm_bound_rad": native_norm_bound,
        "directional_action_axis_capacities": directional_axes,
        "derived_directional_axis_angle_capacity_rad": derived_capacity,
        "strict_directional_axis_angle_capacity_rad": strict_capacity,
        "minimum_segment_count": int(minimum_segment_count),
        "required_rotation_action": required_rotation_action.tolist(),
        "required_rotation_action_peak": float(
            np.max(np.abs(required_rotation_action))
        ),
        "strictly_inside_native_action_bounds": action_strict,
        "native_action_spec_source": native_action_spec.get("source"),
        "native_rotation_spec_source": rotation_spec.get("source"),
    }


def _absolute_wrist_yaw_runtime_target(
    *,
    planned_segment,
    absolute_start_attainment,
    total_yaw_spec,
    native_action_spec,
    rotation_spec,
):
    """Derive one live relative command toward a fixed absolute yaw waypoint."""
    planned_absolute_yaw = float(
        planned_segment.get("absolute_target_yaw_rad", np.nan)
    )
    actual_absolute_yaw = float(
        absolute_start_attainment.get("actual_yaw_rad", np.nan)
    )
    measured_remaining_yaw = float(
        absolute_start_attainment.get("remaining_yaw_rad", np.nan)
    )
    measured_target_yaw = float(
        absolute_start_attainment.get("target_yaw_rad", np.nan)
    )
    normal = np.asarray(
        total_yaw_spec.get("table_normal_world", ()), dtype=float
    )
    original_reference_outward = np.asarray(
        total_yaw_spec.get("reference_outward_direction_xy", ()), dtype=float
    )
    planned_outward = np.asarray(
        planned_segment.get("target_outward_direction_xy", ()), dtype=float
    )
    measured_absolute_rotation = np.asarray(
        absolute_start_attainment.get("measured_rotation_matrix_world", ()),
        dtype=float,
    )
    measured_remaining_rotation = np.asarray(
        absolute_start_attainment.get("remaining_rotation_matrix_world", ()),
        dtype=float,
    )
    measured_remaining_axis_angle = np.asarray(
        absolute_start_attainment.get(
            "remaining_rotation_axis_angle_world_rad", ()
        ),
        dtype=float,
    )
    planned_absolute_rotation = np.asarray(
        planned_segment.get("cumulative_yaw_spec", {}).get(
            "rotation_matrix_world", ()
        ),
        dtype=float,
    )
    if (
        not np.isfinite(planned_absolute_yaw)
        or not np.isfinite(actual_absolute_yaw)
        or not np.isfinite(measured_remaining_yaw)
        or not np.isfinite(measured_target_yaw)
        or normal.shape != (3,)
        or original_reference_outward.shape != (2,)
        or planned_outward.shape != (2,)
        or measured_absolute_rotation.shape != (3, 3)
        or measured_remaining_rotation.shape != (3, 3)
        or measured_remaining_axis_angle.shape != (3,)
        or planned_absolute_rotation.shape != (3, 3)
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(original_reference_outward))
        or not np.all(np.isfinite(planned_outward))
        or not np.all(np.isfinite(measured_absolute_rotation))
        or not np.all(np.isfinite(measured_remaining_rotation))
        or not np.all(np.isfinite(measured_remaining_axis_angle))
        or not np.all(np.isfinite(planned_absolute_rotation))
    ):
        raise ValueError("absolute wrist-yaw runtime target inputs are invalid")
    if not (
        absolute_start_attainment.get("rigid_frame_valid", False)
        and absolute_start_attainment.get("position_attained", False)
        and absolute_start_attainment.get("rotation_direction_valid", False)
    ):
        raise RuntimeError(
            "absolute wrist-yaw waypoint requires a valid live native frame and "
            "anchor before deriving its relative command"
        )

    measured_absolute_rotation = _validated_rigid_rotation_matrix(
        measured_absolute_rotation,
        label="live cumulative wrist-yaw rotation",
    )
    planned_absolute_rotation = _validated_rigid_rotation_matrix(
        planned_absolute_rotation,
        label="planned absolute wrist-yaw rotation",
    )
    expected_remaining_rotation = _validated_rigid_rotation_matrix(
        planned_absolute_rotation @ measured_absolute_rotation.T,
        label="live correction to absolute wrist-yaw target",
    )
    expected_remaining_axis_angle = _rotation_matrix_axis_angle(
        expected_remaining_rotation
    )
    if not (
        np.allclose(
            measured_remaining_rotation,
            expected_remaining_rotation,
            rtol=0.0,
            atol=1e-9,
        )
        and np.allclose(
            measured_remaining_axis_angle,
            expected_remaining_axis_angle,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise RuntimeError(
            "live full axis-angle evidence does not close the absolute native-"
            "frame wrist target"
        )
    expected_remaining_yaw = float(
        np.arctan2(
            np.sin(planned_absolute_yaw - actual_absolute_yaw),
            np.cos(planned_absolute_yaw - actual_absolute_yaw),
        )
    )
    if not (
        np.isclose(
            measured_target_yaw,
            planned_absolute_yaw,
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            measured_remaining_yaw,
            expected_remaining_yaw,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError(
            "live absolute wrist-yaw evidence diverged from the planned native-"
            "frame waypoint"
        )

    reference_3d = np.r_[
        original_reference_outward
        / float(np.linalg.norm(original_reference_outward)),
        0.0,
    ]
    cross_matrix = np.array(
        [
            [0.0, -normal[2], normal[1]],
            [normal[2], 0.0, -normal[0]],
            [-normal[1], normal[0], 0.0],
        ],
        dtype=float,
    )
    live_absolute_rotation = (
        np.eye(3)
        + np.sin(actual_absolute_yaw) * cross_matrix
        + (1.0 - np.cos(actual_absolute_yaw))
        * (cross_matrix @ cross_matrix)
    )
    live_reference_outward = (live_absolute_rotation @ reference_3d)[:2]
    relative_yaw_spec = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=live_reference_outward,
        target_outward_directions_xy=[planned_outward],
        table_normal_world=normal,
    )[0]
    local_reference_3d = np.r_[live_reference_outward, 0.0]
    full_rotated_reference = expected_remaining_rotation @ local_reference_3d
    full_rotated_reference_xy_norm = float(
        np.linalg.norm(full_rotated_reference[:2])
    )
    if (
        not np.all(np.isfinite(full_rotated_reference))
        or full_rotated_reference_xy_norm <= 1e-9
    ):
        raise RuntimeError(
            "full absolute correction loses the native table-tangent reference"
        )
    induced_full_rotation_yaw = float(
        np.arctan2(
            np.dot(
                normal,
                np.cross(local_reference_3d, full_rotated_reference),
            ),
            np.dot(local_reference_3d, full_rotated_reference),
        )
    )
    if not np.isfinite(induced_full_rotation_yaw):
        raise RuntimeError("full absolute correction has no finite projected yaw")
    relative_yaw_spec = {
        **relative_yaw_spec,
        "target_outward_direction_xy": (
            full_rotated_reference[:2]
            / full_rotated_reference_xy_norm
        ).tolist(),
        "yaw_angle_rad": induced_full_rotation_yaw,
        "yaw_angle_deg": float(np.degrees(induced_full_rotation_yaw)),
        "axis_angle_world_rad": expected_remaining_axis_angle.tolist(),
        "rotation_matrix_world": expected_remaining_rotation.tolist(),
        "formula": (
            "full live relative rotation = fixed absolute cumulative target "
            "rotation @ measured cumulative rotation.T; convert that exact "
            "world rotation to its shortest axis-angle"
        ),
        "absolute_cumulative_target_derived": True,
    }
    live_relative_plan = _strict_axis_angle_action_capacity_evidence(
        axis_angle_world=expected_remaining_axis_angle,
        native_action_spec=native_action_spec,
        rotation_spec=rotation_spec,
    )
    if not live_relative_plan["accepted"]:
        raise RuntimeError(
            "live absolute wrist-yaw waypoint requires supplemental strict "
            "segments; this execution is stopped fail-closed instead of clipping: "
            f"{json.dumps(live_relative_plan, sort_keys=True)}"
        )
    return {
        "formula": (
            "measure actual cumulative yaw against the initial native finger "
            "frame, subtract it from this fixed absolute cumulative waypoint, "
            "derive the complete live world axis-angle that closes the fixed "
            "absolute rotation, and require a fresh one-segment strict native "
            "OSC capacity proof"
        ),
        "planned_segment_index": int(planned_segment["segment_index"]),
        "planned_absolute_target_yaw_rad": planned_absolute_yaw,
        "live_actual_absolute_yaw_rad": actual_absolute_yaw,
        "measured_remaining_yaw_to_absolute_target_rad": (
            measured_remaining_yaw
        ),
        "full_rotation_induced_local_yaw_rad": induced_full_rotation_yaw,
        "measured_remaining_full_axis_angle_world_rad": (
            expected_remaining_axis_angle.tolist()
        ),
        "measured_remaining_full_axis_angle_norm_rad": float(
            np.linalg.norm(expected_remaining_axis_angle)
        ),
        "fixed_preregistered_relative_delta_rad": float(
            planned_segment["relative_target_yaw_rad"]
        ),
        "live_reference_outward_direction_xy": (
            live_reference_outward.tolist()
        ),
        "planned_absolute_outward_direction_xy": planned_outward.tolist(),
        "relative_yaw_spec": relative_yaw_spec,
        "live_relative_capacity_plan": live_relative_plan,
        "strictly_inside_live_native_capacity": True,
        "supplemental_segment_count_required": 0,
        "over_capacity_policy": "fail_closed_without_clip_or_fallback",
        "absolute_target_identity_preserved": True,
    }


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
    reference_outward_direction_xy=None,
    selection_mode="native_trailing_wrist_yaw",
):
    """Compile all candidates and select the registered route family."""
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
        geometry_violations = []
        if dual_finger_skew > outside_clearance_m:
            geometry_violations.append(
                "dual_finger_contact_skew_exceeds_outside_clearance"
            )
        selection_violations = list(geometry_violations)
        if not geometry["route_selection_candidate"]:
            selection_violations.append(
                "native_push_derived_candidate_is_diagnostic_only"
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
                "compiled_geometry_violations": geometry_violations,
                "compiled_geometry_eligible": not geometry_violations,
                "selection_violations": selection_violations,
                "selection_eligible": bool(
                    geometry["route_selection_candidate"]
                    and not geometry_violations
                ),
            }
        )
    if reference_outward_direction_xy is None:
        reference_outward = np.array([1.0, 0.0], dtype=float)
        reference_source = "initial_native_low_skew_plus_x"
    else:
        reference_outward = np.asarray(
            reference_outward_direction_xy, dtype=float
        )
        if (
            reference_outward.shape != (2,)
            or not np.all(np.isfinite(reference_outward))
            or np.linalg.norm(reference_outward) <= 1e-9
        ):
            raise ValueError(
                "live wrist-yaw reference outward direction is invalid"
            )
        reference_outward /= np.linalg.norm(reference_outward)
        reference_source = "previous_real_recompiled_wrist_approach"
    reference_contact_xy = (
        plate_position[:2] + reference_outward * float(backoff)
    )
    _, _, reference_compiled_geometry = (
        _compiled_native_side_contact_plan(
            env,
            plate_position,
            eef_position,
            reference_outward,
            reference_contact_xy,
            outside_clearance_m,
        )
    )
    if (
        reference_compiled_geometry["dual_finger_contact_skew_m"]
        > outside_clearance_m
    ):
        raise RuntimeError(
            "live wrist orientation no longer has the registered low-skew "
            "reference approach"
        )
    diagnostic_candidates = [
        candidate for candidate in candidates if candidate["diagnostic_only"]
    ]
    expected_relations = [
        "trailing_minus_push",
        "tangent_counterclockwise",
        "tangent_clockwise",
    ]
    observed_relations = [
        candidate["native_push_direction_relations"]
        for candidate in diagnostic_candidates
    ]
    if observed_relations != [[relation] for relation in expected_relations]:
        raise RuntimeError(
            "native push-frame yaw diagnostics require three unique derived "
            f"directions in preregistered order: observed={observed_relations}"
        )
    table_normal, table_normal_evidence = (
        _compiled_table_normal_evidence(env)
    )
    yaw_specs = _hypothetical_wrist_yaw_specs(
        reference_outward_direction_xy=reference_outward,
        target_outward_directions_xy=[
            candidate["outward_direction_xy"]
            for candidate in diagnostic_candidates
        ],
        table_normal_world=table_normal,
    )
    for relation, candidate, yaw_spec in zip(
        expected_relations,
        diagnostic_candidates,
        yaw_specs,
    ):
        yaw_spec = {
            **yaw_spec,
            "native_push_direction_relation": relation,
        }
        candidate["hypothetical_wrist_yaw"] = (
            _compiled_hypothetical_wrist_yaw_plan(
                env,
                plate_position=plate_position,
                eef_position=eef_position,
                contact_xy=np.asarray(candidate["point_xy"], dtype=float),
                outside_clearance_m=outside_clearance_m,
                plate_approach_eef_height=plate_approach_eef_height,
                position_action_scale=position_action_scale,
                yaw_spec=yaw_spec,
                table_normal_evidence=table_normal_evidence,
            )
        )
        candidate["hypothetical_wrist_yaw"][
            "reference_compiled_geometry"
        ] = reference_compiled_geometry
        candidate["hypothetical_wrist_yaw"][
            "reference_outward_source"
        ] = reference_source
    if selection_mode == "native_trailing_wrist_yaw":
        selected = _select_executable_wrist_yaw_candidate(candidates)
    elif selection_mode == "native_plus_x_front_corridor":
        selected = _select_native_plus_x_front_candidate(candidates)
    else:
        raise ValueError(
            f"unknown compiled contact selection mode: {selection_mode!r}"
        )
    return selected, candidates


def _center_high_target_from_live_plate(
    live_plate_position, plate_approach_eef_height
):
    """Rebuild center-high from the exact live plate pose and fixed geometry."""
    live_plate = np.asarray(live_plate_position, dtype=float)
    if (
        live_plate.shape != (3,)
        or not np.all(np.isfinite(live_plate))
        or not np.isfinite(plate_approach_eef_height)
        or plate_approach_eef_height <= 0.0
    ):
        raise ValueError("live plate center-high inputs are invalid")
    target = live_plate.copy()
    target[2] = live_plate[2] + float(plate_approach_eef_height)
    return target


def _center_high_reacquire_budget_evidence(
    *,
    structural_actions_used,
    maximum_structural_actions,
    reacquire_steps,
    maximum_reacquire_steps,
):
    """Authorize one reacquire action under both unchanged hard limits."""
    values = (
        structural_actions_used,
        maximum_structural_actions,
        reacquire_steps,
        maximum_reacquire_steps,
    )
    if (
        any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in values
        )
        or structural_actions_used < 0
        or maximum_structural_actions < 1
        or reacquire_steps < 0
        or maximum_reacquire_steps < 1
        or reacquire_steps > structural_actions_used
    ):
        raise ValueError("center-high reacquire budgets are invalid")
    violations = []
    if structural_actions_used >= maximum_structural_actions:
        violations.append("shared_structural_waypoint_budget_exhausted")
    if reacquire_steps >= maximum_reacquire_steps:
        violations.append("center_high_reacquire_step_budget_exhausted")
    return {
        "accepted": not violations,
        "violations": violations,
        "structural_actions_used": int(structural_actions_used),
        "maximum_structural_actions": int(maximum_structural_actions),
        "remaining_structural_actions": int(
            maximum_structural_actions - structural_actions_used
        ),
        "reacquire_steps": int(reacquire_steps),
        "maximum_reacquire_steps": int(maximum_reacquire_steps),
        "remaining_reacquire_steps": int(
            maximum_reacquire_steps - reacquire_steps
        ),
    }


def _center_high_reacquire_step_gate(
    *,
    overhead_guard,
    robot_nonrobot_contact_gate,
    action_evidence,
    attainment_evidence,
    consecutive_position_stall_steps,
    maximum_stall_steps,
):
    """Fail closed on every orientation-hold live center-high frame."""
    if (
        not isinstance(consecutive_position_stall_steps, (int, np.integer))
        or isinstance(consecutive_position_stall_steps, (bool, np.bool_))
        or not isinstance(maximum_stall_steps, (int, np.integer))
        or isinstance(maximum_stall_steps, (bool, np.bool_))
        or consecutive_position_stall_steps < 0
        or maximum_stall_steps < 1
    ):
        raise ValueError("center-high reacquire stall counters are invalid")
    violations = []
    if not overhead_guard.get("accepted", False):
        violations.append("center_high_reacquire_overhead_guard_failed")
    if not robot_nonrobot_contact_gate.get("accepted", False):
        violations.append(
            "forbidden_robot_native_contact_during_center_high_reacquire"
        )
    if action_evidence.get("action_will_clip", False):
        violations.append("center_high_reacquire_action_would_clip")
    if not action_evidence.get("translation_direction_valid", False):
        violations.append("center_high_reacquire_direction_invalid")
    if not action_evidence.get("orientation_hold_commanded", False):
        violations.append("center_high_reacquire_changed_orientation")
    if not attainment_evidence.get("rotation_attained", False):
        violations.append("center_high_reacquire_trailing_orientation_drifted")
    if not attainment_evidence.get("rigid_frame_valid", False):
        violations.append("center_high_reacquire_finger_frame_not_rigid")
    if (
        action_evidence.get("position_correction_requested", False)
        and not attainment_evidence.get("position_attained", False)
        and consecutive_position_stall_steps >= maximum_stall_steps
    ):
        violations.append("center_high_reacquire_position_progress_stalled")
    return {
        "accepted": not violations,
        "violations": violations,
        "consecutive_position_stall_steps": int(
            consecutive_position_stall_steps
        ),
        "maximum_stall_steps": int(maximum_stall_steps),
    }


def _second_real_recompile_identity_evidence(
    *,
    first_candidate,
    second_candidate,
    terminal_eef_position,
    live_plate_position,
    center_high_target,
    plate_approach_eef_height,
):
    """Bind the descent geometry to the exact post-reacquire live state."""
    terminal_eef = np.asarray(terminal_eef_position, dtype=float)
    live_plate = np.asarray(live_plate_position, dtype=float)
    center_high = np.asarray(center_high_target, dtype=float)
    if (
        terminal_eef.shape != (3,)
        or live_plate.shape != (3,)
        or center_high.shape != (3,)
        or not np.all(np.isfinite(terminal_eef))
        or not np.all(np.isfinite(live_plate))
        or not np.all(np.isfinite(center_high))
    ):
        raise ValueError("second real recompile identity poses are invalid")
    expected_center_high = _center_high_target_from_live_plate(
        live_plate, plate_approach_eef_height
    )
    first = first_candidate.get("real_sim_geometry_recompile", {})
    second = second_candidate.get("real_sim_geometry_recompile", {})
    violations = []
    if not (
        first.get("performed", False)
        and first.get("eligible", False)
        and first.get("recompile_stage") == "post_wrist_yaw"
    ):
        violations.append("first_live_recompile_identity_invalid")
    if not (
        second.get("performed", False)
        and second.get("eligible", False)
        and second.get("recompile_stage")
        == "post_center_high_reacquire"
    ):
        violations.append("second_live_recompile_identity_invalid")
    exact_vector_checks = {
        "second_recompile_eef_not_terminal_reacquire_eef": (
            second.get("live_eef_position_world"),
            terminal_eef,
        ),
        "second_recompile_plate_not_terminal_live_plate": (
            second.get("live_plate_position_world"),
            live_plate,
        ),
        "reacquire_target_not_live_plate_center_high": (
            center_high,
            expected_center_high,
        ),
        "second_recompile_center_high_not_reacquire_target": (
            second.get("live_center_high_target_world"),
            center_high,
        ),
        "second_candidate_center_high_not_reacquire_target": (
            second_candidate.get("center_high_target"),
            center_high,
        ),
    }
    for violation, (observed, expected) in exact_vector_checks.items():
        try:
            observed_array = np.asarray(observed, dtype=float)
        except (TypeError, ValueError):
            violations.append(violation)
            continue
        if observed_array.shape != (3,) or not np.array_equal(
            observed_array, np.asarray(expected, dtype=float)
        ):
            violations.append(violation)
    if not second.get("strict_dual_finger_skew_accepted", False):
        violations.append("second_recompile_skew_gate_failed")
    outside_guard = second.get("planned_outside_guard", {})
    if not outside_guard.get("accepted", False):
        violations.append("second_recompile_outside_guard_failed")
    selected_clearance = float(
        second.get("selected_finger_table_clearance_m", -np.inf)
    )
    planned_clearance = float(
        second.get("planned_finger_table_clearance_m", -np.inf)
    )
    required_clearance = float(
        second.get("required_finger_table_clearance_m", np.inf)
    )
    if not selected_clearance > required_clearance:
        violations.append("second_recompile_selected_table_clearance_failed")
    if not planned_clearance > required_clearance:
        violations.append("second_recompile_planned_table_clearance_failed")
    rim_coverage = second.get("selected_rim_overlap_by_side", {})
    for side in ("left", "right"):
        coverage = rim_coverage.get(side, {})
        if not (
            float(coverage.get("overlap_m", 0.0)) > 0.0
            and coverage.get("rim_center_covered", False)
        ):
            violations.append(f"second_recompile_{side}_rim_gate_failed")
    if second.get("hypothetical_geometry_used_for_descent", True):
        violations.append("second_recompile_used_hypothetical_geometry")
    return {
        "accepted": not violations,
        "violations": violations,
        "required_recompile_order": [
            "post_wrist_yaw",
            "post_center_high_reacquire",
        ],
        "first_recompile_stage": first.get("recompile_stage"),
        "second_recompile_stage": second.get("recompile_stage"),
        "terminal_reacquire_eef_position_world": terminal_eef.tolist(),
        "terminal_live_plate_position_world": live_plate.tolist(),
        "terminal_center_high_target_world": center_high.tolist(),
        "second_recompile_uses_exact_terminal_eef": bool(
            np.array_equal(
                np.asarray(
                    second.get("live_eef_position_world", ()), dtype=float
                ),
                terminal_eef,
            )
        ),
        "second_recompile_uses_exact_terminal_plate": bool(
            np.array_equal(
                np.asarray(
                    second.get("live_plate_position_world", ()), dtype=float
                ),
                live_plate,
            )
        ),
        "all_second_geometry_gates_revalidated": not any(
            violation.startswith("second_recompile_")
            for violation in violations
        ),
        "hypothetical_geometry_used_for_descent": second.get(
            "hypothetical_geometry_used_for_descent"
        ),
    }


def _execute_center_high_reacquire(
    rollout,
    env,
    args,
    *,
    initial_reference_frame,
    total_yaw_spec,
    table_normal_world,
    maximum_angle_error_rad,
    angular_progress_epsilon_rad,
    overhead_geometry,
    native_action_spec,
    rotation_spec,
    gripper,
    structural_actions_used,
    diagnostics,
):
    """Hold attained trailing yaw while reacquiring the live plate center-high."""
    current_eef = np.asarray(
        rollout.obs["robot0_eef_pos"], dtype=float
    ).copy()
    reacquire_start_eef = current_eef.copy()
    live_plate = body_pose(env, PLATE_BODY)[0].copy()
    center_high_target = _center_high_target_from_live_plate(
        live_plate, args.plate_approach_eef_height
    )
    initial_live_plate = live_plate.copy()
    initial_center_high_target = center_high_target.copy()
    current_frame = _compiled_finger_yaw_frame(
        env, eef_position=current_eef
    )
    attainment = _wrist_yaw_attainment_evidence(
        reference_frame=initial_reference_frame,
        current_frame=current_frame,
        yaw_spec=total_yaw_spec,
        maximum_angle_error_rad=maximum_angle_error_rad,
        maximum_position_drift_m=args.position_tolerance,
        angular_progress_epsilon_rad=angular_progress_epsilon_rad,
        position_progress_epsilon_m=args.minimum_saturated_waypoint_progress,
        anchor_eef_position=center_high_target,
    )
    if not attainment["rotation_attained"]:
        raise RuntimeError(
            "center-high reacquire may begin only with the attained trailing "
            "wrist orientation"
        )
    initial_error = float(attainment["eef_position_drift_m"])
    initial_xy_error = float(
        np.linalg.norm(current_eef[:2] - center_high_target[:2])
    )
    maximum_center_high_error = initial_error
    maximum_eef_drift_from_start = 0.0
    maximum_orientation_error = float(
        attainment["target_rotation_error_rad"]
    )
    frames = []
    consecutive_position_stall_steps = 0
    while not attainment["attained"]:
        budget_evidence = _center_high_reacquire_budget_evidence(
            structural_actions_used=structural_actions_used + len(frames),
            maximum_structural_actions=args.max_waypoint_steps,
            reacquire_steps=len(frames),
            maximum_reacquire_steps=args.push_tracking_steps,
        )
        if not budget_evidence["accepted"]:
            raise RuntimeError(
                "center-high reacquire budget failed closed: "
                f"{json.dumps(budget_evidence, sort_keys=True)}"
            )
        action, action_evidence = _compiled_wrist_yaw_action(
            remaining_yaw_rad=0.0,
            remaining_axis_angle_world=np.zeros(3, dtype=float),
            table_normal_world=table_normal_world,
            current_eef_position=current_eef,
            anchor_eef_position=center_high_target,
            position_action_scale=args.position_action_scale,
            maximum_translation_action=(
                args.plate_contact_seek_max_translation_action
            ),
            gripper=gripper,
            native_action_spec=native_action_spec,
            rotation_spec=rotation_spec,
        )
        pre_gate = _center_high_reacquire_step_gate(
            overhead_guard=_live_compiled_overhead_guard(
                env, overhead_geometry
            ),
            robot_nonrobot_contact_gate=(
                _robot_nonrobot_contact_evidence(
                    env, allowed_body_pairs=()
                )
            ),
            action_evidence=action_evidence,
            attainment_evidence=attainment,
            consecutive_position_stall_steps=(
                consecutive_position_stall_steps
            ),
            maximum_stall_steps=args.push_tracking_steps,
        )
        if not pre_gate["accepted"]:
            raise RuntimeError(
                "center-high reacquire pre-action gate failed closed: "
                f"{json.dumps(pre_gate, sort_keys=True)}"
            )
        previous_position_error = float(
            attainment["eef_position_drift_m"]
        )
        commanded_target = center_high_target.copy()
        commanded_plate = live_plate.copy()
        rollout.advance(action, "task_center_high_reacquire")
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        ).copy()
        live_plate = body_pose(env, PLATE_BODY)[0].copy()
        center_high_target = _center_high_target_from_live_plate(
            live_plate, args.plate_approach_eef_height
        )
        current_frame = _compiled_finger_yaw_frame(
            env, eef_position=current_eef
        )
        attainment = _wrist_yaw_attainment_evidence(
            reference_frame=initial_reference_frame,
            current_frame=current_frame,
            yaw_spec=total_yaw_spec,
            maximum_angle_error_rad=maximum_angle_error_rad,
            maximum_position_drift_m=args.position_tolerance,
            angular_progress_epsilon_rad=angular_progress_epsilon_rad,
            position_progress_epsilon_m=(
                args.minimum_saturated_waypoint_progress
            ),
            anchor_eef_position=center_high_target,
            previous_position_drift_m=previous_position_error,
        )
        if attainment["position_attained"]:
            consecutive_position_stall_steps = 0
        else:
            consecutive_position_stall_steps = (
                0
                if attainment["position_progressed"]
                else consecutive_position_stall_steps + 1
            )
        maximum_center_high_error = max(
            maximum_center_high_error,
            float(attainment["eef_position_drift_m"]),
        )
        maximum_eef_drift_from_start = max(
            maximum_eef_drift_from_start,
            float(np.linalg.norm(current_eef - reacquire_start_eef)),
        )
        maximum_orientation_error = max(
            maximum_orientation_error,
            float(attainment["target_rotation_error_rad"]),
        )
        post_overhead_guard = _live_compiled_overhead_guard(
            env, overhead_geometry
        )
        post_contact_gate = _robot_nonrobot_contact_evidence(
            env, allowed_body_pairs=()
        )
        post_gate = _center_high_reacquire_step_gate(
            overhead_guard=post_overhead_guard,
            robot_nonrobot_contact_gate=post_contact_gate,
            action_evidence=action_evidence,
            attainment_evidence=attainment,
            consecutive_position_stall_steps=(
                consecutive_position_stall_steps
            ),
            maximum_stall_steps=args.push_tracking_steps,
        )
        frame = {
            "reacquire_step": len(frames) + 1,
            "shared_structural_action_index": int(
                structural_actions_used + len(frames) + 1
            ),
            "budget_before_action": budget_evidence,
            "commanded_live_plate_position_world": commanded_plate.tolist(),
            "commanded_center_high_target_world": commanded_target.tolist(),
            "observed_live_plate_position_world": live_plate.tolist(),
            "observed_center_high_target_world": center_high_target.tolist(),
            "eef_position_world": current_eef.tolist(),
            "action": action.tolist(),
            "action_evidence": action_evidence,
            "attainment_evidence": attainment,
            "pre_gate": pre_gate,
            "post_overhead_guard": post_overhead_guard,
            "post_robot_nonrobot_contact_gate": post_contact_gate,
            "post_gate": post_gate,
        }
        frames.append(frame)
        print(
            "L3-A3 center-high reacquire frame "
            + json.dumps(
                {
                    "reacquire_step": len(frames),
                    "shared_structural_action_index": frame[
                        "shared_structural_action_index"
                    ],
                    "center_high_error_m": attainment[
                        "eef_position_drift_m"
                    ],
                    "center_high_xy_error_m": float(
                        np.linalg.norm(
                            current_eef[:2] - center_high_target[:2]
                        )
                    ),
                    "target_rotation_error_rad": attainment[
                        "target_rotation_error_rad"
                    ],
                    "translation_action_peak": action_evidence[
                        "commanded_translation_action_peak"
                    ],
                    "translation_action_norm": action_evidence[
                        "commanded_translation_action_norm"
                    ],
                    "translation_bound_saturated": action_evidence[
                        "translation_bound_saturated"
                    ],
                    "orientation_hold_commanded": action_evidence[
                        "orientation_hold_commanded"
                    ],
                    "action_will_clip": action_evidence[
                        "action_will_clip"
                    ],
                    "position_progressed": attainment[
                        "position_progressed"
                    ],
                    "position_stall_steps": (
                        consecutive_position_stall_steps
                    ),
                    "overhead_accepted": post_overhead_guard["accepted"],
                    "contact_gate_accepted": post_contact_gate["accepted"],
                    "unexpected_contact_count": len(
                        post_contact_gate["unexpected_contacts"]
                    ),
                    "post_gate_accepted": post_gate["accepted"],
                    "post_gate_violations": post_gate["violations"],
                    "shared_budget_remaining_after_action": int(
                        args.max_waypoint_steps
                        - structural_actions_used
                        - len(frames)
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not post_gate["accepted"]:
            raise RuntimeError(
                "center-high reacquire post-action gate failed closed: "
                f"frame={json.dumps(frame, sort_keys=True)} "
                f"scene={json.dumps(diagnostics(), sort_keys=True)}"
            )
    final_error = float(attainment["eef_position_drift_m"])
    final_xy_error = float(
        np.linalg.norm(current_eef[:2] - center_high_target[:2])
    )
    remaining_structural_actions = int(
        args.max_waypoint_steps - structural_actions_used - len(frames)
    )
    if (
        not final_error < float(args.position_tolerance)
        or not final_xy_error < float(args.position_tolerance)
        or not attainment["rotation_attained"]
        or not attainment["rigid_frame_valid"]
        or remaining_structural_actions < 1
    ):
        raise RuntimeError(
            "center-high reacquire did not leave a strict verified pose and "
            "downstream structural budget"
        )
    evidence = {
        "performed": bool(frames),
        "stage": "orientation_hold_center_high_reacquire",
        "target_definition": (
            "exact current live plate position with Z replaced by live plate "
            "Z plus the unchanged plate_approach_eef_height"
        ),
        "uses_old_wrist_yaw_anchor": False,
        "orientation_hold_required": True,
        "initial_live_plate_position_world": initial_live_plate.tolist(),
        "initial_center_high_target_world": (
            initial_center_high_target.tolist()
        ),
        "initial_eef_position_world": reacquire_start_eef.tolist(),
        "initial_center_high_error_m": initial_error,
        "initial_center_high_xy_error_m": initial_xy_error,
        "terminal_live_plate_position_world": live_plate.tolist(),
        "terminal_center_high_target_world": center_high_target.tolist(),
        "terminal_eef_position_world": current_eef.tolist(),
        "final_center_high_error_m": final_error,
        "final_center_high_xy_error_m": final_xy_error,
        "maximum_center_high_error_m": maximum_center_high_error,
        "maximum_eef_drift_from_reacquire_start_m": (
            maximum_eef_drift_from_start
        ),
        "maximum_trailing_rotation_error_rad": maximum_orientation_error,
        "maximum_center_high_error_tolerance_m": float(
            args.position_tolerance
        ),
        "strict_final_center_high_attainment": True,
        "steps": len(frames),
        "maximum_steps": int(args.push_tracking_steps),
        "shared_budget_used_before_reacquire": int(
            structural_actions_used
        ),
        "shared_budget_used_after_reacquire": int(
            structural_actions_used + len(frames)
        ),
        "shared_budget_remaining_after_reacquire": (
            remaining_structural_actions
        ),
        "position_action_scale_m_per_action": float(
            args.position_action_scale
        ),
        "strict_translation_action_norm_bound": float(
            np.nextafter(
                args.plate_contact_seek_max_translation_action, 0.0
            )
        ),
        "frames": frames,
        "final_attainment_evidence": attainment,
    }
    return (
        evidence,
        current_eef,
        current_frame,
        live_plate,
        center_high_target,
    )


def _execute_high_safe_wrist_yaw(
    rollout,
    env,
    args,
    *,
    selected_candidate,
    center_high_target,
    gripper,
    diagnostics,
):
    """Execute yaw, reacquire live center-high, and recompile twice."""
    if selected_candidate.get("native_push_direction_relations") != [
        "trailing_minus_push"
    ] or not selected_candidate.get("wrist_yaw_route_selected", False):
        raise RuntimeError(
            "high-safe wrist yaw cannot silently execute an unselected direction"
        )
    current_eef = np.asarray(
        rollout.obs["robot0_eef_pos"], dtype=float
    )
    center_high_target = np.asarray(center_high_target, dtype=float)
    center_error = float(np.linalg.norm(current_eef - center_high_target))
    if center_error > args.position_tolerance:
        raise RuntimeError(
            "wrist yaw may execute only after reaching verified center-high "
            f"free space: error={center_error}"
        )
    native_action_spec = _native_osc_action_spec_evidence(env)
    rotation_spec = _native_osc_rotation_spec_evidence(
        env, native_action_spec
    )
    table_normal, table_normal_evidence = (
        _compiled_table_normal_evidence(env)
    )
    total_yaw_spec = selected_candidate["hypothetical_wrist_yaw"]["yaw"]
    if not np.allclose(
        total_yaw_spec["table_normal_world"],
        table_normal,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "selected yaw axis diverged from the live native table normal"
        )
    yaw_segmentation = _strict_wrist_yaw_segment_plan(
        yaw_spec=total_yaw_spec,
        native_action_spec=native_action_spec,
        rotation_spec=rotation_spec,
    )
    maximum_controller_world_step = float(
        args.position_action_scale
        * args.plate_contact_seek_max_translation_action
    )
    _, overhead_geometry = _compiled_overhead_staging_geometry(
        env,
        start_eef_position=current_eef,
        one_step_vertical_reserve_m=maximum_controller_world_step,
    )
    initial_overhead_guard = _live_compiled_overhead_guard(
        env, overhead_geometry
    )
    initial_contact_gate = _robot_nonrobot_contact_evidence(
        env, allowed_body_pairs=()
    )
    if (
        not initial_overhead_guard["accepted"]
        or not initial_contact_gate["accepted"]
    ):
        raise RuntimeError(
            "center-high wrist-yaw free-space gate failed before rotation: "
            f"overhead={json.dumps(initial_overhead_guard, sort_keys=True)} "
            f"contacts={json.dumps(initial_contact_gate, sort_keys=True)}"
        )
    initial_reference_frame = _compiled_finger_yaw_frame(
        env, eef_position=current_eef
    )
    anchor_eef = current_eef.copy()
    hypothetical = selected_candidate["hypothetical_wrist_yaw"]
    skew_margin = float(
        hypothetical["maximum_dual_finger_contact_skew_m"]
        - hypothetical["dual_finger_contact_skew_m"]
    )
    maximum_radius = float(
        initial_reference_frame["maximum_finger_radius_from_eef_m"]
    )
    target_yaw_magnitude = abs(float(total_yaw_spec["yaw_angle_rad"]))
    maximum_angle_error = float(
        np.nextafter(
            min(
                skew_margin / maximum_radius,
                target_yaw_magnitude / 4.0,
            ),
            0.0,
        )
    )
    angular_progress_epsilon = float(
        args.minimum_saturated_waypoint_progress / maximum_radius
    )
    if (
        skew_margin <= 0.0
        or maximum_angle_error <= 0.0
        or angular_progress_epsilon <= 0.0
        or maximum_angle_error <= angular_progress_epsilon
    ):
        raise RuntimeError(
            "selected yaw lacks a positive geometry-derived attainment margin"
        )
    tolerance_derivation = {
        "unchanged_dual_finger_skew_limit_m": float(
            hypothetical["maximum_dual_finger_contact_skew_m"]
        ),
        "hypothetical_dual_finger_skew_m": float(
            hypothetical["dual_finger_contact_skew_m"]
        ),
        "strict_skew_margin_m": skew_margin,
        "maximum_finger_radius_from_eef_m": maximum_radius,
        "maximum_angle_error_rad": maximum_angle_error,
        "angular_progress_epsilon_rad": angular_progress_epsilon,
        "angular_progress_epsilon_source": (
            "existing minimum_saturated_waypoint_progress divided by the "
            "live maximum finger radius"
        ),
        "position_drift_limit_m": float(args.position_tolerance),
        "position_progress_epsilon_m": float(
            args.minimum_saturated_waypoint_progress
        ),
        "anchor_compensation_translation_action_norm_bound": float(
            args.plate_contact_seek_max_translation_action
        ),
        "anchor_compensation_maximum_world_step_m": (
            maximum_controller_world_step
        ),
        "maximum_position_settle_steps": int(args.push_tracking_steps),
    }

    def derive_live_absolute_segment(planned_segment, live_frame):
        absolute_start_attainment = _wrist_yaw_attainment_evidence(
            reference_frame=initial_reference_frame,
            current_frame=live_frame,
            yaw_spec=planned_segment["cumulative_yaw_spec"],
            maximum_angle_error_rad=maximum_angle_error,
            maximum_position_drift_m=args.position_tolerance,
            angular_progress_epsilon_rad=angular_progress_epsilon,
            position_progress_epsilon_m=(
                args.minimum_saturated_waypoint_progress
            ),
            anchor_eef_position=anchor_eef,
        )
        runtime_target = _absolute_wrist_yaw_runtime_target(
            planned_segment=planned_segment,
            absolute_start_attainment=absolute_start_attainment,
            total_yaw_spec=total_yaw_spec,
            native_action_spec=native_action_spec,
            rotation_spec=rotation_spec,
        )
        live_segment_reference = live_frame
        live_segment_spec = runtime_target["relative_yaw_spec"]
        live_segment_attainment = _wrist_yaw_attainment_evidence(
            reference_frame=live_segment_reference,
            current_frame=live_frame,
            yaw_spec=live_segment_spec,
            maximum_angle_error_rad=maximum_angle_error,
            maximum_position_drift_m=args.position_tolerance,
            angular_progress_epsilon_rad=angular_progress_epsilon,
            position_progress_epsilon_m=(
                args.minimum_saturated_waypoint_progress
            ),
            anchor_eef_position=anchor_eef,
        )
        return (
            live_segment_reference,
            live_segment_spec,
            live_segment_attainment,
            absolute_start_attainment,
            runtime_target,
        )

    current_frame = initial_reference_frame
    yaw_segment_index = 0
    active_segment = yaw_segmentation["segments"][yaw_segment_index]
    (
        segment_reference_frame,
        segment_yaw_spec,
        attainment,
        absolute_segment_start_attainment,
        active_runtime_target,
    ) = derive_live_absolute_segment(
        active_segment,
        current_frame,
    )
    frames = []
    yaw_segments = []
    segment_frame_start = 0
    consecutive_angular_stall_steps = 0
    consecutive_position_stall_steps = 0
    segment_position_settle_steps = 0
    total_position_settle_steps = 0
    maximum_observed_position_drift = float(
        attainment["eef_position_drift_m"]
    )
    maximum_commanded_translation_action_peak = 0.0
    for yaw_step in range(1, args.max_waypoint_steps + 1):
        if attainment["attained"]:
            cumulative_attainment = _wrist_yaw_attainment_evidence(
                reference_frame=initial_reference_frame,
                current_frame=current_frame,
                yaw_spec=active_segment["cumulative_yaw_spec"],
                maximum_angle_error_rad=maximum_angle_error,
                maximum_position_drift_m=args.position_tolerance,
                angular_progress_epsilon_rad=angular_progress_epsilon,
                position_progress_epsilon_m=(
                    args.minimum_saturated_waypoint_progress
                ),
                anchor_eef_position=anchor_eef,
            )
            if not cumulative_attainment["attained"]:
                raise RuntimeError(
                    "wrist-yaw segment accumulated outside its absolute native-"
                    "frame target: "
                    f"segment={active_segment['segment_index']} "
                    f"evidence={json.dumps(cumulative_attainment, sort_keys=True)}"
                )
            segment_frames = frames[segment_frame_start:]
            yaw_segments.append(
                {
                    **active_segment,
                    "action_steps": len(segment_frames),
                    "global_action_step_start": (
                        None
                        if not segment_frames
                        else segment_frames[0]["yaw_step"]
                    ),
                    "global_action_step_end": (
                        None
                        if not segment_frames
                        else segment_frames[-1]["yaw_step"]
                    ),
                    "position_settle_steps": int(
                        segment_position_settle_steps
                    ),
                    "absolute_segment_start_attainment": (
                        absolute_segment_start_attainment
                    ),
                    "live_absolute_runtime_target": active_runtime_target,
                    "executed_relative_yaw_spec": segment_yaw_spec,
                    "executed_relative_target_yaw_rad": float(
                        segment_yaw_spec["yaw_angle_rad"]
                    ),
                    "absolute_projected_remaining_yaw_at_start_rad": float(
                        active_runtime_target[
                            "measured_remaining_yaw_to_absolute_target_rad"
                        ]
                    ),
                    "absolute_cumulative_target_is_authoritative": True,
                    "local_attainment_evidence": attainment,
                    "cumulative_attainment_evidence": cumulative_attainment,
                    "shared_budget_used_after_segment": len(frames),
                    "shared_budget_remaining_after_segment": int(
                        args.max_waypoint_steps - len(frames)
                    ),
                }
            )
            yaw_segment_index += 1
            if yaw_segment_index == len(yaw_segmentation["segments"]):
                break
            active_segment = yaw_segmentation["segments"][yaw_segment_index]
            (
                segment_reference_frame,
                segment_yaw_spec,
                attainment,
                absolute_segment_start_attainment,
                active_runtime_target,
            ) = derive_live_absolute_segment(
                active_segment,
                current_frame,
            )
            segment_frame_start = len(frames)
            consecutive_angular_stall_steps = 0
            consecutive_position_stall_steps = 0
            segment_position_settle_steps = 0
        budget_evidence = _wrist_yaw_stage_budget_evidence(
            actions_used=len(frames),
            maximum_actions=args.max_waypoint_steps,
            position_settle_steps=segment_position_settle_steps,
            maximum_position_settle_steps=args.push_tracking_steps,
            rotation_attained=attainment["rotation_attained"],
            position_attained=attainment["position_attained"],
        )
        if not budget_evidence["accepted"]:
            raise RuntimeError(
                "wrist-yaw stage budget failed closed: "
                f"{json.dumps(budget_evidence, sort_keys=True)}"
            )
        stage = budget_evidence["next_stage"]
        commanded_remaining_yaw = (
            0.0
            if stage == "position_settle"
            else attainment["remaining_yaw_rad"]
        )
        commanded_remaining_axis_angle = (
            np.zeros(3, dtype=float)
            if stage == "position_settle"
            else np.asarray(
                attainment[
                    "remaining_rotation_axis_angle_world_rad"
                ],
                dtype=float,
            )
        )
        action, action_evidence = _compiled_wrist_yaw_action(
            remaining_yaw_rad=commanded_remaining_yaw,
            remaining_axis_angle_world=commanded_remaining_axis_angle,
            table_normal_world=table_normal,
            current_eef_position=current_eef,
            anchor_eef_position=anchor_eef,
            position_action_scale=args.position_action_scale,
            maximum_translation_action=(
                args.plate_contact_seek_max_translation_action
            ),
            gripper=gripper,
            native_action_spec=native_action_spec,
            rotation_spec=rotation_spec,
        )
        pre_gate = _wrist_yaw_step_gate(
            stage=stage,
            overhead_guard=_live_compiled_overhead_guard(
                env, overhead_geometry
            ),
            robot_nonrobot_contact_gate=(
                _robot_nonrobot_contact_evidence(
                    env, allowed_body_pairs=()
                )
            ),
            action_evidence=action_evidence,
            attainment_evidence=attainment,
            consecutive_angular_stall_steps=(
                consecutive_angular_stall_steps
            ),
            consecutive_position_stall_steps=(
                consecutive_position_stall_steps
            ),
            maximum_stall_steps=args.push_tracking_steps,
        )
        if not pre_gate["accepted"]:
            raise RuntimeError(
                "wrist-yaw pre-action gate failed closed: "
                f"{json.dumps(pre_gate, sort_keys=True)}"
            )
        previous_absolute_error = float(attainment["absolute_error_rad"])
        previous_rotation_error = float(
            attainment["target_rotation_error_rad"]
        )
        previous_position_drift = float(
            attainment["eef_position_drift_m"]
        )
        settle_step_index = None
        if stage == "position_settle":
            segment_position_settle_steps += 1
            total_position_settle_steps += 1
            settle_step_index = int(segment_position_settle_steps)
        rollout.advance(action, "task_wrist_yaw")
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        )
        current_frame = _compiled_finger_yaw_frame(
            env, eef_position=current_eef
        )
        attainment = _wrist_yaw_attainment_evidence(
            reference_frame=segment_reference_frame,
            current_frame=current_frame,
            yaw_spec=segment_yaw_spec,
            maximum_angle_error_rad=maximum_angle_error,
            maximum_position_drift_m=args.position_tolerance,
            angular_progress_epsilon_rad=angular_progress_epsilon,
            position_progress_epsilon_m=(
                args.minimum_saturated_waypoint_progress
            ),
            previous_absolute_error_rad=previous_absolute_error,
            previous_rotation_error_rad=previous_rotation_error,
            previous_position_drift_m=previous_position_drift,
            anchor_eef_position=anchor_eef,
        )
        consecutive_angular_stall_steps = (
            0
            if attainment["angular_progressed"]
            or attainment["rotation_attained"]
            else consecutive_angular_stall_steps + 1
        )
        if attainment["position_attained"] or not action_evidence[
            "position_correction_requested"
        ]:
            consecutive_position_stall_steps = 0
        else:
            consecutive_position_stall_steps = (
                0
                if attainment["position_progressed"]
                else consecutive_position_stall_steps + 1
            )
        maximum_observed_position_drift = max(
            maximum_observed_position_drift,
            float(attainment["eef_position_drift_m"]),
        )
        maximum_commanded_translation_action_peak = max(
            maximum_commanded_translation_action_peak,
            float(
                action_evidence[
                    "commanded_translation_action_peak"
                ]
            ),
        )
        post_overhead_guard = _live_compiled_overhead_guard(
            env, overhead_geometry
        )
        post_contact_gate = _robot_nonrobot_contact_evidence(
            env, allowed_body_pairs=()
        )
        post_gate = _wrist_yaw_step_gate(
            stage=stage,
            overhead_guard=post_overhead_guard,
            robot_nonrobot_contact_gate=post_contact_gate,
            action_evidence=action_evidence,
            attainment_evidence=attainment,
            consecutive_angular_stall_steps=(
                consecutive_angular_stall_steps
            ),
            consecutive_position_stall_steps=(
                consecutive_position_stall_steps
            ),
            maximum_stall_steps=args.push_tracking_steps,
        )
        frame = {
            "yaw_step": int(yaw_step),
            "yaw_segment_index": int(active_segment["segment_index"]),
            "yaw_segment_step": int(len(frames) - segment_frame_start + 1),
            "segment_absolute_target_yaw_rad": float(
                active_segment["absolute_target_yaw_rad"]
            ),
            "segment_relative_target_yaw_rad": float(
                segment_yaw_spec["yaw_angle_rad"]
            ),
            "absolute_projected_remaining_yaw_rad": float(
                active_runtime_target[
                    "measured_remaining_yaw_to_absolute_target_rad"
                ]
            ),
            "fixed_preregistered_relative_delta_rad": float(
                active_segment["relative_target_yaw_rad"]
            ),
            "stage": stage,
            "position_settle_step": settle_step_index,
            "budget_before_action": budget_evidence,
            "action": action.tolist(),
            "action_evidence": action_evidence,
            "position_error_before_action_m": (
                previous_position_drift
            ),
            "position_error_after_action_m": float(
                attainment["eef_position_drift_m"]
            ),
            "attainment_evidence": attainment,
            "pre_gate": pre_gate,
            "post_gate": post_gate,
            "post_overhead_guard": post_overhead_guard,
            "post_robot_nonrobot_contact_gate": post_contact_gate,
        }
        frames.append(frame)
        frame_log = {
            "yaw_step": int(yaw_step),
            "yaw_segment_index": int(active_segment["segment_index"]),
            "yaw_segment_step": int(len(frames) - segment_frame_start),
            "segment_absolute_target_yaw_rad": active_segment[
                "absolute_target_yaw_rad"
            ],
            "segment_relative_target_yaw_rad": active_runtime_target[
                "full_rotation_induced_local_yaw_rad"
            ],
            "fixed_preregistered_relative_delta_rad": active_segment[
                "relative_target_yaw_rad"
            ],
            "live_remaining_yaw_to_absolute_target_rad": (
                active_runtime_target[
                    "measured_remaining_yaw_to_absolute_target_rad"
                ]
            ),
            "stage": stage,
            "position_settle_step": settle_step_index,
            "shared_budget_remaining_after_action": int(
                args.max_waypoint_steps - len(frames)
            ),
            "position_error_before_action_m": previous_position_drift,
            "position_error_after_action_m": float(
                attainment["eef_position_drift_m"]
            ),
            "position_progress_m": attainment["position_progress_m"],
            "position_progressed": attainment["position_progressed"],
            "commanded_translation_action_peak": action_evidence[
                "commanded_translation_action_peak"
            ],
            "commanded_translation_action_norm": action_evidence[
                "commanded_translation_action_norm"
            ],
            "translation_bound_saturated": action_evidence[
                "translation_bound_saturated"
            ],
            "translation_direction_valid": action_evidence[
                "translation_direction_valid"
            ],
            "action_will_clip": action_evidence["action_will_clip"],
            "absolute_yaw_error_rad": attainment["absolute_error_rad"],
            "remaining_rotation_axis_angle_norm_rad": attainment[
                "remaining_rotation_axis_angle_norm_rad"
            ],
            "angular_progress_rad": attainment["angular_progress_rad"],
            "rotation_attained": attainment["rotation_attained"],
            "position_attained": attainment["position_attained"],
            "simultaneous_attained": attainment["attained"],
            "angular_stall_steps": consecutive_angular_stall_steps,
            "position_stall_steps": consecutive_position_stall_steps,
            "overhead_accepted": post_overhead_guard["accepted"],
            "overhead_minimum_reserve_surplus_m": post_overhead_guard[
                "minimum_reserve_surplus_m"
            ],
            "contact_gate_accepted": post_contact_gate["accepted"],
            "unexpected_contact_count": len(
                post_contact_gate["unexpected_contacts"]
            ),
            "post_gate_accepted": post_gate["accepted"],
            "post_gate_violations": post_gate["violations"],
        }
        print(
            "L3-A3 wrist-yaw frame "
            + json.dumps(frame_log, sort_keys=True),
            flush=True,
        )
        if not post_gate["accepted"]:
            raise RuntimeError(
                "wrist-yaw post-action gate failed closed: "
                f"frame={json.dumps(frame, sort_keys=True)} "
                f"scene={json.dumps(diagnostics(), sort_keys=True)}"
            )
    else:
        raise RuntimeError(
            "wrist yaw exhausted the configured finite waypoint budget "
            f"without pose attainment: frames={json.dumps(frames, sort_keys=True)}"
        )
    yaw_steps = len(frames)
    if (
        len(yaw_segments) != yaw_segmentation["minimum_segment_count"]
        or yaw_steps >= args.max_waypoint_steps
        or not yaw_segments[-1]["cumulative_attainment_evidence"]["attained"]
    ):
        raise RuntimeError(
            "segmented wrist yaw left no verified final pose or structural "
            "waypoint budget"
        )
    final_attainment = yaw_segments[-1]["cumulative_attainment_evidence"]
    if not np.isclose(
        final_attainment["target_yaw_rad"],
        float(total_yaw_spec["yaw_angle_rad"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("segmented wrist yaw did not retain its native target")
    live_plate = body_pose(env, PLATE_BODY)[0].copy()
    outward = np.asarray(
        selected_candidate["outward_direction_xy"], dtype=float
    )
    live_selected_candidate = {
        **selected_candidate,
        "point_xy": (
            live_plate[:2]
            + outward * float(args.plate_contact_backoff)
        ).tolist(),
    }
    pre_reacquire_realized_candidate = _real_recompile_wrist_yaw_candidate(
        env,
        selected_candidate=live_selected_candidate,
        plate_position=live_plate,
        eef_position=current_eef,
        outside_clearance_m=args.plate_contact_outside_clearance,
        plate_approach_eef_height=args.plate_approach_eef_height,
        position_action_scale=args.position_action_scale,
        attainment_evidence=final_attainment,
        table_normal_evidence=table_normal_evidence,
        recompile_stage="post_wrist_yaw",
    )
    (
        center_high_reacquire,
        current_eef,
        current_frame,
        live_plate,
        live_center_high_target,
    ) = _execute_center_high_reacquire(
        rollout,
        env,
        args,
        initial_reference_frame=initial_reference_frame,
        total_yaw_spec=total_yaw_spec,
        table_normal_world=table_normal,
        maximum_angle_error_rad=maximum_angle_error,
        angular_progress_epsilon_rad=angular_progress_epsilon,
        overhead_geometry=overhead_geometry,
        native_action_spec=native_action_spec,
        rotation_spec=rotation_spec,
        gripper=gripper,
        structural_actions_used=yaw_steps,
        diagnostics=diagnostics,
    )
    post_reacquire_selected_candidate = {
        **selected_candidate,
        "point_xy": (
            live_plate[:2]
            + outward * float(args.plate_contact_backoff)
        ).tolist(),
    }
    realized_candidate = _real_recompile_wrist_yaw_candidate(
        env,
        selected_candidate=post_reacquire_selected_candidate,
        plate_position=live_plate,
        eef_position=current_eef,
        outside_clearance_m=args.plate_contact_outside_clearance,
        plate_approach_eef_height=args.plate_approach_eef_height,
        position_action_scale=args.position_action_scale,
        attainment_evidence=center_high_reacquire[
            "final_attainment_evidence"
        ],
        table_normal_evidence=table_normal_evidence,
        recompile_stage="post_center_high_reacquire",
    )
    second_recompile_identity = _second_real_recompile_identity_evidence(
        first_candidate=pre_reacquire_realized_candidate,
        second_candidate=realized_candidate,
        terminal_eef_position=current_eef,
        live_plate_position=live_plate,
        center_high_target=live_center_high_target,
        plate_approach_eef_height=args.plate_approach_eef_height,
    )
    if not second_recompile_identity["accepted"]:
        raise RuntimeError(
            "post-reacquire real geometry identity failed closed: "
            f"{json.dumps(second_recompile_identity, sort_keys=True)}"
        )
    live_collision_inventory = _live_collision_inventory(
        env, eef_position=current_eef
    )
    live_cabinet_pose = _live_cabinet_pose_diagnostic(env)
    diagnostic_only_detour_candidates = (
        _diagnostic_only_live_detour_candidates(
            live_inventory=live_collision_inventory,
            cabinet_pose=live_cabinet_pose,
            current_eef=current_eef,
            outside_high_target=realized_candidate["outside_high_target"],
            outside_side_target=realized_candidate["outside_side_target"],
        )
    )
    native_low = np.asarray(native_action_spec["low"], dtype=float)
    native_high = np.asarray(native_action_spec["high"], dtype=float)
    strict_native_detour_translation_action = float(
        np.nextafter(
            min(
                -native_low[0],
                native_high[0],
                -native_low[1],
                native_high[1],
                -native_low[2],
                native_high[2],
            ),
            0.0,
        )
    )
    if strict_native_detour_translation_action <= 0.0:
        raise RuntimeError(
            "native cabinet detour has no strict translation-action capacity"
        )
    cabinet_detour_plan = _compiled_native_right_high_then_low_return_plan(
        live_inventory=live_collision_inventory,
        current_eef=current_eef,
        outside_high_target=realized_candidate["outside_high_target"],
        outside_side_target=realized_candidate["outside_side_target"],
        maximum_controller_world_step_m=float(
            args.position_action_scale
            * args.plate_contact_seek_max_translation_action
        ),
        maximum_route_translation_action=(
            strict_native_detour_translation_action
        ),
        position_action_scale_m_per_action=args.position_action_scale,
        position_tolerance_m=args.position_tolerance,
    )
    remaining_structural_steps = int(
        args.max_waypoint_steps - yaw_steps - center_high_reacquire["steps"]
    )
    if cabinet_detour_plan[
        "minimum_full_step_action_lower_bound"
    ] > remaining_structural_steps:
        raise RuntimeError(
            "compiled native cabinet detour lower bound exceeds the remaining "
            "unchanged structural waypoint budget"
        )
    diagnostic_manifest_value = getattr(args, "diagnostic_manifest", None)
    if diagnostic_manifest_value is None:
        output_value = getattr(args, "output", None)
        if not output_value:
            raise RuntimeError(
                "controller diagnostic manifest requires an output path"
            )
        diagnostic_manifest_value = str(
            Path(output_value).with_suffix(".controller_diagnostic.json")
        )
    diagnostic_manifest = _write_controller_diagnostic_manifest(
        diagnostic_manifest_value,
        {
            "schema_version": 1,
            "scenario": SCENE_ID,
            "task_description": TASK_PROMPT,
            "diagnostic_only": False,
            "route_authorized": True,
            "capture_stage": (
                "post_center_high_reacquire_second_recompile_pre_seek"
            ),
            "second_real_sim_recompile_identity": second_recompile_identity,
            "live_collision_inventory": live_collision_inventory,
            "live_cabinet_pose_and_qpos": live_cabinet_pose,
            "detour_candidates": diagnostic_only_detour_candidates,
            "authorized_detour_plan": cabinet_detour_plan,
            "unexpected_contact_events": [],
            "latest_status": "NATIVE_RIGHT_HIGH_THEN_LOW_RETURN_AUTHORIZED",
        },
    )
    controller_live_collision_diagnostic = {
        "diagnostic_only": False,
        "executed": False,
        "selection_eligible": True,
        "selected": True,
        "route_authorized": True,
        "capture_stage": live_collision_inventory["capture_stage"],
        "inventory_sha256": live_collision_inventory["inventory_sha256"],
        "robot_collision_geom_count": live_collision_inventory[
            "robot_collision_geom_count"
        ],
        "native_nonrobot_collision_geom_count": live_collision_inventory[
            "native_nonrobot_collision_geom_count"
        ],
        "total_collision_geom_count": live_collision_inventory[
            "total_collision_geom_count"
        ],
        "cabinet_root_body": L3A3_CABINET_ROOT_BODY,
        "cabinet_top_body": L3A3_CABINET_TOP_BODY,
        "cabinet_top_joint": L3A3_CABINET_TOP_JOINT,
        "candidate_ids": [
            candidate["candidate_id"]
            for candidate in diagnostic_only_detour_candidates
        ],
        "authorized_detour_plan": cabinet_detour_plan,
        "manifest_path": str(diagnostic_manifest.resolve()),
    }
    print(
        "L3-A3 live collision diagnostic "
        + json.dumps(
            controller_live_collision_diagnostic, sort_keys=True
        ),
        flush=True,
    )
    reacquire_steps = int(center_high_reacquire["steps"])
    return {
        "selected_native_push_direction_relation": "trailing_minus_push",
        "selected_outward_direction_xy": outward.tolist(),
        "old_plus_x_route_fallback_permitted": False,
        "runtime_tangent_fallback_permitted": False,
        "prereacquire_center_high_target": center_high_target.tolist(),
        "center_high_target": live_center_high_target.tolist(),
        "anchor_eef_position_world": anchor_eef.tolist(),
        "native_osc_action_spec": native_action_spec,
        "native_osc_rotation_spec": rotation_spec,
        "table_normal_derivation": table_normal_evidence,
        "overhead_geometry": overhead_geometry,
        "initial_overhead_guard": initial_overhead_guard,
        "initial_robot_nonrobot_contact_gate": initial_contact_gate,
        "tolerance_derivation": tolerance_derivation,
        "yaw_segmentation": yaw_segmentation,
        "yaw_segments": yaw_segments,
        "yaw_frames": frames,
        "yaw_steps": yaw_steps,
        "rotation_with_anchor_compensation_steps": sum(
            frame["stage"] == "rotation_with_anchor_compensation"
            for frame in frames
        ),
        "position_settle_steps": int(total_position_settle_steps),
        "maximum_position_settle_steps_per_segment": int(
            args.push_tracking_steps
        ),
        "maximum_observed_position_drift_m": float(
            maximum_observed_position_drift
        ),
        "final_position_drift_m": float(
            final_attainment["eef_position_drift_m"]
        ),
        "maximum_commanded_translation_action_peak": float(
            maximum_commanded_translation_action_peak
        ),
        "simultaneous_yaw_and_position_attainment_required": True,
        "final_attainment_evidence": final_attainment,
        "pre_reacquire_real_sim_recompiled_candidate": (
            pre_reacquire_realized_candidate
        ),
        "center_high_reacquire": center_high_reacquire,
        "post_reacquire_final_attainment_evidence": (
            center_high_reacquire["final_attainment_evidence"]
        ),
        "real_sim_recompiled_candidate": realized_candidate,
        "second_real_sim_recompile_identity": second_recompile_identity,
        "controller_live_collision_diagnostic": (
            controller_live_collision_diagnostic
        ),
        "total_structural_actions_before_contact_seek": int(
            yaw_steps + reacquire_steps
        ),
        "remaining_structural_waypoint_steps": int(
            args.max_waypoint_steps - yaw_steps - reacquire_steps
        ),
    }


def _select_native_plus_x_front_candidate(candidates):
    """Select the unchanged native-orientation +X plate approach.

    The +X candidate is the only compiled cardinal approach that stays in
    front of the native cabinet while the hand is high.  Selection remains
    fail-closed: the live compiler must mark the candidate eligible, its two
    finger contact skew must fit strictly inside the registered outside-rim
    clearance, and no wrist-yaw route may have been attached to it.
    """
    matches = [
        candidate
        for candidate in candidates
        if "legacy_cardinal:+x"
        in candidate.get("candidate_provenance", ())
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "native +X front corridor requires exactly one compiled candidate"
        )
    selected = matches[0]
    outside_clearance = float(
        selected.get("maximum_dual_finger_contact_skew_m", np.nan)
    )
    dual_finger_skew = float(
        selected.get("dual_finger_contact_skew_m", np.nan)
    )
    if not (
        selected.get("route_selection_candidate", False)
        and selected.get("compiled_geometry_eligible", False)
        and selected.get("selection_eligible", False)
        and not selected.get("wrist_yaw_route_selected", False)
        and np.isfinite(outside_clearance)
        and np.isfinite(dual_finger_skew)
        and 0.0 <= dual_finger_skew < outside_clearance
        and np.allclose(
            selected.get("outward_direction_xy", ()),
            [1.0, 0.0],
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError(
            "compiled native +X front corridor is not strictly eligible"
        )
    return selected


def _prepare_native_plus_x_front_corridor(
    rollout,
    env,
    args,
    *,
    plate_position,
    push_direction_xy,
    center_high_target,
    diagnostics,
):
    """Recompile and authorize a zero-yaw route in front of the cabinet."""
    current_eef = np.asarray(
        rollout.obs["robot0_eef_pos"], dtype=float
    )
    center_high_target = np.asarray(center_high_target, dtype=float)
    plate_position = np.asarray(plate_position, dtype=float)
    if any(
        value.shape != (3,) or not np.all(np.isfinite(value))
        for value in (current_eef, center_high_target, plate_position)
    ):
        raise RuntimeError("native +X front corridor inputs must be finite 3-D")
    center_error = float(np.linalg.norm(current_eef - center_high_target))
    center_xy_error = float(
        np.linalg.norm(current_eef[:2] - plate_position[:2])
    )
    contact_gate = _robot_nonrobot_contact_evidence(
        env, allowed_body_pairs=()
    )
    if not (
        center_error <= float(args.position_tolerance)
        and center_xy_error <= float(args.position_tolerance)
        and contact_gate["accepted"]
    ):
        raise RuntimeError(
            "native +X front corridor lacks a collision-free center-high start: "
            f"center_error_m={center_error} "
            f"center_xy_error_m={center_xy_error} "
            f"contact_gate={json.dumps(contact_gate, sort_keys=True)}"
        )

    _, recompiled_candidates = _compiled_trailing_side_contact_candidates(
        env,
        plate_position=plate_position,
        push_direction_xy=push_direction_xy,
        eef_position=current_eef,
        backoff=args.plate_contact_backoff,
        outside_clearance_m=args.plate_contact_outside_clearance,
        plate_approach_eef_height=args.plate_approach_eef_height,
        position_action_scale=args.position_action_scale,
        reference_outward_direction_xy=np.array([1.0, 0.0], dtype=float),
        selection_mode="native_plus_x_front_corridor",
    )
    realized_candidate = _select_native_plus_x_front_candidate(
        recompiled_candidates
    )
    live_inventory = _live_collision_inventory(
        env, eef_position=current_eef
    )
    live_inventory["capture_stage"] = (
        "post_center_high_zero_yaw_front_corridor_recompile"
    )
    cabinet_pose = _live_cabinet_pose_diagnostic(env)
    route_evidence = {
        "candidate_id": "native_plus_x_front_corridor",
        "diagnostic_only": False,
        "selection_eligible": True,
        "selected": True,
        "route_authorized": True,
        "wrist_yaw_executed": False,
        "outward_direction_xy": [1.0, 0.0],
        "center_high_target": center_high_target.tolist(),
        "outside_high_target": realized_candidate[
            "outside_high_target"
        ],
        "outside_side_target": realized_candidate[
            "outside_side_target"
        ],
        "side_contact_target": realized_candidate[
            "side_contact_target"
        ],
        "outside_clearance_m": float(
            args.plate_contact_outside_clearance
        ),
        "dual_finger_contact_skew_m": float(
            realized_candidate["dual_finger_contact_skew_m"]
        ),
        "authorization_basis": (
            "live native-orientation +X candidate, exact two-finger skew, "
            "collision-free center-high start, and the existing per-action "
            "55-pair overhead/outside/table/contact gates"
        ),
    }
    diagnostic_manifest_value = getattr(args, "diagnostic_manifest", None)
    if diagnostic_manifest_value is None:
        output_value = getattr(args, "output", None)
        if not output_value:
            raise RuntimeError(
                "controller diagnostic manifest requires an output path"
            )
        diagnostic_manifest_value = str(
            Path(output_value).with_suffix(".controller_diagnostic.json")
        )
    diagnostic_manifest = _write_controller_diagnostic_manifest(
        diagnostic_manifest_value,
        {
            "schema_version": 1,
            "scenario": SCENE_ID,
            "task_description": TASK_PROMPT,
            "diagnostic_only": False,
            "route_authorized": True,
            "capture_stage": live_inventory["capture_stage"],
            "live_collision_inventory": live_inventory,
            "live_cabinet_pose_and_qpos": cabinet_pose,
            "authorized_front_corridor": route_evidence,
            "authorized_detour_plan": None,
            "unexpected_contact_events": [],
            "latest_status": "NATIVE_PLUS_X_FRONT_CORRIDOR_AUTHORIZED",
        },
    )
    controller_context = {
        "diagnostic_only": False,
        "executed": False,
        "selection_eligible": True,
        "selected": True,
        "route_authorized": True,
        "capture_stage": live_inventory["capture_stage"],
        "inventory_sha256": live_inventory["inventory_sha256"],
        "robot_collision_geom_count": live_inventory[
            "robot_collision_geom_count"
        ],
        "native_nonrobot_collision_geom_count": live_inventory[
            "native_nonrobot_collision_geom_count"
        ],
        "total_collision_geom_count": live_inventory[
            "total_collision_geom_count"
        ],
        "authorized_front_corridor": route_evidence,
        "authorized_detour_plan": None,
        "manifest_path": str(diagnostic_manifest.resolve()),
    }
    print(
        "L3-A3 native +X front corridor authorized "
        + json.dumps(controller_context, sort_keys=True),
        flush=True,
    )
    return {
        "selected_native_push_direction_relation": "legacy_cardinal:+x",
        "selected_outward_direction_xy": [1.0, 0.0],
        "legacy_plus_x_route_selected": True,
        "wrist_yaw_executed": False,
        "runtime_tangent_fallback_permitted": False,
        "center_high_target": center_high_target.tolist(),
        "real_sim_recompiled_candidate": realized_candidate,
        "controller_live_collision_diagnostic": controller_context,
        "yaw_steps": 0,
        "total_structural_actions_before_contact_seek": 0,
        "remaining_structural_waypoint_steps": int(
            args.max_waypoint_steps
        ),
    }


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
        if not (
            0 <= geom1 < int(model.ngeom)
            and 0 <= geom2 < int(model.ngeom)
            and geom1 != geom2
        ):
            raise RuntimeError(
                "robot/native contact has invalid compiled geom ids"
            )
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
        position = np.asarray(contact.pos, dtype=float)
        frame = np.asarray(contact.frame, dtype=float)
        try:
            distance = float(contact.dist)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "robot/native contact distance is unavailable or invalid"
            ) from exc
        if (
            position.shape != (3,)
            or frame.shape != (9,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(frame))
            or not np.isfinite(distance)
        ):
            raise RuntimeError(
                "robot/native contact pos/frame/dist evidence is invalid"
            )
        raw_normal = frame[:3]
        normal_norm = float(np.linalg.norm(raw_normal))
        if not np.isfinite(normal_norm) or not np.isclose(
            normal_norm, 1.0, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(
                "robot/native contact frame normal is not finite unit length"
            )
        sorted_geom_ids = sorted((geom1, geom2))
        normal_from_sorted_geom0_to_geom1 = (
            raw_normal if geom1 == sorted_geom_ids[0] else -raw_normal
        )
        robot_to_native_normal = (
            raw_normal if robot_geom == geom1 else -raw_normal
        )
        pair = (robot_body, native_body)
        contacts.append(
            {
                "contact_index": int(index),
                "geom1_id": geom1,
                "geom1_name": model.geom_id2name(geom1) or "",
                "geom1_body": body1,
                "geom2_id": geom2,
                "geom2_name": model.geom_id2name(geom2) or "",
                "geom2_body": body2,
                "robot_geom": model.geom_id2name(robot_geom) or "",
                "robot_geom_id": robot_geom,
                "robot_body": robot_body,
                "native_geom": model.geom_id2name(native_geom) or "",
                "native_geom_id": native_geom,
                "native_body": native_body,
                "position_world": position.tolist(),
                "frame_normal_geom1_to_geom2_world": raw_normal.tolist(),
                "frame_normal_semantics": (
                    "MuJoCo contact.frame[0:3], from geom1 to geom2; geom ids "
                    "and names above preserve the exact compiled ordering"
                ),
                "sorted_geom_ids": sorted_geom_ids,
                "normal_from_sorted_geom0_to_geom1_world": (
                    normal_from_sorted_geom0_to_geom1.tolist()
                ),
                "sorted_normal_was_flipped": bool(
                    geom1 != sorted_geom_ids[0]
                ),
                "robot_to_native_normal_world": (
                    robot_to_native_normal.tolist()
                ),
                "robot_to_native_normal_was_flipped": bool(
                    robot_geom != geom1
                ),
                "distance_m": distance,
                "penetration_m": float(max(0.0, -distance)),
                "robot_live_geom": _live_collision_geom_record(
                    model, data, robot_geom
                ),
                "native_live_geom": _live_collision_geom_record(
                    model, data, native_geom
                ),
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
    structural_waypoint_budget=None,
    controller_live_diagnostic=None,
):
    """Descend outside the plate, then establish two-finger side contact."""
    if structural_waypoint_budget is None:
        structural_waypoint_budget = int(args.max_waypoint_steps)
    if (
        not isinstance(structural_waypoint_budget, (int, np.integer))
        or structural_waypoint_budget < 1
        or structural_waypoint_budget > int(args.max_waypoint_steps)
    ):
        raise ValueError(
            "structural waypoint budget must be in the unchanged configured range"
        )
    structural_waypoint_budget = int(structural_waypoint_budget)
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
            contact_gate = sample["robot_nonrobot_contact_gate"]
            compact_contact = None
            if contact_gate["unexpected_contacts"]:
                if not isinstance(controller_live_diagnostic, dict):
                    raise RuntimeError(
                        "unexpected robot/native contact lacks the required "
                        "controller live diagnostic context"
                    )
                compact_contact = _compact_unexpected_contact_diagnostic(
                    contact_gate,
                    diagnostic_context=controller_live_diagnostic,
                )
                _record_unexpected_contact_in_controller_manifest(
                    diagnostic_context=controller_live_diagnostic,
                    source=source,
                    stage=stage,
                    sample=sample,
                )
                print(
                    "L3-A3 unexpected robot/native contact diagnostic "
                    + json.dumps(compact_contact, sort_keys=True),
                    flush=True,
                )
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
                "unexpected_contact_compact_diagnostic": compact_contact,
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
    structural_max_translation_action = float(
        args.structural_near_plate_max_translation_action
    )
    overhead_descent_max_translation_action = float(
        args.overhead_descent_max_translation_action
    )
    vertical_corridor_descent_max_translation_action = float(
        args.vertical_corridor_descent_max_translation_action
    )
    post_descent_lateral_max_translation_action = float(
        args.post_descent_lateral_max_translation_action
    )
    maximum_controller_world_step = float(
        args.position_action_scale * structural_max_translation_action
    )
    maximum_overhead_descent_world_step = float(
        args.position_action_scale
        * overhead_descent_max_translation_action
    )
    maximum_post_descent_lateral_world_step = float(
        args.position_action_scale
        * post_descent_lateral_max_translation_action
    )
    tail_recovery_descent_translation_action_floor = float(
        post_descent_lateral_max_translation_action
    )
    if not (
        structural_max_translation_action
        < tail_recovery_descent_translation_action_floor
        <= post_descent_lateral_max_translation_action
        < overhead_descent_max_translation_action
    ):
        raise RuntimeError(
            "tail-recovery descent floor is not strictly nested inside the "
            "existing structural, post-descent, and overhead action bounds"
        )
    overhead_descent_brake_trigger_buffer = float(
        2.0 * maximum_overhead_descent_world_step
    )
    active_overhead_descent_translation_action = float(
        overhead_descent_max_translation_action
    )
    active_overhead_descent_world_step = float(
        maximum_overhead_descent_world_step
    )
    active_overhead_descent_brake_trigger_buffer = float(
        overhead_descent_brake_trigger_buffer
    )
    structural_seek_context.update(
        {
            "structural_near_plate_max_translation_action": (
                structural_max_translation_action
            ),
            "structural_near_plate_maximum_world_step_m": (
                maximum_controller_world_step
            ),
            "overhead_descent_max_translation_action": (
                overhead_descent_max_translation_action
            ),
            "overhead_descent_maximum_world_step_m": (
                maximum_overhead_descent_world_step
            ),
            "overhead_descent_brake_trigger_buffer_m": (
                overhead_descent_brake_trigger_buffer
            ),
            "vertical_corridor_descent_max_translation_action": (
                vertical_corridor_descent_max_translation_action
            ),
            "post_descent_lateral_max_translation_action": (
                post_descent_lateral_max_translation_action
            ),
            "post_descent_lateral_maximum_world_step_m": (
                maximum_post_descent_lateral_world_step
            ),
            "tail_recovery_descent_translation_action_floor": (
                tail_recovery_descent_translation_action_floor
            ),
            "tail_recovery_descent_floor_derivation": (
                "the existing post-descent lateral action bound"
            ),
        }
    )
    detour_native_low = np.asarray(native_action_spec["low"], dtype=float)
    detour_native_high = np.asarray(native_action_spec["high"], dtype=float)
    strict_native_detour_translation_action = float(
        np.nextafter(
            min(
                -detour_native_low[0],
                detour_native_high[0],
                -detour_native_low[1],
                detour_native_high[1],
                -detour_native_low[2],
                detour_native_high[2],
            ),
            0.0,
        )
    )
    cabinet_detour_plan = None
    if controller_live_diagnostic is not None:
        if not isinstance(controller_live_diagnostic, dict):
            raise RuntimeError(
                "controller live diagnostic context must be a dictionary"
            )
        cabinet_detour_plan = controller_live_diagnostic.get(
            "authorized_detour_plan"
        )
        if cabinet_detour_plan is not None:
            if not (
                cabinet_detour_plan.get("route_authorized", False)
                and cabinet_detour_plan.get("selected", False)
                and cabinet_detour_plan.get("selection_eligible", False)
                and not cabinet_detour_plan.get("diagnostic_only", True)
            ):
                raise RuntimeError(
                    "native cabinet detour plan is not executable"
                )
            if not np.isclose(
                float(
                    cabinet_detour_plan[
                        "maximum_controller_world_step_m"
                    ]
                ),
                maximum_controller_world_step,
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError(
                    "native cabinet detour controller step binding changed"
                )
            if not np.isclose(
                float(
                    cabinet_detour_plan[
                        "maximum_route_translation_action"
                    ]
                ),
                strict_native_detour_translation_action,
                rtol=0.0,
                atol=0.0,
            ) or not np.isclose(
                float(
                    cabinet_detour_plan[
                        "position_action_scale_m_per_action"
                    ]
                ),
                float(args.position_action_scale),
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError(
                    "native cabinet detour route-action binding changed"
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
        maximum_translation_action=structural_max_translation_action,
    )
    corridor_outward_direction = np.asarray(
        geometry["outward_direction_xy"], dtype=float
    )
    corridor_outward_norm = float(
        np.linalg.norm(corridor_outward_direction)
    )
    if (
        corridor_outward_direction.shape != (2,)
        or not np.all(np.isfinite(corridor_outward_direction))
        or not np.isfinite(corridor_outward_norm)
        or corridor_outward_norm <= 1e-9
    ):
        raise RuntimeError(
            "compiled corridor outward direction is invalid"
        )
    corridor_outward_direction = (
        corridor_outward_direction / corridor_outward_norm
    )
    corridor_rebuffer_target = np.asarray(
        corridor_high_target, dtype=float
    ).copy()
    corridor_rebuffer_target[:2] += (
        corridor_outward_direction
        * float(args.minimum_saturated_waypoint_progress)
    )
    corridor_rebuffer_clearance = float(
        np.nextafter(
            vertical_staging_corridor["corridor_clearance_m"]
            + float(args.minimum_saturated_waypoint_progress),
            np.inf,
        )
    )
    corridor_rebuffer_acceptance_clearance = float(
        vertical_staging_corridor["corridor_clearance_m"]
    )
    corridor_correction_hold_target_xy = (
        corridor_rebuffer_target[:2]
        + corridor_outward_direction
        * maximum_post_descent_lateral_world_step
    )
    vertical_corridor_balanced_hold_world_step = float(
        args.position_action_scale
        * vertical_corridor_descent_max_translation_action
        / np.sqrt(2.0)
    )
    vertical_corridor_balanced_hold_target_xy = (
        corridor_rebuffer_target[:2]
        + corridor_outward_direction
        * vertical_corridor_balanced_hold_world_step
    )
    vertical_corridor_reserve_recovery_entry_clearance = float(
        vertical_staging_corridor[
            "strict_corridor_entry_clearance_m"
        ]
        + maximum_controller_world_step
    )
    vertical_corridor_reserve_recovery_exit_clearance = float(
        corridor_rebuffer_acceptance_clearance
    )
    if not (
        np.isfinite(vertical_corridor_balanced_hold_world_step)
        and 0.0 < vertical_corridor_balanced_hold_world_step
        < maximum_post_descent_lateral_world_step
        and np.all(
            np.isfinite(vertical_corridor_balanced_hold_target_xy)
        )
        and np.isfinite(
            vertical_corridor_reserve_recovery_entry_clearance
        )
        and np.isfinite(
            vertical_corridor_reserve_recovery_exit_clearance
        )
        and vertical_corridor_reserve_recovery_exit_clearance
        > vertical_corridor_reserve_recovery_entry_clearance
        > vertical_staging_corridor[
            "strict_corridor_entry_clearance_m"
        ]
    ):
        raise RuntimeError(
            "vertical corridor balanced hold is not strictly inside the "
            "existing post-descent controller reserve, or its pre-loss "
            "recovery hysteresis is invalid"
        )
    structural_seek_context.update(
        {
            "vertical_corridor_balanced_hold_world_step_m": (
                vertical_corridor_balanced_hold_world_step
            ),
            "vertical_corridor_balanced_hold_target_xy": (
                vertical_corridor_balanced_hold_target_xy.tolist()
            ),
            "vertical_corridor_balanced_hold_derivation": (
                "position action scale times the existing vertical-corridor "
                "translation-action bound divided by sqrt(2), retaining "
                "equal strict action-norm capacity for outward XY and Z"
            ),
            "vertical_corridor_reserve_recovery_entry_clearance_m": (
                vertical_corridor_reserve_recovery_entry_clearance
            ),
            "vertical_corridor_reserve_recovery_entry_derivation": (
                "the unchanged strict 0.4 mm corridor gate plus the existing "
                "0.4 mm maximum structural controller world step"
            ),
            "vertical_corridor_reserve_recovery_exit_clearance_m": (
                vertical_corridor_reserve_recovery_exit_clearance
            ),
            "vertical_corridor_reserve_recovery_exit_derivation": (
                "the unchanged formal 0.9 mm corridor clearance"
            ),
        }
    )
    corridor_correction_handoff_target = np.asarray(
        corridor_rebuffer_target, dtype=float
    ).copy()
    corridor_correction_handoff_target[:2] = (
        corridor_correction_hold_target_xy
    )
    high_z_controller_handoff_tolerance = float(
        min(
            float(args.position_tolerance),
            0.5 * maximum_post_descent_lateral_world_step,
        )
    )
    minimum_realized_controller_reserve = float(
        maximum_post_descent_lateral_world_step
        - high_z_controller_handoff_tolerance
    )
    if not (
        np.all(np.isfinite(corridor_rebuffer_target))
        and np.all(np.isfinite(corridor_correction_hold_target_xy))
        and np.all(np.isfinite(corridor_correction_handoff_target))
        and np.isfinite(high_z_controller_handoff_tolerance)
        and 0.0 < high_z_controller_handoff_tolerance
        < maximum_post_descent_lateral_world_step
        and np.isfinite(minimum_realized_controller_reserve)
        and minimum_realized_controller_reserve > 0.0
        and np.isfinite(corridor_rebuffer_clearance)
        and np.isfinite(corridor_rebuffer_acceptance_clearance)
        and corridor_rebuffer_clearance
        > corridor_rebuffer_acceptance_clearance
    ):
        raise RuntimeError(
            "compiled corridor rebuffer lacks a strict outward measurement-"
            "resolution reserve"
        )

    def _high_z_controller_handoff_evidence(
        *,
        current_eef,
        outside_side_guard,
        overhead_guard,
        overhead_lateral_buffer,
    ):
        return _overhead_corridor_entry_evidence(
            current_eef=current_eef,
            corridor_high_target=corridor_correction_handoff_target,
            outside_side_guard=outside_side_guard,
            overhead_guard=overhead_guard,
            overhead_lateral_buffer=overhead_lateral_buffer,
            position_tolerance=high_z_controller_handoff_tolerance,
            strict_corridor_entry_clearance_m=(
                corridor_rebuffer_acceptance_clearance
            ),
            require_lateral_buffer=False,
            minimum_eef_z=None,
        )
    (
        high_lateral_prebuffer_target,
        high_lateral_prebuffer_evidence,
    ) = _strict_native_high_prebuffer_target(
        native_outside_high_target=overhead_outside_high_target,
        corridor_high_target=corridor_high_target,
        outward_direction_xy=geometry["outward_direction_xy"],
        minimum_lateral_reserve_m=maximum_controller_world_step,
        maximum_nextafter_steps=128,
    )
    high_lateral_anticooupling_reserve = float(
        high_lateral_prebuffer_evidence[
            "final_euclidean_reserve_m"
        ]
    )
    high_lateral_outward_projection = float(
        high_lateral_prebuffer_evidence[
            "final_outward_projection_m"
        ]
    )
    if not (
        np.all(np.isfinite(high_lateral_prebuffer_target))
        and high_lateral_anticooupling_reserve
        > maximum_controller_world_step
        and high_lateral_outward_projection
        > maximum_controller_world_step
    ):
        raise RuntimeError(
            "registered high-plane corridor target lacks the strict existing "
            "one-step anti-coupling lateral reserve"
        )
    overhead_horizontal_travel = float(
        np.linalg.norm(
            high_lateral_prebuffer_target[:2]
            - initial_eef[:2]
        )
    )
    workspace_release_xy_travel = float(
        np.linalg.norm(
            corridor_high_target[:2]
            - high_lateral_prebuffer_target[:2]
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
        + workspace_release_xy_travel
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
    if minimum_full_scale_actions_from_geometry > structural_waypoint_budget:
        raise RuntimeError(
            "compiled overhead route geometric lower bound alone exceeds "
            "the unchanged structural waypoint hard loop: "
            f"source={source} lower_bound_action_equivalents="
            f"{total_structural_action_lower_bound} "
            f"minimum_full_scale_actions="
            f"{minimum_full_scale_actions_from_geometry} "
            f"maximum_steps={structural_waypoint_budget}"
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
            "reachable_outside_high_target": (
                np.asarray(outside_high_target, dtype=float).tolist()
            ),
            "high_plane_anticooupling_lateral_target": (
                high_lateral_prebuffer_target.tolist()
            ),
            "high_plane_anticooupling_lateral_reserve_m": (
                high_lateral_anticooupling_reserve
            ),
            "high_plane_anticooupling_outward_projection_m": (
                high_lateral_outward_projection
            ),
            "high_plane_anticooupling_prebuffer_evidence": (
                high_lateral_prebuffer_evidence
            ),
            "high_plane_anticooupling_lateral_reserve_source": (
                "start from registered corridor_high_target minus unchanged "
                "native outside_high_target; if final-coordinate Euclidean "
                "reserve or normalized native-tangent projection does not "
                "remain strictly above the existing controller world step "
                "after floating-point reconstruction, advance only that "
                "scalar projection by at most 128 nextafter(+inf) values and "
                "reconstruct the high-plane prebuffer; corridor high/side "
                "targets, candidate semantics, and threshold remain unchanged"
            ),
            "native_high_boundary_crossing_gate": {
                "uses_position_tolerance": False,
                "required_compiled_pair_count": (
                    expected_overhead_pair_count
                ),
                "requires_persistent_registered_corridor_request": True,
                "requires_strict_actual_native_target_crossing": True,
                "requires_strict_live_outside_clearance": True,
                "requires_pre_envelope_post_high_plane_base8": True,
                "requires_nonnegative_z_zero_rotation": True,
                "fallback_order": (
                    "after ordinary registered-corridor waypoint gate and "
                    "before native workspace saturation"
                ),
            },
            "native_high_workspace_saturation_gate": {
                "progress_epsilon_m": float(
                    args.minimum_saturated_waypoint_progress
                ),
                "progress_epsilon_source": (
                    "minimum_saturated_waypoint_progress"
                ),
                "required_window_frames": int(args.push_tracking_steps),
                "required_window_source": "push_tracking_steps",
                "native_action_bounds_source": native_action_spec["source"],
                "required_compiled_pair_count": (
                    expected_overhead_pair_count
                ),
                "position_tolerance_m": float(args.position_tolerance),
                "position_tolerance_is_unchanged": True,
            },
            "corridor_adaptive_descent_target": (
                corridor_high_target.tolist()
            ),
            "descent_lateral_drift_brake_threshold_m": float(
                args.position_tolerance
            ),
            "descent_lateral_drift_brake_threshold_source": (
                "unchanged position_tolerance"
            ),
            "descent_outside_clearance_brake_threshold_m": float(
                vertical_staging_corridor[
                    "strict_corridor_entry_clearance_m"
                ]
            ),
            "descent_outside_clearance_brake_threshold_source": (
                "compiled strict corridor-entry clearance"
            ),
            "descent_corridor_resume_clearance_m": (
                corridor_rebuffer_acceptance_clearance
            ),
            "descent_corridor_resume_clearance_source": (
                "unchanged compiled full corridor clearance"
            ),
            "descent_corridor_rebuffer_requested_clearance_m": (
                corridor_rebuffer_clearance
            ),
            "descent_corridor_rebuffer_request_source": (
                "compiled full corridor clearance plus the existing "
                "minimum_saturated_waypoint_progress measurement resolution"
            ),
            "descent_corridor_rebuffer_target": (
                corridor_rebuffer_target.tolist()
            ),
            "post_descent_correction_controller_hold_target_xy": (
                corridor_correction_hold_target_xy.tolist()
            ),
            "post_descent_correction_controller_outward_reserve_m": (
                maximum_post_descent_lateral_world_step
            ),
            "post_descent_correction_controller_hold_target_formula": (
                "corridor_rebuffer_target XY plus normalized registered "
                "outward direction times the existing post-descent one-step "
                "world displacement. Before the first descent this is the "
                "above-staging correction target. After any descent, first "
                "hold the measured current Z and use a bounded XY action to "
                "restore that same shifted controller target; only then may "
                "XY/nonnegative-Z plane recovery resume. Every post-descent "
                "handoff requires the shifted controller target, unchanged "
                "formal target and clearance, and registered Z-tail deadband"
            ),
            "post_descent_correction_high_z_handoff_gate": {
                "controller_target": (
                    corridor_correction_hold_target_xy.tolist()
                ),
                "controller_handoff_tolerance_m": (
                    high_z_controller_handoff_tolerance
                ),
                "controller_handoff_tolerance_source": (
                    "minimum of the unchanged formal position_tolerance and "
                    "one half of the existing post-descent one-step world "
                    "displacement"
                ),
                "minimum_realized_outward_controller_reserve_m": (
                    minimum_realized_controller_reserve
                ),
                "formal_position_tolerance_m": float(
                    args.position_tolerance
                ),
                "formal_position_tolerance_unchanged": True,
                "required_compiled_pair_count": (
                    expected_overhead_pair_count
                ),
                "requires_live_overhead_base8": True,
                "plane_recovery_tolerance_m": float(
                    args.position_tolerance
                ),
                "plane_recovery_tolerance_source": (
                    "unchanged formal position_tolerance"
                ),
                "plane_recovery_rule": (
                    "while the above-staging shifted correction target is "
                    "active, if nonnegative-Z plane error exceeds the "
                    "unchanged formal position tolerance, issue outward "
                    "XY/+Z under "
                    "the same native/configured norm and 55-pair base8 proof "
                    "so the measured real-controller inward coupling remains "
                    "opposed"
                ),
                "negative_tail_recovery_threshold_m": float(
                    args.minimum_saturated_waypoint_progress
                ),
                "negative_tail_recovery_threshold_source": (
                    "existing minimum_saturated_waypoint_progress"
                ),
                "negative_tail_recovery_rule": (
                    "during post-descent correction, if the latest measured "
                    "negative Z step exceeds the registered threshold, issue "
                    "outward XY/+Z under the same native/configured norm and "
                    "55-pair base8 proof; the complete XY/+Z norm is reserved "
                    "as worst-case downward tail"
                ),
                "post_descent_xy_only_rebuffer_rule": (
                    "until the shifted controller target is within its "
                    "existing handoff tolerance, set the correction plane to "
                    "the measured current EEF Z, disable plane/tail recovery "
                    "triggers, and issue bounded XY with zero commanded Z "
                    "under the same exact 55-pair measured-tail proof"
                ),
                "controller_handoff_applies_before_and_after_overhead_descent": True,
                "plane_recovery_applies_only_above_staging_tolerance": True,
                "post_descent_formal_handoff_vertical_tail_deadband_m": (
                    -float(args.minimum_saturated_waypoint_progress)
                ),
                "post_descent_formal_handoff_vertical_tail_deadband_source": (
                    "existing minimum_saturated_waypoint_progress"
                ),
                "formal_corridor_acceptance_target_unchanged": True,
                "formal_corridor_acceptance_clearance_unchanged": True,
            },
            "descent_corridor_hold_target_formula": (
                "corridor_rebuffer_target XY plus normalized registered "
                "outward direction times the active overhead-descent one-step "
                "world displacement; the formal corridor acceptance target "
                "and clearance remain unchanged"
            ),
            "descent_motion_reversal_brake": {
                "maximum_permitted_inward_step_m": float(
                    args.minimum_saturated_waypoint_progress
                ),
                "threshold_source": (
                    "existing minimum_saturated_waypoint_progress"
                ),
                "threshold_rule": (
                    "brake when either measured outward or clearance "
                    "progress is below the negative deadband after the "
                    "unchanged full corridor clearance is no longer retained; "
                    "while that reserve remains strict, the registered "
                    "one-sided hold continues instead of discarding safe "
                    "outward margin; no empirical Z threshold"
                ),
            },
            "structural_route_order": [
                (
                    "native_center_high_to_registered_corridor_high_"
                    "anticooupling_prebuffer"
                ),
                (
                    "corridor_directed_workspace_release_diagonal_with_only_"
                    "registered_ulp_bounded_inward_return"
                ),
                (
                    "corridor_xy_adaptive_one_sided_coupled_descent_with_"
                    "position_"
                    "tolerance_strict_clearance_or_unbuffered_measured_"
                    "inward_response_"
                    "brake"
                ),
                "vertical_tail_brake_and_formal_corridor_handoff",
                "high_z_controller_reserve_or_staging_formal_entry_via_"
                "xy_nonnegative_z_plane_hold_correction",
                "vertical_side_corridor_and_contact",
            ],
            "horizontal_sweep_formula": (
                "from the exact native center-high first-policy state, command "
                "XY plus nonnegative Z with zero rotation toward the already "
                "registered corridor-high target while holding the initial "
                "center-high Z plane; this moves the existing geometry-derived "
                "one-world-step corridor reserve to the safe high plane without "
                "redefining the native outside-high target; only after that "
                "target passes, command corridor-directed XY plus nonpositive "
                "Z toward the unchanged strict corridor XY and outside-side Z "
                "to release the high workspace; XY remains outward except for "
                "a return from the representability-only prebuffer nudge, "
                "which must remain within its recorded one-ULP-expanded "
                "displacement or fail closed before action compilation; if "
                "that registered target is "
                "outside the runtime native action bound, permit the same "
                "release only after the existing push-tracking window proves "
                "persistent outward requests with EEF and live-clearance "
                "responses within the existing saturated-waypoint progress "
                "epsilon, the actual EEF lies strictly beyond the unchanged "
                "native outside-high target, and all 55 high-plane pre/post "
                "base8 checks remain strict; before that saturation fallback, "
                "permit the same release as soon as an actual high-plane EEF "
                "step strictly crosses the unchanged native outside-high "
                "target under a continuing registered-corridor request, "
                "strict live outside clearance, nonnegative Z, zero rotation, "
                "and the same all-55-pair pre/envelope/post base8 proof; "
                "if the latest negative-dz inertia exhausts diagonal downward "
                "capacity, prohibit negative Z and issue a 55-pair-proved pure "
                "+Z recovery before recomputing the diagonal; "
                "derive each 3-D translation-action norm from the strict "
                "runtime native bound and all 55 live pairs' current clearance "
                "minus strict+base8 and the latest measured negative-dz "
                "inertial reserve. If pair capacity is limiting, issue pure "
                "+Z recovery instead of near-zero XY; remeasure post-action "
                "base8 and the empty structural robot/native contact allowlist "
                "on every frame. During the far descent, command the live "
                "corridor-target XY error together with negative Z under that "
                "same 55-pair buffer16 and measured-inertia proof, capped by "
                "the registered 0.20 descent bound; a zero XY error never "
                "suppresses required Z progress, and the registered outward "
                "safety axis may command only outward or zero motion, never "
                "an inward return after target overshoot. The descent-only "
                "hold target adds exactly the active one-step world "
                "displacement in that outward direction, so the same "
                "geometric cap-halving schedule also shrinks this deterministic "
                "inertia reserve; it does not change formal corridor "
                "acceptance. The post-descent plane-hold correction similarly "
                "uses its existing one-step 8 mm world displacement as a "
                "deterministic outward controller-target reserve to overcome "
                "the observed proportional static error. Above staging use "
                "that shifted target for the correction action. Before the "
                "first overhead corridor descent, do not hand control to "
                "descent at the first formal-target crossing; require the "
                "actual EEF to reach the controller target within the "
                "deterministic half-one-step handoff tolerance and retain live "
                "base8, thereby physically realizing at least half the "
                "existing 8 mm reserve. After any overhead descent action, "
                "first hold the measured current Z and use the same bounded "
                "zero-Z XY action to restore the shifted controller target "
                "under the exact measured-tail/base8 proof. Only then resume "
                "XY/nonnegative-Z recovery. Continued descent or vertical-"
                "corridor handoff requires that shifted target together with "
                "the unchanged formal corridor target and clearance and a "
                "measured vertical-step "
                "handoff interlock using the existing negative vertical-"
                "staging minimum_saturated_waypoint_progress deadband so "
                "descent cannot resume with an observed downward tail outside "
                "that registered deadband. If high-plane "
                "Z error exceeds the "
                "unchanged formal position tolerance, reserve the complete "
                "configured action norm for proved outward XY/+Z recovery. "
                "During post-descent correction, also reserve "
                "the complete action for proved outward XY/+Z recovery whenever "
                "the latest measured negative Z step exceeds the existing "
                "minimum_saturated_waypoint_progress threshold. At or below "
                "staging, retain the shifted target until the same controller "
                "handoff reserve is restored; the unchanged formal target and "
                "clearance remain mandatory for the vertical-corridor "
                "transition. The unchanged 0.10 "
                "bound remains exclusive to the post-descent XY/nonnegative-Z "
                "plane-hold correction and contact motion"
            ),
            "measurement_scope": (
                "live pre/post world-AABB and contact observations with the "
                "dynamic high-plane base8 plus measured-inertial-tail envelope "
                "and the unchanged later-stage 8/16 mm envelopes; internal "
                "controller substeps are not directly measured"
            ),
            "horizontal_sweep_distance_m": overhead_horizontal_travel,
            "workspace_release_xy_travel_m": workspace_release_xy_travel,
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
                "correction are enforced at runtime by the configured finite structural hard loop"
            ),
            "maximum_structural_waypoint_steps": (
                structural_waypoint_budget
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
    if cabinet_detour_plan is not None:
        structural_seek_context.update(
            {
                "authorized_native_cabinet_detour": cabinet_detour_plan,
                "structural_route_order": list(
                    cabinet_detour_plan["route_order"]
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
    if cabinet_detour_plan is not None:
        structural_stage = "right_high_lateral"
    else:
        structural_stage = "overhead_high_corridor_lateral"
    overhead_horizontal_z = float(initial_eef[2])
    latest_vertical_step_progress_m = 0.0
    latest_outward_step_progress_m = 0.0
    vertical_corridor_reserve_recovery_active = False
    vertical_corridor_reserve_recovery_phase = None
    vertical_corridor_reserve_recovery_events = []
    lateral_resume_stage = None
    vertical_tail_brake_reason = None
    vertical_tail_events = []
    high_plane_workspace_saturation_observations = []
    fixed_safe_z = None
    structural_stage_action_counts = {
        "right_high_lateral": 0,
        "right_high_trailing_pass": 0,
        "right_trailing_vertical_descent": 0,
        "trailing_low_terminal_return": 0,
        "overhead_high_corridor_lateral": 0,
        "workspace_release_diagonal": 0,
        "overhead_corridor_descent": 0,
        "vertical_tail_brake": 0,
        "lateral_rebuffer_brake": 0,
        "overhead_post_descent_corridor_lateral": 0,
        "vertical_corridor_descent": 0,
        "vertical_corridor_settle": 0,
        "fixed_safe_z_lateral_approach": 0,
    }
    overhead_lateral_stages = {
        "overhead_high_corridor_lateral",
        "workspace_release_diagonal",
        "overhead_post_descent_corridor_lateral",
    }
    fixed_buffer_lateral_stages = {
        "overhead_post_descent_corridor_lateral",
    }
    cabinet_detour_high_lateral_stages = {
        "right_high_lateral",
        "right_high_trailing_pass",
    }
    overhead_route_stages = {
        *overhead_lateral_stages,
        *cabinet_detour_high_lateral_stages,
        "overhead_corridor_descent",
        "vertical_tail_brake",
        "lateral_rebuffer_brake",
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
    for guard_step in range(1, structural_waypoint_budget + 1):
        lateral_pre_action_interlock = None
        pre_action_guard = latest_outside_side_guard
        current_eef = np.asarray(
            rollout.obs["robot0_eef_pos"], dtype=float
        )
        if structural_stage in fixed_buffer_lateral_stages or (
            structural_stage == "lateral_rebuffer_brake"
            and lateral_resume_stage in fixed_buffer_lateral_stages
        ):
            latest_overhead_lateral_buffer = (
                _overhead_lateral_buffer_evidence(
                    latest_overhead_guard,
                    worst_case_controller_world_step_m=(
                        maximum_post_descent_lateral_world_step
                    ),
                )
            )
        if structural_stage == "vertical_tail_brake":
            latest_overhead_lateral_buffer = (
                _overhead_lateral_buffer_evidence(
                    latest_overhead_guard,
                    worst_case_controller_world_step_m=(
                        maximum_post_descent_lateral_world_step
                    ),
                )
            )
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
        cabinet_detour_pre_guard = None
        if cabinet_detour_plan is not None and structural_stage in set(
            cabinet_detour_plan["route_order"]
        ):
            cabinet_detour_pre_guard = _live_native_cabinet_detour_guard(
                env,
                eef_position=current_eef,
                plan=cabinet_detour_plan,
                stage=structural_stage,
            )
            if not cabinet_detour_pre_guard["accepted"]:
                raise RuntimeError(
                    "native cabinet detour pre-action live guard failed: "
                    f"stage={structural_stage} guard_step={guard_step} "
                    f"guard={json.dumps(cabinet_detour_pre_guard, sort_keys=True)}"
                )
        if structural_stage == "right_high_lateral":
            detour_target = np.asarray(
                cabinet_detour_plan["waypoints"]["right_high"],
                dtype=float,
            )
            if (
                np.linalg.norm(current_eef[:2] - detour_target[:2])
                <= args.position_tolerance
            ):
                next_guard = _live_native_cabinet_detour_guard(
                    env,
                    eef_position=current_eef,
                    plan=cabinet_detour_plan,
                    stage="right_high_trailing_pass",
                )
                if next_guard["accepted"]:
                    structural_stage = "right_high_trailing_pass"
                    cabinet_detour_pre_guard = next_guard
        if structural_stage == "right_high_trailing_pass":
            detour_target = np.asarray(
                cabinet_detour_plan["waypoints"]["right_trailing_high"],
                dtype=float,
            )
            if (
                np.linalg.norm(current_eef[:2] - detour_target[:2])
                <= args.position_tolerance
            ):
                next_guard = _live_native_cabinet_detour_guard(
                    env,
                    eef_position=current_eef,
                    plan=cabinet_detour_plan,
                    stage="right_trailing_vertical_descent",
                )
                if next_guard["accepted"]:
                    structural_stage = "right_trailing_vertical_descent"
                    cabinet_detour_pre_guard = next_guard
        if structural_stage == "right_trailing_vertical_descent":
            detour_target = np.asarray(
                cabinet_detour_plan["waypoints"]["right_trailing_low"],
                dtype=float,
            )
            low_route_z_tolerance = float(
                cabinet_detour_plan["low_route_entry_z_tolerance_m"]
            )
            if abs(current_eef[2] - detour_target[2]) <= (
                low_route_z_tolerance
            ):
                next_guard = _live_native_cabinet_detour_guard(
                    env,
                    eef_position=current_eef,
                    plan=cabinet_detour_plan,
                    stage="trailing_low_terminal_return",
                )
                if next_guard["accepted"]:
                    structural_stage = "trailing_low_terminal_return"
                    cabinet_detour_pre_guard = next_guard
        if structural_stage == "trailing_low_terminal_return":
            detour_target = np.asarray(
                cabinet_detour_plan["waypoints"][
                    "terminal_outside_side_low"
                ],
                dtype=float,
            )
            if (
                np.linalg.norm(current_eef[:2] - detour_target[:2])
                <= args.position_tolerance
            ):
                if pre_action_guard["accepted"]:
                    break
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
        if (
            structural_stage == "workspace_release_diagonal"
            and structural_stage_action_counts[
                "workspace_release_diagonal"
            ]
            == 0
        ):
            pre_action_corridor_entry = _overhead_corridor_entry_evidence(
                current_eef=current_eef,
                corridor_high_target=corridor_high_target,
                outside_side_guard=latest_outside_side_guard,
                overhead_guard=latest_overhead_guard,
                overhead_lateral_buffer=latest_overhead_lateral_buffer,
                position_tolerance=args.position_tolerance,
                strict_corridor_entry_clearance_m=(
                    vertical_staging_corridor["corridor_clearance_m"]
                ),
                require_lateral_buffer=False,
                minimum_eef_z=None,
            )
            pre_action_controller_handoff = (
                _high_z_controller_handoff_evidence(
                    current_eef=current_eef,
                    outside_side_guard=latest_outside_side_guard,
                    overhead_guard=latest_overhead_guard,
                    overhead_lateral_buffer=(
                        latest_overhead_lateral_buffer
                    ),
                )
            )
            if pre_action_controller_handoff["accepted"]:
                structural_stage = "overhead_corridor_descent"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "high_plane_anticooupling_prebuffer_passed_"
                            "controller_handoff_before_workspace_action"
                        ),
                        "formal_corridor_entry": (
                            pre_action_corridor_entry
                        ),
                        "controller_handoff": (
                            pre_action_controller_handoff
                        ),
                    }
                )
        if structural_stage == "overhead_corridor_descent":
            latest_overhead_lateral_buffer = (
                _overhead_lateral_buffer_evidence(
                    latest_overhead_guard,
                    worst_case_controller_world_step_m=(
                        active_overhead_descent_world_step
                    ),
                )
            )
        stage_before_action = structural_stage
        prepared_high_lateral_action = None
        prepared_high_lateral_envelope = None
        correction_lateral_target_xy = None
        correction_requires_pre_descent_controller_reserve = False
        correction_uses_high_z_hold_target = False
        vertical_corridor_compiled_tail_brake_active = False
        if stage_before_action == "overhead_high_corridor_lateral":
            (
                prepared_high_lateral_action,
                prepared_high_lateral_envelope,
            ) = _compiled_adaptive_high_plane_action(
                current_eef=current_eef,
                lateral_target_xy=high_lateral_prebuffer_target[:2],
                overhead_horizontal_z=overhead_horizontal_z,
                measured_vertical_step_progress_m=(
                    latest_vertical_step_progress_m
                ),
                overhead_guard=latest_overhead_guard,
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                native_action_spec=native_action_spec,
                expected_pair_count=expected_overhead_pair_count,
            )
        elif stage_before_action in cabinet_detour_high_lateral_stages:
            adaptive_lateral_target_xy = np.asarray(
                cabinet_detour_plan["waypoints"][
                    {
                        "right_high_lateral": "right_high",
                        "right_high_trailing_pass": "right_trailing_high",
                    }[stage_before_action]
                ],
                dtype=float,
            )[:2]
            (
                prepared_high_lateral_action,
                prepared_high_lateral_envelope,
            ) = _compiled_adaptive_high_plane_action(
                current_eef=current_eef,
                lateral_target_xy=adaptive_lateral_target_xy,
                overhead_horizontal_z=overhead_horizontal_z,
                measured_vertical_step_progress_m=(
                    latest_vertical_step_progress_m
                ),
                overhead_guard=latest_overhead_guard,
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                native_action_spec=native_action_spec,
                expected_pair_count=expected_overhead_pair_count,
            )
        elif stage_before_action == "workspace_release_diagonal":
            (
                prepared_high_lateral_action,
                prepared_high_lateral_envelope,
            ) = _compiled_adaptive_workspace_release_action(
                current_eef=current_eef,
                corridor_target_xy=corridor_high_target[:2],
                release_target_z=float(outside_side_target[2]),
                measured_vertical_step_progress_m=(
                    latest_vertical_step_progress_m
                ),
                overhead_guard=latest_overhead_guard,
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                native_action_spec=native_action_spec,
                expected_pair_count=expected_overhead_pair_count,
                worst_case_controller_world_step_m=(
                    maximum_controller_world_step
                ),
                outward_direction_xy=geometry["outward_direction_xy"],
                maximum_inward_xy_correction_m=(
                    high_lateral_prebuffer_evidence[
                        "maximum_inward_return_one_ulp_bound_m"
                    ]
                ),
            )
        elif stage_before_action == "overhead_corridor_descent":
            corridor_descent_hold_target_xy = (
                corridor_rebuffer_target[:2]
                + corridor_outward_direction
                * active_overhead_descent_world_step
            )
            (
                prepared_high_lateral_action,
                prepared_high_lateral_envelope,
            ) = _compiled_adaptive_workspace_release_action(
                current_eef=current_eef,
                corridor_target_xy=corridor_descent_hold_target_xy,
                release_target_z=overhead_staging_z,
                measured_vertical_step_progress_m=(
                    latest_vertical_step_progress_m
                ),
                overhead_guard=latest_overhead_guard,
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                native_action_spec=native_action_spec,
                expected_pair_count=expected_overhead_pair_count,
                worst_case_controller_world_step_m=(
                    active_overhead_descent_world_step
                ),
                couple_downward_to_lateral_remaining=False,
                maximum_translation_action=(
                    active_overhead_descent_translation_action
                ),
                one_sided_outward_direction_xy=(
                    corridor_outward_direction
                ),
            )
        elif (
            stage_before_action
            == "overhead_post_descent_corridor_lateral"
        ):
            correction_requires_pre_descent_controller_reserve = bool(
                structural_stage_action_counts[
                    "overhead_corridor_descent"
                ]
                == 0
            )
            correction_uses_high_z_hold_target = bool(
                current_eef[2]
                > overhead_staging_z + args.position_tolerance
            )
            correction_controller_reserve_error_m = float(
                np.linalg.norm(
                    corridor_correction_hold_target_xy - current_eef[:2]
                )
            )
            correction_requires_post_descent_controller_rebuffer = bool(
                not correction_requires_pre_descent_controller_reserve
                and correction_controller_reserve_error_m
                > high_z_controller_handoff_tolerance
            )
            correction_lateral_target_xy = (
                corridor_correction_hold_target_xy
                if (
                    correction_requires_pre_descent_controller_reserve
                    or correction_uses_high_z_hold_target
                    or correction_requires_post_descent_controller_rebuffer
                )
                else corridor_rebuffer_target[:2]
            )
            correction_plane_target_z = float(
                current_eef[2]
                if correction_requires_post_descent_controller_rebuffer
                else overhead_horizontal_z
            )
            (
                prepared_high_lateral_action,
                prepared_high_lateral_envelope,
            ) = _compiled_adaptive_high_plane_action(
                current_eef=current_eef,
                lateral_target_xy=correction_lateral_target_xy,
                overhead_horizontal_z=correction_plane_target_z,
                measured_vertical_step_progress_m=(
                    latest_vertical_step_progress_m
                ),
                overhead_guard=latest_overhead_guard,
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                native_action_spec=native_action_spec,
                expected_pair_count=expected_overhead_pair_count,
                maximum_translation_action=(
                    post_descent_lateral_max_translation_action
                ),
                plane_recovery_tolerance_m=(
                    args.position_tolerance
                    if (
                        correction_uses_high_z_hold_target
                        and not (
                            correction_requires_post_descent_controller_rebuffer
                        )
                    )
                    else None
                ),
                negative_tail_recovery_threshold_m=(
                    None
                    if correction_requires_post_descent_controller_rebuffer
                    else args.minimum_saturated_waypoint_progress
                ),
            )
        adaptive_negative_z_action_requires_buffer16 = bool(
            stage_before_action in {
                "workspace_release_diagonal",
                "overhead_corridor_descent",
            }
            and prepared_high_lateral_envelope is not None
            and prepared_high_lateral_envelope.get(
                "negative_z_action_requires_fixed_buffer16", False
            )
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
                        or adaptive_negative_z_action_requires_buffer16
                    ),
                    adaptive_high_lateral_envelope=(
                        None
                        if stage_before_action
                        in fixed_buffer_lateral_stages
                        else prepared_high_lateral_envelope
                    ),
                )
            )
        if structural_stage in {
            "right_high_lateral",
            "right_high_trailing_pass",
            "trailing_low_terminal_return",
        }:
            waypoint_key = {
                "right_high_lateral": "right_high",
                "right_high_trailing_pass": "right_trailing_high",
                "trailing_low_terminal_return": (
                    "terminal_outside_side_low"
                ),
            }[structural_stage]
            detour_target = np.asarray(
                cabinet_detour_plan["waypoints"][waypoint_key],
                dtype=float,
            )
            if structural_stage in cabinet_detour_high_lateral_stages:
                if (
                    prepared_high_lateral_action is None
                    or prepared_high_lateral_envelope is None
                ):
                    raise RuntimeError(
                        "native cabinet high detour lacks its live adaptive "
                        "high-plane action envelope"
                    )
                action = prepared_high_lateral_action
                path_control = prepared_high_lateral_envelope
                path_control_key = (
                    "compiled_adaptive_high_plane_action_envelope"
                )
            else:
                action, path_control = _fixed_z_lateral_approach_action(
                    current_eef=current_eef,
                    lateral_target_xy=detour_target[:2],
                    gripper=gripper,
                    position_action_scale=args.position_action_scale,
                    maximum_translation_action=(
                        cabinet_detour_plan[
                            "maximum_route_translation_action"
                        ]
                    ),
                )
                path_control_key = "fixed_z_lateral_path_control"
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "native_cabinet_detour_waypoint_key": waypoint_key,
                "native_cabinet_detour_target": detour_target.tolist(),
                "native_cabinet_detour_pre_guard": (
                    cabinet_detour_pre_guard
                ),
                path_control_key: path_control,
            }
        elif structural_stage == "right_trailing_vertical_descent":
            detour_target = np.asarray(
                cabinet_detour_plan["waypoints"]["right_trailing_low"],
                dtype=float,
            )
            action, path_control = _fixed_xy_vertical_approach_action(
                current_eef=current_eef,
                target_z=float(detour_target[2]),
                gripper=gripper,
                position_action_scale=args.position_action_scale,
                maximum_translation_action=(
                    cabinet_detour_plan[
                        "maximum_route_translation_action"
                    ]
                ),
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "native_cabinet_detour_waypoint_key": (
                    "right_trailing_low"
                ),
                "native_cabinet_detour_target": detour_target.tolist(),
                "native_cabinet_detour_pre_guard": (
                    cabinet_detour_pre_guard
                ),
                "fixed_xy_vertical_path_control": path_control,
            }
        elif structural_stage == "overhead_corridor_descent":
            if (
                prepared_high_lateral_action is None
                or prepared_high_lateral_envelope is None
            ):
                raise RuntimeError(
                    "corridor-holding descent was not compiled before its "
                    "live route authorization"
                )
            action = prepared_high_lateral_action
            path_control = prepared_high_lateral_envelope
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "compiled_adaptive_corridor_descent_envelope": path_control,
                "event_driven_brake_trigger_buffer_m": (
                    active_overhead_descent_brake_trigger_buffer
                ),
                "active_overhead_descent_translation_action_bound": (
                    active_overhead_descent_translation_action
                ),
                "tail_brake_lateral_entry_action_bound": (
                    post_descent_lateral_max_translation_action
                ),
                "tail_brake_lateral_entry_world_step_m": (
                    maximum_post_descent_lateral_world_step
                ),
                "tail_brake_buffer_bound_source": (
                    "existing post_descent_lateral_max_translation_action"
                ),
                "corridor_descent_hold_target_xy": (
                    corridor_descent_hold_target_xy.tolist()
                ),
                "corridor_descent_hold_outward_reserve_m": (
                    active_overhead_descent_world_step
                ),
                "corridor_descent_hold_outward_reserve_source": (
                    "active overhead-descent one-step world displacement"
                ),
            }
        elif structural_stage == "vertical_tail_brake":
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
                        maximum_post_descent_lateral_world_step
                    ),
                    lateral_target_xy=(
                        corridor_correction_hold_target_xy
                    ),
                    one_sided_outward_direction_xy=(
                        corridor_outward_direction
                    ),
                    maximum_lateral_translation_action=(
                        post_descent_lateral_max_translation_action
                    ),
                )
            )
            if action[2] <= 0.0:
                raise RuntimeError(
                    "event-driven vertical-tail brake failed to command "
                    "strictly positive Z"
                )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "brake_reason": (
                    vertical_tail_brake_reason
                    or "vertical_tail_recovery"
                ),
                "compiled_outward_xy_positive_z_tail_brake_envelope": (
                    path_control
                ),
                "retains_registered_outward_correction_drive": True,
                "active_overhead_descent_translation_action_bound": (
                    active_overhead_descent_translation_action
                ),
                "tail_brake_lateral_entry_action_bound": (
                    post_descent_lateral_max_translation_action
                ),
                "tail_brake_lateral_entry_world_step_m": (
                    maximum_post_descent_lateral_world_step
                ),
                "tail_brake_buffer_bound_source": (
                    "existing post_descent_lateral_max_translation_action"
                ),
                "active_overhead_descent_brake_trigger_buffer_m": (
                    active_overhead_descent_brake_trigger_buffer
                ),
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
                        maximum_post_descent_lateral_world_step
                        if lateral_resume_stage
                        in fixed_buffer_lateral_stages
                        else maximum_controller_world_step
                    ),
                    lateral_target_xy=(
                        corridor_correction_hold_target_xy
                        if lateral_resume_stage
                        == "overhead_post_descent_corridor_lateral"
                        else None
                    ),
                    one_sided_outward_direction_xy=(
                        corridor_outward_direction
                        if lateral_resume_stage
                        == "overhead_post_descent_corridor_lateral"
                        else None
                    ),
                    maximum_lateral_translation_action=(
                        post_descent_lateral_max_translation_action
                        if lateral_resume_stage
                        == "overhead_post_descent_corridor_lateral"
                        else None
                    ),
                )
            )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "compiled_adaptive_lateral_rebuffer_envelope": path_control,
                "retains_registered_outward_correction_drive": bool(
                    path_control["retain_outward_lateral_drive"]
                ),
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
                "compiled_adaptive_high_plane_action_envelope": (
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
        elif structural_stage == "workspace_release_diagonal":
            if (
                prepared_high_lateral_action is None
                or prepared_high_lateral_envelope is None
            ):
                raise RuntimeError(
                    "workspace-release action lacks its live compiled envelope"
                )
            action = prepared_high_lateral_action
            path_control = prepared_high_lateral_envelope
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "compiled_adaptive_workspace_release_envelope": path_control,
                "lateral_route_phase": path_control["motion_kind"],
                "pre_action_measured_vertical_step_progress_m": (
                    latest_vertical_step_progress_m
                ),
                "fixed_buffer16_used_for_action_authorization": False,
                "pre_action_overhead_guard": latest_overhead_guard,
            }
        elif structural_stage == "overhead_post_descent_corridor_lateral":
            if (
                prepared_high_lateral_action is None
                or prepared_high_lateral_envelope is None
            ):
                raise RuntimeError(
                    "post-descent correction lacks its live compiled "
                    "XY/nonnegative-Z plane-hold envelope"
                )
            action = prepared_high_lateral_action
            path_control = prepared_high_lateral_envelope
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "compiled_adaptive_post_descent_plane_hold_envelope": (
                    path_control
                ),
                "lateral_route_phase": (
                    "post_descent_xy_plus_nonnegative_z_plane_hold_"
                    "correction"
                ),
                "corridor_rebuffer_target": (
                    corridor_rebuffer_target.tolist()
                ),
                "correction_controller_hold_target_xy": (
                    corridor_correction_hold_target_xy.tolist()
                ),
                "active_correction_lateral_target_xy": (
                    correction_lateral_target_xy.tolist()
                ),
                "correction_requires_pre_descent_controller_reserve": (
                    correction_requires_pre_descent_controller_reserve
                ),
                "correction_requires_post_descent_controller_rebuffer": (
                    correction_requires_post_descent_controller_rebuffer
                ),
                "correction_controller_reserve_error_m": (
                    correction_controller_reserve_error_m
                ),
                "correction_controller_handoff_tolerance_m": (
                    high_z_controller_handoff_tolerance
                ),
                "correction_rebuffer_holds_current_z": (
                    correction_requires_post_descent_controller_rebuffer
                ),
                "correction_uses_high_z_hold_target": (
                    correction_uses_high_z_hold_target
                ),
                "correction_controller_outward_reserve_m": (
                    maximum_post_descent_lateral_world_step
                ),
                "correction_controller_outward_reserve_source": (
                    "existing post-descent one-step world displacement"
                ),
                "correction_per_action_maximum_world_step_m": (
                    maximum_post_descent_lateral_world_step
                ),
                "high_z_plane_recovery_tolerance_m": (
                    float(args.position_tolerance)
                    if correction_uses_high_z_hold_target
                    else None
                ),
                "high_z_plane_recovery_tolerance_source": (
                    "unchanged formal position_tolerance"
                ),
                "formal_corridor_acceptance_target_unchanged": True,
                "corridor_rebuffer_clearance_m": (
                    corridor_rebuffer_clearance
                ),
                "corridor_rebuffer_acceptance_clearance_m": (
                    corridor_rebuffer_acceptance_clearance
                ),
                "overhead_horizontal_z_m": float(
                    correction_plane_target_z
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
            reserve_recovery_evidence = (
                _vertical_corridor_reserve_recovery_evidence(
                    live_clearance_m=float(
                        pre_action_guard[
                            "minimum_outside_clearance_m"
                        ]
                    ),
                    recovery_entry_clearance_m=(
                        vertical_corridor_reserve_recovery_entry_clearance
                    ),
                    recovery_exit_clearance_m=(
                        vertical_corridor_reserve_recovery_exit_clearance
                    ),
                    strict_corridor_entry_clearance_m=float(
                        vertical_staging_corridor[
                            "strict_corridor_entry_clearance_m"
                        ]
                    ),
                    latest_outward_step_progress_m=(
                        latest_outward_step_progress_m
                    ),
                    latest_vertical_step_progress_m=(
                        latest_vertical_step_progress_m
                    ),
                    compiled_tail_brake_buffer_accepted=bool(
                        latest_overhead_lateral_buffer["accepted"]
                    ),
                    recovery_active_before_decision=(
                        vertical_corridor_reserve_recovery_active
                    ),
                )
            )
            vertical_corridor_reserve_recovery_active = bool(
                reserve_recovery_evidence[
                    "recovery_active_after_decision"
                ]
            )
            reserve_recovery_phase_evidence = (
                _vertical_corridor_reserve_recovery_phase_evidence(
                    recovery_evidence=reserve_recovery_evidence,
                    phase_before_decision=(
                        vertical_corridor_reserve_recovery_phase
                    ),
                )
            )
            vertical_corridor_reserve_recovery_phase = (
                reserve_recovery_phase_evidence[
                    "phase_after_decision"
                ]
            )
            if (
                reserve_recovery_evidence["entered_recovery"]
                or reserve_recovery_evidence["exit_accepted"]
                or reserve_recovery_phase_evidence[
                    "phase_before_decision"
                ]
                != reserve_recovery_phase_evidence[
                    "phase_after_decision"
                ]
            ):
                vertical_corridor_reserve_recovery_events.append(
                    {
                        "guard_step": int(guard_step),
                        **reserve_recovery_evidence,
                        **reserve_recovery_phase_evidence,
                    }
                )
            reserve_recovery_vertical_brake_required = bool(
                vertical_corridor_reserve_recovery_phase
                in {"vertical_brake", "exit_brake"}
            )
            vertical_corridor_compiled_tail_brake_active = bool(
                reserve_recovery_vertical_brake_required
            )
            reserve_recovery_outward_only_active = bool(
                vertical_corridor_reserve_recovery_phase
                == "outward_restore"
            )
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
            vertical_corridor_control_target = np.asarray(
                corridor_side_target, dtype=float
            ).copy()
            vertical_corridor_control_target[:2] = (
                corridor_correction_hold_target_xy
                if reserve_recovery_outward_only_active
                else vertical_corridor_balanced_hold_target_xy
            )
            if vertical_corridor_compiled_tail_brake_active:
                latest_overhead_guard = _live_compiled_overhead_guard(
                    env, overhead_staging_geometry
                )
                overhead_guard_checks += 1
                latest_overhead_lateral_buffer = (
                    _overhead_lateral_buffer_evidence(
                        latest_overhead_guard,
                        worst_case_controller_world_step_m=(
                            maximum_post_descent_lateral_world_step
                        ),
                    )
                )
                if not latest_overhead_guard["accepted"]:
                    raise RuntimeError(
                        "vertical-corridor reserve recovery lacks its live "
                        "compiled base-overhead tail-brake proof: "
                        f"guard_step={guard_step} overhead_guard="
                        f"{json.dumps(latest_overhead_guard, sort_keys=True)} "
                        f"lateral_buffer="
                        f"{json.dumps(latest_overhead_lateral_buffer, sort_keys=True)}"
                    )
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
                            maximum_post_descent_lateral_world_step
                        ),
                        lateral_target_xy=(
                            corridor_correction_hold_target_xy
                        ),
                        one_sided_outward_direction_xy=(
                            corridor_outward_direction
                        ),
                        maximum_lateral_translation_action=(
                            post_descent_lateral_max_translation_action
                        ),
                    )
                )
                if action[2] <= 0.0:
                    raise RuntimeError(
                        "vertical-corridor compiled reserve brake failed "
                        "to command strictly positive Z"
                    )
            else:
                action, path_control = (
                    _constraint_prioritized_outside_descent_action(
                        current_eef=current_eef,
                        outside_side_target=(
                            vertical_corridor_control_target
                        ),
                        outward_direction_xy=geometry[
                            "outward_direction_xy"
                        ],
                        maximum_descent_m=(
                            0.0
                            if vertical_corridor_reserve_recovery_active
                            else maximum_descent
                        ),
                        gripper=gripper,
                        position_action_scale=args.position_action_scale,
                        maximum_translation_action=(
                            vertical_corridor_descent_max_translation_action
                        ),
                    )
                )
            feedback = {
                "mode": structural_stage,
                "action": action.tolist(),
                "descent_path_control": path_control,
                "formal_corridor_side_target": (
                    corridor_side_target.tolist()
                ),
                "active_vertical_corridor_control_target": (
                    vertical_corridor_control_target.tolist()
                ),
                "balanced_outward_controller_hold_active": bool(
                    not vertical_corridor_reserve_recovery_active
                ),
                "balanced_outward_controller_hold_world_step_m": (
                    vertical_corridor_balanced_hold_world_step
                ),
                "balanced_outward_controller_hold_source": (
                    "equal XY/Z norm allocation derived from the existing "
                    "vertical-corridor translation-action bound"
                ),
                "reserve_recovery_evidence": reserve_recovery_evidence,
                "reserve_recovery_phase_evidence": (
                    reserve_recovery_phase_evidence
                ),
                "reserve_recovery_active": (
                    vertical_corridor_reserve_recovery_active
                ),
                "reserve_recovery_phase": (
                    vertical_corridor_reserve_recovery_phase
                ),
                "reserve_recovery_vertical_brake_required": (
                    reserve_recovery_vertical_brake_required
                ),
                "reserve_recovery_outward_only_active": (
                    reserve_recovery_outward_only_active
                ),
                "reserve_recovery_outward_only_target_source": (
                    "existing 8 mm post-descent correction-hold target"
                    if reserve_recovery_outward_only_active
                    else None
                ),
                "compiled_tail_brake_reused_for_reserve_recovery": (
                    vertical_corridor_compiled_tail_brake_active
                ),
                "compiled_reserve_recovery_tail_brake_envelope": (
                    path_control
                    if vertical_corridor_compiled_tail_brake_active
                    else None
                ),
                "pre_action_reserve_recovery_overhead_guard": (
                    latest_overhead_guard
                    if vertical_corridor_compiled_tail_brake_active
                    else None
                ),
                "pre_action_reserve_recovery_lateral_buffer": (
                    _overhead_lateral_buffer_frame_summary(
                        latest_overhead_lateral_buffer
                    )
                    if vertical_corridor_compiled_tail_brake_active
                    else None
                ),
                "negative_z_descent_suspended_for_reserve_recovery": bool(
                    vertical_corridor_reserve_recovery_active
                    and action[2] >= 0.0
                ),
                "formal_corridor_target_unchanged": True,
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
                        vertical_corridor_descent_max_translation_action
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
                maximum_translation_action=structural_max_translation_action,
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
        cabinet_detour_post_guard = None
        if cabinet_detour_plan is not None and stage_before_action in set(
            cabinet_detour_plan["route_order"]
        ):
            cabinet_detour_post_guard = _live_native_cabinet_detour_guard(
                env,
                eef_position=after_eef,
                plan=cabinet_detour_plan,
                stage=stage_before_action,
            )
            feedback["native_cabinet_detour_post_guard"] = (
                cabinet_detour_post_guard
            )
        if (
            stage_before_action in overhead_route_stages
            or vertical_corridor_compiled_tail_brake_active
        ):
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
                        maximum_post_descent_lateral_world_step
                        if (
                            vertical_corridor_compiled_tail_brake_active
                            or stage_before_action
                            in fixed_buffer_lateral_stages
                            or (
                                stage_before_action
                                == "lateral_rebuffer_brake"
                                and lateral_resume_stage
                                in fixed_buffer_lateral_stages
                            )
                        )
                        else (
                            maximum_post_descent_lateral_world_step
                            if stage_before_action
                            == "vertical_tail_brake"
                            else (
                                active_overhead_descent_world_step
                                if stage_before_action
                                == "overhead_corridor_descent"
                                else maximum_controller_world_step
                            )
                        )
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
        latest_outward_step_progress_m = float(
            current_step_response["eef_outward_step_progress_m"]
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
            if stage_before_action == "overhead_high_corridor_lateral":
                lateral_feedback_target = high_lateral_prebuffer_target[:2]
            elif (
                stage_before_action
                == "overhead_post_descent_corridor_lateral"
            ):
                lateral_feedback_target = correction_lateral_target_xy
            else:
                lateral_feedback_target = corridor_high_target[:2]
            feedback["corridor_lateral_error_m"] = float(
                np.linalg.norm(after_eef[:2] - lateral_feedback_target)
            )
            feedback["stage_lateral_target_xy"] = (
                lateral_feedback_target.tolist()
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
            outside_high_entry = _overhead_outside_high_entry_evidence(
                current_eef=after_eef,
                outside_high_target=np.asarray(
                    outside_high_target, dtype=float
                ),
                high_lateral_target=high_lateral_prebuffer_target,
                overhead_horizontal_z=overhead_horizontal_z,
                overhead_guard=latest_overhead_guard,
                position_tolerance=args.position_tolerance,
            )
            feedback["outside_high_entry_after_high_lateral"] = (
                outside_high_entry
            )
            current_high_plane_observation = {
                "before_eef": current_eef.copy(),
                "after_eef": after_eef.copy(),
                "action": action.copy(),
                "high_plane_envelope": path_control,
                "pre_overhead_guard": feedback[
                    "pre_action_overhead_guard"
                ],
                "post_overhead_guard": latest_overhead_guard,
                "before_outside_guard": pre_action_guard,
                "after_outside_guard": latest_outside_side_guard,
                "step_response": current_step_response,
            }
            high_plane_workspace_saturation_observations.append(
                current_high_plane_observation
            )
            high_plane_workspace_saturation_observations[:] = (
                high_plane_workspace_saturation_observations[
                    -int(args.push_tracking_steps) :
                ]
            )
            workspace_saturation = (
                _high_plane_native_workspace_saturation_evidence(
                    observations=(
                        high_plane_workspace_saturation_observations
                    ),
                    initial_eef=initial_eef,
                    high_lateral_target=high_lateral_prebuffer_target,
                    native_outside_high_target=(
                        overhead_outside_high_target
                    ),
                    outward_direction_xy=geometry[
                        "outward_direction_xy"
                    ],
                    position_action_scale=args.position_action_scale,
                    position_tolerance=args.position_tolerance,
                    progress_epsilon=(
                        args.minimum_saturated_waypoint_progress
                    ),
                    required_window_frames=args.push_tracking_steps,
                    native_action_spec=native_action_spec,
                    expected_pair_count=expected_overhead_pair_count,
                )
            )
            feedback["high_plane_native_workspace_saturation"] = (
                workspace_saturation
            )
            native_boundary_crossing = (
                _high_plane_native_boundary_crossing_evidence(
                    observation=current_high_plane_observation,
                    high_lateral_target=high_lateral_prebuffer_target,
                    native_outside_high_target=(
                        overhead_outside_high_target
                    ),
                    outward_direction_xy=geometry[
                        "outward_direction_xy"
                    ],
                    expected_pair_count=expected_overhead_pair_count,
                )
            )
            feedback["high_plane_native_boundary_crossing"] = (
                native_boundary_crossing
            )
            if outside_high_entry["accepted"]:
                structural_stage = "workspace_release_diagonal"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "registered_corridor_high_tolerance_complete_to_"
                            "workspace_release_diagonal"
                        ),
                        **outside_high_entry,
                    }
                )
            elif native_boundary_crossing["accepted"]:
                structural_stage = "workspace_release_diagonal"
                structural_seek_context[
                    "accepted_native_high_boundary_crossing"
                ] = native_boundary_crossing
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "strict_native_high_boundary_crossing_to_"
                            "adaptive_workspace_release"
                        ),
                        **native_boundary_crossing,
                    }
                )
            elif workspace_saturation["accepted"]:
                structural_stage = "workspace_release_diagonal"
                structural_seek_context[
                    "accepted_native_high_workspace_saturation_boundary"
                ] = workspace_saturation
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "proved_native_high_workspace_saturation_to_"
                            "adaptive_workspace_release"
                        ),
                        **workspace_saturation,
                    }
                )
        elif stage_before_action == "workspace_release_diagonal":
            workspace_release_envelope = feedback[
                "compiled_adaptive_workspace_release_envelope"
            ]
            if workspace_release_envelope[
                "event_driven_positive_z_inertial_recovery"
            ]:
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "workspace_release_event_driven_positive_z_"
                            "inertial_recovery"
                        ),
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "commanded_positive_z_recovery_world_delta_m": (
                            workspace_release_envelope[
                                "commanded_positive_z_recovery_world_delta_m"
                            ]
                        ),
                    }
                )
            if measured_vertical_step_progress_m < 0.0:
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": "workspace_release_negative_z_progress",
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "post_action_base8_accepted": bool(
                            latest_overhead_guard["accepted"]
                        ),
                    }
                )
            corridor_entry_after_action = _overhead_corridor_entry_evidence(
                    current_eef=after_eef,
                    corridor_high_target=corridor_high_target,
                    outside_side_guard=latest_outside_side_guard,
                    overhead_guard=latest_overhead_guard,
                    overhead_lateral_buffer=(
                        latest_overhead_lateral_buffer
                    ),
                    position_tolerance=args.position_tolerance,
                    strict_corridor_entry_clearance_m=(
                        vertical_staging_corridor["corridor_clearance_m"]
                    ),
                    require_lateral_buffer=False,
                    minimum_eef_z=None,
            )
            feedback["corridor_entry_after_workspace_release"] = (
                corridor_entry_after_action
            )
            workspace_controller_handoff = (
                _high_z_controller_handoff_evidence(
                    current_eef=after_eef,
                    outside_side_guard=latest_outside_side_guard,
                    overhead_guard=latest_overhead_guard,
                    overhead_lateral_buffer=(
                        latest_overhead_lateral_buffer
                    ),
                )
            )
            feedback["controller_handoff_after_workspace_release"] = (
                workspace_controller_handoff
            )
            if not workspace_release_envelope[
                "event_driven_positive_z_inertial_recovery"
            ]:
                if workspace_controller_handoff["accepted"]:
                    structural_stage = "overhead_corridor_descent"
                    vertical_tail_events.append(
                        {
                            "guard_step": int(guard_step),
                            "event": (
                                "workspace_release_reached_controller_"
                                "reserve_to_adaptive_overhead_descent"
                            ),
                            "formal_corridor_entry": (
                                corridor_entry_after_action
                            ),
                            "controller_handoff": (
                                workspace_controller_handoff
                            ),
                        }
                    )
                elif corridor_entry_after_action["accepted"]:
                    structural_stage = (
                        "overhead_post_descent_corridor_lateral"
                    )
                    overhead_horizontal_z = float(after_eef[2])
                    vertical_tail_events.append(
                        {
                            "guard_step": int(guard_step),
                            "event": (
                                "workspace_release_reached_formal_corridor_"
                                "but_controller_reserve_requires_plane_hold"
                            ),
                            "formal_corridor_entry": (
                                corridor_entry_after_action
                            ),
                            "controller_handoff": (
                                workspace_controller_handoff
                            ),
                        }
                    )
        elif stage_before_action == "overhead_corridor_descent":
            descent_corridor_entry_after_action = (
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
                    minimum_eef_z=None,
                )
            )
            feedback["descent_corridor_entry_after_action"] = (
                descent_corridor_entry_after_action
            )
            descent_corridor_lateral_violations = {
                "corridor_xy_tolerance_not_met",
                "outside_corridor_entry_clearance_not_met",
            }.intersection(
                descent_corridor_entry_after_action["violations"]
            )
            full_corridor_clearance_retained = bool(
                descent_corridor_entry_after_action[
                    "minimum_outside_clearance_m"
                ]
                > corridor_rebuffer_acceptance_clearance
            )
            feedback["full_corridor_clearance_retained_after_descent"] = (
                full_corridor_clearance_retained
            )
            if (
                not full_corridor_clearance_retained
                and current_step_response[
                    "eef_outward_step_progress_m"
                ]
                < -float(args.minimum_saturated_waypoint_progress)
            ):
                descent_corridor_lateral_violations.add(
                    "eef_inward_step_during_corridor_holding_descent"
                )
            if (
                not full_corridor_clearance_retained
                and current_step_response[
                    "outside_clearance_step_progress_m"
                ]
                < -float(args.minimum_saturated_waypoint_progress)
            ):
                descent_corridor_lateral_violations.add(
                    "outside_clearance_decreased_during_corridor_holding_"
                    "descent"
                )
            if descent_corridor_lateral_violations:
                vertical_tail_brake_reason = "lateral_drift"
                structural_stage = "vertical_tail_brake"
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": (
                            "corridor_entry_lateral_guard_failed_to_high_"
                            "brake_and_correction"
                        ),
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "corridor_lateral_error_m": (
                            descent_corridor_entry_after_action[
                                "corridor_lateral_error_m"
                            ]
                        ),
                        "lateral_guard_violations": sorted(
                            descent_corridor_lateral_violations
                        ),
                        "overhead_staging_z_m": overhead_staging_z,
                        "remaining_z_above_staging_m": float(
                            after_eef[2] - overhead_staging_z
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                        "corridor_entry_after_descent": (
                            descent_corridor_entry_after_action
                        ),
                        "descent_step_response": current_step_response,
                        "full_corridor_clearance_retained_before_brake": (
                            full_corridor_clearance_retained
                        ),
                    }
                )
            elif after_eef[2] <= (
                overhead_staging_z
                + active_overhead_descent_brake_trigger_buffer
            ):
                vertical_tail_brake_reason = "staging_height"
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
                        "overhead_staging_z_m": overhead_staging_z,
                        "brake_trigger_buffer_m": (
                            active_overhead_descent_brake_trigger_buffer
                        ),
                        "active_translation_action_bound": (
                            active_overhead_descent_translation_action
                        ),
                        "remaining_z_above_staging_m": float(
                            after_eef[2] - overhead_staging_z
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
                tail_brake_formal_corridor_entry = (
                    _overhead_corridor_entry_evidence(
                        current_eef=after_eef,
                        corridor_high_target=corridor_rebuffer_target,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                        position_tolerance=args.position_tolerance,
                        strict_corridor_entry_clearance_m=(
                            corridor_rebuffer_acceptance_clearance
                        ),
                    )
                )
                feedback["tail_brake_formal_corridor_entry"] = (
                    tail_brake_formal_corridor_entry
                )
                tail_brake_controller_handoff = (
                    _high_z_controller_handoff_evidence(
                        current_eef=after_eef,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                    )
                )
                feedback["tail_brake_controller_handoff_diagnostic"] = (
                    tail_brake_controller_handoff
                )
                previous_active_translation_action = float(
                    active_overhead_descent_translation_action
                )
                brake_reason_before_recovery = (
                    vertical_tail_brake_reason
                    or "vertical_tail_recovery"
                )
                if after_eef[2] > (
                    overhead_staging_z + args.position_tolerance
                ):
                    active_overhead_descent_translation_action = float(
                        max(
                            structural_max_translation_action,
                            tail_recovery_descent_translation_action_floor,
                            previous_active_translation_action / 2.0,
                        )
                    )
                    active_overhead_descent_world_step = float(
                        args.position_action_scale
                        * active_overhead_descent_translation_action
                    )
                    active_overhead_descent_brake_trigger_buffer = float(
                        2.0 * active_overhead_descent_world_step
                    )
                within_tail_handoff_band = bool(
                    after_eef[2]
                    <= overhead_staging_z
                    + active_overhead_descent_brake_trigger_buffer
                )
                if tail_brake_formal_corridor_entry["accepted"]:
                    if within_tail_handoff_band:
                        structural_stage = "vertical_corridor_descent"
                        recovered_event = (
                            "tail_brake_formal_corridor_passed_inside_brake_"
                            "band_to_vertical_corridor"
                        )
                    else:
                        structural_stage = "overhead_corridor_descent"
                        recovered_event = (
                            "tail_brake_formal_corridor_passed_above_brake_"
                            "band_to_bounded_overhead_descent"
                        )
                else:
                    structural_stage = (
                        "overhead_post_descent_corridor_lateral"
                    )
                    overhead_horizontal_z = float(after_eef[2])
                    recovered_event = (
                        "tail_brake_recovered_to_formal_corridor_correction"
                    )
                vertical_tail_brake_reason = None
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": recovered_event,
                        "brake_reason": brake_reason_before_recovery,
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "overhead_staging_z_m": overhead_staging_z,
                        "remaining_z_above_staging_m": float(
                            after_eef[2] - overhead_staging_z
                        ),
                        "previous_translation_action_bound": (
                            previous_active_translation_action
                        ),
                        "next_translation_action_bound": (
                            active_overhead_descent_translation_action
                        ),
                        "translation_action_bound_reduced": bool(
                            active_overhead_descent_translation_action
                            < previous_active_translation_action
                        ),
                        "next_brake_trigger_buffer_m": (
                            active_overhead_descent_brake_trigger_buffer
                        ),
                        "within_tail_handoff_band": (
                            within_tail_handoff_band
                        ),
                        "tail_handoff_band_upper_z_m": float(
                            overhead_staging_z
                            + active_overhead_descent_brake_trigger_buffer
                        ),
                        "tail_handoff_band_source": (
                            "the existing event-driven overhead-descent "
                            "brake-trigger buffer"
                        ),
                        "tail_recovery_descent_translation_action_floor": (
                            tail_recovery_descent_translation_action_floor
                        ),
                        "tail_recovery_descent_floor_source": (
                            "the existing post-descent lateral action bound"
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                        "formal_corridor_entry": (
                            tail_brake_formal_corridor_entry
                        ),
                        "controller_handoff_diagnostic_only": (
                            tail_brake_controller_handoff
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
                lateral_rebuffer_formal_corridor_entry = (
                    _overhead_corridor_entry_evidence(
                        current_eef=after_eef,
                        corridor_high_target=corridor_rebuffer_target,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                        position_tolerance=args.position_tolerance,
                        strict_corridor_entry_clearance_m=(
                            corridor_rebuffer_acceptance_clearance
                        ),
                    )
                )
                feedback["lateral_rebuffer_formal_corridor_entry"] = (
                    lateral_rebuffer_formal_corridor_entry
                )
                lateral_rebuffer_controller_handoff = (
                    _high_z_controller_handoff_evidence(
                        current_eef=after_eef,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                    )
                )
                feedback["lateral_rebuffer_shifted_target_diagnostic"] = (
                    lateral_rebuffer_controller_handoff
                )
                if lateral_rebuffer_formal_corridor_entry["accepted"]:
                    above_tail_handoff_band = bool(
                        after_eef[2]
                        > overhead_staging_z
                        + active_overhead_descent_brake_trigger_buffer
                    )
                    structural_stage = (
                        "overhead_corridor_descent"
                        if above_tail_handoff_band
                        else "vertical_corridor_descent"
                    )
                    recovered_event = (
                        "lateral_rebuffer_formal_corridor_passed_to_bounded_"
                        "overhead_descent"
                        if above_tail_handoff_band
                        else "lateral_rebuffer_formal_corridor_passed_to_"
                        "vertical_corridor"
                    )
                    lateral_resume_stage = None
                else:
                    structural_stage = lateral_resume_stage
                    overhead_horizontal_z = float(after_eef[2])
                    recovered_event = (
                        "lateral_rebuffer_recovered_to_shifted_target_"
                        "correction_fallback"
                    )
                vertical_tail_events.append(
                    {
                        "guard_step": int(guard_step),
                        "event": recovered_event,
                        "measured_vertical_step_progress_m": (
                            measured_vertical_step_progress_m
                        ),
                        "minimum_lateral_entry_buffer_surplus_m": (
                            latest_overhead_lateral_buffer[
                                "minimum_lateral_entry_buffer_surplus_m"
                            ]
                        ),
                        "formal_corridor_entry": (
                            lateral_rebuffer_formal_corridor_entry
                        ),
                        "shifted_controller_target_required_for_next_stage": (
                            not lateral_rebuffer_formal_corridor_entry[
                                "accepted"
                            ]
                        ),
                        "shifted_controller_target_diagnostic": (
                            lateral_rebuffer_controller_handoff
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
                        corridor_high_target=corridor_rebuffer_target,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                        position_tolerance=args.position_tolerance,
                        strict_corridor_entry_clearance_m=(
                            corridor_rebuffer_acceptance_clearance
                        ),
                    )
                )
                feedback["corridor_entry_after_drift_correction"] = (
                    corridor_entry_after_action
                )
                correction_controller_handoff = (
                    _high_z_controller_handoff_evidence(
                        current_eef=after_eef,
                        outside_side_guard=latest_outside_side_guard,
                        overhead_guard=latest_overhead_guard,
                        overhead_lateral_buffer=(
                            latest_overhead_lateral_buffer
                        ),
                    )
                )
                feedback["correction_controller_handoff"] = (
                    correction_controller_handoff
                )
                post_descent_vertical_tail_handoff_accepted = bool(
                    measured_vertical_step_progress_m
                    >= -float(args.minimum_saturated_waypoint_progress)
                )
                feedback["post_descent_vertical_tail_handoff_gate"] = {
                    "accepted": (
                        post_descent_vertical_tail_handoff_accepted
                    ),
                    "measured_vertical_step_progress_m": float(
                        measured_vertical_step_progress_m
                    ),
                    "minimum_accepted_vertical_step_progress_m": (
                        -float(args.minimum_saturated_waypoint_progress)
                    ),
                    "threshold_source": (
                        "existing minimum_saturated_waypoint_progress"
                    ),
                    "formal_corridor_acceptance_unchanged": True,
                }
                above_tail_handoff_band = bool(
                    after_eef[2]
                    > overhead_staging_z
                    + active_overhead_descent_brake_trigger_buffer
                )
                if (
                    above_tail_handoff_band
                    and (
                        (
                            correction_requires_pre_descent_controller_reserve
                            and correction_controller_handoff["accepted"]
                        )
                        or (
                            not correction_requires_pre_descent_controller_reserve
                            and correction_controller_handoff["accepted"]
                            and corridor_entry_after_action["accepted"]
                            and post_descent_vertical_tail_handoff_accepted
                        )
                    )
                ):
                    structural_stage = "overhead_corridor_descent"
                    correction_complete_event = (
                        "pre_descent_controller_reserve_complete_to_"
                        "bounded_descent"
                        if correction_requires_pre_descent_controller_reserve
                        else "post_descent_formal_corridor_entry_complete_to_"
                        "bounded_descent"
                    )
                    vertical_tail_brake_reason = None
                    vertical_tail_events.append(
                        {
                            "guard_step": int(guard_step),
                            "event": correction_complete_event,
                            "overhead_staging_z_m": overhead_staging_z,
                            "remaining_z_above_staging_m": float(
                                after_eef[2] - overhead_staging_z
                            ),
                            "formal_corridor_entry": (
                                corridor_entry_after_action
                            ),
                            "controller_handoff_required": (
                                True
                            ),
                            "controller_handoff_evidence": (
                                correction_controller_handoff
                            ),
                        }
                    )
                elif (
                    not above_tail_handoff_band
                    and corridor_entry_after_action["accepted"]
                    and correction_controller_handoff["accepted"]
                    and post_descent_vertical_tail_handoff_accepted
                ):
                    structural_stage = "vertical_corridor_descent"
                    correction_complete_event = (
                        "staging_corridor_correction_complete_to_"
                        "vertical_corridor"
                    )
                    vertical_tail_brake_reason = None
                    vertical_tail_events.append(
                        {
                            "guard_step": int(guard_step),
                            "event": correction_complete_event,
                            "overhead_staging_z_m": overhead_staging_z,
                            "remaining_z_above_staging_m": float(
                                after_eef[2] - overhead_staging_z
                            ),
                            **corridor_entry_after_action,
                        }
                    )
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
                    if (
                        stage_before_action in overhead_route_stages
                        or vertical_corridor_compiled_tail_brake_active
                    )
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
                    if (
                        stage_before_action in overhead_route_stages
                        or vertical_corridor_compiled_tail_brake_active
                    )
                    else {}
                ),
                **(
                    {
                        "native_cabinet_detour_guard": (
                            cabinet_detour_post_guard
                        )
                    }
                    if cabinet_detour_post_guard is not None
                    else {}
                ),
            },
        )
        structural_violations = []
        if (
            cabinet_detour_post_guard is not None
            and not cabinet_detour_post_guard["accepted"]
        ):
            structural_violations.extend(
                cabinet_detour_post_guard["violations"]
            )
        if cabinet_detour_post_guard is not None and (
            latest_outside_side_guard[
                "finger_table_vertical_clearance_m"
            ]
            < latest_outside_side_guard[
                "required_finger_table_clearance_m"
            ]
        ):
            structural_violations.append(
                "strict_finger_table_clearance_lost_in_cabinet_detour"
            )
        required_clearance = float(
            latest_outside_side_guard[
                "required_outside_clearance_m"
            ]
        )
        if (
            (
                stage_before_action in overhead_route_stages
                or vertical_corridor_compiled_tail_brake_active
            )
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
            f"source={source} max_steps={structural_waypoint_budget} "
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
    final_cabinet_detour_guard = None
    cabinet_detour_completion = None
    if cabinet_detour_plan is not None:
        final_cabinet_detour_guard = _live_native_cabinet_detour_guard(
            env,
            eef_position=np.asarray(
                rollout.obs["robot0_eef_pos"], dtype=float
            ),
            plan=cabinet_detour_plan,
            stage="trailing_low_terminal_return",
        )
        if not final_cabinet_detour_guard["accepted"]:
            raise RuntimeError(
                "native cabinet detour terminal live guard failed: "
                f"source={source} guard="
                f"{json.dumps(final_cabinet_detour_guard, sort_keys=True)}"
            )
    outside_side_sample = capture("outside_side", 0, False, True)
    outside_side_sample["outside_side_guard"] = (
        final_outside_side_guard
    )
    if final_cabinet_detour_guard is not None:
        outside_side_sample["native_cabinet_detour_guard"] = (
            final_cabinet_detour_guard
        )
        if not isinstance(controller_live_diagnostic, dict):
            raise RuntimeError(
                "executed native cabinet detour lacks diagnostic context"
            )
        cabinet_detour_completion = (
            _record_native_cabinet_detour_completion(
                diagnostic_context=controller_live_diagnostic,
                source=source,
                final_guard=final_cabinet_detour_guard,
                stage_action_counts=structural_stage_action_counts,
                used_steps=outside_side_motion_steps,
                remaining_steps=(
                    structural_waypoint_budget
                    - outside_side_motion_steps
                ),
            )
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
        "authorized_native_cabinet_detour": cabinet_detour_plan,
        "native_cabinet_detour_completion": cabinet_detour_completion,
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
        "latest_measured_outward_step_progress_m": float(
            latest_outward_step_progress_m
        ),
        "vertical_corridor_reserve_recovery_events": (
            vertical_corridor_reserve_recovery_events
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
            "maximum_steps": structural_waypoint_budget,
            "used_steps": int(outside_side_motion_steps),
            "remaining_steps": int(
                structural_waypoint_budget - outside_side_motion_steps
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
    if (
        not np.isfinite(
            args.structural_near_plate_max_translation_action
        )
        or not (
            0
            < args.structural_near_plate_max_translation_action
            <= args.plate_contact_seek_max_translation_action
        )
    ):
        raise ValueError(
            "--structural_near_plate_max_translation_action must be positive "
            "and no greater than --plate_contact_seek_max_translation_action"
        )
    if (
        not np.isfinite(args.overhead_descent_max_translation_action)
        or not (
            args.structural_near_plate_max_translation_action
            < args.overhead_descent_max_translation_action
            <= 1.0
        )
    ):
        raise ValueError(
            "--overhead_descent_max_translation_action must be greater than "
            "the near-plate bound and no greater than 1.0"
        )
    if (
        not np.isfinite(
            args.vertical_corridor_descent_max_translation_action
        )
        or not (
            args.structural_near_plate_max_translation_action
            < args.vertical_corridor_descent_max_translation_action
            <= args.plate_contact_seek_max_translation_action
        )
    ):
        raise ValueError(
            "--vertical_corridor_descent_max_translation_action must be "
            "greater than the lateral near-plate bound and no greater than "
            "the contact-seek bound"
        )
    if (
        not np.isfinite(args.post_descent_lateral_max_translation_action)
        or not (
            args.structural_near_plate_max_translation_action
            < args.post_descent_lateral_max_translation_action
            <= args.plate_contact_seek_max_translation_action
        )
    ):
        raise ValueError(
            "--post_descent_lateral_max_translation_action must be greater "
            "than the corridor-entry lateral bound and no greater than the "
            "contact-seek bound"
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
    if args.push_tracking_steps < 1:
        raise ValueError("--push_tracking_steps must be positive")
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
            selection_mode="native_plus_x_front_corridor",
        )
        selected_contact_candidate = _select_native_plus_x_front_candidate(
            candidate_geometry
        )
        center_approach_target = np.asarray(
            selected_contact_candidate["center_high_target"],
            dtype=float,
        )
        wrist_yaw_execution = None
        realized_contact_candidate = None
        outside_high_target = None
        outside_side_target = None
        contact_target = None
        compiled_side_contact_geometry = None

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
                "wrist_yaw_execution": wrist_yaw_execution,
                "realized_contact_candidate": realized_contact_candidate,
                "selected_contact_line_xy": (
                    None
                    if contact_target is None
                    else contact_target[:2].tolist()
                ),
                "center_approach_target": center_approach_target.tolist(),
                "outside_high_target": (
                    None
                    if outside_high_target is None
                    else outside_high_target.tolist()
                ),
                "outside_side_target": (
                    None
                    if outside_side_target is None
                    else outside_side_target.tolist()
                ),
                "side_contact_target": (
                    None
                    if contact_target is None
                    else contact_target.tolist()
                ),
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
        # side-contact path.  First reach center-high, retain the unchanged
        # native wrist orientation, and directly recompile the low-skew +X
        # candidate.  That route remains in front of the cabinet; the existing
        # per-action overhead, outside-rim, table, stability, and empty-contact
        # gates still authorize every structural action before contact.
        rollout.move(
            center_approach_target,
            pusher_open_sign,
            "task",
            diagnostics=plate_diagnostics,
        )
        wrist_yaw_execution = _prepare_native_plus_x_front_corridor(
            rollout,
            env,
            args,
            plate_position=body_pose(env, PLATE_BODY)[0],
            push_direction_xy=direction_xy,
            center_high_target=center_approach_target,
            diagnostics=plate_diagnostics,
        )
        realized_contact_candidate = wrist_yaw_execution[
            "real_sim_recompiled_candidate"
        ]
        outside_high_target = np.asarray(
            realized_contact_candidate["outside_high_target"], dtype=float
        )
        outside_side_target = np.asarray(
            realized_contact_candidate["outside_side_target"], dtype=float
        )
        contact_target = np.asarray(
            realized_contact_candidate["side_contact_target"], dtype=float
        )
        compiled_side_contact_geometry = realized_contact_candidate[
            "compiled_geometry"
        ]
        print(
            "L3-A3 native +X front corridor recompiled "
            + json.dumps(plate_diagnostics(), sort_keys=True),
            flush=True,
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
            structural_waypoint_budget=wrist_yaw_execution[
                "remaining_structural_waypoint_steps"
            ],
            controller_live_diagnostic=wrist_yaw_execution[
                "controller_live_collision_diagnostic"
            ],
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
        active_wrist_approach_outward = np.asarray(
            realized_contact_candidate["outward_direction_xy"], dtype=float
        )
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
                    reference_outward_direction_xy=(
                        active_wrist_approach_outward
                    ),
                    selection_mode="native_plus_x_front_corridor",
                )
                recontact_center_target = np.asarray(
                    recontact_selected_candidate[
                        "center_high_target"
                    ],
                    dtype=float,
                )
                recontact_retreat_target = recontact_eef.copy()
                recontact_retreat_target[2] = max(
                    recontact_eef[2], recontact_center_target[2]
                )
                recontact_wrist_yaw_execution = None
                recontact_realized_candidate = None
                recontact_high_target = None
                recontact_outside_side_target = None
                recontact_side_contact_target = None
                recontact_compiled_geometry = None
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
                    "wrist_yaw_execution": recontact_wrist_yaw_execution,
                    "realized_contact_candidate": (
                        recontact_realized_candidate
                    ),
                    "retreat_target": recontact_retreat_target.tolist(),
                    "center_target": recontact_center_target.tolist(),
                    "outside_high_target": None,
                    "outside_side_target": None,
                    "side_contact_target": None,
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

                # Every recovery waypoint uses OSC env.step.  Retreat and
                # cross above the live plate, retain the native wrist at
                # center-high, recompile the +X front corridor, descend outside
                # its rim, then seek inward.
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
                recontact_wrist_yaw_execution = (
                    _prepare_native_plus_x_front_corridor(
                        rollout,
                        env,
                        args,
                        plate_position=body_pose(env, PLATE_BODY)[0],
                        push_direction_xy=recontact_direction,
                        center_high_target=recontact_center_target,
                        diagnostics=recontact_diagnostics,
                    )
                )
                recontact_realized_candidate = (
                    recontact_wrist_yaw_execution[
                        "real_sim_recompiled_candidate"
                    ]
                )
                recontact_high_target = np.asarray(
                    recontact_realized_candidate[
                        "outside_high_target"
                    ],
                    dtype=float,
                )
                recontact_outside_side_target = np.asarray(
                    recontact_realized_candidate[
                        "outside_side_target"
                    ],
                    dtype=float,
                )
                recontact_side_contact_target = np.asarray(
                    recontact_realized_candidate[
                        "side_contact_target"
                    ],
                    dtype=float,
                )
                recontact_compiled_geometry = (
                    recontact_realized_candidate["compiled_geometry"]
                )
                active_wrist_approach_outward = np.asarray(
                    recontact_realized_candidate[
                        "outward_direction_xy"
                    ],
                    dtype=float,
                )
                recontact_event.update(
                    {
                        "wrist_yaw_execution": (
                            recontact_wrist_yaw_execution
                        ),
                        "realized_contact_candidate": (
                            recontact_realized_candidate
                        ),
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
                    structural_waypoint_budget=(
                        recontact_wrist_yaw_execution[
                            "remaining_structural_waypoint_steps"
                        ]
                    ),
                    controller_live_diagnostic=(
                        recontact_wrist_yaw_execution[
                            "controller_live_collision_diagnostic"
                        ]
                    ),
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
            "initial_contact_candidate_diagnostics": candidate_geometry,
            "initial_selected_contact_candidate": (
                selected_contact_candidate
            ),
            "initial_wrist_yaw_execution": wrist_yaw_execution,
            "initial_realized_contact_candidate": (
                realized_contact_candidate
            ),
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
    parser.add_argument(
        "--diagnostic_manifest",
        default=None,
        help=(
            "standalone diagnostic-only live collision inventory JSON; "
            "defaults beside --output"
        ),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position_action_scale", type=float, default=0.08)
    parser.add_argument("--position_tolerance", type=float, default=0.005)
    parser.add_argument("--max_waypoint_steps", type=int, default=240)
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
        "--plate_contact_outside_clearance", type=float, default=0.0005
    )
    parser.add_argument(
        "--plate_contact_seek_max_translation_action",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--structural_near_plate_max_translation_action",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--vertical_corridor_descent_max_translation_action",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--post_descent_lateral_max_translation_action",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--overhead_descent_max_translation_action",
        type=float,
        default=0.20,
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

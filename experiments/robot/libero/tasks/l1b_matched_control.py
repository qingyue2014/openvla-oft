"""Pure geometry helpers for paired L1-B matched controls.

This module intentionally has no LIBERO or MuJoCo imports so its matching
contract can be unit-tested locally without initializing the simulator.
"""

from __future__ import annotations

import numpy as np


def _xy(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite XY vector")
    return array


def dual_radius_reflection(
    target_xy,
    eb_obstacle_xy,
    er_obstacle_xy,
    *,
    minimum_axis_length_m: float = 1e-6,
) -> np.ndarray:
    """Reflect ER about the target-to-EB axis to obtain the unique paired EC.

    Reflection about a line containing both the target and the EB obstacle
    preserves (1) target-to-obstacle distance and (2) displacement magnitude
    from the EB obstacle.  The reflected point changes only the intervention
    direction, which is the intended risk variable for Outcome V2.
    """

    target = _xy(target_xy, "target_xy")
    eb = _xy(eb_obstacle_xy, "eb_obstacle_xy")
    er = _xy(er_obstacle_xy, "er_obstacle_xy")
    axis = eb - target
    axis_length = float(np.linalg.norm(axis))
    if axis_length < minimum_axis_length_m:
        raise ValueError("target and EB obstacle do not define a reflection axis")
    axis_unit = axis / axis_length
    er_relative = er - target
    projection = axis_unit * float(np.dot(er_relative, axis_unit))
    reflected_relative = 2.0 * projection - er_relative
    return target + reflected_relative


def angular_separation_deg(first, second) -> float:
    first_xy = _xy(first, "first")
    second_xy = _xy(second, "second")
    first_radius = float(np.linalg.norm(first_xy))
    second_radius = float(np.linalg.norm(second_xy))
    if min(first_radius, second_radius) < 1e-9:
        return 0.0
    cosine = float(
        np.clip(
            np.dot(first_xy, second_xy) / (first_radius * second_radius),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cosine)))


def dual_radius_metrics(
    target_xy,
    eb_obstacle_xy,
    er_obstacle_xy,
    ec_obstacle_xy,
) -> dict[str, float]:
    """Return settled-pose diagnostics for the dual-radius EC contract."""

    target = _xy(target_xy, "target_xy")
    eb = _xy(eb_obstacle_xy, "eb_obstacle_xy")
    er = _xy(er_obstacle_xy, "er_obstacle_xy")
    ec = _xy(ec_obstacle_xy, "ec_obstacle_xy")
    expected_ec = dual_radius_reflection(target, eb, er)
    er_target_vector = er - target
    ec_target_vector = ec - target
    er_intervention_vector = er - eb
    ec_intervention_vector = ec - eb
    er_target_radius = float(np.linalg.norm(er_target_vector))
    ec_target_radius = float(np.linalg.norm(ec_target_vector))
    er_intervention_radius = float(np.linalg.norm(er_intervention_vector))
    ec_intervention_radius = float(np.linalg.norm(ec_intervention_vector))
    return {
        "er_target_radius_m": er_target_radius,
        "ec_target_radius_m": ec_target_radius,
        "target_radius_mismatch_m": abs(
            er_target_radius - ec_target_radius
        ),
        "er_intervention_radius_m": er_intervention_radius,
        "ec_intervention_radius_m": ec_intervention_radius,
        "intervention_radius_mismatch_m": abs(
            er_intervention_radius - ec_intervention_radius
        ),
        "angular_separation_deg": angular_separation_deg(
            er_target_vector, ec_target_vector
        ),
        "reflection_residual_m": float(np.linalg.norm(ec - expected_ec)),
        "er_ec_separation_m": float(np.linalg.norm(er - ec)),
    }


def dual_radius_match_passes(metrics: dict[str, float], spec: dict) -> bool:
    """Apply the preregistered fail-closed Outcome V2 tolerances."""

    return bool(
        metrics["target_radius_mismatch_m"]
        <= float(spec["matched_target_radius_tolerance_m"])
        and metrics["intervention_radius_mismatch_m"]
        <= float(spec["matched_intervention_radius_tolerance_m"])
        and metrics["reflection_residual_m"]
        <= float(spec["matched_reflection_residual_tolerance_m"])
        and metrics["angular_separation_deg"]
        >= float(spec["min_control_angle_separation_deg"])
    )

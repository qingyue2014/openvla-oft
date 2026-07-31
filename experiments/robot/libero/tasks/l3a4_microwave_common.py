"""Shared constants and simulator helpers for native-only L3-A4.

L3-A4 uses the native ``libero_10`` microwave task without changing its BDDL,
prompt, or asset inventory.  The only serialized-state intervention is the
pose / velocity of ``porcelain_mug_1``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


SCENARIO = "L3-A4"
TASK_SUITE = "libero_10"
TASK_ID = 9
TASK_FILE = (
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_"
    "microwave_and_close_it.bddl"
)
TASK_PROMPT = "put the yellow and white mug in the microwave and close it"
TASK_KEY = TASK_PROMPT.replace(" ", "_")

PORCELAIN_BODY = "porcelain_mug_1_main"
TARGET_BODY = "white_yellow_mug_1_main"
MICROWAVE_PREFIX = "microwave_1"
MICROWAVE_ROOT_CANDIDATES = (
    "microwave_1",
    "microwave_1_object",
    "microwave_1_main",
)
DOOR_BODY_CANDIDATES = (
    "microwave_1_microdoorroot",
    "microwave_1_object_microdoorroot",
    "microdoorroot",
)
DOOR_JOINT_CANDIDATES = (
    "microwave_1_microjoint",
    "microwave_1_object_microjoint",
    "microjoint",
)
HEATING_SITE_CANDIDATES = (
    "microwave_1_heating_region",
    "microwave_1_object_heating_region",
    "heating_region",
)

DUMMY_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
RUNTIME_WAIT_STEPS = 10
MAX_MUG_TILT_DEG = 1.0
MAX_WAIT_TRANSLATION_M = 0.003
MAX_WAIT_LINEAR_SPEED_MPS = 0.015
MAX_WAIT_ANGULAR_SPEED_RADPS = 0.15
MIN_CASCADE_DISPLACEMENT_M = 0.015
MIN_CASCADE_TILT_CHANGE_DEG = 10.0
MAX_HINGE_RADIUS_ERROR_M = 0.002
EC_ANGULAR_CANDIDATE_COUNT = 48
EC_RADIUS_INITIAL_STEP_M = 0.0005
EC_RADIUS_MAX_OFFSET_M = 0.002
EC_RADIUS_MIN_BRACKET_M = 0.000015625
MAX_EC_RADIUS_CALIBRATION_STEPS = 16


def all_model_names(model, kind: str) -> list[str]:
    count = int(getattr(model, f"n{kind}"))
    lookup = getattr(model, f"{kind}_id2name")
    return [name for index in range(count) if (name := lookup(index))]


def resolve_name(model, kind: str, candidates: Iterable[str], suffix: str) -> str:
    lookup = getattr(model, f"{kind}_name2id")
    for candidate in candidates:
        try:
            lookup(candidate)
            return candidate
        except Exception:
            continue
    matches = [name for name in all_model_names(model, kind) if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"cannot uniquely resolve {kind} suffix {suffix!r}; matches={matches}"
        )
    return matches[0]


def resolve_microwave_names(model) -> dict[str, str]:
    root = resolve_name(model, "body", MICROWAVE_ROOT_CANDIDATES, "microwave_1")
    door = resolve_name(model, "body", DOOR_BODY_CANDIDATES, "microdoorroot")
    joint = resolve_name(model, "joint", DOOR_JOINT_CANDIDATES, "microjoint")
    site = resolve_name(model, "site", HEATING_SITE_CANDIDATES, "heating_region")
    return {"fixture_root": root, "door_body": door, "door_joint": joint, "heating_site": site}


def free_joint_addresses(sim, body_name: str) -> tuple[int, int]:
    candidates = (
        body_name.replace("_main", "") + "_joint0",
        body_name.replace("_main", "_joint0"),
        body_name + "_joint0",
        body_name,
    )
    for candidate in candidates:
        try:
            joint_id = int(sim.model.joint_name2id(candidate))
        except Exception:
            continue
        return (
            int(sim.model.jnt_qposadr[joint_id]),
            int(sim.model.jnt_dofadr[joint_id]),
        )
    raise RuntimeError(f"free joint not found for {body_name!r}")


def body_tilt_deg(sim, body_name: str) -> float:
    body_id = int(sim.model.body_name2id(body_name))
    w, x, y, z = np.asarray(sim.data.body_xquat[body_id], dtype=float)
    del w, z
    up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(up_z)))


def body_pose(sim, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = int(sim.model.body_name2id(body_name))
    pos = np.asarray(sim.data.body_xpos[body_id], dtype=float).copy()
    mat = np.asarray(sim.data.body_xmat[body_id], dtype=float).reshape(3, 3).copy()
    return pos, mat


def body_speeds(sim, body_name: str) -> tuple[float, float]:
    body_id = int(sim.model.body_name2id(body_name))
    try:
        linear = np.asarray(sim.data.body_xvelp[body_id], dtype=float)
        angular = np.asarray(sim.data.body_xvelr[body_id], dtype=float)
    except AttributeError:
        angular = np.asarray(sim.data.cvel[body_id, :3], dtype=float)
        linear = np.asarray(sim.data.cvel[body_id, 3:6], dtype=float)
    return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))


def descendant_body_ids(model, root_name: str) -> set[int]:
    root_id = int(model.body_name2id(root_name))
    result = {root_id}
    changed = True
    while changed:
        changed = False
        for body_id in range(int(model.nbody)):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def descendant_geom_ids(model, root_name: str) -> set[int]:
    body_ids = descendant_body_ids(model, root_name)
    return {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def contacts_between(sim, first_geoms: set[int], second_geoms: set[int]) -> bool:
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        if (
            contact.geom1 in first_geoms
            and contact.geom2 in second_geoms
        ) or (
            contact.geom2 in first_geoms
            and contact.geom1 in second_geoms
        ):
            return True
    return False


def contact_body_names(sim, root_name: str) -> set[str]:
    geoms = descendant_geom_ids(sim.model, root_name)
    names: set[str] = set()
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        if contact.geom1 in geoms:
            other = contact.geom2
        elif contact.geom2 in geoms:
            other = contact.geom1
        else:
            continue
        name = sim.model.body_id2name(int(sim.model.geom_bodyid[other]))
        if name:
            names.add(name)
    return names


def closest_point_on_oriented_box(
    point,
    center,
    rotation,
    half_size,
) -> tuple[np.ndarray, bool]:
    """Return a box-surface point using only compiled pose and half extents."""
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    half_size = np.asarray(half_size, dtype=float)
    if point.shape != (3,) or center.shape != (3,) or half_size.shape != (3,):
        raise ValueError("box point, center, and half_size must have shape (3,)")
    if rotation.shape != (3, 3):
        raise ValueError("box rotation must have shape (3, 3)")
    if (
        not np.all(np.isfinite(point))
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(half_size))
        or np.any(half_size <= 0.0)
    ):
        raise ValueError("compiled box inputs must be finite with positive size")
    local = rotation.T @ (point - center)
    inside = bool(np.all(np.abs(local) <= half_size))
    if inside:
        distance_to_face = half_size - np.abs(local)
        axis = int(np.argmin(distance_to_face))
        closest_local = local.copy()
        closest_local[axis] = (
            half_size[axis]
            if local[axis] >= 0.0
            else -half_size[axis]
        )
    else:
        closest_local = np.clip(local, -half_size, half_size)
    return center + rotation @ closest_local, inside


def segment_aabb_distance(segment_start, segment_end, half_size) -> float:
    """Return the exact distance between a segment and a centered AABB."""
    start = np.asarray(segment_start, dtype=float)
    end = np.asarray(segment_end, dtype=float)
    half = np.asarray(half_size, dtype=float)
    if start.shape != (3,) or end.shape != (3,) or half.shape != (3,):
        raise ValueError("segment endpoints and half_size must have shape (3,)")
    if (
        not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
        or not np.all(np.isfinite(half))
        or np.any(half <= 0.0)
    ):
        raise ValueError("segment/AABB inputs must be finite with positive size")

    direction = end - start
    breakpoints = {0.0, 1.0}
    for axis in range(3):
        if abs(float(direction[axis])) <= np.finfo(float).eps:
            continue
        for face in (-half[axis], half[axis]):
            value = float((face - start[axis]) / direction[axis])
            if 0.0 < value < 1.0:
                breakpoints.add(value)
    ordered = sorted(breakpoints)
    candidates = set(ordered)
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        midpoint = 0.5 * (lower + upper)
        position = start + direction * midpoint
        active = []
        for axis in range(3):
            if position[axis] < -half[axis]:
                boundary = -half[axis]
            elif position[axis] > half[axis]:
                boundary = half[axis]
            else:
                continue
            active.append(
                (
                    float(direction[axis]),
                    float(start[axis] - boundary),
                )
            )
        denominator = sum(slope * slope for slope, _ in active)
        if denominator > np.finfo(float).eps:
            root = -sum(
                slope * intercept for slope, intercept in active
            ) / denominator
            if lower <= root <= upper:
                candidates.add(float(root))

    minimum_squared = float("inf")
    for parameter in candidates:
        position = start + direction * float(parameter)
        outside = np.maximum(np.abs(position) - half, 0.0)
        minimum_squared = min(
            minimum_squared, float(np.dot(outside, outside))
        )
    return float(np.sqrt(minimum_squared))


def oriented_box_separating_clearance(
    first_center,
    first_rotation,
    first_half_size,
    second_center,
    second_rotation,
    second_half_size,
) -> float:
    """Return the largest SAT gap; positive means two OBBs are separate."""
    center_a = np.asarray(first_center, dtype=float)
    rotation_a = np.asarray(first_rotation, dtype=float)
    half_a = np.asarray(first_half_size, dtype=float)
    center_b = np.asarray(second_center, dtype=float)
    rotation_b = np.asarray(second_rotation, dtype=float)
    half_b = np.asarray(second_half_size, dtype=float)
    if center_a.shape != (3,) or center_b.shape != (3,):
        raise ValueError("oriented-box centers must have shape (3,)")
    if rotation_a.shape != (3, 3) or rotation_b.shape != (3, 3):
        raise ValueError("oriented-box rotations must have shape (3, 3)")
    if half_a.shape != (3,) or half_b.shape != (3,):
        raise ValueError("oriented-box half sizes must have shape (3,)")
    arrays = (
        center_a,
        rotation_a,
        half_a,
        center_b,
        rotation_b,
        half_b,
    )
    if (
        not all(np.all(np.isfinite(value)) for value in arrays)
        or np.any(half_a <= 0.0)
        or np.any(half_b <= 0.0)
    ):
        raise ValueError("oriented-box inputs must be finite with positive size")

    axes = [rotation_a[:, index] for index in range(3)]
    axes.extend(rotation_b[:, index] for index in range(3))
    axes.extend(
        np.cross(rotation_a[:, first], rotation_b[:, second])
        for first in range(3)
        for second in range(3)
    )
    delta = center_b - center_a
    gaps = []
    for raw_axis in axes:
        norm = float(np.linalg.norm(raw_axis))
        if norm <= 32.0 * np.finfo(float).eps:
            continue
        axis = raw_axis / norm
        projected_a = float(
            np.sum(half_a * np.abs(rotation_a.T @ axis))
        )
        projected_b = float(
            np.sum(half_b * np.abs(rotation_b.T @ axis))
        )
        gaps.append(
            abs(float(np.dot(delta, axis))) - projected_a - projected_b
        )
    if not gaps:
        raise ValueError("oriented boxes produced no separating axes")
    return float(max(gaps))


def native_site_contains_point(
    site_position,
    site_rotation,
    site_half_size,
    point,
) -> bool:
    """Match LIBERO SiteObject.in_box's strict world-AABB predicate."""
    position = np.asarray(site_position, dtype=float)
    rotation = np.asarray(site_rotation, dtype=float)
    half_size = np.asarray(site_half_size, dtype=float)
    point = np.asarray(point, dtype=float)
    if position.shape != (3,) or half_size.shape != (3,) or point.shape != (3,):
        raise ValueError("site position, half size, and point need shape (3,)")
    if rotation.shape != (3, 3):
        raise ValueError("site rotation must have shape (3, 3)")
    arrays = (position, rotation, half_size, point)
    if (
        not all(np.all(np.isfinite(value)) for value in arrays)
        or np.any(half_size <= 0.0)
    ):
        raise ValueError("site geometry must be finite with positive size")
    world_half_size = np.abs(rotation @ half_size)
    lower = position - world_half_size
    upper = position + world_half_size
    # This is the native SiteObject.in_box lower-z allowance, preserved
    # verbatim as geometry semantics rather than introduced as a task margin.
    lower[2] -= 0.01
    return bool(np.all(point > lower) and np.all(point < upper))


def collision_masks_compatible(
    first_contype: int,
    first_conaffinity: int,
    second_contype: int,
    second_conaffinity: int,
) -> bool:
    """Apply MuJoCo's bidirectional geom collision bitmask rule."""
    values = (
        int(first_contype),
        int(first_conaffinity),
        int(second_contype),
        int(second_conaffinity),
    )
    if any(value < 0 for value in values):
        raise ValueError("collision masks must be nonnegative integers")
    return bool(
        (values[0] & values[3]) != 0
        or (values[2] & values[1]) != 0
    )


def planar_park_clearances(
    candidate_xy,
    table_center_xy,
    table_rotation_xy,
    table_half_size_xy,
    object_radius_xy: float,
    obstacle_centers_xy,
    obstacle_radii_xy,
) -> dict[str, float]:
    """Measure table-edge and sampled-obstacle clearance for a park point."""
    candidate = np.asarray(candidate_xy, dtype=float)
    table_center = np.asarray(table_center_xy, dtype=float)
    table_rotation = np.asarray(table_rotation_xy, dtype=float)
    table_half_size = np.asarray(table_half_size_xy, dtype=float)
    obstacle_centers = np.asarray(obstacle_centers_xy, dtype=float)
    obstacle_radii = np.asarray(obstacle_radii_xy, dtype=float)
    object_radius = float(object_radius_xy)
    if candidate.shape != (2,) or table_center.shape != (2,):
        raise ValueError("candidate and table center must have shape (2,)")
    if table_rotation.shape != (2, 2) or table_half_size.shape != (2,):
        raise ValueError("table rotation and half size have invalid shape")
    if obstacle_centers.ndim != 2 or obstacle_centers.shape[1:] != (2,):
        raise ValueError("obstacle centers must have shape (N, 2)")
    if obstacle_radii.shape != (len(obstacle_centers),):
        raise ValueError("obstacle radii must have shape (N,)")
    arrays = (
        candidate,
        table_center,
        table_rotation,
        table_half_size,
        obstacle_centers,
        obstacle_radii,
    )
    if (
        not all(np.all(np.isfinite(array)) for array in arrays)
        or not np.isfinite(object_radius)
        or object_radius < 0.0
        or np.any(table_half_size <= 0.0)
        or np.any(obstacle_radii < 0.0)
    ):
        raise ValueError("planar park geometry must be finite and nonnegative")
    table_local = table_rotation.T @ (candidate - table_center)
    table_axis_clearance = (
        table_half_size - np.abs(table_local) - object_radius
    )
    if len(obstacle_centers):
        obstacle_clearance = float(
            np.min(
                np.linalg.norm(
                    obstacle_centers - candidate[None, :], axis=1
                )
                - obstacle_radii
                - object_radius
            )
        )
    else:
        obstacle_clearance = float("inf")
    return {
        "table_edge_clearance_m": float(np.min(table_axis_clearance)),
        "obstacle_clearance_m": obstacle_clearance,
    }


def fixture_local_position(sim, fixture_root: str, world_position) -> np.ndarray:
    root_pos, root_mat = body_pose(sim, fixture_root)
    return root_mat.T @ (np.asarray(world_position, dtype=float) - root_pos)


def fixture_world_position(sim, fixture_root: str, local_position) -> np.ndarray:
    root_pos, root_mat = body_pose(sim, fixture_root)
    return root_pos + root_mat @ np.asarray(local_position, dtype=float)


def hinge_radius_m(local_xy, hinge_local_xy) -> float:
    local = np.asarray(local_xy, dtype=float)
    hinge = np.asarray(hinge_local_xy, dtype=float)
    if local.shape != (2,) or hinge.shape != (2,):
        raise ValueError("hinge-radius coordinates must both have shape (2,)")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(hinge)):
        raise ValueError("hinge-radius coordinates must be finite")
    return float(np.linalg.norm(local - hinge))


def radially_adjusted_input_xy(
    input_local_xy,
    hinge_local_xy,
    signed_post_wait_radius_error_m: float,
    radial_offset_m: float,
) -> np.ndarray:
    """Move a local intervention by a bounded amount along its original ray."""
    input_xy = np.asarray(input_local_xy, dtype=float)
    hinge_xy = np.asarray(hinge_local_xy, dtype=float)
    if input_xy.shape != (2,):
        raise ValueError("input_local_xy must have shape (2,)")
    input_radius = hinge_radius_m(input_xy, hinge_xy)
    signed_error = float(signed_post_wait_radius_error_m)
    offset = float(radial_offset_m)
    if not np.isfinite(signed_error):
        raise ValueError("signed_post_wait_radius_error_m must be finite")
    if (
        not np.isfinite(offset)
        or offset < 0.0
        or offset > EC_RADIUS_MAX_OFFSET_M
    ):
        raise ValueError(
            "radial_offset_m must be finite and within the calibration bound"
        )
    if input_radius <= np.finfo(float).eps:
        raise ValueError("cannot radially calibrate an input at the hinge")
    radial_unit = (input_xy - hinge_xy) / input_radius
    return input_xy - np.sign(signed_error) * offset * radial_unit


def policy_image(obs: dict) -> np.ndarray:
    """Return the exact agentview orientation consumed by OpenVLA."""
    return np.asarray(obs["agentview_image"])[::-1, ::-1].copy()

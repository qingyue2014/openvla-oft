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
    closest_point_on_oriented_box,
    collision_masks_compatible,
    contact_body_names,
    contacts_between,
    convex_mesh_aabb_distance,
    descendant_body_ids,
    descendant_geom_ids,
    native_site_contains_point,
    oriented_box_separating_clearance,
    planar_park_clearances,
    policy_image,
    resolve_microwave_names,
    segment_aabb_distance,
)
from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)


APPROACH_HEIGHT = 0.16
GRASP_HEIGHT = 0.060
PORCELAIN_GRASP_HEIGHT = 0.080
PORCELAIN_GRASP_CLEARANCE_OFFSET = 0.040
PORCELAIN_CONTACT_SEEK_STEPS = 80
PORCELAIN_CONTACT_SEEK_GAIN = 12.0
PORCELAIN_CONTACT_SEEK_ACTION_LIMIT = 0.25
PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M = 0.030
TARGET_GRASP_CLEARANCE_OFFSET = 0.040
TARGET_CONTACT_SEEK_STEPS = 80
TARGET_CONTACT_SEEK_GAIN = 12.0
TARGET_CONTACT_SEEK_ACTION_LIMIT = 0.25
SAFE_PARK_MIN_OUTWARD_DISTANCE_M = 0.060
SAFE_PARK_MAX_OUTWARD_DISTANCE_M = 0.400
SAFE_PARK_SEARCH_STEP_M = 0.010
SAFE_PARK_TABLE_EDGE_MARGIN_M = 0.020
SAFE_PARK_DOOR_SWEEP_MARGIN_M = 0.020
SAFE_PARK_STATIC_MARGIN_M = 0.020
SAFE_PARK_DOOR_SWEEP_SAMPLES = 49
TARGET_INSERTION_SEARCH_STEP_M = 0.005
TARGET_INSERTION_SWEEP_STEP_M = 0.005
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


def _eef_body_name(model) -> str:
    for name in ("gripper0_eef", "robot0_gripper0_eef", "robot0_eef"):
        try:
            model.body_name2id(name)
            return name
        except Exception:
            continue
    matches = [
        model.body_id2name(index)
        for index in range(int(model.nbody))
        if (model.body_id2name(index) or "").endswith("gripper0_eef")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve end-effector body; matches={matches}")
    return matches[0]


def _eef_position(env) -> np.ndarray:
    model = env.sim.model
    body_id = int(model.body_name2id(_eef_body_name(model)))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _robot_contact_body_names(env) -> set[str]:
    return {
        pair["other_body_name"]
        for pair in _robot_contact_pairs(env)
        if pair["other_body_name"]
    }


def _robot_contact_pairs(env) -> list[dict]:
    robot_geoms = _robot_geom_ids(env.sim.model)
    contacts = {}
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in robot_geoms:
            robot = int(contact.geom1)
            other = int(contact.geom2)
        elif contact.geom2 in robot_geoms:
            robot = int(contact.geom2)
            other = int(contact.geom1)
        else:
            continue
        robot_body = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[robot])
        ) or ""
        other_body = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[other])
        ) or ""
        if other_body.startswith(("robot0_", "gripper0_")):
            continue
        key = (robot, other)
        contacts[key] = {
            "contact_index": int(index),
            "robot_geom_id": robot,
            "robot_geom_name": _geom_name(env.sim.model, robot),
            "robot_body_name": str(robot_body),
            "other_geom_id": other,
            "other_geom_name": _geom_name(env.sim.model, other),
            "other_body_name": str(other_body),
        }
    return [contacts[key] for key in sorted(contacts)]


def _has_microwave_contact(body_names) -> bool:
    return any("microwave" in name.lower() for name in body_names)


def _geom_name(model, geom_id: int) -> str:
    name = model.geom_id2name(int(geom_id))
    return "" if name is None else str(name)


def _closest_point_on_compiled_geom(env, geom_id: int, point):
    """Return a conservative closest point from compiled MuJoCo geometry."""
    model = env.sim.model
    center = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(
        env.sim.data.geom_xmat[geom_id], dtype=float
    ).reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=float) - center)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    inside = False

    if geom_type == 6:  # mjGEOM_BOX
        return closest_point_on_oriented_box(
            point,
            center,
            rotation,
            size,
        )
    elif geom_type == 2:  # mjGEOM_SPHERE
        norm = float(np.linalg.norm(local))
        inside = norm <= float(size[0])
        direction = (
            local / norm
            if norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0, 0.0])
        )
        closest_local = direction * float(size[0])
    elif geom_type == 3:  # mjGEOM_CAPSULE, local axis is z
        axis_point = np.asarray(
            [0.0, 0.0, np.clip(local[2], -size[1], size[1])]
        )
        radial = local - axis_point
        norm = float(np.linalg.norm(radial))
        inside = norm <= float(size[0])
        direction = (
            radial / norm
            if norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0, 0.0])
        )
        closest_local = axis_point + direction * float(size[0])
    elif geom_type == 5:  # mjGEOM_CYLINDER, local axis is z
        radial = local[:2]
        radial_norm = float(np.linalg.norm(radial))
        radial_direction = (
            radial / radial_norm
            if radial_norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0])
        )
        radial_gap = radial_norm - float(size[0])
        axial_gap = abs(float(local[2])) - float(size[1])
        inside = radial_gap <= 0.0 and axial_gap <= 0.0
        closest_local = local.copy()
        if inside:
            if -radial_gap <= -axial_gap:
                closest_local[:2] = radial_direction * float(size[0])
            else:
                closest_local[2] = (
                    float(size[1]) if local[2] >= 0.0 else -float(size[1])
                )
        else:
            closest_local[:2] = (
                radial_direction * min(radial_norm, float(size[0]))
            )
            closest_local[2] = np.clip(
                local[2], -float(size[1]), float(size[1])
            )
    else:
        # Native microwave collision geoms are boxes.  For any other compiled
        # native geom type, the model's bounding radius gives a conservative
        # direction without consulting or changing source XML.
        radius = float(model.geom_rbound[geom_id])
        norm = float(np.linalg.norm(local))
        inside = norm <= radius
        direction = (
            local / norm
            if norm > np.finfo(float).eps
            else np.asarray([1.0, 0.0, 0.0])
        )
        closest_local = direction * radius

    closest = center + rotation @ closest_local
    return closest, inside


def _collision_compatible_geom_ids(model, candidates, references) -> list[int]:
    references = tuple(int(geom_id) for geom_id in references)
    return sorted(
        int(geom_id)
        for geom_id in candidates
        if any(
            collision_masks_compatible(
                model.geom_contype[geom_id],
                model.geom_conaffinity[geom_id],
                model.geom_contype[reference_id],
                model.geom_conaffinity[reference_id],
            )
            for reference_id in references
        )
    )


def _compiled_microwave_clearance(env, names, mug_position):
    """Resolve an outward XY direction from the compiled static microwave."""
    model = env.sim.model
    fixture_geoms = descendant_geom_ids(model, names["fixture_root"])
    door_geoms = descendant_geom_ids(model, names["door_body"])
    static_fixture_geoms = sorted(fixture_geoms - door_geoms)
    robot_geoms = sorted(_robot_geom_ids(model))
    if not robot_geoms:
        raise RuntimeError("compiled model has no robot geoms")
    collision_geoms = _collision_compatible_geom_ids(
        model,
        static_fixture_geoms,
        robot_geoms,
    )
    if not collision_geoms:
        fixture_masks = sorted(
            {
                (
                    int(model.geom_group[geom_id]),
                    int(model.geom_contype[geom_id]),
                    int(model.geom_conaffinity[geom_id]),
                )
                for geom_id in static_fixture_geoms
            }
        )
        robot_masks = sorted(
            {
                (
                    int(model.geom_group[geom_id]),
                    int(model.geom_contype[geom_id]),
                    int(model.geom_conaffinity[geom_id]),
                )
                for geom_id in robot_geoms
            }
        )
        raise RuntimeError(
            "compiled microwave has no robot-compatible static collision "
            f"geoms; fixture(group,contype,conaffinity)={fixture_masks}; "
            f"robot(group,contype,conaffinity)={robot_masks}"
        )

    mug_position = np.asarray(mug_position, dtype=float)
    candidates = []
    for geom_id in collision_geoms:
        closest, inside = _closest_point_on_compiled_geom(
            env, geom_id, mug_position
        )
        delta = mug_position - closest
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        if horizontal_distance <= np.finfo(float).eps:
            continue
        body_name = model.body_id2name(
            int(model.geom_bodyid[geom_id])
        ) or ""
        candidates.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "body_name": str(body_name),
                "geom_type": int(model.geom_type[geom_id]),
                "geom_group": int(model.geom_group[geom_id]),
                "geom_contype": int(model.geom_contype[geom_id]),
                "geom_conaffinity": int(
                    model.geom_conaffinity[geom_id]
                ),
                "closest_point": closest.tolist(),
                "distance_m": float(np.linalg.norm(delta)),
                "horizontal_distance_m": horizontal_distance,
                "inside": bool(inside),
                "direction_xy": (
                    delta[:2] / horizontal_distance
                ).tolist(),
            }
        )
    if not candidates:
        raise RuntimeError(
            "cannot resolve a horizontal clearance direction from compiled "
            "microwave geometry"
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["distance_m"],
            item["horizontal_distance_m"],
            item["geom_id"],
        ),
    )
    if selected["inside"]:
        raise RuntimeError(
            "porcelain mug center is inside compiled microwave collision "
            f"geom {selected['geom_name'] or selected['geom_id']}"
        )
    direction_xy = np.asarray(selected["direction_xy"], dtype=float)
    hinge_position, _ = body_pose(env.sim, names["door_body"])
    hinge_delta = mug_position[:2] - hinge_position[:2]
    hinge_norm = float(np.linalg.norm(hinge_delta))
    hinge_direction = (
        hinge_delta / hinge_norm
        if hinge_norm > np.finfo(float).eps
        else np.zeros(2, dtype=float)
    )
    return direction_xy, {
        "method": "nearest_compiled_static_microwave_collision_surface",
        "selected": selected,
        "candidate_count": len(candidates),
        "static_fixture_geom_count": len(static_fixture_geoms),
        "robot_geom_count": len(robot_geoms),
        "robot_compatible_collision_geom_count": len(collision_geoms),
        "collision_filter": (
            "(fixture.contype & robot.conaffinity) != 0 or "
            "(robot.contype & fixture.conaffinity) != 0; "
            "geom_group is diagnostic only"
        ),
        "hinge_position": hinge_position.tolist(),
        "hinge_away_direction_xy": hinge_direction.tolist(),
        "surface_hinge_direction_dot": float(
            np.dot(direction_xy, hinge_direction)
        ),
        "requested_outward_offset_m": PORCELAIN_GRASP_CLEARANCE_OFFSET,
        "predicted_eef_surface_horizontal_clearance_m": (
            selected["horizontal_distance_m"]
            + PORCELAIN_GRASP_CLEARANCE_OFFSET
        ),
    }


def _support_contact_box(env, support_body: str):
    model = env.sim.model
    mug_geoms = descendant_geom_ids(model, PORCELAIN_BODY)
    support_geoms = descendant_geom_ids(model, support_body)
    contacted_support_geoms = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in mug_geoms and contact.geom2 in support_geoms:
            contacted_support_geoms.add(int(contact.geom2))
        elif contact.geom2 in mug_geoms and contact.geom1 in support_geoms:
            contacted_support_geoms.add(int(contact.geom1))
    boxes = [
        geom_id
        for geom_id in contacted_support_geoms
        if int(model.geom_type[geom_id]) == 6
    ]
    if not boxes:
        raise RuntimeError(
            "porcelain mug has no compiled box contact with its recorded "
            f"support body {support_body!r}"
        )
    geom_id = max(
        boxes,
        key=lambda candidate: (
            float(
                model.geom_size[candidate][0]
                * model.geom_size[candidate][1]
            ),
            -int(candidate),
        ),
    )
    rotation = np.asarray(
        env.sim.data.geom_xmat[geom_id], dtype=float
    ).reshape(3, 3)
    normal = rotation[:, 2]
    normal_tilt_deg = float(
        np.degrees(
            np.arccos(np.clip(float(normal[2]), -1.0, 1.0))
        )
    )
    if normal_tilt_deg > MAX_MUG_TILT_DEG:
        raise RuntimeError(
            "compiled support contact is not an upright table top; "
            f"normal_tilt_deg={normal_tilt_deg}"
        )
    return geom_id, {
        "support_body": support_body,
        "geom_id": int(geom_id),
        "geom_name": _geom_name(model, geom_id),
        "geom_group": int(model.geom_group[geom_id]),
        "geom_contype": int(model.geom_contype[geom_id]),
        "geom_conaffinity": int(model.geom_conaffinity[geom_id]),
        "center": np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        ).tolist(),
        "rotation": rotation.tolist(),
        "half_size": np.asarray(
            model.geom_size[geom_id], dtype=float
        ).tolist(),
        "normal_tilt_deg": normal_tilt_deg,
        "selection": "largest compiled box in the actual mug-support contact",
    }


def _compiled_mug_horizontal_radius(env, mug_position) -> tuple[float, list[int]]:
    model = env.sim.model
    mug_geoms = sorted(
        geom_id
        for geom_id in descendant_geom_ids(model, PORCELAIN_BODY)
        if int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    )
    if not mug_geoms:
        raise RuntimeError("porcelain mug has no collision-capable compiled geoms")
    mug_position = np.asarray(mug_position, dtype=float)
    radius = max(
        float(
            np.linalg.norm(
                np.asarray(
                    env.sim.data.geom_xpos[geom_id], dtype=float
                )[:2]
                - mug_position[:2]
            )
            + model.geom_rbound[geom_id]
        )
        for geom_id in mug_geoms
    )
    if not np.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("invalid compiled porcelain horizontal radius")
    return radius, mug_geoms


def _compiled_door_sweep_samples(env, names):
    model = env.sim.model
    robot_geoms = _robot_geom_ids(model)
    door_geoms = _collision_compatible_geom_ids(
        model,
        descendant_geom_ids(model, names["door_body"]),
        robot_geoms,
    )
    if not door_geoms:
        raise RuntimeError("compiled door has no robot-compatible collision geoms")
    joint_id = int(model.joint_name2id(names["door_joint"]))
    qadr = int(model.jnt_qposadr[joint_id])
    start_qpos = float(env.sim.data.qpos[qadr])
    closed_qpos = float(model.jnt_range[joint_id][1])
    close_angle = closed_qpos - start_qpos
    hinge_position, hinge_rotation = body_pose(
        env.sim, names["door_body"]
    )
    hinge_axis = hinge_rotation @ np.asarray(
        model.jnt_axis[joint_id], dtype=float
    )
    centers = []
    radii = []
    geom_records = []
    fractions = np.linspace(0.0, 1.0, SAFE_PARK_DOOR_SWEEP_SAMPLES)
    for geom_id in door_geoms:
        initial_center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        radius = float(model.geom_rbound[geom_id])
        if not np.isfinite(radius) or radius <= 0.0:
            raise RuntimeError(
                f"compiled door geom {_geom_name(model, geom_id)!r} "
                "has invalid bounding radius"
            )
        radial_vector = initial_center - hinge_position
        for fraction in fractions:
            center = hinge_position + _rotation_about_axis(
                radial_vector,
                hinge_axis,
                close_angle * float(fraction),
            )
            centers.append(center[:2])
            radii.append(radius)
        geom_records.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "geom_type": int(model.geom_type[geom_id]),
                "geom_rbound_m": radius,
                "geom_contype": int(model.geom_contype[geom_id]),
                "geom_conaffinity": int(
                    model.geom_conaffinity[geom_id]
                ),
            }
        )
    return np.asarray(centers), np.asarray(radii), {
        "door_body": names["door_body"],
        "door_start_qpos": start_qpos,
        "door_closed_qpos": closed_qpos,
        "door_close_angle": close_angle,
        "hinge_position": hinge_position.tolist(),
        "hinge_axis": hinge_axis.tolist(),
        "samples_per_geom": SAFE_PARK_DOOR_SWEEP_SAMPLES,
        "door_collision_geoms": geom_records,
    }


def _compiled_safe_outward_park(
    env,
    names,
    support_body,
    start_mug_position,
    outward_direction_xy,
):
    """Find the nearest table-supported point outside the compiled door sweep."""
    model = env.sim.model
    start = np.asarray(start_mug_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    outward_norm = float(np.linalg.norm(outward))
    if outward.shape != (2,) or outward_norm <= np.finfo(float).eps:
        raise RuntimeError("safe-park outward direction is invalid")
    outward = outward / outward_norm
    support_geom, support_record = _support_contact_box(env, support_body)
    table_center = np.asarray(
        env.sim.data.geom_xpos[support_geom], dtype=float
    )
    table_rotation = np.asarray(
        env.sim.data.geom_xmat[support_geom], dtype=float
    ).reshape(3, 3)
    table_half_size = np.asarray(
        model.geom_size[support_geom], dtype=float
    )
    table_normal = table_rotation[:, 2]
    table_top = table_center + table_normal * table_half_size[2]
    support_offset = float(np.dot(start - table_top, table_normal))
    mug_radius, mug_geoms = _compiled_mug_horizontal_radius(env, start)
    door_centers, door_radii, door_record = _compiled_door_sweep_samples(
        env, names
    )
    fixture_geoms = descendant_geom_ids(model, names["fixture_root"])
    door_geoms = descendant_geom_ids(model, names["door_body"])
    static_geoms = _collision_compatible_geom_ids(
        model,
        fixture_geoms - door_geoms,
        mug_geoms,
    )
    if not static_geoms:
        raise RuntimeError(
            "compiled microwave has no mug-compatible static collision geoms"
        )

    trace = []
    distances = np.arange(
        SAFE_PARK_MIN_OUTWARD_DISTANCE_M,
        SAFE_PARK_MAX_OUTWARD_DISTANCE_M
        + 0.5 * SAFE_PARK_SEARCH_STEP_M,
        SAFE_PARK_SEARCH_STEP_M,
    )
    selected = None
    for distance in distances:
        candidate_xy = start[:2] + outward * float(distance)
        plane_rhs = (
            support_offset
            + float(np.dot(table_normal, table_top))
            - float(np.dot(table_normal[:2], candidate_xy))
        )
        if abs(float(table_normal[2])) <= np.finfo(float).eps:
            raise RuntimeError("compiled table top has a vertical normal")
        candidate = np.asarray(
            [
                candidate_xy[0],
                candidate_xy[1],
                plane_rhs / float(table_normal[2]),
            ]
        )
        planar = planar_park_clearances(
            candidate_xy,
            table_center[:2],
            table_rotation[:2, :2],
            table_half_size[:2],
            mug_radius,
            door_centers,
            door_radii,
        )
        static_candidates = []
        for geom_id in static_geoms:
            closest, inside = _closest_point_on_compiled_geom(
                env, geom_id, candidate
            )
            clearance = (
                -float("inf")
                if inside
                else float(np.linalg.norm(candidate - closest)) - mug_radius
            )
            static_candidates.append((clearance, int(geom_id)))
        static_clearance, closest_static_geom = min(static_candidates)
        passed = bool(
            planar["table_edge_clearance_m"]
            >= SAFE_PARK_TABLE_EDGE_MARGIN_M
            and planar["obstacle_clearance_m"]
            >= SAFE_PARK_DOOR_SWEEP_MARGIN_M
            and static_clearance >= SAFE_PARK_STATIC_MARGIN_M
        )
        record = {
            "outward_distance_m": float(distance),
            "candidate_position": candidate.tolist(),
            "table_edge_clearance_m": planar["table_edge_clearance_m"],
            "door_sweep_clearance_m": planar["obstacle_clearance_m"],
            "static_microwave_clearance_m": static_clearance,
            "closest_static_geom_id": closest_static_geom,
            "closest_static_geom_name": _geom_name(
                model, closest_static_geom
            ),
            "passed": passed,
        }
        trace.append(record)
        if passed:
            selected = record
            break
    if selected is None:
        raise RuntimeError(
            "no compiled table-supported outward park point clears the "
            f"door sweep within {SAFE_PARK_MAX_OUTWARD_DISTANCE_M} m; "
            f"last_candidate={trace[-1] if trace else None}"
        )
    return np.asarray(selected["candidate_position"], dtype=float), {
        "method": (
            "nearest outward point inside actual compiled support box and "
            "outside sampled compiled door sweep/static microwave"
        ),
        "outward_direction_xy": outward.tolist(),
        "selected": selected,
        "mug_horizontal_radius_bound_m": mug_radius,
        "support_offset_along_normal_m": support_offset,
        "support_geometry": support_record,
        "door_sweep_geometry": door_record,
        "table_edge_margin_m": SAFE_PARK_TABLE_EDGE_MARGIN_M,
        "door_sweep_margin_m": SAFE_PARK_DOOR_SWEEP_MARGIN_M,
        "static_microwave_margin_m": SAFE_PARK_STATIC_MARGIN_M,
        "search_min_outward_m": SAFE_PARK_MIN_OUTWARD_DISTANCE_M,
        "search_max_outward_m": SAFE_PARK_MAX_OUTWARD_DISTANCE_M,
        "search_step_m": SAFE_PARK_SEARCH_STEP_M,
        "candidate_trace": trace,
        "reachability_gate": (
            "bounded outward corridor plus real OSC env.step waypoint"
        ),
    }


def _compiled_geom_support_radius(
    model,
    geom_id,
    rotation,
    direction,
):
    """Return the compiled geom's support radius along a world direction."""
    geom_id = int(geom_id)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    local_direction = rotation.T @ direction
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == 2:
        radius = float(size[0])
        method = "compiled sphere support"
    elif geom_type == 3:
        radius = float(
            size[0] + size[1] * abs(float(local_direction[2]))
        )
        method = "compiled capsule support"
    elif geom_type == 4:
        radius = float(np.linalg.norm(size * local_direction))
        method = "compiled ellipsoid support"
    elif geom_type == 5:
        axial = abs(float(local_direction[2]))
        radial = float(
            np.sqrt(max(0.0, 1.0 - axial * axial))
        )
        radius = float(size[1] * axial + size[0] * radial)
        method = "compiled cylinder support"
    elif geom_type == 6:
        radius = float(np.sum(size * np.abs(local_direction)))
        method = "compiled box support"
    else:
        radius = float(model.geom_rbound[geom_id])
        method = "conservative compiled bounding-sphere support"
    return radius, method


def _compiled_target_support_geometry(env, desired_up):
    """Measure the target body's native table-support offset from contacts."""
    model = env.sim.model
    desired_up = np.asarray(desired_up, dtype=float)
    desired_up = desired_up / np.linalg.norm(desired_up)
    target_geoms = descendant_geom_ids(model, TARGET_BODY)
    target_position, _ = body_pose(env.sim, TARGET_BODY)
    candidates = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in target_geoms:
            target_geom = int(contact.geom1)
            support_geom = int(contact.geom2)
        elif contact.geom2 in target_geoms:
            target_geom = int(contact.geom2)
            support_geom = int(contact.geom1)
        else:
            continue
        support_body = model.body_id2name(
            int(model.geom_bodyid[support_geom])
        ) or ""
        if (
            support_body.startswith(("robot0_", "gripper0_"))
            or "microwave" in support_body.lower()
            or support_geom in target_geoms
            or int(model.geom_type[support_geom]) != 6
        ):
            continue
        rotation = np.asarray(
            env.sim.data.geom_xmat[support_geom], dtype=float
        ).reshape(3, 3)
        half_size = np.asarray(
            model.geom_size[support_geom], dtype=float
        )
        alignment = rotation.T @ desired_up
        normal_axis = int(np.argmax(np.abs(alignment)))
        normal = rotation[:, normal_axis] * (
            1.0 if alignment[normal_axis] >= 0.0 else -1.0
        )
        normal_tilt_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(normal, desired_up)), -1.0, 1.0)
                )
            )
        )
        surface = (
            np.asarray(
                env.sim.data.geom_xpos[support_geom], dtype=float
            )
            + normal * half_size[normal_axis]
        )
        support_offset = float(
            np.dot(target_position - surface, normal)
        )
        tangent_axes = [
            axis for axis in range(3) if axis != normal_axis
        ]
        candidates.append(
            {
                "contact_index": int(index),
                "target_geom_id": target_geom,
                "target_geom_name": _geom_name(model, target_geom),
                "support_geom_id": support_geom,
                "support_geom_name": _geom_name(model, support_geom),
                "support_body": str(support_body),
                "support_surface_position": surface.tolist(),
                "support_normal": normal.tolist(),
                "normal_axis": normal_axis,
                "normal_tilt_deg": normal_tilt_deg,
                "support_offset_m": support_offset,
                "support_planar_area_m2": float(
                    4.0
                    * half_size[tangent_axes[0]]
                    * half_size[tangent_axes[1]]
                ),
            }
        )
    level = [
        candidate
        for candidate in candidates
        if candidate["normal_tilt_deg"] <= MAX_MUG_TILT_DEG
    ]
    if not level:
        raise RuntimeError(
            "target mug has no compiled level non-microwave support contact; "
            f"contacts={candidates}"
        )
    selected = max(
        level,
        key=lambda item: (
            item["support_planar_area_m2"],
            -item["normal_tilt_deg"],
            -item["support_geom_id"],
        ),
    )
    selected_support = int(selected["support_geom_id"])
    contacting_target_geoms = sorted(
        {
            int(candidate["target_geom_id"])
            for candidate in level
            if int(candidate["support_geom_id"]) == selected_support
        }
    )
    selected_normal = np.asarray(
        selected["support_normal"], dtype=float
    )
    bottom_candidates = []
    for geom_id in sorted(target_geoms):
        if not collision_masks_compatible(
            model.geom_contype[geom_id],
            model.geom_conaffinity[geom_id],
            model.geom_contype[selected_support],
            model.geom_conaffinity[selected_support],
        ):
            continue
        center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        rotation = np.asarray(
            env.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        support_radius, support_method = _compiled_geom_support_radius(
            model,
            geom_id,
            rotation,
            selected_normal,
        )
        bottom_candidates.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "geom_type": int(model.geom_type[geom_id]),
                "center_relative_to_target": (
                    center - target_position
                ).tolist(),
                "rotation": rotation.tolist(),
                "support_radius_m": support_radius,
                "support_radius_method": support_method,
                "bottom_projection_m": float(
                    np.dot(center, selected_normal) - support_radius
                ),
            }
        )
    if not bottom_candidates:
        raise RuntimeError(
            "target mug has no collision-compatible compiled bottom geoms"
        )
    minimum_bottom = min(
        item["bottom_projection_m"] for item in bottom_candidates
    )
    compiled_bottom_tolerance = float(
        getattr(model, "geom_margin", np.zeros(int(model.ngeom)))[
            selected_support
        ]
    )
    compiled_bottom_tolerance += max(
        (
            float(
                getattr(
                    model,
                    "geom_margin",
                    np.zeros(int(model.ngeom)),
                )[item["geom_id"]]
            )
            for item in bottom_candidates
        ),
        default=0.0,
    )
    numerical_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(1.0, abs(minimum_bottom))
    )
    bottom_tolerance = max(
        compiled_bottom_tolerance, numerical_tolerance
    )
    supporting_target_geoms = sorted(
        {
            *contacting_target_geoms,
            *(
                item["geom_id"]
                for item in bottom_candidates
                if item["bottom_projection_m"]
                <= minimum_bottom + bottom_tolerance
            ),
        }
    )
    return {
        "method": "actual compiled target-to-table support contacts",
        "target_initial_position": target_position.tolist(),
        "desired_up": desired_up.tolist(),
        "selected": selected,
        "contacting_target_geom_ids": contacting_target_geoms,
        "supporting_target_geom_ids": supporting_target_geoms,
        "compiled_bottom_tolerance_m": bottom_tolerance,
        "bottom_geom_candidates": bottom_candidates,
        "candidate_contacts": candidates,
    }


def _compiled_held_target_support_geometry(
    env,
    support_geometry,
    floor_geom_id,
    floor_normal,
):
    """Calibrate floor height for the target's actual held orientation."""
    model = env.sim.model
    floor_geom_id = int(floor_geom_id)
    floor_normal = np.asarray(floor_normal, dtype=float)
    floor_normal = floor_normal / np.linalg.norm(floor_normal)
    source_normal = np.asarray(
        support_geometry["selected"]["support_normal"], dtype=float
    )
    normal_alignment = float(np.dot(source_normal, floor_normal))
    normal_mismatch_deg = float(
        np.degrees(
            np.arccos(np.clip(normal_alignment, -1.0, 1.0))
        )
    )
    if normal_mismatch_deg > MAX_MUG_TILT_DEG:
        raise RuntimeError(
            "native table support and compiled microwave floor normals "
            f"differ by {normal_mismatch_deg}deg"
        )

    current_target, current_target_rotation = body_pose(
        env.sim, TARGET_BODY
    )
    initial_candidates = []
    held_candidates = []
    for source in support_geometry["bottom_geom_candidates"]:
        geom_id = int(source["geom_id"])
        if not collision_masks_compatible(
            model.geom_contype[geom_id],
            model.geom_conaffinity[geom_id],
            model.geom_contype[floor_geom_id],
            model.geom_conaffinity[floor_geom_id],
        ):
            continue
        initial_rotation = np.asarray(
            source["rotation"], dtype=float
        ).reshape(3, 3)
        initial_radius, initial_method = (
            _compiled_geom_support_radius(
                model,
                geom_id,
                initial_rotation,
                floor_normal,
            )
        )
        initial_relative_center = np.asarray(
            source["center_relative_to_target"], dtype=float
        )
        initial_relative_bottom = float(
            np.dot(initial_relative_center, floor_normal)
            - initial_radius
        )

        held_center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        held_rotation = np.asarray(
            env.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        held_radius, held_method = _compiled_geom_support_radius(
            model,
            geom_id,
            held_rotation,
            floor_normal,
        )
        held_relative_center = held_center - current_target
        held_relative_bottom = float(
            np.dot(held_relative_center, floor_normal) - held_radius
        )
        initial_candidates.append(
            {
                "geom_id": geom_id,
                "geom_name": _geom_name(model, geom_id),
                "relative_bottom_m": initial_relative_bottom,
                "support_radius_m": initial_radius,
                "support_radius_method": initial_method,
            }
        )
        held_candidates.append(
            {
                "geom_id": geom_id,
                "geom_name": _geom_name(model, geom_id),
                "relative_center": held_relative_center.tolist(),
                "rotation": held_rotation.tolist(),
                "relative_bottom_m": held_relative_bottom,
                "support_radius_m": held_radius,
                "support_radius_method": held_method,
            }
        )
    if not initial_candidates or not held_candidates:
        raise RuntimeError(
            "target has no compiled collision geom compatible with the "
            "microwave floor"
        )
    initial_relative_bottom = min(
        item["relative_bottom_m"] for item in initial_candidates
    )
    held_relative_bottom = min(
        item["relative_bottom_m"] for item in held_candidates
    )
    source_support_offset = float(
        support_geometry["selected"]["support_offset_m"]
    )
    held_support_offset = float(
        source_support_offset
        + initial_relative_bottom
        - held_relative_bottom
    )
    return {
        "method": (
            "actual held-pose compiled geom centers and rotations, "
            "calibrated to native table support contact"
        ),
        "floor_geom_id": floor_geom_id,
        "floor_geom_name": _geom_name(model, floor_geom_id),
        "floor_normal": floor_normal.tolist(),
        "source_support_normal": source_normal.tolist(),
        "support_normal_mismatch_deg": normal_mismatch_deg,
        "source_support_offset_m": source_support_offset,
        "initial_relative_bottom_m": initial_relative_bottom,
        "held_relative_bottom_m": held_relative_bottom,
        "held_support_offset_m": held_support_offset,
        "target_position_at_planning": current_target.tolist(),
        "target_rotation_at_planning": current_target_rotation.tolist(),
        "target_tilt_at_planning_deg": body_tilt_deg(
            env.sim, TARGET_BODY
        ),
        "initial_geom_candidates": initial_candidates,
        "held_geom_candidates": held_candidates,
    }


def _compiled_microwave_floor(
    env,
    names,
    site_position,
    site_rotation,
    target_geom_ids,
):
    """Resolve the level compiled box surface directly below heating site."""
    model = env.sim.model
    site_position = np.asarray(site_position, dtype=float)
    site_up = np.asarray(site_rotation, dtype=float)[:, 2]
    static_geoms = (
        descendant_geom_ids(model, names["fixture_root"])
        - descendant_geom_ids(model, names["door_body"])
    )
    compatible = _collision_compatible_geom_ids(
        model,
        static_geoms,
        target_geom_ids,
    )
    candidates = []
    for geom_id in compatible:
        if int(model.geom_type[geom_id]) != 6:
            continue
        center = np.asarray(
            env.sim.data.geom_xpos[geom_id], dtype=float
        )
        rotation = np.asarray(
            env.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        half_size = np.asarray(model.geom_size[geom_id], dtype=float)
        alignment = rotation.T @ site_up
        normal_axis = int(np.argmax(np.abs(alignment)))
        normal = rotation[:, normal_axis] * (
            1.0 if alignment[normal_axis] >= 0.0 else -1.0
        )
        tilt_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(normal, site_up)), -1.0, 1.0)
                )
            )
        )
        surface = center + normal * half_size[normal_axis]
        below_site_m = float(np.dot(site_position - surface, site_up))
        local_site = rotation.T @ (site_position - center)
        tangent_axes = [
            axis for axis in range(3) if axis != normal_axis
        ]
        covers_site_center = bool(
            all(
                abs(float(local_site[axis]))
                <= float(half_size[axis])
                for axis in tangent_axes
            )
        )
        candidates.append(
            {
                "geom_id": int(geom_id),
                "geom_name": _geom_name(model, geom_id),
                "body_name": str(
                    model.body_id2name(
                        int(model.geom_bodyid[geom_id])
                    )
                    or ""
                ),
                "center": center.tolist(),
                "rotation": rotation.tolist(),
                "half_size": half_size.tolist(),
                "normal_axis": normal_axis,
                "normal": normal.tolist(),
                "surface_position": surface.tolist(),
                "normal_tilt_deg": tilt_deg,
                "below_site_m": below_site_m,
                "covers_site_center": covers_site_center,
                "geom_contype": int(model.geom_contype[geom_id]),
                "geom_conaffinity": int(
                    model.geom_conaffinity[geom_id]
                ),
            }
        )
    floors = [
        candidate
        for candidate in candidates
        if candidate["normal_tilt_deg"] <= MAX_MUG_TILT_DEG
        and candidate["covers_site_center"]
        and candidate["below_site_m"] > 0.0
    ]
    if not floors:
        raise RuntimeError(
            "cannot resolve compiled level microwave floor below native "
            f"heating site; candidates={candidates}"
        )
    selected = min(
        floors,
        key=lambda item: (item["below_site_m"], item["geom_id"]),
    )
    return selected, {
        "method": (
            "nearest compiled target-compatible level box surface below "
            "native heating-site center"
        ),
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _signed_point_box_clearance(point, center, rotation, half_size):
    local = np.asarray(rotation, dtype=float).T @ (
        np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    )
    outside = np.maximum(
        np.abs(local) - np.asarray(half_size, dtype=float), 0.0
    )
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance > 0.0:
        return outside_distance
    return -float(
        np.min(np.asarray(half_size, dtype=float) - np.abs(local))
    )


def _compiled_convex_mesh_geometry(model, geom_id):
    """Decode the exact convex hull used by MuJoCo mesh collision."""
    geom_id = int(geom_id)
    if int(model.geom_type[geom_id]) != 7:
        raise RuntimeError(
            f"geom {_geom_name(model, geom_id)!r} is not a mesh"
        )
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise RuntimeError(
            f"mesh geom {_geom_name(model, geom_id)!r} has no data id"
        )
    graph_address = int(model.mesh_graphadr[mesh_id])
    if graph_address < 0:
        raise RuntimeError(
            f"mesh geom {_geom_name(model, geom_id)!r} has no compiled "
            "MuJoCo convex hull"
        )
    graph = np.asarray(model.mesh_graph, dtype=int).reshape(-1)
    if graph_address + 2 > len(graph):
        raise RuntimeError(
            f"compiled convex hull header for mesh {mesh_id} is truncated"
        )
    hull_vertex_count = int(graph[graph_address])
    hull_face_count = int(graph[graph_address + 1])
    if hull_vertex_count < 4 or hull_face_count < 4:
        raise RuntimeError(
            f"invalid compiled convex hull for {_geom_name(model, geom_id)!r}"
        )
    edge_record_count = hull_vertex_count + 3 * hull_face_count
    face_address = (
        graph_address
        + 2
        + 2 * hull_vertex_count
        + edge_record_count
    )
    graph_end = face_address + 3 * hull_face_count
    if graph_end > len(graph):
        raise RuntimeError(
            f"compiled convex hull record for mesh {mesh_id} is truncated"
        )
    face_ids = np.asarray(
        graph[face_address:graph_end],
        dtype=int,
    ).reshape(hull_face_count, 3)
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    hull_id_address = graph_address + 2 + hull_vertex_count
    hull_global_ids = np.asarray(
        graph[hull_id_address : hull_id_address + hull_vertex_count],
        dtype=int,
    )
    if (
        vertex_address < 0
        or vertex_count < 4
        or vertex_address + vertex_count > len(model.mesh_vert)
        or np.any(hull_global_ids < 0)
        or np.any(hull_global_ids >= vertex_count)
        or len(set(int(index) for index in hull_global_ids))
        != hull_vertex_count
        or np.any(face_ids < 0)
        or np.any(face_ids >= vertex_count)
    ):
        raise RuntimeError(
            f"compiled convex hull vertices are invalid for mesh {mesh_id}"
        )
    full_vertices = np.asarray(
        model.mesh_vert[
            vertex_address : vertex_address + vertex_count
        ],
        dtype=float,
    ).reshape(vertex_count, 3)
    global_to_hull = {
        int(global_id): hull_id
        for hull_id, global_id in enumerate(hull_global_ids)
    }
    if any(
        int(index) not in global_to_hull for index in face_ids.reshape(-1)
    ):
        raise RuntimeError(
            f"compiled convex hull faces reference non-hull vertices "
            f"for mesh {mesh_id}"
        )
    hull_vertices = full_vertices[hull_global_ids]
    hull_faces = np.asarray(
        [
            [global_to_hull[int(index)] for index in face]
            for face in face_ids
        ],
        dtype=int,
    )
    return hull_vertices, hull_faces, {
        "geom_id": geom_id,
        "geom_name": _geom_name(model, geom_id),
        "geom_type": int(model.geom_type[geom_id]),
        "geom_size": np.asarray(
            model.geom_size[geom_id], dtype=float
        ).tolist(),
        "geom_rbound": float(model.geom_rbound[geom_id]),
        "mesh_id": mesh_id,
        "mesh_vertex_count": vertex_count,
        "convex_hull_vertex_count": len(hull_vertices),
        "convex_hull_face_count": len(hull_faces),
        "mesh_graph_address": graph_address,
    }


def _compiled_geom_evidence(model, geom_id):
    """Record the exact compiled shape inputs used by clearance."""
    geom_id = int(geom_id)
    evidence = {
        "geom_id": geom_id,
        "geom_name": _geom_name(model, geom_id),
        "geom_type": int(model.geom_type[geom_id]),
        "geom_size": np.asarray(
            model.geom_size[geom_id], dtype=float
        ).tolist(),
        "geom_rbound": float(model.geom_rbound[geom_id]),
        "geom_dataid": int(model.geom_dataid[geom_id]),
        "geom_margin": float(
            getattr(
                model,
                "geom_margin",
                np.zeros(int(model.ngeom)),
            )[geom_id]
        ),
    }
    if evidence["geom_type"] == 7:
        _, _, mesh_evidence = _compiled_convex_mesh_geometry(
            model, geom_id
        )
        evidence["convex_mesh"] = mesh_evidence
    return evidence


def _compiled_geom_pair_clearance(
    env,
    moving_geom,
    fixture_geom,
    translation,
    guard_margin,
    *,
    fixture_center_override=None,
    fixture_rotation_override=None,
):
    """Conservatively clear a translated geom against native fixture geom."""
    model = env.sim.model
    moving_geom = int(moving_geom)
    fixture_geom = int(fixture_geom)
    translation = np.asarray(translation, dtype=float)
    moving_center = (
        np.asarray(env.sim.data.geom_xpos[moving_geom], dtype=float)
        + translation
    )
    moving_rotation = np.asarray(
        env.sim.data.geom_xmat[moving_geom], dtype=float
    ).reshape(3, 3)
    fixture_center = (
        np.asarray(
            env.sim.data.geom_xpos[fixture_geom], dtype=float
        )
        if fixture_center_override is None
        else np.asarray(fixture_center_override, dtype=float)
    )
    fixture_rotation = (
        np.asarray(
            env.sim.data.geom_xmat[fixture_geom], dtype=float
        ).reshape(3, 3)
        if fixture_rotation_override is None
        else np.asarray(fixture_rotation_override, dtype=float).reshape(
            3, 3
        )
    )
    moving_type = int(model.geom_type[moving_geom])
    fixture_type = int(model.geom_type[fixture_geom])
    native_geom_margin = 0.0
    if hasattr(model, "geom_margin"):
        native_geom_margin = float(
            model.geom_margin[moving_geom]
            + model.geom_margin[fixture_geom]
        )
    compiled_margin = float(guard_margin) + native_geom_margin

    if moving_type == 6 and fixture_type in (3, 5):
        fixture_size = np.asarray(
            model.geom_size[fixture_geom], dtype=float
        )
        fixture_axis = fixture_rotation[:, 2]
        start = fixture_center - fixture_axis * fixture_size[1]
        end = fixture_center + fixture_axis * fixture_size[1]
        start_local = moving_rotation.T @ (start - moving_center)
        end_local = moving_rotation.T @ (end - moving_center)
        clearance = (
            segment_aabb_distance(
                start_local,
                end_local,
                np.asarray(model.geom_size[moving_geom], dtype=float),
            )
            - float(fixture_size[0])
        )
        method = (
            "compiled box-capsule distance"
            if fixture_type == 3
            else "conservative compiled box-cylinder-as-capsule distance"
        )
    elif fixture_type != 6:
        clearance = float(
            np.linalg.norm(moving_center - fixture_center)
            - model.geom_rbound[moving_geom]
            - model.geom_rbound[fixture_geom]
        )
        method = "conservative compiled bounding spheres"
    else:
        fixture_half_size = np.asarray(
            model.geom_size[fixture_geom], dtype=float
        )
        if moving_type == 7:
            sphere_lower_bound = (
                _signed_point_box_clearance(
                    moving_center,
                    fixture_center,
                    fixture_rotation,
                    fixture_half_size,
                )
                - float(model.geom_rbound[moving_geom])
            )
            if sphere_lower_bound > compiled_margin:
                clearance = sphere_lower_bound
                method = (
                    "certified compiled bounding-sphere-to-box positive "
                    "lower bound"
                )
            else:
                (
                    mesh_vertices,
                    mesh_faces,
                    _,
                ) = _compiled_convex_mesh_geometry(model, moving_geom)
                world_vertices = (
                    moving_center
                    + (moving_rotation @ mesh_vertices.T).T
                )
                fixture_local_vertices = (
                    fixture_rotation.T
                    @ (world_vertices - fixture_center).T
                ).T
                clearance = convex_mesh_aabb_distance(
                    fixture_local_vertices,
                    mesh_faces,
                    fixture_half_size,
                )
                method = (
                    "exact compiled MuJoCo convex-mesh-to-box distance"
                )
        elif moving_type == 6:
            clearance = oriented_box_separating_clearance(
                moving_center,
                moving_rotation,
                np.asarray(model.geom_size[moving_geom], dtype=float),
                fixture_center,
                fixture_rotation,
                fixture_half_size,
            )
            method = "compiled box-box separating-axis gap"
        elif moving_type in (3, 5):
            moving_size = np.asarray(
                model.geom_size[moving_geom], dtype=float
            )
            axis = moving_rotation[:, 2]
            start = moving_center - axis * moving_size[1]
            end = moving_center + axis * moving_size[1]
            start_local = fixture_rotation.T @ (
                start - fixture_center
            )
            end_local = fixture_rotation.T @ (end - fixture_center)
            clearance = (
                segment_aabb_distance(
                    start_local,
                    end_local,
                    fixture_half_size,
                )
                - float(moving_size[0])
            )
            method = (
                "compiled capsule-box distance"
                if moving_type == 3
                else "conservative compiled cylinder-as-capsule distance"
            )
        else:
            clearance = (
                _signed_point_box_clearance(
                    moving_center,
                    fixture_center,
                    fixture_rotation,
                    fixture_half_size,
                )
                - float(model.geom_rbound[moving_geom])
            )
            method = "conservative compiled bounding-sphere-to-box distance"
    return (
        float(clearance - compiled_margin),
        method,
        {
            "primitive_clearance_m": float(clearance),
            "native_geom_margin_m": native_geom_margin,
            "continuous_guard_m": float(guard_margin),
            "net_clearance_m": float(clearance - compiled_margin),
        },
    )


def _compiled_target_door_sweep_clearance(
    env,
    names,
    target_geoms,
    candidate_target_position,
    current_target_position,
):
    """Check the placed target against the entire compiled closing door arc."""
    model = env.sim.model
    door_geoms = _collision_compatible_geom_ids(
        model,
        descendant_geom_ids(model, names["door_body"]),
        target_geoms,
    )
    collision_target_geoms = _collision_compatible_geom_ids(
        model,
        target_geoms,
        door_geoms,
    )
    if not door_geoms or not collision_target_geoms:
        raise RuntimeError(
            "compiled target/door sweep has no compatible collision geoms"
        )
    joint_id = int(model.joint_name2id(names["door_joint"]))
    qadr = int(model.jnt_qposadr[joint_id])
    start_qpos = float(env.sim.data.qpos[qadr])
    closed_qpos = float(model.jnt_range[joint_id][1])
    close_angle = closed_qpos - start_qpos
    hinge_position, hinge_rotation = body_pose(
        env.sim, names["door_body"]
    )
    hinge_axis = hinge_rotation @ np.asarray(
        model.jnt_axis[joint_id], dtype=float
    )
    hinge_axis = hinge_axis / np.linalg.norm(hinge_axis)
    fractions = np.linspace(
        0.0, 1.0, SAFE_PARK_DOOR_SWEEP_SAMPLES
    )
    angular_spacing = abs(float(close_angle)) / max(
        len(fractions) - 1, 1
    )
    target_translation = (
        np.asarray(candidate_target_position, dtype=float)
        - np.asarray(current_target_position, dtype=float)
    )
    minimum = float("inf")
    limiting = None
    evaluations = 0
    for door_geom in door_geoms:
        initial_center = np.asarray(
            env.sim.data.geom_xpos[door_geom], dtype=float
        )
        initial_rotation = np.asarray(
            env.sim.data.geom_xmat[door_geom], dtype=float
        ).reshape(3, 3)
        radial = initial_center - hinge_position
        axial = hinge_axis * float(np.dot(radial, hinge_axis))
        swept_radius = float(
            np.linalg.norm(radial - axial)
            + model.geom_rbound[door_geom]
        )
        continuous_guard = 0.5 * swept_radius * angular_spacing
        for sample_index, fraction in enumerate(fractions):
            angle = close_angle * float(fraction)
            center = hinge_position + _rotation_about_axis(
                radial, hinge_axis, angle
            )
            rotation = np.column_stack(
                [
                    _rotation_about_axis(
                        initial_rotation[:, axis],
                        hinge_axis,
                        angle,
                    )
                    for axis in range(3)
                ]
            )
            for target_geom in collision_target_geoms:
                if not collision_masks_compatible(
                    model.geom_contype[target_geom],
                    model.geom_conaffinity[target_geom],
                    model.geom_contype[door_geom],
                    model.geom_conaffinity[door_geom],
                ):
                    continue
                evaluations += 1
                (
                    clearance,
                    method,
                    clearance_components,
                ) = _compiled_geom_pair_clearance(
                    env,
                    target_geom,
                    door_geom,
                    target_translation,
                    continuous_guard,
                    fixture_center_override=center,
                    fixture_rotation_override=rotation,
                )
                if clearance < minimum:
                    minimum = clearance
                    limiting = {
                        "sample_index": int(sample_index),
                        "sample_fraction": float(fraction),
                        "door_angle_rad": float(angle),
                        "target_geom_id": int(target_geom),
                        "target_geom_name": _geom_name(
                            model, target_geom
                        ),
                        "door_geom_id": int(door_geom),
                        "door_geom_name": _geom_name(
                            model, door_geom
                        ),
                        "clearance_m": float(clearance),
                        "continuous_guard_m": continuous_guard,
                        "method": method,
                        "clearance_components": clearance_components,
                    }
    if evaluations == 0 or limiting is None:
        raise RuntimeError(
            "compiled target/door sweep produced no compatible evaluations"
        )
    limiting["target_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["target_geom_id"]
    )
    limiting["door_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["door_geom_id"]
    )
    return minimum, {
        "door_start_qpos": start_qpos,
        "door_closed_qpos": closed_qpos,
        "door_close_angle_rad": close_angle,
        "hinge_position": hinge_position.tolist(),
        "hinge_axis": hinge_axis.tolist(),
        "samples": len(fractions),
        "compatible_pair_evaluations": evaluations,
        "collision_filter": (
            "MuJoCo bidirectional contype/conaffinity compatibility; "
            "native visual-only 0/0 geoms are excluded"
        ),
        "minimum_clearance_m": minimum,
        "limiting_pair": limiting,
    }


def _translated_swept_clearance(
    env,
    moving_geoms,
    fixture_geoms,
    start_position,
    end_position,
    reference_position,
):
    """Bound the whole straight translation, inflating by half sample step."""
    model = env.sim.model
    start = np.asarray(start_position, dtype=float)
    end = np.asarray(end_position, dtype=float)
    reference = np.asarray(reference_position, dtype=float)
    distance = float(np.linalg.norm(end - start))
    intervals = max(
        1, int(np.ceil(distance / TARGET_INSERTION_SWEEP_STEP_M))
    )
    spacing = distance / intervals
    sweep_guard = 0.5 * spacing
    minimum = float("inf")
    limiting = None
    compatible_pairs = 0
    for sample_index, fraction in enumerate(
        np.linspace(0.0, 1.0, intervals + 1)
    ):
        translated_position = (
            start + (end - start) * float(fraction)
        )
        translation = translated_position - reference
        for moving_geom in moving_geoms:
            for fixture_geom in fixture_geoms:
                if not collision_masks_compatible(
                    model.geom_contype[moving_geom],
                    model.geom_conaffinity[moving_geom],
                    model.geom_contype[fixture_geom],
                    model.geom_conaffinity[fixture_geom],
                ):
                    continue
                compatible_pairs += 1
                (
                    clearance,
                    method,
                    clearance_components,
                ) = _compiled_geom_pair_clearance(
                    env,
                    moving_geom,
                    fixture_geom,
                    translation,
                    sweep_guard,
                )
                if clearance < minimum:
                    minimum = clearance
                    limiting = {
                        "sample_index": int(sample_index),
                        "sample_fraction": float(fraction),
                        "translated_reference_position": (
                            translated_position.tolist()
                        ),
                        "moving_geom_id": int(moving_geom),
                        "moving_geom_name": _geom_name(
                            model, moving_geom
                        ),
                        "moving_body_name": str(
                            model.body_id2name(
                                int(model.geom_bodyid[moving_geom])
                            )
                            or ""
                        ),
                        "fixture_geom_id": int(fixture_geom),
                        "fixture_geom_name": _geom_name(
                            model, fixture_geom
                        ),
                        "fixture_body_name": str(
                            model.body_id2name(
                                int(model.geom_bodyid[fixture_geom])
                            )
                            or ""
                        ),
                        "clearance_m": float(clearance),
                        "method": method,
                        "clearance_components": clearance_components,
                    }
    if compatible_pairs == 0 or limiting is None:
        raise RuntimeError(
            "compiled insertion sweep has no collision-compatible geom pairs"
        )
    limiting["moving_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["moving_geom_id"]
    )
    limiting["fixture_compiled_geometry"] = _compiled_geom_evidence(
        model, limiting["fixture_geom_id"]
    )
    return minimum, {
        "path_start": start.tolist(),
        "path_end": end.tolist(),
        "reference_position": reference.tolist(),
        "sample_zero_is_current_pose": bool(
            np.allclose(start, reference, rtol=0.0, atol=1e-12)
        ),
        "sample_zero_semantics": (
            "synthetic translated path start; it equals current mjData "
            "geometry only when path_start equals reference_position"
        ),
        "path_length_m": distance,
        "sample_intervals": intervals,
        "sample_spacing_m": spacing,
        "continuous_sweep_guard_m": sweep_guard,
        "compatible_pair_evaluations": compatible_pairs,
        "minimum_clearance_m": minimum,
        "limiting_pair": limiting,
    }


def _compiled_rigid_gripper_fixture_geoms(env, names):
    """Resolve collision geoms that translate rigidly with the controlled EEF."""
    model = env.sim.model
    eef_root = _eef_body_name(model)
    eef_body_ids = descendant_body_ids(model, eef_root)
    eef_body_ids.update(
        body_id
        for body_id in range(int(model.nbody))
        if (
            (
                model.body_id2name(body_id) or ""
            ).startswith(("robot0_", "gripper0_"))
            and any(
                token
                in (model.body_id2name(body_id) or "").lower()
                for token in ("gripper", "hand", "finger")
            )
        )
    )
    gripper_geoms = sorted(
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in eef_body_ids
    )
    fixture_geoms = sorted(
        descendant_geom_ids(model, names["fixture_root"])
    )
    collision_gripper_geoms = _collision_compatible_geom_ids(
        model, gripper_geoms, fixture_geoms
    )
    collision_fixture_geoms = _collision_compatible_geom_ids(
        model, fixture_geoms, collision_gripper_geoms
    )
    if not collision_gripper_geoms or not collision_fixture_geoms:
        raise RuntimeError(
            "compiled target path has no gripper/microwave collision "
            f"geometry; eef_root={eef_root!r}; "
            f"gripper_geoms={gripper_geoms}"
        )
    return (
        collision_gripper_geoms,
        collision_fixture_geoms,
        {
            "eef_root_body": eef_root,
            "rigid_gripper_body_names": sorted(
                str(model.body_id2name(body_id) or "")
                for body_id in eef_body_ids
            ),
            "rigid_gripper_geom_ids": gripper_geoms,
            "collision_gripper_geom_ids": collision_gripper_geoms,
            "collision_fixture_geom_ids": collision_fixture_geoms,
        },
    )


def _compiled_target_grasp_clearance(
    env,
    names,
    target_position,
    outward_direction_xy,
):
    """Derive a no-contact grasp corridor around every native scene object."""
    model = env.sim.model
    target_position = np.asarray(target_position, dtype=float)
    outward = np.asarray(outward_direction_xy, dtype=float)
    if outward.shape != (2,):
        raise RuntimeError("target grasp outward direction is not planar")
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm <= np.finfo(float).eps:
        raise RuntimeError("target grasp outward direction is zero")
    outward = outward / outward_norm
    current_eef = _eef_position(env)
    (
        fixture_compatible_gripper_geoms,
        fixture_geoms,
        rigid_gripper_geometry,
    ) = _compiled_rigid_gripper_fixture_geoms(env, names)
    rigid_gripper_geoms = rigid_gripper_geometry[
        "rigid_gripper_geom_ids"
    ]
    target_geoms = sorted(descendant_geom_ids(model, TARGET_BODY))
    collision_target_geoms = _collision_compatible_geom_ids(
        model,
        target_geoms,
        rigid_gripper_geoms,
    )
    collision_gripper_geoms = _collision_compatible_geom_ids(
        model,
        rigid_gripper_geoms,
        collision_target_geoms,
    )
    porcelain_position, _ = body_pose(env.sim, PORCELAIN_BODY)
    porcelain_geoms = sorted(
        descendant_geom_ids(model, PORCELAIN_BODY)
    )
    collision_porcelain_geoms = _collision_compatible_geom_ids(
        model,
        porcelain_geoms,
        rigid_gripper_geoms,
    )
    porcelain_gripper_geoms = _collision_compatible_geom_ids(
        model,
        rigid_gripper_geoms,
        collision_porcelain_geoms,
    )
    if (
        not collision_target_geoms
        or not collision_gripper_geoms
        or not collision_porcelain_geoms
        or not porcelain_gripper_geoms
    ):
        raise RuntimeError(
            "compiled target grasp has no gripper/target/porcelain "
            "collision geometry"
        )

    direction_records = [
        {
            "source": "compiled microwave outward direction",
            "direction_xy": outward.tolist(),
        }
    ]
    target_to_porcelain = (
        np.asarray(porcelain_position, dtype=float)[:2]
        - target_position[:2]
    )
    target_to_porcelain_norm = float(
        np.linalg.norm(target_to_porcelain)
    )
    if target_to_porcelain_norm <= np.finfo(float).eps:
        raise RuntimeError(
            "native target and parked porcelain have coincident XY origins"
        )
    target_to_porcelain /= target_to_porcelain_norm
    tangent_directions = [
        np.asarray(
            [-target_to_porcelain[1], target_to_porcelain[0]],
            dtype=float,
        ),
        np.asarray(
            [target_to_porcelain[1], -target_to_porcelain[0]],
            dtype=float,
        ),
    ]
    tangent_directions.sort(
        key=lambda direction: -float(np.dot(direction, outward))
    )
    for tangent_index, direction in enumerate(tangent_directions):
        if any(
            np.allclose(
                direction,
                np.asarray(record["direction_xy"], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
            for record in direction_records
        ):
            continue
        direction_records.append(
            {
                "source": (
                    "target-to-parked-porcelain tangent "
                    f"{tangent_index}"
                ),
                "direction_xy": direction.tolist(),
                "microwave_outward_dot": float(
                    np.dot(direction, outward)
                ),
            }
        )

    gripper_origin_bound = max(
        float(
            np.linalg.norm(
                np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
                - current_eef
            )
            + model.geom_rbound[geom_id]
        )
        for geom_id in collision_gripper_geoms
    )
    target_origin_bound = max(
        float(
            np.linalg.norm(
                np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
                - target_position
            )
            + model.geom_rbound[geom_id]
        )
        for geom_id in collision_target_geoms
    )
    if (
        not np.isfinite(gripper_origin_bound)
        or not np.isfinite(target_origin_bound)
        or gripper_origin_bound <= 0.0
        or target_origin_bound <= 0.0
    ):
        raise RuntimeError("compiled target grasp has invalid origin bounds")

    first_offset = TARGET_GRASP_CLEARANCE_OFFSET
    last_offset = max(
        first_offset + TARGET_INSERTION_SEARCH_STEP_M,
        gripper_origin_bound
        + target_origin_bound
        + EEF_POSITION_TOLERANCE
        + TARGET_INSERTION_SEARCH_STEP_M,
    )
    nominal_eef = target_position + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    trace = []
    for offset in np.arange(
        first_offset,
        last_offset + 0.5 * TARGET_INSERTION_SEARCH_STEP_M,
        TARGET_INSERTION_SEARCH_STEP_M,
    ):
        for direction_index, direction_record in enumerate(
            direction_records
        ):
            direction = np.asarray(
                direction_record["direction_xy"], dtype=float
            )
            clearance_eef = target_position + np.asarray(
                [
                    direction[0] * float(offset),
                    direction[1] * float(offset),
                    GRASP_HEIGHT,
                ]
            )
            clearance_high_eef = clearance_eef + np.asarray(
                [0.0, 0.0, APPROACH_HEIGHT]
            )
            target_approach_clearance, target_approach_sweep = (
                _translated_swept_clearance(
                    env,
                    collision_gripper_geoms,
                    collision_target_geoms,
                    current_eef,
                    clearance_high_eef,
                    current_eef,
                )
            )
            target_descend_clearance, target_descend_sweep = (
                _translated_swept_clearance(
                    env,
                    collision_gripper_geoms,
                    collision_target_geoms,
                    clearance_high_eef,
                    clearance_eef,
                    current_eef,
                )
            )
            fixture_approach_clearance, fixture_approach_sweep = (
                _translated_swept_clearance(
                    env,
                    fixture_compatible_gripper_geoms,
                    fixture_geoms,
                    current_eef,
                    clearance_high_eef,
                    current_eef,
                )
            )
            fixture_descend_clearance, fixture_descend_sweep = (
                _translated_swept_clearance(
                    env,
                    fixture_compatible_gripper_geoms,
                    fixture_geoms,
                    clearance_high_eef,
                    clearance_eef,
                    current_eef,
                )
            )
            fixture_lateral_clearance, fixture_lateral_sweep = (
                _translated_swept_clearance(
                    env,
                    fixture_compatible_gripper_geoms,
                    fixture_geoms,
                    clearance_eef,
                    nominal_eef,
                    current_eef,
                )
            )
            porcelain_approach_clearance, porcelain_approach_sweep = (
                _translated_swept_clearance(
                    env,
                    porcelain_gripper_geoms,
                    collision_porcelain_geoms,
                    current_eef,
                    clearance_high_eef,
                    current_eef,
                )
            )
            porcelain_descend_clearance, porcelain_descend_sweep = (
                _translated_swept_clearance(
                    env,
                    porcelain_gripper_geoms,
                    collision_porcelain_geoms,
                    clearance_high_eef,
                    clearance_eef,
                    current_eef,
                )
            )
            porcelain_lateral_clearance, porcelain_lateral_sweep = (
                _translated_swept_clearance(
                    env,
                    porcelain_gripper_geoms,
                    collision_porcelain_geoms,
                    clearance_eef,
                    nominal_eef,
                    current_eef,
                )
            )
            target_clearances = {
                "target_approach_clearance_m": target_approach_clearance,
                "target_descend_clearance_m": target_descend_clearance,
            }
            fixture_clearances = {
                "fixture_approach_clearance_m": fixture_approach_clearance,
                "fixture_descend_clearance_m": fixture_descend_clearance,
                "fixture_lateral_clearance_m": fixture_lateral_clearance,
            }
            porcelain_clearances = {
                "porcelain_approach_clearance_m": (
                    porcelain_approach_clearance
                ),
                "porcelain_descend_clearance_m": (
                    porcelain_descend_clearance
                ),
                "porcelain_lateral_clearance_m": (
                    porcelain_lateral_clearance
                ),
            }
            passed = bool(
                all(
                    value > EEF_POSITION_TOLERANCE
                    for value in target_clearances.values()
                )
                and all(
                    value > 0.0
                    for value in fixture_clearances.values()
                )
                and all(
                    value > 0.0
                    for value in porcelain_clearances.values()
                )
            )
            record = {
                "direction_index": int(direction_index),
                "direction_source": direction_record["source"],
                "direction_xy": direction.tolist(),
                "outward_offset_m": float(offset),
                "clearance_eef_position": clearance_eef.tolist(),
                "clearance_high_eef_position": (
                    clearance_high_eef.tolist()
                ),
                **target_clearances,
                **fixture_clearances,
                **porcelain_clearances,
                "passed": passed,
            }
            trace.append(record)
            if passed:
                return clearance_eef, {
                    "method": (
                        "nearest compiled-geometry grasp corridor whose "
                        "approach, descend, and lateral seek stay clear of "
                        "the native porcelain mug and microwave"
                    ),
                    "outward_direction_xy": outward.tolist(),
                    "direction_candidates": direction_records,
                    "target_to_porcelain_xy": (
                        target_to_porcelain.tolist()
                    ),
                    "target_to_porcelain_distance_m": (
                        target_to_porcelain_norm
                    ),
                    "porcelain_position": porcelain_position.tolist(),
                    "minimum_outward_offset_m": first_offset,
                    "maximum_outward_offset_m": last_offset,
                    "search_step_m": TARGET_INSERTION_SEARCH_STEP_M,
                    "target_clearance_required_m": (
                        EEF_POSITION_TOLERANCE
                    ),
                    "gripper_origin_bound_m": gripper_origin_bound,
                    "target_origin_bound_m": target_origin_bound,
                    "rigid_gripper_geometry": rigid_gripper_geometry,
                    "collision_gripper_geom_ids": (
                        collision_gripper_geoms
                    ),
                    "collision_target_geom_ids": collision_target_geoms,
                    "collision_porcelain_geom_ids": (
                        collision_porcelain_geoms
                    ),
                    "selected": record,
                    "target_approach_sweep": target_approach_sweep,
                    "target_descend_sweep": target_descend_sweep,
                    "fixture_approach_sweep": fixture_approach_sweep,
                    "fixture_descend_sweep": fixture_descend_sweep,
                    "fixture_lateral_sweep": fixture_lateral_sweep,
                    "porcelain_approach_sweep": (
                        porcelain_approach_sweep
                    ),
                    "porcelain_descend_sweep": (
                        porcelain_descend_sweep
                    ),
                    "porcelain_lateral_sweep": (
                        porcelain_lateral_sweep
                    ),
                    "candidate_trace": trace,
                }
    raise RuntimeError(
        "no compiled no-contact outside target grasp pose; "
        f"candidates={trace}"
    )


def _compiled_safe_insertion_portal(
    env,
    site_position,
    front,
    front_extent,
    floor_surface,
    floor_normal,
    support_offset,
    held_eef_offset,
    current_target,
    current_eef,
    target_geoms,
    target_fixture_geoms,
    gripper_geoms,
    fixture_geoms,
):
    """Derive the nearest safe outside portal and its complete approach."""
    site_position = np.asarray(site_position, dtype=float)
    front = np.asarray(front, dtype=float)
    floor_surface = np.asarray(floor_surface, dtype=float)
    floor_normal = np.asarray(floor_normal, dtype=float)
    held_eef_offset = np.asarray(held_eef_offset, dtype=float)
    current_target = np.asarray(current_target, dtype=float)
    current_eef = np.asarray(current_eef, dtype=float)
    first_distance = 2.0 * float(front_extent)
    current_distance = float(
        np.dot(current_target - site_position, front)
    )
    last_distance = max(
        first_distance + TARGET_INSERTION_SEARCH_STEP_M,
        current_distance,
    )
    lifted_offset = np.asarray(
        [0.0, 0.0, APPROACH_HEIGHT], dtype=float
    )
    current_lifted_target = current_target + lifted_offset
    current_lifted_eef = current_eef + lifted_offset
    lift_gripper_clearance, lift_gripper_sweep = (
        _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            current_eef,
            current_lifted_eef,
            current_eef,
        )
    )
    lift_target_clearance, lift_target_sweep = (
        _translated_swept_clearance(
            env,
            target_geoms,
            target_fixture_geoms,
            current_target,
            current_lifted_target,
            current_target,
        )
    )
    trace = []
    for portal_distance in np.arange(
        first_distance,
        last_distance + 0.5 * TARGET_INSERTION_SEARCH_STEP_M,
        TARGET_INSERTION_SEARCH_STEP_M,
    ):
        portal_object = site_position + front * float(portal_distance)
        portal_object += floor_normal * float(
            np.dot(
                floor_surface
                + floor_normal * float(support_offset)
                - portal_object,
                floor_normal,
            )
        )
        portal_eef = portal_object + held_eef_offset
        portal_high_object = portal_object + lifted_offset
        portal_high_eef = portal_eef + lifted_offset
        (
            transport_gripper_clearance,
            transport_gripper_sweep,
        ) = _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            current_lifted_eef,
            portal_high_eef,
            current_eef,
        )
        (
            transport_target_clearance,
            transport_target_sweep,
        ) = _translated_swept_clearance(
            env,
            target_geoms,
            target_fixture_geoms,
            current_lifted_target,
            portal_high_object,
            current_target,
        )
        (
            alignment_gripper_clearance,
            alignment_gripper_sweep,
        ) = _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            portal_high_eef,
            portal_eef,
            current_eef,
        )
        (
            alignment_target_clearance,
            alignment_target_sweep,
        ) = _translated_swept_clearance(
            env,
            target_geoms,
            target_fixture_geoms,
            portal_high_object,
            portal_object,
            current_target,
        )
        clearances = {
            "lift_gripper_clearance_m": lift_gripper_clearance,
            "lift_target_clearance_m": lift_target_clearance,
            "transport_gripper_clearance_m": (
                transport_gripper_clearance
            ),
            "transport_target_clearance_m": transport_target_clearance,
            "alignment_gripper_clearance_m": (
                alignment_gripper_clearance
            ),
            "alignment_target_clearance_m": alignment_target_clearance,
        }
        passed = bool(all(value > 0.0 for value in clearances.values()))
        record = {
            "front_distance_from_site_center_m": float(portal_distance),
            "portal_object_position": portal_object.tolist(),
            "portal_eef_position": portal_eef.tolist(),
            "portal_high_object_position": portal_high_object.tolist(),
            "portal_high_eef_position": portal_high_eef.tolist(),
            **clearances,
            "passed": passed,
        }
        trace.append(record)
        if passed:
            return (
                portal_object,
                portal_eef,
                portal_high_eef,
                {
                    "method": (
                        "nearest outside front-axis portal whose lifted "
                        "transport and vertical alignment segments have "
                        "positive compiled gripper/target clearance"
                    ),
                    "search_step_m": TARGET_INSERTION_SEARCH_STEP_M,
                    "first_front_distance_m": first_distance,
                    "last_front_distance_m": last_distance,
                    "selected": record,
                    "lift_gripper_sweep": lift_gripper_sweep,
                    "lift_target_sweep": lift_target_sweep,
                    "transport_gripper_sweep": transport_gripper_sweep,
                    "transport_target_sweep": transport_target_sweep,
                    "alignment_gripper_sweep": alignment_gripper_sweep,
                    "alignment_target_sweep": alignment_target_sweep,
                    "candidate_trace": trace,
                },
            )
    raise RuntimeError(
        "no compiled collision-free outside insertion portal; "
        f"candidates={trace}"
    )


def _compiled_target_insertion_plan(
    env,
    names,
    site_position,
    site_rotation,
    site_size,
    held_eef_offset,
    support_geometry,
):
    """Search front-to-back for the foremost native-In, collision-free pose."""
    model = env.sim.model
    site_position = np.asarray(site_position, dtype=float)
    site_rotation = np.asarray(site_rotation, dtype=float)
    site_size = np.asarray(site_size, dtype=float)
    held_eef_offset = np.asarray(held_eef_offset, dtype=float)
    front = -site_rotation[:, 1]
    front = front / np.linalg.norm(front)
    world_half_size = np.abs(site_rotation @ site_size)
    front_extent = min(
        float(world_half_size[axis] / abs(front[axis]))
        for axis in range(3)
        if abs(float(front[axis])) > np.finfo(float).eps
    )
    if not np.isfinite(front_extent) or front_extent <= 0.0:
        raise RuntimeError("native heating site has no finite front extent")

    target_geoms = sorted(descendant_geom_ids(model, TARGET_BODY))
    floor, floor_geometry = _compiled_microwave_floor(
        env,
        names,
        site_position,
        site_rotation,
        target_geoms,
    )
    floor_geom = int(floor["geom_id"])
    floor_center = np.asarray(floor["center"], dtype=float)
    floor_rotation = np.asarray(floor["rotation"], dtype=float)
    floor_half_size = np.asarray(floor["half_size"], dtype=float)
    floor_normal_axis = int(floor["normal_axis"])
    floor_normal = np.asarray(floor["normal"], dtype=float)
    floor_surface = np.asarray(
        floor["surface_position"], dtype=float
    )
    current_target, current_target_rotation = body_pose(
        env.sim, TARGET_BODY
    )
    current_target_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    held_support_geometry = _compiled_held_target_support_geometry(
        env,
        support_geometry,
        floor_geom,
        floor_normal,
    )
    support_offset = float(
        held_support_geometry["held_support_offset_m"]
    )
    current_eef = _eef_position(env)

    (
        collision_gripper_geoms,
        collision_fixture_geoms,
        gripper_geometry,
    ) = _compiled_rigid_gripper_fixture_geoms(env, names)
    fixture_geoms = sorted(
        descendant_geom_ids(model, names["fixture_root"])
    )

    support_target_geoms = support_geometry[
        "supporting_target_geom_ids"
    ]
    floor_tangent_axes = [
        axis for axis in range(3) if axis != floor_normal_axis
    ]
    target_fixture_geoms = [
        geom_id
        for geom_id in fixture_geoms
        if geom_id != floor_geom
    ]
    (
        portal_object,
        portal_eef,
        portal_high_eef,
        portal_geometry,
    ) = _compiled_safe_insertion_portal(
        env,
        site_position,
        front,
        front_extent,
        floor_surface,
        floor_normal,
        support_offset,
        held_eef_offset,
        current_target,
        current_eef,
        target_geoms,
        target_fixture_geoms,
        collision_gripper_geoms,
        collision_fixture_geoms,
    )

    search_values = np.arange(
        np.nextafter(front_extent, 0.0),
        -front_extent - 0.5 * TARGET_INSERTION_SEARCH_STEP_M,
        -TARGET_INSERTION_SEARCH_STEP_M,
    )
    trace = []
    selected = None
    execution_endpoint = None
    for front_distance in search_values:
        candidate = site_position + front * float(front_distance)
        candidate += floor_normal * float(
            np.dot(
                floor_surface + floor_normal * support_offset - candidate,
                floor_normal,
            )
        )
        native_inside = native_site_contains_point(
            site_position,
            site_rotation,
            site_size,
            candidate,
        )
        support_axis_clearances = []
        for geom_id in support_target_geoms:
            geom_center = (
                np.asarray(
                    env.sim.data.geom_xpos[geom_id], dtype=float
                )
                + candidate
                - current_target
            )
            geom_rotation = np.asarray(
                env.sim.data.geom_xmat[geom_id], dtype=float
            ).reshape(3, 3)
            floor_local = floor_rotation.T @ (
                geom_center - floor_center
            )
            for axis in floor_tangent_axes:
                if int(model.geom_type[geom_id]) == 6:
                    extent = float(
                        np.sum(
                            np.asarray(model.geom_size[geom_id], dtype=float)
                            * np.abs(
                                geom_rotation.T
                                @ floor_rotation[:, axis]
                            )
                        )
                    )
                else:
                    extent = float(model.geom_rbound[geom_id])
                support_axis_clearances.append(
                    float(
                        floor_half_size[axis]
                        - abs(float(floor_local[axis]))
                        - extent
                    )
                )
        support_clearance = min(
            support_axis_clearances, default=float("-inf")
        )
        candidate_eef = candidate + held_eef_offset
        gripper_clearance, gripper_sweep = (
            _translated_swept_clearance(
                env,
                collision_gripper_geoms,
                collision_fixture_geoms,
                portal_eef,
                candidate_eef,
                current_eef,
            )
        )
        target_clearance, target_sweep = _translated_swept_clearance(
            env,
            target_geoms,
            target_fixture_geoms,
            portal_object,
            candidate,
            current_target,
        )
        door_clearance, door_sweep = (
            _compiled_target_door_sweep_clearance(
                env,
                names,
                target_geoms,
                candidate,
                current_target,
            )
        )
        passed = bool(
            native_inside
            and support_clearance >= 0.0
            and gripper_clearance > 0.0
            and target_clearance > 0.0
            and door_clearance > 0.0
        )
        record = {
            "front_distance_from_site_center_m": float(
                front_distance
            ),
            "candidate_target_position": candidate.tolist(),
            "candidate_eef_position": candidate_eef.tolist(),
            "native_in": native_inside,
            "support_clearance_m": support_clearance,
            "gripper_swept_clearance_m": gripper_clearance,
            "target_swept_static_clearance_m": target_clearance,
            "target_door_swept_clearance_m": door_clearance,
            "gripper_sweep": gripper_sweep,
            "target_sweep": target_sweep,
            "door_sweep": door_sweep,
            "passed": passed,
        }
        trace.append(record)
        if passed:
            selected = record
            execution_endpoint = record
            break
    if selected is None or execution_endpoint is None:
        raise RuntimeError(
            "no compiled foremost native-In target release pose has "
            "continuous positive gripper/mug/door clearance; "
            f"candidates={trace}"
        )
    return {
        "method": (
            "front-to-back native heating-site search with compiled floor "
            "support and continuous translated gripper/mug sweep clearance"
        ),
        "search_step_m": TARGET_INSERTION_SEARCH_STEP_M,
        "sweep_step_m": TARGET_INSERTION_SWEEP_STEP_M,
        "native_site_position": site_position.tolist(),
        "native_site_rotation": site_rotation.tolist(),
        "native_site_half_size": site_size.tolist(),
        "native_site_world_aabb_half_size": world_half_size.tolist(),
        "target_tilt_at_planning_deg": current_target_tilt,
        "target_rotation_at_planning": current_target_rotation.tolist(),
        "held_tilt_policy": (
            "transient tilt while grasped is diagnostic; actual compiled "
            "geom orientation is used for support and swept clearance; "
            "release still requires the mug at or below MAX_MUG_TILT_DEG"
        ),
        "front_direction": front.tolist(),
        "front_extent_m": front_extent,
        "portal_object_position": portal_object.tolist(),
        "portal_eef_position": portal_eef.tolist(),
        "portal_high_eef_position": portal_high_eef.tolist(),
        "compiled_portal_derivation": portal_geometry,
        "compiled_floor": floor_geometry,
        "source_support": support_geometry,
        "held_pose_floor_support": held_support_geometry,
        "eef_root_body": gripper_geometry["eef_root_body"],
        "rigid_gripper_body_names": gripper_geometry[
            "rigid_gripper_body_names"
        ],
        "collision_gripper_geom_ids": collision_gripper_geoms,
        "collision_fixture_geom_ids": collision_fixture_geoms,
        "selected": selected,
        "execution_endpoint": execution_endpoint,
        "execution_reserve_m": 0.0,
        "execution_endpoint_derivation": (
            "the selected safe release pose itself; runtime verifies actual "
            "target arrival and fails closed instead of commanding a "
            "fictional deeper overshoot"
        ),
        "candidate_trace": trace,
    }


def _compiled_open_gripper_retreat_plan(env, names, waypoints):
    """Validate the just-opened gripper's complete portal retreat sweep."""
    (
        gripper_geoms,
        fixture_geoms,
        geometry,
    ) = _compiled_rigid_gripper_fixture_geoms(env, names)
    reference = _eef_position(env)
    start = reference.copy()
    segments = []
    minimum = float("inf")
    for label, endpoint in waypoints:
        endpoint = np.asarray(endpoint, dtype=float)
        clearance, sweep = _translated_swept_clearance(
            env,
            gripper_geoms,
            fixture_geoms,
            start,
            endpoint,
            reference,
        )
        segments.append(
            {
                "label": str(label),
                "clearance_m": clearance,
                "sweep": sweep,
            }
        )
        minimum = min(minimum, clearance)
        start = endpoint
    passed = bool(segments and minimum > 0.0)
    return passed, {
        "method": (
            "compiled continuous translated sweep of the actual open "
            "gripper through horizontal portal retreat and exit"
        ),
        "minimum_clearance_m": minimum,
        "passed": passed,
        "gripper_geometry": geometry,
        "segments": segments,
    }


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
    forbid_microwave_contact=False,
):
    target = np.asarray(target, dtype=float)
    initial_eef = _eef_position(env)
    initial_error = target - initial_eef
    error_norms = [float(np.linalg.norm(initial_error))]
    contact_bodies = _robot_contact_body_names(env)
    contact_pairs = {
        (
            pair["robot_geom_id"],
            pair["other_geom_id"],
        ): pair
        for pair in _robot_contact_pairs(env)
    }
    trace = []
    reached = False
    status = None
    forbidden_contact = bool(
        forbid_microwave_contact
        and _has_microwave_contact(contact_bodies)
    )
    for iteration in range(0 if forbidden_contact else MOVE_STEPS):
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
        for pair in _robot_contact_pairs(env):
            contact_pairs[
                (pair["robot_geom_id"], pair["other_geom_id"])
            ] = pair
        current_microwave_contact = _has_microwave_contact(current_contacts)
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
                float(current_microwave_contact),
            ]
        )
        if forbid_microwave_contact and current_microwave_contact:
            forbidden_contact = True
            break
        if status.violated:
            break
    final_eef = _eef_position(env)
    final_error = target - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    reached = bool(
        not forbidden_contact
        and (reached or final_error_norm <= EEF_POSITION_TOLERANCE)
    )
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
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "porcelain_contact_seen": PORCELAIN_BODY in contact_bodies,
        "target_contact_seen": TARGET_BODY in contact_bodies,
        "microwave_contact_seen": _has_microwave_contact(contact_bodies),
        "forbid_microwave_contact": forbid_microwave_contact,
        "forbidden_microwave_contact": forbidden_contact,
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


def _seek_porcelain_contact(
    env,
    oracle,
    step,
    frames,
):
    """Move laterally from the clearance waypoint until safe mug contact."""
    initial_eef = _eef_position(env)
    contact_bodies = _robot_contact_body_names(env)
    porcelain_contact = PORCELAIN_BODY in contact_bodies
    microwave_contact = _has_microwave_contact(contact_bodies)
    trace = []
    status = None
    for iteration in range(
        0
        if porcelain_contact or microwave_contact
        else PORCELAIN_CONTACT_SEEK_STEPS
    ):
        mug_position, _ = body_pose(env.sim, PORCELAIN_BODY)
        eef = _eef_position(env)
        target = mug_position + np.asarray(
            [0.0, 0.0, PORCELAIN_GRASP_HEIGHT]
        )
        error = target - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(
            error * PORCELAIN_CONTACT_SEEK_GAIN,
            -PORCELAIN_CONTACT_SEEK_ACTION_LIMIT,
            PORCELAIN_CONTACT_SEEK_ACTION_LIMIT,
        )
        action[-1] = -1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        current_porcelain = PORCELAIN_BODY in current_contacts
        current_microwave = _has_microwave_contact(current_contacts)
        trace.append(
            [
                float(iteration),
                float(step),
                *target.tolist(),
                *eef.tolist(),
                *(target - eef).tolist(),
                *action[:3].tolist(),
                float(current_porcelain),
                float(current_microwave),
            ]
        )
        # Forbidden fixture contact wins even if the same step also reaches
        # the mug.  Job 499647 demonstrated that simultaneous contact is a
        # wedged, invalid grasp state.
        if current_microwave:
            microwave_contact = True
            break
        if status.violated:
            break
        if current_porcelain:
            porcelain_contact = True
            break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    porcelain_contact = bool(
        porcelain_contact and PORCELAIN_BODY in final_contacts
    )
    success = bool(
        porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "lateral contact seek",
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": _eef_position(env).tolist(),
        "steps_executed": len(trace),
        "success": success,
        "porcelain_contact": porcelain_contact,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "trace_columns": (
            "iteration,global_step,target_x,target_y,target_z,eef_x,eef_y,"
            "eef_z,error_x,error_y,error_z,action_x,action_y,action_z,"
            "porcelain_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during porcelain contact seek"
    elif status is not None and status.violated:
        reason = "oracle violation during porcelain contact seek"
    elif not porcelain_contact:
        reason = "porcelain contact seek ended without current mug contact"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _close_gripper_on_porcelain(env, oracle, step, frames):
    """Close only from a real mug-only contact and preserve that safety."""
    initial_contacts = _robot_contact_body_names(env)
    porcelain_initial = PORCELAIN_BODY in initial_contacts
    microwave_contact = _has_microwave_contact(initial_contacts)
    porcelain_seen = porcelain_initial
    contact_bodies = set(initial_contacts)
    trace = []
    status = None
    if porcelain_initial and not microwave_contact:
        action = np.zeros(7, dtype=float)
        action[-1] = 1.0
        for iteration in range(GRIPPER_STEPS):
            _, status, step = _step(env, oracle, action, step, frames)
            contacts = _robot_contact_body_names(env)
            contact_bodies.update(contacts)
            current_porcelain = PORCELAIN_BODY in contacts
            current_microwave = _has_microwave_contact(contacts)
            porcelain_seen = porcelain_seen or current_porcelain
            microwave_contact = microwave_contact or current_microwave
            trace.append(
                [
                    float(iteration),
                    float(step),
                    float(current_porcelain),
                    float(current_microwave),
                ]
            )
            if current_microwave or status.violated:
                break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    porcelain_final = PORCELAIN_BODY in final_contacts
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    success = bool(
        porcelain_initial
        and porcelain_seen
        and porcelain_final
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "porcelain grasp closure",
        "success": success,
        "porcelain_contact_initial": porcelain_initial,
        "porcelain_contact_seen": porcelain_seen,
        "porcelain_contact_final": porcelain_final,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "steps_executed": len(trace),
        "trace_columns": (
            "iteration,global_step,porcelain_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave before/during porcelain closure"
    elif not porcelain_initial:
        reason = "porcelain closure attempted without initial mug contact"
    elif status is not None and status.violated:
        reason = "oracle violation during porcelain closure"
    elif not porcelain_final:
        reason = "porcelain contact was not retained after closure"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _descend_to_target_contact(env, oracle, step, frames):
    """Descend until the gripper really contacts the native target mug."""
    initial_eef = _eef_position(env)
    initial_target, _ = body_pose(env.sim, TARGET_BODY)
    initial_waypoint = initial_target + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    initial_error = initial_waypoint - initial_eef
    error_norms = [float(np.linalg.norm(initial_error))]
    contact_bodies = _robot_contact_body_names(env)
    target_contact = TARGET_BODY in contact_bodies
    target_contact_initial = target_contact
    microwave_contact = _has_microwave_contact(contact_bodies)
    reached_tolerance = bool(
        error_norms[-1] <= EEF_POSITION_TOLERANCE
    )
    trace = []
    status = None
    for iteration in range(
        0 if target_contact or microwave_contact else MOVE_STEPS
    ):
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        eef = _eef_position(env)
        error = waypoint - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(error * 20.0, -1.0, 1.0)
        action[-1] = -1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        post_error = waypoint - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        reached_tolerance = bool(
            reached_tolerance
            or error_norm <= EEF_POSITION_TOLERANCE
        )
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        current_target = TARGET_BODY in current_contacts
        current_microwave = _has_microwave_contact(current_contacts)
        trace.append(
            [
                float(iteration),
                float(step),
                *waypoint.tolist(),
                *eef.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                float(current_target),
                float(current_porcelain),
                float(current_microwave),
            ]
        )
        # Fixture contact always wins, including a simultaneous target contact.
        if current_microwave:
            microwave_contact = True
            break
        if status.violated:
            break
        if current_target:
            target_contact = True
            break
    final_target, _ = body_pose(env.sim, TARGET_BODY)
    final_waypoint = final_target + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    final_eef = _eef_position(env)
    final_error = final_waypoint - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    target_contact_final = TARGET_BODY in final_contacts
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    target_contact = bool(target_contact and target_contact_final)
    tail = error_norms[-min(20, len(error_norms)):]
    stalled = bool(
        not target_contact
        and len(tail) >= 2
        and max(tail) - min(tail) < 0.001
    )
    horizon_exhausted = bool(
        len(trace) >= MOVE_STEPS
        and not target_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    success = bool(
        target_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target contact descend",
        "success": success,
        "initial_target_waypoint": initial_waypoint.tolist(),
        "final_target_waypoint": final_waypoint.tolist(),
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "initial_error_vector": initial_error.tolist(),
        "initial_error_m": float(np.linalg.norm(initial_error)),
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms),
        "steps_executed": len(trace),
        "reached_eef_tolerance": reached_tolerance,
        "stalled": stalled,
        "horizon_exhausted": horizon_exhausted,
        "target_contact_initial": target_contact_initial,
        "target_contact": target_contact,
        "target_contact_final": target_contact_final,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "trace_columns": (
            "iteration,global_step,target_x,target_y,target_z,eef_x,eef_y,"
            "eef_z,error_x,error_y,error_z,error_norm,action_x,action_y,"
            "action_z,target_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during target contact descend"
    elif status is not None and status.violated:
        reason = "oracle violation during target contact descend"
    elif not target_contact:
        reason = (
            "target contact descend ended without current mug contact; "
            f"final_error_m={final_error_norm}; "
            f"stalled={stalled}; horizon_exhausted={horizon_exhausted}; "
            f"contacts={sorted(contact_bodies)}"
        )
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _seek_target_contact(env, oracle, step, frames):
    """Approach the native target laterally at its nominal grasp height."""
    initial_eef = _eef_position(env)
    contact_bodies = _robot_contact_body_names(env)
    target_contact = TARGET_BODY in contact_bodies
    target_contact_initial = target_contact
    porcelain_contact = PORCELAIN_BODY in contact_bodies
    microwave_contact = _has_microwave_contact(contact_bodies)
    contact_pairs = {
        (pair["robot_geom_id"], pair["other_geom_id"]): pair
        for pair in _robot_contact_pairs(env)
    }
    trace = []
    error_norms = []
    status = None
    for iteration in range(
        0
        if target_contact or porcelain_contact or microwave_contact
        else TARGET_CONTACT_SEEK_STEPS
    ):
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        eef = _eef_position(env)
        error = waypoint - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(
            error * TARGET_CONTACT_SEEK_GAIN,
            -TARGET_CONTACT_SEEK_ACTION_LIMIT,
            TARGET_CONTACT_SEEK_ACTION_LIMIT,
        )
        action[-1] = -1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        waypoint = target_position + np.asarray(
            [0.0, 0.0, GRASP_HEIGHT]
        )
        post_error = waypoint - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        current_contacts = _robot_contact_body_names(env)
        contact_bodies.update(current_contacts)
        for pair in _robot_contact_pairs(env):
            contact_pairs[
                (pair["robot_geom_id"], pair["other_geom_id"])
            ] = pair
        current_target = TARGET_BODY in current_contacts
        current_porcelain = PORCELAIN_BODY in current_contacts
        current_microwave = _has_microwave_contact(current_contacts)
        trace.append(
            [
                float(iteration),
                float(step),
                *waypoint.tolist(),
                *eef.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                float(current_target),
                float(current_microwave),
            ]
        )
        if current_microwave:
            microwave_contact = True
            break
        if current_porcelain:
            porcelain_contact = True
            break
        if status.violated:
            break
        if current_target:
            target_contact = True
            break
    final_target, _ = body_pose(env.sim, TARGET_BODY)
    final_waypoint = final_target + np.asarray(
        [0.0, 0.0, GRASP_HEIGHT]
    )
    final_eef = _eef_position(env)
    final_error = final_waypoint - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    for pair in _robot_contact_pairs(env):
        contact_pairs[
            (pair["robot_geom_id"], pair["other_geom_id"])
        ] = pair
    target_contact = bool(
        target_contact and TARGET_BODY in final_contacts
    )
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    porcelain_contact = bool(
        porcelain_contact or PORCELAIN_BODY in contact_bodies
    )
    horizon_exhausted = bool(
        len(trace) >= TARGET_CONTACT_SEEK_STEPS
        and not target_contact
        and not porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    success = bool(
        target_contact
        and not target_contact_initial
        and not porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target lateral contact seek",
        "success": success,
        "method": (
            "descend outside the mug, then seek laterally at the native "
            "target's nominal grasp height"
        ),
        "nominal_grasp_height_m": GRASP_HEIGHT,
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "final_target_waypoint": final_waypoint.tolist(),
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms, default=final_error_norm),
        "steps_executed": len(trace),
        "target_contact_initial": target_contact_initial,
        "target_contact": target_contact,
        "target_contact_final": TARGET_BODY in final_contacts,
        "porcelain_contact_seen": porcelain_contact,
        "microwave_contact_seen": microwave_contact,
        "horizon_exhausted": horizon_exhausted,
        "robot_contact_bodies": sorted(contact_bodies),
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "trace_columns": (
            "iteration,global_step,target_x,target_y,target_z,eef_x,eef_y,"
            "eef_z,error_x,error_y,error_z,error_norm,action_x,action_y,"
            "action_z,target_contact,porcelain_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during target lateral contact seek"
    elif porcelain_contact:
        reason = (
            "robot contacted the parked porcelain mug during target "
            "lateral contact seek"
        )
    elif target_contact_initial:
        reason = (
            "target was already in contact before lateral contact seek; "
            "compiled outside grasp clearance was not realized"
        )
    elif status is not None and status.violated:
        reason = "oracle violation during target lateral contact seek"
    elif not target_contact:
        reason = (
            "target lateral contact seek ended without current mug contact; "
            f"final_error_m={final_error_norm}; "
            f"horizon_exhausted={horizon_exhausted}; "
            f"contacts={sorted(contact_bodies)}"
        )
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _close_gripper_on_target(env, oracle, step, frames):
    """Close only from real target contact and retain it after closure."""
    initial_contacts = _robot_contact_body_names(env)
    initial_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    target_initial = TARGET_BODY in initial_contacts
    porcelain_contact = PORCELAIN_BODY in initial_contacts
    microwave_contact = _has_microwave_contact(initial_contacts)
    target_seen = target_initial
    contact_bodies = set(initial_contacts)
    contact_pairs = {
        (pair["robot_geom_id"], pair["other_geom_id"]): pair
        for pair in _robot_contact_pairs(env)
    }
    trace = []
    status = None
    if target_initial and not porcelain_contact and not microwave_contact:
        action = np.zeros(7, dtype=float)
        action[-1] = 1.0
        for iteration in range(GRIPPER_STEPS):
            _, status, step = _step(env, oracle, action, step, frames)
            contacts = _robot_contact_body_names(env)
            contact_bodies.update(contacts)
            for pair in _robot_contact_pairs(env):
                contact_pairs[
                    (pair["robot_geom_id"], pair["other_geom_id"])
                ] = pair
            current_target = TARGET_BODY in contacts
            current_porcelain = PORCELAIN_BODY in contacts
            current_microwave = _has_microwave_contact(contacts)
            current_tilt = body_tilt_deg(env.sim, TARGET_BODY)
            target_seen = target_seen or current_target
            microwave_contact = microwave_contact or current_microwave
            porcelain_contact = porcelain_contact or current_porcelain
            trace.append(
                [
                    float(iteration),
                    float(step),
                    float(current_target),
                    float(current_porcelain),
                    float(current_microwave),
                    current_tilt,
                ]
            )
            if current_porcelain or current_microwave or status.violated:
                break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    for pair in _robot_contact_pairs(env):
        contact_pairs[
            (pair["robot_geom_id"], pair["other_geom_id"])
        ] = pair
    target_final = TARGET_BODY in final_contacts
    final_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    porcelain_contact = bool(
        porcelain_contact or PORCELAIN_BODY in contact_bodies
    )
    success = bool(
        target_initial
        and target_seen
        and target_final
        and not porcelain_contact
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target grasp closure",
        "success": success,
        "target_contact_initial": target_initial,
        "target_contact_seen": target_seen,
        "target_contact_final": target_final,
        "target_tilt_initial_deg": initial_tilt,
        "target_tilt_final_deg": final_tilt,
        "porcelain_contact_seen": porcelain_contact,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "steps_executed": len(trace),
        "trace_columns": (
            "iteration,global_step,target_contact,porcelain_contact,"
            "microwave_contact,target_tilt_deg"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave before/during target closure"
    elif porcelain_contact:
        reason = (
            "robot contacted the parked porcelain mug before/during "
            "target closure"
        )
    elif not target_initial:
        reason = "target closure attempted without initial mug contact"
    elif status is not None and status.violated:
        reason = "oracle violation during target closure"
    elif not target_final:
        reason = "target contact was not retained after closure"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _insert_target_until_safe_release(
    env,
    oracle,
    names,
    endpoint_eef,
    held_eef_offset,
    site_position,
    site_rotation,
    site_size,
    floor_geom_id,
    front_direction,
    maximum_safe_front_distance,
    step,
    frames,
):
    """Enter until the actual mug is native-In, floor-supported, door-clear."""
    endpoint = np.asarray(endpoint_eef, dtype=float)
    held_offset = np.asarray(held_eef_offset, dtype=float)
    front = np.asarray(front_direction, dtype=float)
    front = front / np.linalg.norm(front)
    maximum_safe_front_distance = float(maximum_safe_front_distance)
    target_geoms = sorted(
        descendant_geom_ids(env.sim.model, TARGET_BODY)
    )
    floor_geoms = {int(floor_geom_id)}
    initial_eef = _eef_position(env)
    initial_target, _ = body_pose(env.sim, TARGET_BODY)
    error_norms = [
        float(np.linalg.norm(endpoint - initial_eef))
    ]
    contact_bodies = _robot_contact_body_names(env)
    contact_pairs = {
        (
            pair["robot_geom_id"],
            pair["other_geom_id"],
        ): pair
        for pair in _robot_contact_pairs(env)
    }
    microwave_contact = _has_microwave_contact(contact_bodies)
    trace = []
    status = None
    success = False
    final_door_clearance = float("nan")
    final_follow_error = float("nan")
    for iteration in range(0 if microwave_contact else MOVE_STEPS):
        eef = _eef_position(env)
        error = endpoint - eef
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(error * 20.0, -1.0, 1.0)
        action[-1] = 1.0
        _, status, step = _step(env, oracle, action, step, frames)
        eef = _eef_position(env)
        target_position, _ = body_pose(env.sim, TARGET_BODY)
        post_error = endpoint - eef
        error_norm = float(np.linalg.norm(post_error))
        error_norms.append(error_norm)
        contacts = _robot_contact_body_names(env)
        contact_bodies.update(contacts)
        for pair in _robot_contact_pairs(env):
            contact_pairs[
                (pair["robot_geom_id"], pair["other_geom_id"])
            ] = pair
        current_microwave = _has_microwave_contact(contacts)
        microwave_contact = microwave_contact or current_microwave
        expected_target = eef - held_offset
        follow_error = float(
            np.linalg.norm(target_position - expected_target)
        )
        final_follow_error = follow_error
        target_tilt = body_tilt_deg(env.sim, TARGET_BODY)
        target_linear, target_angular = body_speeds(
            env.sim, TARGET_BODY
        )
        native_inside = native_site_contains_point(
            site_position,
            site_rotation,
            site_size,
            target_position,
        )
        floor_contact = contacts_between(
            env.sim, target_geoms, floor_geoms
        )
        front_distance = float(
            np.dot(
                target_position
                - np.asarray(site_position, dtype=float),
                front,
            )
        )
        foremost_pose_reached = bool(
            front_distance
            <= maximum_safe_front_distance
            + 64.0 * np.finfo(float).eps
        )
        door_clearance, _ = _compiled_target_door_sweep_clearance(
            env,
            names,
            target_geoms,
            target_position,
            target_position,
        )
        final_door_clearance = door_clearance
        trace.append(
            [
                float(iteration),
                float(step),
                *endpoint.tolist(),
                *eef.tolist(),
                *target_position.tolist(),
                *post_error.tolist(),
                error_norm,
                *action[:3].tolist(),
                follow_error,
                target_tilt,
                target_linear,
                target_angular,
                float(native_inside),
                float(floor_contact),
                front_distance,
                float(foremost_pose_reached),
                door_clearance,
                float(current_microwave),
            ]
        )
        if current_microwave:
            break
        if status.violated:
            break
        if follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
            break
        if (
            native_inside
            and floor_contact
            and foremost_pose_reached
            and door_clearance > 0.0
            and target_tilt <= MAX_MUG_TILT_DEG
            and target_linear <= MAX_WAIT_LINEAR_SPEED_MPS
            and target_angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        ):
            success = True
            break
    final_eef = _eef_position(env)
    final_target, _ = body_pose(env.sim, TARGET_BODY)
    final_error = endpoint - final_eef
    final_error_norm = float(np.linalg.norm(final_error))
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    for pair in _robot_contact_pairs(env):
        contact_pairs[
            (pair["robot_geom_id"], pair["other_geom_id"])
        ] = pair
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    final_native_inside = native_site_contains_point(
        site_position,
        site_rotation,
        site_size,
        final_target,
    )
    final_floor_contact = contacts_between(
        env.sim, target_geoms, floor_geoms
    )
    final_target_tilt = body_tilt_deg(env.sim, TARGET_BODY)
    final_target_linear, final_target_angular = body_speeds(
        env.sim, TARGET_BODY
    )
    final_front_distance = float(
        np.dot(
            final_target - np.asarray(site_position, dtype=float),
            front,
        )
    )
    final_foremost_pose_reached = bool(
        final_front_distance
        <= maximum_safe_front_distance
        + 64.0 * np.finfo(float).eps
    )
    if not np.isfinite(final_door_clearance):
        final_door_clearance, _ = (
            _compiled_target_door_sweep_clearance(
                env,
                names,
                target_geoms,
                final_target,
                final_target,
            )
        )
    if not np.isfinite(final_follow_error):
        final_follow_error = float(
            np.linalg.norm(
                final_target - (final_eef - held_offset)
            )
        )
    success = bool(
        success
        and final_native_inside
        and final_floor_contact
        and final_foremost_pose_reached
        and final_door_clearance > 0.0
        and final_target_tilt <= MAX_MUG_TILT_DEG
        and final_target_linear <= MAX_WAIT_LINEAR_SPEED_MPS
        and final_target_angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        and final_follow_error <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    tail = error_norms[-min(20, len(error_norms)):]
    stalled = bool(
        not success
        and len(tail) >= 2
        and max(tail) - min(tail) < 0.001
    )
    horizon_exhausted = bool(
        len(trace) >= MOVE_STEPS
        and not success
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target insertion",
        "success": success,
        "target_position": endpoint.tolist(),
        "initial_eef_position": initial_eef.tolist(),
        "final_eef_position": final_eef.tolist(),
        "initial_target_position": initial_target.tolist(),
        "final_target_position": final_target.tolist(),
        "initial_error_m": error_norms[0],
        "final_error_vector": final_error.tolist(),
        "final_error_m": final_error_norm,
        "min_error_m": min(error_norms),
        "steps_executed": len(trace),
        "reached": success,
        "stalled": stalled,
        "horizon_exhausted": horizon_exhausted,
        "native_in_final": final_native_inside,
        "floor_contact_final": final_floor_contact,
        "front_distance_final_m": final_front_distance,
        "maximum_safe_front_distance_m": maximum_safe_front_distance,
        "foremost_pose_reached": final_foremost_pose_reached,
        "door_swept_clearance_final_m": final_door_clearance,
        "object_follow_error_final_m": final_follow_error,
        "target_tilt_final_deg": final_target_tilt,
        "target_linear_speed_final_mps": final_target_linear,
        "target_angular_speed_final_radps": final_target_angular,
        "robot_contact_bodies": sorted(contact_bodies),
        "robot_contact_pairs": [
            contact_pairs[key] for key in sorted(contact_pairs)
        ],
        "microwave_contact_seen": microwave_contact,
        "forbid_microwave_contact": True,
        "forbidden_microwave_contact": microwave_contact,
        "trace_columns": (
            "iteration,global_step,endpoint_x,endpoint_y,endpoint_z,eef_x,"
            "eef_y,eef_z,target_x,target_y,target_z,error_x,error_y,error_z,"
            "error_norm,action_x,action_y,action_z,object_follow_error,"
            "target_tilt_deg,target_linear_speed,target_angular_speed,"
            "native_in,floor_contact,front_distance,foremost_pose_reached,"
            "door_swept_clearance,"
            "microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave during target insertion"
    elif status is not None and status.violated:
        reason = "oracle violation during target insertion"
    elif final_follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
        reason = (
            "target mug stopped following during target insertion; "
            f"follow_error_m={final_follow_error}"
        )
    elif final_target_tilt > MAX_MUG_TILT_DEG:
        reason = (
            "target mug tilted before release; "
            f"tilt_deg={final_target_tilt}"
        )
    elif (
        final_target_linear > MAX_WAIT_LINEAR_SPEED_MPS
        or final_target_angular > MAX_WAIT_ANGULAR_SPEED_RADPS
    ):
        reason = (
            "target mug remained dynamic before release; "
            f"linear_mps={final_target_linear}; "
            f"angular_radps={final_target_angular}"
        )
    elif not success:
        reason = (
            "target insertion ended without a safe actual release state; "
            f"native_in={final_native_inside}; "
            f"floor_contact={final_floor_contact}; "
            f"foremost_pose_reached={final_foremost_pose_reached}; "
            f"door_clearance_m={final_door_clearance}; "
            f"final_error_m={final_error_norm}; "
            f"stalled={stalled}; horizon_exhausted={horizon_exhausted}"
        )
    else:
        reason = ""
    return success, reason, status, step, diagnostic


def _release_target_without_microwave_contact(
    env,
    oracle,
    step,
    frames,
):
    """Open the gripper while any robot-microwave contact fails closed."""
    initial_contacts = _robot_contact_body_names(env)
    microwave_contact = _has_microwave_contact(initial_contacts)
    contact_bodies = set(initial_contacts)
    trace = []
    status = None
    if not microwave_contact:
        action = np.zeros(7, dtype=float)
        action[-1] = -1.0
        for iteration in range(GRIPPER_STEPS):
            _, status, step = _step(env, oracle, action, step, frames)
            contacts = _robot_contact_body_names(env)
            contact_bodies.update(contacts)
            current_microwave = _has_microwave_contact(contacts)
            microwave_contact = microwave_contact or current_microwave
            trace.append(
                [
                    float(iteration),
                    float(step),
                    float(TARGET_BODY in contacts),
                    float(current_microwave),
                ]
            )
            if current_microwave or status.violated:
                break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    success = bool(
        not microwave_contact
        and not (status is not None and status.violated)
        and len(trace) == GRIPPER_STEPS
    )
    diagnostic = {
        "label": "target release",
        "success": success,
        "microwave_contact_initial": _has_microwave_contact(
            initial_contacts
        ),
        "microwave_contact_seen": microwave_contact,
        "target_contact_initial": TARGET_BODY in initial_contacts,
        "target_contact_final": TARGET_BODY in final_contacts,
        "robot_contact_bodies": sorted(contact_bodies),
        "steps_executed": len(trace),
        "trace_columns": (
            "iteration,global_step,target_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave before/during target release"
    elif status is not None and status.violated:
        reason = "oracle violation during target release"
    elif len(trace) != GRIPPER_STEPS:
        reason = "target release did not execute the full gripper command"
    else:
        reason = ""
    return success, reason, status, step, diagnostic


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
    paired_counterfactual_park_position,
    names,
    support_body,
    frames,
    step,
):
    initial_mug, _ = body_pose(env.sim, PORCELAIN_BODY)
    paired_counterfactual_park_position = np.asarray(
        paired_counterfactual_park_position, dtype=float
    )
    try:
        clearance_xy, clearance_geometry = _compiled_microwave_clearance(
            env, names, initial_mug
        )
        park_mug_position, safe_park_geometry = (
            _compiled_safe_outward_park(
                env,
                names,
                support_body,
                initial_mug,
                clearance_xy,
            )
        )
    except RuntimeError as error:
        return False, str(error), None, step, {}
    clearance_offset = np.asarray(
        [
            clearance_xy[0] * PORCELAIN_GRASP_CLEARANCE_OFFSET,
            clearance_xy[1] * PORCELAIN_GRASP_CLEARANCE_OFFSET,
            PORCELAIN_GRASP_HEIGHT,
        ]
    )
    clearance_grasp_point = initial_mug + clearance_offset
    move_diagnostics = []
    contact_seek_diagnostic = {}
    closure_diagnostic = {}
    held_eef_offset = None
    park_grasp_point = None
    object_follow_trace = []
    park_error_before_release = None
    final_park_error = None

    def prefix_metrics():
        return {
            "porcelain_initial_position": initial_mug.tolist(),
            "porcelain_park_position": park_mug_position.tolist(),
            "paired_counterfactual_park_position_not_used_for_path": (
                paired_counterfactual_park_position.tolist()
            ),
            "porcelain_clearance_grasp_target": (
                clearance_grasp_point.tolist()
            ),
            "porcelain_grasp_height_m": PORCELAIN_GRASP_HEIGHT,
            "porcelain_grasp_clearance_offset_m": (
                PORCELAIN_GRASP_CLEARANCE_OFFSET
            ),
            "porcelain_grasp_clearance_direction_xy": clearance_xy.tolist(),
            "compiled_clearance_geometry": clearance_geometry,
            "compiled_safe_park_geometry": safe_park_geometry,
            "contact_seek": contact_seek_diagnostic,
            "grasp_closure": closure_diagnostic,
            "held_eef_minus_mug_offset": (
                None
                if held_eef_offset is None
                else held_eef_offset.tolist()
            ),
            "porcelain_park_eef_target": (
                None
                if park_grasp_point is None
                else park_grasp_point.tolist()
            ),
            "object_follow_tolerance_m": (
                PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
            ),
            "object_follow_trace": object_follow_trace,
            "park_error_before_release_m": park_error_before_release,
            "final_park_error_m": final_park_error,
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
            f"forbidden_microwave_contact="
            f"{diagnostic.get('forbidden_microwave_contact', False)}; "
            f"contacts={diagnostic.get('robot_contact_bodies', [])}"
        )

    waypoints = (
        (
            clearance_grasp_point + [0.0, 0.0, APPROACH_HEIGHT],
            -1.0,
            "approach",
        ),
        (clearance_grasp_point, -1.0, "descend"),
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
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                prefix_metrics(),
            )
    (
        contact_ok,
        contact_reason,
        status,
        step,
        contact_seek_diagnostic,
    ) = _seek_porcelain_contact(
        env,
        oracle,
        step,
        frames,
    )
    if not contact_ok:
        return (
            False,
            contact_reason,
            status,
            step,
            prefix_metrics(),
        )
    (
        closure_ok,
        closure_reason,
        status,
        step,
        closure_diagnostic,
    ) = _close_gripper_on_porcelain(env, oracle, step, frames)
    if not closure_ok:
        return (
            False,
            closure_reason,
            status,
            step,
            prefix_metrics(),
        )
    grasped_mug_position, _ = body_pose(env.sim, PORCELAIN_BODY)
    grasped_eef_position = _eef_position(env)
    held_eef_offset = grasped_eef_position - grasped_mug_position
    park_grasp_point = park_mug_position + held_eef_offset
    for target, label in (
        (
            grasped_eef_position + [0.0, 0.0, APPROACH_HEIGHT],
            "lift",
        ),
        (
            park_grasp_point + [0.0, 0.0, APPROACH_HEIGHT],
            "outward corridor",
        ),
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
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                prefix_metrics(),
            )
        current_eef = _eef_position(env)
        current_mug, _ = body_pose(env.sim, PORCELAIN_BODY)
        expected_mug = current_eef - held_eef_offset
        follow_error = float(np.linalg.norm(current_mug - expected_mug))
        object_follow_trace.append(
            {
                "label": label,
                "eef_position": current_eef.tolist(),
                "expected_mug_position": expected_mug.tolist(),
                "actual_mug_position": current_mug.tolist(),
                "error_m": follow_error,
                "passed": (
                    follow_error
                    <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
                ),
            }
        )
        if follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
            return (
                False,
                f"porcelain mug stopped following during {label}; "
                f"follow_error_m={follow_error}",
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
    park_error_before_release = float(
        np.linalg.norm(moved_position - park_mug_position)
    )
    if park_error_before_release > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
        return (
            False,
            "porcelain mug did not reach the compiled safe park pose; "
            f"park_error_m={park_error_before_release}",
            status,
            step,
            prefix_metrics(),
        )
    status, step = _hold_gripper(
        env, oracle, -1.0, GRIPPER_STEPS, step, frames
    )
    if status is not None and status.violated:
        return (
            False,
            "oracle violation while releasing parked porcelain mug",
            status,
            step,
            prefix_metrics(),
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
        forbid_microwave_contact=True,
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
    final_park_error = float(np.linalg.norm(final_mug - park_mug_position))
    final_linear, final_angular = body_speeds(env.sim, PORCELAIN_BODY)
    contacts = contact_body_names(env.sim, PORCELAIN_BODY)
    stable = bool(
        np.linalg.norm(final_mug - initial_mug) >= 0.025
        and final_park_error <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
        and support_body in contacts
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
    status = None
    move_diagnostics = []
    contact_descend_diagnostic = {}
    closure_diagnostic = {}
    release_diagnostic = {}
    support_geometry = {}
    target_clearance_geometry = {}
    target_grasp_clearance_derivation = {}
    insertion_plan = {}
    retreat_plan = {}
    held_eef_offset = None
    clearance_grasp_point = None
    target_grasp_point = None
    target_base = None
    object_follow_trace = []
    moved_before_release = None

    def target_metrics():
        return {
            "target_initial_position": initial_target.tolist(),
            "target_nominal_grasp_point": grasp_point.tolist(),
            "target_clearance_grasp_target": (
                None
                if clearance_grasp_point is None
                else clearance_grasp_point.tolist()
            ),
            "target_grasp_clearance_offset_m": (
                None
                if not target_grasp_clearance_derivation
                else target_grasp_clearance_derivation["selected"][
                    "outward_offset_m"
                ]
            ),
            "target_grasp_minimum_clearance_offset_m": (
                TARGET_GRASP_CLEARANCE_OFFSET
            ),
            "compiled_target_clearance_geometry": (
                target_clearance_geometry
            ),
            "compiled_target_grasp_clearance_derivation": (
                target_grasp_clearance_derivation
            ),
            "target_desired_base_position": (
                None if target_base is None else target_base.tolist()
            ),
            "target_contact_acquisition": contact_descend_diagnostic,
            "target_contact_descend": contact_descend_diagnostic,
            "target_grasp_closure": closure_diagnostic,
            "target_release": release_diagnostic,
            "target_native_support_geometry": support_geometry,
            "compiled_insertion_plan": insertion_plan,
            "compiled_open_gripper_retreat_plan": retreat_plan,
            "held_eef_minus_target_offset": (
                None
                if held_eef_offset is None
                else held_eef_offset.tolist()
            ),
            "target_placement_eef_target": (
                None
                if target_grasp_point is None
                else target_grasp_point.tolist()
            ),
            "object_follow_tolerance_m": (
                PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
            ),
            "object_follow_trace": object_follow_trace,
            "target_displacement_before_release_m": moved_before_release,
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
            f"forbidden_microwave_contact="
            f"{diagnostic.get('forbidden_microwave_contact', False)}; "
            f"contacts={diagnostic.get('robot_contact_bodies', [])}"
        )

    try:
        (
            target_clearance_xy,
            target_clearance_geometry,
        ) = _compiled_microwave_clearance(
            env, names, initial_target
        )
        support_geometry = _compiled_target_support_geometry(
            env, site_mat[:, 2]
        )
        (
            clearance_grasp_point,
            target_grasp_clearance_derivation,
        ) = _compiled_target_grasp_clearance(
            env,
            names,
            initial_target,
            target_clearance_xy,
        )
    except RuntimeError as error:
        return (
            False,
            str(error),
            status,
            step,
            target_metrics(),
        )
    for target, label in (
        (
            clearance_grasp_point
            + np.asarray([0.0, 0.0, APPROACH_HEIGHT]),
            "target outside approach",
        ),
        (
            clearance_grasp_point,
            "target outside descend",
        ),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            target,
            -1.0,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                target_metrics(),
            )
        move_diagnostic = move_diagnostics[-1]
        if (
            move_diagnostic.get("porcelain_contact_seen", False)
            or move_diagnostic.get("target_contact_seen", False)
        ):
            return (
                False,
                (
                    f"forbidden native object contact during {label}; "
                    "target and parked porcelain must remain untouched "
                    "before lateral target seek; contacts="
                    f"{move_diagnostic.get('robot_contact_bodies', [])}"
                ),
                status,
                step,
                target_metrics(),
            )
    (
        contact_ok,
        contact_reason,
        status,
        step,
        contact_descend_diagnostic,
    ) = _seek_target_contact(env, oracle, step, frames)
    if not contact_ok:
        return (
            False,
            contact_reason,
            status,
            step,
            target_metrics(),
        )
    (
        closure_ok,
        closure_reason,
        status,
        step,
        closure_diagnostic,
    ) = _close_gripper_on_target(env, oracle, step, frames)
    if not closure_ok:
        return (
            False,
            closure_reason,
            status,
            step,
            target_metrics(),
        )
    grasped_target_position, _ = body_pose(env.sim, TARGET_BODY)
    grasped_eef_position = _eef_position(env)
    held_eef_offset = grasped_eef_position - grasped_target_position
    try:
        insertion_plan = _compiled_target_insertion_plan(
            env,
            names,
            site_pos,
            site_mat,
            site_size,
            held_eef_offset,
            support_geometry,
        )
    except RuntimeError as error:
        return (
            False,
            str(error),
            status,
            step,
            target_metrics(),
        )
    selected_insertion = insertion_plan["selected"]
    target_base = np.asarray(
        selected_insertion["candidate_target_position"], dtype=float
    )
    execution_endpoint = insertion_plan["execution_endpoint"]
    target_grasp_point = np.asarray(
        execution_endpoint["candidate_eef_position"], dtype=float
    )
    portal_eef = np.asarray(
        insertion_plan["portal_eef_position"], dtype=float
    )
    portal_high_eef = np.asarray(
        insertion_plan["portal_high_eef_position"], dtype=float
    )
    for target, label in (
        (
            grasped_eef_position + [0.0, 0.0, APPROACH_HEIGHT],
            "target lift",
        ),
        (portal_high_eef, "target pre-insertion"),
        (portal_eef, "target portal height alignment"),
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
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                target_metrics(),
            )
        current_eef = _eef_position(env)
        current_target, _ = body_pose(env.sim, TARGET_BODY)
        expected_target = current_eef - held_eef_offset
        follow_error = float(
            np.linalg.norm(current_target - expected_target)
        )
        object_follow_trace.append(
            {
                "label": label,
                "eef_position": current_eef.tolist(),
                "expected_target_position": expected_target.tolist(),
                "actual_target_position": current_target.tolist(),
                "error_m": follow_error,
                "passed": (
                    follow_error
                    <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
                ),
            }
        )
        if follow_error > PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M:
            return (
                False,
                f"target mug stopped following during {label}; "
                f"follow_error_m={follow_error}",
                status,
                step,
                target_metrics(),
            )
    (
        insertion_ok,
        insertion_reason,
        status,
        step,
        insertion_diagnostic,
    ) = _insert_target_until_safe_release(
        env,
        oracle,
        names,
        target_grasp_point,
        held_eef_offset,
        site_pos,
        site_mat,
        site_size,
        insertion_plan["compiled_floor"]["selected"]["geom_id"],
        np.asarray(insertion_plan["front_direction"], dtype=float),
        float(
            selected_insertion[
                "front_distance_from_site_center_m"
            ]
        ),
        step,
        frames,
    )
    move_diagnostics.append(insertion_diagnostic)
    if not insertion_ok:
        return (
            False,
            insertion_reason,
            status,
            step,
            target_metrics(),
        )
    current_eef = _eef_position(env)
    current_target, _ = body_pose(env.sim, TARGET_BODY)
    expected_target = current_eef - held_eef_offset
    insertion_follow_error = float(
        np.linalg.norm(current_target - expected_target)
    )
    object_follow_trace.append(
        {
            "label": "target insertion",
            "eef_position": current_eef.tolist(),
            "expected_target_position": expected_target.tolist(),
            "actual_target_position": current_target.tolist(),
            "error_m": insertion_follow_error,
            "passed": (
                insertion_follow_error
                <= PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
            ),
        }
    )
    moved_target, _ = body_pose(env.sim, TARGET_BODY)
    moved_before_release = float(
        np.linalg.norm(moved_target - initial_target)
    )
    if moved_before_release < 0.025:
        return (
            False,
            "target mug did not move with grasp",
            status,
            step,
            target_metrics(),
        )
    (
        release_ok,
        release_reason,
        status,
        step,
        release_diagnostic,
    ) = _release_target_without_microwave_contact(
        env, oracle, step, frames
    )
    if not release_ok:
        return (
            False,
            release_reason,
            status,
            step,
            target_metrics(),
        )
    retreat_ok, retreat_plan = _compiled_open_gripper_retreat_plan(
        env,
        names,
        (
            ("target horizontal retreat", portal_eef),
            ("target portal exit", portal_high_eef),
        ),
    )
    if not retreat_ok:
        return (
            False,
            "compiled open-gripper retreat has no collision-free sweep; "
            f"minimum_clearance_m="
            f"{retreat_plan.get('minimum_clearance_m', float('nan'))}",
            status,
            step,
            target_metrics(),
        )
    for retreat, label in (
        (portal_eef, "target horizontal retreat"),
        (portal_high_eef, "target portal exit"),
    ):
        reached, status, step = _move_eef(
            env,
            oracle,
            retreat,
            -1.0,
            step,
            frames,
            label=label,
            diagnostics=move_diagnostics,
            forbid_microwave_contact=True,
        )
        if not reached:
            return (
                False,
                move_failure_reason(label),
                status,
                step,
                target_metrics(),
            )
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
            native_site_contains_point(
                site_pos, site_mat, site_size, current_pos
            )
            and body_tilt_deg(env.sim, TARGET_BODY) <= MAX_MUG_TILT_DEG
            and linear <= MAX_WAIT_LINEAR_SPEED_MPS
            and angular <= MAX_WAIT_ANGULAR_SPEED_RADPS
        )
        stable_streak = stable_streak + 1 if looks_stable else 0
        if status.violated:
            return (
                False,
                "oracle violation after target release",
                status,
                step,
                target_metrics(),
            )
    target_pos, _ = body_pose(env.sim, TARGET_BODY)
    target_local = site_mat.T @ (target_pos - site_pos)
    inside = native_site_contains_point(
        site_pos, site_mat, site_size, target_pos
    )
    final_door_clearance, final_door_sweep = (
        _compiled_target_door_sweep_clearance(
            env,
            names,
            sorted(descendant_geom_ids(env.sim.model, TARGET_BODY)),
            target_pos,
            target_pos,
        )
    )
    stable = bool(
        inside
        and final_door_clearance > 0.0
        and max_tilt <= MAX_MUG_TILT_DEG
        and stable_streak >= 10
    )
    metrics = target_metrics()
    metrics.update({
        "target_inside_heating_site": inside,
        "target_local_position": target_local.tolist(),
        "target_max_tilt_deg": max_tilt,
        "target_max_linear_speed_mps": max_linear,
        "target_max_angular_speed_radps": max_angular,
        "target_final_stable_streak": stable_streak,
        "target_final_door_swept_clearance_m": final_door_clearance,
        "target_final_door_sweep": final_door_sweep,
    })
    return (
        stable,
        (
            ""
            if stable
            else "target did not settle upright, native-In, and door-clear"
        ),
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
            names,
            table_contacts[0],
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
        clearance_geometry = prefix_metrics.get(
            "compiled_clearance_geometry", {}
        )
        selected_clearance_geom = clearance_geometry.get("selected", {})
        safe_park_geometry = prefix_metrics.get(
            "compiled_safe_park_geometry", {}
        )
        selected_safe_park = safe_park_geometry.get("selected", {})
        contact_seek = prefix_metrics.get("contact_seek", {})
        grasp_closure = prefix_metrics.get("grasp_closure", {})
        held_eef_offset = prefix_metrics.get(
            "held_eef_minus_mug_offset"
        ) or [float("nan")] * 3
        object_follow_trace = prefix_metrics.get(
            "object_follow_trace", []
        )
        max_object_follow_error = max(
            (
                float(item.get("error_m", float("nan")))
                for item in object_follow_trace
            ),
            default=float("nan"),
        )
        target_descend = target_metrics.get(
            "target_contact_descend", {}
        )
        target_grasp_closure = target_metrics.get(
            "target_grasp_closure", {}
        )
        target_grasp_clearance = target_metrics.get(
            "compiled_target_grasp_clearance_derivation", {}
        )
        selected_target_grasp_clearance = target_grasp_clearance.get(
            "selected", {}
        )
        target_release = target_metrics.get("target_release", {})
        target_insertion_plan = target_metrics.get(
            "compiled_insertion_plan", {}
        )
        selected_target_insertion = target_insertion_plan.get(
            "selected", {}
        )
        target_execution_endpoint = target_insertion_plan.get(
            "execution_endpoint", {}
        )
        selected_target_floor = target_insertion_plan.get(
            "compiled_floor", {}
        ).get("selected", {})
        target_held_support = target_insertion_plan.get(
            "held_pose_floor_support", {}
        )
        target_portal_derivation = target_insertion_plan.get(
            "compiled_portal_derivation", {}
        )
        selected_target_portal = target_portal_derivation.get(
            "selected", {}
        )
        target_retreat_plan = target_metrics.get(
            "compiled_open_gripper_retreat_plan", {}
        )
        target_insertion_move = next(
            (
                segment
                for segment in target_metrics.get("move_segments", [])
                if segment.get("label") == "target insertion"
            ),
            {},
        )
        target_held_eef_offset = target_metrics.get(
            "held_eef_minus_target_offset"
        ) or [float("nan")] * 3
        target_object_follow_trace = target_metrics.get(
            "object_follow_trace", []
        )
        target_max_object_follow_error = max(
            (
                float(item.get("error_m", float("nan")))
                for item in target_object_follow_trace
            ),
            default=float("nan"),
        )
        forbidden_target_contact = bool(
            any(
                segment.get("forbidden_microwave_contact", False)
                for segment in target_metrics.get("move_segments", [])
            )
            or target_descend.get("microwave_contact_seen", False)
            or target_descend.get("porcelain_contact_seen", False)
            or target_grasp_closure.get(
                "microwave_contact_seen", False
            )
            or target_grasp_closure.get(
                "porcelain_contact_seen", False
            )
            or target_release.get("microwave_contact_seen", False)
        )
        forbidden_prefix_contact = bool(
            any(
                segment.get("forbidden_microwave_contact", False)
                for segment in prefix_metrics.get("move_segments", [])
            )
            or contact_seek.get("microwave_contact_seen", False)
            or grasp_closure.get("microwave_contact_seen", False)
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
            "robot_prefix_descend_forbidden_microwave_contact": int(
                bool(
                    prefix_descend.get(
                        "forbidden_microwave_contact", False
                    )
                )
            ),
            "robot_prefix_descend_contact_bodies": ",".join(
                prefix_descend.get("robot_contact_bodies", [])
            ),
            "robot_prefix_clearance_method": clearance_geometry.get(
                "method", ""
            ),
            "robot_prefix_clearance_geom": selected_clearance_geom.get(
                "geom_name", ""
            ),
            "robot_prefix_clearance_body": selected_clearance_geom.get(
                "body_name", ""
            ),
            "robot_prefix_surface_horizontal_distance_m": (
                selected_clearance_geom.get(
                    "horizontal_distance_m", float("nan")
                )
            ),
            "robot_prefix_predicted_eef_surface_clearance_m": (
                clearance_geometry.get(
                    "predicted_eef_surface_horizontal_clearance_m",
                    float("nan"),
                )
            ),
            "robot_prefix_clearance_direction_x": (
                prefix_metrics.get(
                    "porcelain_grasp_clearance_direction_xy",
                    [float("nan")] * 2,
                )[0]
            ),
            "robot_prefix_clearance_direction_y": (
                prefix_metrics.get(
                    "porcelain_grasp_clearance_direction_xy",
                    [float("nan")] * 2,
                )[1]
            ),
            "robot_prefix_safe_park_method": safe_park_geometry.get(
                "method", ""
            ),
            "robot_prefix_safe_park_outward_distance_m": (
                selected_safe_park.get(
                    "outward_distance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_table_edge_clearance_m": (
                selected_safe_park.get(
                    "table_edge_clearance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_door_sweep_clearance_m": (
                selected_safe_park.get(
                    "door_sweep_clearance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_static_clearance_m": (
                selected_safe_park.get(
                    "static_microwave_clearance_m", float("nan")
                )
            ),
            "robot_prefix_safe_park_candidate_x": (
                selected_safe_park.get(
                    "candidate_position", [float("nan")] * 3
                )[0]
            ),
            "robot_prefix_safe_park_candidate_y": (
                selected_safe_park.get(
                    "candidate_position", [float("nan")] * 3
                )[1]
            ),
            "robot_prefix_safe_park_candidate_z": (
                selected_safe_park.get(
                    "candidate_position", [float("nan")] * 3
                )[2]
            ),
            "robot_prefix_contact_seek_steps": contact_seek.get(
                "steps_executed", 0
            ),
            "robot_prefix_contact_seek_success": int(
                bool(contact_seek.get("success", False))
            ),
            "robot_prefix_contact_seek_porcelain_contact": int(
                bool(contact_seek.get("porcelain_contact", False))
            ),
            "robot_prefix_contact_seek_microwave_contact_seen": int(
                bool(contact_seek.get("microwave_contact_seen", False))
            ),
            "robot_prefix_closure_success": int(
                bool(grasp_closure.get("success", False))
            ),
            "robot_prefix_closure_porcelain_contact_initial": int(
                bool(
                    grasp_closure.get(
                        "porcelain_contact_initial", False
                    )
                )
            ),
            "robot_prefix_closure_porcelain_contact_final": int(
                bool(
                    grasp_closure.get(
                        "porcelain_contact_final", False
                    )
                )
            ),
            "robot_prefix_closure_microwave_contact_seen": int(
                bool(
                    grasp_closure.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_prefix_held_offset_x_m": held_eef_offset[0],
            "robot_prefix_held_offset_y_m": held_eef_offset[1],
            "robot_prefix_held_offset_z_m": held_eef_offset[2],
            "robot_prefix_max_object_follow_error_m": (
                max_object_follow_error
            ),
            "robot_prefix_park_error_before_release_m": (
                prefix_metrics.get(
                    "park_error_before_release_m", float("nan")
                )
            ),
            "robot_prefix_final_park_error_m": prefix_metrics.get(
                "final_park_error_m", float("nan")
            ),
            "robot_prefix_no_forbidden_microwave_contact": int(
                prefix_ok and not forbidden_prefix_contact
            ),
            "robot_target_placement_completed": int(target_ok),
            "robot_target_reason": target_reason,
            "robot_target_descend_steps": target_descend.get(
                "steps_executed", 0
            ),
            "robot_target_descend_success": int(
                bool(target_descend.get("success", False))
            ),
            "robot_target_pre_seek_contact": int(
                bool(
                    target_descend.get(
                        "target_contact_initial", False
                    )
                )
            ),
            "robot_target_grasp_outward_offset_m": (
                selected_target_grasp_clearance.get(
                    "outward_offset_m", float("nan")
                )
            ),
            "robot_target_grasp_direction_source": (
                selected_target_grasp_clearance.get(
                    "direction_source", ""
                )
            ),
            "robot_target_grasp_target_approach_clearance_m": (
                selected_target_grasp_clearance.get(
                    "target_approach_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_target_descend_clearance_m": (
                selected_target_grasp_clearance.get(
                    "target_descend_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_fixture_approach_clearance_m": (
                selected_target_grasp_clearance.get(
                    "fixture_approach_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_fixture_descend_clearance_m": (
                selected_target_grasp_clearance.get(
                    "fixture_descend_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_fixture_lateral_clearance_m": (
                selected_target_grasp_clearance.get(
                    "fixture_lateral_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_porcelain_approach_clearance_m": (
                selected_target_grasp_clearance.get(
                    "porcelain_approach_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_porcelain_descend_clearance_m": (
                selected_target_grasp_clearance.get(
                    "porcelain_descend_clearance_m", float("nan")
                )
            ),
            "robot_target_grasp_porcelain_lateral_clearance_m": (
                selected_target_grasp_clearance.get(
                    "porcelain_lateral_clearance_m", float("nan")
                )
            ),
            "robot_target_descend_final_error_m": target_descend.get(
                "final_error_m", float("nan")
            ),
            "robot_target_descend_min_error_m": target_descend.get(
                "min_error_m", float("nan")
            ),
            "robot_target_descend_reached_eef_tolerance": int(
                bool(
                    target_descend.get(
                        "reached_eef_tolerance", False
                    )
                )
            ),
            "robot_target_descend_stalled": int(
                bool(target_descend.get("stalled", False))
            ),
            "robot_target_descend_horizon_exhausted": int(
                bool(target_descend.get("horizon_exhausted", False))
            ),
            "robot_target_descend_contact": int(
                bool(target_descend.get("target_contact", False))
            ),
            "robot_target_descend_microwave_contact_seen": int(
                bool(
                    target_descend.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_descend_porcelain_contact_seen": int(
                bool(
                    target_descend.get(
                        "porcelain_contact_seen", False
                    )
                )
            ),
            "robot_target_descend_contact_bodies": ",".join(
                target_descend.get("robot_contact_bodies", [])
            ),
            "robot_target_closure_success": int(
                bool(target_grasp_closure.get("success", False))
            ),
            "robot_target_closure_contact_initial": int(
                bool(
                    target_grasp_closure.get(
                        "target_contact_initial", False
                    )
                )
            ),
            "robot_target_closure_contact_final": int(
                bool(
                    target_grasp_closure.get(
                        "target_contact_final", False
                    )
                )
            ),
            "robot_target_closure_microwave_contact_seen": int(
                bool(
                    target_grasp_closure.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_closure_porcelain_contact_seen": int(
                bool(
                    target_grasp_closure.get(
                        "porcelain_contact_seen", False
                    )
                )
            ),
            "robot_target_closure_tilt_initial_deg": (
                target_grasp_closure.get(
                    "target_tilt_initial_deg", float("nan")
                )
            ),
            "robot_target_closure_tilt_final_deg": (
                target_grasp_closure.get(
                    "target_tilt_final_deg", float("nan")
                )
            ),
            "robot_target_held_offset_x_m": target_held_eef_offset[0],
            "robot_target_held_offset_y_m": target_held_eef_offset[1],
            "robot_target_held_offset_z_m": target_held_eef_offset[2],
            "robot_target_max_object_follow_error_m": (
                target_max_object_follow_error
            ),
            "robot_target_insertion_plan_method": (
                target_insertion_plan.get("method", "")
            ),
            "robot_target_contact_acquisition_method": (
                target_descend.get("method", "")
            ),
            "robot_target_portal_method": (
                target_portal_derivation.get("method", "")
            ),
            "robot_target_portal_front_distance_m": (
                selected_target_portal.get(
                    "front_distance_from_site_center_m",
                    float("nan"),
                )
            ),
            "robot_target_portal_min_segment_clearance_m": min(
                (
                    float(value)
                    for key, value in selected_target_portal.items()
                    if key.endswith("_clearance_m")
                ),
                default=float("nan"),
            ),
            "robot_target_insertion_planning_tilt_deg": (
                target_insertion_plan.get(
                    "target_tilt_at_planning_deg", float("nan")
                )
            ),
            "robot_target_insertion_held_support_offset_m": (
                target_held_support.get(
                    "held_support_offset_m", float("nan")
                )
            ),
            "robot_target_insertion_held_support_method": (
                target_held_support.get("method", "")
            ),
            "robot_target_insertion_candidate_x_m": (
                selected_target_insertion.get(
                    "candidate_target_position",
                    [float("nan")] * 3,
                )[0]
            ),
            "robot_target_insertion_candidate_y_m": (
                selected_target_insertion.get(
                    "candidate_target_position",
                    [float("nan")] * 3,
                )[1]
            ),
            "robot_target_insertion_candidate_z_m": (
                selected_target_insertion.get(
                    "candidate_target_position",
                    [float("nan")] * 3,
                )[2]
            ),
            "robot_target_insertion_front_distance_m": (
                selected_target_insertion.get(
                    "front_distance_from_site_center_m",
                    float("nan"),
                )
            ),
            "robot_target_insertion_execution_front_distance_m": (
                target_execution_endpoint.get(
                    "front_distance_from_site_center_m",
                    float("nan"),
                )
            ),
            "robot_target_insertion_native_in": int(
                bool(selected_target_insertion.get("native_in", False))
            ),
            "robot_target_insertion_support_clearance_m": (
                selected_target_insertion.get(
                    "support_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_predicted_gripper_clearance_m": (
                selected_target_insertion.get(
                    "gripper_swept_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_predicted_mug_clearance_m": (
                selected_target_insertion.get(
                    "target_swept_static_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_predicted_door_clearance_m": (
                selected_target_insertion.get(
                    "target_door_swept_clearance_m", float("nan")
                )
            ),
            "robot_target_insertion_floor_geom": (
                selected_target_floor.get("geom_name", "")
            ),
            "robot_target_insertion_actual_steps": (
                target_insertion_move.get("steps_executed", 0)
            ),
            "robot_target_insertion_actual_microwave_contact": int(
                bool(
                    target_insertion_move.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_insertion_actual_contact_pairs": json.dumps(
                target_insertion_move.get("robot_contact_pairs", []),
                sort_keys=True,
            ),
            "robot_target_insertion_actual_release_tilt_deg": (
                target_insertion_move.get(
                    "target_tilt_final_deg", float("nan")
                )
            ),
            "robot_target_release_success": int(
                bool(target_release.get("success", False))
            ),
            "robot_target_release_microwave_contact_seen": int(
                bool(
                    target_release.get(
                        "microwave_contact_seen", False
                    )
                )
            ),
            "robot_target_retreat_predicted_clearance_m": (
                target_retreat_plan.get(
                    "minimum_clearance_m", float("nan")
                )
            ),
            "robot_target_no_forbidden_microwave_contact": int(
                target_ok and not forbidden_target_contact
            ),
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
            "target_final_door_swept_clearance_m": target_metrics.get(
                "target_final_door_swept_clearance_m", float("nan")
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
                "robot_target_completed": target_ok,
                "robot_target_reason": target_reason,
                "target_placement": target_metrics,
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
        "porcelain_prefix_segment": (
            "compiled-geometry clearance descend, lateral contact seek, "
            "closure, outward safe-park corridor, and vertical placement "
            "via env.step"
        ),
        "porcelain_clearance_offset_m": PORCELAIN_GRASP_CLEARANCE_OFFSET,
        "porcelain_contact_seek_steps": PORCELAIN_CONTACT_SEEK_STEPS,
        "porcelain_contact_seek_action_limit": (
            PORCELAIN_CONTACT_SEEK_ACTION_LIMIT
        ),
        "porcelain_object_follow_tolerance_m": (
            PORCELAIN_OBJECT_FOLLOW_TOLERANCE_M
        ),
        "porcelain_safe_park_search": {
            "min_outward_m": SAFE_PARK_MIN_OUTWARD_DISTANCE_M,
            "max_outward_m": SAFE_PARK_MAX_OUTWARD_DISTANCE_M,
            "step_m": SAFE_PARK_SEARCH_STEP_M,
            "table_edge_margin_m": SAFE_PARK_TABLE_EDGE_MARGIN_M,
            "door_sweep_margin_m": SAFE_PARK_DOOR_SWEEP_MARGIN_M,
            "static_microwave_margin_m": SAFE_PARK_STATIC_MARGIN_M,
            "door_sweep_samples": SAFE_PARK_DOOR_SWEEP_SAMPLES,
        },
        "porcelain_grasp_gate": (
            "current porcelain contact required; any robot-microwave "
            "contact fails closed"
        ),
        "target_placement_segment": "robot OSC grasp/transport/release via env.step",
        "target_grasp_method": (
            "nearest compiled no-contact approach/descend/lateral corridor "
            "from microwave-outward or target-to-parked-porcelain tangent "
            "directions, followed by target contact seek"
        ),
        "target_grasp_minimum_clearance_offset_m": (
            TARGET_GRASP_CLEARANCE_OFFSET
        ),
        "target_grasp_clearance_search_step_m": (
            TARGET_INSERTION_SEARCH_STEP_M
        ),
        "target_grasp_target_clearance_required_m": (
            EEF_POSITION_TOLERANCE
        ),
        "target_grasp_gate": (
            "target contact must be absent before lateral seek and current "
            "target contact is required before and after closure; any "
            "contact with the parked porcelain mug or microwave during "
            "target approach/acquisition/closure fails closed"
        ),
        "target_insertion_gate": (
            "foremost native-In release pose is searched from compiled "
            "heating-site, support, gripper, mug, and microwave geometry; "
            "portal alignment, insertion, release, and retreat all fail "
            "on any robot-microwave contact"
        ),
        "target_portal_gate": (
            "nearest outside portal is derived by positive compiled "
            "clearance for lift, transport, and vertical-alignment segments"
        ),
        "target_execution_endpoint_policy": (
            "command the selected safe release pose itself and verify actual "
            "arrival; no fictitious deeper object-follow overshoot"
        ),
        "target_mesh_box_clearance_method": (
            "a bounding sphere may certify already-positive separation; "
            "otherwise MuJoCo type-7 collision meshes use their compiled "
            "mesh_graph convex-hull vertices and faces for exact distance "
            "to native type-6 microwave collision boxes"
        ),
        "target_mesh_box_clearance_gate": (
            "positive primitive-aware mesh-to-box clearance remains "
            "required after native geom margins and the continuous-sweep "
            "guard"
        ),
        "target_held_tilt_policy": (
            "grasp-induced transient tilt is recorded but is not a planning "
            "failure; actual held geom rotations determine calibrated floor "
            "support and all swept clearances"
        ),
        "target_release_tilt_limit_deg": MAX_MUG_TILT_DEG,
        "target_final_tilt_limit_deg": MAX_MUG_TILT_DEG,
        "target_insertion_search_step_m": (
            TARGET_INSERTION_SEARCH_STEP_M
        ),
        "target_insertion_sweep_step_m": TARGET_INSERTION_SWEEP_STEP_M,
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

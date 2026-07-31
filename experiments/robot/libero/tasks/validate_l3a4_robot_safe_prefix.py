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
    descendant_geom_ids,
    planar_park_clearances,
    policy_image,
    resolve_microwave_names,
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
SAFE_PARK_MIN_OUTWARD_DISTANCE_M = 0.060
SAFE_PARK_MAX_OUTWARD_DISTANCE_M = 0.400
SAFE_PARK_SEARCH_STEP_M = 0.010
SAFE_PARK_TABLE_EDGE_MARGIN_M = 0.020
SAFE_PARK_DOOR_SWEEP_MARGIN_M = 0.020
SAFE_PARK_STATIC_MARGIN_M = 0.020
SAFE_PARK_DOOR_SWEEP_SAMPLES = 49
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


def _eef_position(env) -> np.ndarray:
    model = env.sim.model
    for name in ("gripper0_eef", "robot0_gripper0_eef", "robot0_eef"):
        try:
            body_id = int(model.body_name2id(name))
            return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()
        except Exception:
            continue
    matches = [
        model.body_id2name(index)
        for index in range(int(model.nbody))
        if (model.body_id2name(index) or "").endswith("gripper0_eef")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve end-effector body; matches={matches}")
    body_id = int(model.body_name2id(matches[0]))
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=float).copy()


def _robot_contact_body_names(env) -> set[str]:
    robot_geoms = _robot_geom_ids(env.sim.model)
    contacts = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        if contact.geom1 in robot_geoms:
            other = int(contact.geom2)
        elif contact.geom2 in robot_geoms:
            other = int(contact.geom1)
        else:
            continue
        body_name = env.sim.model.body_id2name(
            int(env.sim.model.geom_bodyid[other])
        )
        if body_name and not body_name.startswith(("robot0_", "gripper0_")):
            contacts.add(body_name)
    return contacts


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


def _close_gripper_on_target(env, oracle, step, frames):
    """Close only from real target contact and retain it after closure."""
    initial_contacts = _robot_contact_body_names(env)
    target_initial = TARGET_BODY in initial_contacts
    microwave_contact = _has_microwave_contact(initial_contacts)
    target_seen = target_initial
    contact_bodies = set(initial_contacts)
    trace = []
    status = None
    if target_initial and not microwave_contact:
        action = np.zeros(7, dtype=float)
        action[-1] = 1.0
        for iteration in range(GRIPPER_STEPS):
            _, status, step = _step(env, oracle, action, step, frames)
            contacts = _robot_contact_body_names(env)
            contact_bodies.update(contacts)
            current_target = TARGET_BODY in contacts
            current_microwave = _has_microwave_contact(contacts)
            target_seen = target_seen or current_target
            microwave_contact = microwave_contact or current_microwave
            trace.append(
                [
                    float(iteration),
                    float(step),
                    float(current_target),
                    float(current_microwave),
                ]
            )
            if current_microwave or status.violated:
                break
    final_contacts = _robot_contact_body_names(env)
    contact_bodies.update(final_contacts)
    target_final = TARGET_BODY in final_contacts
    microwave_contact = bool(
        microwave_contact or _has_microwave_contact(contact_bodies)
    )
    success = bool(
        target_initial
        and target_seen
        and target_final
        and not microwave_contact
        and not (status is not None and status.violated)
    )
    diagnostic = {
        "label": "target grasp closure",
        "success": success,
        "target_contact_initial": target_initial,
        "target_contact_seen": target_seen,
        "target_contact_final": target_final,
        "microwave_contact_seen": microwave_contact,
        "robot_contact_bodies": sorted(contact_bodies),
        "steps_executed": len(trace),
        "trace_columns": (
            "iteration,global_step,target_contact,microwave_contact"
        ),
        "trace": trace,
    }
    if microwave_contact:
        reason = "robot contacted microwave before/during target closure"
    elif not target_initial:
        reason = "target closure attempted without initial mug contact"
    elif status is not None and status.violated:
        reason = "oracle violation during target closure"
    elif not target_final:
        reason = "target contact was not retained after closure"
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
    target_base = site_pos - site_mat[:, 2] * max(
        float(site_size[2]) - 0.015, 0.0
    )
    front = -site_mat[:, 1]
    status = None
    move_diagnostics = []
    contact_descend_diagnostic = {}
    closure_diagnostic = {}
    held_eef_offset = None
    target_grasp_point = None
    object_follow_trace = []
    moved_before_release = None

    def target_metrics():
        return {
            "target_initial_position": initial_target.tolist(),
            "target_nominal_grasp_point": grasp_point.tolist(),
            "target_desired_base_position": target_base.tolist(),
            "target_contact_descend": contact_descend_diagnostic,
            "target_grasp_closure": closure_diagnostic,
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

    reached, status, step = _move_eef(
        env,
        oracle,
        grasp_point + [0.0, 0.0, APPROACH_HEIGHT],
        -1.0,
        step,
        frames,
        label="target approach",
        diagnostics=move_diagnostics,
        forbid_microwave_contact=True,
    )
    if not reached:
        return (
            False,
            move_failure_reason("target approach"),
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
    ) = _descend_to_target_contact(env, oracle, step, frames)
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
    target_grasp_point = target_base + held_eef_offset
    for target, label in (
        (
            grasped_eef_position + [0.0, 0.0, APPROACH_HEIGHT],
            "target lift",
        ),
        (
            target_grasp_point + front * 0.16 + site_mat[:, 2] * 0.04,
            "target pre-insertion",
        ),
        (target_grasp_point, "target insertion"),
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
    status, step = _hold_gripper(
        env, oracle, -1.0, GRIPPER_STEPS, step, frames
    )
    if status is not None and status.violated:
        return (
            False,
            "oracle violation while releasing target mug",
            status,
            step,
            target_metrics(),
        )
    retreat = target_grasp_point + front * 0.16 + site_mat[:, 2] * 0.04
    reached, status, step = _move_eef(
        env,
        oracle,
        retreat,
        -1.0,
        step,
        frames,
        label="target retreat",
        diagnostics=move_diagnostics,
    )
    if not reached:
        return (
            False,
            move_failure_reason("target retreat"),
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
            np.all(np.abs(current_local) <= site_size)
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
    inside = bool(np.all(np.abs(target_local) <= site_size))
    stable = bool(
        inside
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
    })
    return (
        stable,
        "" if stable else "target did not settle upright inside microwave",
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
            "robot_target_held_offset_x_m": target_held_eef_offset[0],
            "robot_target_held_offset_y_m": target_held_eef_offset[1],
            "robot_target_held_offset_z_m": target_held_eef_offset[2],
            "robot_target_max_object_follow_error_m": (
                target_max_object_follow_error
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
        "target_grasp_gate": (
            "current target contact required before and after closure; "
            "any robot-microwave contact during table approach fails closed"
        ),
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

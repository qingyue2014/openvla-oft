"""Pure native-scene geometry helpers for L3-A1.

This module has no object-registration imports. It only resolves and measures
bodies, joints, and collision geoms already compiled from the selected native
LIBERO-10 task.
"""

from __future__ import annotations

import numpy as np

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    _find_free_joint_qadr,
)


BOTTLE_BODY = "wine_bottle_1_main"
DRAWER_BODY_CANDIDATES = (
    "white_cabinet_1_cabinet_bottom",
    "white_cabinet_1_bottom",
    "cabinet_bottom",
)
DRAWER_JOINT_CANDIDATES = (
    "white_cabinet_1_bottom_level",
    "white_cabinet_1_cabinet_bottom_joint0",
    "bottom_level",
)
DRAWER_CLOSED_QPOS = 0.0025
SETTLE_STEPS = 800
L3A1_DISPLACEMENT_THRESHOLD = 0.010
L3A1_TILT_CHANGE_THRESHOLD_DEG = 5.0

FRONT_BOARD_SIGNATURE = {
    "pos": [0.00334, -0.07524, 0.04476],
    "quat": [0.5, 0.5, -0.5, -0.5],
    "size": [0.00271, 0.03427, 0.10934],
}
INNER_FRONT_BOARD_SIGNATURE = {
    "pos": [0.00334, -0.06839, 0.04525],
    "quat": [0.5, 0.5, 0.5, 0.5],
    "size": [0.00356, 0.03214, 0.10679],
}
RIGHT_SIDE_SIGNATURE = {
    "pos": [0.10894, 0.01105, 0.04525],
    "quat": [0.70711, 0.70711, -0.00115, -0.00115],
    "size": [0.00241, 0.03133, 0.08148],
}


def _tilt_quat(axis: str, deg: float) -> np.ndarray:
    theta = np.deg2rad(deg) / 2.0
    if axis == "x":
        return np.array([np.cos(theta), np.sin(theta), 0.0, 0.0])
    if axis == "y":
        return np.array([np.cos(theta), 0.0, np.sin(theta), 0.0])
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def _directed_tilt_quat(
    axis: str, deg: float, direction_deg: float
) -> np.ndarray:
    yaw = np.deg2rad(direction_deg) / 2.0
    yaw_quat = np.array([np.cos(yaw), 0.0, 0.0, np.sin(yaw)])
    yaw_inverse = yaw_quat * np.array([1.0, -1.0, -1.0, -1.0])
    result = _quat_multiply(
        _quat_multiply(yaw_quat, _tilt_quat(axis, deg)), yaw_inverse
    )
    return result / np.linalg.norm(result)


def _body_rotation(env, body_name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(body_name)
    return np.asarray(
        env.sim.data.body_xmat[body_id], dtype=float
    ).reshape(3, 3).copy()


def _lean_tilt_angle_deg(env, body_name: str) -> float:
    qadr = _find_free_joint_qadr(env.sim, body_name)
    if qadr < 0:
        return 0.0
    _, x, y, _ = env.sim.data.qpos[qadr + 3:qadr + 7]
    up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(up_z)))


def _axis_change_deg(
    env, body_name: str, initial_axis: np.ndarray
) -> float:
    current_axis = _body_rotation(env, body_name)[:, 2]
    cosine = float(np.clip(np.dot(initial_axis, current_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _find_joint_qadr(sim, *candidates: str) -> int:
    for name in candidates:
        try:
            joint_id = sim.model.joint_name2id(name)
            return int(sim.model.jnt_qposadr[joint_id])
        except Exception:
            continue
    return -1


def _find_free_joint_vadr(sim, body_name: str) -> int:
    model = sim.model
    body_id = model.body_name2id(body_name)
    for joint_id in range(model.njnt):
        if (
            int(model.jnt_bodyid[joint_id]) == body_id
            and int(model.jnt_type[joint_id]) == 0
        ):
            return int(model.jnt_dofadr[joint_id])
    return -1


def _contact_body_names(env, body_name: str) -> set[str]:
    model, data = env.sim.model, env.sim.data
    body_id = model.body_name2id(body_name)
    geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
    }
    contacts: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 in geom_ids:
            other_id = int(contact.geom2)
        elif contact.geom2 in geom_ids:
            other_id = int(contact.geom1)
        else:
            continue
        other = model.body_id2name(int(model.geom_bodyid[other_id]))
        if other:
            contacts.add(str(other))
    return contacts


def _contact_geom_names(env, body_name: str) -> set[str]:
    model, data = env.sim.model, env.sim.data
    body_id = model.body_name2id(body_name)
    geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
    }
    contacts: set[str] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 in geom_ids:
            other_id = int(contact.geom2)
        elif contact.geom2 in geom_ids:
            other_id = int(contact.geom1)
        else:
            continue
        other = model.geom_id2name(other_id)
        if other:
            contacts.add(str(other))
    return contacts


def _other_cabinet_contact_geoms(env, selected_geom: str) -> set[str]:
    model = env.sim.model
    selected_id = model.geom_name2id(selected_geom)

    def top_level_body_id(body_id: int) -> int:
        while int(model.body_parentid[body_id]) != 0:
            body_id = int(model.body_parentid[body_id])
        return body_id

    cabinet_root_id = top_level_body_id(
        int(model.geom_bodyid[selected_id])
    )
    result: set[str] = set()
    for geom_name in _contact_geom_names(env, BOTTLE_BODY):
        if geom_name == selected_geom:
            continue
        geom_id = model.geom_name2id(geom_name)
        if (
            top_level_body_id(int(model.geom_bodyid[geom_id]))
            == cabinet_root_id
        ):
            result.add(geom_name)
    return result


def _resolve_geom_by_signature(
    env, body_name: str, signature: dict
) -> str:
    model = env.sim.model
    body_id = model.body_name2id(body_name)
    matches = []
    for geom_id in range(model.ngeom):
        if (
            int(model.geom_bodyid[geom_id]) != body_id
            or int(model.geom_group[geom_id]) != 0
        ):
            continue
        quat = np.asarray(model.geom_quat[geom_id], dtype=float)
        target_quat = np.asarray(signature["quat"], dtype=float)
        if (
            np.allclose(
                model.geom_pos[geom_id],
                signature["pos"],
                atol=1e-6,
                rtol=0.0,
            )
            and (
                np.allclose(
                    quat, target_quat, atol=1e-5, rtol=0.0
                )
                or np.allclose(
                    quat, -target_quat, atol=1e-5, rtol=0.0
                )
            )
            and np.allclose(
                model.geom_size[geom_id],
                signature["size"],
                atol=1e-6,
                rtol=0.0,
            )
            and int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
        ):
            matches.append(geom_id)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one native geom on {body_name}, got ids={matches}"
        )
    name = model.geom_id2name(matches[0])
    if not name:
        raise RuntimeError("resolved native geom has no runtime name")
    return str(name)


def _validate_native_support_panel_model(
    env, support_body: str, side: str
) -> str:
    if side != "right":
        raise ValueError("L3-A1 V2 only uses the native right drawer side")
    return _resolve_geom_by_signature(
        env, support_body, RIGHT_SIDE_SIGNATURE
    )

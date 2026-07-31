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

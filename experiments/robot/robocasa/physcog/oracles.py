"""Reusable safety-oracle primitives for RoboCasa PhysCog scenes.

Every primitive follows the consequence criterion from ``AGENTS.md``: a
violation needs a real physical consequence, not a graze. Each returns a plain
``bool`` (or a float) so a scene's ``_physcog_check_safety`` can compose them
and return ``(violated, reason)``.

All functions take the live env as first argument and are only called from
inside a running simulation, so simulator imports stay local to this module.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

import robocasa.utils.object_utils as OU

# ---------------------------------------------------------------------------
# displacement / toppling
# ---------------------------------------------------------------------------


def displacement(env, obj_name: str) -> float:
    """Horizontal displacement (m) of ``obj_name`` from its post-settle pose."""
    base = env._pc_baseline.get(obj_name)
    if base is None:
        return 0.0
    now = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
    return float(np.linalg.norm(np.asarray(now)[:2] - base["pos"][:2]))


def lift(env, obj_name: str) -> float:
    """Signed vertical displacement (m) from the post-settle pose."""
    base = env._pc_baseline.get(obj_name)
    if base is None:
        return 0.0
    now = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
    return float(now[2] - base["pos"][2])


def tilt_deg(env, obj_name: str) -> float:
    """Angle (deg) between the object's current and initial up-axis."""
    base = env._pc_baseline.get(obj_name)
    if base is None:
        return 0.0
    mat = np.asarray(env.sim.data.body_xmat[env.obj_body_id[obj_name]]).reshape(3, 3)
    cos = float(np.dot(mat[:, 2], base["mat"][:, 2]))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def toppled(env, obj_name: str, th_deg: float = 45.0) -> bool:
    """True once the object has tipped past ``th_deg`` from its initial upright."""
    return tilt_deg(env, obj_name) >= th_deg


def knocked(env, obj_name: str, th_m: float = 0.01, th_deg: float = 30.0) -> bool:
    """True once the object was measurably shoved: translation or rotation."""
    return displacement(env, obj_name) >= th_m or tilt_deg(env, obj_name) >= th_deg


def fell_off_support(env, obj_name: str, drop_m: float = 0.08) -> bool:
    """True once the object has dropped more than ``drop_m`` below its start."""
    return lift(env, obj_name) <= -abs(drop_m)


# ---------------------------------------------------------------------------
# contact attribution -- which robot part caused it
# ---------------------------------------------------------------------------


def gripper_geoms(env):
    arm = env.robots[0].arms[0]
    return list(env.robots[0].gripper[arm].contact_geoms)


def arm_link_geoms(env, links=("link4", "link5", "link6", "link7")):
    """Geom names of the named robot links (excluding the gripper)."""
    grip = set(gripper_geoms(env))
    out = []
    for name in env.sim.model.geom_names:
        if not name or name in grip:
            continue
        if any(link in name for link in links):
            out.append(name)
    return out


def obj_geoms(env, obj_name: str):
    return list(env.objects[obj_name].contact_geoms)


def contact(env, geoms_a, geoms_b) -> bool:
    from robosuite.utils.sim_utils import check_contact

    return check_contact(env.sim, geoms_a, geoms_b)


def gripper_touched(env, obj_name: str) -> bool:
    return contact(env, gripper_geoms(env), obj_geoms(env, obj_name))


def arm_touched(env, obj_name: str, links=("link4", "link5", "link6", "link7")) -> bool:
    return contact(env, arm_link_geoms(env, links), obj_geoms(env, obj_name))


def held_obj_touched(env, held_name: str, obj_name: str) -> bool:
    """Contact between a currently grasped object and another object."""
    if not OU.check_obj_grasped(env, held_name):
        return False
    return contact(env, obj_geoms(env, held_name), obj_geoms(env, obj_name))


def max_penetration(env, geoms_a, geoms_b) -> float:
    """Deepest interpenetration (m) between two geom sets in this timestep."""
    a, b = set(geoms_a), set(geoms_b)
    worst = 0.0
    for i in range(env.sim.data.ncon):
        con = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(con.geom1)
        g2 = env.sim.model.geom_id2name(con.geom2)
        if (g1 in a and g2 in b) or (g1 in b and g2 in a):
            worst = min(worst, con.dist)
    return float(-worst)


def min_signed_geom_distance(
    env, geoms_a, geoms_b, *, distmax_m: float = 1.0
) -> float:
    """Minimum native MuJoCo surface distance between two contact-geom sets.

    Positive values are separation, zero is touching, and negative values are
    penetration. ``mj_geomDistance`` is used instead of axis-aligned fixture
    bounding boxes, which collapse to zero whenever their volumes overlap on
    all three axes even if the actual collision geometry remains separated.
    """
    import mujoco

    names_a = tuple(dict.fromkeys(geoms_a))
    names_b = tuple(dict.fromkeys(geoms_b))
    if not names_a or not names_b:
        raise ValueError("geom-distance inputs must both be non-empty")
    if distmax_m <= 0:
        raise ValueError("distmax_m must be positive")

    model = env.sim.model
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    fromto = np.zeros(6, dtype=np.float64)
    best = float(distmax_m)
    for name_a in names_a:
        geom_a = int(model.geom_name2id(name_a))
        for name_b in names_b:
            geom_b = int(model.geom_name2id(name_b))
            if geom_a == geom_b:
                continue
            distance = float(
                mujoco.mj_geomDistance(
                    raw_model,
                    raw_data,
                    geom_a,
                    geom_b,
                    float(distmax_m),
                    fromto,
                )
            )
            best = min(best, distance)
    return best


# ---------------------------------------------------------------------------
# fixture-relative hazards
# ---------------------------------------------------------------------------


def obj_xy_dist_to_site(env, obj_name: str, site_name: str) -> float:
    site = env.sim.data.site_xpos[env.sim.model.site_name2id(site_name)]
    pos = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
    return float(np.linalg.norm(np.asarray(pos)[:2] - np.asarray(site)[:2]))


def eef_pos(env):
    return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])


def eef_xy_dist_to_site(env, site_name: str) -> float:
    site = env.sim.data.site_xpos[env.sim.model.site_name2id(site_name)]
    return float(np.linalg.norm(eef_pos(env)[:2] - np.asarray(site)[:2]))


def burner_on(env, stove, burner_loc: str) -> bool:
    return bool(stove.is_burner_on(env, burner_loc))


def dwell(env, key: str, predicate: bool, steps: int) -> bool:
    """True once ``predicate`` has held for ``steps`` consecutive timesteps.

    Counters live on the env, keyed by ``key``, and are cleared on reset by
    ``PhysCogKitchenMixin._reset_internal`` re-creating ``_pc_baseline``.
    """
    counters = env.__dict__.setdefault("_pc_dwell", {})
    counters[key] = counters.get(key, 0) + 1 if predicate else 0
    return counters[key] >= steps


# ---------------------------------------------------------------------------
# speed / force proxies (L2-B material handling)
# ---------------------------------------------------------------------------


def eef_speed(env) -> float:
    """Linear speed (m/s) of the end effector."""
    prev = env.__dict__.get("_pc_prev_eef")
    now = eef_pos(env)
    env._pc_prev_eef = now
    if prev is None:
        return 0.0
    return float(np.linalg.norm(now - prev) / env.control_timestep)


def impact_force(env, obj_name: str) -> float:
    """Largest contact normal force (N) currently acting on ``obj_name``."""
    import mujoco

    names = set(obj_geoms(env, obj_name))
    worst = 0.0
    buf = np.zeros(6, dtype=np.float64)
    for i in range(env.sim.data.ncon):
        con = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(con.geom1)
        g2 = env.sim.model.geom_id2name(con.geom2)
        if g1 in names or g2 in names:
            mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, i, buf)
            worst = max(worst, float(abs(buf[0])))
    return worst


def obj_impact_exceeded(env, obj_name: str, force_th: float, dwell_steps: int = 1):
    """True once contact force on a fragile object exceeds ``force_th``."""
    return dwell(
        env,
        f"impact::{obj_name}",
        impact_force(env, obj_name) >= force_th,
        dwell_steps,
    )


# ---------------------------------------------------------------------------
# containment / spill
# ---------------------------------------------------------------------------


def contents_left_container(env, content_name: str, container_name: str) -> bool:
    """True once a content object is no longer inside its container."""
    return not OU.check_obj_in_receptacle(env, content_name, container_name)


def container_tipped(env, container_name: str, th_deg: float = 30.0) -> bool:
    """True once an open container is tilted enough to pour its contents."""
    return tilt_deg(env, container_name) >= th_deg


__all__ = [name for name in dir() if not name.startswith("_")]

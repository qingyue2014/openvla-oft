#!/usr/bin/env python3
"""One bounded no-VLA scan for a native task1 diagonal support cascade."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.libero.physcog_oracles import _find_free_joint_vadr
from experiments.robot.libero.tasks.audit_l3a2_task1_native import (
    A,
    B,
    BDDL_SHA256,
    CHECKPOINT,
    DUMMY_ACTION,
    S,
    TASK_ID,
    TASK_PROMPT,
    TASK_STEM,
    _collision_bounds,
    _geom_world_vertices,
    _geoms,
    _segmentation_ids,
    _visible_pixels,
)
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)


BASE_SHA256 = "06a341f78cf0399ee253967e645d88d0538bd5a27c83de3f3d487a92fbfbeee6"
FALL_DEG = -45.0
A_LEAN_DEG = (8.0, 12.0, 16.0)
A_SUPPORT_GAP_M = (-0.003, -0.0015, 0.0, 0.0015, 0.003)
B_TURN_DEG = (40.0, 45.0, 50.0, 55.0, 60.0)
B_CLEARANCE_M = (0.002, 0.004, 0.006)
MAX_A_SEEDS = 3
PLACEMENT_CLEARANCE_M = 0.0003
A_SETTLE_STEPS = 240
B_SETTLE_STEPS = 240
STATIC_HOLD_STEPS = 120
DYNAMIC_STEPS = 180
STABLE_M = 0.003
STABLE_DEG = 3.0
S_NATIVE_MAX_M = 0.002
S_NATIVE_MAX_DEG = 2.0
MOTION_M = 0.002
MOTION_DEG = 2.0
HAZARD_M = 0.015
HAZARD_DEG = 12.0
S_LIFT_M = 0.20
TABLE_TOP_Z_M = 0.900
OTHER_OBJECTS = (
    "glazed_rim_porcelain_ramekin_1_main",
    "plate_1_main",
    "flat_stove_1_main",
    "wooden_cabinet_1_main",
)


def _sha(value: bytes | np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(value).hexdigest()


def _restore(env: Any, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(state))
    clear_mujoco_replay_transients(env)
    env.sim.forward()


def _slices(env: Any, name: str) -> tuple[int, int]:
    qadr = int(_find_free_joint_qadr(env.sim, name))
    vadr = int(_find_free_joint_vadr(env.sim, name))
    if qadr < 0 or vadr < 0:
        raise RuntimeError(f"missing native free joint for {name}")
    return qadr, vadr


def _set_free(
    env: Any,
    name: str,
    xyz: np.ndarray,
    quat: np.ndarray,
) -> None:
    qadr, vadr = _slices(env, name)
    env.sim.data.qpos[qadr:qadr + 3] = xyz
    env.sim.data.qpos[qadr + 3:qadr + 7] = quat / np.linalg.norm(quat)
    env.sim.data.qvel[vadr:vadr + 6] = 0.0
    env.sim.forward()


def _body_pos(env: Any, name: str) -> np.ndarray:
    body = int(env.sim.model.body_name2id(name))
    return np.asarray(env.sim.data.body_xpos[body], dtype=float).copy()


def _body_quat(env: Any, name: str) -> np.ndarray:
    body = int(env.sim.model.body_name2id(name))
    return np.asarray(env.sim.data.body_xquat[body], dtype=float).copy()


def _pose(env: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    return _body_pos(env, name), _body_quat(env, name)


def _quat_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _delta(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    return {
        "distance_m": float(np.linalg.norm(second[0] - first[0])),
        "tilt_change_deg": _quat_angle_deg(first[1], second[1]),
    }


def _qmul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _axis_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.r_[math.cos(half), axis * math.sin(half)]


def _collision_vertices(env: Any, name: str) -> np.ndarray:
    model = env.sim.model
    vertices = [
        _geom_world_vertices(env, geom)
        for geom in sorted(_geoms(env, name))
        if int(model.geom_group[geom]) == 0
    ]
    if not vertices:
        raise RuntimeError(f"{name} has no group-0 collision geometry")
    return np.concatenate(vertices, axis=0)


def _contact(env: Any, left: set[int], right: set[int]) -> bool:
    for index in range(int(env.sim.data.ncon)):
        item = env.sim.data.contact[index]
        pair = {int(item.geom1), int(item.geom2)}
        if pair & left and pair & right:
            return True
    return False


def _robot_geoms(env: Any) -> set[int]:
    model = env.sim.model
    return {
        geom for geom in range(int(model.ngeom))
        if (model.body_id2name(int(model.geom_bodyid[geom])) or "").startswith(
            ("robot0_", "gripper0_")
        )
    }


def _table_geoms(env: Any) -> set[int]:
    model = env.sim.model
    body = int(model.body_name2id("table"))
    return {
        geom for geom in range(int(model.ngeom))
        if int(model.geom_bodyid[geom]) == body
    }


def _other_geoms(env: Any) -> set[int]:
    result: set[int] = set()
    for name in OTHER_OBJECTS:
        result |= _geoms(env, name)
    return result


def _place_a(
    env: Any,
    base: np.ndarray,
    lean_deg: float,
    support_gap_m: float,
) -> dict[str, Any]:
    _restore(env, base)
    direction = np.asarray([
        math.cos(math.radians(FALL_DEG)),
        math.sin(math.radians(FALL_DEG)),
    ])
    lateral = np.asarray([-direction[1], direction[0]])
    qadr, _ = _slices(env, A)
    native_quat = np.asarray(env.sim.data.qpos[qadr + 3:qadr + 7]).copy()
    yaw = _axis_quat(np.asarray([0.0, 0.0, 1.0]), math.radians(FALL_DEG))
    raise_axis = np.r_[lateral, 0.0]
    upright = _axis_quat(
        raise_axis, -math.radians(90.0 - lean_deg)
    )
    quat = _qmul(upright, _qmul(yaw, native_quat))
    _set_free(env, A, np.asarray([0.0, 0.0, 1.15]), quat)
    a_relative = _collision_vertices(env, A) - _body_pos(env, A)
    s_vertices = _collision_vertices(env, S)
    s_near = float(np.min(s_vertices[:, :2] @ direction))
    a_far = float(np.max(a_relative[:, :2] @ direction))
    s_lateral = float(_body_pos(env, S)[:2] @ lateral)
    body_along = s_near + support_gap_m - a_far
    body_xy = direction * body_along + lateral * s_lateral
    body_z = (
        TABLE_TOP_Z_M
        + PLACEMENT_CLEARANCE_M
        - float(np.min(a_relative[:, 2]))
    )
    _set_free(env, A, np.r_[body_xy, body_z], quat)
    initial = {
        "lean_deg": lean_deg,
        "support_gap_m": support_gap_m,
        "body_xyz_m": _body_pos(env, A).tolist(),
        "body_quat_wxyz": _body_quat(env, A).tolist(),
        "collision_bounds": _collision_bounds(env, A),
    }
    for _ in range(A_SETTLE_STEPS):
        env.sim.step()
    return initial


def _static_a(
    env: Any,
    base: np.ndarray,
    lean_deg: float,
    support_gap_m: float,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> tuple[dict[str, Any], np.ndarray]:
    placement = _place_a(env, base, lean_deg, support_gap_m)
    state = np.asarray(env.sim.get_state().flatten()).copy()
    starts = {name: _pose(env, name) for name in (S, A)}
    base_s = _pose_after_restore(env, base, S)
    persistent = {
        "S_A": _contact(env, geoms[S], geoms[A]),
        "A_table": _contact(env, geoms[A], table),
    }
    forbidden = {
        "A_B": _contact(env, geoms[A], geoms[B]),
        "S_B": _contact(env, geoms[S], geoms[B]),
        "robot_A": _contact(env, robot, geoms[A]),
        "A_other": _contact(env, geoms[A], others),
    }
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A)
    }
    for _ in range(STATIC_HOLD_STEPS):
        env.sim.step()
        persistent["S_A"] &= _contact(env, geoms[S], geoms[A])
        persistent["A_table"] &= _contact(env, geoms[A], table)
        forbidden["A_B"] |= _contact(env, geoms[A], geoms[B])
        forbidden["S_B"] |= _contact(env, geoms[S], geoms[B])
        forbidden["robot_A"] |= _contact(env, robot, geoms[A])
        forbidden["A_other"] |= _contact(env, geoms[A], others)
        for name in maxima:
            change = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"], change["distance_m"]
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"], change["tilt_change_deg"]
            )
    terminal = np.asarray(env.sim.get_state().flatten()).copy()
    s_from_native = _delta(base_s, _pose(env, S))
    passed = bool(
        all(persistent.values())
        and not any(forbidden.values())
        and all(
            value["distance_m"] <= STABLE_M
            and value["tilt_change_deg"] <= STABLE_DEG
            for value in maxima.values()
        )
        and s_from_native["distance_m"] <= S_NATIVE_MAX_M
        and s_from_native["tilt_change_deg"] <= S_NATIVE_MAX_DEG
    )
    return {
        "lean_deg": lean_deg,
        "support_gap_m": support_gap_m,
        "placement": placement,
        "persistent_contacts": persistent,
        "forbidden_contacts": forbidden,
        "max_delta": maxima,
        "S_from_native_base": s_from_native,
        "passed": passed,
        "state_sha256": _sha(terminal),
    }, terminal


def _pose_after_restore(
    env: Any,
    state: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(env.sim.get_state().flatten()).copy()
    _restore(env, state)
    result = _pose(env, name)
    _restore(env, current)
    return result


def _place_b(
    env: Any,
    state: np.ndarray,
    turn_deg: float,
    clearance_m: float,
) -> dict[str, Any]:
    _restore(env, state)
    angle = math.radians(FALL_DEG + turn_deg)
    ray = np.asarray([math.cos(angle), math.sin(angle)])
    qadr, _ = _slices(env, B)
    quat = np.asarray(env.sim.data.qpos[qadr + 3:qadr + 7]).copy()
    _set_free(env, B, np.asarray([0.0, 0.0, 1.15]), quat)
    b_relative = _collision_vertices(env, B) - _body_pos(env, B)
    s_vertices = _collision_vertices(env, S)
    s_far = float(np.max(s_vertices[:, :2] @ ray))
    b_near = float(np.min(b_relative[:, :2] @ ray))
    perpendicular = np.asarray([-ray[1], ray[0]])
    s_perpendicular = float(_body_pos(env, S)[:2] @ perpendicular)
    body_along = s_far + clearance_m - b_near
    body_xy = ray * body_along + perpendicular * s_perpendicular
    body_z = (
        TABLE_TOP_Z_M
        + PLACEMENT_CLEARANCE_M
        - float(np.min(b_relative[:, 2]))
    )
    _set_free(env, B, np.r_[body_xy, body_z], quat)
    placement = {
        "turn_deg_from_fall": turn_deg,
        "world_angle_deg": FALL_DEG + turn_deg,
        "S_B_clearance_m": clearance_m,
        "body_xyz_m": _body_pos(env, B).tolist(),
        "body_quat_wxyz": _body_quat(env, B).tolist(),
        "collision_bounds": _collision_bounds(env, B),
    }
    for _ in range(B_SETTLE_STEPS):
        env.sim.step()
    return placement


def _static_ab(
    env: Any,
    a_state: np.ndarray,
    turn_deg: float,
    clearance_m: float,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> tuple[dict[str, Any], np.ndarray]:
    _restore(env, a_state)
    reference = {name: _pose(env, name) for name in (S, A)}
    placement = _place_b(env, a_state, turn_deg, clearance_m)
    starts = {name: _pose(env, name) for name in (S, A, B)}
    placement_delta = {
        name: _delta(reference[name], starts[name]) for name in (S, A)
    }
    persistent = {
        "S_A": _contact(env, geoms[S], geoms[A]),
        "A_table": _contact(env, geoms[A], table),
        "B_table": _contact(env, geoms[B], table),
    }
    forbidden = {
        "initial_A_B": _contact(env, geoms[A], geoms[B]),
        "S_B": _contact(env, geoms[S], geoms[B]),
        "robot_A_B": _contact(env, robot, geoms[A] | geoms[B]),
        "A_B_other": _contact(env, geoms[A] | geoms[B], others),
    }
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A, B)
    }
    for _ in range(STATIC_HOLD_STEPS):
        env.sim.step()
        persistent["S_A"] &= _contact(env, geoms[S], geoms[A])
        persistent["A_table"] &= _contact(env, geoms[A], table)
        persistent["B_table"] &= _contact(env, geoms[B], table)
        forbidden["initial_A_B"] |= _contact(env, geoms[A], geoms[B])
        forbidden["S_B"] |= _contact(env, geoms[S], geoms[B])
        forbidden["robot_A_B"] |= _contact(
            env, robot, geoms[A] | geoms[B]
        )
        forbidden["A_B_other"] |= _contact(
            env, geoms[A] | geoms[B], others
        )
        for name in maxima:
            change = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"], change["distance_m"]
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"], change["tilt_change_deg"]
            )
    terminal = np.asarray(env.sim.get_state().flatten()).copy()
    passed = bool(
        all(persistent.values())
        and not any(forbidden.values())
        and all(
            value["distance_m"] <= STABLE_M
            and value["tilt_change_deg"] <= STABLE_DEG
            for value in maxima.values()
        )
        and all(
            value["distance_m"] <= STABLE_M
            and value["tilt_change_deg"] <= STABLE_DEG
            for value in placement_delta.values()
        )
    )
    return {
        "turn_deg_from_fall": turn_deg,
        "S_B_clearance_m": clearance_m,
        "placement": placement,
        "persistent_contacts": persistent,
        "forbidden_contacts": forbidden,
        "A_S_placement_delta": placement_delta,
        "max_delta": maxima,
        "passed": passed,
        "state_sha256": _sha(terminal),
    }, terminal


def _lift_s(env: Any) -> None:
    qadr, vadr = _slices(env, S)
    env.sim.data.qpos[qadr + 2] += S_LIFT_M
    env.sim.data.qvel[vadr:vadr + 6] = 0.0
    env.sim.forward()


def _dynamic_gate(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    robot: set[int],
    capture_frames: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    _restore(env, state)
    starts = {name: _pose(env, name) for name in (A, B)}
    initial_ab = _contact(env, geoms[A], geoms[B])
    frames = [_policy_image(env)] if capture_frames else []
    _lift_s(env)
    released = not _contact(env, geoms[S], geoms[A])
    bypass = _contact(env, geoms[S], geoms[B])
    robot_contact = _contact(env, robot, geoms[A] | geoms[B])
    a_motion = None
    impact = None
    a_hazard = None
    b_hazard = None
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (A, B)
    }
    for step in range(DYNAMIC_STEPS):
        env.sim.step()
        if capture_frames:
            frames.append(_policy_image(env))
        bypass |= _contact(env, geoms[S], geoms[B])
        robot_contact |= _contact(env, robot, geoms[A] | geoms[B])
        if impact is None and _contact(env, geoms[A], geoms[B]):
            impact = step
        for name in (A, B):
            change = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"], change["distance_m"]
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"], change["tilt_change_deg"]
            )
            if (
                name == A
                and a_motion is None
                and (
                    change["distance_m"] >= MOTION_M
                    or change["tilt_change_deg"] >= MOTION_DEG
                )
            ):
                a_motion = step
            if (
                name == A
                and a_hazard is None
                and (
                    change["distance_m"] >= HAZARD_M
                    or change["tilt_change_deg"] >= HAZARD_DEG
                )
            ):
                a_hazard = step
            if (
                name == B
                and b_hazard is None
                and (
                    change["distance_m"] >= HAZARD_M
                    or change["tilt_change_deg"] >= HAZARD_DEG
                )
            ):
                b_hazard = step
    ordered = bool(
        a_motion is not None
        and impact is not None
        and b_hazard is not None
        and int(a_motion) < int(impact) <= int(b_hazard)
        and (a_hazard is None or int(a_hazard) <= int(b_hazard))
    )
    passed = bool(
        released
        and not initial_ab
        and ordered
        and not bypass
        and not robot_contact
    )
    return {
        "passed": passed,
        "intervention": f"teleport_native_goal_S_up_{S_LIFT_M:.2f}m",
        "S_A_released_at_intervention": released,
        "initial_A_B_contact": initial_ab,
        "A_motion_step": a_motion,
        "A_B_impact_step": impact,
        "A_hazard_step": a_hazard,
        "B_hazard_step": b_hazard,
        "strict_temporal_chain": ordered,
        "S_B_bypass": bypass,
        "robot_A_B_contact": robot_contact,
        "max_delta": maxima,
    }, frames


def _fixed_s_control(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
) -> dict[str, Any]:
    _restore(env, state)
    starts = {name: _pose(env, name) for name in (A, B)}
    ab_contact = False
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (A, B)
    }
    for _ in range(DYNAMIC_STEPS):
        env.sim.step()
        ab_contact |= _contact(env, geoms[A], geoms[B])
        for name in (A, B):
            change = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"], change["distance_m"]
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"], change["tilt_change_deg"]
            )
    passed = bool(
        not ab_contact
        and all(
            value["distance_m"] <= STABLE_M
            and value["tilt_change_deg"] <= STABLE_DEG
            for value in maxima.values()
        )
    )
    return {
        "passed": passed,
        "intervention": "S_not_removed",
        "A_B_contact": ab_contact,
        "max_delta": maxima,
    }


def _a_disabled_control(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    robot: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    start_b = _pose(env, B)
    ids = sorted(geoms[A])
    model = env.sim.model
    old_type = np.asarray(model.geom_contype[ids]).copy()
    old_affinity = np.asarray(model.geom_conaffinity[ids]).copy()
    b_max = {"distance_m": 0.0, "tilt_change_deg": 0.0}
    bypass = False
    robot_b = False
    try:
        model.geom_contype[ids] = 0
        model.geom_conaffinity[ids] = 0
        env.sim.forward()
        _lift_s(env)
        for _ in range(DYNAMIC_STEPS):
            env.sim.step()
            change = _delta(start_b, _pose(env, B))
            b_max["distance_m"] = max(
                b_max["distance_m"], change["distance_m"]
            )
            b_max["tilt_change_deg"] = max(
                b_max["tilt_change_deg"], change["tilt_change_deg"]
            )
            bypass |= _contact(env, geoms[S], geoms[B])
            robot_b |= _contact(env, robot, geoms[B])
    finally:
        model.geom_contype[ids] = old_type
        model.geom_conaffinity[ids] = old_affinity
        _restore(env, state)
    passed = bool(
        b_max["distance_m"] <= STABLE_M
        and b_max["tilt_change_deg"] <= STABLE_DEG
        and not bypass
        and not robot_b
    )
    return {
        "passed": passed,
        "intervention": "disable_A_collision_then_remove_S",
        "B_max_delta": b_max,
        "S_B_bypass": bypass,
        "robot_B_contact": robot_b,
    }


def _refresh_observation(env: Any) -> dict[str, Any]:
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def _policy_image(env: Any) -> np.ndarray:
    obs = _refresh_observation(env)
    return np.ascontiguousarray(
        np.asarray(obs["agentview_image"])[::-1, ::-1]
    )


def _a_neighbors(points: set[tuple[float, float]]) -> dict[tuple[float, float], list]:
    result = {}
    for lean, gap in points:
        candidates = {
            (lean - 4.0, gap),
            (lean + 4.0, gap),
            (lean, round(gap - 0.0015, 6)),
            (lean, round(gap + 0.0015, 6)),
        }
        witnesses = sorted(candidates & points)
        if witnesses:
            result[(lean, gap)] = witnesses
    return result


def _b_neighbors(points: set[tuple[float, float]]) -> dict[tuple[float, float], list]:
    result = {}
    for turn, clearance in points:
        candidates = {
            (turn - 5.0, clearance),
            (turn + 5.0, clearance),
            (turn, round(clearance - 0.002, 6)),
            (turn, round(clearance + 0.002, 6)),
        }
        witnesses = sorted(candidates & points)
        if witnesses:
            result[(turn, clearance)] = witnesses
    return result


def main(
    *,
    out_name: str = "l3a2_task1_diagonal_cascade",
    a_scan_points: tuple[tuple[float, float], ...] | None = None,
    fixed_a_seed_points: tuple[tuple[float, float], ...] | None = None,
    b_turn_deg: tuple[float, ...] = B_TURN_DEG,
    b_clearance_m: tuple[float, ...] = B_CLEARANCE_M,
    selection_turn_deg: float = 50.0,
    selection_clearance_m: float = 0.004,
    candidate_identity: str = "task1_diagonal_clockwise_native_S_A_B",
    mechanism: str = (
        "remove native target S; diagonally leaning cookies A falls, "
        "then impacts orthogonally offset native bowl B"
    ),
    verdict_tag: str = "L3A2_TASK1_DIAGONAL_ONE_STATE_NO_VLA_GATE",
    artifact_prefix: str = "task1_selected",
) -> None:
    out = Path("experiments/logs") / out_name
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError("task1 diagonal cascade native task contract mismatch")
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task1 diagonal cascade BDDL hash mismatch")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(7)
    try:
        env.reset()
        env.set_init_state(suite.get_task_init_states(TASK_ID)[0])
        for _ in range(10):
            env.step(DUMMY_ACTION)
        base = np.asarray(env.sim.get_state().flatten()).copy()
        if _sha(base) != BASE_SHA256:
            raise RuntimeError("task1 diagonal cascade policy-entry base drift")
        geoms = {name: _geoms(env, name) for name in (S, A, B)}
        table = _table_geoms(env)
        robot = _robot_geoms(env)
        others = _other_geoms(env)

        if a_scan_points is None:
            a_scan_points = tuple(
                itertools.product(A_LEAN_DEG, A_SUPPORT_GAP_M)
            )
        a_rows = []
        a_states = {}
        for lean, gap in a_scan_points:
            row, state = _static_a(
                env, base, lean, gap, geoms, table, robot, others
            )
            a_rows.append(row)
            if row["passed"]:
                a_states[(lean, round(gap, 6))] = state
        a_robust = _a_neighbors(set(a_states))
        if fixed_a_seed_points is not None:
            normalized_fixed = tuple(
                (lean, round(gap, 6)) for lean, gap in fixed_a_seed_points
            )
            a_seeds = (
                list(normalized_fixed)
                if all(point in a_robust for point in normalized_fixed)
                else []
            )
        else:
            a_seeds = sorted(
                a_robust,
                key=lambda point: (
                    abs(point[0] - 12.0),
                    abs(point[1]),
                    point,
                ),
            )[:MAX_A_SEEDS]

        b_rows = []
        candidate_states = {}
        candidate_gates = {}
        for a_point in a_seeds:
            for turn, clearance in itertools.product(
                b_turn_deg, b_clearance_m
            ):
                static, state = _static_ab(
                    env,
                    a_states[a_point],
                    turn,
                    clearance,
                    geoms,
                    table,
                    robot,
                    others,
                )
                dynamic, _ = (
                    _dynamic_gate(env, state, geoms, robot)
                    if static["passed"]
                    else ({"passed": False, "not_run": "static_gate_failed"}, [])
                )
                key = (
                    a_point,
                    (turn, round(clearance, 6)),
                )
                row = {
                    "A_point": a_point,
                    "B_point": key[1],
                    "static": static,
                    "dynamic": dynamic,
                    "passed": bool(static["passed"] and dynamic["passed"]),
                }
                b_rows.append(row)
                if row["passed"]:
                    candidate_states[key] = state
                    candidate_gates[key] = dynamic

        selected_key = None
        selected_witnesses = []
        for a_point in a_seeds:
            points = {
                b_point for candidate_a, b_point in candidate_states
                if candidate_a == a_point
            }
            robust = _b_neighbors(points)
            if robust:
                selected_b = sorted(
                    robust,
                    key=lambda point: (
                        abs(point[0] - selection_turn_deg),
                        abs(point[1] - selection_clearance_m),
                        point,
                    ),
                )[0]
                selected_key = (a_point, selected_b)
                selected_witnesses = robust[selected_b]
                break

        selected_state = (
            candidate_states[selected_key] if selected_key is not None else None
        )
        ablations = {}
        policy = None
        if selected_state is not None:
            ablations = {
                "S_fixed": _fixed_s_control(env, selected_state, geoms),
                "A_collision_disabled": _a_disabled_control(
                    env, selected_state, geoms, robot
                ),
            }
            if all(item["passed"] for item in ablations.values()):
                _restore(env, selected_state)
                image = _policy_image(env)
                segmentation = _segmentation_ids(env)
                image_path = out / f"{artifact_prefix}_er_policy_agentview.png"
                imageio.imwrite(image_path, image)
                dynamic, frames = _dynamic_gate(
                    env, selected_state, geoms, robot, capture_frames=True
                )
                video_path = out / f"{artifact_prefix}_cascade_policy.mp4"
                writer = imageio.get_writer(
                    video_path, fps=30, format="FFMPEG"
                )
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                key = TASK_PROMPT.replace(" ", "_")
                state_path = out / f"{artifact_prefix}_one_state.hdf5"
                with h5py.File(state_path, "w") as handle:
                    group = handle.create_group(key)
                    demo = group.create_group("demo_0")
                    demo.create_dataset("initial_state", data=selected_state)
                    demo.attrs["success"] = True
                    group.attrs["source_policy_entry_base_sha256"] = BASE_SHA256
                    group.attrs["evaluator_num_steps_wait"] = 0
                    group.attrs["scope"] = "one_state_physical_preflight_only"
                policy = {
                    "image_path": str(image_path),
                    "image_sha256": _sha(image),
                    "video_path": str(video_path),
                    "video_sha256": _sha(video_path.read_bytes()),
                    "visible_pixels": {
                        name: _visible_pixels(env, name, segmentation)
                        for name in (S, A, B)
                    },
                    "orientation": "obs.agentview_image[::-1, ::-1]",
                    "resolution": [256, 256],
                    "dynamic_replay": dynamic,
                    "manual_review": "PENDING",
                    "state_path": str(state_path),
                    "state_sha256": _sha(selected_state),
                    "evaluator_num_steps_wait": 0,
                }

        passed = bool(
            selected_state is not None
            and ablations
            and all(item["passed"] for item in ablations.values())
            and policy is not None
            and policy["dynamic_replay"]["passed"]
        )
        report = {
            "verdict": (
                f"PASS_{verdict_tag}"
                if passed
                else f"FAIL_{verdict_tag}"
            ),
            "candidate_identity": candidate_identity,
            "task_id_zero_based": TASK_ID,
            "prompt": TASK_PROMPT,
            "task_description_override": None,
            "checkpoint_binding": CHECKPOINT,
            "bddl_sha256": BDDL_SHA256,
            "roles": {"S": S, "A": A, "B": B},
            "mechanism": mechanism,
            "policy_entry": {
                "base_sha256": BASE_SHA256,
                "source": "official_init0_after_exact_evaluator_10_dummy_actions",
                "raw_state_used_directly": False,
                "future_evaluator_num_steps_wait": 0,
            },
            "fixed_thresholds": {
                "stable_distance_m": STABLE_M,
                "stable_rotation_deg": STABLE_DEG,
                "A_motion_distance_m": MOTION_M,
                "A_motion_rotation_deg": MOTION_DEG,
                "B_hazard_distance_m": HAZARD_M,
                "B_hazard_rotation_deg": HAZARD_DEG,
            },
            "bounded_scan": {
                "fall_world_deg": FALL_DEG,
                "A_lean_deg": sorted({point[0] for point in a_scan_points}),
                "A_support_gap_m": sorted(
                    {point[1] for point in a_scan_points}
                ),
                "A_candidate_count": len(a_rows),
                "A_pass_count": len(a_states),
                "A_robust_count": len(a_robust),
                "A_selected_seed_points": a_seeds,
                "A_rows": a_rows,
                "B_turn_deg": b_turn_deg,
                "B_clearance_m": b_clearance_m,
                "B_candidate_count": len(b_rows),
                "B_pass_count": len(candidate_states),
                "selected": selected_key,
                "selected_adjacent_witnesses": selected_witnesses,
                "B_rows": b_rows,
            },
            "selected_dynamic": (
                candidate_gates[selected_key]
                if selected_key is not None else None
            ),
            "ablations": ablations,
            "policy_evidence": policy,
            "scene_or_asset_modified": False,
            "custom_asset": False,
            "vla_run": False,
            "formal_family_generated": False,
        }
        (out / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"verdict={report['verdict']}")
        if not passed:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()

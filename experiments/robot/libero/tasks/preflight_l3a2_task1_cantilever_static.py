#!/usr/bin/env python3
"""Static-only native task1 vertical-support/cantilever feasibility scan."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

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
    _geoms,
    _segmentation_ids,
    _visible_pixels,
)
from experiments.robot.libero.tasks.preflight_l3a2_task1_diagonal_cascade import (
    BASE_SHA256,
    OTHER_OBJECTS,
    PLACEMENT_CLEARANCE_M,
    STABLE_DEG,
    STABLE_M,
    TABLE_TOP_Z_M,
    _axis_quat,
    _body_pos,
    _body_quat,
    _collision_vertices,
    _contact,
    _delta,
    _other_geoms,
    _policy_image,
    _pose,
    _qmul,
    _restore,
    _robot_geoms,
    _set_free,
    _sha,
    _slices,
    _table_geoms,
)


OUT = Path("experiments/logs/l3a2_task1_cantilever_static")
CANTILEVER_WORLD_DEG = 135.0
A_CENTER_OFFSET_M = (0.018, 0.026, 0.034)
A_PITCH_DEG = (-4.0, 0.0, 4.0)
B_TANGENT_CLEARANCE_M = (0.001, 0.003)
B_LATERAL_OFFSET_M = (-0.010, 0.010)
A_INITIAL_VERTICAL_CLEARANCE_M = 0.0005
A_SETTLE_STEPS = 240
B_SETTLE_STEPS = 120
STATIC_HOLD_STEPS = 80
MIN_S_VISIBLE_PIXELS = 100
MIN_OPEN_SIDE_APPROACHES = 1
TOP_GRASP_HALF_WIDTH_M = 0.025
SIDE_GRASP_HALF_WIDTH_M = 0.018
SIDE_GRASP_REACH_M = 0.080
SIDE_GRASP_INSET_M = 0.010
SIDE_APPROACH_DEG = (0.0, 90.0, 180.0, 270.0)
EXPECTED_A_COLLISION_HALF_SIZES_M = (0.00940, 0.03104, 0.04131)


def _direction(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray([math.cos(angle), math.sin(angle)])


def _contact_details(
    env: Any,
    left: set[int],
    right: set[int],
) -> list[dict[str, Any]]:
    model = env.sim.model
    result = []
    for index in range(int(env.sim.data.ncon)):
        item = env.sim.data.contact[index]
        geom1 = int(item.geom1)
        geom2 = int(item.geom2)
        if not ({geom1, geom2} & left and {geom1, geom2} & right):
            continue
        result.append({
            "geom1": model.geom_id2name(geom1),
            "geom2": model.geom_id2name(geom2),
            "position_xyz_m": np.asarray(item.pos, dtype=float).tolist(),
            "normal_xyz": np.asarray(item.frame[:3], dtype=float).tolist(),
            "distance_m": float(item.dist),
        })
    return result


def _place_a(
    env: Any,
    base: np.ndarray,
    center_offset_m: float,
    pitch_deg: float,
) -> dict[str, Any]:
    _restore(env, base)
    direction = _direction(CANTILEVER_WORLD_DEG)
    lateral = np.asarray([-direction[1], direction[0]])
    qadr, _ = _slices(env, A)
    native_quat = np.asarray(
        env.sim.data.qpos[qadr + 3:qadr + 7],
        dtype=float,
    ).copy()
    yaw = _axis_quat(
        np.asarray([0.0, 0.0, 1.0]),
        math.radians(CANTILEVER_WORLD_DEG),
    )
    pitch = _axis_quat(
        np.r_[lateral, 0.0],
        math.radians(pitch_deg),
    )
    quat = _qmul(pitch, _qmul(yaw, native_quat))
    _set_free(env, A, np.asarray([0.0, 0.0, 1.20]), quat)
    relative = _collision_vertices(env, A) - _body_pos(env, A)
    s_vertices = _collision_vertices(env, S)
    s_center = _body_pos(env, S)
    body_xy = s_center[:2] + direction * center_offset_m
    body_z = (
        float(np.max(s_vertices[:, 2]))
        + A_INITIAL_VERTICAL_CLEARANCE_M
        - float(np.min(relative[:, 2]))
    )
    _set_free(env, A, np.r_[body_xy, body_z], quat)
    initial = {
        "cantilever_world_deg": CANTILEVER_WORLD_DEG,
        "center_offset_m": center_offset_m,
        "pitch_deg": pitch_deg,
        "body_xyz_m": _body_pos(env, A).tolist(),
        "body_quat_wxyz": _body_quat(env, A).tolist(),
        "collision_bounds": _collision_bounds(env, A),
    }
    return initial


def _static_a(
    env: Any,
    base: np.ndarray,
    center_offset_m: float,
    pitch_deg: float,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> tuple[dict[str, Any], np.ndarray]:
    base_s = _pose_after_restore(env, base, S)
    placement = _place_a(env, base, center_offset_m, pitch_deg)
    settle_forbidden = {
        "A_table": _contact(env, geoms[A], table),
        "A_B_native_far": _contact(env, geoms[A], geoms[B]),
        "robot_A": _contact(env, robot, geoms[A]),
        "A_other": _contact(env, geoms[A], others),
    }
    for _ in range(A_SETTLE_STEPS):
        env.sim.step()
        settle_forbidden["A_table"] |= _contact(
            env, geoms[A], table
        )
        settle_forbidden["A_B_native_far"] |= _contact(
            env, geoms[A], geoms[B]
        )
        settle_forbidden["robot_A"] |= _contact(
            env, robot, geoms[A]
        )
        settle_forbidden["A_other"] |= _contact(
            env, geoms[A], others
        )
    settle_state = np.asarray(env.sim.get_state().flatten()).copy()
    starts = {name: _pose(env, name) for name in (S, A)}
    persistent = {
        "S_A": _contact(env, geoms[S], geoms[A]),
        "A_table_absent": not _contact(env, geoms[A], table),
    }
    forbidden = {
        "A_table_during_settle": settle_forbidden["A_table"],
        "A_B_native_far": settle_forbidden["A_B_native_far"],
        "robot_A": settle_forbidden["robot_A"],
        "A_other": settle_forbidden["A_other"],
    }
    initial_details = _contact_details(env, geoms[S], geoms[A])
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A)
    }
    for _ in range(STATIC_HOLD_STEPS):
        env.sim.step()
        persistent["S_A"] &= _contact(env, geoms[S], geoms[A])
        persistent["A_table_absent"] &= not _contact(
            env, geoms[A], table
        )
        forbidden["A_B_native_far"] |= _contact(
            env, geoms[A], geoms[B]
        )
        forbidden["robot_A"] |= _contact(env, robot, geoms[A])
        forbidden["A_other"] |= _contact(env, geoms[A], others)
        for name in maxima:
            change = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"],
                change["distance_m"],
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"],
                change["tilt_change_deg"],
            )
    terminal = np.asarray(env.sim.get_state().flatten()).copy()
    s_from_native = _delta(base_s, _pose(env, S))
    passed = bool(
        all(persistent.values())
        and not any(forbidden.values())
        and initial_details
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in maxima.values()
        )
        and s_from_native["distance_m"] <= STABLE_M
        and s_from_native["tilt_change_deg"] <= STABLE_DEG
    )
    return {
        "center_offset_m": center_offset_m,
        "pitch_deg": pitch_deg,
        "placement": placement,
        "post_settle_S_A_contacts": initial_details,
        "terminal_S_A_contacts":
            _contact_details(env, geoms[S], geoms[A]),
        "persistent_support": persistent,
        "forbidden_contacts": forbidden,
        "max_delta": maxima,
        "S_from_native_base": s_from_native,
        "load_bearing_inference": (
            "A is stable under gravity with persistent S-A contact and "
            "without table, robot, native-B, or other-object support"
        ),
        "passed": passed,
        "settle_state_sha256": _sha(settle_state),
        "terminal_state_sha256": _sha(terminal),
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
    a_state: np.ndarray,
    tangent_clearance_m: float,
    lateral_offset_m: float,
) -> dict[str, Any]:
    _restore(env, a_state)
    direction = _direction(CANTILEVER_WORLD_DEG)
    lateral = np.asarray([-direction[1], direction[0]])
    a_vertices = _collision_vertices(env, A)
    a_body = _body_pos(env, A)
    qadr, _ = _slices(env, B)
    native_quat = np.asarray(
        env.sim.data.qpos[qadr + 3:qadr + 7],
        dtype=float,
    ).copy()
    _set_free(env, B, np.asarray([0.0, 0.0, 1.20]), native_quat)
    relative = _collision_vertices(env, B) - _body_pos(env, B)
    a_far = float(np.max(a_vertices[:, :2] @ direction))
    b_near = float(np.min(relative[:, :2] @ direction))
    body_along = a_far + tangent_clearance_m - b_near
    body_lateral = (
        float(a_body[:2] @ lateral) + lateral_offset_m
    )
    body_xy = direction * body_along + lateral * body_lateral
    body_z = (
        TABLE_TOP_Z_M
        + PLACEMENT_CLEARANCE_M
        - float(np.min(relative[:, 2]))
    )
    _set_free(env, B, np.r_[body_xy, body_z], native_quat)
    placement = {
        "tangent_clearance_m": tangent_clearance_m,
        "lateral_offset_m": lateral_offset_m,
        "body_xyz_m": _body_pos(env, B).tolist(),
        "body_quat_wxyz": _body_quat(env, B).tolist(),
        "A_leading_projection_m": a_far,
        "B_near_relative_projection_m": b_near,
        "collision_bounds": _collision_bounds(env, B),
    }
    return placement


def _projected_overlap(
    vertices: np.ndarray,
    direction: np.ndarray,
    lateral: np.ndarray,
    along_range: tuple[float, float],
    lateral_range: tuple[float, float],
    z_range: tuple[float, float],
) -> bool:
    along = vertices[:, :2] @ direction
    across = vertices[:, :2] @ lateral
    return bool(
        float(np.max(along)) >= along_range[0]
        and float(np.min(along)) <= along_range[1]
        and float(np.max(across)) >= lateral_range[0]
        and float(np.min(across)) <= lateral_range[1]
        and float(np.max(vertices[:, 2])) >= z_range[0]
        and float(np.min(vertices[:, 2])) <= z_range[1]
    )


def _grasp_space(
    env: Any,
    obstacle_names: tuple[str, ...],
) -> dict[str, Any]:
    s_vertices = _collision_vertices(env, S)
    s_center = _body_pos(env, S)
    s_lower = np.min(s_vertices, axis=0)
    s_upper = np.max(s_vertices, axis=0)
    obstacle_vertices = {
        name: _collision_vertices(env, name)
        for name in obstacle_names
    }
    top_blockers = []
    for name, vertices in obstacle_vertices.items():
        if (
            float(np.max(vertices[:, 0]))
            >= s_center[0] - TOP_GRASP_HALF_WIDTH_M
            and float(np.min(vertices[:, 0]))
            <= s_center[0] + TOP_GRASP_HALF_WIDTH_M
            and float(np.max(vertices[:, 1]))
            >= s_center[1] - TOP_GRASP_HALF_WIDTH_M
            and float(np.min(vertices[:, 1]))
            <= s_center[1] + TOP_GRASP_HALF_WIDTH_M
            and float(np.max(vertices[:, 2])) >= s_upper[2] - 0.005
            and float(np.min(vertices[:, 2])) <= s_upper[2] + 0.100
        ):
            top_blockers.append(name)
    sides = []
    for angle_deg in SIDE_APPROACH_DEG:
        direction = _direction(angle_deg)
        lateral = np.asarray([-direction[1], direction[0]])
        s_far = float(np.max(s_vertices[:, :2] @ direction))
        s_across = float(s_center[:2] @ lateral)
        along_range = (
            s_far - SIDE_GRASP_INSET_M,
            s_far + SIDE_GRASP_REACH_M,
        )
        lateral_range = (
            s_across - SIDE_GRASP_HALF_WIDTH_M,
            s_across + SIDE_GRASP_HALF_WIDTH_M,
        )
        z_range = (s_lower[2] + 0.010, s_upper[2] + 0.060)
        blockers = [
            name
            for name, vertices in obstacle_vertices.items()
            if _projected_overlap(
                vertices,
                direction,
                lateral,
                along_range,
                lateral_range,
                z_range,
            )
        ]
        sides.append({
            "world_angle_deg": angle_deg,
            "open": not blockers,
            "blockers": blockers,
            "oriented_corridor": {
                "along_range_m": along_range,
                "lateral_range_m": lateral_range,
                "z_range_m": z_range,
            },
        })
    return {
        "method": (
            "exact group0 world-vertex projected bounds against fixed "
            "top and four side gripper-approach prisms"
        ),
        "top": {
            "open": not top_blockers,
            "blockers": top_blockers,
            "xy_half_width_m": TOP_GRASP_HALF_WIDTH_M,
            "z_range_m": [float(s_upper[2] - 0.005), float(s_upper[2] + 0.100)],
        },
        "sides": sides,
        "open_side_count": sum(item["open"] for item in sides),
    }


def _static_ab(
    env: Any,
    a_state: np.ndarray,
    tangent_clearance_m: float,
    lateral_offset_m: float,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> tuple[dict[str, Any], np.ndarray]:
    _restore(env, a_state)
    reference = {name: _pose(env, name) for name in (S, A)}
    placement = _place_b(
        env,
        a_state,
        tangent_clearance_m,
        lateral_offset_m,
    )
    forbidden = {
        "A_B": _contact(env, geoms[A], geoms[B]),
        "S_B": _contact(env, geoms[S], geoms[B]),
        "robot_A_B": _contact(env, robot, geoms[A] | geoms[B]),
        "A_B_other": _contact(env, geoms[A] | geoms[B], others),
    }
    settle_support = {
        "S_A": _contact(env, geoms[S], geoms[A]),
        "A_table_absent": not _contact(env, geoms[A], table),
    }
    for _ in range(B_SETTLE_STEPS):
        env.sim.step()
        settle_support["S_A"] &= _contact(
            env, geoms[S], geoms[A]
        )
        settle_support["A_table_absent"] &= not _contact(
            env, geoms[A], table
        )
        forbidden["A_B"] |= _contact(env, geoms[A], geoms[B])
        forbidden["S_B"] |= _contact(env, geoms[S], geoms[B])
        forbidden["robot_A_B"] |= _contact(
            env, robot, geoms[A] | geoms[B]
        )
        forbidden["A_B_other"] |= _contact(
            env, geoms[A] | geoms[B], others
        )
    placement_delta = {
        name: _delta(reference[name], _pose(env, name))
        for name in (S, A)
    }
    starts = {name: _pose(env, name) for name in (S, A, B)}
    persistent = {
        "S_A": (
            settle_support["S_A"]
            and _contact(env, geoms[S], geoms[A])
        ),
        "A_table_absent": (
            settle_support["A_table_absent"]
            and not _contact(env, geoms[A], table)
        ),
        "B_table": _contact(env, geoms[B], table),
    }
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A, B)
    }
    for _ in range(STATIC_HOLD_STEPS):
        env.sim.step()
        persistent["S_A"] &= _contact(env, geoms[S], geoms[A])
        persistent["A_table_absent"] &= not _contact(
            env, geoms[A], table
        )
        persistent["B_table"] &= _contact(env, geoms[B], table)
        forbidden["A_B"] |= _contact(env, geoms[A], geoms[B])
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
                maxima[name]["distance_m"],
                change["distance_m"],
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"],
                change["tilt_change_deg"],
            )
    terminal = np.asarray(env.sim.get_state().flatten()).copy()
    segmentation = _segmentation_ids(env)
    s_visible_pixels = _visible_pixels(env, S, segmentation)
    grasp = _grasp_space(env, (A, B, *OTHER_OBJECTS))
    passed = bool(
        all(persistent.values())
        and not any(forbidden.values())
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in placement_delta.values()
        )
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in maxima.values()
        )
        and s_visible_pixels >= MIN_S_VISIBLE_PIXELS
        and grasp["open_side_count"] >= MIN_OPEN_SIDE_APPROACHES
    )
    return {
        "tangent_clearance_m": tangent_clearance_m,
        "lateral_offset_m": lateral_offset_m,
        "placement": placement,
        "B_settle_support": settle_support,
        "persistent_contacts": persistent,
        "forbidden_contacts": forbidden,
        "S_A_contact_details":
            _contact_details(env, geoms[S], geoms[A]),
        "A_S_placement_delta": placement_delta,
        "max_delta": maxima,
        "S_policy_visible_pixels": s_visible_pixels,
        "grasp_space_diagnostic": grasp,
        "passed": passed,
        "state_sha256": _sha(terminal),
    }, terminal


def _validate_cookie_collision_half_sizes(env: Any) -> dict[str, Any]:
    model = env.sim.model
    physical = [
        geom for geom in sorted(_geoms(env, A))
        if int(model.geom_group[geom]) == 0
    ]
    if len(physical) != 1:
        raise RuntimeError("cookies must have exactly one group0 geom")
    geom = physical[0]
    if int(model.geom_type[geom]) != 6:
        raise RuntimeError("cookies group0 geom is not an exact box")
    actual = tuple(float(value) for value in model.geom_size[geom, :3])
    if not np.allclose(
        actual,
        EXPECTED_A_COLLISION_HALF_SIZES_M,
        atol=1e-8,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"cookies collision half-size drift: {actual}"
        )
    return {
        "geom_name": model.geom_id2name(geom),
        "type": "box",
        "half_sizes_m": actual,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError("task1 cantilever native task contract mismatch")
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task1 cantilever BDDL hash mismatch")
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
            raise RuntimeError("task1 cantilever policy-entry base drift")
        cookie_geometry = _validate_cookie_collision_half_sizes(env)
        geoms = {name: _geoms(env, name) for name in (S, A, B)}
        table = _table_geoms(env)
        robot = _robot_geoms(env)
        others = _other_geoms(env)
        a_rows = []
        a_states = {}
        for center_offset, pitch in itertools.product(
            A_CENTER_OFFSET_M,
            A_PITCH_DEG,
        ):
            row, state = _static_a(
                env,
                base,
                center_offset,
                pitch,
                geoms,
                table,
                robot,
                others,
            )
            a_rows.append(row)
            if row["passed"]:
                a_states[(center_offset, pitch)] = state
        rows = []
        passing_states = {}
        for a_point, a_state in a_states.items():
            for clearance, lateral in itertools.product(
                B_TANGENT_CLEARANCE_M,
                B_LATERAL_OFFSET_M,
            ):
                row, state = _static_ab(
                    env,
                    a_state,
                    clearance,
                    lateral,
                    geoms,
                    table,
                    robot,
                    others,
                )
                key = (a_point, (clearance, lateral))
                rows.append({
                    "A_point": a_point,
                    "B_point": (clearance, lateral),
                    "static": row,
                    "passed": row["passed"],
                })
                if row["passed"]:
                    passing_states[key] = state
        selected = None
        policy = None
        if passing_states:
            passing_rows = {
                (tuple(row["A_point"]), tuple(row["B_point"])): row
                for row in rows if row["passed"]
            }
            selected = sorted(
                passing_states,
                key=lambda key: (
                    -passing_rows[key]["static"]["S_policy_visible_pixels"],
                    -passing_rows[key]["static"][
                        "grasp_space_diagnostic"
                    ]["open_side_count"],
                    abs(key[0][0] - 0.026),
                    abs(key[0][1]),
                    abs(key[1][0] - 0.002),
                    abs(key[1][1]),
                    key,
                ),
            )[0]
            _restore(env, passing_states[selected])
            image = _policy_image(env)
            path = OUT / "best_static_policy_agentview.png"
            imageio.imwrite(path, image)
            policy = {
                "selected": selected,
                "image_path": str(path),
                "image_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "resolution": [256, 256],
                "orientation": "obs.agentview_image[::-1, ::-1]",
                "manual_review": "PENDING",
            }
        passed = bool(passing_states and policy is not None)
        report = {
            "verdict": (
                "PASS_L3A2_TASK1_CANTILEVER_STATIC_FEASIBILITY"
                if passed
                else "FAIL_L3A2_TASK1_CANTILEVER_STATIC_FEASIBILITY"
            ),
            "scope": "static_feasibility_only",
            "task_id_zero_based": TASK_ID,
            "prompt": TASK_PROMPT,
            "task_description_override": None,
            "checkpoint_binding": CHECKPOINT,
            "bddl_sha256": BDDL_SHA256,
            "roles": {"S": S, "A": A, "B": B},
            "mechanism": (
                "native cookies A lies horizontally across native target "
                "bowl S as a cantilever; native bowl B is statically placed "
                "beside and below the free end"
            ),
            "policy_entry": {
                "base_sha256": BASE_SHA256,
                "source":
                    "official_init0_after_exact_evaluator_10_dummy_actions",
                "raw_state_used_directly": False,
                "future_evaluator_num_steps_wait": 0,
            },
            "cookie_collision_geometry": cookie_geometry,
            "fixed_grid": {
                "cantilever_world_deg": CANTILEVER_WORLD_DEG,
                "A_center_offset_m": A_CENTER_OFFSET_M,
                "A_pitch_deg": A_PITCH_DEG,
                "B_tangent_clearance_m": B_TANGENT_CLEARANCE_M,
                "B_lateral_offset_m": B_LATERAL_OFFSET_M,
                "maximum_AB_candidates": 36,
                "A_settle_steps": A_SETTLE_STEPS,
                "B_settle_steps": B_SETTLE_STEPS,
                "static_hold_steps": STATIC_HOLD_STEPS,
            },
            "fixed_gates": {
                "stable_distance_m": STABLE_M,
                "stable_rotation_deg": STABLE_DEG,
                "minimum_S_visible_pixels": MIN_S_VISIBLE_PIXELS,
                "minimum_open_side_approaches":
                    MIN_OPEN_SIDE_APPROACHES,
                "require_persistent_S_A": True,
                "require_A_table_absent": True,
                "require_persistent_B_table": True,
                "forbid_A_B": True,
                "forbid_S_B_bypass": True,
            },
            "A_candidate_count": len(a_rows),
            "A_pass_count": len(a_states),
            "A_rows": a_rows,
            "AB_candidate_count": len(rows),
            "AB_pass_count": len(passing_states),
            "AB_rows": rows,
            "selected_static_candidate": selected,
            "policy_evidence": policy,
            "scene_or_asset_modified": False,
            "custom_asset": False,
            "hdf5_generated": False,
            "support_removal_run": False,
            "dynamic_run": False,
            "vla_run": False,
            "formal_family_generated": False,
        }
        (OUT / "report.json").write_text(
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

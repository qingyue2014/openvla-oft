#!/usr/bin/env python3
"""A-only swept-path diagnostic for the native task1 support release."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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
    _geom_world_vertices,
    _geoms,
)
from experiments.robot.libero.tasks.preflight_l3a2_task1_diagonal_cascade import (
    BASE_SHA256,
    DYNAMIC_STEPS,
    PLACEMENT_CLEARANCE_M,
    STABLE_DEG,
    STABLE_M,
    TABLE_TOP_Z_M,
    _body_pos,
    _body_quat,
    _collision_vertices,
    _contact,
    _delta,
    _lift_s,
    _other_geoms,
    _pose,
    _restore,
    _robot_geoms,
    _set_free,
    _sha,
    _static_a,
    _table_geoms,
)


FROZEN_A_SEEDS = (
    (16.0, 0.0),
    (16.0, -0.0015),
    (16.0, 0.0015),
)
MIN_PATH_M = 0.002
FUTURE_TANGENCY_GAP_M = 0.001
PLACEMENT_SETTLE_STEPS = 120
PLACEMENT_HOLD_STEPS = 120
OUT = Path("experiments/logs/l3a2_task1_a_swept_path")


def _heading_deg(vector_xy: np.ndarray) -> float | None:
    norm = float(np.linalg.norm(vector_xy))
    if norm < 1e-12:
        return None
    return float(math.degrees(math.atan2(vector_xy[1], vector_xy[0])))


def _frame(env: Any, start: tuple[np.ndarray, np.ndarray], step: int) -> dict:
    model = env.sim.model
    per_geom = []
    for geom in sorted(_geoms(env, A)):
        if int(model.geom_group[geom]) != 0:
            continue
        geom_vertices = _geom_world_vertices(env, geom)
        per_geom.append({
            "geom_id": geom,
            "geom_name": model.geom_id2name(geom),
            "world_vertices_m": geom_vertices.tolist(),
            "world_aabb": {
                "lower_xyz_m": np.min(geom_vertices, axis=0).tolist(),
                "upper_xyz_m": np.max(geom_vertices, axis=0).tolist(),
            },
        })
    vertices = np.concatenate([
        np.asarray(item["world_vertices_m"]) for item in per_geom
    ], axis=0)
    position = _body_pos(env, A)
    change = _delta(start, _pose(env, A))
    planar = position[:2] - start[0][:2]
    return {
        "step": step,
        "body_xyz_m": position.tolist(),
        "body_quat_wxyz": _body_quat(env, A).tolist(),
        "translation_from_start_xyz_m": (position - start[0]).tolist(),
        "planar_translation_heading_deg": _heading_deg(planar),
        "distance_from_start_m": change["distance_m"],
        "rotation_from_start_deg": change["tilt_change_deg"],
        "group0_geometries": per_geom,
        "group0_world_vertices_m": vertices.tolist(),
        "group0_world_aabb": {
            "lower_xyz_m": np.min(vertices, axis=0).tolist(),
            "upper_xyz_m": np.max(vertices, axis=0).tolist(),
        },
    }


def _a_only_trace(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    start = _pose(env, A)
    native_b_start = _pose(env, B)
    _lift_s(env)
    released = not _contact(env, geoms[S], geoms[A])
    frames = []
    contacts = []
    for step in range(DYNAMIC_STEPS + 1):
        if step:
            env.sim.step()
        frames.append(_frame(env, start, step))
        contacts.append({
            "step": step,
            "S_A": _contact(env, geoms[S], geoms[A]),
            "A_table": _contact(env, geoms[A], table),
            "A_B_native_far": _contact(env, geoms[A], geoms[B]),
            "robot_A": _contact(env, robot, geoms[A]),
            "A_other": _contact(env, geoms[A], others),
        })
    for index, frame in enumerate(frames):
        if index == 0:
            step_translation = np.zeros(3)
        else:
            step_translation = (
                np.asarray(frame["body_xyz_m"])
                - np.asarray(frames[index - 1]["body_xyz_m"])
            )
        frame["step_translation_xyz_m"] = step_translation.tolist()
        frame["step_planar_translation_heading_deg"] = _heading_deg(
            step_translation[:2]
        )
    all_vertices = np.concatenate([
        np.asarray(frame["group0_world_vertices_m"]) for frame in frames
    ], axis=0)
    body_path = np.asarray([frame["body_xyz_m"] for frame in frames])
    segments = np.diff(body_path, axis=0)
    planar_segments = segments[:, :2]
    first_motion = next(
        (
            frame["step"] for frame in frames
            if frame["distance_from_start_m"] >= MIN_PATH_M
        ),
        None,
    )
    first_clear = next(
        (
            item["step"] for item in contacts
            if not item["S_A"]
        ),
        None,
    )
    max_frame = max(frames, key=lambda item: item["distance_from_start_m"])
    return {
        "S_A_released_at_intervention": released,
        "B_held_at_native_far_pose": True,
        "B_native_pose_drift_after_trace": _delta(
            native_b_start, _pose(env, B)
        ),
        "first_A_motion_step": first_motion,
        "first_step_without_S_A_contact": first_clear,
        "max_translation_m": max(
            item["distance_from_start_m"] for item in frames
        ),
        "max_rotation_deg": max(
            item["rotation_from_start_deg"] for item in frames
        ),
        "maximum_translation_frame": max_frame,
        "net_translation_xyz_m": (
            body_path[-1] - body_path[0]
        ).tolist(),
        "net_planar_heading_deg": _heading_deg(
            body_path[-1, :2] - body_path[0, :2]
        ),
        "maximum_translation_heading_deg": max_frame[
            "planar_translation_heading_deg"
        ],
        "cumulative_body_path_m": float(
            np.sum(np.linalg.norm(segments, axis=1))
        ),
        "cumulative_planar_path_m": float(
            np.sum(np.linalg.norm(planar_segments, axis=1))
        ),
        "swept_group0_envelope": {
            "lower_xyz_m": np.min(all_vertices, axis=0).tolist(),
            "upper_xyz_m": np.max(all_vertices, axis=0).tolist(),
            "sampled_steps": len(frames),
            "vertices_per_step": len(
                frames[0]["group0_world_vertices_m"]
            ),
        },
        "contacts": contacts,
        "frames": frames,
    }


def _b_relative_vertices(
    env: Any,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _restore(env, state)
    quat = _body_quat(env, B)
    _set_free(env, B, np.asarray([0.0, 0.0, 1.15]), quat)
    relative = _collision_vertices(env, B) - _body_pos(env, B)
    return relative, quat


def _future_tangent_candidate(
    frame: dict[str, Any],
    start_xy: np.ndarray,
    b_relative: np.ndarray,
) -> dict[str, Any] | None:
    body_xy = np.asarray(frame["body_xyz_m"][:2], dtype=float)
    direction = body_xy - start_xy
    norm = float(np.linalg.norm(direction))
    if norm < MIN_PATH_M:
        return None
    direction /= norm
    lateral = np.asarray([-direction[1], direction[0]])
    a_vertices = np.asarray(frame["group0_world_vertices_m"], dtype=float)
    a_far = float(np.max(a_vertices[:, :2] @ direction))
    b_near = float(np.min(b_relative[:, :2] @ direction))
    body_along = a_far + FUTURE_TANGENCY_GAP_M - b_near
    body_lateral = float(body_xy @ lateral)
    candidate_xy = direction * body_along + lateral * body_lateral
    candidate_z = (
        TABLE_TOP_Z_M
        + PLACEMENT_CLEARANCE_M
        - float(np.min(b_relative[:, 2]))
    )
    return {
        "source_trace_step": frame["step"],
        "actual_path_heading_deg": _heading_deg(direction),
        "future_tangency_gap_m": FUTURE_TANGENCY_GAP_M,
        "body_xyz_m": [
            float(candidate_xy[0]),
            float(candidate_xy[1]),
            candidate_z,
        ],
    }


def _static_placement_check(
    env: Any,
    state: np.ndarray,
    candidate: dict[str, Any],
    b_quat: np.ndarray,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    reference = {name: _pose(env, name) for name in (S, A)}
    _set_free(
        env,
        B,
        np.asarray(candidate["body_xyz_m"]),
        b_quat,
    )
    forbidden = {
        "initial_A_B": _contact(env, geoms[A], geoms[B]),
        "initial_S_B": _contact(env, geoms[S], geoms[B]),
        "initial_robot_B": _contact(env, robot, geoms[B]),
        "initial_B_other": _contact(env, geoms[B], others),
    }
    placement_start = {name: _pose(env, name) for name in (S, A, B)}
    for _ in range(PLACEMENT_SETTLE_STEPS):
        env.sim.step()
        forbidden["initial_A_B"] |= _contact(env, geoms[A], geoms[B])
        forbidden["initial_S_B"] |= _contact(env, geoms[S], geoms[B])
        forbidden["initial_robot_B"] |= _contact(env, robot, geoms[B])
        forbidden["initial_B_other"] |= _contact(env, geoms[B], others)
    placement_delta = {
        name: _delta(placement_start[name], _pose(env, name))
        for name in (S, A, B)
    }
    reference_delta = {
        name: _delta(reference[name], _pose(env, name))
        for name in (S, A)
    }
    persistent_b_table = _contact(env, geoms[B], table)
    starts = {name: _pose(env, name) for name in (S, A, B)}
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A, B)
    }
    for _ in range(PLACEMENT_HOLD_STEPS):
        env.sim.step()
        forbidden["initial_A_B"] |= _contact(env, geoms[A], geoms[B])
        forbidden["initial_S_B"] |= _contact(env, geoms[S], geoms[B])
        forbidden["initial_robot_B"] |= _contact(env, robot, geoms[B])
        forbidden["initial_B_other"] |= _contact(env, geoms[B], others)
        persistent_b_table &= _contact(env, geoms[B], table)
        for name in maxima:
            change = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"], change["distance_m"]
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"],
                change["tilt_change_deg"],
            )
    passed = bool(
        persistent_b_table
        and not any(forbidden.values())
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in placement_delta.values()
        )
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in reference_delta.values()
        )
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in maxima.values()
        )
    )
    return {
        "diagnostic_placeable": passed,
        "persistent_B_table": persistent_b_table,
        "forbidden_contacts": forbidden,
        "placement_settle_delta": placement_delta,
        "S_A_from_frozen_reference": reference_delta,
        "max_delta": maxima,
        "settled_B_pose": {
            "body_xyz_m": _body_pos(env, B).tolist(),
            "body_quat_wxyz": _body_quat(env, B).tolist(),
        },
    }


def _first_placeable_point(
    env: Any,
    state: np.ndarray,
    trace: dict[str, Any],
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> dict[str, Any]:
    b_relative, b_quat = _b_relative_vertices(env, state)
    start_xy = np.asarray(trace["frames"][0]["body_xyz_m"][:2], dtype=float)
    attempts = []
    first = None
    contacts_by_step = {
        item["step"]: item for item in trace["contacts"]
    }
    for frame in trace["frames"]:
        if contacts_by_step[frame["step"]]["S_A"]:
            continue
        candidate = _future_tangent_candidate(frame, start_xy, b_relative)
        if candidate is None:
            continue
        check = _static_placement_check(
            env,
            state,
            candidate,
            b_quat,
            geoms,
            table,
            robot,
            others,
        )
        row = {**candidate, "static_check": check}
        attempts.append(row)
        if check["diagnostic_placeable"]:
            first = row
            break
    return {
        "definition": (
            "earliest sampled post-release A pose whose actual group0 leading "
            "surface admits a native upright B tangent 1 mm farther along "
            "the measured body-translation heading, while the pre-lift S/A "
            "state remains contact-free and stable for 120 steps"
        ),
        "first_placeable": first,
        "attempt_count_until_first_or_exhaustion": len(attempts),
        "attempts": attempts,
        "diagnostic_only_not_scene_selection": True,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError("task1 A-only diagnostic native contract mismatch")
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task1 A-only diagnostic BDDL hash mismatch")
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
            raise RuntimeError("task1 A-only diagnostic policy-entry drift")
        geoms = {name: _geoms(env, name) for name in (S, A, B)}
        table = _table_geoms(env)
        robot = _robot_geoms(env)
        others = _other_geoms(env)
        seeds = []
        for lean_deg, support_gap_m in FROZEN_A_SEEDS:
            static, state = _static_a(
                env,
                base,
                lean_deg,
                support_gap_m,
                geoms,
                table,
                robot,
                others,
            )
            if not static["passed"]:
                raise RuntimeError(
                    "frozen robust A seed failed revalidation: "
                    f"{lean_deg}, {support_gap_m}"
                )
            trace = _a_only_trace(
                env, state, geoms, table, robot, others
            )
            placement = _first_placeable_point(
                env, state, trace, geoms, table, robot, others
            )
            seeds.append({
                "seed": {
                    "lean_deg": lean_deg,
                    "support_gap_m": support_gap_m,
                },
                "static_revalidation": static,
                "a_only_trace": trace,
                "first_placeable_point": placement,
            })
        report = {
            "diagnostic_status":
                "COMPLETE_L3A2_TASK1_A_ONLY_SWEPT_PATH_DIAGNOSTIC",
            "scene_verdict": None,
            "candidate_selected": False,
            "task_id_zero_based": TASK_ID,
            "prompt": TASK_PROMPT,
            "task_description_override": None,
            "checkpoint_binding": CHECKPOINT,
            "bddl_sha256": BDDL_SHA256,
            "policy_entry": {
                "base_sha256": BASE_SHA256,
                "source":
                    "official_init0_after_exact_evaluator_10_dummy_actions",
                "raw_state_used_directly": False,
                "future_evaluator_num_steps_wait": 0,
            },
            "roles": {"S": S, "A": A, "B_native_far": B},
            "seed_count": len(seeds),
            "frozen_A_seeds": FROZEN_A_SEEDS,
            "dynamic_steps": DYNAMIC_STEPS,
            "all_group0_vertices_recorded_each_step": True,
            "scene_or_asset_modified": False,
            "custom_asset": False,
            "policy_image_or_video_generated": False,
            "hdf5_generated": False,
            "vla_run": False,
            "formal_family_generated": False,
            "seeds": seeds,
        }
        (OUT / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"diagnostic_status={report['diagnostic_status']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

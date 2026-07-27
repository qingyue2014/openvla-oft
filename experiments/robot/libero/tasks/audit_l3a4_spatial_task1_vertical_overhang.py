#!/usr/bin/env python3
"""Exact-geometry and 80-step hold audit for a bowl-on-bowl overhang."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_l3a4_spatial_task1_bowl2_feasibility import (
    EXPECTED_BDDL_SHA256,
    PROMPT,
    TASK_ID,
)
from audit_l3a4_spatial_task1_native import (
    DUMMY_ACTION,
    EVALUATOR_NUM_STEPS_WAIT,
    descendant_geoms,
    resolve_body,
)
from probe_l3a4_spatial_task1_momentum import (
    _bounds,
    _contact,
    _extent,
    _free_pose,
    _jsonable,
    _policy_image,
    _restore,
    _robot_geoms,
    _set_pose,
    _sha,
)


ROLES = {
    "S": "akita_black_bowl_1",
    "A": "akita_black_bowl_2",
    "B": "cookies_1",
    "landmark": "glazed_rim_porcelain_ramekin_1",
    "goal": "plate_1",
}
DIRECTIONS = {
    "+y": np.asarray([0.0, 1.0]),
    "+x": np.asarray([1.0, 0.0]),
    "-x": np.asarray([-1.0, 0.0]),
    "-y": np.asarray([0.0, -1.0]),
}
A_OFFSETS_M = (0.012, 0.020, 0.028)
B_CLEARANCES_M = (0.002, 0.006, 0.010)
VERTICAL_OVERLAP_M = -0.002
TABLE_BODY = "table"


def _pose(env, body: str) -> tuple[np.ndarray, float]:
    body_id = int(env.sim.model.body_name2id(body))
    pos = np.asarray(env.sim.data.body_xpos[body_id]).copy()
    mat = np.asarray(env.sim.data.body_xmat[body_id]).reshape(3, 3)
    tilt = float(
        np.degrees(np.arccos(np.clip(mat[2, 2], -1.0, 1.0)))
    )
    return pos, tilt


def _contact_force(env, left: set[int], right: set[int]) -> float:
    maximum = 0.0
    for index in range(int(env.sim.data.ncon)):
        item = env.sim.data.contact[index]
        if not (
            (
                int(item.geom1) in left
                and int(item.geom2) in right
            )
            or (
                int(item.geom2) in left
                and int(item.geom1) in right
            )
        ):
            continue
        address = int(item.efc_address)
        if address >= 0:
            maximum = max(
                maximum, abs(float(env.sim.data.efc_force[address]))
            )
    return maximum


def _segmentation_geom_ids(env) -> np.ndarray:
    segmentation = env.sim.render(
        width=256,
        height=256,
        camera_name="agentview",
        segmentation=True,
    )
    if segmentation is None:
        raise RuntimeError("segmentation render returned None")
    segmentation = np.asarray(segmentation)
    return segmentation[..., -1] if segmentation.ndim == 3 else segmentation


def _visible_pixels(env, body: str) -> int:
    geom_ids = descendant_geoms(env.sim, body)
    return int(np.isin(_segmentation_geom_ids(env), tuple(geom_ids)).sum())


def _candidate_state(
    env,
    base: np.ndarray,
    bodies: dict[str, str],
    direction: np.ndarray,
    offset_m: float,
    b_clearance_m: float,
) -> tuple[np.ndarray, dict]:
    _restore(env, base)
    s_pose = _free_pose(env, bodies["S"])
    table_z = float(_bounds(env, bodies["S"])[0][2])
    s_hi_z = float(_bounds(env, bodies["S"])[1][2])

    a_pose = _free_pose(env, bodies["A"])
    a_pose[:2] = s_pose[:2] + direction * offset_m
    _set_pose(env, bodies["A"], a_pose, zero_velocity=False)
    env.sim.forward()
    a_pose[2] += (
        s_hi_z + VERTICAL_OVERLAP_M
        - float(_bounds(env, bodies["A"])[0][2])
    )
    _set_pose(env, bodies["A"], a_pose, zero_velocity=False)
    env.sim.forward()

    # B is beyond the overhanging edge on the expected A roll-off side.
    forward_extent = max(
        _extent(env, bodies["S"], direction),
        offset_m + _extent(env, bodies["A"], direction),
    )
    b_pose = _free_pose(env, bodies["B"])
    b_pose[:2] = s_pose[:2] + direction * (
        forward_extent
        + _extent(env, bodies["B"], -direction)
        + b_clearance_m
    )
    _set_pose(env, bodies["B"], b_pose, zero_velocity=False)
    env.sim.forward()
    b_pose[2] += (
        table_z - 0.001 - float(_bounds(env, bodies["B"])[0][2])
    )
    _set_pose(env, bodies["B"], b_pose, zero_velocity=False)
    env.sim.forward()

    state = np.asarray(env.sim.get_state().flatten()).copy()
    allowed = set()
    for role in ("A", "B"):
        body_id = int(env.sim.model.body_name2id(bodies[role]))
        for joint_id in range(env.sim.model.njnt):
            if (
                int(env.sim.model.jnt_bodyid[joint_id]) == body_id
                and int(env.sim.model.jnt_type[joint_id]) == 0
            ):
                qadr = int(env.sim.model.jnt_qposadr[joint_id])
                allowed.update(range(1 + qadr, 1 + qadr + 7))
    changed = set(
        np.flatnonzero(~np.isclose(state, base, rtol=0, atol=1e-12))
    )
    if not changed or not changed <= allowed:
        raise RuntimeError("candidate changed outside A/B qpos")
    return state, {
        "S_A_initial_contact": None,
        "table_z_m": table_z,
        "S_top_z_m": s_hi_z,
    }


def _hold(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    table_geoms: set[int],
    robot: set[int],
    steps: int,
) -> dict:
    _restore(env, state)
    starts = {
        role: _pose(env, bodies[role]) for role in ("S", "A", "B")
    }
    counts = {"S_A": 0, "B_table": 0}
    forbidden = {
        "A_table": False,
        "A_B": False,
        "S_B": False,
        "robot_A": False,
        "robot_B": False,
    }
    max_force = 0.0
    max_drift = {role: 0.0 for role in ("S", "A", "B")}
    max_tilt = {role: 0.0 for role in ("S", "A", "B")}
    for step in range(steps + 1):
        counts["S_A"] += int(_contact(env, geoms["S"], geoms["A"]))
        counts["B_table"] += int(
            _contact(env, geoms["B"], table_geoms)
        )
        max_force = max(
            max_force, _contact_force(env, geoms["S"], geoms["A"])
        )
        forbidden["A_table"] |= _contact(
            env, geoms["A"], table_geoms
        )
        forbidden["A_B"] |= _contact(env, geoms["A"], geoms["B"])
        forbidden["S_B"] |= _contact(env, geoms["S"], geoms["B"])
        forbidden["robot_A"] |= _contact(env, robot, geoms["A"])
        forbidden["robot_B"] |= _contact(env, robot, geoms["B"])
        for role in ("S", "A", "B"):
            pos, tilt = _pose(env, bodies[role])
            max_drift[role] = max(
                max_drift[role],
                float(np.linalg.norm(pos - starts[role][0])),
            )
            max_tilt[role] = max(
                max_tilt[role], abs(tilt - starts[role][1])
            )
        if step < steps:
            env.sim.step()
    occupancy = {
        key: count / float(steps + 1) for key, count in counts.items()
    }
    passed = bool(
        occupancy["S_A"] >= 0.95
        and occupancy["B_table"] >= 0.95
        and max_force > 1e-6
        and not any(forbidden.values())
        and max(max_drift.values()) <= 0.002
        and max(max_tilt.values()) <= 3.0
    )
    return {
        "passed": passed,
        "contact_occupancy": occupancy,
        "max_S_A_normal_force_N": max_force,
        "forbidden_seen": forbidden,
        "max_drift_m": max_drift,
        "max_tilt_change_deg": max_tilt,
    }


def _grasp_corridor(
    env,
    bodies: dict[str, str],
    direction: np.ndarray,
    offset_m: float,
) -> dict:
    s_center = _pose(env, bodies["S"])[0]
    s_exposed_extent = _extent(env, bodies["S"], -direction)
    grasp_center = s_center[:2] - direction * (s_exposed_extent - 0.008)
    a_center = _pose(env, bodies["A"])[0]
    a_near_projection = (
        float(a_center[:2] @ direction)
        - _extent(env, bodies["A"], -direction)
    )
    grasp_projection = float(grasp_center @ direction)
    center_to_a_margin = a_near_projection - grasp_projection
    corridor_radius = 0.015
    free_margin = center_to_a_margin - corridor_radius
    return {
        "exposed_offset_m": offset_m,
        "grasp_center_xy": grasp_center,
        "corridor_diameter_m": 2 * corridor_radius,
        "center_to_A_near_surface_m": center_to_a_margin,
        "free_margin_m": free_margin,
        "passed": bool(offset_m >= 0.020 and free_margin >= 0.005),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_spatial_task1_vertical_overhang",
    )
    parser.add_argument("--hold_steps", type=int, default=80)
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID))
    if task.language != PROMPT:
        raise RuntimeError("native prompt drift")
    if hashlib.sha256(bddl.read_bytes()).hexdigest() != EXPECTED_BDDL_SHA256:
        raise RuntimeError("native BDDL drift")
    raw = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    try:
        env.seed(42)
        env.reset()
        env.set_init_state(raw)
        for _ in range(EVALUATOR_NUM_STEPS_WAIT):
            env.step(DUMMY_ACTION)
        base = np.asarray(env.sim.get_state().flatten()).copy()
        bodies = {
            role: resolve_body(env.sim, stem) for role, stem in ROLES.items()
        }
        geoms = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        table_geoms = descendant_geoms(env.sim, TABLE_BODY)
        robot = _robot_geoms(env)
        baseline_pixels = _visible_pixels(env, bodies["S"])
        rows, eligible = [], []
        for direction_name, direction in DIRECTIONS.items():
            for offset_m in A_OFFSETS_M:
                for b_clearance_m in B_CLEARANCES_M:
                    state, placement = _candidate_state(
                        env, base, bodies, direction, offset_m, b_clearance_m
                    )
                    hold = _hold(
                        env, state, bodies, geoms, table_geoms, robot,
                        args.hold_steps,
                    )
                    row = {
                        "direction": direction_name,
                        "A_offset_m": offset_m,
                        "B_clearance_m": b_clearance_m,
                        "vertical_overlap_m": VERTICAL_OVERLAP_M,
                        "state_sha256": _sha(state),
                        "placement": placement,
                        "hold": hold,
                        "passed": False,
                    }
                    rows.append(row)
                    if not hold["passed"]:
                        continue
                    _restore(env, state)
                    pixels = _visible_pixels(env, bodies["S"])
                    corridor = _grasp_corridor(
                        env, bodies, direction, offset_m
                    )
                    visibility = {
                        "S_visible_pixels": pixels,
                        "native_baseline_pixels": baseline_pixels,
                        "retained_fraction": (
                            pixels / baseline_pixels
                            if baseline_pixels else 0.0
                        ),
                        "passed": bool(
                            pixels >= 40
                            and baseline_pixels > 0
                            and pixels / baseline_pixels >= 0.30
                        ),
                    }
                    row["visibility"] = visibility
                    row["grasp_corridor"] = corridor
                    row["passed"] = bool(
                        visibility["passed"] and corridor["passed"]
                    )
                    if row["passed"]:
                        eligible.append((row, state))

        selected, artifacts = None, {}
        if eligible:
            row, state = eligible[0]
            selected = {
                key: row[key]
                for key in (
                    "direction", "A_offset_m", "B_clearance_m",
                    "state_sha256", "visibility", "grasp_corridor",
                )
            }
            _restore(env, state)
            png = out / "selected_static_policy_agentview.png"
            imageio.imwrite(png, _policy_image(env))
            segmentation = _segmentation_geom_ids(env)
            mask = np.isin(
                segmentation, tuple(geoms["S"])
            ).astype(np.uint8) * 255
            mask_png = out / "selected_static_S_segmentation.png"
            imageio.imwrite(mask_png, mask)
            npz = out / "selected_static_wait0_state.npz"
            np.savez_compressed(
                npz, policy_entry_base_state=base, candidate_state=state
            )
            artifacts = {
                "policy_png": str(png),
                "S_segmentation_png": str(mask_png),
                "state_npz": str(npz),
            }

        report = {
            "verdict": (
                "PASS_L3A4_VERTICAL_OVERHANG_STATIC_HOLD"
                if eligible else
                "FAIL_L3A4_VERTICAL_OVERHANG_STATIC_HOLD"
            ),
            "task_suite": "libero_spatial",
            "task_id": TASK_ID,
            "prompt": PROMPT,
            "bddl_sha256": EXPECTED_BDDL_SHA256,
            "policy_entry_base_sha256": _sha(base),
            "future_evaluator_num_steps_wait": 0,
            "grid": {
                "directions": list(DIRECTIONS),
                "A_offsets_m": A_OFFSETS_M,
                "B_clearances_m": B_CLEARANCES_M,
                "vertical_overlap_m": VERTICAL_OVERLAP_M,
                "candidate_count": len(rows),
            },
            "native_S_visible_pixels": baseline_pixels,
            "hold_pass_count": sum(row["hold"]["passed"] for row in rows),
            "visibility_grasp_pass_count": len(eligible),
            "selected": selected,
            "rows": rows,
            "artifacts": artifacts,
            "dynamic_run": False,
            "vla_run": False,
            "custom_assets": False,
        }
        (out / "audit.json").write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n"
        )
        print(report["verdict"])
    finally:
        env.close()


if __name__ == "__main__":
    main()

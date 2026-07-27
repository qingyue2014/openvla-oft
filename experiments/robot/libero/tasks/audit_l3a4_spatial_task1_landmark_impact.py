#!/usr/bin/env python3
"""Static exact-geometry audit for cookies -> ramekin landmark impact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_l3a4_spatial_task1_native import (
    DUMMY_ACTION,
    EVALUATOR_NUM_STEPS_WAIT,
    PROMPT,
    TASK_ID,
    descendant_geoms,
    resolve_body,
)
from probe_l3a4_spatial_task1_momentum import (
    EXPECTED_BDDL_SHA256,
    _bounds,
    _contact,
    _extent,
    _free_pose,
    _geom_vertices,
    _jsonable,
    _quat_axis_angle,
    _restore,
    _robot_geoms,
    _set_pose,
    _sha,
)


ROLES = {
    "S": "akita_black_bowl_1",
    "A": "cookies_1",
    "B": "glazed_rim_porcelain_ramekin_1",
    "other_bowl": "akita_black_bowl_2",
    "goal": "plate_1",
}
DIRECTIONS = {
    "+y": np.asarray([0.0, 1.0]),
    "+x": np.asarray([1.0, 0.0]),
    "-x": np.asarray([-1.0, 0.0]),
    "-y": np.asarray([0.0, -1.0]),
    "-x-y": np.asarray([-1.0, -1.0]) / np.sqrt(2.0),
}
# Coupled lean/overlap pairs keep the frozen audit at 5*3*2 = 30 points.
A_LEAN_OVERLAP = (
    (65.0, -0.003),
    (72.0, -0.004),
    (78.0, -0.005),
)
B_CLEARANCES_M = (0.002, 0.006)
TABLE_BODY = "table"


def _interval(env, body: str, axis_xy: np.ndarray) -> tuple[float, float]:
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0])
    values = np.concatenate(
        [
            _geom_vertices(env, geom) @ axis
            for geom in descendant_geoms(env.sim, body)
            if int(env.sim.model.geom_group[geom]) == 0
        ]
    )
    return float(values.min()), float(values.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_spatial_task1_landmark_impact",
    )
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
        if env.sim.model.body_name2id(TABLE_BODY) < 0:
            raise RuntimeError("compiled runtime table body missing")
        table_geoms = descendant_geoms(env.sim, TABLE_BODY)
        geoms = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        robot = _robot_geoms(env)
        rows = []
        for direction_name, direction in DIRECTIONS.items():
            for lean, overlap in A_LEAN_OVERLAP:
                for b_clearance in B_CLEARANCES_M:
                    _restore(env, base)
                    s_pose = _free_pose(env, bodies["S"])
                    table_z = float(_bounds(env, bodies["S"])[0][2])
                    a_pose = _free_pose(env, bodies["A"])
                    # Local cookie-box plane becomes near vertical and leans
                    # through the volume vacated when S is lifted.
                    a_pose[3:7] = _quat_axis_angle(
                        np.asarray(
                            [direction[1], -direction[0], 0.0]
                        ),
                        lean,
                    )
                    _set_pose(env, bodies["A"], a_pose)
                    env.sim.forward()
                    a_pose[:2] = s_pose[:2] + direction * (
                        _extent(env, bodies["S"], direction)
                        + _extent(env, bodies["A"], -direction)
                        + overlap
                    )
                    _set_pose(env, bodies["A"], a_pose)
                    env.sim.forward()
                    a_pose[2] += (
                        table_z - 0.001
                        - float(_bounds(env, bodies["A"])[0][2])
                    )
                    _set_pose(env, bodies["A"], a_pose)
                    env.sim.forward()

                    b_pose = _free_pose(env, bodies["B"])
                    b_pose[:2] = s_pose[:2] - direction * (
                        _extent(env, bodies["S"], -direction)
                        + _extent(env, bodies["B"], direction)
                        + b_clearance
                    )
                    _set_pose(env, bodies["B"], b_pose)
                    env.sim.forward()
                    b_pose[2] += (
                        table_z - 0.001
                        - float(_bounds(env, bodies["B"])[0][2])
                    )
                    _set_pose(env, bodies["B"], b_pose)
                    env.sim.forward()

                    si = _interval(env, bodies["S"], direction)
                    ai = _interval(env, bodies["A"], direction)
                    bi = _interval(env, bodies["B"], direction)
                    contacts = {
                        "S_A": _contact(env, geoms["S"], geoms["A"]),
                        "A_table": _contact(
                            env, geoms["A"], table_geoms
                        ),
                        "B_table": _contact(
                            env, geoms["B"], table_geoms
                        ),
                        "A_B": _contact(env, geoms["A"], geoms["B"]),
                        "S_B": _contact(env, geoms["S"], geoms["B"]),
                        "robot_A": _contact(env, robot, geoms["A"]),
                        "robot_B": _contact(env, robot, geoms["B"]),
                    }
                    rows.append(
                        {
                            "direction": direction_name,
                            "A_lean_deg": lean,
                            "SA_overlap_m": overlap,
                            "B_clearance_m": b_clearance,
                            "signed_SA_directional_clearance_m": (
                                ai[0] - si[1]
                            ),
                            "signed_SB_directional_clearance_m": (
                                si[0] - bi[1]
                            ),
                            "contacts": contacts,
                            "feasible": bool(
                                contacts["S_A"]
                                and contacts["A_table"]
                                and contacts["B_table"]
                                and not any(
                                    contacts[key]
                                    for key in (
                                        "A_B", "S_B",
                                        "robot_A", "robot_B",
                                    )
                                )
                            ),
                        }
                    )
        feasible = [row for row in rows if row["feasible"]]
        report = {
            "verdict": (
                "PASS_L3A4_LANDMARK_IMPACT_STATIC_FEASIBILITY"
                if feasible else
                "FAIL_L3A4_LANDMARK_IMPACT_STATIC_FEASIBILITY"
            ),
            "scope": "read_only_exact_geometry_no_candidate_sim_step",
            "task_suite": "libero_spatial",
            "task_id": TASK_ID,
            "prompt": PROMPT,
            "bddl_sha256": EXPECTED_BDDL_SHA256,
            "policy_entry_base_sha256": _sha(base),
            "future_evaluator_num_steps_wait": 0,
            "roles": bodies,
            "grid": {
                "directions": list(DIRECTIONS),
                "A_lean_overlap": A_LEAN_OVERLAP,
                "B_clearances_m": B_CLEARANCES_M,
                "candidate_count": len(rows),
            },
            "feasible_count": len(feasible),
            "rows": rows,
            "custom_assets": False,
            "candidate_raw_mujoco_steps": 0,
            "vla_run": False,
        }
        (out / "audit.json").write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n"
        )
        print(report["verdict"])
    finally:
        env.close()


if __name__ == "__main__":
    main()

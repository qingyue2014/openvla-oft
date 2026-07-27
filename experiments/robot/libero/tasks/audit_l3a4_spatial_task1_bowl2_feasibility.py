#!/usr/bin/env python3
"""Read-only exact-compiled-geometry audit for the bowl2 relay candidate.

No raw MuJoCo step, policy model, rollout, or force is executed.  Hypothetical
A/B free poses are forwarded only to inspect compiled collision geometry and
initial contact / bypass feasibility.
"""

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
TILTS_DEG = (20.0, 30.0, 40.0)
SA_OVERLAPS_M = (-0.002, -0.004, -0.006)
B_GAPS_M = (0.002, 0.006)


def _projection_interval(
    env, body: str, axis_xy: np.ndarray
) -> tuple[float, float]:
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
        default="experiments/logs/l3a4_spatial_task1_bowl2_feasibility",
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
        geoms = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        robot = _robot_geoms(env)
        native = {}
        for role, body in bodies.items():
            lo, hi = _bounds(env, body)
            native[role] = {
                "position": _free_pose(env, body)[:3]
                if role != "goal" else
                np.asarray(
                    env.sim.data.body_xpos[
                        env.sim.model.body_name2id(body)
                    ]
                ),
                "collision_AABB_lo": lo,
                "collision_AABB_hi": hi,
                "collision_AABB_size_m": hi - lo,
                "group0_count": sum(
                    int(env.sim.model.geom_group[g]) == 0
                    for g in geoms[role]
                ),
                "group1_count": sum(
                    int(env.sim.model.geom_group[g]) == 1
                    for g in geoms[role]
                ),
            }

        rows = []
        for name, direction in DIRECTIONS.items():
            for tilt in TILTS_DEG:
                for overlap in SA_OVERLAPS_M:
                    for b_gap in B_GAPS_M:
                        _restore(env, base)
                        s_pose = _free_pose(env, bodies["S"])
                        table_z = float(_bounds(env, bodies["S"])[0][2])
                        a_pose = _free_pose(env, bodies["A"])
                        a_pose[3:7] = _quat_axis_angle(
                            np.asarray(
                                [direction[1], -direction[0], 0.0]
                            ),
                            tilt,
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
                            table_z - float(_bounds(env, bodies["A"])[0][2])
                        )
                        _set_pose(env, bodies["A"], a_pose)
                        env.sim.forward()

                        b_pose = _free_pose(env, bodies["B"])
                        b_pose[:2] = s_pose[:2] - direction * (
                            _extent(env, bodies["S"], -direction)
                            + _extent(env, bodies["B"], direction)
                            + b_gap
                        )
                        _set_pose(env, bodies["B"], b_pose)
                        env.sim.forward()
                        b_pose[2] += (
                            table_z - float(_bounds(env, bodies["B"])[0][2])
                        )
                        _set_pose(env, bodies["B"], b_pose)
                        env.sim.forward()

                        s_interval = _projection_interval(
                            env, bodies["S"], direction
                        )
                        a_interval = _projection_interval(
                            env, bodies["A"], direction
                        )
                        b_interval = _projection_interval(
                            env, bodies["B"], direction
                        )
                        rows.append(
                            {
                                "direction": name,
                                "tilt_deg": tilt,
                                "SA_overlap_m": overlap,
                                "B_gap_m": b_gap,
                                "signed_SA_directional_clearance_m": (
                                    a_interval[0] - s_interval[1]
                                ),
                                "signed_SB_directional_clearance_m": (
                                    s_interval[0] - b_interval[1]
                                ),
                                "contacts": {
                                    "S_A": _contact(
                                        env, geoms["S"], geoms["A"]
                                    ),
                                    "A_B": _contact(
                                        env, geoms["A"], geoms["B"]
                                    ),
                                    "S_B": _contact(
                                        env, geoms["S"], geoms["B"]
                                    ),
                                    "robot_A": _contact(
                                        env, robot, geoms["A"]
                                    ),
                                    "robot_B": _contact(
                                        env, robot, geoms["B"]
                                    ),
                                },
                            }
                        )

        feasible = [
            row for row in rows
            if row["contacts"]["S_A"]
            and not any(
                row["contacts"][key]
                for key in ("A_B", "S_B", "robot_A", "robot_B")
            )
        ]
        report = {
            "verdict": (
                "PASS_L3A4_SPATIAL_TASK1_BOWL2_STATIC_FEASIBILITY"
                if feasible else
                "FAIL_L3A4_SPATIAL_TASK1_BOWL2_STATIC_FEASIBILITY"
            ),
            "scope": "read_only_exact_compiled_geometry_no_raw_sim_step",
            "task_suite": "libero_spatial",
            "task_id": TASK_ID,
            "prompt": PROMPT,
            "bddl_sha256": EXPECTED_BDDL_SHA256,
            "policy_entry_base_sha256": _sha(base),
            "future_evaluator_num_steps_wait": 0,
            "roles": bodies,
            "native_geometry": native,
            "proposed_bounded_scan": {
                "directions": list(DIRECTIONS),
                "tilts_deg": TILTS_DEG,
                "SA_overlaps_m": SA_OVERLAPS_M,
                "B_gaps_m": B_GAPS_M,
                "candidate_count": len(rows),
            },
            "static_feasible_count": len(feasible),
            "rows": rows,
            "custom_assets": False,
            "raw_mujoco_steps_after_candidate_pose": 0,
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

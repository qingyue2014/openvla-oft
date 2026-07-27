#!/usr/bin/env python3
"""Static-only cookie cantilever -> goal-plate scene feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_l3a4_spatial_task1_native import (
    DUMMY_ACTION,
    EVALUATOR_NUM_STEPS_WAIT,
    PROMPT,
    TASK_ID,
    balanced_form,
    descendant_geoms,
    resolve_body,
)
from audit_l3a4_spatial_task1_vertical_overhang import (
    _contact_force,
    _pose,
    _segmentation_geom_ids,
    _visible_pixels,
)
from probe_l3a4_spatial_task1_momentum import (
    EXPECTED_BDDL_SHA256,
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


EXPECTED_BASE_SHA256 = (
    "06a341f78cf0399ee253967e645d88d0538bd5a27c83de3f3d487a92fbfbeee6"
)
EXPECTED_GOAL = "(:goal (And (On akita_black_bowl_1 plate_1)) )"
ROLES = {
    "S": "akita_black_bowl_1",
    "A": "cookies_1",
    "B": "plate_1",
    "landmark": "glazed_rim_porcelain_ramekin_1",
    "other_bowl": "akita_black_bowl_2",
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


def _free_joint_addresses(env, body: str) -> tuple[int, int]:
    body_id = int(env.sim.model.body_name2id(body))
    for joint_id in range(env.sim.model.njnt):
        if (
            int(env.sim.model.jnt_bodyid[joint_id]) == body_id
            and int(env.sim.model.jnt_type[joint_id]) == 0
        ):
            return (
                int(env.sim.model.jnt_qposadr[joint_id]),
                int(env.sim.model.jnt_dofadr[joint_id]),
            )
    raise RuntimeError(f"no free joint for {body}")


def _place_and_scratch_settle(
    env,
    base: np.ndarray,
    bodies: dict[str, str],
    direction: np.ndarray,
    offset_m: float,
    b_clearance_m: float,
    scratch_steps: int,
) -> np.ndarray:
    _restore(env, base)
    s_pose = _free_pose(env, bodies["S"])
    table_z = float(_bounds(env, bodies["S"])[0][2])
    s_top = float(_bounds(env, bodies["S"])[1][2])
    a_pose = _free_pose(env, bodies["A"])
    a_pose[:2] = s_pose[:2] + direction * offset_m
    _set_pose(env, bodies["A"], a_pose)
    env.sim.forward()
    a_pose[2] += (
        s_top + VERTICAL_OVERLAP_M
        - float(_bounds(env, bodies["A"])[0][2])
    )
    _set_pose(env, bodies["A"], a_pose)
    env.sim.forward()

    # B's near edge is just outside S, underneath A's potential free-end
    # fall path. Vertical separation keeps A/B initially disjoint.
    b_pose = _free_pose(env, bodies["B"])
    b_pose[:2] = s_pose[:2] + direction * (
        _extent(env, bodies["S"], direction)
        + _extent(env, bodies["B"], -direction)
        + b_clearance_m
    )
    _set_pose(env, bodies["B"], b_pose)
    env.sim.forward()
    b_pose[2] += (
        table_z - 0.001 - float(_bounds(env, bodies["B"])[0][2])
    )
    _set_pose(env, bodies["B"], b_pose)
    env.sim.forward()

    # Scratch settling derives only A/B qpos. S is held at its exact
    # policy-entry pose, preventing load from contaminating the common base.
    for _ in range(scratch_steps):
        env.sim.step()
        _set_pose(env, bodies["S"], s_pose)
        env.sim.forward()
    settled = {
        role: _free_pose(env, bodies[role]) for role in ("A", "B")
    }

    _restore(env, base)
    for role in ("A", "B"):
        _set_pose(
            env, bodies[role], settled[role], zero_velocity=False
        )
    env.sim.forward()
    state = np.asarray(env.sim.get_state().flatten()).copy()
    allowed = set()
    for role in ("A", "B"):
        qadr, _ = _free_joint_addresses(env, bodies[role])
        allowed.update(range(1 + qadr, 1 + qadr + 7))
    changed = set(
        np.flatnonzero(~np.isclose(state, base, rtol=0, atol=1e-12))
    )
    if not changed or not changed <= allowed:
        raise RuntimeError("candidate changed outside A/B free qpos")
    return state


def _stability_gate(
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
        "A_landmark": False,
        "A_other_bowl": False,
        "B_landmark": False,
        "B_other_bowl": False,
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
        for role, other in (
            ("A_landmark", "landmark"),
            ("A_other_bowl", "other_bowl"),
        ):
            forbidden[role] |= _contact(
                env, geoms["A"], geoms[other]
            )
        for role, other in (
            ("B_landmark", "landmark"),
            ("B_other_bowl", "other_bowl"),
        ):
            forbidden[role] |= _contact(
                env, geoms["B"], geoms[other]
            )
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
        key: value / float(steps + 1) for key, value in counts.items()
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
        "steps": steps,
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
    exposed_extent = _extent(env, bodies["S"], -direction)
    grasp_center = s_center[:2] - direction * (exposed_extent - 0.008)
    a_center = _pose(env, bodies["A"])[0]
    a_near = (
        float(a_center[:2] @ direction)
        - _extent(env, bodies["A"], -direction)
    )
    separation = a_near - float(grasp_center @ direction)
    radius = 0.015
    free_margin = separation - radius
    return {
        "grasp_center_xy": grasp_center,
        "corridor_diameter_m": 2 * radius,
        "A_offset_m": offset_m,
        "center_to_A_near_surface_m": separation,
        "free_margin_m": free_margin,
        "passed": bool(offset_m >= 0.020 and free_margin >= 0.005),
    }


def _asset_gate(env, bodies: dict[str, str]) -> dict:
    result = {}
    for role in ("S", "A", "B"):
        geoms = descendant_geoms(env.sim, bodies[role])
        collision = [
            geom for geom in geoms
            if int(env.sim.model.geom_group[geom]) == 0
            and int(env.sim.model.geom_contype[geom])
            and int(env.sim.model.geom_conaffinity[geom])
        ]
        visible = [
            geom for geom in geoms
            if int(env.sim.model.geom_group[geom]) == 1
        ]
        result[role] = {
            "collision_group0_count": len(collision),
            "visible_group1_count": len(visible),
            "passed": bool(collision and visible),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_spatial_task1_cookie_goal_plate",
    )
    parser.add_argument("--scratch_settle_steps", type=int, default=40)
    parser.add_argument("--settle_gate_steps", type=int, default=240)
    parser.add_argument("--hold_steps", type=int, default=80)
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID))
    bddl_bytes = bddl.read_bytes()
    if task.language != PROMPT:
        raise RuntimeError("native prompt drift")
    if hashlib.sha256(bddl_bytes).hexdigest() != EXPECTED_BDDL_SHA256:
        raise RuntimeError("native BDDL drift")
    goal = balanced_form(bddl_bytes.decode(), "goal")
    if goal != EXPECTED_GOAL:
        raise RuntimeError(f"native goal semantics drift: {goal}")
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
        if _sha(base) != EXPECTED_BASE_SHA256:
            raise RuntimeError("policy-entry base hash drift")
        bodies = {
            role: resolve_body(env.sim, stem) for role, stem in ROLES.items()
        }
        geoms = {
            role: descendant_geoms(env.sim, body)
            for role, body in bodies.items()
        }
        table_geoms = descendant_geoms(env.sim, TABLE_BODY)
        robot = _robot_geoms(env)
        assets = _asset_gate(env, bodies)
        if not all(item["passed"] for item in assets.values()):
            raise RuntimeError("native asset collision/visibility gate failed")
        baseline_pixels = {
            role: _visible_pixels(env, bodies[role])
            for role in ("S", "A", "B")
        }

        rows, eligible = [], []
        for direction_name, direction in DIRECTIONS.items():
            for offset_m in A_OFFSETS_M:
                for b_clearance_m in B_CLEARANCES_M:
                    state = _place_and_scratch_settle(
                        env, base, bodies, direction, offset_m,
                        b_clearance_m, args.scratch_settle_steps,
                    )
                    settle = _stability_gate(
                        env, state, bodies, geoms, table_geoms, robot,
                        args.settle_gate_steps,
                    )
                    hold = _stability_gate(
                        env, state, bodies, geoms, table_geoms, robot,
                        args.hold_steps,
                    )
                    row = {
                        "direction": direction_name,
                        "A_offset_m": offset_m,
                        "B_clearance_m": b_clearance_m,
                        "vertical_overlap_m": VERTICAL_OVERLAP_M,
                        "state_sha256": _sha(state),
                        "settle_gate": settle,
                        "hold_gate": hold,
                        "passed": False,
                    }
                    rows.append(row)
                    if not settle["passed"] or not hold["passed"]:
                        continue
                    _restore(env, state)
                    pixels = {
                        role: _visible_pixels(env, bodies[role])
                        for role in ("S", "A", "B")
                    }
                    visibility = {
                        "pixels": pixels,
                        "baseline_pixels": baseline_pixels,
                        "retained_fraction": {
                            role: pixels[role] / baseline_pixels[role]
                            if baseline_pixels[role] else 0.0
                            for role in ("S", "A", "B")
                        },
                    }
                    visibility["passed"] = bool(
                        pixels["S"] >= 40
                        and visibility["retained_fraction"]["S"] >= 0.30
                        and pixels["A"] >= 30
                        and pixels["B"] >= 30
                    )
                    corridor = _grasp_corridor(
                        env, bodies, direction, offset_m
                    )
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
            rgb = out / "selected_static_policy_agentview.png"
            imageio.imwrite(rgb, _policy_image(env))
            segmentation = _segmentation_geom_ids(env)
            masks = {}
            for role in ("S", "A", "B"):
                mask = (
                    np.isin(segmentation, tuple(geoms[role])).astype(np.uint8)
                    * 255
                )
                mask_path = out / f"selected_static_{role}_segmentation.png"
                imageio.imwrite(mask_path, mask)
                masks[role] = str(mask_path)
            state_npz = out / "selected_static_wait0_state.npz"
            np.savez_compressed(
                state_npz,
                policy_entry_base_state=base,
                candidate_state=state,
            )
            artifacts = {
                "policy_rgb": str(rgb),
                "segmentation_masks": masks,
                "state_npz": str(state_npz),
            }

        report = {
            "verdict": (
                "PASS_L3A4_COOKIE_GOAL_PLATE_STATIC"
                if eligible else
                "FAIL_L3A4_COOKIE_GOAL_PLATE_STATIC"
            ),
            "task_suite": "libero_spatial",
            "task_id": TASK_ID,
            "prompt": PROMPT,
            "prompt_override": False,
            "bddl_sha256": EXPECTED_BDDL_SHA256,
            "goal_form": goal,
            "goal_semantics_preserved": True,
            "policy_entry": {
                "base_state_sha256": _sha(base),
                "native_wait_applied_once": EVALUATOR_NUM_STEPS_WAIT,
                "future_evaluator_num_steps_wait": 0,
            },
            "roles": bodies,
            "assets": assets,
            "grid": {
                "directions": list(DIRECTIONS),
                "A_offsets_m": A_OFFSETS_M,
                "B_clearances_m": B_CLEARANCES_M,
                "vertical_overlap_m": VERTICAL_OVERLAP_M,
                "candidate_count": len(rows),
                "scratch_settle_steps": args.scratch_settle_steps,
                "settle_gate_steps": args.settle_gate_steps,
                "hold_steps": args.hold_steps,
            },
            "native_policy_pixels": baseline_pixels,
            "settle_pass_count": sum(
                row["settle_gate"]["passed"] for row in rows
            ),
            "hold_pass_count": sum(
                row["hold_gate"]["passed"] for row in rows
            ),
            "visibility_grasp_pass_count": len(eligible),
            "selected": selected,
            "rows": rows,
            "artifacts": artifacts,
            "manual_policy_rgb_review": (
                "PENDING" if eligible else "NOT_REVIEWABLE_PHYSICS_FAILED"
            ),
            "release_run": False,
            "dynamic_run": False,
            "vla_run": False,
            "hdf5_written": False,
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

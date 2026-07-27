"""Bounded native can-on-can alignment diagnostic for task59 L3-A3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.robot.libero.tasks.generate_l3a3_stack_tray_candidate import (
    apply_templates_to_base,
    descendants_geoms,
    free_template,
    settle,
)
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    bodies_in_contact,
    body_pose,
    pose_delta,
    set_free_pose,
    zero_body_velocity,
)
from experiments.robot.libero.tasks.probe_l3a3_task57_native import balanced_form


TASK_ID = 59
PROMPT = "pick up the tomato sauce and put it in the tray"
NATIVE_BDDL_SHA256 = "7580a3282b33142c441a3a4f906e7f88415a9a734b3ef14e59e22f3a8d7d3315"
GOAL_SHA256 = "a3cb4109ca75f8e64024e9cf63066478505f44b9b95946a4d98fb95c59bb00b9"
SUPPORT = "tomato_sauce_1_main"
MIDDLE = "alphabet_soup_1_main"
FORBIDDEN = (
    "butter_1_main",
    "cream_cheese_1_main",
    "ketchup_1_main",
    "wooden_tray_1_main",
)
GRID_M = (-0.004, -0.002, 0.0, 0.002, 0.004)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def geom_world_z_bounds(sim, geom_id: int) -> tuple[float, float]:
    """Exact compiled-geometry z AABB; never substitutes geom rbound."""
    model, data = sim.model, sim.data
    geom_type = int(model.geom_type[geom_id])
    center = np.asarray(data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == 2:  # sphere
        z_extent = float(size[0])
    elif geom_type == 3:  # capsule
        z_extent = float(
            abs(rotation[2, 2]) * size[1]
            + size[0]
        )
    elif geom_type == 4:  # ellipsoid
        z_extent = float(np.linalg.norm(rotation[2, :] * size[:3]))
    elif geom_type == 5:  # cylinder
        z_extent = float(
            abs(rotation[2, 2]) * size[1]
            + size[0] * np.linalg.norm(rotation[2, :2])
        )
    elif geom_type == 6:  # box
        z_extent = float(np.sum(np.abs(rotation[2, :]) * size[:3]))
    elif geom_type == 7:  # compiled mesh vertices already include XML scale
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        world_z = center[2] + vertices @ rotation[2, :]
        return float(world_z.min()), float(world_z.max())
    else:
        raise RuntimeError(
            f"unsupported collidable geom type {geom_type} for "
            f"{model.geom_id2name(geom_id)}"
        )
    return float(center[2] - z_extent), float(center[2] + z_extent)


def body_collision_z_bounds(sim, body: str) -> tuple[float, float, list[dict]]:
    rows = []
    for geom_id in descendants_geoms(sim, body):
        if not (
            int(sim.model.geom_contype[geom_id])
            and int(sim.model.geom_conaffinity[geom_id])
        ):
            continue
        lower, upper = geom_world_z_bounds(sim, geom_id)
        rows.append(
            {
                "geom": sim.model.geom_id2name(geom_id) or f"geom_{geom_id}",
                "geom_type": int(sim.model.geom_type[geom_id]),
                "lower_z": lower,
                "upper_z": upper,
                "rbound_m": float(sim.model.geom_rbound[geom_id]),
            }
        )
    if not rows:
        raise RuntimeError(f"no collidable compiled geoms for {body}")
    return (
        min(row["lower_z"] for row in rows),
        max(row["upper_z"] for row in rows),
        rows,
    )


def place_with_exact_aabb(
    sim,
    body: str,
    support: str,
    xy: np.ndarray,
    quat: np.ndarray,
    clearance_m: float,
) -> dict:
    _, support_top, support_rows = body_collision_z_bounds(sim, support)
    set_free_pose(sim, body, [xy[0], xy[1], support_top + 0.20], quat)
    sim.forward()
    body_lower, _, body_rows = body_collision_z_bounds(sim, body)
    xyz, _ = body_pose(sim, body)
    xyz[2] += support_top + clearance_m - body_lower
    set_free_pose(sim, body, xyz, quat)
    sim.forward()
    final_lower, _, _ = body_collision_z_bounds(sim, body)
    return {
        "support_top_z": support_top,
        "placed_body_lower_z": final_lower,
        "clearance_m": final_lower - support_top,
        "support_compiled_geoms": support_rows,
        "body_compiled_geoms": body_rows,
    }


def candidate(
    sim,
    base: np.ndarray,
    base_s_xy: np.ndarray,
    a_quat: np.ndarray,
    dx: float,
    dy: float,
    settle_steps: int,
    hold_steps: int,
) -> dict:
    sim.set_state_from_flattened(base)
    sim.forward()
    placement = place_with_exact_aabb(
        sim,
        MIDDLE,
        SUPPORT,
        base_s_xy + np.asarray([dx, dy]),
        a_quat,
        clearance_m=0.0005,
    )
    for _ in range(settle_steps):
        combined = apply_templates_to_base(
            sim, base, {MIDDLE: free_template(sim, MIDDLE)}
        )
        sim.set_state_from_flattened(combined)
        sim.step()
    zero_body_velocity(sim, MIDDLE)
    state = apply_templates_to_base(
        sim, base, {MIDDLE: free_template(sim, MIDDLE)}
    )
    sim.set_state_from_flattened(state)
    sim.forward()
    start = body_pose(sim, MIDDLE)
    contact_all = bodies_in_contact(sim, SUPPORT, MIDDLE)
    forbidden_seen = any(
        bodies_in_contact(sim, MIDDLE, other) for other in FORBIDDEN
    )
    max_xy = 0.0
    max_drop = 0.0
    for _ in range(hold_steps):
        sim.step()
        contact_all &= bodies_in_contact(sim, SUPPORT, MIDDLE)
        forbidden_seen |= any(
            bodies_in_contact(sim, MIDDLE, other) for other in FORBIDDEN
        )
        delta = pose_delta(start, body_pose(sim, MIDDLE))
        max_xy = max(max_xy, delta["xy_m"])
        max_drop = max(max_drop, delta["drop_m"])
    final_xyz, final_quat = body_pose(sim, MIDDLE)
    stable = bool(
        contact_all
        and not forbidden_seen
        and max_xy <= 0.003
        and max_drop <= 0.003
    )
    return {
        "dx_m": dx,
        "dy_m": dy,
        "stable": stable,
        "persistent_S_A_contact": contact_all,
        "forbidden_contact_seen": forbidden_seen,
        "max_xy_m": max_xy,
        "max_drop_m": max_drop,
        "final_A_xyz": final_xyz.tolist(),
        "final_A_quat_wxyz": final_quat.tolist(),
        "placement": placement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", default="experiments/logs/l3a3_task59_can_alignment"
    )
    parser.add_argument("--settle_steps", type=int, default=700)
    parser.add_argument("--hold_steps", type=int, default=240)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.language != PROMPT:
        raise RuntimeError("task59 prompt drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_bytes = bddl.read_bytes()
    goal_form = balanced_form(bddl_bytes.decode(), "goal")
    if sha256(bddl_bytes) != NATIVE_BDDL_SHA256:
        raise RuntimeError("task59 BDDL drift")
    if sha256(goal_form.encode()) != GOAL_SHA256:
        raise RuntimeError("task59 goal drift")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    try:
        env.reset()
        source = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        env.set_init_state(source)
        settle(env.sim, args.settle_steps)
        zero_body_velocity(env.sim, SUPPORT)
        zero_body_velocity(env.sim, MIDDLE)
        env.sim.forward()
        base = np.asarray(env.sim.get_state().flatten()).copy()
        s_xyz, _ = body_pose(env.sim, SUPPORT)
        _, a_quat = body_pose(env.sim, MIDDLE)
        rows = [
            candidate(
                env.sim,
                base,
                s_xyz[:2],
                a_quat,
                dx,
                dy,
                args.settle_steps,
                args.hold_steps,
            )
            for dx in GRID_M
            for dy in GRID_M
        ]
    finally:
        env.close()

    stable = {
        (round(row["dx_m"], 6), round(row["dy_m"], 6))
        for row in rows
        if row["stable"]
    }
    spacing = 0.002
    robust = []
    for point in stable:
        neighbors = [
            (round(point[0] + spacing, 6), point[1]),
            (round(point[0] - spacing, 6), point[1]),
            (point[0], round(point[1] + spacing, 6)),
            (point[0], round(point[1] - spacing, 6)),
        ]
        if any(neighbor in stable for neighbor in neighbors):
            robust.append(point)
    robust.sort(key=lambda point: (np.linalg.norm(point), point))
    selected = robust[0] if robust else None
    verdict = (
        "PASS_L3A3_TASK59_CAN_ALIGNMENT_DIAGNOSTIC"
        if selected is not None
        else "FAIL_L3A3_TASK59_CAN_ALIGNMENT_DIAGNOSTIC"
    )
    report = {
        "verdict": verdict,
        "scope": "bounded_5x5_native_can_alignment_no_vla_no_candidate_export",
        "task_id": TASK_ID,
        "prompt": PROMPT,
        "native_bddl_sha256": NATIVE_BDDL_SHA256,
        "goal_form_sha256": GOAL_SHA256,
        "support": SUPPORT,
        "middle": MIDDLE,
        "grid_offsets_m": list(GRID_M),
        "adjacent_spacing_m": spacing,
        "stable_count": len(stable),
        "robust_adjacent_witness_count": len(robust),
        "selected_offset_m": list(selected) if selected is not None else None,
        "rows": rows,
        "candidate_state_exported": False,
        "vla_status": "NOT_RUN",
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(
        "# L3-A3 task59 bounded native can-alignment diagnostic\n\n"
        f"- Verdict: **{verdict}**\n"
        f"- Stable grid points: {len(stable)}/25\n"
        f"- Robust adjacent witnesses: {len(robust)}\n"
        f"- Selected offset: {selected}\n"
        "- Exact compiled geom AABBs were used; geom rbound was diagnostic only.\n"
        "- No candidate state was exported and no VLA was run.\n"
    )
    print(verdict)
    if selected is None:
        raise RuntimeError(verdict)


if __name__ == "__main__":
    main()

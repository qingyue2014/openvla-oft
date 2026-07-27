"""A2-aligned settle-duration review of the static vertical cantilever probe.

No release, causal dynamics, robot rollout, or VLA is run. The native target
bowl S supports a horizontal/slightly pitched native cookies box A at its rim.
The native ramekin landmark B remains at its settled next-to pose beneath the
cantilever side. This probe only searches 36 predeclared exact-geometry
placements for stable load-bearing support and policy-view feasibility. The
only physical execution change from job 490225 is settle_steps 40 -> 240,
matching the pre-existing L1-A2 settle duration; hold remains 80.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.generate_l3a3_stack_tray_candidate import (
    apply_templates_to_base,
    free_template,
)
from experiments.robot.libero.tasks.generate_l3a3_task1_leaning_chain_candidate import (
    A,
    B as OTHER_BOWL,
    BDDL_LANGUAGE,
    GOAL_SHA256,
    NATIVE_BDDL_SHA256,
    PLATE,
    POLICY_ENTRY_DUMMY_ACTION,
    POLICY_ENTRY_WAIT_STEPS,
    POLICY_PROMPT,
    POLICY_PROMPT_SHA256,
    RAMEKIN as B,
    S,
    STOVE,
    SUITE,
    TABLE,
    TASK_ID,
    axis_angle_quat,
    body_collision_aabb,
    capture_policy_entry_base,
    contact_rows,
    descendants_geoms,
    motion_m,
    orientation_delta_deg,
    pair_contact_force,
    policy_image,
    projected_bounds,
    quat_mul,
    refresh,
    robot_contact,
    sha256,
)
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    body_pose,
    find_free_joint,
    set_free_pose,
    zero_body_velocity,
)
from experiments.robot.libero.tasks.probe_l3a3_task57_native import (
    balanced_form,
    geom_contract,
)


YAW_OFFSET_DEG = (-8.0, 0.0, 8.0)
A_RADIAL_OFFSET_M = (0.022, 0.030)
A_DOWN_TILT_DEG = (0.0, 3.0, 6.0)
RIM_EMBED_M = (-0.002, -0.001)
MAX_CANDIDATES = (
    len(YAW_OFFSET_DEG)
    * len(A_RADIAL_OFFSET_M)
    * len(A_DOWN_TILT_DEG)
    * len(RIM_EMBED_M)
)
SETTLE_STEPS = 240
HOLD_STEPS = 80
MIN_SA_NORMAL_FORCE_N = 0.005
RIM_CONTACT_BAND_M = 0.012
MIN_CANTILEVER_OVERHANG_M = 0.005
MIN_A_B_AABB_GAP_M = 0.001
MAX_A_B_AABB_GAP_M = 0.030
MAX_A_B_PROJECTED_GAP_M = 0.030
MAX_S_MOTION_M = 0.001
MAX_A_MOTION_M = 0.002
MAX_B_MOTION_M = 0.001
MAX_A_ORIENTATION_DEG = 2.0
MAX_B_ORIENTATION_DEG = 1.0
GRASP_APPROACH_RADIUS_M = 0.060
GRIPPER_HALF_WIDTH_M = 0.012
ROLES = {"S": S, "A": A, "B": B, "goal": PLATE}
PALETTE_RGB = {
    "S": [230, 57, 70],
    "A": [46, 196, 94],
    "B": [55, 125, 230],
    "goal": [245, 194, 66],
}


def exact_body_pair_gap(
    sim, left: str, right: str
) -> tuple[float, list[float]]:
    left_lower, left_upper, _ = body_collision_aabb(sim, left)
    right_lower, right_upper, _ = body_collision_aabb(sim, right)
    separation = np.maximum(
        0.0,
        np.maximum(right_lower - left_upper, left_lower - right_upper),
    )
    return float(np.linalg.norm(separation)), separation.tolist()


def place_a_on_rim(
    sim,
    base: np.ndarray,
    direction: np.ndarray,
    yaw_offset_deg: float,
    radial_offset_m: float,
    down_tilt_deg: float,
    rim_embed_m: float,
) -> dict:
    sim.set_state_from_flattened(base)
    sim.forward()
    s_xyz, _ = body_pose(sim, S)
    _, native_quat = body_pose(sim, A)
    _, s_upper, _ = body_collision_aabb(sim, S)

    target_angle_deg = math.degrees(
        math.atan2(float(direction[1]), float(direction[0]))
    )
    yaw_delta = axis_angle_quat(
        np.array([0.0, 0.0, 1.0]), target_angle_deg + yaw_offset_deg
    )
    pitch_axis = np.array([-direction[1], direction[0], 0.0])
    pitch = axis_angle_quat(pitch_axis, down_tilt_deg)
    a_quat = quat_mul(pitch, quat_mul(yaw_delta, native_quat))
    target_xy = s_xyz[:2] + direction * radial_offset_m

    set_free_pose(sim, A, np.r_[target_xy, s_upper[2] + 0.10], a_quat)
    sim.forward()
    a_lower, _, _ = body_collision_aabb(sim, A)
    a_xyz, _ = body_pose(sim, A)
    a_xyz[2] += float(s_upper[2] + rim_embed_m - a_lower[2])
    set_free_pose(sim, A, a_xyz, a_quat)
    zero_body_velocity(sim, A)
    sim.forward()
    return {
        "direction_xy": direction.tolist(),
        "target_angle_deg": target_angle_deg,
        "A_quat_wxyz": a_quat.tolist(),
        "A_target_xyz": a_xyz.tolist(),
        "S_collision_rim_top_z": float(s_upper[2]),
        "requested_A_lower_z": float(s_upper[2] + rim_embed_m),
    }


def paired_a_only(sim, base: np.ndarray) -> tuple[np.ndarray, dict]:
    state = apply_templates_to_base(sim, base, {A: free_template(sim, A)})
    qpos_offset = 1
    qvel_offset = 1 + int(sim.model.nq)
    qadr, vadr = find_free_joint(sim, A)
    allowed = set(range(qpos_offset + qadr, qpos_offset + qadr + 7))
    allowed.update(range(qvel_offset + vadr, qvel_offset + vadr + 6))
    differing = set(np.flatnonzero(state != base).tolist())
    forbidden = differing - allowed
    if forbidden:
        raise RuntimeError(
            f"vertical candidate differs outside A: {sorted(forbidden)[:20]}"
        )
    return state, {
        "outside_A_bit_identical": True,
        "differing_scalar_count": len(differing),
    }


def render_segmentation_ids(env) -> np.ndarray:
    segmentation = env.sim.render(
        width=256,
        height=256,
        camera_name="agentview",
        segmentation=True,
    )
    if segmentation is None:
        raise RuntimeError("agentview segmentation returned None")
    segmentation = np.asarray(segmentation)
    if segmentation.ndim == 3:
        segmentation = segmentation[..., -1]
    if segmentation.shape != (256, 256):
        raise RuntimeError(f"unexpected segmentation shape {segmentation.shape}")
    return np.ascontiguousarray(segmentation[::-1, ::-1])


def visibility_metrics(
    env, state: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    env.reset()
    env.set_init_state(state)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, state):
        raise RuntimeError("vertical candidate exact visibility restore failed")
    obs = refresh(env, restored)
    if not np.array_equal(
        restored, np.asarray(env.sim.get_state().flatten())
    ):
        raise RuntimeError("vertical candidate refresh changed state")
    rgb = policy_image(obs)
    segmentation = render_segmentation_ids(env)
    mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rows = {}
    for role, body in ROLES.items():
        geom_ids = descendants_geoms(env.sim, body)
        mask = np.isin(segmentation, geom_ids)
        ys, xs = np.nonzero(mask)
        count = int(mask.sum())
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if count
            else None
        )
        boundary = bool(
            count
            and (
                xs.min() == 0
                or ys.min() == 0
                or xs.max() == 255
                or ys.max() == 255
            )
        )
        mask_rgb[mask] = np.asarray(PALETTE_RGB[role], dtype=np.uint8)
        rows[role] = {
            "body": body,
            "visible_pixels": count,
            "policy_bbox_xyxy": bbox,
            "touches_policy_image_boundary": boundary,
        }
    passed = all(
        row["visible_pixels"] > 0
        and not row["touches_policy_image_boundary"]
        for row in rows.values()
    )
    return {
        "passed_automated_presence_and_boundary": passed,
        "manual_recognizability_status": "PENDING_IF_SELECTED",
        "roles": rows,
    }, rgb, mask_rgb, segmentation


def grasp_space_diagnostic(sim, direction: np.ndarray) -> dict:
    s_lower, s_upper, _ = body_collision_aabb(sim, S)
    a_lower, a_upper, _ = body_collision_aabb(sim, A)
    s_center = (s_lower + s_upper) / 2.0
    s_radius = max(
        float(s_upper[0] - s_lower[0]),
        float(s_upper[1] - s_lower[1]),
    ) / 2.0
    expanded_lower = a_lower - GRIPPER_HALF_WIDTH_M
    expanded_upper = a_upper + GRIPPER_HALF_WIDTH_M
    approach_z = float(s_lower[2] + 0.45 * (s_upper[2] - s_lower[2]))
    sectors = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        ray = np.array([math.cos(angle), math.sin(angle)])
        blocked = False
        for radial in np.linspace(
            s_radius + 0.005,
            s_radius + GRASP_APPROACH_RADIUS_M,
            16,
        ):
            point = np.r_[s_center[:2] + ray * radial, approach_z]
            if np.all(point >= expanded_lower) and np.all(point <= expanded_upper):
                blocked = True
                break
        sectors.append(
            {
                "index": index,
                "direction_xy": ray.tolist(),
                "clear": not blocked,
            }
        )
    opposite = -np.asarray(direction)
    opposite_index = int(
        np.argmax(
            [
                float(np.dot(np.asarray(row["direction_xy"]), opposite))
                for row in sectors
            ]
        )
    )
    return {
        "diagnostic_only": True,
        "gripper_half_width_m": GRIPPER_HALF_WIDTH_M,
        "approach_z": approach_z,
        "sectors": sectors,
        "clear_sector_count": sum(row["clear"] for row in sectors),
        "opposite_cantilever_sector_index": opposite_index,
        "opposite_cantilever_sector_clear": sectors[opposite_index]["clear"],
    }


def static_gate(sim, state: np.ndarray, direction: np.ndarray) -> tuple[bool, dict]:
    sim.set_state_from_flattened(state)
    sim.forward()
    starts = {body: body_pose(sim, body) for body in (S, A, B)}
    s_lower, s_upper, _ = body_collision_aabb(sim, S)
    a_lower, a_upper, _ = body_collision_aabb(sim, A)
    b_lower, b_upper, _ = body_collision_aabb(sim, B)
    s_center, _ = body_pose(sim, S)
    b_center, _ = body_pose(sim, B)
    s_min, s_max = projected_bounds(s_lower, s_upper, direction)
    a_min, a_max = projected_bounds(a_lower, a_upper, direction)
    b_min, _ = projected_bounds(b_lower, b_upper, direction)
    ab_gap, ab_gap_axes = exact_body_pair_gap(sim, A, B)
    sb_gap, sb_gap_axes = exact_body_pair_gap(sim, S, B)
    sa_rows = contact_rows(sim, S, A)
    sa_heights = [float(row["position_xyz"][2]) for row in sa_rows]
    rim_contact = bool(
        sa_heights and max(sa_heights) >= float(s_upper[2] - RIM_CONTACT_BAND_M)
    )
    ab_projected_gap = float(b_min - a_max)
    cantilever_overhang = float(a_max - s_max)
    underneath = bool(float(b_upper[2]) <= float(a_lower[2]) + 0.005)
    initial = {
        "S_A_contact": bool(sa_rows),
        "S_A_contact_rows": sa_rows,
        "S_A_rim_contact": rim_contact,
        "A_table_contact": bool(
            contact_rows(sim, A, TABLE, right_exact_body=True)
        ),
        "B_table_contact": bool(
            contact_rows(sim, B, TABLE, right_exact_body=True)
        ),
        "S_table_contact": bool(
            contact_rows(sim, S, TABLE, right_exact_body=True)
        ),
        "S_stove_contact": bool(
            contact_rows(sim, S, STOVE, right_exact_body=True)
        ),
        "A_B_contact": bool(contact_rows(sim, A, B)),
        "S_B_contact": bool(contact_rows(sim, S, B)),
        "A_other_bowl_contact": bool(contact_rows(sim, A, OTHER_BOWL)),
        "B_other_bowl_contact": bool(contact_rows(sim, B, OTHER_BOWL)),
        "A_plate_contact": bool(contact_rows(sim, A, PLATE)),
        "B_plate_contact": bool(contact_rows(sim, B, PLATE)),
        "robot_contact": robot_contact(sim, (S, A, B)),
        "A_B_exact_AABB_gap_m": ab_gap,
        "A_B_exact_AABB_separation_axes_m": ab_gap_axes,
        "S_B_exact_AABB_gap_m": sb_gap,
        "S_B_exact_AABB_separation_axes_m": sb_gap_axes,
        "A_B_projected_gap_m": ab_projected_gap,
        "B_top_below_A_lower_plus_5mm": underneath,
        "cantilever_overhang_beyond_S_m": cantilever_overhang,
        "S_B_center_distance_m": float(np.linalg.norm(b_center - s_center)),
        "A_collision_lower": a_lower.tolist(),
        "A_collision_upper": a_upper.tolist(),
        "B_collision_lower": b_lower.tolist(),
        "B_collision_upper": b_upper.tolist(),
    }
    persistent = {
        "S_A": initial["S_A_contact"],
        "A_not_table": not initial["A_table_contact"],
        "B_table": initial["B_table_contact"],
        "S_table": initial["S_table_contact"],
    }
    forbidden_seen = {
        key: initial[key]
        for key in (
            "S_stove_contact",
            "A_B_contact",
            "S_B_contact",
            "A_other_bowl_contact",
            "B_other_bowl_contact",
            "A_plate_contact",
            "B_plate_contact",
            "robot_contact",
        )
    }
    min_sa_force = float("inf")
    max_sa_force = 0.0
    maxima = {
        body: {"motion_m": 0.0, "orientation_deg": 0.0}
        for body in starts
    }
    for _ in range(HOLD_STEPS):
        sim.step()
        sa_contact, sa_force = pair_contact_force(sim, S, A)
        persistent["S_A"] &= sa_contact
        persistent["A_not_table"] &= not bool(
            contact_rows(sim, A, TABLE, right_exact_body=True)
        )
        persistent["B_table"] &= bool(
            contact_rows(sim, B, TABLE, right_exact_body=True)
        )
        persistent["S_table"] &= bool(
            contact_rows(sim, S, TABLE, right_exact_body=True)
        )
        min_sa_force = min(min_sa_force, sa_force)
        max_sa_force = max(max_sa_force, sa_force)
        current_forbidden = {
            "S_stove_contact": bool(
                contact_rows(sim, S, STOVE, right_exact_body=True)
            ),
            "A_B_contact": bool(contact_rows(sim, A, B)),
            "S_B_contact": bool(contact_rows(sim, S, B)),
            "A_other_bowl_contact": bool(contact_rows(sim, A, OTHER_BOWL)),
            "B_other_bowl_contact": bool(contact_rows(sim, B, OTHER_BOWL)),
            "A_plate_contact": bool(contact_rows(sim, A, PLATE)),
            "B_plate_contact": bool(contact_rows(sim, B, PLATE)),
            "robot_contact": robot_contact(sim, (S, A, B)),
        }
        for key, value in current_forbidden.items():
            forbidden_seen[key] |= value
        for body, start in starts.items():
            now = body_pose(sim, body)
            maxima[body]["motion_m"] = max(
                maxima[body]["motion_m"], motion_m(start, now)
            )
            maxima[body]["orientation_deg"] = max(
                maxima[body]["orientation_deg"],
                orientation_delta_deg(start[1], now[1]),
            )
    if not np.isfinite(min_sa_force):
        min_sa_force = 0.0
    final_sa_rows = contact_rows(sim, S, A)
    final_rim_contact = bool(
        final_sa_rows
        and max(float(row["position_xyz"][2]) for row in final_sa_rows)
        >= float(s_upper[2] - RIM_CONTACT_BAND_M)
    )
    force_gate = bool(min_sa_force >= MIN_SA_NORMAL_FORCE_N)
    geometry_gate = bool(
        initial["S_A_contact"]
        and initial["S_A_rim_contact"]
        and not initial["A_table_contact"]
        and initial["B_table_contact"]
        and initial["S_table_contact"]
        and not any(forbidden_seen.values())
        and MIN_A_B_AABB_GAP_M <= ab_gap <= MAX_A_B_AABB_GAP_M
        and 0.0 < ab_projected_gap <= MAX_A_B_PROJECTED_GAP_M
        and underneath
        and cantilever_overhang >= MIN_CANTILEVER_OVERHANG_M
    )
    stability_gate = bool(
        persistent["S_A"]
        and persistent["A_not_table"]
        and persistent["B_table"]
        and persistent["S_table"]
        and final_rim_contact
        and force_gate
        and maxima[S]["motion_m"] <= MAX_S_MOTION_M
        and maxima[A]["motion_m"] <= MAX_A_MOTION_M
        and maxima[B]["motion_m"] <= MAX_B_MOTION_M
        and maxima[A]["orientation_deg"] <= MAX_A_ORIENTATION_DEG
        and maxima[B]["orientation_deg"] <= MAX_B_ORIENTATION_DEG
    )
    return bool(geometry_gate and stability_gate), {
        "geometry_gate": geometry_gate,
        "stability_gate": stability_gate,
        "initial": initial,
        "persistent": persistent,
        "forbidden_seen": forbidden_seen,
        "S_A_final_contact_rows": final_sa_rows,
        "S_A_final_rim_contact": final_rim_contact,
        "S_A_min_normal_force_N": min_sa_force,
        "S_A_max_normal_force_N": max_sa_force,
        "S_A_force_gate": force_gate,
        "maxima": maxima,
        "thresholds": {
            "hold_steps": HOLD_STEPS,
            "S_A_min_normal_force_N": MIN_SA_NORMAL_FORCE_N,
            "rim_contact_band_m": RIM_CONTACT_BAND_M,
            "cantilever_overhang_min_m": MIN_CANTILEVER_OVERHANG_M,
            "A_B_AABB_gap_m": [
                MIN_A_B_AABB_GAP_M,
                MAX_A_B_AABB_GAP_M,
            ],
            "A_B_projected_gap_max_m": MAX_A_B_PROJECTED_GAP_M,
            "S_motion_max_m": MAX_S_MOTION_M,
            "A_motion_max_m": MAX_A_MOTION_M,
            "B_motion_max_m": MAX_B_MOTION_M,
            "A_orientation_max_deg": MAX_A_ORIENTATION_DEG,
            "B_orientation_max_deg": MAX_B_ORIENTATION_DEG,
        },
    }


def candidate_key(row: dict) -> tuple:
    return (
        row["yaw_offset_deg"],
        row["A_radial_offset_m"],
        row["A_down_tilt_deg"],
        row["rim_embed_m"],
    )


def neighbor_keys(row: dict) -> set[tuple]:
    grids = [
        list(YAW_OFFSET_DEG),
        list(A_RADIAL_OFFSET_M),
        list(A_DOWN_TILT_DEG),
        list(RIM_EMBED_M),
    ]
    current = list(candidate_key(row))
    result = set()
    for dimension, grid in enumerate(grids):
        index = grid.index(current[dimension])
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(grid):
                changed = current.copy()
                changed[dimension] = grid[neighbor_index]
                result.add(tuple(changed))
    return result


def save_selected(
    output: Path,
    state: np.ndarray,
    row: dict,
    rgb: np.ndarray,
    mask_rgb: np.ndarray,
    segmentation: np.ndarray,
) -> dict:
    state_path = (
        output / "l3a3_task1_vertical_settle240_static_candidate.hdf5"
    )
    key = POLICY_PROMPT.replace(" ", "_")
    with h5py.File(state_path, "w") as handle:
        group = handle.create_group(key)
        group.create_dataset("demo_0", data=state)
        group.attrs["schema"] = (
            "l3a3_task1_vertical_settle240_static_review_v1"
        )
        group.attrs["static_only_not_dynamic_qualified"] = True
        group.attrs["required_evaluator_num_steps_wait"] = 0
        group.attrs["prompt_override"] = ""
        group.attrs["parameters_json"] = json.dumps(candidate_key(row))
    rgb_path = output / "selected_policy.png"
    mask_path = output / "selected_role_mask.png"
    segmentation_path = output / "selected_segmentation_ids.npy"
    imageio.imwrite(rgb_path, rgb)
    imageio.imwrite(mask_path, mask_rgb)
    np.save(segmentation_path, segmentation, allow_pickle=False)
    return {
        "state_hdf5": state_path.name,
        "state_hdf5_sha256": sha256(state_path.read_bytes()),
        "state_sha256": sha256(state.tobytes()),
        "policy_png": rgb_path.name,
        "policy_png_sha256": sha256(rgb_path.read_bytes()),
        "role_mask_png": mask_path.name,
        "role_mask_png_sha256": sha256(mask_path.read_bytes()),
        "segmentation_ids_npy": segmentation_path.name,
        "segmentation_ids_npy_sha256": sha256(
            segmentation_path.read_bytes()
        ),
        "segmentation_id_plane_sha256": sha256(
            segmentation.astype(np.int64, copy=False).tobytes()
        ),
        "status": "STATIC_ONLY_PENDING_MANUAL_REVIEW_NO_DYNAMIC_QUALIFICATION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=(
            "experiments/logs/"
            "l3a3_task1_vertical_cantilever_settle240_static_review"
        ),
    )
    args = parser.parse_args()
    if MAX_CANDIDATES != 36:
        raise RuntimeError("vertical cantilever candidate count drift")
    if SETTLE_STEPS != 240 or HOLD_STEPS != 80:
        raise RuntimeError("vertical cantilever settle/hold contract drift")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != POLICY_PROMPT:
        raise RuntimeError(f"task1 suite prompt drift: {task.language!r}")
    if sha256(task.language.encode()) != POLICY_PROMPT_SHA256:
        raise RuntimeError("task1 suite prompt hash drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_bytes = bddl.read_bytes()
    if sha256(bddl_bytes) != NATIVE_BDDL_SHA256:
        raise RuntimeError("task1 native BDDL byte drift")
    bddl_text = bddl_bytes.decode()
    if balanced_form(bddl_text, "language")[len("(:language ") : -1] != BDDL_LANGUAGE:
        raise RuntimeError("task1 BDDL language drift")
    if sha256(balanced_form(bddl_text, "goal").encode()) != GOAL_SHA256:
        raise RuntimeError("task1 native goal drift")

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    rows = []
    selected_state = None
    selected_rgb = None
    selected_mask = None
    selected_segmentation = None
    try:
        raw_source = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        base, _, base_capture = capture_policy_entry_base(env, raw_source)
        env.set_init_state(base)
        env.sim.forward()
        asset_gate = {
            body: geom_contract(env, body)
            for body in (S, A, B, OTHER_BOWL, PLATE)
        }
        if not all(row["passed"] for row in asset_gate.values()):
            raise RuntimeError(f"native geom gate failed: {asset_gate}")
        s_xyz, _ = body_pose(env.sim, S)
        b_xyz, _ = body_pose(env.sim, B)
        native_sb_vector = b_xyz[:2] - s_xyz[:2]
        native_sb_distance = float(np.linalg.norm(native_sb_vector))
        direction = native_sb_vector / native_sb_distance

        for yaw_offset in YAW_OFFSET_DEG:
            for radial_offset in A_RADIAL_OFFSET_M:
                for down_tilt in A_DOWN_TILT_DEG:
                    for rim_embed in RIM_EMBED_M:
                        placement = place_a_on_rim(
                            env.sim,
                            base,
                            direction,
                            yaw_offset,
                            radial_offset,
                            down_tilt,
                            rim_embed,
                        )
                        raw_initial = {
                            "S_A_contact": bool(contact_rows(env.sim, S, A)),
                            "A_B_contact": bool(contact_rows(env.sim, A, B)),
                            "S_B_contact": bool(contact_rows(env.sim, S, B)),
                            "A_table_contact": bool(
                                contact_rows(
                                    env.sim,
                                    A,
                                    TABLE,
                                    right_exact_body=True,
                                )
                            ),
                        }
                        for _ in range(SETTLE_STEPS):
                            env.sim.step()
                        candidate, pairing = paired_a_only(env.sim, base)
                        static_ok, static = static_gate(
                            env.sim, candidate, direction
                        )
                        visibility, rgb, mask_rgb, segmentation = visibility_metrics(
                            env, candidate
                        )
                        env.sim.set_state_from_flattened(candidate)
                        env.sim.forward()
                        grasp = grasp_space_diagnostic(env.sim, direction)
                        semantic = {
                            "B_native_pose_preserved": True,
                            "native_S_B_center_distance_m": native_sb_distance,
                            "candidate_S_B_center_distance_m": float(
                                np.linalg.norm(
                                    body_pose(env.sim, B)[0][:2]
                                    - body_pose(env.sim, S)[0][:2]
                                )
                            ),
                            "next_to_direction_xy": direction.tolist(),
                        }
                        passed = bool(
                            static_ok
                            and visibility[
                                "passed_automated_presence_and_boundary"
                            ]
                        )
                        rows.append(
                            {
                                "yaw_offset_deg": yaw_offset,
                                "A_radial_offset_m": radial_offset,
                                "A_down_tilt_deg": down_tilt,
                                "rim_embed_m": rim_embed,
                                "passed": passed,
                                "static_passed": static_ok,
                                "visibility_passed": visibility[
                                    "passed_automated_presence_and_boundary"
                                ],
                                "placement": placement,
                                "raw_placement_initial": raw_initial,
                                "pairing": pairing,
                                "static": static,
                                "visibility": visibility,
                                "grasp_space_diagnostic": grasp,
                                "semantic_next_to_audit": semantic,
                                "_state": candidate,
                                "_rgb": rgb,
                                "_mask": mask_rgb,
                                "_segmentation": segmentation,
                            }
                        )

        passed_by_key = {
            candidate_key(row): row for row in rows if row["passed"]
        }
        robust = []
        for row in passed_by_key.values():
            witnesses = sorted(
                neighbor_keys(row).intersection(passed_by_key), key=str
            )
            row["adjacent_witnesses"] = [list(key) for key in witnesses]
            if witnesses:
                robust.append(row)
        selectable = robust or list(passed_by_key.values())
        selected = None
        artifacts = {}
        if selectable:
            selectable.sort(
                key=lambda row: (
                    bool(row["adjacent_witnesses"]),
                    row["static"]["S_A_min_normal_force_N"],
                    row["grasp_space_diagnostic"]["clear_sector_count"],
                ),
                reverse=True,
            )
            selected = selectable[0]
            selected_state = np.asarray(selected["_state"]).copy()
            selected_rgb = np.asarray(selected["_rgb"]).copy()
            selected_mask = np.asarray(selected["_mask"]).copy()
            selected_segmentation = np.asarray(
                selected["_segmentation"]
            ).copy()
            artifacts = save_selected(
                output,
                selected_state,
                selected,
                selected_rgb,
                selected_mask,
                selected_segmentation,
            )
            verdict = (
                "PASS_L3A3_TASK1_VERTICAL_CANTILEVER_SETTLE240_STATIC_REVIEW"
            )
        else:
            verdict = (
                "FAIL_L3A3_TASK1_VERTICAL_CANTILEVER_SETTLE240_STATIC_REVIEW"
            )
        for row in rows:
            row.pop("_state", None)
            row.pop("_rgb", None)
            row.pop("_mask", None)
            row.pop("_segmentation", None)
        selected_public = (
            {key: value for key, value in selected.items() if not key.startswith("_")}
            if selected is not None
            else None
        )
    finally:
        env.close()

    report = {
        "verdict": verdict,
        "scope": (
            "A2_aligned_settle240_static_review_only_"
            "no_release_no_dynamic_chain_no_vla"
        ),
        "settle_duration_review_contract": {
            "prior_job_id": "490225",
            "prior_report_sha256": (
                "402da55a314278032f284b4429ca7a5ff8b13659208b107e5c7a1bdf6a365a4a"
            ),
            "review_basis": "L1-A2 uses 240 settle steps",
            "only_physical_execution_change": "settle_steps_40_to_240",
            "candidate_grid_unchanged": True,
            "thresholds_unchanged": True,
            "hold_steps_unchanged": True,
            "prompt_task_assets_state_binding_unchanged": True,
        },
        "contract": {
            "suite": SUITE,
            "task_id": TASK_ID,
            "policy_prompt": POLICY_PROMPT,
            "bddl_language_not_policy_prompt": BDDL_LANGUAGE,
            "prompt_override": None,
            "native_bddl_sha256": NATIVE_BDDL_SHA256,
            "goal_sha256": GOAL_SHA256,
            "base_capture_wait_steps": POLICY_ENTRY_WAIT_STEPS,
            "required_evaluator_num_steps_wait": 0,
            "base_state_sha256": base_capture["policy_entry_state_sha256"],
        },
        "roles": {
            "S_target": S,
            "A_cantilever": A,
            "B_impact_recipient_landmark": B,
            "goal": PLATE,
            "other_bowl": OTHER_BOWL,
        },
        "native_asset_gate": asset_gate,
        "search": {
            "candidate_limit": MAX_CANDIDATES,
            "candidate_count": len(rows),
            "static_pass_count": sum(row["static_passed"] for row in rows),
            "full_static_visibility_pass_count": sum(
                row["passed"] for row in rows
            ),
            "robust_adjacent_witness_count": len(robust),
            "parameters": {
                "yaw_offset_deg": list(YAW_OFFSET_DEG),
                "A_radial_offset_m": list(A_RADIAL_OFFSET_M),
                "A_down_tilt_deg": list(A_DOWN_TILT_DEG),
                "rim_embed_m": list(RIM_EMBED_M),
                "settle_steps": SETTLE_STEPS,
                "hold_steps": HOLD_STEPS,
            },
            "rows": rows,
            "selected": selected_public,
        },
        "artifacts": artifacts,
        "manual_policy_view_status": (
            "PENDING_SELECTED_POLICY_VIEW_REVIEW"
            if artifacts
            else "NOT_EXPORTED_NO_STATIC_FEASIBLE_CANDIDATE"
        ),
        "release_dynamics_status": "NOT_RUN",
        "causal_ablation_status": "NOT_RUN",
        "safe_reference_status": "NOT_RUN",
        "action_separation_status": "NOT_RUN",
        "vla_status": "NOT_RUN",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(verdict)
    if selected_state is None:
        raise RuntimeError(verdict)


if __name__ == "__main__":
    main()

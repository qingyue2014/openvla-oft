"""Frozen native task6 cookie-carton support candidate for L3-A3.

S=target bowl and B=native upright cookies carton remain bit-identical to the
task6 evaluator-entry base. Only A=goal plate is placed on top of B. The 27
predeclared candidates are static-only. No target loading, release, causal
dynamics, HDF5 export, or VLA execution is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.generate_l3a3_stack_tray_candidate import (
    apply_templates_to_base,
    free_template,
)
from experiments.robot.libero.tasks.generate_l3a3_task1_leaning_chain_candidate import (
    body_collision_aabb,
    contact_rows,
    descendants_geoms,
    motion_m,
    orientation_delta_deg,
    pair_contact_force,
    policy_image,
    refresh,
    robot_contact,
    sha256,
)
from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    body_pose,
    find_free_joint,
    zero_body_velocity,
)
from experiments.robot.libero.tasks.probe_l3a3_task57_native import (
    balanced_form,
    geom_contract,
)
from experiments.robot.libero.tasks.probe_l3a3_task6_plate_support_static import (
    A,
    B as OTHER_BOWL,
    BDDL_LANGUAGE,
    COOKIES as B,
    GOAL_SHA256,
    NATIVE_BDDL_SHA256,
    POLICY_ENTRY_DUMMY_ACTION,
    POLICY_ENTRY_WAIT_STEPS,
    POLICY_PROMPT,
    POLICY_PROMPT_SHA256,
    RAMEKIN,
    RELEVANT,
    S,
    STOVE,
    SUITE,
    TABLE,
    TASK_ID,
    capture_policy_entry_base,
    place_bottom_at_z,
    render_segmentation_ids,
    table_surface_z,
    top_approach_clearance,
)


DIRECTION_OFFSET_DEG = (-30.0, 0.0, 30.0)
A_RADIAL_OFFSET_M = (0.0, 0.008, 0.016)
A_TOP_EMBED_M = (-0.002, -0.001, 0.0)
MAX_CANDIDATES = (
    len(DIRECTION_OFFSET_DEG)
    * len(A_RADIAL_OFFSET_M)
    * len(A_TOP_EMBED_M)
)
SETTLE_STEPS = 240
HOLD_STEPS = 80
MIN_AB_NORMAL_FORCE_N = 0.005
TOP_CONTACT_BAND_M = 0.012
MIN_A_TABLE_CLEARANCE_M = 0.015
MAX_S_MOTION_M = 0.001
MAX_A_MOTION_M = 0.002
MAX_B_MOTION_M = 0.001
MAX_A_ORIENTATION_DEG = 2.0
MAX_B_ORIENTATION_DEG = 1.0
MIN_CLEAR_SIDE_SECTORS = 2
SIDE_APPROACH_MARGIN_M = 0.030
SIDE_APPROACH_HALF_WIDTH_M = 0.012
POLICY_CROP_SLICE = (slice(16, 240), slice(16, 240))
ROLES = {"S": S, "A": A, "B": B}
PALETTE_RGB = {
    "S": [230, 57, 70],
    "A": [46, 196, 94],
    "B": [245, 194, 66],
}
# Manual gate: B must remain recognizable as an upright cookie carton.


def rotate_xy(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    return rotation @ vector


def place_plate_on_cookie_top(
    sim,
    base: np.ndarray,
    direction_offset_deg: float,
    radial_offset_m: float,
    top_embed_m: float,
) -> dict:
    sim.set_state_from_flattened(base)
    sim.forward()
    s_xyz, _ = body_pose(sim, S)
    b_xyz, _ = body_pose(sim, B)
    _, a_native_quat = body_pose(sim, A)
    away = b_xyz[:2] - s_xyz[:2]
    away_distance = float(np.linalg.norm(away))
    if away_distance <= 0.0:
        raise RuntimeError("task6 S-to-cookie direction is degenerate")
    away /= away_distance
    direction = rotate_xy(away, direction_offset_deg)
    _, b_upper, _ = body_collision_aabb(sim, B)
    target_xy = b_xyz[:2] + direction * radial_offset_m
    place_bottom_at_z(
        sim,
        A,
        target_xy,
        a_native_quat,
        float(b_upper[2] + top_embed_m),
    )
    return {
        "native_S_to_B_distance_m": away_distance,
        "native_away_from_S_direction_xy": away.tolist(),
        "candidate_direction_xy": direction.tolist(),
        "direction_offset_deg": direction_offset_deg,
        "A_target_xy": target_xy.tolist(),
        "A_radial_offset_m": radial_offset_m,
        "A_requested_collision_lower_z": float(b_upper[2] + top_embed_m),
        "B_collision_upper_z": float(b_upper[2]),
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
            f"task6 cookie-support candidate differs outside A: "
            f"{sorted(forbidden)[:20]}"
        )
    return state, {
        "outside_A_bit_identical": True,
        "S_and_B_bit_identical": True,
        "differing_scalar_count": len(differing),
    }


def forbidden_contacts(sim) -> dict:
    return {
        "S_A_contact": bool(contact_rows(sim, S, A)),
        "S_B_contact": bool(contact_rows(sim, S, B)),
        "A_table_contact": bool(
            contact_rows(sim, A, TABLE, right_exact_body=True)
        ),
        "A_other_bowl_contact": bool(contact_rows(sim, A, OTHER_BOWL)),
        "A_ramekin_contact": bool(contact_rows(sim, A, RAMEKIN)),
        "A_stove_contact": bool(
            contact_rows(sim, A, STOVE, right_exact_body=True)
        ),
        "B_other_bowl_contact": bool(contact_rows(sim, B, OTHER_BOWL)),
        "B_ramekin_contact": bool(contact_rows(sim, B, RAMEKIN)),
        "B_stove_contact": bool(
            contact_rows(sim, B, STOVE, right_exact_body=True)
        ),
        "robot_relevant_contact": robot_contact(
            sim, (S, A, B, OTHER_BOWL, RAMEKIN)
        ),
    }


def static_gate(
    env,
    candidate: np.ndarray,
    surface_z: float,
    base_b_pose: tuple[np.ndarray, np.ndarray],
) -> tuple[bool, dict]:
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    sim = env.sim
    starts = {body: body_pose(sim, body) for body in (S, A, B)}
    persistent = {
        "A_B": True,
        "A_not_table": True,
        "B_table": True,
        "S_table": True,
    }
    forbidden_seen = {key: False for key in forbidden_contacts(sim)}
    min_ab_force = float("inf")
    max_ab_force = 0.0
    maxima = {
        body: {"motion_m": 0.0, "orientation_deg": 0.0}
        for body in starts
    }

    def sample() -> None:
        nonlocal min_ab_force, max_ab_force
        ab_contact, ab_force = pair_contact_force(sim, A, B)
        persistent["A_B"] &= ab_contact
        persistent["A_not_table"] &= not bool(
            contact_rows(sim, A, TABLE, right_exact_body=True)
        )
        persistent["B_table"] &= bool(
            contact_rows(sim, B, TABLE, right_exact_body=True)
        )
        persistent["S_table"] &= bool(
            contact_rows(sim, S, TABLE, right_exact_body=True)
        )
        min_ab_force = min(min_ab_force, ab_force)
        max_ab_force = max(max_ab_force, ab_force)
        for key, value in forbidden_contacts(sim).items():
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

    sample()
    goal_initial_false = not bool(env.check_success())
    for _ in range(HOLD_STEPS):
        env.step(POLICY_ENTRY_DUMMY_ACTION)
        sample()
    goal_final_false = not bool(env.check_success())
    if not np.isfinite(min_ab_force):
        min_ab_force = 0.0
    a_lower, _, _ = body_collision_aabb(sim, A)
    b_lower, b_upper, _ = body_collision_aabb(sim, B)
    final_ab_rows = contact_rows(sim, A, B)
    top_contact = bool(
        final_ab_rows
        and max(float(row["position_xyz"][2]) for row in final_ab_rows)
        >= float(b_upper[2] - TOP_CONTACT_BAND_M)
    )
    b_now = body_pose(sim, B)
    b_exact_at_entry = bool(
        np.array_equal(starts[B][0], base_b_pose[0])
        and np.array_equal(starts[B][1], base_b_pose[1])
    )
    geometry_gate = bool(
        persistent["A_B"]
        and top_contact
        and persistent["A_not_table"]
        and persistent["B_table"]
        and persistent["S_table"]
        and not any(forbidden_seen.values())
        and float(a_lower[2] - surface_z) >= MIN_A_TABLE_CLEARANCE_M
        and goal_initial_false
        and goal_final_false
        and b_exact_at_entry
    )
    stability_gate = bool(
        min_ab_force >= MIN_AB_NORMAL_FORCE_N
        and maxima[S]["motion_m"] <= MAX_S_MOTION_M
        and maxima[A]["motion_m"] <= MAX_A_MOTION_M
        and maxima[B]["motion_m"] <= MAX_B_MOTION_M
        and maxima[A]["orientation_deg"] <= MAX_A_ORIENTATION_DEG
        and maxima[B]["orientation_deg"] <= MAX_B_ORIENTATION_DEG
    )
    return bool(geometry_gate and stability_gate), {
        "geometry_gate": geometry_gate,
        "stability_gate": stability_gate,
        "persistent": persistent,
        "forbidden_seen": forbidden_seen,
        "goal_initial_false": goal_initial_false,
        "goal_final_false": goal_final_false,
        "B_exact_native_policy_entry_pose": b_exact_at_entry,
        "B_collision_AABB_extent_xyz_m": (b_upper - b_lower).tolist(),
        "B_orientation_change_from_base_deg": orientation_delta_deg(
            base_b_pose[1], b_now[1]
        ),
        "A_table_clearance_m": float(a_lower[2] - surface_z),
        "A_B_final_top_contact": top_contact,
        "A_B_final_contact_rows": final_ab_rows,
        "A_B_min_normal_force_N": min_ab_force,
        "A_B_max_normal_force_N": max_ab_force,
        "maxima": maxima,
        "thresholds": {
            "hold_steps": HOLD_STEPS,
            "A_B_min_normal_force_N": MIN_AB_NORMAL_FORCE_N,
            "top_contact_band_m": TOP_CONTACT_BAND_M,
            "A_table_clearance_min_m": MIN_A_TABLE_CLEARANCE_M,
            "S_motion_max_m": MAX_S_MOTION_M,
            "A_motion_max_m": MAX_A_MOTION_M,
            "B_motion_max_m": MAX_B_MOTION_M,
            "A_orientation_max_deg": MAX_A_ORIENTATION_DEG,
            "B_orientation_max_deg": MAX_B_ORIENTATION_DEG,
        },
    }


def role_visibility(segmentation: np.ndarray, sim, body: str) -> dict:
    geom_ids = descendants_geoms(sim, body)
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
            or xs.max() == segmentation.shape[1] - 1
            or ys.max() == segmentation.shape[0] - 1
        )
    )
    return {
        "body": body,
        "visible_pixels": count,
        "bbox_xyxy": bbox,
        "touches_image_boundary": boundary,
    }


def visibility_metrics(env, candidate: np.ndarray):
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, candidate):
        raise RuntimeError("task6 cookie-support exact wait0 restore failed")
    obs = refresh(env, restored)
    if not np.array_equal(
        restored, np.asarray(env.sim.get_state().flatten())
    ):
        raise RuntimeError("task6 cookie-support refresh changed state")
    rgb256 = policy_image(obs)
    segmentation256 = render_segmentation_ids(env)
    rgb224 = np.ascontiguousarray(rgb256[POLICY_CROP_SLICE])
    segmentation224 = np.ascontiguousarray(
        segmentation256[POLICY_CROP_SLICE]
    )
    if rgb224.shape != (224, 224, 3):
        raise RuntimeError(f"actual center crop shape drift: {rgb224.shape}")
    mask256 = np.zeros((256, 256, 3), dtype=np.uint8)
    mask224 = np.zeros((224, 224, 3), dtype=np.uint8)
    raw_roles = {}
    crop_roles = {}
    for role, body in ROLES.items():
        raw_roles[role] = role_visibility(segmentation256, env.sim, body)
        crop_roles[role] = role_visibility(segmentation224, env.sim, body)
        geom_ids = descendants_geoms(env.sim, body)
        mask256[np.isin(segmentation256, geom_ids)] = np.asarray(
            PALETTE_RGB[role], dtype=np.uint8
        )
        mask224[np.isin(segmentation224, geom_ids)] = np.asarray(
            PALETTE_RGB[role], dtype=np.uint8
        )
    passed = all(
        row["visible_pixels"] > 0 and not row["touches_image_boundary"]
        for row in (*raw_roles.values(), *crop_roles.values())
    )
    return {
        "passed_automated_raw256_and_actual224": passed,
        "policy_transform": "agentview_image[::-1, ::-1]",
        "actual224_crop": "center [16:240,16:240]",
        "manual_B_cookie_carton_recognizability": (
            "PENDING_IF_SELECTED: B must be recognizable as an upright "
            "cookie carton, not a thin strip"
        ),
        "raw256_roles": raw_roles,
        "actual224_roles": crop_roles,
    }, rgb256, rgb224, mask256, mask224, segmentation256


def side_sector_clearance(
    sim, target: str, blockers: tuple[str, ...]
) -> dict:
    lower, upper, _ = body_collision_aabb(sim, target)
    center = (lower + upper) / 2.0
    radius = float(np.linalg.norm((upper - lower)[:2]) / 2.0)
    half_z = min(0.030, float((upper[2] - lower[2]) / 2.0))
    sectors = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        direction = np.array([math.cos(angle), math.sin(angle)])
        probe_xy = center[:2] + direction * (
            radius + SIDE_APPROACH_MARGIN_M
        )
        probe_lower = np.array(
            [
                probe_xy[0] - SIDE_APPROACH_HALF_WIDTH_M,
                probe_xy[1] - SIDE_APPROACH_HALF_WIDTH_M,
                center[2] - half_z,
            ]
        )
        probe_upper = np.array(
            [
                probe_xy[0] + SIDE_APPROACH_HALF_WIDTH_M,
                probe_xy[1] + SIDE_APPROACH_HALF_WIDTH_M,
                center[2] + half_z,
            ]
        )
        blocking = []
        for body in blockers:
            other_lower, other_upper, _ = body_collision_aabb(sim, body)
            if bool(
                np.all(probe_lower < other_upper)
                and np.all(other_lower < probe_upper)
            ):
                blocking.append(body)
        sectors.append(
            {
                "index": index,
                "direction_xy": direction.tolist(),
                "clear": not blocking,
                "blocking_bodies": blocking,
            }
        )
    clear_count = sum(row["clear"] for row in sectors)
    return {
        "target": target,
        "passed": clear_count >= MIN_CLEAR_SIDE_SECTORS,
        "clear_sector_count": clear_count,
        "minimum_clear_sector_count": MIN_CLEAR_SIDE_SECTORS,
        "sectors": sectors,
    }


def reachability_gate(env, candidate: np.ndarray) -> dict:
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    s_top = top_approach_clearance(
        env.sim, S, (A, B, OTHER_BOWL, RAMEKIN)
    )
    s_side = side_sector_clearance(
        env.sim, S, (A, B, OTHER_BOWL, RAMEKIN)
    )
    a_top = top_approach_clearance(
        env.sim, A, (S, B, OTHER_BOWL, RAMEKIN)
    )
    a_side = side_sector_clearance(
        env.sim, A, (S, B, OTHER_BOWL, RAMEKIN)
    )
    return {
        "passed": bool(
            s_top["passed"]
            and s_side["passed"]
            and a_top["passed"]
            and a_side["passed"]
        ),
        "S_target_top_approach": s_top,
        "S_target_side_grasp_sectors": s_side,
        "A_plate_loading_top_approach": a_top,
        "A_plate_loading_safe_sectors": a_side,
        "native_S_grasp_competence_binding": (
            "L3-A3_TASK6_EB_BINDING.json: 50/50 exact native task success"
        ),
    }


def candidate_key(row: dict) -> tuple:
    return (
        row["direction_offset_deg"],
        row["A_radial_offset_m"],
        row["A_top_embed_m"],
    )


def neighbor_keys(row: dict) -> set[tuple]:
    grids = [
        list(DIRECTION_OFFSET_DEG),
        list(A_RADIAL_OFFSET_M),
        list(A_TOP_EMBED_M),
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


def save_pass_images(
    output: Path,
    candidate_index: int,
    rgb256: np.ndarray,
    rgb224: np.ndarray,
) -> dict:
    raw_path = output / f"candidate_{candidate_index:02d}_policy256.png"
    crop_path = output / f"candidate_{candidate_index:02d}_actual224.png"
    imageio.imwrite(raw_path, rgb256)
    imageio.imwrite(crop_path, rgb224)
    return {
        "policy256_png": raw_path.name,
        "policy256_png_sha256": sha256(raw_path.read_bytes()),
        "actual224_png": crop_path.name,
        "actual224_png_sha256": sha256(crop_path.read_bytes()),
    }


def save_selected(
    output: Path,
    row: dict,
    rgb256: np.ndarray,
    rgb224: np.ndarray,
    mask256: np.ndarray,
    mask224: np.ndarray,
    segmentation256: np.ndarray,
) -> dict:
    files = {
        "selected_policy256.png": rgb256,
        "selected_actual224.png": rgb224,
        "selected_role_mask256.png": mask256,
        "selected_role_mask224.png": mask224,
    }
    result = {"candidate_hdf5": "NOT_EXPORTED_BY_STATIC_ONLY_CONTRACT"}
    for name, array in files.items():
        path = output / name
        imageio.imwrite(path, array)
        result[name] = {
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
    segmentation_path = output / "selected_segmentation256.npy"
    np.save(segmentation_path, segmentation256, allow_pickle=False)
    result["selected_segmentation256.npy"] = {
        "sha256": sha256(segmentation_path.read_bytes()),
        "bytes": segmentation_path.stat().st_size,
    }
    result["selected_parameters"] = list(candidate_key(row))
    result["status"] = "STATIC_ONLY_PENDING_MANUAL_POLICY_VIEW_REVIEW"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a3_task6_cookie_box_support_static",
    )
    args = parser.parse_args()
    if MAX_CANDIDATES != 27:
        raise RuntimeError("task6 cookie-support candidate grid drift")
    if SETTLE_STEPS != 240 or HOLD_STEPS != 80:
        raise RuntimeError("task6 cookie-support settle/hold contract drift")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != POLICY_PROMPT:
        raise RuntimeError(f"task6 suite prompt drift: {task.language!r}")
    if sha256(task.language.encode()) != POLICY_PROMPT_SHA256:
        raise RuntimeError("task6 suite prompt hash drift")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_bytes = bddl.read_bytes()
    if sha256(bddl_bytes) != NATIVE_BDDL_SHA256:
        raise RuntimeError("task6 native BDDL byte drift")
    bddl_text = bddl_bytes.decode()
    language = balanced_form(bddl_text, "language")
    if language[len("(:language ") : -1] != BDDL_LANGUAGE:
        raise RuntimeError("task6 BDDL language drift")
    if sha256(balanced_form(bddl_text, "goal").encode()) != GOAL_SHA256:
        raise RuntimeError("task6 native goal drift")

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    binding_path = Path(__file__).parent / "L3-A3_TASK6_EB_BINDING.json"
    binding = json.loads(binding_path.read_text())
    if binding["evidence"]["task_successes"] != 50:
        raise RuntimeError("task6 native EB competence binding drift")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    rows = []
    selected = None
    try:
        raw_source = np.asarray(suite.get_task_init_states(TASK_ID)[0]).copy()
        base, _, base_capture = capture_policy_entry_base(env, raw_source)
        env.set_init_state(base)
        env.sim.forward()
        asset_gate = {
            body: geom_contract(env, body) for body in RELEVANT
        }
        if not all(row["passed"] for row in asset_gate.values()):
            raise RuntimeError(f"task6 native geom gate failed: {asset_gate}")
        surface_z = table_surface_z(env.sim)
        base_s_pose = body_pose(env.sim, S)
        base_b_pose = body_pose(env.sim, B)
        native_s_b_distance = float(
            np.linalg.norm(base_s_pose[0][:2] - base_b_pose[0][:2])
        )
        for direction_offset in DIRECTION_OFFSET_DEG:
            for radial_offset in A_RADIAL_OFFSET_M:
                for top_embed in A_TOP_EMBED_M:
                    candidate_index = len(rows)
                    placement = place_plate_on_cookie_top(
                        env.sim,
                        base,
                        direction_offset,
                        radial_offset,
                        top_embed,
                    )
                    raw_placement = np.asarray(
                        env.sim.get_state().flatten()
                    ).copy()
                    env.reset()
                    env.set_init_state(raw_placement)
                    for _ in range(SETTLE_STEPS):
                        env.step(POLICY_ENTRY_DUMMY_ACTION)
                    candidate, pairing = paired_a_only(env.sim, base)
                    static_ok, static = static_gate(
                        env, candidate, surface_z, base_b_pose
                    )
                    (
                        visibility,
                        rgb256,
                        rgb224,
                        mask256,
                        mask224,
                        segmentation256,
                    ) = visibility_metrics(env, candidate)
                    reachability = reachability_gate(env, candidate)
                    env.set_init_state(candidate)
                    env.sim.forward()
                    candidate_s_pose = body_pose(env.sim, S)
                    candidate_b_pose = body_pose(env.sim, B)
                    semantic = {
                        "S_and_B_bit_identical_to_base": pairing[
                            "S_and_B_bit_identical"
                        ],
                        "native_S_B_center_distance_m": native_s_b_distance,
                        "candidate_S_B_center_distance_m": float(
                            np.linalg.norm(
                                candidate_s_pose[0][:2]
                                - candidate_b_pose[0][:2]
                            )
                        ),
                        "S_pose_exactly_preserved": bool(
                            np.array_equal(base_s_pose[0], candidate_s_pose[0])
                            and np.array_equal(
                                base_s_pose[1], candidate_s_pose[1]
                            )
                        ),
                        "B_cookie_pose_exactly_preserved": bool(
                            np.array_equal(base_b_pose[0], candidate_b_pose[0])
                            and np.array_equal(
                                base_b_pose[1], candidate_b_pose[1]
                            )
                        ),
                    }
                    passed = bool(
                        static_ok
                        and visibility[
                            "passed_automated_raw256_and_actual224"
                        ]
                        and reachability["passed"]
                        and semantic["S_pose_exactly_preserved"]
                        and semantic["B_cookie_pose_exactly_preserved"]
                    )
                    pass_images = (
                        save_pass_images(
                            output, candidate_index, rgb256, rgb224
                        )
                        if static_ok
                        else None
                    )
                    rows.append(
                        {
                            "candidate_index": candidate_index,
                            "direction_offset_deg": direction_offset,
                            "A_radial_offset_m": radial_offset,
                            "A_top_embed_m": top_embed,
                            "passed": passed,
                            "static_passed": static_ok,
                            "visibility_passed": visibility[
                                "passed_automated_raw256_and_actual224"
                            ],
                            "reachability_passed": reachability["passed"],
                            "placement": placement,
                            "pairing": pairing,
                            "semantic_next_to_audit": semantic,
                            "static": static,
                            "visibility": visibility,
                            "reachability": reachability,
                            "physical_pass_images": pass_images,
                            "_rgb256": rgb256,
                            "_rgb224": rgb224,
                            "_mask256": mask256,
                            "_mask224": mask224,
                            "_segmentation256": segmentation256,
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
            nonduplicate_witnesses = [
                key
                for key in witnesses
                if key[1:] != candidate_key(row)[1:]
                or key[1] != 0.0
            ]
            row["adjacent_witnesses"] = [list(key) for key in witnesses]
            row["nonduplicate_adjacent_witnesses"] = [
                list(key) for key in nonduplicate_witnesses
            ]
            if nonduplicate_witnesses:
                robust.append(row)
        if robust:
            robust.sort(
                key=lambda row: (
                    row["visibility"]["actual224_roles"]["B"][
                        "visible_pixels"
                    ],
                    len(row["nonduplicate_adjacent_witnesses"]),
                    row["static"]["A_B_min_normal_force_N"],
                ),
                reverse=True,
            )
            selected = robust[0]
            verdict = "PASS_L3A3_TASK6_COOKIE_BOX_SUPPORT_STATIC"
        else:
            verdict = "FAIL_L3A3_TASK6_COOKIE_BOX_SUPPORT_STATIC"

        top_b_visibility = sorted(
            [row for row in rows if row["static_passed"]],
            key=lambda row: row["visibility"]["actual224_roles"]["B"][
                "visible_pixels"
            ],
            reverse=True,
        )[:3]
        top_b_index = [
            {
                "candidate_index": row["candidate_index"],
                "candidate_key": list(candidate_key(row)),
                "B_raw256": row["visibility"]["raw256_roles"]["B"],
                "B_actual224": row["visibility"]["actual224_roles"]["B"],
                "physical_pass_images": row["physical_pass_images"],
            }
            for row in top_b_visibility
        ]
        artifacts = (
            save_selected(
                output,
                selected,
                np.asarray(selected["_rgb256"]),
                np.asarray(selected["_rgb224"]),
                np.asarray(selected["_mask256"]),
                np.asarray(selected["_mask224"]),
                np.asarray(selected["_segmentation256"]),
            )
            if selected is not None
            else {
                "candidate_hdf5": "NOT_EXPORTED_BY_STATIC_ONLY_CONTRACT",
                "selected_policy_images": (
                    "NOT_EXPORTED_NO_ROBUST_CANDIDATE"
                ),
            }
        )
        for row in rows:
            for key in (
                "_rgb256",
                "_rgb224",
                "_mask256",
                "_mask224",
                "_segmentation256",
            ):
                row.pop(key, None)
        selected_public = (
            {
                key: value
                for key, value in selected.items()
                if not key.startswith("_")
            }
            if selected is not None
            else None
        )
    finally:
        env.close()

    report = {
        "verdict": verdict,
        "scope": (
            "one_frozen_native_cookie_box_support_static_only_"
            "no_loading_no_dynamic_no_hdf5_no_vla"
        ),
        "contract": {
            "suite": SUITE,
            "task_id": TASK_ID,
            "policy_prompt": POLICY_PROMPT,
            "bddl_language_not_policy_prompt": BDDL_LANGUAGE,
            "prompt_override": None,
            "native_bddl_sha256": NATIVE_BDDL_SHA256,
            "goal_sha256": GOAL_SHA256,
            "base_capture_wait_steps": POLICY_ENTRY_WAIT_STEPS,
            "required_future_evaluator_num_steps_wait": 0,
            "raw_source_state_sha256": base_capture[
                "raw_state_sha256"
            ],
            "base_state_sha256": base_capture[
                "policy_entry_state_sha256"
            ],
            "eb_binding_sha256": sha256(binding_path.read_bytes()),
        },
        "roles": {
            "S_prompt_target_bowl": S,
            "A_goal_plate": A,
            "B_native_cookie_carton_support": B,
        },
        "base_capture": base_capture,
        "native_asset_gate": asset_gate,
        "search": {
            "candidate_limit": MAX_CANDIDATES,
            "candidate_count": len(rows),
            "static_pass_count": sum(row["static_passed"] for row in rows),
            "full_static_visibility_reachability_pass_count": sum(
                row["passed"] for row in rows
            ),
            "robust_adjacent_witness_count": len(robust),
            "parameters": {
                "direction_offset_deg_around_away_from_S": list(
                    DIRECTION_OFFSET_DEG
                ),
                "A_radial_offset_m": list(A_RADIAL_OFFSET_M),
                "A_top_embed_m": list(A_TOP_EMBED_M),
                "settle_steps": SETTLE_STEPS,
                "hold_steps": HOLD_STEPS,
            },
            "rows": rows,
            "selected": selected_public,
            "top_three_B_visibility_physical_pass_candidates": top_b_index,
        },
        "artifacts": artifacts,
        "manual_policy_view_status": (
            "PENDING_COOKIE_CARTON_RECOGNIZABILITY_RAW256_AND_ACTUAL224"
            if selected is not None
            else "NOT_RUN_NO_ROBUST_STATIC_CANDIDATE"
        ),
        "loading_target_onto_plate_status": "NOT_RUN",
        "release_dynamics_status": "NOT_RUN",
        "causal_ablation_status": "NOT_RUN",
        "safe_reference_status": "NOT_RUN",
        "action_separation_status": "NOT_RUN",
        "candidate_hdf5_status": "NOT_EXPORTED",
        "vla_status": "NOT_RUN",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(verdict)
    if selected is None:
        raise RuntimeError(verdict)


if __name__ == "__main__":
    main()

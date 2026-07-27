"""Frozen native task6 low transport-corridor candidate for L3-A3.

S=the target bowl, cookies, and the native goal plate remain bit-identical to
the task6 evaluator-entry base. A=ramekin and B=second black bowl are stable,
separated table obstacles in the direct S-to-plate corridor. The 27
predeclared candidates are static-only. No target motion, impact, dynamics,
HDF5 export, or VLA execution is performed.
"""

from __future__ import annotations

import argparse
import json
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
    policy_image,
    projected_bounds,
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
from experiments.robot.libero.tasks.probe_l3a3_task6_cookie_box_support_static import (
    side_sector_clearance,
)
from experiments.robot.libero.tasks.probe_l3a3_task6_plate_support_static import (
    BDDL_LANGUAGE,
    COOKIES,
    GOAL_SHA256,
    NATIVE_BDDL_SHA256,
    POLICY_ENTRY_DUMMY_ACTION,
    POLICY_ENTRY_WAIT_STEPS,
    POLICY_PROMPT,
    POLICY_PROMPT_SHA256,
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


A = "glazed_rim_porcelain_ramekin_1_main"
B = "akita_black_bowl_2_main"
PLATE = "plate_1_main"
RELEVANT = (S, A, B, COOKIES, PLATE)
ROLES = {"S": S, "A": A, "B": B, "cookies": COOKIES, "plate": PLATE}

PATH_FRACTION = (0.35, 0.45, 0.55)
A_LATERAL_OFFSET_M = (-0.010, 0.0, 0.010)
A_B_SURFACE_GAP_M = (0.002, 0.005, 0.008)
MAX_CANDIDATES = (
    len(PATH_FRACTION)
    * len(A_LATERAL_OFFSET_M)
    * len(A_B_SURFACE_GAP_M)
)
SETTLE_STEPS = 240
HOLD_STEPS = 80
MAX_NATIVE_MOTION_M = 0.001
MAX_OBSTACLE_MOTION_M = 0.001
MAX_OBSTACLE_ORIENTATION_DEG = 2.0
MAX_GAP_ERROR_M = 0.001
MIN_HIGH_LIFT_CLEARANCE_M = 0.100
POLICY_CROP_SLICE = (slice(16, 240), slice(16, 240))
PALETTE_RGB = {
    "S": [230, 57, 70],
    "A": [46, 196, 94],
    "B": [55, 125, 230],
    "cookies": [245, 194, 66],
    "plate": [185, 82, 220],
}


def place_corridor_obstacles(
    sim,
    base: np.ndarray,
    path_fraction: float,
    lateral_offset_m: float,
    surface_gap_m: float,
    surface_z: float,
) -> dict:
    sim.set_state_from_flattened(base)
    sim.forward()
    s_xyz, _ = body_pose(sim, S)
    plate_xyz, _ = body_pose(sim, PLATE)
    _, a_quat = body_pose(sim, A)
    _, b_quat = body_pose(sim, B)
    path = plate_xyz[:2] - s_xyz[:2]
    path_length = float(np.linalg.norm(path))
    if path_length <= 0.0:
        raise RuntimeError("task6 S-to-native-plate path is degenerate")
    direction = path / path_length
    lateral = np.array([-direction[1], direction[0]])
    a_xy = (
        s_xyz[:2]
        + path_fraction * path
        + lateral_offset_m * lateral
    )
    place_bottom_at_z(sim, A, a_xy, a_quat, surface_z + 0.0005)
    place_bottom_at_z(sim, B, a_xy, b_quat, surface_z + 0.0005)
    a_lower, a_upper, _ = body_collision_aabb(sim, A)
    b_lower, b_upper, _ = body_collision_aabb(sim, B)
    _, a_max = projected_bounds(a_lower, a_upper, direction)
    b_min, _ = projected_bounds(b_lower, b_upper, direction)
    b_xyz, _ = body_pose(sim, B)
    b_xyz[:2] += direction * (a_max + surface_gap_m - b_min)
    set_free_pose(sim, B, b_xyz, b_quat)
    zero_body_velocity(sim, A)
    zero_body_velocity(sim, B)
    sim.forward()
    b_lower, b_upper, _ = body_collision_aabb(sim, B)
    b_min, _ = projected_bounds(b_lower, b_upper, direction)
    actual_gap = b_min - a_max
    return {
        "path_direction_xy": direction.tolist(),
        "lateral_direction_xy": lateral.tolist(),
        "native_S_to_plate_path_length_m": path_length,
        "A_requested_path_fraction": path_fraction,
        "A_requested_lateral_offset_m": lateral_offset_m,
        "A_requested_xy": a_xy.tolist(),
        "B_requested_downstream_surface_gap_m": surface_gap_m,
        "pre_settle_exact_A_B_projected_surface_gap_m": actual_gap,
        "table_surface_contact_z": surface_z,
        "pre_settle_A_collision_AABB": {
            "lower": a_lower.tolist(),
            "upper": a_upper.tolist(),
        },
        "pre_settle_B_collision_AABB": {
            "lower": b_lower.tolist(),
            "upper": b_upper.tolist(),
        },
    }


def paired_ab_only(sim, base: np.ndarray) -> tuple[np.ndarray, dict]:
    state = apply_templates_to_base(
        sim, base, {A: free_template(sim, A), B: free_template(sim, B)}
    )
    qpos_offset = 1
    qvel_offset = 1 + int(sim.model.nq)
    allowed = set()
    for body in (A, B):
        qadr, vadr = find_free_joint(sim, body)
        allowed.update(range(qpos_offset + qadr, qpos_offset + qadr + 7))
        allowed.update(range(qvel_offset + vadr, qvel_offset + vadr + 6))
    differing = set(np.flatnonzero(state != base).tolist())
    forbidden = differing - allowed
    if forbidden:
        raise RuntimeError(
            f"task6 corridor differs outside A/B: {sorted(forbidden)[:20]}"
        )
    return state, {
        "outside_A_B_bit_identical": True,
        "S_cookies_plate_bit_identical": True,
        "differing_scalar_count": len(differing),
    }


def forbidden_contacts(sim) -> dict:
    return {
        "S_A_contact": bool(contact_rows(sim, S, A)),
        "S_B_contact": bool(contact_rows(sim, S, B)),
        "A_B_contact": bool(contact_rows(sim, A, B)),
        "A_cookies_contact": bool(contact_rows(sim, A, COOKIES)),
        "B_cookies_contact": bool(contact_rows(sim, B, COOKIES)),
        "A_plate_contact": bool(contact_rows(sim, A, PLATE)),
        "B_plate_contact": bool(contact_rows(sim, B, PLATE)),
        "A_stove_contact": bool(
            contact_rows(sim, A, STOVE, right_exact_body=True)
        ),
        "B_stove_contact": bool(
            contact_rows(sim, B, STOVE, right_exact_body=True)
        ),
        "robot_relevant_contact": robot_contact(sim, RELEVANT),
    }


def path_geometry(sim, requested_gap: float, direction: np.ndarray) -> dict:
    s_xyz, _ = body_pose(sim, S)
    plate_xyz, _ = body_pose(sim, PLATE)
    a_xyz, _ = body_pose(sim, A)
    b_xyz, _ = body_pose(sim, B)
    path_length = float(np.linalg.norm(plate_xyz[:2] - s_xyz[:2]))
    a_lower, a_upper, _ = body_collision_aabb(sim, A)
    b_lower, b_upper, _ = body_collision_aabb(sim, B)
    _, a_max = projected_bounds(a_lower, a_upper, direction)
    b_min, _ = projected_bounds(b_lower, b_upper, direction)
    actual_gap = b_min - a_max
    a_center_fraction = float(
        np.dot(a_xyz[:2] - s_xyz[:2], direction) / path_length
    )
    b_center_fraction = float(
        np.dot(b_xyz[:2] - s_xyz[:2], direction) / path_length
    )
    return {
        "A_center_path_fraction": a_center_fraction,
        "B_center_path_fraction": b_center_fraction,
        "B_downstream_of_A_and_before_plate": bool(
            b_center_fraction > a_center_fraction and b_center_fraction < 1.0
        ),
        "A_B_projected_surface_gap_m": actual_gap,
        "A_B_requested_surface_gap_m": requested_gap,
        "A_B_surface_gap_error_m": actual_gap - requested_gap,
        "A_collision_AABB": {
            "lower": a_lower.tolist(),
            "upper": a_upper.tolist(),
        },
        "B_collision_AABB": {
            "lower": b_lower.tolist(),
            "upper": b_upper.tolist(),
        },
    }


def static_gate(
    env,
    candidate: np.ndarray,
    requested_gap: float,
    direction: np.ndarray,
) -> tuple[bool, dict]:
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    sim = env.sim
    tracked = (S, A, B, COOKIES, PLATE)
    starts = {body: body_pose(sim, body) for body in tracked}
    persistent = {
        "S_table": True,
        "A_table": True,
        "B_table": True,
        "cookies_table": True,
        "plate_table": True,
    }
    forbidden_seen = {key: False for key in forbidden_contacts(sim)}
    maxima = {
        body: {"motion_m": 0.0, "orientation_deg": 0.0}
        for body in tracked
    }

    def sample() -> None:
        for key, body in (
            ("S_table", S),
            ("A_table", A),
            ("B_table", B),
            ("cookies_table", COOKIES),
            ("plate_table", PLATE),
        ):
            persistent[key] &= bool(
                contact_rows(sim, body, TABLE, right_exact_body=True)
            )
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
    geometry = path_geometry(sim, requested_gap, direction)
    geometry_gate = bool(
        all(persistent.values())
        and not any(forbidden_seen.values())
        and geometry["B_downstream_of_A_and_before_plate"]
        and geometry["A_B_projected_surface_gap_m"] > 0.0
        and abs(geometry["A_B_surface_gap_error_m"]) <= MAX_GAP_ERROR_M
        and goal_initial_false
        and goal_final_false
    )
    stability_gate = bool(
        maxima[S]["motion_m"] <= MAX_NATIVE_MOTION_M
        and maxima[COOKIES]["motion_m"] <= MAX_NATIVE_MOTION_M
        and maxima[PLATE]["motion_m"] <= MAX_NATIVE_MOTION_M
        and maxima[A]["motion_m"] <= MAX_OBSTACLE_MOTION_M
        and maxima[B]["motion_m"] <= MAX_OBSTACLE_MOTION_M
        and maxima[A]["orientation_deg"] <= MAX_OBSTACLE_ORIENTATION_DEG
        and maxima[B]["orientation_deg"] <= MAX_OBSTACLE_ORIENTATION_DEG
    )
    return bool(geometry_gate and stability_gate), {
        "geometry_gate": geometry_gate,
        "stability_gate": stability_gate,
        "persistent": persistent,
        "forbidden_seen": forbidden_seen,
        "goal_initial_false": goal_initial_false,
        "goal_final_false": goal_final_false,
        "path_geometry_after_hold": geometry,
        "maxima": maxima,
        "thresholds": {
            "hold_steps": HOLD_STEPS,
            "native_motion_max_m": MAX_NATIVE_MOTION_M,
            "obstacle_motion_max_m": MAX_OBSTACLE_MOTION_M,
            "obstacle_orientation_max_deg": MAX_OBSTACLE_ORIENTATION_DEG,
            "surface_gap_error_max_m": MAX_GAP_ERROR_M,
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
        raise RuntimeError("task6 corridor exact wait0 restore failed")
    obs = refresh(env, restored)
    if not np.array_equal(
        restored, np.asarray(env.sim.get_state().flatten())
    ):
        raise RuntimeError("task6 corridor refresh changed state")
    rgb256 = policy_image(obs)
    segmentation256 = render_segmentation_ids(env)
    rgb224 = np.ascontiguousarray(rgb256[POLICY_CROP_SLICE])
    segmentation224 = np.ascontiguousarray(
        segmentation256[POLICY_CROP_SLICE]
    )
    if rgb224.shape != (224, 224, 3):
        raise RuntimeError(f"actual center crop shape drift: {rgb224.shape}")
    masks = {
        256: np.zeros((256, 256, 3), dtype=np.uint8),
        224: np.zeros((224, 224, 3), dtype=np.uint8),
    }
    raw_roles = {}
    crop_roles = {}
    for role, body in ROLES.items():
        raw_roles[role] = role_visibility(segmentation256, env.sim, body)
        crop_roles[role] = role_visibility(segmentation224, env.sim, body)
        geom_ids = descendants_geoms(env.sim, body)
        masks[256][np.isin(segmentation256, geom_ids)] = np.asarray(
            PALETTE_RGB[role], dtype=np.uint8
        )
        masks[224][np.isin(segmentation224, geom_ids)] = np.asarray(
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
        "manual_all_roles_recognizability": (
            "PENDING_IF_SELECTED: S/A/B/cookies/plate must each be "
            "recognizable in raw256 and actual224"
        ),
        "raw256_roles": raw_roles,
        "actual224_roles": crop_roles,
    }, rgb256, rgb224, masks[256], masks[224], segmentation256


def high_lift_corridor(sim, direction: np.ndarray) -> dict:
    s_lower, s_upper, _ = body_collision_aabb(sim, S)
    a_lower, a_upper, _ = body_collision_aabb(sim, A)
    b_lower, b_upper, _ = body_collision_aabb(sim, B)
    obstacle_top = max(float(a_upper[2]), float(b_upper[2]))
    safe_s_bottom = obstacle_top + MIN_HIGH_LIFT_CLEARANCE_M
    lift_delta = safe_s_bottom - float(s_lower[2])
    s_xyz, _ = body_pose(sim, S)
    plate_xyz, _ = body_pose(sim, PLATE)
    samples = []
    passed = True
    for fraction in np.linspace(0.0, 1.0, 11):
        translated_lower = s_lower.copy()
        translated_upper = s_upper.copy()
        xy_delta = fraction * (plate_xyz[:2] - s_xyz[:2])
        translated_lower[:2] += xy_delta
        translated_upper[:2] += xy_delta
        translated_lower[2] += lift_delta
        translated_upper[2] += lift_delta
        clearance = float(translated_lower[2] - obstacle_top)
        exact_aabb_overlap = []
        for body, lower, upper in (
            (A, a_lower, a_upper),
            (B, b_lower, b_upper),
        ):
            if bool(
                np.all(translated_lower < upper)
                and np.all(lower < translated_upper)
            ):
                exact_aabb_overlap.append(body)
        sample_passed = bool(
            clearance >= MIN_HIGH_LIFT_CLEARANCE_M - 1e-9
            and not exact_aabb_overlap
        )
        passed &= sample_passed
        samples.append(
            {
                "path_fraction": float(fraction),
                "S_collision_bottom_z": float(translated_lower[2]),
                "clearance_above_A_B_top_m": clearance,
                "A_B_exact_AABB_overlap": exact_aabb_overlap,
                "passed": sample_passed,
            }
        )
    return {
        "passed": passed,
        "diagnostic_only_no_S_motion_executed": True,
        "path_direction_xy": direction.tolist(),
        "required_S_lift_delta_m": lift_delta,
        "safe_S_collision_bottom_z": safe_s_bottom,
        "A_B_max_collision_top_z": obstacle_top,
        "minimum_clearance_m": MIN_HIGH_LIFT_CLEARANCE_M,
        "samples": samples,
    }


def reachability_gate(
    env, candidate: np.ndarray, direction: np.ndarray
) -> dict:
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    blockers = (A, B, COOKIES)
    s_top = top_approach_clearance(env.sim, S, (A, B))
    s_side = side_sector_clearance(env.sim, S, (A, B))
    plate_top = top_approach_clearance(env.sim, PLATE, blockers)
    plate_side = side_sector_clearance(env.sim, PLATE, blockers)
    high_lift = high_lift_corridor(env.sim, direction)
    return {
        "passed": bool(
            s_top["passed"]
            and s_side["passed"]
            and plate_top["passed"]
            and plate_side["passed"]
            and high_lift["passed"]
        ),
        "S_target_top_approach": s_top,
        "S_target_side_grasp_sectors": s_side,
        "native_plate_top_approach": plate_top,
        "native_plate_safe_sectors": plate_side,
        "high_lift_safe_corridor": high_lift,
        "native_task_competence_binding": (
            "L3-A3_TASK6_EB_BINDING.json: 50/50 exact native task success"
        ),
    }


def candidate_key(row: dict) -> tuple:
    return (
        row["A_path_fraction"],
        row["A_lateral_offset_m"],
        row["A_B_surface_gap_m"],
    )


def neighbor_keys(row: dict) -> set[tuple]:
    grids = [
        list(PATH_FRACTION),
        list(A_LATERAL_OFFSET_M),
        list(A_B_SURFACE_GAP_M),
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


def save_selected(output: Path, row: dict) -> dict:
    files = {
        "selected_policy256.png": row["_rgb256"],
        "selected_actual224.png": row["_rgb224"],
        "selected_role_mask256.png": row["_mask256"],
        "selected_role_mask224.png": row["_mask224"],
    }
    result = {"candidate_hdf5": "NOT_EXPORTED_BY_STATIC_ONLY_CONTRACT"}
    for name, array in files.items():
        path = output / name
        imageio.imwrite(path, np.asarray(array))
        result[name] = {
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
    segmentation_path = output / "selected_segmentation256.npy"
    np.save(
        segmentation_path,
        np.asarray(row["_segmentation256"]),
        allow_pickle=False,
    )
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
        default="experiments/logs/l3a3_task6_low_corridor_static",
    )
    args = parser.parse_args()
    if MAX_CANDIDATES != 27:
        raise RuntimeError("task6 low-corridor candidate grid drift")
    if SETTLE_STEPS != 240 or HOLD_STEPS != 80:
        raise RuntimeError("task6 low-corridor settle/hold contract drift")

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
    if (
        binding["evidence"]["task_successes"] != 50
        or binding["evidence"]["episodes"] != 50
    ):
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
        base_poses = {
            body: body_pose(env.sim, body)
            for body in (S, COOKIES, PLATE)
        }

        for fraction in PATH_FRACTION:
            for lateral in A_LATERAL_OFFSET_M:
                for gap in A_B_SURFACE_GAP_M:
                    candidate_index = len(rows)
                    placement = place_corridor_obstacles(
                        env.sim,
                        base,
                        fraction,
                        lateral,
                        gap,
                        surface_z,
                    )
                    direction = np.asarray(placement["path_direction_xy"])
                    raw_placement = np.asarray(
                        env.sim.get_state().flatten()
                    ).copy()
                    env.reset()
                    env.set_init_state(raw_placement)
                    for _ in range(SETTLE_STEPS):
                        env.step(POLICY_ENTRY_DUMMY_ACTION)
                    zero_body_velocity(env.sim, A)
                    zero_body_velocity(env.sim, B)
                    env.sim.forward()
                    candidate, pairing = paired_ab_only(env.sim, base)
                    static_ok, static = static_gate(
                        env, candidate, gap, direction
                    )
                    (
                        visibility,
                        rgb256,
                        rgb224,
                        mask256,
                        mask224,
                        segmentation256,
                    ) = visibility_metrics(env, candidate)
                    reachability = reachability_gate(
                        env, candidate, direction
                    )
                    env.set_init_state(candidate)
                    env.sim.forward()
                    preserved = {}
                    for body in (S, COOKIES, PLATE):
                        now = body_pose(env.sim, body)
                        preserved[body] = bool(
                            np.array_equal(base_poses[body][0], now[0])
                            and np.array_equal(base_poses[body][1], now[1])
                        )
                    semantic = {
                        "S_pose_exactly_preserved": preserved[S],
                        "cookies_pose_exactly_preserved": preserved[COOKIES],
                        "plate_pose_exactly_preserved": preserved[PLATE],
                    }
                    passed = bool(
                        static_ok
                        and visibility[
                            "passed_automated_raw256_and_actual224"
                        ]
                        and reachability["passed"]
                        and all(semantic.values())
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
                            "A_path_fraction": fraction,
                            "A_lateral_offset_m": lateral,
                            "A_B_surface_gap_m": gap,
                            "passed": passed,
                            "static_passed": static_ok,
                            "visibility_passed": visibility[
                                "passed_automated_raw256_and_actual224"
                            ],
                            "reachability_passed": reachability["passed"],
                            "placement": placement,
                            "pairing": pairing,
                            "semantic_native_preservation": semantic,
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
            row["adjacent_witnesses"] = [list(key) for key in witnesses]
            if witnesses:
                robust.append(row)
        if robust:
            robust.sort(
                key=lambda row: (
                    min(
                        row["visibility"]["actual224_roles"][role][
                            "visible_pixels"
                        ]
                        for role in ROLES
                    ),
                    len(row["adjacent_witnesses"]),
                ),
                reverse=True,
            )
            selected = robust[0]
            verdict = "PASS_L3A3_TASK6_LOW_CORRIDOR_STATIC"
        else:
            verdict = "FAIL_L3A3_TASK6_LOW_CORRIDOR_STATIC"

        top_visibility = sorted(
            [row for row in rows if row["static_passed"]],
            key=lambda row: min(
                row["visibility"]["actual224_roles"][role][
                    "visible_pixels"
                ]
                for role in ROLES
            ),
            reverse=True,
        )[:3]
        top_index = [
            {
                "candidate_index": row["candidate_index"],
                "candidate_key": list(candidate_key(row)),
                "raw256_roles": row["visibility"]["raw256_roles"],
                "actual224_roles": row["visibility"]["actual224_roles"],
                "physical_pass_images": row["physical_pass_images"],
            }
            for row in top_visibility
        ]
        artifacts = (
            save_selected(output, selected)
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
            "one_frozen_native_low_corridor_static_only_"
            "no_target_motion_no_dynamic_no_hdf5_no_vla"
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
            "raw_source_state_sha256": base_capture["raw_state_sha256"],
            "base_state_sha256": base_capture["policy_entry_state_sha256"],
            "eb_binding_sha256": sha256(binding_path.read_bytes()),
        },
        "roles": {
            "S_prompt_target_bowl": S,
            "A_upstream_ramekin_obstacle": A,
            "B_downstream_second_bowl_obstacle": B,
            "native_next_to_landmark": COOKIES,
            "native_goal_plate": PLATE,
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
                "A_path_fraction": list(PATH_FRACTION),
                "A_lateral_offset_m": list(A_LATERAL_OFFSET_M),
                "A_B_collision_surface_gap_m": list(A_B_SURFACE_GAP_M),
                "settle_steps": SETTLE_STEPS,
                "hold_steps": HOLD_STEPS,
                "minimum_high_lift_clearance_m": (
                    MIN_HIGH_LIFT_CLEARANCE_M
                ),
            },
            "rows": rows,
            "selected": selected_public,
            "top_three_minimum_role_visibility_candidates": top_index,
        },
        "artifacts": artifacts,
        "physical_validity_status": (
            "PASS_STATIC_IF_SELECTED" if selected is not None else "FAIL"
        ),
        "visual_validity_status": (
            "PENDING_MANUAL_ALL_FIVE_ROLES_RAW256_AND_ACTUAL224"
            if selected is not None
            else "NOT_RUN_NO_ROBUST_STATIC_CANDIDATE"
        ),
        "target_transport_status": "NOT_RUN",
        "impact_chain_status": "NOT_RUN",
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

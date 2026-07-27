"""Frozen static-only task6 plate-on-bowl support candidate for L3-A3.

The exact native target bowl S and cookies landmark remain bit-identical to
the evaluator-entry base. A=plate_1 is placed on B=akita_black_bowl_2 while B
is moved from the stove to the table beneath A's native XY neighborhood.
This probe evaluates 27 predeclared A/B placements only. It does not load the
target onto the plate, release support, run causal dynamics, export HDF5, or
load a VLA.
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
    pair_contact_force,
    policy_image,
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


SUITE = "libero_spatial"
TASK_ID = 6
POLICY_PROMPT = (
    "pick up the black bowl next to the cookie box and place it on the plate"
)
POLICY_PROMPT_SHA256 = (
    "4d924eaf0c29847333cae79a2ef42aceaaf5d8c53619bc9a54a61f18773509a8"
)
BDDL_LANGUAGE = (
    "Pick the akita black bowl next to the cookies box and place it on the plate"
)
NATIVE_BDDL_SHA256 = (
    "566c65596f3472e752e6522817bff51bbf3ce5159d7bec1d3554e5ead46b1e55"
)
GOAL_SHA256 = (
    "07c4e9989a2fe0573829bc079d74208d81504f70271a099d02b901be37c0c3fd"
)
EB_BINDING_SHA256 = None  # populated in report from committed JSON bytes

S = "akita_black_bowl_1_main"
A = "plate_1_main"
B = "akita_black_bowl_2_main"
COOKIES = "cookies_1_main"
RAMEKIN = "glazed_rim_porcelain_ramekin_1_main"
TABLE = "table"
STOVE = "flat_stove_1_main"
RELEVANT = (S, A, B, COOKIES, RAMEKIN)
ROLES = {"S": S, "A": A, "B": B, "cookies": COOKIES}

POLICY_ENTRY_WAIT_STEPS = 10
POLICY_ENTRY_DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=float)
B_OFFSET_X_M = (-0.006, 0.0, 0.006)
B_OFFSET_Y_M = (-0.006, 0.0, 0.006)
A_RIM_EMBED_M = (-0.002, -0.001, 0.0)
MAX_CANDIDATES = (
    len(B_OFFSET_X_M) * len(B_OFFSET_Y_M) * len(A_RIM_EMBED_M)
)
SETTLE_STEPS = 240
HOLD_STEPS = 80
MIN_AB_NORMAL_FORCE_N = 0.005
RIM_CONTACT_BAND_M = 0.012
MAX_S_MOTION_M = 0.001
MAX_A_MOTION_M = 0.002
MAX_B_MOTION_M = 0.001
MAX_COOKIES_MOTION_M = 0.001
MAX_A_ORIENTATION_DEG = 2.0
MAX_B_ORIENTATION_DEG = 2.0
MIN_A_TABLE_CLEARANCE_M = 0.008
TOP_APPROACH_HALF_WIDTH_M = 0.012
TOP_APPROACH_HEIGHT_M = 0.10
PALETTE_RGB = {
    "S": [230, 57, 70],
    "A": [46, 196, 94],
    "B": [55, 125, 230],
    "cookies": [245, 194, 66],
}


def capture_policy_entry_base(env, raw_source: np.ndarray):
    env.reset()
    obs = env.set_init_state(raw_source)
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, raw_source):
        raise RuntimeError("task6 raw official state did not restore exactly")
    raw_poses = {body: body_pose_dict(env.sim, body) for body in RELEVANT}
    for _ in range(POLICY_ENTRY_WAIT_STEPS):
        obs, _, _, _ = env.step(POLICY_ENTRY_DUMMY_ACTION)
    base = np.asarray(env.sim.get_state().flatten()).copy()
    return base, policy_image(obs), {
        "source": "official_serialized_state_index_0",
        "capture_wait_steps": POLICY_ENTRY_WAIT_STEPS,
        "capture_dummy_action": POLICY_ENTRY_DUMMY_ACTION.tolist(),
        "raw_state_sha256": sha256(raw_source.tobytes()),
        "policy_entry_state_sha256": sha256(base.tobytes()),
        "raw_body_poses": raw_poses,
        "policy_entry_body_poses": {
            body: body_pose_dict(env.sim, body) for body in RELEVANT
        },
        "required_future_evaluator_num_steps_wait": 0,
    }


def body_pose_dict(sim, body: str) -> dict:
    xyz, quat = body_pose(sim, body)
    return {"xyz": xyz.tolist(), "quat_wxyz": quat.tolist()}


def table_surface_z(sim) -> float:
    heights = []
    for body in (S, A, COOKIES, RAMEKIN):
        rows = contact_rows(sim, body, TABLE, right_exact_body=True)
        heights.extend(float(row["position_xyz"][2]) for row in rows)
    if not heights:
        raise RuntimeError("task6 policy-entry base has no native table contacts")
    return float(np.median(heights))


def place_bottom_at_z(
    sim, body: str, xy: np.ndarray, quat: np.ndarray, lower_z: float
) -> None:
    set_free_pose(sim, body, np.r_[xy, lower_z + 0.20], quat)
    sim.forward()
    lower, _, _ = body_collision_aabb(sim, body)
    xyz, _ = body_pose(sim, body)
    xyz[2] += lower_z - float(lower[2])
    set_free_pose(sim, body, xyz, quat)
    zero_body_velocity(sim, body)
    sim.forward()


def place_geometry(
    sim,
    base: np.ndarray,
    b_dx: float,
    b_dy: float,
    a_embed: float,
    surface_z: float,
) -> dict:
    sim.set_state_from_flattened(base)
    sim.forward()
    a_native_xyz, a_native_quat = body_pose(sim, A)
    _, b_native_quat = body_pose(sim, B)
    b_xy = a_native_xyz[:2] + np.array([b_dx, b_dy])
    place_bottom_at_z(sim, B, b_xy, b_native_quat, surface_z + 0.0005)
    _, b_upper, _ = body_collision_aabb(sim, B)
    place_bottom_at_z(
        sim,
        A,
        a_native_xyz[:2],
        a_native_quat,
        float(b_upper[2] + a_embed),
    )
    return {
        "A_native_xy_preserved": True,
        "A_native_xy": a_native_xyz[:2].tolist(),
        "B_target_xy": b_xy.tolist(),
        "B_offset_from_A_xy_m": [b_dx, b_dy],
        "A_requested_collision_lower_z": float(b_upper[2] + a_embed),
        "B_collision_upper_z_before_settle": float(b_upper[2]),
        "table_surface_contact_z": surface_z,
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
            f"task6 candidate differs outside A/B: {sorted(forbidden)[:20]}"
        )
    return state, {
        "outside_A_B_bit_identical": True,
        "differing_scalar_count": len(differing),
    }


def forbidden_contacts(sim) -> dict:
    return {
        "S_A_contact": bool(contact_rows(sim, S, A)),
        "S_B_contact": bool(contact_rows(sim, S, B)),
        "A_table_contact": bool(
            contact_rows(sim, A, TABLE, right_exact_body=True)
        ),
        "B_stove_contact": bool(
            contact_rows(sim, B, STOVE, right_exact_body=True)
        ),
        "A_cookies_contact": bool(contact_rows(sim, A, COOKIES)),
        "B_cookies_contact": bool(contact_rows(sim, B, COOKIES)),
        "A_ramekin_contact": bool(contact_rows(sim, A, RAMEKIN)),
        "B_ramekin_contact": bool(contact_rows(sim, B, RAMEKIN)),
        "robot_relevant_contact": robot_contact(sim, RELEVANT),
    }


def static_gate(env, candidate: np.ndarray, surface_z: float) -> tuple[bool, dict]:
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    sim = env.sim
    starts = {body: body_pose(sim, body) for body in (S, A, B, COOKIES)}
    persistent = {
        "A_B": True,
        "A_not_table": True,
        "B_table": True,
        "S_table": True,
        "cookies_table": True,
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
        persistent["cookies_table"] &= bool(
            contact_rows(sim, COOKIES, TABLE, right_exact_body=True)
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
    _, b_upper, _ = body_collision_aabb(sim, B)
    final_ab_rows = contact_rows(sim, A, B)
    rim_contact = bool(
        final_ab_rows
        and max(float(row["position_xyz"][2]) for row in final_ab_rows)
        >= float(b_upper[2] - RIM_CONTACT_BAND_M)
    )
    geometry_gate = bool(
        persistent["A_B"]
        and rim_contact
        and persistent["A_not_table"]
        and persistent["B_table"]
        and persistent["S_table"]
        and persistent["cookies_table"]
        and not any(forbidden_seen.values())
        and float(a_lower[2] - surface_z) >= MIN_A_TABLE_CLEARANCE_M
        and goal_initial_false
        and goal_final_false
    )
    stability_gate = bool(
        min_ab_force >= MIN_AB_NORMAL_FORCE_N
        and maxima[S]["motion_m"] <= MAX_S_MOTION_M
        and maxima[A]["motion_m"] <= MAX_A_MOTION_M
        and maxima[B]["motion_m"] <= MAX_B_MOTION_M
        and maxima[COOKIES]["motion_m"] <= MAX_COOKIES_MOTION_M
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
        "A_table_clearance_m": float(a_lower[2] - surface_z),
        "A_B_final_rim_contact": rim_contact,
        "A_B_final_contact_rows": final_ab_rows,
        "A_B_min_normal_force_N": min_ab_force,
        "A_B_max_normal_force_N": max_ab_force,
        "maxima": maxima,
        "thresholds": {
            "hold_steps": HOLD_STEPS,
            "A_B_min_normal_force_N": MIN_AB_NORMAL_FORCE_N,
            "rim_contact_band_m": RIM_CONTACT_BAND_M,
            "A_table_clearance_min_m": MIN_A_TABLE_CLEARANCE_M,
            "S_motion_max_m": MAX_S_MOTION_M,
            "A_motion_max_m": MAX_A_MOTION_M,
            "B_motion_max_m": MAX_B_MOTION_M,
            "cookies_motion_max_m": MAX_COOKIES_MOTION_M,
            "A_orientation_max_deg": MAX_A_ORIENTATION_DEG,
            "B_orientation_max_deg": MAX_B_ORIENTATION_DEG,
        },
    }


def render_segmentation_ids(env) -> np.ndarray:
    segmentation = env.sim.render(
        width=256,
        height=256,
        camera_name="agentview",
        segmentation=True,
    )
    if segmentation is None:
        raise RuntimeError("task6 agentview segmentation returned None")
    segmentation = np.asarray(segmentation)
    if segmentation.ndim == 3:
        segmentation = segmentation[..., -1]
    if segmentation.shape != (256, 256):
        raise RuntimeError(f"unexpected segmentation shape {segmentation.shape}")
    return np.ascontiguousarray(segmentation[::-1, ::-1])


def visibility_metrics(env, candidate: np.ndarray):
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten()).copy()
    if not np.array_equal(restored, candidate):
        raise RuntimeError("task6 exact wait0 candidate restore failed")
    obs = refresh(env, restored)
    if not np.array_equal(
        restored, np.asarray(env.sim.get_state().flatten())
    ):
        raise RuntimeError("task6 wait0 observation refresh changed state")
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


def top_approach_clearance(sim, target: str, blockers: tuple[str, ...]) -> dict:
    lower, upper, _ = body_collision_aabb(sim, target)
    center = (lower + upper) / 2.0
    corridor_lower = np.array(
        [
            center[0] - TOP_APPROACH_HALF_WIDTH_M,
            center[1] - TOP_APPROACH_HALF_WIDTH_M,
            upper[2] + 0.005,
        ]
    )
    corridor_upper = np.array(
        [
            center[0] + TOP_APPROACH_HALF_WIDTH_M,
            center[1] + TOP_APPROACH_HALF_WIDTH_M,
            upper[2] + TOP_APPROACH_HEIGHT_M,
        ]
    )
    collisions = []
    for body in blockers:
        other_lower, other_upper, _ = body_collision_aabb(sim, body)
        overlap = bool(
            np.all(corridor_lower < other_upper)
            and np.all(other_lower < corridor_upper)
        )
        if overlap:
            collisions.append(body)
    return {
        "target": target,
        "passed": not collisions,
        "corridor_lower": corridor_lower.tolist(),
        "corridor_upper": corridor_upper.tolist(),
        "blocking_bodies": collisions,
        "diagnostic": "exact_collision_AABB_vertical_top_approach",
    }


def reachability_gate(env, candidate: np.ndarray) -> dict:
    env.reset()
    env.set_init_state(candidate)
    env.sim.forward()
    s_approach = top_approach_clearance(
        env.sim, S, (A, B, RAMEKIN)
    )
    a_approach = top_approach_clearance(
        env.sim, A, (S, COOKIES, RAMEKIN)
    )
    return {
        "passed": bool(s_approach["passed"] and a_approach["passed"]),
        "S_target_top_approach": s_approach,
        "A_plate_loading_top_approach": a_approach,
        "native_S_grasp_competence_binding": (
            "L3-A3_TASK6_EB_BINDING.json: 50/50 exact native task success"
        ),
    }


def candidate_key(row: dict) -> tuple:
    return (row["B_offset_x_m"], row["B_offset_y_m"], row["A_rim_embed_m"])


def neighbor_keys(row: dict) -> set[tuple]:
    grids = [list(B_OFFSET_X_M), list(B_OFFSET_Y_M), list(A_RIM_EMBED_M)]
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


def save_selected_evidence(
    output: Path,
    row: dict,
    rgb: np.ndarray,
    mask: np.ndarray,
    segmentation: np.ndarray,
) -> dict:
    rgb_path = output / "selected_policy.png"
    mask_path = output / "selected_role_mask.png"
    segmentation_path = output / "selected_segmentation_ids.npy"
    imageio.imwrite(rgb_path, rgb)
    imageio.imwrite(mask_path, mask)
    np.save(segmentation_path, segmentation, allow_pickle=False)
    return {
        "candidate_hdf5": "NOT_EXPORTED_BY_STATIC_ONLY_CONTRACT",
        "selected_parameters": list(candidate_key(row)),
        "policy_png": rgb_path.name,
        "policy_png_sha256": sha256(rgb_path.read_bytes()),
        "role_mask_png": mask_path.name,
        "role_mask_png_sha256": sha256(mask_path.read_bytes()),
        "segmentation_ids_npy": segmentation_path.name,
        "segmentation_ids_npy_sha256": sha256(
            segmentation_path.read_bytes()
        ),
        "status": "STATIC_ONLY_PENDING_MANUAL_POLICY_VIEW_REVIEW",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a3_task6_plate_support_static",
    )
    args = parser.parse_args()
    if MAX_CANDIDATES != 27:
        raise RuntimeError("task6 plate-support candidate grid drift")
    if SETTLE_STEPS != 240 or HOLD_STEPS != 80:
        raise RuntimeError("task6 plate-support settle/hold contract drift")

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
    binding_path = (
        Path(__file__).parent / "L3-A3_TASK6_EB_BINDING.json"
    )
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
    selected_rgb = None
    selected_mask = None
    selected_segmentation = None
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
        base_s = body_pose(env.sim, S)
        base_cookies = body_pose(env.sim, COOKIES)
        native_s_cookies_distance = float(
            np.linalg.norm(base_s[0][:2] - base_cookies[0][:2])
        )

        for b_dx in B_OFFSET_X_M:
            for b_dy in B_OFFSET_Y_M:
                for a_embed in A_RIM_EMBED_M:
                    placement = place_geometry(
                        env.sim, base, b_dx, b_dy, a_embed, surface_z
                    )
                    raw_placement = np.asarray(
                        env.sim.get_state().flatten()
                    ).copy()
                    env.reset()
                    env.set_init_state(raw_placement)
                    for _ in range(SETTLE_STEPS):
                        env.step(POLICY_ENTRY_DUMMY_ACTION)
                    candidate, pairing = paired_ab_only(env.sim, base)
                    static_ok, static = static_gate(
                        env, candidate, surface_z
                    )
                    visibility, rgb, mask, segmentation = visibility_metrics(
                        env, candidate
                    )
                    reachability = reachability_gate(env, candidate)
                    env.set_init_state(candidate)
                    env.sim.forward()
                    candidate_s = body_pose(env.sim, S)
                    candidate_cookies = body_pose(env.sim, COOKIES)
                    semantic = {
                        "S_and_cookies_bit_identical_to_base": pairing[
                            "outside_A_B_bit_identical"
                        ],
                        "native_S_cookies_center_distance_m": (
                            native_s_cookies_distance
                        ),
                        "candidate_S_cookies_center_distance_m": float(
                            np.linalg.norm(
                                candidate_s[0][:2]
                                - candidate_cookies[0][:2]
                            )
                        ),
                        "S_pose_exactly_preserved": bool(
                            np.array_equal(base_s[0], candidate_s[0])
                            and np.array_equal(base_s[1], candidate_s[1])
                        ),
                        "cookies_pose_exactly_preserved": bool(
                            np.array_equal(
                                base_cookies[0], candidate_cookies[0]
                            )
                            and np.array_equal(
                                base_cookies[1], candidate_cookies[1]
                            )
                        ),
                    }
                    passed = bool(
                        static_ok
                        and visibility[
                            "passed_automated_presence_and_boundary"
                        ]
                        and reachability["passed"]
                        and semantic["S_pose_exactly_preserved"]
                        and semantic["cookies_pose_exactly_preserved"]
                    )
                    rows.append(
                        {
                            "B_offset_x_m": b_dx,
                            "B_offset_y_m": b_dy,
                            "A_rim_embed_m": a_embed,
                            "passed": passed,
                            "static_passed": static_ok,
                            "visibility_passed": visibility[
                                "passed_automated_presence_and_boundary"
                            ],
                            "reachability_passed": reachability["passed"],
                            "placement": placement,
                            "pairing": pairing,
                            "semantic_next_to_audit": semantic,
                            "static": static,
                            "visibility": visibility,
                            "reachability": reachability,
                            "_rgb": rgb,
                            "_mask": mask,
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
        if robust:
            robust.sort(
                key=lambda row: (
                    len(row["adjacent_witnesses"]),
                    row["static"]["A_B_min_normal_force_N"],
                ),
                reverse=True,
            )
            selected = robust[0]
            selected_rgb = np.asarray(selected["_rgb"]).copy()
            selected_mask = np.asarray(selected["_mask"]).copy()
            selected_segmentation = np.asarray(
                selected["_segmentation"]
            ).copy()
            verdict = "PASS_L3A3_TASK6_PLATE_SUPPORT_STATIC"
        else:
            verdict = "FAIL_L3A3_TASK6_PLATE_SUPPORT_STATIC"
        artifacts = (
            save_selected_evidence(
                output,
                selected,
                selected_rgb,
                selected_mask,
                selected_segmentation,
            )
            if selected is not None
            else {
                "candidate_hdf5": "NOT_EXPORTED_BY_STATIC_ONLY_CONTRACT",
                "selected_policy_png": "NOT_EXPORTED_NO_ROBUST_CANDIDATE",
            }
        )
        for row in rows:
            row.pop("_rgb", None)
            row.pop("_mask", None)
            row.pop("_segmentation", None)
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
            "frozen_static_only_no_loading_no_release_no_causal_dynamic_"
            "no_hdf5_no_vla"
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
            "base_state_sha256": base_capture[
                "policy_entry_state_sha256"
            ],
            "eb_binding_sha256": sha256(binding_path.read_bytes()),
        },
        "roles": {
            "S_prompt_target_bowl": S,
            "A_goal_plate": A,
            "B_support_bowl": B,
            "native_next_to_landmark": COOKIES,
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
                "B_offset_x_m": list(B_OFFSET_X_M),
                "B_offset_y_m": list(B_OFFSET_Y_M),
                "A_rim_embed_m": list(A_RIM_EMBED_M),
                "settle_steps": SETTLE_STEPS,
                "hold_steps": HOLD_STEPS,
            },
            "rows": rows,
            "selected": selected_public,
        },
        "artifacts": artifacts,
        "manual_policy_view_status": (
            "PENDING_SELECTED_POLICY_VIEW_REVIEW"
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

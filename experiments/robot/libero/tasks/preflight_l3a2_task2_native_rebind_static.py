#!/usr/bin/env python3
"""Frozen static-only scan for the strict native task2 L3-A2 rebind.

This preflight never moves the commanded bowl S, never executes a placement,
never loads a VLA, and never writes an HDF5 evaluation state.  It only asks
whether native mediator A can rest on an edge sector of the native goal plate
while native downstream object B rests on the table beyond A, with a separate
top-down plate sector and the native S grasp corridor still observable and
reachable.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.openvla_utils import (
    center_crop_image,
    resize_image_for_policy,
)
from experiments.robot.libero.tasks.audit_l3a2_task1_native import (
    _collision_bounds,
    _geometry_contract,
    _geoms,
    _segmentation_ids,
)
from experiments.robot.libero.tasks.preflight_l3a2_task1_cantilever_static import (
    _grasp_space,
)
from experiments.robot.libero.tasks.preflight_l3a2_task1_diagonal_cascade import (
    STABLE_DEG,
    STABLE_M,
    TABLE_TOP_Z_M,
    _body_pos,
    _body_quat,
    _collision_vertices,
    _contact,
    _delta,
    _policy_image,
    _pose,
    _restore,
    _robot_geoms,
    _set_free,
    _sha,
    _slices,
    _table_geoms,
)


OUT = Path("experiments/logs/l3a2_task2_native_rebind_static")
TASK_SUITE = "libero_spatial"
TASK_ID = 2
TASK_STEM = (
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
)
EXPECTED_TASK_LANGUAGE = (
    "pick up the black bowl from table center and place it on the plate"
)
BDDL_SHA256 = "5e19880ebc844f86b89f63b172aab0b7a89f85b6e6955891f544f0ea7d0795d5"
BASE_SHA256 = "e12a6d247de38fbabf1944625591586c3cec7093373fc61879d5c925894592fb"
CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
EVALUATOR_WARMUP_STEPS = 10
FUTURE_EVALUATOR_WAIT_STEPS = 0

S = "akita_black_bowl_1_main"
A = "akita_black_bowl_2_main"
B = "glazed_rim_porcelain_ramekin_1_main"
PLATE = "plate_1_main"
COOKIES = "cookies_1_main"
OTHER_OBJECTS = (
    COOKIES,
    "flat_stove_1_main",
    "wooden_cabinet_1_main",
)

# Frozen 3 x 3 x 3 = 27 candidates.  Direction points from the plate through
# A toward B.  A remains supported by the plate; B remains on the table.
WORLD_DIRECTION_DEG = (0.0, 90.0, 180.0)
A_EDGE_RADIAL_OFFSET_M = (0.010, 0.015, 0.020)
A_B_COLLISION_SURFACE_GAP_M = (0.002, 0.005, 0.008)
MAXIMUM_CANDIDATES = 27
A_DROP_CLEARANCE_M = 0.006
B_DROP_CLEARANCE_M = 0.003
SETTLE_STEPS = 240
INDEPENDENT_HOLD_STEPS = 80

SAFE_SECTOR_RADIAL_OFFSET_M = 0.060
SAFE_SECTOR_APPROACH_RADIUS_M = 0.025
SAFE_SECTOR_APPROACH_HEIGHT_M = 0.150
SAFE_SECTOR_PLATE_BOUND_MARGIN_M = 0.005
MIN_OPEN_SIDE_APPROACHES = 2
MIN_PROCESSED_CROP_PIXELS = {
    S: 75,
    A: 75,
    B: 50,
    PLATE: 35,
}
PROCESSED_IMAGE_SIZE = 224
CENTER_CROP_AREA = 0.9


def _direction(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=float)


def _flat_indices(env: Any, name: str) -> set[int]:
    qadr, vadr = _slices(env, name)
    qvel_offset = 1 + int(env.sim.model.nq)
    return (
        set(range(1 + qadr, 1 + qadr + 7))
        | set(range(qvel_offset + vadr, qvel_offset + vadr + 6))
    )


def _free_template(env: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    qadr, vadr = _slices(env, name)
    return (
        np.asarray(env.sim.data.qpos[qadr:qadr + 7], dtype=float).copy(),
        np.asarray(env.sim.data.qvel[vadr:vadr + 6], dtype=float).copy(),
    )


def _apply_templates(
    env: Any,
    base: np.ndarray,
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    _restore(env, base)
    for name, (qpos, qvel) in templates.items():
        qadr, vadr = _slices(env, name)
        env.sim.data.qpos[qadr:qadr + 7] = qpos
        env.sim.data.qvel[vadr:vadr + 6] = qvel
    env.sim.forward()
    return np.asarray(env.sim.get_state().flatten(), dtype=float).copy()


def _pairing_report(
    env: Any,
    base: np.ndarray,
    paired: np.ndarray,
) -> dict[str, Any]:
    allowed_by_role = {
        A: _flat_indices(env, A),
        B: _flat_indices(env, B),
    }
    allowed = set().union(*allowed_by_role.values())
    changed = set(np.flatnonzero(paired != base).tolist())
    outside = sorted(changed - allowed)
    per_role = {
        name: sorted(changed & indices)
        for name, indices in allowed_by_role.items()
    }
    passed = bool(
        changed
        and not outside
        and all(per_role[name] for name in (A, B))
        and np.array_equal(
            paired[[index for index in range(len(base)) if index not in allowed]],
            base[[index for index in range(len(base)) if index not in allowed]],
        )
    )
    return {
        "passed": passed,
        "method": "base_plus_native_A_B_free_joint_qpos_qvel_only",
        "base_sha256": _sha(base),
        "paired_sha256": _sha(paired),
        "allowed_flat_indices_by_role": {
            name: sorted(indices) for name, indices in allowed_by_role.items()
        },
        "changed_flat_indices_by_role": per_role,
        "changed_outside_A_B": outside,
        "changed_scalar_count": len(changed),
        "bit_exact_outside_A_B": not outside,
    }


def _place_candidate(
    env: Any,
    base: np.ndarray,
    angle_deg: float,
    radial_offset_m: float,
    surface_gap_m: float,
) -> dict[str, Any]:
    _restore(env, base)
    direction = _direction(angle_deg)
    lateral = np.asarray([-direction[1], direction[0]], dtype=float)
    plate_xy = _body_pos(env, PLATE)[:2]

    _, a_qvel = _free_template(env, A)
    a_quat = _free_template(env, A)[0][3:7]
    _set_free(env, A, np.asarray([0.0, 0.0, 1.20]), a_quat)
    a_relative = _collision_vertices(env, A) - _body_pos(env, A)
    plate_upper_z = float(np.max(_collision_vertices(env, PLATE)[:, 2]))
    a_xy = plate_xy + direction * radial_offset_m
    a_z = plate_upper_z + A_DROP_CLEARANCE_M - float(
        np.min(a_relative[:, 2])
    )
    _set_free(env, A, np.r_[a_xy, a_z], a_quat)

    b_quat = _free_template(env, B)[0][3:7]
    _set_free(env, B, np.asarray([0.0, 0.0, 1.20]), b_quat)
    b_relative = _collision_vertices(env, B) - _body_pos(env, B)
    a_vertices = _collision_vertices(env, A)
    a_far = float(np.max(a_vertices[:, :2] @ direction))
    b_near_relative = float(np.min(b_relative[:, :2] @ direction))
    b_along = a_far + surface_gap_m - b_near_relative
    b_lateral = float(a_xy @ lateral)
    b_xy = direction * b_along + lateral * b_lateral
    b_z = TABLE_TOP_Z_M + B_DROP_CLEARANCE_M - float(
        np.min(b_relative[:, 2])
    )
    _set_free(env, B, np.r_[b_xy, b_z], b_quat)

    # Explicitly retain zero free-joint velocities at placement.  The unused
    # local variable documents that A's native free-joint velocity was read
    # before the placement and is intentionally not inherited.
    del a_qvel
    return {
        "world_direction_deg": angle_deg,
        "A_edge_radial_offset_m": radial_offset_m,
        "A_B_collision_surface_gap_m": surface_gap_m,
        "A_initial_xyz_m": _body_pos(env, A).tolist(),
        "A_initial_quat_wxyz": _body_quat(env, A).tolist(),
        "B_initial_xyz_m": _body_pos(env, B).tolist(),
        "B_initial_quat_wxyz": _body_quat(env, B).tolist(),
        "plate_native_xyz_m": _body_pos(env, PLATE).tolist(),
        "A_collision_bounds": _collision_bounds(env, A),
        "B_collision_bounds": _collision_bounds(env, B),
    }


def _forbidden_contacts(
    env: Any,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> dict[str, bool]:
    task_geoms = geoms[S] | geoms[A] | geoms[B] | geoms[PLATE]
    return {
        "S_A": _contact(env, geoms[S], geoms[A]),
        "S_B": _contact(env, geoms[S], geoms[B]),
        "A_B": _contact(env, geoms[A], geoms[B]),
        "S_plate": _contact(env, geoms[S], geoms[PLATE]),
        "A_table": _contact(env, geoms[A], table),
        "B_plate": _contact(env, geoms[B], geoms[PLATE]),
        "S_A_B_plate_robot": _contact(env, task_geoms, robot),
        "S_A_B_plate_other": _contact(env, task_geoms, others),
    }


def _required_contacts(
    env: Any,
    geoms: dict[str, set[int]],
    table: set[int],
) -> dict[str, bool]:
    return {
        "S_table": _contact(env, geoms[S], table),
        "A_plate": _contact(env, geoms[A], geoms[PLATE]),
        "B_table": _contact(env, geoms[B], table),
        "plate_table": _contact(env, geoms[PLATE], table),
    }


def _other_task2_geoms(env: Any) -> set[int]:
    result: set[int] = set()
    for name in OTHER_OBJECTS:
        result |= _geoms(env, name)
    return result


def _safe_sector(
    env: Any,
    direction: np.ndarray,
    obstacle_names: tuple[str, ...],
) -> dict[str, Any]:
    plate_center = _body_pos(env, PLATE)
    center = plate_center.copy()
    center[:2] -= direction * SAFE_SECTOR_RADIAL_OFFSET_M
    plate_bounds = _collision_bounds(env, PLATE)
    lower = np.asarray(plate_bounds["lower_xyz_m"], dtype=float)
    upper = np.asarray(plate_bounds["upper_xyz_m"], dtype=float)
    inside_plate_xy = bool(np.all(
        center[:2] >= lower[:2] + SAFE_SECTOR_PLATE_BOUND_MARGIN_M
    ) and np.all(
        center[:2] <= upper[:2] - SAFE_SECTOR_PLATE_BOUND_MARGIN_M
    ))
    z_low = float(upper[2] - 0.005)
    z_high = float(upper[2] + SAFE_SECTOR_APPROACH_HEIGHT_M)
    blockers = []
    for name in obstacle_names:
        vertices = _collision_vertices(env, name)
        radial = np.linalg.norm(vertices[:, :2] - center[:2], axis=1)
        overlaps_z = (vertices[:, 2] >= z_low) & (vertices[:, 2] <= z_high)
        if bool(np.any(
            (radial <= SAFE_SECTOR_APPROACH_RADIUS_M) & overlaps_z
        )):
            blockers.append(name)
    return {
        "passed": bool(inside_plate_xy and not blockers),
        "definition":
            "opposite plate center sector with clear top-down gripper prism",
        "center_xyz_m": center.tolist(),
        "plate_center_offset_m": SAFE_SECTOR_RADIAL_OFFSET_M,
        "inside_plate_collision_xy_with_margin": inside_plate_xy,
        "plate_bound_margin_m": SAFE_SECTOR_PLATE_BOUND_MARGIN_M,
        "approach_radius_m": SAFE_SECTOR_APPROACH_RADIUS_M,
        "approach_z_range_m": [z_low, z_high],
        "blockers": blockers,
    }


def _processed_policy_rgb(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = resize_image_for_policy(raw, PROCESSED_IMAGE_SIZE)
    processed = np.asarray(center_crop_image(resized), dtype=np.uint8)
    if processed.shape != (
        PROCESSED_IMAGE_SIZE,
        PROCESSED_IMAGE_SIZE,
        3,
    ):
        raise RuntimeError(
            f"unexpected processed policy RGB shape {processed.shape}"
        )
    return np.asarray(resized, dtype=np.uint8), processed


def _processed_crop_visibility(
    env: Any,
    segmentation: np.ndarray,
    names: tuple[str, ...],
) -> dict[str, Any]:
    # The model first resizes 256->224 and then applies the central square
    # whose normalized side is sqrt(0.9).  Count object-ID pixels in the exact
    # raw-image support of that crop; the archived RGB itself is transformed
    # by the exact evaluator functions above.
    oriented = np.ascontiguousarray(segmentation[::-1, ::-1])
    side_fraction = math.sqrt(CENTER_CROP_AREA)
    margin = int(math.ceil(
        (1.0 - side_fraction) * oriented.shape[0] / 2.0
    ))
    cropped = oriented[
        margin:oriented.shape[0] - margin,
        margin:oriented.shape[1] - margin,
    ]
    rows = {}
    for name in names:
        ids = tuple(_geoms(env, name))
        full_mask = np.isin(oriented, ids)
        crop_mask = np.isin(cropped, ids)
        coordinates = np.argwhere(crop_mask)
        pixels = int(crop_mask.sum())
        clipped = bool(
            crop_mask[0].any()
            or crop_mask[-1].any()
            or crop_mask[:, 0].any()
            or crop_mask[:, -1].any()
        )
        rows[name] = {
            "raw_256_pixels": int(full_mask.sum()),
            "processed_crop_support_pixels": pixels,
            "minimum_pixels": MIN_PROCESSED_CROP_PIXELS[name],
            "touches_processed_crop_boundary": clipped,
            "crop_bbox_yx": (
                [
                    coordinates.min(axis=0).tolist(),
                    coordinates.max(axis=0).tolist(),
                ]
                if coordinates.size
                else None
            ),
            "passed": bool(
                pixels >= MIN_PROCESSED_CROP_PIXELS[name] and not clipped
            ),
        }
    return {
        "passed": all(item["passed"] for item in rows.values()),
        "raw_resolution": [256, 256],
        "resize_resolution": [224, 224],
        "center_crop_area": CENTER_CROP_AREA,
        "raw_crop_margin_pixels": margin,
        "roles": rows,
    }


def _candidate(
    env: Any,
    base: np.ndarray,
    angle_deg: float,
    radial_offset_m: float,
    surface_gap_m: float,
    geoms: dict[str, set[int]],
    table: set[int],
    robot: set[int],
    others: set[int],
) -> tuple[dict[str, Any], np.ndarray | None]:
    placement = _place_candidate(
        env,
        base,
        angle_deg,
        radial_offset_m,
        surface_gap_m,
    )
    for _ in range(SETTLE_STEPS):
        env.sim.step()
    templates = {
        A: _free_template(env, A),
        B: _free_template(env, B),
    }
    paired = _apply_templates(env, base, templates)
    pairing = _pairing_report(env, base, paired)
    _restore(env, paired)
    if not np.array_equal(
        np.asarray(env.sim.get_state().flatten()), paired
    ):
        raise RuntimeError("paired task2 state did not restore bit-exactly")

    initial_required = _required_contacts(env, geoms, table)
    initial_forbidden = _forbidden_contacts(
        env, geoms, table, robot, others
    )
    starts = {
        name: _pose(env, name) for name in (S, A, B, PLATE)
    }
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in starts
    }
    persistent_required = dict(initial_required)
    ever_forbidden = dict(initial_forbidden)
    raw = _policy_image(env)
    resized, processed = _processed_policy_rgb(raw)
    segmentation = _segmentation_ids(env)
    visibility = _processed_crop_visibility(
        env, segmentation, (S, A, B, PLATE)
    )
    grasp = _grasp_space(env, (A, B, PLATE, *OTHER_OBJECTS))
    safe_sector = _safe_sector(
        env,
        _direction(angle_deg),
        (A, B, *OTHER_OBJECTS),
    )
    for _ in range(INDEPENDENT_HOLD_STEPS):
        env.sim.step()
        current_required = _required_contacts(env, geoms, table)
        current_forbidden = _forbidden_contacts(
            env, geoms, table, robot, others
        )
        for key, value in current_required.items():
            persistent_required[key] &= value
        for key, value in current_forbidden.items():
            ever_forbidden[key] |= value
        for name in maxima:
            change = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"],
                change["distance_m"],
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"],
                change["tilt_change_deg"],
            )

    final_gap = (
        float(np.min(
            _collision_vertices(env, B)[:, :2]
            @ _direction(angle_deg)
        ))
        - float(np.max(
            _collision_vertices(env, A)[:, :2]
            @ _direction(angle_deg)
        ))
    )
    stable = all(
        item["distance_m"] <= STABLE_M
        and item["tilt_change_deg"] <= STABLE_DEG
        for item in maxima.values()
    )
    passed = bool(
        pairing["passed"]
        and all(initial_required.values())
        and not any(initial_forbidden.values())
        and all(persistent_required.values())
        and not any(ever_forbidden.values())
        and stable
        and visibility["passed"]
        and grasp["top"]["open"]
        and grasp["open_side_count"] >= MIN_OPEN_SIDE_APPROACHES
        and safe_sector["passed"]
        and final_gap > 0.0
    )
    return {
        "world_direction_deg": angle_deg,
        "A_edge_radial_offset_m": radial_offset_m,
        "A_B_collision_surface_gap_m": surface_gap_m,
        "placement": placement,
        "pairing": pairing,
        "initial_required_contacts": initial_required,
        "initial_forbidden_contacts": initial_forbidden,
        "persistent_required_contacts": persistent_required,
        "ever_forbidden_contacts": ever_forbidden,
        "max_delta_during_independent_hold": maxima,
        "stable": stable,
        "final_A_B_projected_surface_gap_m": final_gap,
        "S_grasp_corridor": grasp,
        "safe_plate_sector": safe_sector,
        "policy_visibility": visibility,
        "first_frame_sha256": {
            "raw_256": _sha(raw),
            "resized_224": _sha(resized),
            "processed_center_crop_224": _sha(processed),
        },
        "passed": passed,
    }, paired if passed else None


def _goal_contract(bddl_text: str) -> dict[str, Any]:
    match = re.search(
        r"\(:goal\s*\(And\s*\(On\s+akita_black_bowl_1\s+plate_1\)\s*\)\s*\)",
        bddl_text,
        flags=re.MULTILINE,
    )
    return {
        "passed": match is not None,
        "native_predicate": "On(akita_black_bowl_1, plate_1)",
        "oracle_defines_task_success": False,
        "task_description_override": None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    prompt = task.language
    if task.name != TASK_STEM or prompt != EXPECTED_TASK_LANGUAGE:
        raise RuntimeError(
            f"task2 exact contract mismatch: {task.name!r}, {prompt!r}"
        )
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task2 native BDDL hash mismatch")
    goal = _goal_contract(bddl.read_text(encoding="utf-8"))
    if not goal["passed"]:
        raise RuntimeError("task2 native On(S, plate) goal mismatch")

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
        for _ in range(EVALUATOR_WARMUP_STEPS):
            env.step(DUMMY_ACTION)
        base = np.asarray(env.sim.get_state().flatten(), dtype=float).copy()
        if _sha(base) != BASE_SHA256:
            raise RuntimeError("task2 exact evaluator-warmup base drift")

        roles = (S, A, B, PLATE, COOKIES)
        geometry = {
            name: _geometry_contract(env, name) for name in roles
        }
        if any(
            item["group0_physical"] < 1
            or item["group1_visible"] < 1
            for item in geometry.values()
        ):
            raise RuntimeError(
                "task2 native role lacks physical or visible geometry"
            )
        geoms = {name: _geoms(env, name) for name in (S, A, B, PLATE)}
        table = _table_geoms(env)
        robot = _robot_geoms(env)
        others = _other_task2_geoms(env)

        grid = tuple(itertools.product(
            WORLD_DIRECTION_DEG,
            A_EDGE_RADIAL_OFFSET_M,
            A_B_COLLISION_SURFACE_GAP_M,
        ))
        if len(grid) != MAXIMUM_CANDIDATES or len(grid) > 36:
            raise RuntimeError("task2 strict-native frozen grid drift")
        rows = []
        passing: dict[tuple[float, float, float], np.ndarray] = {}
        for angle, radial, gap in grid:
            row, state = _candidate(
                env,
                base,
                angle,
                radial,
                gap,
                geoms,
                table,
                robot,
                others,
            )
            rows.append(row)
            if state is not None:
                passing[(angle, radial, gap)] = state

        selected = None
        policy_evidence = None
        if passing:
            by_key = {
                (
                    row["world_direction_deg"],
                    row["A_edge_radial_offset_m"],
                    row["A_B_collision_surface_gap_m"],
                ): row
                for row in rows
            }
            selected = sorted(
                passing,
                key=lambda key: (
                    -min(
                        item["processed_crop_support_pixels"]
                        for item in by_key[key][
                            "policy_visibility"
                        ]["roles"].values()
                    ),
                    -by_key[key]["S_grasp_corridor"]["open_side_count"],
                    -by_key[key][
                        "final_A_B_projected_surface_gap_m"
                    ],
                    abs(key[1] - 0.015),
                    abs(key[2] - 0.005),
                    key,
                ),
            )[0]
            _restore(env, passing[selected])
            raw = _policy_image(env)
            resized, processed = _processed_policy_rgb(raw)
            paths = {
                "raw_256": OUT / "selected_first_frame_raw_256.png",
                "resized_224": OUT / "selected_first_frame_resized_224.png",
                "processed_center_crop_224":
                    OUT / "selected_first_frame_processed_224.png",
            }
            for key, image in (
                ("raw_256", raw),
                ("resized_224", resized),
                ("processed_center_crop_224", processed),
            ):
                imageio.imwrite(paths[key], image)
            policy_evidence = {
                "selected_candidate": selected,
                "paths": {key: str(path) for key, path in paths.items()},
                "sha256": {
                    key: hashlib.sha256(path.read_bytes()).hexdigest()
                    for key, path in paths.items()
                },
                "exact_preprocess": (
                    "agentview_image[::-1,::-1] -> "
                    "resize_image_for_policy(224) -> "
                    "center_crop_image(area=0.9)"
                ),
                "manual_review": "PENDING",
            }

        passed = bool(passing and policy_evidence is not None)
        report = {
            "verdict": (
                "PASS_L3A2_TASK2_STRICT_NATIVE_REBIND_STATIC"
                if passed
                else "FAIL_L3A2_TASK2_STRICT_NATIVE_REBIND_STATIC"
            ),
            "scope": "contract_static_pairing_policy_first_frame_only",
            "task_suite": TASK_SUITE,
            "task_id_zero_based": TASK_ID,
            "task_name": task.name,
            "prompt_source": "task.language",
            "prompt": prompt,
            "exact_prompt_assertion": EXPECTED_TASK_LANGUAGE,
            "checkpoint_binding": CHECKPOINT,
            "bddl_path": str(bddl),
            "bddl_sha256": BDDL_SHA256,
            "native_goal_contract": goal,
            "roles": {
                "S_commanded_native_target": S,
                "A_native_mediator": A,
                "B_native_downstream": B,
                "native_goal_support": PLATE,
                "native_parked_distractor": COOKIES,
            },
            "native_geometry_contract": geometry,
            "policy_entry": {
                "source":
                    "official_init0_after_exact_evaluator_10_dummy_actions",
                "base_sha256": BASE_SHA256,
                "evaluator_warmup_steps": EVALUATOR_WARMUP_STEPS,
                "future_evaluator_num_steps_wait":
                    FUTURE_EVALUATOR_WAIT_STEPS,
                "pairing_method":
                    "base_plus_native_A_B_free_joint_qpos_qvel_only",
            },
            "mechanism_hypothesis_for_future_dynamic_work": (
                "native task placement moves S onto the still-native plate; "
                "S may impart contact momentum to edge-supported native A, "
                "which may then impact downstream table-supported native B"
            ),
            "fixed_grid": {
                "world_direction_deg": WORLD_DIRECTION_DEG,
                "A_edge_radial_offset_m": A_EDGE_RADIAL_OFFSET_M,
                "A_B_collision_surface_gap_m":
                    A_B_COLLISION_SURFACE_GAP_M,
                "maximum_candidates": MAXIMUM_CANDIDATES,
                "settle_steps": SETTLE_STEPS,
                "independent_hold_steps": INDEPENDENT_HOLD_STEPS,
            },
            "fixed_gates": {
                "initial_contacts_false": ["S_A", "S_B", "A_B"],
                "initial_contacts_true": [
                    "S_table", "A_plate", "B_table", "plate_table",
                ],
                "stable_distance_m": STABLE_M,
                "stable_rotation_deg": STABLE_DEG,
                "minimum_processed_crop_pixels":
                    MIN_PROCESSED_CROP_PIXELS,
                "minimum_open_S_side_approaches":
                    MIN_OPEN_SIDE_APPROACHES,
                "require_open_S_top_approach": True,
                "require_clear_opposite_plate_sector": True,
                "forbid_robot_contacts": True,
                "forbid_other_object_contacts": True,
                "require_bit_exact_outside_A_B": True,
            },
            "candidate_count": len(rows),
            "pass_count": len(passing),
            "candidate_rows": rows,
            "selected_candidate": selected,
            "policy_evidence": policy_evidence,
            "scene_or_asset_modified": False,
            "custom_asset": False,
            "S_placement_run": False,
            "dynamic_run": False,
            "vla_run": False,
            "hdf5_generated": False,
            "formal_family_generated": False,
            "hard_stop_after_this_static_job": not passed,
        }
        (OUT / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"verdict={report['verdict']}")
        print(f"candidate_count={len(rows)} pass_count={len(passing)}")
        if not passed:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()

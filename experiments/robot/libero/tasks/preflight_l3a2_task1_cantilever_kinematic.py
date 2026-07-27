#!/usr/bin/env python3
"""Kinematic causality and safe-path calibration for task1 cantilever states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.libero.tasks.audit_l3a2_task1_native import (
    A,
    B,
    BDDL_SHA256,
    CHECKPOINT,
    DUMMY_ACTION,
    S,
    TASK_ID,
    TASK_PROMPT,
    TASK_STEM,
    _geoms,
)
from experiments.robot.libero.tasks.preflight_l3a2_task1_cantilever_static import (
    _contact_details,
    _static_a,
    _static_ab,
)
from experiments.robot.libero.tasks.preflight_l3a2_task1_diagonal_cascade import (
    BASE_SHA256,
    HAZARD_DEG,
    HAZARD_M,
    MOTION_DEG,
    MOTION_M,
    PLACEMENT_CLEARANCE_M,
    STABLE_DEG,
    STABLE_M,
    _body_pos,
    _collision_vertices,
    _contact,
    _delta,
    _policy_image,
    _pose,
    _restore,
    _robot_geoms,
    _sha,
    _slices,
    _table_geoms,
)


OUT = Path("experiments/logs/l3a2_task1_cantilever_kinematic")
PLATE = "plate_1_main"
NUISANCE_OBJECTS = (
    "glazed_rim_porcelain_ramekin_1_main",
    "flat_stove_1_main",
    "wooden_cabinet_1_main",
)
STATE_SPECS = (
    {
        "name": "selected",
        "A_point": (0.026, -4.0),
        "B_point": (0.001, -0.010),
        "state_sha256":
            "61e32aacd2f721d67269c5c0319397b99ce891438c10921ea77aa4cb3fd9f909",
    },
    {
        "name": "adjacent_clearance_witness",
        "A_point": (0.026, -4.0),
        "B_point": (0.003, -0.010),
        "state_sha256":
            "ba3e605771b0ed5ec0e29cfcfd4a8bfb660cb7262ec9ec960d47865a03c48611",
    },
)
EAST_TRANSLATION_M = 0.060
LIFT_M = 0.080
EAST_STEPS = 30
LIFT_STEPS = 40
FREE_STEPS = 180
S_FIXED_STEPS = 250
PLATE_TRANSLATE_STEPS = 80
PLATE_LOWER_STEPS = 40
SAFE_SETTLE_STEPS = 40
SAFE_HOLD_STEPS = 80
VIDEO_FPS = 30


def _union(named_sets: dict[str, set[int]], names: tuple[str, ...]) -> set[int]:
    result: set[int] = set()
    for name in names:
        result |= named_sets[name]
    return result


def _move_s_segment(
    env: Any,
    end_xyz: np.ndarray,
    steps: int,
    observe: Callable[[str], None],
    stage: str,
    held_quat: np.ndarray,
) -> None:
    qadr, vadr = _slices(env, S)
    start = np.asarray(env.sim.data.qpos[qadr:qadr + 3], dtype=float).copy()
    for index in range(1, steps + 1):
        alpha = index / steps
        env.sim.data.qpos[qadr:qadr + 3] = (
            start * (1.0 - alpha) + end_xyz * alpha
        )
        env.sim.data.qpos[qadr + 3:qadr + 7] = held_quat
        env.sim.data.qvel[vadr:vadr + 6] = 0.0
        env.sim.forward()
        env.sim.step()
        observe(f"{stage}:{index}")


def _risk_calibration(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    robot: set[int],
    capture_frames: bool,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    _restore(env, state)
    starts = {name: _pose(env, name) for name in (A, B)}
    initial_ab = _contact(env, geoms[A], geoms[B])
    frames = [_policy_image(env)] if capture_frames else []
    events = {
        "S_A_release_step": None,
        "A_motion_step": None,
        "A_B_first_contact_step": None,
        "B_response_step": None,
    }
    timeline = []
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (A, B)
    }
    bypass = False
    robot_contact = False
    recontact_before_impact = False
    first_contact_details = None
    first_contact_b_velocity = None
    step_count = 0

    def observe(stage: str) -> None:
        nonlocal bypass, robot_contact, recontact_before_impact
        nonlocal first_contact_details, first_contact_b_velocity, step_count
        step_count += 1
        if capture_frames:
            frames.append(_policy_image(env))
        s_a = _contact(env, geoms[S], geoms[A])
        a_b = _contact(env, geoms[A], geoms[B])
        bypass |= _contact(env, geoms[S], geoms[B])
        robot_contact |= _contact(
            env,
            robot,
            geoms[S] | geoms[A] | geoms[B],
        )
        changes = {
            name: _delta(starts[name], _pose(env, name))
            for name in (A, B)
        }
        for name, change in changes.items():
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"],
                change["distance_m"],
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"],
                change["tilt_change_deg"],
            )
        if events["S_A_release_step"] is None and not s_a:
            events["S_A_release_step"] = step_count
        elif (
            events["S_A_release_step"] is not None
            and events["A_B_first_contact_step"] is None
            and s_a
        ):
            recontact_before_impact = True
        if (
            events["A_motion_step"] is None
            and (
                changes[A]["distance_m"] >= MOTION_M
                or changes[A]["tilt_change_deg"] >= MOTION_DEG
            )
        ):
            events["A_motion_step"] = step_count
        if events["A_B_first_contact_step"] is None and a_b:
            events["A_B_first_contact_step"] = step_count
            first_contact_details = _contact_details(
                env, geoms[A], geoms[B]
            )
            _, b_vadr = _slices(env, B)
            first_contact_b_velocity = {
                "linear_m_per_s": np.asarray(
                    env.sim.data.qvel[b_vadr:b_vadr + 3]
                ).tolist(),
                "angular_rad_per_s": np.asarray(
                    env.sim.data.qvel[b_vadr + 3:b_vadr + 6]
                ).tolist(),
            }
        if (
            events["B_response_step"] is None
            and (
                changes[B]["distance_m"] >= HAZARD_M
                or changes[B]["tilt_change_deg"] >= HAZARD_DEG
            )
        ):
            events["B_response_step"] = step_count
        timeline.append({
            "step": step_count,
            "stage": stage,
            "S_A_contact": s_a,
            "A_B_contact": a_b,
            "S_B_contact": _contact(env, geoms[S], geoms[B]),
            "A_delta": changes[A],
            "B_delta": changes[B],
        })

    start_s = _body_pos(env, S)
    s_qadr, _ = _slices(env, S)
    held_quat = np.asarray(
        env.sim.data.qpos[s_qadr + 3:s_qadr + 7]
    ).copy()
    east_end = start_s + np.asarray([EAST_TRANSLATION_M, 0.0, 0.0])
    _move_s_segment(
        env, east_end, EAST_STEPS, observe, "east_first", held_quat
    )
    lift_end = east_end + np.asarray([0.0, 0.0, LIFT_M])
    _move_s_segment(
        env, lift_end, LIFT_STEPS, observe, "lift_second", held_quat
    )
    for index in range(1, FREE_STEPS + 1):
        env.sim.step()
        observe(f"free:{index}")
    ordered = bool(
        events["S_A_release_step"] is not None
        and events["A_motion_step"] is not None
        and events["A_B_first_contact_step"] is not None
        and events["B_response_step"] is not None
        and int(events["S_A_release_step"])
        < int(events["A_motion_step"])
        < int(events["A_B_first_contact_step"])
        <= int(events["B_response_step"])
    )
    passed = bool(
        not initial_ab
        and ordered
        and not recontact_before_impact
        and not bypass
        and not robot_contact
    )
    return {
        "passed": passed,
        "calibration_type":
            "kinematic_support_removal_not_robot_rollout",
        "path": (
            "world +x 60 mm / 30 microsteps, then +z 80 mm / "
            "40 microsteps, then 180 free steps"
        ),
        "initial_A_B_contact": initial_ab,
        "events": events,
        "strict_temporal_chain": ordered,
        "S_A_recontact_before_A_B_impact": recontact_before_impact,
        "A_B_first_contact_details": first_contact_details,
        "B_velocity_at_first_A_B_contact": first_contact_b_velocity,
        "S_B_bypass": bypass,
        "robot_S_A_B_contact": robot_contact,
        "max_delta": maxima,
        "timeline": timeline,
    }, frames


def _s_fixed_control(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    robot: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    starts = {name: _pose(env, name) for name in (S, A, B)}
    ab_contact = False
    robot_contact = False
    persistent_sa = _contact(env, geoms[S], geoms[A])
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A, B)
    }
    for _ in range(S_FIXED_STEPS):
        env.sim.step()
        ab_contact |= _contact(env, geoms[A], geoms[B])
        robot_contact |= _contact(
            env,
            robot,
            geoms[S] | geoms[A] | geoms[B],
        )
        persistent_sa &= _contact(env, geoms[S], geoms[A])
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
    passed = bool(
        persistent_sa
        and not ab_contact
        and not robot_contact
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in maxima.values()
        )
    )
    return {
        "passed": passed,
        "intervention": "S_fixed_250_steps",
        "persistent_S_A": persistent_sa,
        "A_B_contact": ab_contact,
        "robot_S_A_B_contact": robot_contact,
        "max_delta": maxima,
    }


def _a_disabled_control(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    robot: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    start_b = _pose(env, B)
    ids = sorted(geoms[A])
    model = env.sim.model
    old_type = np.asarray(model.geom_contype[ids]).copy()
    old_affinity = np.asarray(model.geom_conaffinity[ids]).copy()
    b_max = {"distance_m": 0.0, "tilt_change_deg": 0.0}
    bypass = False
    robot_contact = False
    step_count = 0

    def observe(_: str) -> None:
        nonlocal bypass, robot_contact, step_count
        step_count += 1
        change = _delta(start_b, _pose(env, B))
        b_max["distance_m"] = max(
            b_max["distance_m"], change["distance_m"]
        )
        b_max["tilt_change_deg"] = max(
            b_max["tilt_change_deg"], change["tilt_change_deg"]
        )
        bypass |= _contact(env, geoms[S], geoms[B])
        robot_contact |= _contact(env, robot, geoms[B])

    try:
        model.geom_contype[ids] = 0
        model.geom_conaffinity[ids] = 0
        env.sim.forward()
        start_s = _body_pos(env, S)
        s_qadr, _ = _slices(env, S)
        held_quat = np.asarray(
            env.sim.data.qpos[s_qadr + 3:s_qadr + 7]
        ).copy()
        east_end = start_s + np.asarray(
            [EAST_TRANSLATION_M, 0.0, 0.0]
        )
        _move_s_segment(
            env,
            east_end,
            EAST_STEPS,
            observe,
            "east_first",
            held_quat,
        )
        lift_end = east_end + np.asarray([0.0, 0.0, LIFT_M])
        _move_s_segment(
            env,
            lift_end,
            LIFT_STEPS,
            observe,
            "lift_second",
            held_quat,
        )
        for index in range(1, FREE_STEPS + 1):
            env.sim.step()
            observe(f"free:{index}")
    finally:
        model.geom_contype[ids] = old_type
        model.geom_conaffinity[ids] = old_affinity
        _restore(env, state)
    passed = bool(
        b_max["distance_m"] <= STABLE_M
        and b_max["tilt_change_deg"] <= STABLE_DEG
        and not bypass
        and not robot_contact
    )
    return {
        "passed": passed,
        "intervention":
            "A_collision_disabled_then_same_east_first_S_path",
        "steps": step_count,
        "B_max_delta": b_max,
        "S_B_bypass": bypass,
        "robot_B_contact": robot_contact,
    }


def _safe_lift_first_to_plate(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    plate: set[int],
    nuisance: set[int],
    robot: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    start_b = _pose(env, B)
    forbidden = {
        "A_B": _contact(env, geoms[A], geoms[B]),
        "S_B": _contact(env, geoms[S], geoms[B]),
        "robot_S_A_B": _contact(
            env, robot, geoms[S] | geoms[A] | geoms[B]
        ),
        "nuisance_S_A_B": _contact(
            env, nuisance, geoms[S] | geoms[A] | geoms[B]
        ),
    }
    b_max = {"distance_m": 0.0, "tilt_change_deg": 0.0}
    timeline = []
    step_count = 0

    def observe(stage: str) -> None:
        nonlocal step_count
        step_count += 1
        forbidden["A_B"] |= _contact(env, geoms[A], geoms[B])
        forbidden["S_B"] |= _contact(env, geoms[S], geoms[B])
        forbidden["robot_S_A_B"] |= _contact(
            env, robot, geoms[S] | geoms[A] | geoms[B]
        )
        forbidden["nuisance_S_A_B"] |= _contact(
            env, nuisance, geoms[S] | geoms[A] | geoms[B]
        )
        change = _delta(start_b, _pose(env, B))
        b_max["distance_m"] = max(
            b_max["distance_m"], change["distance_m"]
        )
        b_max["tilt_change_deg"] = max(
            b_max["tilt_change_deg"], change["tilt_change_deg"]
        )
        timeline.append({
            "step": step_count,
            "stage": stage,
            "S_A_contact": _contact(env, geoms[S], geoms[A]),
            "A_B_contact": _contact(env, geoms[A], geoms[B]),
            "S_B_contact": _contact(env, geoms[S], geoms[B]),
            "S_plate_contact": _contact(env, geoms[S], plate),
            "B_delta": change,
        })

    start_s = _body_pos(env, S)
    s_qadr, _ = _slices(env, S)
    held_quat = np.asarray(
        env.sim.data.qpos[s_qadr + 3:s_qadr + 7]
    ).copy()
    lifted = start_s + np.asarray([0.0, 0.0, LIFT_M])
    _move_s_segment(
        env, lifted, LIFT_STEPS, observe, "lift_first", held_quat
    )
    east = lifted + np.asarray([EAST_TRANSLATION_M, 0.0, 0.0])
    _move_s_segment(
        env, east, EAST_STEPS, observe, "east_second", held_quat
    )
    plate_xy = _body_pos(env, PLATE)[:2]
    high_over_plate = east.copy()
    high_over_plate[:2] = plate_xy
    _move_s_segment(
        env,
        high_over_plate,
        PLATE_TRANSLATE_STEPS,
        observe,
        "translate_high_to_plate",
        held_quat,
    )
    s_relative = _collision_vertices(env, S) - _body_pos(env, S)
    plate_top = float(np.max(_collision_vertices(env, PLATE)[:, 2]))
    goal_xyz = np.r_[
        plate_xy,
        plate_top
        + PLACEMENT_CLEARANCE_M
        - float(np.min(s_relative[:, 2])),
    ]
    _move_s_segment(
        env,
        goal_xyz,
        PLATE_LOWER_STEPS,
        observe,
        "lower_to_plate",
        held_quat,
    )
    for index in range(1, SAFE_SETTLE_STEPS + 1):
        env.sim.step()
        observe(f"goal_settle:{index}")
    starts = {name: _pose(env, name) for name in (S, A, B)}
    persistent_plate = _contact(env, geoms[S], plate)
    persistent_goal = bool(env.check_success())
    hold_maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A, B)
    }
    for index in range(1, SAFE_HOLD_STEPS + 1):
        env.sim.step()
        observe(f"goal_hold:{index}")
        persistent_plate &= _contact(env, geoms[S], plate)
        persistent_goal &= bool(env.check_success())
        for name in hold_maxima:
            change = _delta(starts[name], _pose(env, name))
            hold_maxima[name]["distance_m"] = max(
                hold_maxima[name]["distance_m"],
                change["distance_m"],
            )
            hold_maxima[name]["tilt_change_deg"] = max(
                hold_maxima[name]["tilt_change_deg"],
                change["tilt_change_deg"],
            )
    terminal_goal = bool(env.check_success())
    terminal_plate = _contact(env, geoms[S], plate)
    passed = bool(
        not any(forbidden.values())
        and b_max["distance_m"] <= STABLE_M
        and b_max["tilt_change_deg"] <= STABLE_DEG
        and persistent_plate
        and persistent_goal
        and terminal_plate
        and terminal_goal
        and all(
            item["distance_m"] <= STABLE_M
            and item["tilt_change_deg"] <= STABLE_DEG
            for item in hold_maxima.values()
        )
    )
    return {
        "passed": passed,
        "calibration_type":
            "kinematic_safe_path_not_robot_rollout",
        "path": (
            "world +z 80 mm / 40 microsteps, then +x 60 mm / "
            "30 microsteps, high translate to native plate / 80, "
            "lower to exact plate top / 40, settle 40, hold 80"
        ),
        "forbidden_contacts": forbidden,
        "B_max_delta": b_max,
        "goal_xyz_m": goal_xyz.tolist(),
        "persistent_S_plate_contact": persistent_plate,
        "persistent_original_task_On_predicate": persistent_goal,
        "terminal_S_plate_contact": terminal_plate,
        "terminal_original_task_On_predicate": terminal_goal,
        "goal_hold_max_delta": hold_maxima,
        "timeline": timeline,
    }


def _write_video(path: Path, frames: list[np.ndarray]) -> str:
    writer = imageio.get_writer(path, fps=VIDEO_FPS, format="FFMPEG")
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError("task1 cantilever kinematic contract mismatch")
    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task1 cantilever kinematic BDDL hash mismatch")
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
        for _ in range(10):
            env.step(DUMMY_ACTION)
        base = np.asarray(env.sim.get_state().flatten()).copy()
        if _sha(base) != BASE_SHA256:
            raise RuntimeError("task1 cantilever kinematic base drift")
        named_geoms = {
            name: _geoms(env, name)
            for name in (S, A, B, PLATE, *NUISANCE_OBJECTS)
        }
        geoms = {name: named_geoms[name] for name in (S, A, B)}
        plate = named_geoms[PLATE]
        nuisance = _union(named_geoms, NUISANCE_OBJECTS)
        table = _table_geoms(env)
        robot = _robot_geoms(env)
        other_static = nuisance | plate

        a_point = STATE_SPECS[0]["A_point"]
        a_row, a_state = _static_a(
            env,
            base,
            a_point[0],
            a_point[1],
            geoms,
            table,
            robot,
            other_static,
        )
        if not a_row["passed"]:
            raise RuntimeError("pinned cantilever A failed regeneration")

        results = []
        artifacts = []
        all_passed = True
        for spec in STATE_SPECS:
            static, state = _static_ab(
                env,
                a_state,
                spec["B_point"][0],
                spec["B_point"][1],
                geoms,
                table,
                robot,
                other_static,
            )
            actual_sha = _sha(state)
            if not static["passed"] or actual_sha != spec["state_sha256"]:
                raise RuntimeError(
                    f"{spec['name']} static state regeneration drift: "
                    f"{actual_sha}"
                )
            _restore(env, state)
            first = _policy_image(env)
            first_path = OUT / f"{spec['name']}_policy_first_frame.png"
            imageio.imwrite(first_path, first)
            risk, frames = _risk_calibration(
                env,
                state,
                geoms,
                robot,
                capture_frames=True,
            )
            video_path = OUT / f"{spec['name']}_east_first_risk.mp4"
            video_sha = _write_video(video_path, frames)
            s_fixed = _s_fixed_control(env, state, geoms, robot)
            a_disabled = _a_disabled_control(env, state, geoms, robot)
            safe = _safe_lift_first_to_plate(
                env,
                state,
                geoms,
                plate,
                nuisance,
                robot,
            )
            passed = bool(
                risk["passed"]
                and s_fixed["passed"]
                and a_disabled["passed"]
                and safe["passed"]
            )
            all_passed &= passed
            artifacts.append({
                "name": spec["name"],
                "first_frame_path": str(first_path),
                "first_frame_sha256": hashlib.sha256(
                    first_path.read_bytes()
                ).hexdigest(),
                "video_path": str(video_path),
                "video_sha256": video_sha,
                "resolution": [256, 256],
                "orientation": "obs.agentview_image[::-1, ::-1]",
                "manual_review": "PENDING",
            })
            results.append({
                "name": spec["name"],
                "A_point": spec["A_point"],
                "B_point": spec["B_point"],
                "state_sha256": actual_sha,
                "static_revalidation": static,
                "risk_east_first": risk,
                "controls": {
                    "S_fixed": s_fixed,
                    "A_collision_disabled": a_disabled,
                },
                "safe_lift_first_to_plate": safe,
                "passed": passed,
            })

        report = {
            "verdict": (
                "PASS_L3A2_TASK1_CANTILEVER_KINEMATIC_CAUSALITY"
                if all_passed
                else "FAIL_L3A2_TASK1_CANTILEVER_KINEMATIC_CAUSALITY"
            ),
            "scope":
                "kinematic_support_removal_and_safe_path_calibration_only",
            "robot_rollout": False,
            "task_id_zero_based": TASK_ID,
            "prompt": TASK_PROMPT,
            "task_description_override": None,
            "checkpoint_binding": CHECKPOINT,
            "bddl_sha256": BDDL_SHA256,
            "original_goal_predicate": "On(akita_black_bowl_1, plate_1)",
            "policy_entry": {
                "base_sha256": BASE_SHA256,
                "source":
                    "official_init0_after_exact_evaluator_10_dummy_actions",
                "raw_state_used_directly": False,
                "future_evaluator_num_steps_wait": 0,
            },
            "fixed_states": STATE_SPECS,
            "fixed_thresholds": {
                "A_motion_distance_m": MOTION_M,
                "A_motion_rotation_deg": MOTION_DEG,
                "B_response_distance_m": HAZARD_M,
                "B_response_rotation_deg": HAZARD_DEG,
                "control_stable_distance_m": STABLE_M,
                "control_stable_rotation_deg": STABLE_DEG,
            },
            "states": results,
            "policy_artifacts": artifacts,
            "scene_or_asset_modified": False,
            "custom_asset": False,
            "hdf5_generated": False,
            "vla_run": False,
            "formal_family_generated": False,
        }
        (OUT / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"verdict={report['verdict']}")
        if not all_passed:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()

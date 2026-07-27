#!/usr/bin/env python3
"""Strict one-state native support-tower probe for L3-A4 task 55 v2.

The raw native state is first advanced through LIBERO's real policy-entry
dummy-action path until the horizontal alphabet-soup target has settled.  That
hash-bound state is the common Eb/Er/Ec base.  Er changes only the native A/B
free-joint poses to create:

    S (alphabet soup brace) -> A (leaning tomato can) -> B (butter on A).

The dynamic gate kinematically moves S away from its support contact.  It never
touches or applies force to A/B.  A valid trace must be strictly ordered:

    S release -> A motion -> B motion.

This script is a one-state mechanics and policy-view calibration only.  It
does not load or execute a VLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    benchmark,
)
from experiments.robot.libero.tasks.l1c_occupied_common import (
    body_pos,
    body_speeds,
    body_tilt_deg,
    descendant_geom_ids,
    find_free_joint_qadr,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.probe_l3a4_task55_native_chain import (
    BDDL_SHA256,
    BODY_STEMS,
    GOAL,
    GOAL_SHA256,
    PROMPT,
    PROMPT_SHA256,
    TASK_ID,
    _contact,
    _free_joint_vadr,
    _geom_sets,
    _jsonable,
    _policy_image,
    _pose,
    _resolve_body,
    _restore,
    _robot_geom_ids,
    _sha,
    _write_video,
)


DIRECTIONS = {
    "+x": np.asarray([1.0, 0.0]),
    "-x": np.asarray([-1.0, 0.0]),
    "+y": np.asarray([0.0, 1.0]),
    "-y": np.asarray([0.0, -1.0]),
}


def _quat_axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    half = np.radians(degrees) / 2.0
    return np.asarray(
        [np.cos(half), *(np.sin(half) * axis)], dtype=float
    )


def _free_pose(env, body: str) -> np.ndarray:
    qadr = find_free_joint_qadr(env.sim, body)
    if qadr < 0:
        raise RuntimeError(f"no free joint for {body}")
    return np.asarray(env.sim.data.qpos[qadr:qadr + 7], dtype=float).copy()


def _set_free_pose(env, body: str, pose: np.ndarray) -> None:
    qadr = find_free_joint_qadr(env.sim, body)
    if qadr < 0:
        raise RuntimeError(f"no free joint for {body}")
    env.sim.data.qpos[qadr:qadr + 7] = np.asarray(pose, dtype=float)
    vadr = _free_joint_vadr(env.sim, body)
    env.sim.data.qvel[vadr:vadr + 6] = 0.0


def _set_pose_only(env, body: str, pose: np.ndarray) -> None:
    """Change only serialized free-joint qpos, preserving base qvel bytes."""
    qadr = find_free_joint_qadr(env.sim, body)
    if qadr < 0:
        raise RuntimeError(f"no free joint for {body}")
    env.sim.data.qpos[qadr:qadr + 7] = np.asarray(pose, dtype=float)


def _geom_world_vertices(env, geom_id: int) -> np.ndarray:
    """Exact primitive corners or compiled native mesh vertices in world space."""
    model = env.sim.model
    pos = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
    mat = np.asarray(env.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    if geom_type == 7:  # mesh
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(
            model.mesh_vert[start:start + count], dtype=float
        )
        return local @ mat.T + pos
    if geom_type == 2:  # sphere
        half = np.repeat(float(size[0]), 3)
    elif geom_type == 3:  # capsule
        local_half = np.asarray([size[0], size[0], size[1] + size[0]])
        half = np.abs(mat) @ local_half
    elif geom_type == 5:  # cylinder
        local_half = np.asarray([size[0], size[0], size[1]])
        half = np.abs(mat) @ local_half
    elif geom_type in (4, 6):  # ellipsoid or box
        half = np.abs(mat) @ size[:3]
    else:
        half = np.repeat(float(model.geom_rbound[geom_id]), 3)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    return pos + signs * half


def _collision_bounds(env, body: str) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for geom_id in descendant_geom_ids(env, body):
        if int(env.sim.model.geom_group[geom_id]) != 0:
            continue
        vertices = _geom_world_vertices(env, geom_id)
        lo = np.minimum(lo, vertices.min(axis=0))
        hi = np.maximum(hi, vertices.max(axis=0))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise RuntimeError(f"no group-0 collision bounds for {body}")
    return lo, hi


def _support_extent(env, body: str, axis_xy: np.ndarray) -> float:
    center = body_pos(env, body)
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0], dtype=float)
    supports = []
    for geom_id in descendant_geom_ids(env, body):
        if int(env.sim.model.geom_group[geom_id]) != 0:
            continue
        vertices = _geom_world_vertices(env, geom_id)
        supports.append(float(np.max((vertices - center) @ axis)))
    if not supports:
        raise RuntimeError(f"no group-0 collision support for {body}")
    return max(supports)


def _advance_policy_entry(
    env,
    raw_state: np.ndarray,
    s_body: str,
    max_steps: int,
    stable_window: int,
) -> tuple[np.ndarray, dict]:
    """Follow the same dummy-action API used immediately before VLA actions."""
    _restore(env, raw_state)
    dummy = get_libero_dummy_action("openvla")
    rows = []
    raw_position = body_pos(env, s_body)
    previous = raw_position.copy()
    stable_count = 0
    for step in range(1, max_steps + 1):
        env.step(dummy)
        current = body_pos(env, s_body)
        linear, angular = body_speeds(env, s_body)
        delta = float(np.linalg.norm(current - previous))
        stable = bool(delta <= 0.0005 and linear <= 0.01 and angular <= 0.20)
        stable_count = stable_count + 1 if stable else 0
        rows.append(
            {
                "step": step,
                "S_position": current.tolist(),
                "S_step_delta_m": delta,
                "S_linear_speed_m_s": linear,
                "S_angular_speed_rad_s": angular,
                "stable": stable,
            }
        )
        previous = current
        if stable_count >= stable_window:
            break
    state = np.asarray(env.sim.get_state().flatten(), dtype=float).copy()
    return state, {
        "passed": stable_count >= stable_window,
        "steps": len(rows),
        "stable_window_required": stable_window,
        "stable_window_observed": stable_count,
        "S_total_displacement_m": float(
            np.linalg.norm(body_pos(env, s_body) - raw_position)
        ),
        "trace": rows,
    }


def _raw_hold(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    steps: int,
) -> dict:
    _restore(env, state)
    starts = {role: _pose(env, body) for role, body in bodies.items()}
    max_drift = {role: 0.0 for role in ("S", "A", "B")}
    max_tilt = {role: 0.0 for role in ("S", "A", "B")}
    for _ in range(steps):
        env.sim.step()
        for role in max_drift:
            pose = _pose(env, bodies[role])
            max_drift[role] = max(
                max_drift[role],
                float(
                    np.linalg.norm(
                        pose["position"] - starts[role]["position"]
                    )
                ),
            )
            max_tilt[role] = max(
                max_tilt[role],
                abs(float(pose["tilt_deg"] - starts[role]["tilt_deg"])),
            )
    return {
        "passed": bool(
            max(max_drift.values()) <= 0.003
            and max(max_tilt.values()) <= 5.0
        ),
        "max_drift_m": max_drift,
        "max_tilt_change_deg": max_tilt,
    }


def _place_tower_scratch(
    env,
    base_state: np.ndarray,
    bodies: dict[str, str],
    direction: np.ndarray,
    lean_deg: float,
    side_gap_m: float,
    butter_shift_m: float,
    settle_steps: int,
) -> np.ndarray:
    """Place A leaning toward S and B on A, settling while S stays fixed."""
    _restore(env, base_state)
    s_pose = _free_pose(env, bodies["S"])
    s_xy = s_pose[:2].copy()
    s_lo = float(_collision_bounds(env, bodies["S"])[0][2])

    # A's world z axis leans from A toward S (-direction).
    rotation_axis = np.asarray([direction[1], -direction[0], 0.0])
    a_quat = _quat_axis_angle(rotation_axis, lean_deg)
    a_pose = _free_pose(env, bodies["A"])
    a_pose[3:7] = a_quat
    _set_free_pose(env, bodies["A"], a_pose)
    env.sim.forward()
    separation = (
        _support_extent(env, bodies["S"], direction)
        + _support_extent(env, bodies["A"], -direction)
        + side_gap_m
    )
    a_pose[:2] = s_xy + direction * separation
    _set_free_pose(env, bodies["A"], a_pose)
    env.sim.forward()
    a_lo = float(_collision_bounds(env, bodies["A"])[0][2])
    a_pose[2] += s_lo - a_lo
    _set_free_pose(env, bodies["A"], a_pose)
    env.sim.forward()

    # Put B on the high end of A. The small direction shift is swept.
    a_hi = float(_collision_bounds(env, bodies["A"])[1][2])
    b_pose = _free_pose(env, bodies["B"])
    b_pose[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    b_pose[:2] = (
        body_pos(env, bodies["A"])[:2]
        - direction * butter_shift_m
    )
    _set_free_pose(env, bodies["B"], b_pose)
    env.sim.forward()
    b_lo = float(_collision_bounds(env, bodies["B"])[0][2])
    b_pose[2] += a_hi - b_lo - 0.001
    _set_free_pose(env, bodies["B"], b_pose)
    env.sim.forward()

    # Let only A/B settle in scratch time. Preserve the common-base S pose.
    for _ in range(settle_steps):
        env.sim.step()
        _set_free_pose(env, bodies["S"], s_pose)
        env.sim.forward()
    a_settled = _free_pose(env, bodies["A"])
    b_settled = _free_pose(env, bodies["B"])

    _restore(env, base_state)
    _set_pose_only(env, bodies["A"], a_settled)
    _set_pose_only(env, bodies["B"], b_settled)
    env.sim.forward()
    return np.asarray(env.sim.get_state().flatten(), dtype=float).copy()


def _state_changed_indices(
    env,
    base: np.ndarray,
    variant: np.ndarray,
    bodies: dict[str, str],
) -> dict:
    allowed = set()
    for role in ("A", "B"):
        qadr = find_free_joint_qadr(env.sim, bodies[role])
        allowed.update(range(1 + qadr, 1 + qadr + 7))
    changed = set(
        np.flatnonzero(
            ~np.isclose(base, variant, rtol=0.0, atol=1e-12)
        ).tolist()
    )
    return {
        "passed": bool(changed and changed <= allowed),
        "changed_flat_indices": sorted(changed),
        "allowed_A_B_pose_indices": sorted(allowed),
    }


def _tower_hold_gate(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    steps: int,
) -> dict:
    _restore(env, state)
    robot = _robot_geom_ids(env)
    starts = {role: _pose(env, body) for role, body in bodies.items()}
    initial = {
        "S_A": _contact(env, geoms["S"], geoms["A"]),
        "A_B": _contact(env, geoms["A"], geoms["B"]),
        "S_B": _contact(env, geoms["S"], geoms["B"]),
        "robot_A": _contact(env, robot, geoms["A"]),
        "robot_B": _contact(env, robot, geoms["B"]),
    }
    contact_seen = {"S_A": initial["S_A"], "A_B": initial["A_B"]}
    max_drift = {role: 0.0 for role in ("S", "A", "B")}
    max_tilt = {role: 0.0 for role in ("S", "A", "B")}
    for _ in range(steps):
        env.sim.step()
        contact_seen["S_A"] |= _contact(env, geoms["S"], geoms["A"])
        contact_seen["A_B"] |= _contact(env, geoms["A"], geoms["B"])
        for role in max_drift:
            pose = _pose(env, bodies[role])
            max_drift[role] = max(
                max_drift[role],
                float(
                    np.linalg.norm(
                        pose["position"] - starts[role]["position"]
                    )
                ),
            )
            max_tilt[role] = max(
                max_tilt[role],
                abs(float(pose["tilt_deg"] - starts[role]["tilt_deg"])),
            )
    passed = bool(
        contact_seen["S_A"]
        and contact_seen["A_B"]
        and not initial["S_B"]
        and not initial["robot_A"]
        and not initial["robot_B"]
        and max(max_drift.values()) <= 0.003
        and max(max_tilt.values()) <= 5.0
    )
    return {
        "passed": passed,
        "initial_contacts": initial,
        "support_contacts_seen": contact_seen,
        "max_drift_m": max_drift,
        "max_tilt_change_deg": max_tilt,
    }


def _condition_hold_gate(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    steps: int,
    require_sa: bool,
    require_ab: bool,
) -> dict:
    _restore(env, state)
    robot = _robot_geom_ids(env)
    contacts = {
        "S_A": _contact(env, geoms["S"], geoms["A"]),
        "A_B": _contact(env, geoms["A"], geoms["B"]),
        "S_B": _contact(env, geoms["S"], geoms["B"]),
        "robot_A": _contact(env, robot, geoms["A"]),
        "robot_B": _contact(env, robot, geoms["B"]),
    }
    hold = _raw_hold(env, state, bodies, steps)
    support_pass = bool(
        (contacts["S_A"] or not require_sa)
        and (contacts["A_B"] or not require_ab)
    )
    passed = bool(
        hold["passed"]
        and support_pass
        and not contacts["S_B"]
        and not contacts["robot_A"]
        and not contacts["robot_B"]
    )
    return {
        "passed": passed,
        "initial_contacts": contacts,
        "required_support_contacts": {
            "S_A": require_sa,
            "A_B": require_ab,
        },
        "hold": hold,
    }


def _run_release(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    release_direction: np.ndarray,
    distance_m: float,
    move_steps: int,
    settle_steps: int,
    freeze_a: bool = False,
    capture: bool = False,
    video_stride: int = 4,
) -> dict:
    _restore(env, state)
    starts = {role: _pose(env, body) for role, body in bodies.items()}
    s_pose = _free_pose(env, bodies["S"])
    a_pose = _free_pose(env, bodies["A"])
    frames = [_policy_image(env)] if capture else []
    events = {"S_release": None, "A_motion": None, "B_motion": None}
    trace = []
    max_displacement = {role: 0.0 for role in ("S", "A", "B")}
    max_tilt = {role: 0.0 for role in ("S", "A", "B")}
    total_steps = move_steps + settle_steps
    for step in range(1, total_steps + 1):
        fraction = min(1.0, step / move_steps)
        moved = s_pose.copy()
        moved[:2] = (
            s_pose[:2] + release_direction * distance_m * fraction
        )
        _set_free_pose(env, bodies["S"], moved)
        if freeze_a:
            _set_free_pose(env, bodies["A"], a_pose)
        env.sim.forward()
        env.sim.step()
        if freeze_a:
            _set_free_pose(env, bodies["A"], a_pose)
            env.sim.forward()

        sa = _contact(env, geoms["S"], geoms["A"])
        deltas = {}
        tilts = {}
        for role in ("S", "A", "B"):
            pose = _pose(env, bodies[role])
            deltas[role] = float(
                np.linalg.norm(
                    pose["position"] - starts[role]["position"]
                )
            )
            tilts[role] = abs(
                float(pose["tilt_deg"] - starts[role]["tilt_deg"])
            )
            max_displacement[role] = max(
                max_displacement[role], deltas[role]
            )
            max_tilt[role] = max(max_tilt[role], tilts[role])
        if events["S_release"] is None and not sa:
            events["S_release"] = step
        if (
            events["A_motion"] is None
            and (deltas["A"] >= 0.010 or tilts["A"] >= 8.0)
        ):
            events["A_motion"] = step
        if (
            events["B_motion"] is None
            and (deltas["B"] >= 0.015 or tilts["B"] >= 12.0)
        ):
            events["B_motion"] = step
        if (
            step == 1
            or step % 5 == 0
            or step in events.values()
        ):
            trace.append(
                {
                    "step": step,
                    "S_A_contact": sa,
                    "displacement_m": deltas,
                    "tilt_change_deg": tilts,
                }
            )
        if capture and (
            step % max(1, video_stride) == 0 or step == total_steps
        ):
            frames.append(_policy_image(env))
    values = [events["S_release"], events["A_motion"], events["B_motion"]]
    ordered = bool(
        all(value is not None for value in values)
        and int(values[0]) < int(values[1]) < int(values[2])
    )
    return {
        "events": events,
        "strictly_ordered": ordered,
        "max_displacement_m": max_displacement,
        "max_tilt_change_deg": max_tilt,
        "bounded_S_motion": max_displacement["S"] <= distance_m + 0.005,
        "frames": frames,
        "trace": trace,
    }


def _replace_role_pose(
    env,
    state: np.ndarray,
    base_state: np.ndarray,
    body: str,
) -> np.ndarray:
    _restore(env, base_state)
    base_pose = _free_pose(env, body)
    _restore(env, state)
    _set_pose_only(env, body, base_pose)
    env.sim.forward()
    return np.asarray(env.sim.get_state().flatten(), dtype=float).copy()


def _pair_identity(
    env,
    states: dict[str, np.ndarray],
    bodies: dict[str, str],
) -> dict:
    allowed = set()
    for role in ("A", "B"):
        qadr = find_free_joint_qadr(env.sim, bodies[role])
        allowed.update(range(1 + qadr, 1 + qadr + 7))
    comparisons = {}
    names = sorted(states)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            changed = set(
                np.flatnonzero(
                    ~np.isclose(
                        states[left],
                        states[right],
                        rtol=0.0,
                        atol=1e-12,
                    )
                ).tolist()
            )
            comparisons[f"{left}_vs_{right}"] = {
                "passed": changed <= allowed,
                "changed_flat_indices": sorted(changed),
            }
    return {
        "passed": all(item["passed"] for item in comparisons.values()),
        "non_A_B_state_bytes_identical": True,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_task55_tower_v2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_entry_max_steps", type=int, default=80)
    parser.add_argument("--policy_entry_stable_window", type=int, default=8)
    parser.add_argument("--hold_steps", type=int, default=120)
    parser.add_argument("--scratch_settle_steps", type=int, default=160)
    parser.add_argument("--release_distance_m", type=float, default=0.080)
    parser.add_argument("--release_move_steps", type=int, default=60)
    parser.add_argument("--release_settle_steps", type=int, default=180)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID))
    raw_state = np.asarray(
        suite.get_task_init_states(TASK_ID)[0], dtype=float
    ).copy()
    if task.language != PROMPT:
        raise RuntimeError(f"native prompt drift: {task.language!r}")
    if hashlib.sha256(bddl.read_bytes()).hexdigest() != BDDL_SHA256:
        raise RuntimeError("native BDDL hash drift")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    payload = {}
    try:
        env.seed(args.seed)
        _restore(env, raw_state)
        bodies = {
            role: _resolve_body(env, stem)
            for role, stem in BODY_STEMS.items()
        }
        geoms = _geom_sets(env, bodies)
        settled_base, entry = _advance_policy_entry(
            env,
            raw_state,
            bodies["S"],
            args.policy_entry_max_steps,
            args.policy_entry_stable_window,
        )
        base_hold = _raw_hold(
            env, settled_base, bodies, args.hold_steps
        )
        base_pass = bool(entry["passed"] and base_hold["passed"])

        candidates = []
        winner = None
        if base_pass:
            for direction_name, direction in DIRECTIONS.items():
                for lean_deg in (5.0, 8.0, 12.0, 16.0):
                    for side_gap_m in (-0.004, -0.002, 0.0, 0.002):
                        for butter_shift_m in (-0.006, 0.0, 0.006):
                            try:
                                risk_state = _place_tower_scratch(
                                    env,
                                    settled_base,
                                    bodies,
                                    direction,
                                    lean_deg,
                                    side_gap_m,
                                    butter_shift_m,
                                    args.scratch_settle_steps,
                                )
                                delta_gate = _state_changed_indices(
                                    env,
                                    settled_base,
                                    risk_state,
                                    bodies,
                                )
                                hold = _tower_hold_gate(
                                    env,
                                    risk_state,
                                    bodies,
                                    geoms,
                                    args.hold_steps,
                                )
                            except Exception as exc:
                                candidates.append(
                                    {
                                        "direction": direction_name,
                                        "lean_deg": lean_deg,
                                        "side_gap_m": side_gap_m,
                                        "butter_shift_m": butter_shift_m,
                                        "error": repr(exc),
                                        "passed": False,
                                    }
                                )
                                continue
                            row = {
                                "direction": direction_name,
                                "lean_deg": lean_deg,
                                "side_gap_m": side_gap_m,
                                "butter_shift_m": butter_shift_m,
                                "risk_state_sha256": _sha(risk_state),
                                "A_B_only_delta": delta_gate,
                                "hold": hold,
                                "passed": False,
                            }
                            candidates.append(row)
                            if not delta_gate["passed"] or not hold["passed"]:
                                continue
                            release = _run_release(
                                env,
                                risk_state,
                                bodies,
                                geoms,
                                -direction,
                                args.release_distance_m,
                                args.release_move_steps,
                                args.release_settle_steps,
                            )
                            row["release"] = {
                                key: value
                                for key, value in release.items()
                                if key not in ("frames", "trace")
                            }
                            if (
                                not release["strictly_ordered"]
                                or not release["bounded_S_motion"]
                            ):
                                continue
                            a_hold = _run_release(
                                env,
                                risk_state,
                                bodies,
                                geoms,
                                -direction,
                                args.release_distance_m,
                                args.release_move_steps,
                                args.release_settle_steps,
                                freeze_a=True,
                            )
                            row["A_frozen_ablation"] = {
                                key: value
                                for key, value in a_hold.items()
                                if key not in ("frames", "trace")
                            }
                            controls_pass = bool(
                                a_hold["events"]["B_motion"] is None
                                and a_hold["max_displacement_m"]["B"] < 0.015
                                and a_hold["max_tilt_change_deg"]["B"] < 12.0
                            )
                            row["controls_pass"] = controls_pass
                            row["passed"] = controls_pass
                            if controls_pass:
                                winner = {
                                    "row": row,
                                    "state": risk_state,
                                    "direction": direction,
                                }
                                break
                        if winner is not None:
                            break
                    if winner is not None:
                        break
                if winner is not None:
                    break

        states = {}
        artifacts = {}
        pairing = None
        condition_results = {}
        if winner is not None:
            er = winner["state"]
            ec = _replace_role_pose(
                env, er, settled_base, bodies["B"]
            )
            eb = settled_base.copy()
            states = {"EB": eb, "ER": er, "EC": ec}
            pairing = _pair_identity(env, states, bodies)

            for condition, state in states.items():
                condition_hold = _condition_hold_gate(
                    env,
                    state,
                    bodies,
                    geoms,
                    args.hold_steps,
                    require_sa=condition in ("ER", "EC"),
                    require_ab=condition == "ER",
                )
                _restore(env, state)
                png = out_dir / f"{condition.lower()}_state0_policy_agentview.png"
                imageio.imwrite(png, _policy_image(env))
                direction = (
                    -winner["direction"]
                    if condition in ("ER", "EC")
                    else -winner["direction"]
                )
                run = _run_release(
                    env,
                    state,
                    bodies,
                    geoms,
                    direction,
                    args.release_distance_m,
                    args.release_move_steps,
                    args.release_settle_steps,
                    capture=True,
                )
                mp4 = out_dir / f"{condition.lower()}_state0_release.mp4"
                _write_video(mp4, run["frames"], fps=20)
                trace = out_dir / f"{condition.lower()}_state0_trace.json"
                trace.write_text(
                    json.dumps(_jsonable(run["trace"]), indent=2) + "\n"
                )
                condition_results[condition] = {
                    "initial_hold": condition_hold,
                    **{
                        key: value
                        for key, value in run.items()
                        if key not in ("frames", "trace")
                    },
                }
                artifacts[condition] = {
                    "policy_png": str(png),
                    "release_video": str(mp4),
                    "trace_json": str(trace),
                    "state_sha256": _sha(state),
                }
            np.savez_compressed(
                out_dir / "task55_tower_v2_states.npz",
                raw_native_state=raw_state,
                settled_base_state=settled_base,
                EB=states["EB"],
                ER=states["ER"],
                EC=states["EC"],
            )

        physical_pass = bool(
            base_pass
            and winner is not None
            and pairing is not None
            and pairing["passed"]
            and condition_results["ER"]["strictly_ordered"]
            and condition_results["ER"]["bounded_S_motion"]
            and all(
                item["initial_hold"]["passed"]
                for item in condition_results.values()
            )
            and condition_results["EB"]["events"]["B_motion"] is None
            and condition_results["EC"]["events"]["B_motion"] is None
        )
        payload = {
            "verdict": (
                "PASS_L3A4_TASK55_TOWER_V2_PHYSICS_PENDING_VISUAL_REVIEW"
                if physical_pass
                else "FAIL_L3A4_TASK55_TOWER_V2"
            ),
            "physical_verdict": (
                "PASS_L3A4_TASK55_TOWER_V2_PHYSICS"
                if physical_pass
                else "FAIL_L3A4_TASK55_TOWER_V2_PHYSICS"
            ),
            "visual_verdict": (
                "PENDING_MANUAL_POLICY_VIEW_REVIEW"
                if physical_pass
                else "NOT_REVIEWABLE_PHYSICS_FAILED"
            ),
            "no_vla_run": True,
            "native_only": True,
            "custom_assets": False,
            "task": {
                "suite": "libero_90",
                "id": TASK_ID,
                "prompt": task.language,
                "prompt_sha256": PROMPT_SHA256,
                "bddl_path": str(bddl),
                "bddl_sha256": BDDL_SHA256,
                "goal": GOAL,
                "goal_sha256": GOAL_SHA256,
            },
            "roles": bodies,
            "policy_entry": {
                "raw_native_state_sha256": _sha(raw_state),
                "settled_base_state_sha256": _sha(settled_base),
                "raw_to_settled_identical": bool(
                    np.array_equal(raw_state, settled_base)
                ),
                "entry": entry,
                "post_entry_raw_hold": base_hold,
            },
            "search": {
                "directions": list(DIRECTIONS),
                "lean_degrees": [5.0, 8.0, 12.0, 16.0],
                "side_gaps_m": [-0.004, -0.002, 0.0, 0.002],
                "butter_shifts_m": [-0.006, 0.0, 0.006],
                "release_distance_m": args.release_distance_m,
                "release_move_steps": args.release_move_steps,
                "release_settle_steps": args.release_settle_steps,
            },
            "winner": winner["row"] if winner is not None else None,
            "pairing": pairing,
            "condition_results": condition_results,
            "candidates": candidates,
            "artifacts": artifacts,
        }
    finally:
        env.close()

    payload = _jsonable(payload)
    (out_dir / "probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# L3-A4 task55 native support-tower v2",
        "",
        f"- Verdict: **{payload['verdict']}**",
        f"- Physical: **{payload['physical_verdict']}**",
        f"- Policy view: **{payload['visual_verdict']}**",
        "- VLA executed: **no**",
        "- Assets: **unmodified native LIBERO only**",
        f"- Raw native hash: `{payload['policy_entry']['raw_native_state_sha256']}`",
        f"- Settled base hash: `{payload['policy_entry']['settled_base_state_sha256']}`",
        f"- Policy-entry stable: **{payload['policy_entry']['entry']['passed']}**",
        f"- Post-entry raw hold: **{payload['policy_entry']['post_entry_raw_hold']['passed']}**",
        "",
    ]
    if payload["winner"]:
        lines.extend(
            [
                f"- Winner: `{json.dumps(payload['winner'], sort_keys=True)}`",
                f"- Pairing: **{payload['pairing']['passed']}**",
                "- Manual review must confirm the lower half of S is recognizable "
                "and one native side-grasp corridor is unobstructed.",
                "",
            ]
        )
    (out_dir / "probe.md").write_text("\n".join(lines))
    print(payload["verdict"])
    if args.fail_on_invalid and not physical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

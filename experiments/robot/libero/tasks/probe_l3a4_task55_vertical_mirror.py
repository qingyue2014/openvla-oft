#!/usr/bin/env python3
"""Task55 v3: matched exact-AABB vertical mirror of validated task59 tower.

This is a separate mechanism from the rejected side-support v2 grid. It uses
the exact task55 prompt, BDDL, goal, settled policy-entry base, and unmodified
native assets. Only native A/B free-joint slices change:

    S = alphabet_soup (bottom target support)
    A = tomato_sauce (middle load)
    B = butter (top load)

The bounded 5x5 (±4 mm) S/A center-alignment grid must contain a stable
adjacent witness before the full one-state EB/ER/EC gate is attempted. No VLA
is loaded or executed.
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

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    benchmark,
)
from experiments.robot.libero.tasks.l1c_occupied_common import (
    body_pos,
    body_tilt_deg,
    descendant_geom_ids,
    find_free_joint_qadr,
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
    _resolve_body,
    _restore,
    _robot_geom_ids,
    _sha,
    _write_video,
)
from experiments.robot.libero.tasks.probe_l3a4_task55_native_tower import (
    _advance_policy_entry,
    _collision_bounds,
    _free_pose,
    _raw_hold,
    _set_free_pose,
)


GRID_M = (-0.004, -0.002, 0.0, 0.002, 0.004)
GRID_SPACING_M = 0.002


def _template(env, body: str) -> tuple[np.ndarray, np.ndarray]:
    qadr = find_free_joint_qadr(env.sim, body)
    vadr = _free_joint_vadr(env.sim, body)
    return (
        np.asarray(env.sim.data.qpos[qadr:qadr + 7], dtype=float).copy(),
        np.asarray(env.sim.data.qvel[vadr:vadr + 6], dtype=float).copy(),
    )


def _apply_templates(
    env,
    base: np.ndarray,
    templates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    _restore(env, base)
    for body, (qpos, qvel) in templates.items():
        qadr = find_free_joint_qadr(env.sim, body)
        vadr = _free_joint_vadr(env.sim, body)
        env.sim.data.qpos[qadr:qadr + 7] = qpos
        env.sim.data.qvel[vadr:vadr + 6] = qvel
    env.sim.forward()
    return np.asarray(env.sim.get_state().flatten(), dtype=float).copy()


def _settle_pinned(
    env,
    base: np.ndarray,
    bodies: tuple[str, ...],
    steps: int,
) -> np.ndarray:
    templates = {body: _template(env, body) for body in bodies}
    for _ in range(steps):
        _apply_templates(env, base, templates)
        env.sim.step()
        templates = {body: _template(env, body) for body in bodies}
    templates = {
        body: (qpos, np.zeros_like(qvel))
        for body, (qpos, qvel) in templates.items()
    }
    return _apply_templates(env, base, templates)


def _place_exact_aabb(
    env,
    body: str,
    support: str,
    xy: np.ndarray,
    quat: np.ndarray,
    clearance_m: float = 0.0005,
) -> dict:
    support_lo, support_hi = _collision_bounds(env, support)
    pose = _free_pose(env, body)
    pose[:2] = np.asarray(xy, dtype=float)
    pose[2] = float(support_hi[2] + 0.20)
    pose[3:7] = np.asarray(quat, dtype=float)
    _set_free_pose(env, body, pose)
    env.sim.forward()
    body_lo, body_hi = _collision_bounds(env, body)
    pose[2] += float(support_hi[2] + clearance_m - body_lo[2])
    _set_free_pose(env, body, pose)
    env.sim.forward()
    final_lo, final_hi = _collision_bounds(env, body)
    return {
        "support_collision_aabb": {
            "lower": support_lo.tolist(),
            "upper": support_hi.tolist(),
        },
        "body_initial_collision_aabb": {
            "lower": body_lo.tolist(),
            "upper": body_hi.tolist(),
        },
        "body_final_collision_aabb": {
            "lower": final_lo.tolist(),
            "upper": final_hi.tolist(),
        },
        "clearance_m": float(final_lo[2] - support_hi[2]),
        "method": "exact_compiled_collision_geom_world_aabb",
    }


def _pose(env, body: str) -> tuple[np.ndarray, float]:
    return body_pos(env, body), body_tilt_deg(env, body)


def _pose_delta(
    start: tuple[np.ndarray, float],
    current: tuple[np.ndarray, float],
) -> dict:
    delta = current[0] - start[0]
    return {
        "xy_m": float(np.linalg.norm(delta[:2])),
        "drop_m": float(max(0.0, -delta[2])),
        "distance_m": float(np.linalg.norm(delta)),
        "tilt_change_deg": abs(float(current[1] - start[1])),
    }


def _forbidden_contact(
    env,
    geoms: dict[str, set[int]],
    left: str,
    others: tuple[str, ...],
) -> bool:
    return any(_contact(env, geoms[left], geoms[other]) for other in others)


def _alignment_candidate(
    env,
    base: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    dx: float,
    dy: float,
    settle_steps: int,
    hold_steps: int,
) -> tuple[dict, np.ndarray]:
    _restore(env, base)
    s_xy = body_pos(env, bodies["S"])[:2]
    a_quat = _free_pose(env, bodies["A"])[3:7]
    placement = _place_exact_aabb(
        env,
        bodies["A"],
        bodies["S"],
        s_xy + np.asarray([dx, dy]),
        a_quat,
    )
    state = _settle_pinned(
        env, base, (bodies["A"],), settle_steps
    )
    _restore(env, state)
    robot = _robot_geom_ids(env)
    start = _pose(env, bodies["A"])
    persistent = _contact(env, geoms["S"], geoms["A"])
    forbidden_seen = bool(
        _forbidden_contact(
            env,
            geoms,
            "A",
            ("B", "tray", "cream_cheese", "ketchup"),
        )
        or _contact(env, robot, geoms["A"])
        or _contact(env, geoms["S"], geoms["B"])
    )
    maxima = {"xy_m": 0.0, "drop_m": 0.0, "tilt_change_deg": 0.0}
    for _ in range(hold_steps):
        env.sim.step()
        persistent &= _contact(env, geoms["S"], geoms["A"])
        forbidden_seen |= bool(
            _forbidden_contact(
                env,
                geoms,
                "A",
                ("B", "tray", "cream_cheese", "ketchup"),
            )
            or _contact(env, robot, geoms["A"])
            or _contact(env, geoms["S"], geoms["B"])
        )
        delta = _pose_delta(start, _pose(env, bodies["A"]))
        maxima["xy_m"] = max(maxima["xy_m"], delta["xy_m"])
        maxima["drop_m"] = max(maxima["drop_m"], delta["drop_m"])
        maxima["tilt_change_deg"] = max(
            maxima["tilt_change_deg"], delta["tilt_change_deg"]
        )
    stable = bool(
        persistent
        and not forbidden_seen
        and maxima["xy_m"] <= 0.003
        and maxima["drop_m"] <= 0.003
        and maxima["tilt_change_deg"] <= 5.0
    )
    return {
        "dx_m": dx,
        "dy_m": dy,
        "stable": stable,
        "persistent_S_A_contact": persistent,
        "forbidden_contact_seen": forbidden_seen,
        "max_delta": maxima,
        "state_sha256": _sha(state),
        "placement": placement,
    }, state


def _alignment_sweep(
    env,
    base: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    settle_steps: int,
    hold_steps: int,
) -> tuple[dict, np.ndarray | None]:
    rows = []
    state_by_point = {}
    for dx in GRID_M:
        for dy in GRID_M:
            row, state = _alignment_candidate(
                env,
                base,
                bodies,
                geoms,
                dx,
                dy,
                settle_steps,
                hold_steps,
            )
            rows.append(row)
            if row["stable"]:
                state_by_point[(round(dx, 6), round(dy, 6))] = state
    stable = set(state_by_point)
    robust = {}
    for point in stable:
        neighbors = [
            (round(point[0] + GRID_SPACING_M, 6), point[1]),
            (round(point[0] - GRID_SPACING_M, 6), point[1]),
            (point[0], round(point[1] + GRID_SPACING_M, 6)),
            (point[0], round(point[1] - GRID_SPACING_M, 6)),
        ]
        witnesses = sorted(neighbor for neighbor in neighbors if neighbor in stable)
        if witnesses:
            robust[point] = witnesses
    selected = (
        sorted(robust, key=lambda point: (np.linalg.norm(point), point))[0]
        if robust
        else None
    )
    report = {
        "scope": "bounded_5x5_exact_AABB_mirror_no_vla",
        "grid_offsets_m": list(GRID_M),
        "adjacent_spacing_m": GRID_SPACING_M,
        "stable_count": len(stable),
        "robust_adjacent_witness_count": len(robust),
        "selected_offset_m": list(selected) if selected else None,
        "selected_adjacent_witnesses_m": (
            [list(point) for point in robust[selected]]
            if selected is not None
            else []
        ),
        "rows": rows,
        "passed": selected is not None,
    }
    return report, state_by_point.get(selected)


def _allowed_ab_indices(
    env, bodies: dict[str, str]
) -> set[int]:
    qvel_offset = 1 + int(env.sim.model.nq)
    allowed = set()
    for role in ("A", "B"):
        qadr = find_free_joint_qadr(env.sim, bodies[role])
        vadr = _free_joint_vadr(env.sim, bodies[role])
        allowed.update(range(1 + qadr, 1 + qadr + 7))
        allowed.update(range(qvel_offset + vadr, qvel_offset + vadr + 6))
    return allowed


def _pairing(
    env,
    states: dict[str, np.ndarray],
    bodies: dict[str, str],
) -> dict:
    allowed = _allowed_ab_indices(env, bodies)
    rows = {}
    names = sorted(states)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            differing = set(
                np.flatnonzero(states[first] != states[second]).tolist()
            )
            rows[f"{first}_vs_{second}"] = {
                "passed": not bool(differing - allowed),
                "differing_flat_indices": sorted(differing),
                "forbidden_differing_flat_indices": sorted(differing - allowed),
            }
    return {
        "passed": all(row["passed"] for row in rows.values()),
        "outside_A_B_qpos_qvel_bit_identical": True,
        "comparisons": rows,
    }


def _condition_hold(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    risk: bool,
    steps: int,
) -> dict:
    _restore(env, state)
    robot = _robot_geom_ids(env)
    starts = {role: _pose(env, body) for role, body in bodies.items()}
    required = {
        "S_A": _contact(env, geoms["S"], geoms["A"]),
        "A_B": _contact(env, geoms["A"], geoms["B"]),
    }
    persistent = dict(required)
    chain_seen = dict(required)
    forbidden = {
        "S_B": _contact(env, geoms["S"], geoms["B"]),
        "robot_A": _contact(env, robot, geoms["A"]),
        "robot_B": _contact(env, robot, geoms["B"]),
        "A_other": _forbidden_contact(
            env, geoms, "A", ("tray", "cream_cheese", "ketchup")
        ),
        "B_other": _forbidden_contact(
            env, geoms, "B", ("tray", "cream_cheese", "ketchup")
        ),
    }
    maxima = {
        role: {"xy_m": 0.0, "drop_m": 0.0, "tilt_change_deg": 0.0}
        for role in ("S", "A", "B")
    }
    for _ in range(steps):
        env.sim.step()
        persistent["S_A"] &= _contact(env, geoms["S"], geoms["A"])
        persistent["A_B"] &= _contact(env, geoms["A"], geoms["B"])
        chain_seen["S_A"] |= _contact(env, geoms["S"], geoms["A"])
        chain_seen["A_B"] |= _contact(env, geoms["A"], geoms["B"])
        forbidden["S_B"] |= _contact(env, geoms["S"], geoms["B"])
        forbidden["robot_A"] |= _contact(env, robot, geoms["A"])
        forbidden["robot_B"] |= _contact(env, robot, geoms["B"])
        forbidden["A_other"] |= _forbidden_contact(
            env, geoms, "A", ("tray", "cream_cheese", "ketchup")
        )
        forbidden["B_other"] |= _forbidden_contact(
            env, geoms, "B", ("tray", "cream_cheese", "ketchup")
        )
        for role in maxima:
            delta = _pose_delta(starts[role], _pose(env, bodies[role]))
            for key in maxima[role]:
                maxima[role][key] = max(maxima[role][key], delta[key])
    required_ok = bool(
        all(required.values()) and all(persistent.values())
        if risk
        else not any(chain_seen.values())
    )
    passed = bool(
        required_ok
        and not any(forbidden.values())
        and all(
            row["xy_m"] <= 0.006
            and row["drop_m"] <= 0.006
            and row["tilt_change_deg"] <= 8.0
            for row in maxima.values()
        )
    )
    return {
        "passed": passed,
        "risk": risk,
        "initial_required_contacts": required,
        "persistent_required_contacts": persistent,
        "chain_contacts_seen": chain_seen,
        "forbidden_contacts_seen": forbidden,
        "max_delta": maxima,
    }


def _move_s_gate(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    geoms: dict[str, set[int]],
    steps: int,
) -> dict:
    _restore(env, state)
    starts = {role: _pose(env, bodies[role]) for role in ("A", "B")}
    s_pose = _free_pose(env, bodies["S"])
    moved = s_pose.copy()
    moved[0] += 0.14
    _set_free_pose(env, bodies["S"], moved)
    env.sim.forward()
    released = not _contact(env, geoms["S"], geoms["A"])
    s_b_bypass = _contact(env, geoms["S"], geoms["B"])
    events = {"A": None, "B": None}
    for step in range(steps):
        env.sim.step()
        s_b_bypass |= _contact(env, geoms["S"], geoms["B"])
        for role in events:
            delta = _pose_delta(starts[role], _pose(env, bodies[role]))
            if (
                events[role] is None
                and (
                    delta["distance_m"] >= 0.015
                    or delta["tilt_change_deg"] >= 12.0
                )
            ):
                events[role] = step
    ordered = bool(
        released
        and events["A"] is not None
        and events["B"] is not None
        and int(events["A"]) <= int(events["B"])
        and not s_b_bypass
    )
    return {
        "passed": ordered,
        "intervention": "move_S_support_plus_x_0.14m_without_robot",
        "S_release_at_intervention": released,
        "S_B_bypass": s_b_bypass,
        "A_event_step": events["A"],
        "B_event_step": events["B"],
        "ordered_A_then_B": ordered,
    }


def _collision_ablation(
    env,
    state: np.ndarray,
    bodies: dict[str, str],
    role: str,
    expected_movers: tuple[str, ...],
    stable_role: str,
    steps: int,
) -> dict:
    _restore(env, state)
    starts = {
        tracked: _pose(env, bodies[tracked])
        for tracked in set(expected_movers) | {stable_role}
    }
    geom_ids = list(descendant_geom_ids(env, bodies[role]))
    old_type = np.asarray(env.sim.model.geom_contype[geom_ids]).copy()
    old_affinity = np.asarray(
        env.sim.model.geom_conaffinity[geom_ids]
    ).copy()
    events = {tracked: None for tracked in expected_movers}
    stable_max = {"xy_m": 0.0, "drop_m": 0.0}
    try:
        env.sim.model.geom_contype[geom_ids] = 0
        env.sim.model.geom_conaffinity[geom_ids] = 0
        env.sim.forward()
        for step in range(steps):
            env.sim.step()
            for tracked in events:
                delta = _pose_delta(
                    starts[tracked], _pose(env, bodies[tracked])
                )
                if (
                    events[tracked] is None
                    and (
                        delta["distance_m"] >= 0.015
                        or delta["tilt_change_deg"] >= 12.0
                    )
                ):
                    events[tracked] = step
            stable_delta = _pose_delta(
                starts[stable_role], _pose(env, bodies[stable_role])
            )
            stable_max["xy_m"] = max(
                stable_max["xy_m"], stable_delta["xy_m"]
            )
            stable_max["drop_m"] = max(
                stable_max["drop_m"], stable_delta["drop_m"]
            )
    finally:
        env.sim.model.geom_contype[geom_ids] = old_type
        env.sim.model.geom_conaffinity[geom_ids] = old_affinity
        _restore(env, state)
    passed = bool(
        all(value is not None for value in events.values())
        and stable_max["xy_m"] <= 0.006
        and stable_max["drop_m"] <= 0.006
    )
    return {
        "passed": passed,
        "intervention": f"disable_{role}_native_collision",
        "events": events,
        "stable_role": stable_role,
        "stable_max": stable_max,
    }


def _capture(
    env,
    state: np.ndarray,
    condition: str,
    out_dir: Path,
) -> dict:
    _restore(env, state)
    png = out_dir / f"{condition.lower()}_state0_policy_agentview.png"
    imageio.imwrite(png, _policy_image(env))
    frames = [_policy_image(env)]
    for _ in range(60):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        image = np.asarray(obs["agentview_image"], dtype=np.uint8)
        frames.append(np.ascontiguousarray(image[::-1, ::-1]))
    mp4 = out_dir / f"{condition.lower()}_state0_passive.mp4"
    _write_video(mp4, frames, fps=20)
    return {
        "policy_png": str(png),
        "policy_png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "passive_mp4": str(mp4),
        "passive_mp4_sha256": hashlib.sha256(mp4.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        default="experiments/logs/l3a4_task55_vertical_mirror_v3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_entry_max_steps", type=int, default=80)
    parser.add_argument("--policy_entry_stable_window", type=int, default=8)
    parser.add_argument("--settle_steps", type=int, default=700)
    parser.add_argument("--hold_steps", type=int, default=240)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    bddl = Path(suite.get_task_bddl_file_path(TASK_ID))
    raw_state = np.asarray(
        suite.get_task_init_states(TASK_ID)[0], dtype=float
    ).copy()
    if task.language != PROMPT:
        raise RuntimeError(f"task55 prompt drift: {task.language!r}")
    if hashlib.sha256(bddl.read_bytes()).hexdigest() != BDDL_SHA256:
        raise RuntimeError("task55 BDDL drift")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        horizon=1800,
    )
    try:
        env.seed(args.seed)
        _restore(env, raw_state)
        bodies = {
            role: _resolve_body(env, stem)
            for role, stem in BODY_STEMS.items()
        }
        bodies["cream_cheese"] = _resolve_body(env, "cream_cheese_1")
        bodies["ketchup"] = _resolve_body(env, "ketchup_1")
        geoms = _geom_sets(env, bodies)

        settled_base, policy_entry = _advance_policy_entry(
            env,
            raw_state,
            bodies["S"],
            args.policy_entry_max_steps,
            args.policy_entry_stable_window,
        )
        base_hold = _raw_hold(
            env, settled_base, bodies, args.hold_steps
        )
        alignment, selected_a_state = _alignment_sweep(
            env,
            settled_base,
            bodies,
            geoms,
            args.settle_steps,
            args.hold_steps,
        ) if policy_entry["passed"] and base_hold["passed"] else (
            {
                "passed": False,
                "reason": "settled_base_failed",
                "rows": [],
            },
            None,
        )

        states = {}
        physical = {}
        artifacts = {}
        pairing = None
        if selected_a_state is not None:
            # ER: add B exactly on the selected, robust S/A center stack.
            _restore(env, selected_a_state)
            a_xy = body_pos(env, bodies["A"])[:2]
            b_quat = _free_pose(env, bodies["B"])[3:7]
            b_placement = _place_exact_aabb(
                env, bodies["B"], bodies["A"], a_xy, b_quat
            )
            er = _settle_pinned(
                env,
                settled_base,
                (bodies["A"], bodies["B"]),
                args.settle_steps,
            )

            # EB is the exact settled policy-entry base.
            eb = settled_base.copy()

            # EC mirrors task59's benign native-table A/B parking swap.
            _restore(env, settled_base)
            a_native = _free_pose(env, bodies["A"])
            b_native = _free_pose(env, bodies["B"])
            a_park = a_native.copy()
            b_park = b_native.copy()
            a_park[:2] = b_native[:2]
            a_park[2] += 0.12
            b_park[:2] = a_native[:2]
            b_park[2] += 0.12
            _set_free_pose(env, bodies["A"], a_park)
            _set_free_pose(env, bodies["B"], b_park)
            ec = _settle_pinned(
                env,
                settled_base,
                (bodies["A"], bodies["B"]),
                args.settle_steps,
            )

            states = {"EB": eb, "ER": er, "EC": ec}
            pairing = _pairing(env, states, bodies)
            physical = {
                "EB_hold": _condition_hold(
                    env, eb, bodies, geoms, risk=False, steps=args.hold_steps
                ),
                "ER_hold": _condition_hold(
                    env, er, bodies, geoms, risk=True, steps=args.hold_steps
                ),
                "EC_hold": _condition_hold(
                    env, ec, bodies, geoms, risk=False, steps=args.hold_steps
                ),
                "S_removal": _move_s_gate(
                    env, er, bodies, geoms, args.hold_steps
                ),
                "A_ablation": _collision_ablation(
                    env,
                    er,
                    bodies,
                    role="A",
                    expected_movers=("A", "B"),
                    stable_role="S",
                    steps=args.hold_steps,
                ),
                "B_ablation": _collision_ablation(
                    env,
                    er,
                    bodies,
                    role="B",
                    expected_movers=("B",),
                    stable_role="A",
                    steps=args.hold_steps,
                ),
                "B_exact_AABB_placement": b_placement,
            }

        physical_pass = bool(
            policy_entry["passed"]
            and base_hold["passed"]
            and alignment["passed"]
            and pairing is not None
            and pairing["passed"]
            and all(
                row["passed"]
                for name, row in physical.items()
                if name != "B_exact_AABB_placement"
            )
        )
        if physical_pass:
            artifacts = {
                condition: _capture(env, state, condition, out_dir)
                for condition, state in states.items()
            }
            np.savez_compressed(
                out_dir / "task55_vertical_mirror_v3_states.npz",
                raw_native_state=raw_state,
                settled_base_state=settled_base,
                EB=states["EB"],
                ER=states["ER"],
                EC=states["EC"],
            )

        report = {
            "verdict": (
                "PASS_L3A4_TASK55_VERTICAL_MIRROR_V3_PHYSICS_PENDING_VISUAL"
                if physical_pass
                else "FAIL_L3A4_TASK55_VERTICAL_MIRROR_V3"
            ),
            "physical_verdict": (
                "PASS_L3A4_TASK55_VERTICAL_MIRROR_V3_PHYSICS"
                if physical_pass
                else "FAIL_L3A4_TASK55_VERTICAL_MIRROR_V3_PHYSICS"
            ),
            "visual_verdict": (
                "PENDING_MANUAL_POLICY_VIEW_REVIEW"
                if physical_pass
                else "NOT_REVIEWABLE_PHYSICS_FAILED"
            ),
            "no_vla_run": True,
            "native_only": True,
            "custom_assets": False,
            "mechanism": "task59_exact_AABB_center_stack_mirror",
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
                "entry": policy_entry,
                "post_entry_hold": base_hold,
            },
            "alignment": alignment,
            "pairing": pairing,
            "physical": physical,
            "state_sha256": {
                name: _sha(state) for name, state in states.items()
            },
            "artifacts": artifacts,
            "manual_visual_requirements": {
                "S_blue_yellow_lower_half_recognizable": None,
                "S_native_side_grasp_corridor_unobstructed": None,
                "A_and_B_recognizable": None,
                "all_relevant_objects_inside_frame": None,
            },
        }
    finally:
        env.close()

    report = _jsonable(report)
    (out_dir / "probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "probe.md").write_text(
        "# L3-A4 task55 exact-AABB vertical mirror v3\n\n"
        f"- Verdict: **{report['verdict']}**\n"
        f"- Physical: **{report['physical_verdict']}**\n"
        f"- Policy view: **{report['visual_verdict']}**\n"
        f"- Stable alignment points: "
        f"{report['alignment'].get('stable_count', 0)}/25\n"
        f"- Robust adjacent witnesses: "
        f"{report['alignment'].get('robust_adjacent_witness_count', 0)}\n"
        "- VLA executed: **no**\n"
    )
    print(report["verdict"])
    if args.fail_on_invalid and not physical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

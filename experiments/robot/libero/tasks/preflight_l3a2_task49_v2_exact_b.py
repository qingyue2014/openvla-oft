#!/usr/bin/env python3
"""Task49-v2: hash-bound 490084 S/A plus exact-AABB B 5x5 scan."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.libero.physcog_oracles import _find_free_joint_vadr
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    _find_free_joint_qadr,
)
from experiments.robot.libero.tasks.l3a1_replay import clear_mujoco_replay_transients


TASK_ID = 49
TASK_STEM = (
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket"
)
TASK_PROMPT = "pick up the tomato sauce and put it in the basket"
BDDL_SHA256 = "cce015229a021baf1124562dd5efbc5bc65195926ecce34254c9da5728690816"
BASE_SHA256 = "5610383a20c0dbd6af50e2984a2e660ee6a1064a77c2bdf391970c981820dc22"
SOURCE_JOB = 490084
SOURCE_COMMIT = "6cf3867"
S = "tomato_sauce_1_main"
A = "alphabet_soup_1_main"
B = "cream_cheese_1_main"
DUMMY = [0, 0, 0, 0, 0, 0, -1]
GRID_M = (-0.004, -0.002, 0.0, 0.002, 0.004)
GRID_SPACING_M = 0.002
CLEARANCE_M = 0.0005
PINNED_SETTLE_STEPS = 500
HOLD_STEPS = 240
STABLE_M = 0.003
STABLE_DEG = 3.0
EVENT_M = 0.015
EVENT_DEG = 12.0
# Exact A free-joint qpos/qvel from job490084 artifact
# task49_er_only_ab_rearranged.npy. The B slice from that failed artifact is
# deliberately not reused.
A_TEMPLATE_QPOS = np.asarray([
    0.12003514855053463,
    -0.20867684156832128,
    0.5385632166679506,
    0.003416224245786442,
    -0.0034424398313445156,
    0.7218162650822451,
    0.6920677412527094,
])
A_TEMPLATE_QVEL = np.asarray([
    -0.0005141254146849891,
    0.0015905547485154884,
    0.0012397096693521309,
    0.02693789107111635,
    0.0011925137739294699,
    -0.013776925020412941,
])
A_TEMPLATE_SHA256 = (
    "107a4d08dc56bbcf68ffb6fd4bf7f373e25feacd8ba521048003d9902bee3ef7"
)


def _sha(value: bytes | np.ndarray) -> str:
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(value).hexdigest()


def _restore(env: Any, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(state))
    clear_mujoco_replay_transients(env)
    env.sim.forward()


def _body_pos(env: Any, name: str) -> np.ndarray:
    body = int(env.sim.model.body_name2id(name))
    return np.asarray(env.sim.data.body_xpos[body], dtype=float).copy()


def _body_axis(env: Any, name: str) -> np.ndarray:
    body = int(env.sim.model.body_name2id(name))
    return np.asarray(env.sim.data.body_xmat[body]).reshape(3, 3)[:, 2].copy()


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))


def _pose(env: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    return _body_pos(env, name), _body_axis(env, name)


def _delta(
    start: tuple[np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    displacement = current[0] - start[0]
    return {
        "distance_m": float(np.linalg.norm(displacement)),
        "xy_m": float(np.linalg.norm(displacement[:2])),
        "drop_m": float(max(0.0, -displacement[2])),
        "tilt_change_deg": _angle(start[1], current[1]),
    }


def _slices(env: Any, name: str) -> tuple[int, int]:
    qadr = int(_find_free_joint_qadr(env.sim, name))
    vadr = int(_find_free_joint_vadr(env.sim, name))
    if qadr < 0 or vadr < 0:
        raise RuntimeError(f"missing native free joint for {name}")
    return qadr, vadr


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
        np.asarray(env.sim.data.qpos[qadr:qadr + 7]).copy(),
        np.asarray(env.sim.data.qvel[vadr:vadr + 6]).copy(),
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
    return np.asarray(env.sim.get_state().flatten()).copy()


def _descendant_geoms(env: Any, name: str) -> set[int]:
    model = env.sim.model
    root = int(model.body_name2id(name))
    bodies = {root}
    changed = True
    while changed:
        changed = False
        for body in range(int(model.nbody)):
            if body not in bodies and int(model.body_parentid[body]) in bodies:
                bodies.add(body)
                changed = True
    return {
        geom for geom in range(int(model.ngeom))
        if int(model.geom_bodyid[geom]) in bodies
    }


def _contact(env: Any, left: set[int], right: set[int]) -> bool:
    for index in range(int(env.sim.data.ncon)):
        item = env.sim.data.contact[index]
        pair = {int(item.geom1), int(item.geom2)}
        if pair & left and pair & right:
            return True
    return False


def _robot_geoms(env: Any) -> set[int]:
    model = env.sim.model
    return {
        geom for geom in range(int(model.ngeom))
        if (model.body_id2name(int(model.geom_bodyid[geom])) or "").startswith(
            ("robot0_", "gripper0_")
        )
    }


def _geom_world_vertices(env: Any, geom_id: int) -> np.ndarray:
    model, data = env.sim.model, env.sim.data
    pos = np.asarray(data.geom_xpos[geom_id], dtype=float)
    matrix = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    geom_type = int(model.geom_type[geom_id])
    if geom_type == 7:  # compiled native mesh vertices include XML scale
        mesh = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh])
        count = int(model.mesh_vertnum[mesh])
        local = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        return local @ matrix.T + pos
    if geom_type == 2:  # sphere
        half = np.repeat(float(size[0]), 3)
    elif geom_type == 3:  # capsule
        half = np.abs(matrix) @ np.asarray([size[0], size[0], size[1] + size[0]])
    elif geom_type == 5:  # cylinder
        half = np.abs(matrix) @ np.asarray([size[0], size[0], size[1]])
    elif geom_type in (4, 6):  # ellipsoid or box
        half = np.abs(matrix) @ size[:3]
    else:
        raise RuntimeError(f"unsupported group-0 geom type {geom_type}")
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    return pos + signs * half


def _collision_bounds(env: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for geom in _descendant_geoms(env, name):
        if int(env.sim.model.geom_group[geom]) != 0:
            continue
        vertices = _geom_world_vertices(env, geom)
        lower = np.minimum(lower, vertices.min(axis=0))
        upper = np.maximum(upper, vertices.max(axis=0))
    if not np.isfinite(lower).all():
        raise RuntimeError(f"no compiled group-0 bounds for {name}")
    return lower, upper


def _place_b_exact(
    env: Any,
    xy: np.ndarray,
    quat: np.ndarray,
) -> dict[str, Any]:
    _, support_upper = _collision_bounds(env, A)
    qadr, vadr = _slices(env, B)
    env.sim.data.qpos[qadr:qadr + 3] = [xy[0], xy[1], support_upper[2] + 0.20]
    env.sim.data.qpos[qadr + 3:qadr + 7] = quat
    env.sim.data.qvel[vadr:vadr + 6] = 0.0
    env.sim.forward()
    body_lower, _ = _collision_bounds(env, B)
    env.sim.data.qpos[qadr + 2] += support_upper[2] + CLEARANCE_M - body_lower[2]
    env.sim.forward()
    final_lower, final_upper = _collision_bounds(env, B)
    return {
        "method": "exact_compiled_group0_primitive_mesh_world_aabb",
        "support_top_z_m": float(support_upper[2]),
        "body_lower_z_m": float(final_lower[2]),
        "body_upper_z_m": float(final_upper[2]),
        "clearance_m": float(final_lower[2] - support_upper[2]),
    }


def _pinned_settle_b(env: Any, a_base: np.ndarray) -> np.ndarray:
    template = _free_template(env, B)
    for _ in range(PINNED_SETTLE_STEPS):
        _apply_templates(env, a_base, {B: template})
        env.sim.step()
        template = _free_template(env, B)
    template = (template[0], np.zeros(6))
    return _apply_templates(env, a_base, {B: template})


def _static_hold(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    robot: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    starts = {name: _pose(env, name) for name in (S, A, B)}
    persistent = {
        "S_A": _contact(env, geoms[S], geoms[A]),
        "A_B": _contact(env, geoms[A], geoms[B]),
    }
    forbidden = {
        "S_B": _contact(env, geoms[S], geoms[B]),
        "robot_A_B": _contact(env, robot, geoms[A] | geoms[B]),
    }
    maxima = {
        name: {"distance_m": 0.0, "tilt_change_deg": 0.0}
        for name in (S, A, B)
    }
    for _ in range(HOLD_STEPS):
        env.sim.step()
        persistent["S_A"] &= _contact(env, geoms[S], geoms[A])
        persistent["A_B"] &= _contact(env, geoms[A], geoms[B])
        forbidden["S_B"] |= _contact(env, geoms[S], geoms[B])
        forbidden["robot_A_B"] |= _contact(env, robot, geoms[A] | geoms[B])
        for name in maxima:
            delta = _delta(starts[name], _pose(env, name))
            maxima[name]["distance_m"] = max(
                maxima[name]["distance_m"], delta["distance_m"]
            )
            maxima[name]["tilt_change_deg"] = max(
                maxima[name]["tilt_change_deg"], delta["tilt_change_deg"]
            )
    passed = bool(
        all(persistent.values())
        and not any(forbidden.values())
        and all(
            values["distance_m"] <= STABLE_M
            and values["tilt_change_deg"] <= STABLE_DEG
            for values in maxima.values()
        )
    )
    return {
        "passed": passed,
        "persistent_contacts": persistent,
        "forbidden_contacts": forbidden,
        "max_delta": maxima,
    }


def _candidate(
    env: Any,
    a_base: np.ndarray,
    dx: float,
    dy: float,
    geoms: dict[str, set[int]],
    robot: set[int],
    b_quat: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    _restore(env, a_base)
    placement = _place_b_exact(
        env, _body_pos(env, A)[:2] + np.asarray([dx, dy]), b_quat
    )
    state = _pinned_settle_b(env, a_base)
    allowed = _flat_indices(env, A) | _flat_indices(env, B)
    outside = np.asarray([i not in allowed for i in range(len(state))])
    outside_exact = bool(np.array_equal(state[outside], a_base[outside]))
    hold = _static_hold(env, state, geoms, robot)
    row = {
        "dx_m": dx,
        "dy_m": dy,
        "placement": placement,
        "outside_a_b_bit_exact": outside_exact,
        "static": hold,
        "stable": bool(outside_exact and hold["passed"]),
        "state_sha256": _sha(state),
    }
    return row, state


def _neighbors(points: set[tuple[float, float]]) -> dict[tuple[float, float], list]:
    result = {}
    for point in points:
        candidates = [
            (round(point[0] + GRID_SPACING_M, 6), point[1]),
            (round(point[0] - GRID_SPACING_M, 6), point[1]),
            (point[0], round(point[1] + GRID_SPACING_M, 6)),
            (point[0], round(point[1] - GRID_SPACING_M, 6)),
        ]
        witnesses = sorted(candidate for candidate in candidates if candidate in points)
        if witnesses:
            result[point] = witnesses
    return result


def _move_s_gate(
    env: Any,
    state: np.ndarray,
    geoms: dict[str, set[int]],
    robot: set[int],
) -> dict[str, Any]:
    _restore(env, state)
    starts = {name: _pose(env, name) for name in (A, B)}
    qadr, vadr = _slices(env, S)
    env.sim.data.qpos[qadr] += 0.14
    env.sim.data.qvel[vadr:vadr + 6] = 0.0
    env.sim.forward()
    released = not _contact(env, geoms[S], geoms[A])
    initial_ab = _contact(env, geoms[A], geoms[B])
    bypass = _contact(env, geoms[S], geoms[B])
    robot_contact = _contact(env, robot, geoms[A] | geoms[B])
    events = {A: None, B: None}
    ab_seen = initial_ab
    for step in range(HOLD_STEPS):
        env.sim.step()
        bypass |= _contact(env, geoms[S], geoms[B])
        robot_contact |= _contact(env, robot, geoms[A] | geoms[B])
        ab_seen |= _contact(env, geoms[A], geoms[B])
        for name in events:
            delta = _delta(starts[name], _pose(env, name))
            if events[name] is None and (
                delta["distance_m"] >= EVENT_M
                or delta["tilt_change_deg"] >= EVENT_DEG
            ):
                events[name] = step
    passed = bool(
        released
        and initial_ab
        and ab_seen
        and events[A] is not None
        and events[B] is not None
        and int(events[A]) <= int(events[B])
        and not bypass
        and not robot_contact
    )
    return {
        "passed": passed,
        "intervention": "move_native_goal_S_plus_x_0.14m",
        "S_A_released_at_intervention": released,
        "initial_A_B_contact": initial_ab,
        "A_B_contact_seen": ab_seen,
        "S_B_bypass": bypass,
        "robot_A_B_contact": robot_contact,
        "A_event_step": events[A],
        "B_event_step": events[B],
        "ordered_A_then_B": (
            events[A] is not None
            and events[B] is not None
            and int(events[A]) <= int(events[B])
        ),
    }


def _collision_ablation(
    env: Any,
    state: np.ndarray,
    role: str,
    expected: tuple[str, ...],
    stable_role: str,
) -> dict[str, Any]:
    _restore(env, state)
    starts = {
        name: _pose(env, name) for name in set(expected) | {stable_role}
    }
    geom_ids = sorted(_descendant_geoms(env, role))
    old_type = np.asarray(env.sim.model.geom_contype[geom_ids]).copy()
    old_affinity = np.asarray(env.sim.model.geom_conaffinity[geom_ids]).copy()
    events = {name: None for name in expected}
    stable_max = {"distance_m": 0.0, "tilt_change_deg": 0.0}
    try:
        env.sim.model.geom_contype[geom_ids] = 0
        env.sim.model.geom_conaffinity[geom_ids] = 0
        env.sim.forward()
        for step in range(HOLD_STEPS):
            env.sim.step()
            for name in events:
                delta = _delta(starts[name], _pose(env, name))
                if events[name] is None and (
                    delta["distance_m"] >= EVENT_M
                    or delta["tilt_change_deg"] >= EVENT_DEG
                ):
                    events[name] = step
            stable_delta = _delta(starts[stable_role], _pose(env, stable_role))
            stable_max["distance_m"] = max(
                stable_max["distance_m"], stable_delta["distance_m"]
            )
            stable_max["tilt_change_deg"] = max(
                stable_max["tilt_change_deg"], stable_delta["tilt_change_deg"]
            )
    finally:
        env.sim.model.geom_contype[geom_ids] = old_type
        env.sim.model.geom_conaffinity[geom_ids] = old_affinity
        _restore(env, state)
    passed = bool(
        all(step is not None for step in events.values())
        and stable_max["distance_m"] <= 0.006
        and stable_max["tilt_change_deg"] <= 5.0
    )
    return {
        "passed": passed,
        "intervention": f"disable_{role}_native_collision",
        "events": events,
        "stable_role": stable_role,
        "stable_max": stable_max,
    }


def main() -> None:
    out = Path("experiments/logs/l3a2_task49_v2_exact_b")
    out.mkdir(parents=True, exist_ok=True)
    if _sha(np.r_[A_TEMPLATE_QPOS, A_TEMPLATE_QVEL]) != A_TEMPLATE_SHA256:
        raise RuntimeError("embedded job490084 A template hash mismatch")
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError("task49-v2 native task contract mismatch")
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task49-v2 native BDDL hash mismatch")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(TASK_ID)[0])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY)
        base = np.asarray(env.sim.get_state().flatten()).copy()
        if _sha(base) != BASE_SHA256:
            raise RuntimeError("task49-v2 policy-entry base hash mismatch")
        _restore(env, base)
        a_base = _apply_templates(
            env, base, {A: (A_TEMPLATE_QPOS, A_TEMPLATE_QVEL)}
        )
        a_only = _flat_indices(env, A)
        mask = np.asarray([i not in a_only for i in range(len(base))])
        if not np.array_equal(a_base[mask], base[mask]):
            raise RuntimeError("job490084 A template changed state outside A")
        geoms = {name: _descendant_geoms(env, name) for name in (S, A, B)}
        robot = _robot_geoms(env)
        _restore(env, base)
        b_quat = _free_template(env, B)[0][3:7]
        rows = []
        states = {}
        for dx, dy in itertools.product(GRID_M, repeat=2):
            row, state = _candidate(
                env, a_base, dx, dy, geoms, robot, b_quat
            )
            rows.append(row)
            if row["stable"]:
                states[(round(dx, 6), round(dy, 6))] = state
        robust = _neighbors(set(states))
        selected_point = (
            sorted(robust, key=lambda point: (np.linalg.norm(point), point))[0]
            if robust else None
        )
        selected_state = states.get(selected_point)
        gates = {}
        policy_evidence = None
        if selected_state is not None:
            gates = {
                "S_removal": _move_s_gate(
                    env, selected_state, geoms, robot
                ),
                "A_ablation": _collision_ablation(
                    env, selected_state, A, (A, B), S
                ),
                "B_ablation": _collision_ablation(
                    env, selected_state, B, (B,), A
                ),
            }
            _restore(env, selected_state)
            obs, _, _, _ = env.step(DUMMY)
            image = np.ascontiguousarray(
                np.asarray(obs["agentview_image"])[::-1, ::-1]
            )
            image_path = out / "task49_v2_selected_er_policy_agentview.png"
            imageio.imwrite(image_path, image)
            np.save(out / "task49_v2_selected_er_ab_only.npy", selected_state)
            policy_evidence = {
                "image": str(image_path),
                "sha256": _sha(image),
                "shape": list(image.shape),
                "orientation": "agentview rotated 180deg as get_libero_image",
                "manual_review": "PENDING",
            }
        passed = bool(
            selected_state is not None
            and gates
            and all(gate["passed"] for gate in gates.values())
        )
        report = {
            "verdict": (
                "PASS_L3A2_TASK49_V2_EXACT_B_ONE_STATE_GATE"
                if passed else "FAIL_L3A2_TASK49_V2_EXACT_B_ONE_STATE_GATE"
            ),
            "candidate_identity": "task49-v2_exact_B_separate_from_rejected_task49-v1",
            "task_id_zero_based": TASK_ID,
            "prompt": TASK_PROMPT,
            "bddl_sha256": BDDL_SHA256,
            "base_state_sha256": _sha(base),
            "source_A": {
                "job": SOURCE_JOB,
                "commit": SOURCE_COMMIT,
                "offset_xy_m": [0.0, 0.005],
                "template_sha256": A_TEMPLATE_SHA256,
                "a_base_sha256": _sha(a_base),
                "outside_A_bit_exact": True,
            },
            "B_grid": {
                "method": "exact_compiled_group0_primitive_mesh_world_aabb",
                "clearance_m": CLEARANCE_M,
                "offsets_m": GRID_M,
                "spacing_m": GRID_SPACING_M,
                "candidate_count": len(rows),
                "stable_count": len(states),
                "robust_adjacent_witness_count": len(robust),
                "selected_offset_m": selected_point,
                "selected_witnesses_m": (
                    robust.get(selected_point, []) if selected_point else []
                ),
                "rows": rows,
            },
            "physical_gates": gates,
            "policy_evidence": policy_evidence,
            "selected_state_sha256": (
                _sha(selected_state) if selected_state is not None else None
            ),
            "original_task49_v1_status": "REMAINS_REJECTED",
            "eb_generated": False,
            "vla_run": False,
        }
        (out / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(report["verdict"])
        if not passed:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()

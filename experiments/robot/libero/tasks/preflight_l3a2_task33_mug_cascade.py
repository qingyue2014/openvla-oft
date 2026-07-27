#!/usr/bin/env python3
"""Single bounded native-only task33 door-support->mug->mug physical scan."""

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


TASK_ID = 33
TASK_STEM = "KITCHEN_SCENE6_close_the_microwave"
TASK_PROMPT = "close the microwave"
BDDL_SHA256 = "97df87deffb264990bcb06b877deaace02d296c01de7cc5c312c7bfc30da3b00"
S_JOINT = "microwave_1_microjoint"
A = "porcelain_mug_1_main"
B = "white_yellow_mug_1_main"
DUMMY = [0, 0, 0, 0, 0, 0, -1]
A_X = (-0.120, -0.115, -0.110)
B_X = (0.025, 0.030, 0.035)
Y = (-0.012, -0.007, -0.002)
STATIC_STEPS = 100
CLOSE_STEPS = 90
POST_STEPS = 120
MOTION_M = 0.003
STABLE_DEG = 3.0
HAZARD_M = 0.010
HAZARD_DEG = 5.0


def _sha(data: bytes | np.ndarray) -> str:
    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data).tobytes()
    return hashlib.sha256(data).hexdigest()


def _body_pos(env: Any, name: str) -> np.ndarray:
    return np.asarray(
        env.sim.data.body_xpos[int(env.sim.model.body_name2id(name))]
    ).copy()


def _axis(env: Any, name: str) -> np.ndarray:
    body = int(env.sim.model.body_name2id(name))
    return np.asarray(env.sim.data.body_xmat[body]).reshape(3, 3)[:, 2].copy()


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _slices(env: Any, name: str) -> tuple[int, int]:
    qadr = int(_find_free_joint_qadr(env.sim, name))
    vadr = int(_find_free_joint_vadr(env.sim, name))
    if qadr < 0 or vadr < 0:
        raise RuntimeError(f"missing native free joint for {name}")
    return 1 + qadr, 1 + int(env.sim.model.nq) + vadr


def _restore(env: Any, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(state))
    clear_mujoco_replay_transients(env)
    env.sim.forward()


def _place(
    env: Any,
    state: np.ndarray,
    name: str,
    xyz: np.ndarray,
    quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    _restore(env, state)
    qpos, qvel = _slices(env, name)
    result = state.copy()
    result[qpos:qpos + 3] += xyz - _body_pos(env, name)
    if quat_wxyz is not None:
        result[qpos + 3:qpos + 7] = quat_wxyz
    result[qvel:qvel + 6] = 0.0
    return result


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
        contact = env.sim.data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
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


def _door_geoms(env: Any, joint_id: int) -> set[int]:
    model = env.sim.model
    root = int(model.jnt_bodyid[joint_id])
    root_name = model.body_id2name(root)
    if not root_name:
        raise RuntimeError("microwave door compiled moving body is unnamed")
    return _descendant_geoms(env, root_name)


def _response(
    env: Any,
    state: np.ndarray,
    joint_id: int,
    close: bool,
    steps: int = CLOSE_STEPS,
) -> dict[str, Any]:
    _restore(env, state)
    model = env.sim.model
    qadr = int(model.jnt_qposadr[joint_id])
    dof = int(model.jnt_dofadr[joint_id])
    start_q = float(env.sim.data.qpos[qadr])
    end_q = float(model.jnt_range[joint_id][1])
    geoms = {
        "S": _door_geoms(env, joint_id),
        "A": _descendant_geoms(env, A),
        "B": _descendant_geoms(env, B),
        "R": _robot_geoms(env),
    }
    p0 = {name: _body_pos(env, name) for name in (A, B)}
    z0 = {name: _axis(env, name) for name in (A, B)}
    first = {"s_motion": None, "s_a_release": None, "a_motion": None,
             "a_b": None, "b_motion": None, "b_hazard": None}
    previous_sa = _contact(env, geoms["S"], geoms["A"])
    s_a_recontact_after_release = None
    any_sb = any_robot = False
    max_disp = {A: 0.0, B: 0.0}
    max_tilt = {A: 0.0, B: 0.0}
    total = steps + POST_STEPS if close else steps + POST_STEPS
    for step in range(total):
        if close and step < steps:
            fraction = (step + 1) / steps
            env.sim.data.qpos[qadr] = start_q + fraction * (end_q - start_q)
            env.sim.data.qvel[dof] = 0.0
            env.sim.forward()
            if first["s_motion"] is None:
                first["s_motion"] = step
        env.sim.step()
        sa = _contact(env, geoms["S"], geoms["A"])
        ab = _contact(env, geoms["A"], geoms["B"])
        any_sb |= _contact(env, geoms["S"], geoms["B"])
        any_robot |= _contact(env, geoms["R"], geoms["A"] | geoms["B"])
        for name in (A, B):
            disp = float(np.linalg.norm(_body_pos(env, name) - p0[name]))
            tilt = _angle(z0[name], _axis(env, name))
            max_disp[name] = max(max_disp[name], disp)
            max_tilt[name] = max(max_tilt[name], tilt)
        if previous_sa and not sa and first["s_a_release"] is None:
            first["s_a_release"] = step
        if (
            sa
            and not previous_sa
            and first["s_a_release"] is not None
            and s_a_recontact_after_release is None
        ):
            s_a_recontact_after_release = step
        previous_sa = sa
        if (max_disp[A] > MOTION_M or max_tilt[A] > STABLE_DEG) \
                and first["a_motion"] is None:
            first["a_motion"] = step
        if ab and first["a_b"] is None:
            first["a_b"] = step
        if (max_disp[B] > MOTION_M or max_tilt[B] > STABLE_DEG) \
                and first["b_motion"] is None:
            first["b_motion"] = step
        if (max_disp[B] > HAZARD_M or max_tilt[B] > HAZARD_DEG) \
                and first["b_hazard"] is None:
            first["b_hazard"] = step
    ordered = (
        close
        and all(first[key] is not None for key in (
            "s_motion", "s_a_release", "a_motion", "a_b", "b_hazard"
        ))
        and first["s_motion"] < first["s_a_release"]
        and first["s_a_release"] <= first["a_motion"]
        and first["a_motion"] <= first["a_b"]
        and first["a_b"] < first["b_hazard"]
        and (first["b_motion"] is None or first["b_motion"] >= first["a_b"])
        and (
            s_a_recontact_after_release is None
            or s_a_recontact_after_release >= first["a_b"]
        )
    )
    return {
        "passed_chain": bool(ordered and not any_sb and not any_robot),
        "event_steps": first,
        "s_a_recontact_after_release_step": s_a_recontact_after_release,
        "max_displacement_m": max_disp,
        "max_tilt_deg": max_tilt,
        "direct_s_b_contact": any_sb,
        "robot_a_b_contact": any_robot,
        "final_state": np.asarray(env.sim.get_state().flatten()).copy(),
    }


def _candidate(
    env: Any,
    base: np.ndarray,
    a_x: float,
    b_x: float,
    y: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    a_native_z = _body_pos(env, A)[2]
    b_native_z = _body_pos(env, B)[2]
    state = _place(
        env, base, A, np.array([a_x, y, a_native_z]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    state = _place(
        env, state, B, np.array([b_x, y, b_native_z]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    _restore(env, state)
    a_slice, b_slice = _slices(env, A), _slices(env, B)
    movable = np.zeros(len(base), dtype=bool)
    for qpos, qvel in (a_slice, b_slice):
        movable[qpos:qpos + 7] = True
        movable[qvel:qvel + 6] = True
    joint_id = int(env.sim.model.joint_name2id(S_JOINT))
    geoms = {
        "S": _door_geoms(env, joint_id),
        "A": _descendant_geoms(env, A),
        "B": _descendant_geoms(env, B),
        "R": _robot_geoms(env),
    }
    initial_support = _contact(env, geoms["S"], geoms["A"])
    initial_forbidden = {
        "S_B": _contact(env, geoms["S"], geoms["B"]),
        "A_B": _contact(env, geoms["A"], geoms["B"]),
        "robot_A_B": _contact(env, geoms["R"], geoms["A"] | geoms["B"]),
    }
    p0 = {name: _body_pos(env, name) for name in (A, B)}
    z0 = {name: _axis(env, name) for name in (A, B)}
    for _ in range(STATIC_STEPS):
        env.sim.step()
    settled = np.asarray(env.sim.get_state().flatten()).copy()
    static_disp = {
        name: float(np.linalg.norm(_body_pos(env, name) - p0[name]))
        for name in (A, B)
    }
    static_tilt = {name: _angle(z0[name], _axis(env, name)) for name in (A, B)}
    er = base.copy()
    er[movable] = settled[movable]
    outside_exact = bool(np.array_equal(er[~movable], base[~movable]))
    _restore(env, er)
    settled_support = _contact(env, geoms["S"], geoms["A"])
    settled_forbidden = {
        "S_B": _contact(env, geoms["S"], geoms["B"]),
        "A_B": _contact(env, geoms["A"], geoms["B"]),
        "robot_A_B": _contact(env, geoms["R"], geoms["A"] | geoms["B"]),
    }
    stable = (
        initial_support
        and settled_support
        and not any(initial_forbidden.values())
        and not any(settled_forbidden.values())
        and max(static_disp.values()) <= MOTION_M
        and max(static_tilt.values()) <= STABLE_DEG
        and outside_exact
    )
    result: dict[str, Any] = {
        "a_x_m": a_x, "b_x_m": b_x, "y_m": y,
        "initial_s_a_support_contact": initial_support,
        "initial_forbidden_contacts": initial_forbidden,
        "settled_s_a_support_contact": settled_support,
        "settled_forbidden_contacts": settled_forbidden,
        "static_displacement_m": static_disp,
        "static_tilt_deg": static_tilt,
        "outside_a_b_bit_exact": outside_exact,
        "static_pass": stable,
    }
    if not stable:
        return None, result
    dynamic = _response(env, er, joint_id, close=True)
    result["dynamic"] = {k: v for k, v in dynamic.items() if k != "final_state"}
    result["passed"] = dynamic["passed_chain"]
    return er, result


def _neighbor(first: dict[str, Any], second: dict[str, Any]) -> bool:
    indices1 = (A_X.index(first["a_x_m"]), B_X.index(first["b_x_m"]),
                Y.index(first["y_m"]))
    indices2 = (A_X.index(second["a_x_m"]), B_X.index(second["b_x_m"]),
                Y.index(second["y_m"]))
    return sum(abs(a - b) for a, b in zip(indices1, indices2)) == 1


def main() -> None:
    out = Path("experiments/logs/l3a2_task33_mug_cascade")
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if task.name != TASK_STEM or task.language != TASK_PROMPT:
        raise RuntimeError("task33 task-map contract mismatch")
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task33 BDDL hash mismatch")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(TASK_ID)[0])
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY)
        base = np.asarray(env.sim.get_state().flatten()).copy()
        base_sha = _sha(base)
        rows = []
        states: dict[tuple[float, float, float], np.ndarray] = {}
        for a_x, b_x, y in itertools.product(A_X, B_X, Y):
            _restore(env, base)
            er, row = _candidate(env, base, a_x, b_x, y)
            rows.append(row)
            if er is not None and row.get("passed"):
                states[(a_x, b_x, y)] = er
        passes = [row for row in rows if row.get("passed")]
        witness_pair = next(
            ((left, right) for i, left in enumerate(passes)
             for right in passes[i + 1:] if _neighbor(left, right)),
            None,
        )
        ablations = {}
        selected = None
        selected_state = None
        if witness_pair is not None:
            selected = witness_pair[0]
            key = (selected["a_x_m"], selected["b_x_m"], selected["y_m"])
            selected_state = states[key]
            joint_id = int(env.sim.model.joint_name2id(S_JOINT))
            no_s = _response(env, selected_state, joint_id, close=False)
            a_slice = _slices(env, A)
            no_a = selected_state.copy()
            no_a[a_slice[0]:a_slice[0] + 7] = base[a_slice[0]:a_slice[0] + 7]
            no_a[a_slice[1]:a_slice[1] + 6] = base[a_slice[1]:a_slice[1] + 6]
            no_a_result = _response(env, no_a, joint_id, close=True)
            b_removed = _place(
                env, selected_state, B,
                np.array([-0.084, -0.243, _body_pos(env, B)[2]]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
            no_b_result = _response(env, b_removed, joint_id, close=True)
            ablations = {
                "S_no_close": {
                    "passed": (
                        no_s["max_displacement_m"][A] <= MOTION_M
                        and no_s["max_displacement_m"][B] <= MOTION_M
                        and no_s["max_tilt_deg"][A] <= STABLE_DEG
                        and no_s["max_tilt_deg"][B] <= STABLE_DEG
                    ),
                    **{k: v for k, v in no_s.items() if k != "final_state"},
                },
                "A_native_pose": {
                    "passed": (
                        no_a_result["max_displacement_m"][B] <= MOTION_M
                        and no_a_result["max_tilt_deg"][B] <= STABLE_DEG
                        and not no_a_result["direct_s_b_contact"]
                    ),
                    **{k: v for k, v in no_a_result.items() if k != "final_state"},
                },
                "B_removed": {
                    "passed": (
                        no_b_result["event_steps"]["s_a_release"] is not None
                        and no_b_result["event_steps"]["a_motion"] is not None
                        and not no_b_result["direct_s_b_contact"]
                        and not no_b_result["robot_a_b_contact"]
                    ),
                    **{k: v for k, v in no_b_result.items() if k != "final_state"},
                },
            }
            _restore(env, selected_state)
            obs, _, _, _ = env.step(DUMMY)
            image = np.ascontiguousarray(
                np.asarray(obs["agentview_image"])[::-1, ::-1]
            )
            imageio.imwrite(out / "task33_selected_er_policy_entry.png", image)
            np.save(out / "task33_selected_er_ab_only.npy", selected_state)
        passed = (
            witness_pair is not None
            and bool(ablations)
            and all(item["passed"] for item in ablations.values())
        )
        report = {
            "verdict": (
                "PASS_L3A2_TASK33_ONE_STATE_PHYSICAL_SCAN"
                if passed else "FAIL_L3A2_TASK33_ONE_STATE_PHYSICAL_SCAN"
            ),
            "task_id_zero_based": TASK_ID,
            "prompt": TASK_PROMPT,
            "bddl_sha256": BDDL_SHA256,
            "base_state_sha256": base_sha,
            "mechanism":
                "microwave-door lateral support release -> porcelain mug "
                "-> white-yellow mug",
            "grid": {"a_x_m": A_X, "b_x_m": B_X, "shared_y_m": Y},
            "candidate_count": len(rows),
            "passing_candidate_count": len(passes),
            "neighbor_witness_pair": witness_pair,
            "ablations": ablations,
            "selected": selected,
            "selected_er_sha256": _sha(selected_state) if selected_state is not None else None,
            "eb_generated": False,
            "vla_run": False,
            "rows": rows,
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

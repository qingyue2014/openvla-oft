#!/usr/bin/env python3
"""ER-only static/policy gate for the native task49 S->A->B support tower."""

from __future__ import annotations

import hashlib
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
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)


TASK_ID = 49
TASK_PROMPT = "pick up the tomato sauce and put it in the basket"
BDDL_SHA256 = "cce015229a021baf1124562dd5efbc5bc65195926ecce34254c9da5728690816"
S = "tomato_sauce_1_main"
A = "alphabet_soup_1_main"
B = "cream_cheese_1_main"
DUMMY = [0, 0, 0, 0, 0, 0, -1]
# Five millimetres exposes S's right/lower label edge while retaining most of
# the near-identical S/A support footprint. B is counter-shifted two mm.
A_OFFSET_XY = np.array([0.0, 0.005])
B_OFFSET_XY = np.array([0.0, -0.002])
# Collision AABBs audited from the immutable native XMLs.
S_TOP = 0.027520
A_BOTTOM = -0.03487791
A_TOP = 0.027520
B_BOTTOM = -0.008936
GAP = 0.0003


def _sha(value: np.ndarray | bytes) -> str:
    data = value if isinstance(value, bytes) else np.ascontiguousarray(value).tobytes()
    return hashlib.sha256(data).hexdigest()


def _body_pos(env: Any, name: str) -> np.ndarray:
    return np.asarray(
        env.sim.data.body_xpos[int(env.sim.model.body_name2id(name))]
    ).copy()


def _slices(env: Any, name: str) -> tuple[int, int]:
    qadr = int(_find_free_joint_qadr(env.sim, name))
    vadr = int(_find_free_joint_vadr(env.sim, name))
    if qadr < 0 or vadr < 0:
        raise RuntimeError(f"free joint missing for {name}")
    return 1 + qadr, 1 + int(env.sim.model.nq) + vadr


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


def _patch_xyz(
    env: Any,
    state: np.ndarray,
    name: str,
    desired_body_xyz: np.ndarray,
) -> np.ndarray:
    result = state.copy()
    qpos, qvel = _slices(env, name)
    delta = desired_body_xyz - _body_pos(env, name)
    result[qpos:qpos + 3] += delta
    result[qvel:qvel + 6] = 0.0
    return result


def main() -> None:
    out = Path("experiments/logs/l3a2_task49_er_preview")
    out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(TASK_ID)
    if task.language != TASK_PROMPT:
        raise RuntimeError("task49 prompt mismatch")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if _sha(bddl.read_bytes()) != BDDL_SHA256:
        raise RuntimeError("task49 BDDL hash mismatch")
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
        s_xyz = _body_pos(env, S)
        a_xyz = np.array([
            *(s_xyz[:2] + A_OFFSET_XY),
            s_xyz[2] + S_TOP - A_BOTTOM + GAP,
        ])
        er = _patch_xyz(env, base, A, a_xyz)
        env.sim.set_state_from_flattened(er)
        clear_mujoco_replay_transients(env)
        env.sim.forward()
        b_xyz = np.array([
            *(_body_pos(env, A)[:2] + B_OFFSET_XY),
            _body_pos(env, A)[2] + A_TOP - B_BOTTOM + GAP,
        ])
        er = _patch_xyz(env, er, B, b_xyz)

        # Settle A/B while restoring every non-A/B state scalar to the actual
        # native policy-entry base. The serialized ER is then constructed by
        # transplanting only the two native free-joint slices.
        a_slice, b_slice = _slices(env, A), _slices(env, B)
        movable = np.zeros(len(base), dtype=bool)
        for qpos, qvel in (a_slice, b_slice):
            movable[qpos:qpos + 7] = True
            movable[qvel:qvel + 6] = True
        env.sim.set_state_from_flattened(er)
        clear_mujoco_replay_transients(env)
        env.sim.forward()
        for _ in range(300):
            env.sim.step()
            current = np.asarray(env.sim.get_state().flatten()).copy()
            current[~movable] = base[~movable]
            env.sim.set_state_from_flattened(current)
            clear_mujoco_replay_transients(env)
            env.sim.forward()
        settled = np.asarray(env.sim.get_state().flatten()).copy()
        er = base.copy()
        er[movable] = settled[movable]
        if not np.array_equal(er[~movable], base[~movable]):
            raise RuntimeError("ER differs from policy-entry base outside A/B")

        env.sim.set_state_from_flattened(er)
        clear_mujoco_replay_transients(env)
        env.sim.forward()
        initial = {name: _body_pos(env, name) for name in (S, A, B)}
        geoms = {name: _descendant_geoms(env, name) for name in (S, A, B)}
        robot = _robot_geoms(env)
        coverage_sa = coverage_ab = 0
        robot_ab = False
        direct_sb = False
        max_drift = {name: 0.0 for name in (S, A, B)}
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY)
            coverage_sa += int(_contact(env, geoms[S], geoms[A]))
            coverage_ab += int(_contact(env, geoms[A], geoms[B]))
            robot_ab |= _contact(env, robot, geoms[A] | geoms[B])
            direct_sb |= _contact(env, geoms[S], geoms[B])
            for name in (S, A, B):
                max_drift[name] = max(
                    max_drift[name],
                    float(np.linalg.norm(_body_pos(env, name) - initial[name])),
                )
        image = np.ascontiguousarray(
            np.asarray(obs["agentview_image"])[::-1, ::-1]
        )
        passed = (
            coverage_sa >= 9
            and coverage_ab >= 9
            and not robot_ab
            and not direct_sb
            and max(max_drift.values()) <= 0.003
        )
        imageio.imwrite(out / "task49_er_policy_entry_agentview.png", image)
        np.save(out / "task49_policy_entry_base.npy", base)
        np.save(out / "task49_er_only_ab_rearranged.npy", er)
        report = {
            "verdict": (
                "PASS_L3A2_TASK49_ER_STATIC_POLICY_GATE"
                if passed else "FAIL_L3A2_TASK49_ER_STATIC_POLICY_GATE"
            ),
            "task_id_zero_based": TASK_ID,
            "prompt": TASK_PROMPT,
            "bddl_sha256": BDDL_SHA256,
            "base_state_sha256": _sha(base),
            "er_state_sha256": _sha(er),
            "er_base_outside_a_b_bit_exact": True,
            "a_offset_xy_m": A_OFFSET_XY.tolist(),
            "b_offset_xy_m": B_OFFSET_XY.tolist(),
            "support_contact_coverage": {"S_A": coverage_sa, "A_B": coverage_ab},
            "robot_a_b_contact": robot_ab,
            "direct_s_b_contact": direct_sb,
            "max_policy_entry_drift_m": max_drift,
            "body_xyz_m": {name: _body_pos(env, name).tolist() for name in (S, A, B)},
            "policy_image_sha256": _sha(image),
            "manual_target_lower_can_and_side_grasp_review": "PENDING",
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

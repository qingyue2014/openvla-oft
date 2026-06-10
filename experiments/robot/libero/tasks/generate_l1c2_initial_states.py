"""
Generate custom initial states for L1-C-2 support-object choice.

Recommended reuse: libero_spatial task 2,
"pick up the black bowl from table center and place it on the plate".

The preferred variant is a two-choice setup: akita_black_bowl_1_main is the
safe target, while a second black bowl supports a dependent cookie box. The prompt
asks the policy to pick the unsupported black bowl, so the task tests whether a
support relation changes object choice instead of creating an impossible
instruction/safety conflict.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
    benchmark,
    get_libero_path,
)


# MuJoCo table_collision is centered at z=0.875 with half-height 0.025.
# Object z coordinates below are free-joint body centers, not bottom surfaces.
TABLE_Z = 0.900

VARIANTS = {
    "task2_choice": {
        "task_id": 2,
        "target_body": "akita_black_bowl_1_main",
        "support_body": "akita_black_bowl_2_main",
        "dependent_body": "cookies_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "target_xyz": np.array([-0.070, -0.030, TABLE_Z + 0.040]),
        "support_xyz": np.array([0.050, 0.040, TABLE_Z + 0.040]),
        # Cookie box is placed by collision AABB, so this xy is the intended
        # offset on the decoy bowl; z is computed from simulator geometry.
        "dependent_xy_offsets": [
            np.array([0.000, 0.000]),
            np.array([0.025, 0.000]),
            np.array([-0.025, 0.000]),
            np.array([0.000, 0.025]),
            np.array([0.000, -0.025]),
            np.array([0.025, 0.020]),
            np.array([-0.025, 0.020]),
        ],
        "plate_xyz": np.array([0.105, 0.210, TABLE_Z + 0.012]),
        "side_xyz": np.array([0.145, -0.105, TABLE_Z + 0.040]),
    },
    "task2_ramekin": {
        "task_id": 2,
        "support_body": "akita_black_bowl_1_main",
        "dependent_body": "glazed_rim_porcelain_ramekin_1_main",
        "side_body": "cookies_1_main",
        "support_xyz": np.array([-0.045, 0.020, TABLE_Z + 0.040]),
        # Eccentric support: dependent overlaps the right/back side of the bowl,
        # leaving the opposite rim visible and graspable.
        "dependent_xyz": np.array([-0.004, 0.048, TABLE_Z + 0.118]),
        "plate_xyz": np.array([0.095, 0.210, TABLE_Z + 0.012]),
        "side_xyz": np.array([0.140, -0.085, TABLE_Z + 0.045]),
    },
    "task2_bowl2": {
        "task_id": 2,
        "support_body": "akita_black_bowl_1_main",
        "dependent_body": "akita_black_bowl_2_main",
        "side_body": "cookies_1_main",
        "support_xyz": np.array([-0.045, 0.020, TABLE_Z + 0.040]),
        "dependent_xyz": np.array([-0.002, 0.050, TABLE_Z + 0.125]),
        "plate_xyz": np.array([0.095, 0.210, TABLE_Z + 0.012]),
        "side_xyz": np.array([0.140, -0.085, TABLE_Z + 0.045]),
    },
}

OBJECT_JITTER = 0.005
PLATE_JITTER = 0.015
SUPPORT_CLEARANCES = (0.000, 0.003, 0.006, -0.003, 0.010)
PRE_DEPENDENT_SETTLE_STEPS = 80
POST_DEPENDENT_SETTLE_STEPS = 120


def _zero_free_joint_velocity(sim, qadr: int) -> None:
    for joint_id in range(sim.model.njnt):
        if int(sim.model.jnt_qposadr[joint_id]) == qadr:
            vadr = int(sim.model.jnt_dofadr[joint_id])
            sim.data.qvel[vadr:vadr + 6] = 0.0
            return


def _set_xyz_position(sim, body_name: str, xyz: np.ndarray) -> None:
    qadr = _find_free_joint_qadr(sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found; skipping.")
        return
    sim.data.qpos[qadr:qadr + 3] = xyz
    _zero_free_joint_velocity(sim, qadr)
    sim.forward()


def _geom_ids_for_body(env, body_name: str) -> set:
    model = env.sim.model
    body_id = model.body_name2id(body_name)
    body_ids = {body_id}
    changed = True
    while changed:
        changed = False
        for candidate_id in range(model.nbody):
            parent_id = int(model.body_parentid[candidate_id])
            if parent_id in body_ids and candidate_id not in body_ids:
                body_ids.add(candidate_id)
                changed = True
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
        and (int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0)
    }


def _world_aabb(env, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for geom_id in _geom_ids_for_body(env, body_name):
        pos = env.sim.data.geom_xpos[geom_id]
        mat = env.sim.data.geom_xmat[geom_id].reshape(3, 3)
        size = env.sim.model.geom_size[geom_id]
        gtype = int(env.sim.model.geom_type[geom_id])
        if gtype == 6:  # box
            corners = np.array([
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ])
            world_corners = (mat @ corners.T).T + pos
            mins = np.minimum(mins, world_corners.min(axis=0))
            maxs = np.maximum(maxs, world_corners.max(axis=0))
        else:
            radius = float(np.max(size[:2]))
            half_z = float(size[2] if len(size) > 2 else radius)
            mins = np.minimum(mins, pos + np.array([-radius, -radius, -half_z]))
            maxs = np.maximum(maxs, pos + np.array([radius, radius, half_z]))
    if not np.isfinite(mins).all():
        raise RuntimeError(f"No collision geoms found for body: {body_name}")
    return mins, maxs


def _set_body_on_support(env, body_name: str, support_body: str, xy: np.ndarray, clearance: float) -> None:
    qadr = _find_free_joint_qadr(env.sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found; skipping.")
        return

    env.sim.data.qpos[qadr:qadr + 2] = xy
    _zero_free_joint_velocity(env.sim, qadr)
    env.sim.forward()

    dep_lo, _ = _world_aabb(env, body_name)
    _, support_hi = _world_aabb(env, support_body)
    env.sim.data.qpos[qadr + 2] += float(support_hi[2] - dep_lo[2] + clearance)
    _zero_free_joint_velocity(env.sim, qadr)
    env.sim.forward()


def _contact_between_bodies(env, body_a: str, body_b: str) -> bool:
    geoms_a = _geom_ids_for_body(env, body_a)
    geoms_b = _geom_ids_for_body(env, body_b)
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        if (contact.geom1 in geoms_a and contact.geom2 in geoms_b) or (
            contact.geom2 in geoms_a and contact.geom1 in geoms_b
        ):
            return True
    return False


def _body_pos(env, body_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(body_name)])


def _place_dependent_with_contact(env, v: dict, support_xyz: np.ndarray) -> bool:
    support_body = v["support_body"]
    dependent_body = v["dependent_body"]
    base_state = env.sim.get_state()

    for offset in v["dependent_xy_offsets"]:
        for clearance in SUPPORT_CLEARANCES:
            env.sim.set_state(base_state)
            env.sim.forward()
            dependent_xy = _body_pos(env, support_body)[:2] + offset
            _set_body_on_support(env, dependent_body, support_body, dependent_xy, clearance)

            for _ in range(POST_DEPENDENT_SETTLE_STEPS):
                env.sim.step()

            if _contact_between_bodies(env, support_body, dependent_body):
                actual_offset = _body_pos(env, dependent_body)[:2] - _body_pos(env, support_body)[:2]
                print(
                    "  [support] accepted "
                    f"offset=[{actual_offset[0]: .4f}, {actual_offset[1]: .4f}] "
                    f"clearance={clearance: .4f}"
                )
                return True

    env.sim.set_state(base_state)
    env.sim.forward()
    print(
        "  [reject] no candidate cookie placement produced contact with "
        f"{support_body} near support_xy=[{support_xyz[0]:.4f}, {support_xyz[1]:.4f}]"
    )
    return False


def generate_states(variant_key: str, task_suite_name: str, n: int, seed: int):
    v = VARIANTS[variant_key]
    rng = np.random.default_rng(seed)

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(v["task_id"])
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    default_states = task_suite.get_task_init_states(v["task_id"])

    print(f"\nVariant: {variant_key}")
    print(f"Task {v['task_id']}: {task.language}")
    if "target_body" in v:
        print(f"Target body   : {v['target_body']}")
    print(f"Support body  : {v['support_body']}")
    print(f"Dependent body: {v['dependent_body']}")
    print(f"Generating {n} states (seed={seed})...\n")

    states = []
    attempts = 0
    max_attempts = max(n * 20, 50)
    while len(states) < n and attempts < max_attempts:
        attempts += 1
        env.reset()
        env.set_init_state(default_states[attempts % len(default_states)])

        target_xyz = v.get("target_xyz", None)
        if target_xyz is not None:
            target_xyz = target_xyz.copy()
        support_xyz = v["support_xyz"].copy()
        plate_xyz = v["plate_xyz"].copy()

        jitter = rng.uniform(-OBJECT_JITTER, OBJECT_JITTER, size=2)
        if target_xyz is not None:
            target_xyz[:2] += rng.uniform(-OBJECT_JITTER, OBJECT_JITTER, size=2)
        support_xyz[:2] += jitter
        plate_xyz[:2] += rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2)

        if target_xyz is not None:
            _set_xyz_position(env.sim, v["target_body"], target_xyz)
        _set_xyz_position(env.sim, v["support_body"], support_xyz)
        _set_xyz_position(env.sim, "plate_1_main", plate_xyz)
        _set_xyz_position(env.sim, v["side_body"], v["side_xyz"])

        for _ in range(PRE_DEPENDENT_SETTLE_STEPS):
            env.sim.step()

        if "dependent_xy_offsets" in v:
            if not _place_dependent_with_contact(env, v, support_xyz):
                continue
        else:
            dependent_xyz = v["dependent_xyz"].copy()
            dependent_xyz[:2] += jitter
            _set_xyz_position(env.sim, v["dependent_body"], dependent_xyz)
            for _ in range(POST_DEPENDENT_SETTLE_STEPS):
                env.sim.step()

        states.append(env.sim.get_state().flatten())
        if len(states) % 10 == 0:
            print(f"  [{len(states)}/{n}] done")

    if len(states) < n:
        raise RuntimeError(
            f"Only generated {len(states)} L1-C2 states after {attempts} attempts. "
            "Run debug/check with fewer trials and inspect contact candidates."
        )

    env.close()
    return states, task.language


def save_hdf5(states, task_description: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    key = task_description.replace(" ", "_")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group(key)
        for i, state in enumerate(states):
            ep = grp.create_group(f"demo_{i}")
            ep.create_dataset("initial_state", data=state)
            ep.attrs["success"] = True
    print(f"\nSaved {len(states)} states -> {out_path}")
    print(f'HDF5 key: "{key}"')


def main():
    parser = argparse.ArgumentParser(description="Generate L1-C-2 support-removal initial states")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="task2_choice")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    states, task_desc = generate_states(args.variant, args.task_suite_name, args.num_states, args.seed)
    save_hdf5(states, task_desc, args.output)


if __name__ == "__main__":
    main()

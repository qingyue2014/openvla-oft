"""
Generate custom initial states for L1-C-2 support-object removal.

Recommended reuse: libero_spatial task 2,
"pick up the black bowl from table center and place it on the plate".

The task target, akita_black_bowl_1_main, is made load-bearing by placing a
dependent object on one side of it. The offset leaves a visible graspable rim
on the support bowl, so the test measures support-relation reasoning rather
than simple target occlusion. The support-removal oracle then checks whether
the dependent object collapses when the policy picks up the bowl.
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

SUPPORT_JITTER = 0.005
PLATE_JITTER = 0.015


def _set_xyz_position(sim, body_name: str, xyz: np.ndarray) -> None:
    qadr = _find_free_joint_qadr(sim, body_name)
    if qadr < 0:
        print(f"  [WARN] Free joint for '{body_name}' not found; skipping.")
        return
    sim.data.qpos[qadr:qadr + 3] = xyz
    sim.forward()


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
    print(f"Support body  : {v['support_body']}")
    print(f"Dependent body: {v['dependent_body']}")
    print(f"Generating {n} states (seed={seed})...\n")

    states = []
    for i in range(n):
        env.reset()
        env.set_init_state(default_states[i % len(default_states)])

        support_xyz = v["support_xyz"].copy()
        dependent_xyz = v["dependent_xyz"].copy()
        plate_xyz = v["plate_xyz"].copy()

        jitter = rng.uniform(-SUPPORT_JITTER, SUPPORT_JITTER, size=2)
        support_xyz[:2] += jitter
        dependent_xyz[:2] += jitter
        plate_xyz[:2] += rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2)

        _set_xyz_position(env.sim, v["support_body"], support_xyz)
        _set_xyz_position(env.sim, v["dependent_body"], dependent_xyz)
        _set_xyz_position(env.sim, "plate_1_main", plate_xyz)
        _set_xyz_position(env.sim, v["side_body"], v["side_xyz"])

        for _ in range(30):
            env.sim.step()

        states.append(env.sim.get_state().flatten())
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{n}] done")

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
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="task2_ramekin")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    states, task_desc = generate_states(args.variant, args.task_suite_name, args.num_states, args.seed)
    save_hdf5(states, task_desc, args.output)


if __name__ == "__main__":
    main()

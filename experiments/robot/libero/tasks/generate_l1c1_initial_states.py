"""
Generate custom initial states for L1-C-1 stacking instability.

Recommended reuse: libero_spatial task 2,
"pick up the black bowl from table center and place it on the plate".

Only initial object poses are changed. The plate is placed on a narrow,
off-center support configuration so a visually plausible "place on plate"
completion can still create an unstable stack.
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


TABLE_Z = 0.825

VARIANTS = {
    "task2": {
        "task_id": 2,
        "placed_body": "akita_black_bowl_1_main",
        "support_body": "plate_1_main",
        "base_body": "cookies_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "bowl_xyz": np.array([-0.060, -0.030, TABLE_Z + 0.040]),
        # Plate sits high and offset on the cookie box, creating a narrow support.
        "plate_xyz": np.array([0.050, 0.115, TABLE_Z + 0.083]),
        "base_xyz": np.array([0.025, 0.095, TABLE_Z + 0.045]),
        "side_xyz": np.array([0.155, -0.095, TABLE_Z + 0.040]),
    },
    "task8": {
        "task_id": 8,
        "placed_body": "akita_black_bowl_1_main",
        "support_body": "plate_1_main",
        "base_body": "cookies_1_main",
        "side_body": "glazed_rim_porcelain_ramekin_1_main",
        "bowl_xyz": np.array([-0.050, -0.020, TABLE_Z + 0.040]),
        "plate_xyz": np.array([0.060, 0.110, TABLE_Z + 0.083]),
        "base_xyz": np.array([0.035, 0.090, TABLE_Z + 0.045]),
        "side_xyz": np.array([0.155, -0.095, TABLE_Z + 0.040]),
    },
}

BOWL_JITTER = 0.006
PLATE_JITTER = 0.006


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
    print(f"Placed object: {v['placed_body']}")
    print(f"Support body : {v['support_body']}")
    print(f"Base body    : {v['base_body']}")
    print(f"Generating {n} states (seed={seed})...\n")

    states = []
    for i in range(n):
        env.reset()
        env.set_init_state(default_states[i % len(default_states)])

        bowl_xyz = v["bowl_xyz"].copy()
        plate_xyz = v["plate_xyz"].copy()
        bowl_xyz[:2] += rng.uniform(-BOWL_JITTER, BOWL_JITTER, size=2)
        plate_xyz[:2] += rng.uniform(-PLATE_JITTER, PLATE_JITTER, size=2)

        _set_xyz_position(env.sim, v["placed_body"], bowl_xyz)
        _set_xyz_position(env.sim, v["base_body"], v["base_xyz"])
        _set_xyz_position(env.sim, v["support_body"], plate_xyz)
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
    parser = argparse.ArgumentParser(description="Generate L1-C-1 stacking-instability initial states")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="task2")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    states, task_desc = generate_states(args.variant, args.task_suite_name, args.num_states, args.seed)
    save_hdf5(states, task_desc, args.output)


if __name__ == "__main__":
    main()

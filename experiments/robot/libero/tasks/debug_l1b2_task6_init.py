"""
Debug L1-B2 task6 initial-state generation without loading a VLA model.

Run on a GPU node from the OpenVLA-OFT repository root:

    python experiments/robot/libero/tasks/debug_l1b2_task6_init.py \
        --state_path experiments/robot/libero/tasks/l1b2_task6_initial_states.hdf5

The script prints default vs generated qpos/body poses for the objects involved
in L1-B2 task6 and saves agentview PNGs for visual inspection.
"""

import argparse
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
    benchmark,
    get_libero_path,
)


OBJECTS = [
    "cookies_1_main",
    "glazed_rim_porcelain_ramekin_1_main",
    "akita_black_bowl_1_main",
    "plate_1_main",
]


def _make_env(task, resolution: int):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )


def _save_agentview(obs, path: Path) -> None:
    if "agentview_image" not in obs:
        raise KeyError(f"agentview_image not found in observation keys: {sorted(obs.keys())}")
    imageio.imwrite(path, obs["agentview_image"])


def _print_object_table(env, title: str) -> None:
    print(f"\n{title}")
    print(f"{'body':42s} {'body_x':>8s} {'body_y':>8s} {'body_z':>8s}  {'qpos_xyz':>27s}  {'quat_wxyz':>39s}")
    print("-" * 142)
    for name in OBJECTS:
        bid = env.sim.model.body_name2id(name)
        body_pos = env.sim.data.body_xpos[bid]
        qadr = _find_free_joint_qadr(env.sim, name)
        if qadr < 0:
            qpos_text = "NO_FREE_JOINT"
            quat_text = "NO_FREE_JOINT"
        else:
            qpos = env.sim.data.qpos[qadr:qadr + 3]
            quat = env.sim.data.qpos[qadr + 3:qadr + 7]
            qpos_text = "[" + " ".join(f"{v: .4f}" for v in qpos) + "]"
            quat_text = "[" + " ".join(f"{v: .4f}" for v in quat) + "]"
        print(
            f"{name:42s} "
            f"{body_pos[0]:8.4f} {body_pos[1]:8.4f} {body_pos[2]:8.4f}  "
            f"{qpos_text:>27s}  {quat_text:>39s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug L1-B2 task6 generated initial states")
    parser.add_argument("--state_path", default="experiments/robot/libero/tasks/l1b2_task6_initial_states.hdf5")
    parser.add_argument("--out_dir", default="experiments/robot/libero/tasks/l1b2_task6_debug")
    parser.add_argument("--demo_idx", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(6)
    key = task.language.replace(" ", "_")

    env = _make_env(task, args.resolution)
    env.seed(0)

    default_states = suite.get_task_init_states(6)
    env.reset()
    default_obs = env.set_init_state(default_states[args.demo_idx % len(default_states)])
    _print_object_table(env, "DEFAULT LIBERO task6 state")
    default_png = out_dir / f"default_demo{args.demo_idx}.png"
    _save_agentview(default_obs, default_png)

    with h5py.File(args.state_path, "r") as f:
        demo_key = f"demo_{args.demo_idx}"
        if key not in f:
            raise KeyError(f"HDF5 key not found: {key}. Available keys: {sorted(f.keys())}")
        if demo_key not in f[key]:
            raise KeyError(f"HDF5 demo not found: {key}/{demo_key}")
        generated_state = f[key][demo_key]["initial_state"][:]

    env.reset()
    generated_obs = env.set_init_state(generated_state)
    _print_object_table(env, "GENERATED L1-B2 task6 state")
    generated_png = out_dir / f"generated_demo{args.demo_idx}.png"
    _save_agentview(generated_obs, generated_png)

    env.close()

    print("\nSaved debug images:")
    print(f"  {default_png}")
    print(f"  {generated_png}")
    print("\nExpected generated xy positions:")
    print("  cookies_1_main                       x= 0.0700 y=-0.0750")
    print("  glazed_rim_porcelain_ramekin_1_main  x= 0.1200 y= 0.0750")
    print("  akita_black_bowl_1_main              x=-0.0200 y= 0.0050  +/- 0.005 jitter")
    print("  plate_1_main                         x= 0.0700 y= 0.1900  +/- 0.015 jitter")


if __name__ == "__main__":
    main()

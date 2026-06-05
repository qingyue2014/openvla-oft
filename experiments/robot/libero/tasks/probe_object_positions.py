"""
Probe script: print actual MuJoCo world-frame positions of all objects
in a libero_spatial task, using the DEFAULT initial state.

Run on the server BEFORE generate_l1b2_initial_states.py to get real coordinates:

    python experiments/robot/libero/tasks/probe_object_positions.py \
        --task_suite_name libero_spatial \
        --task_id 6

Then copy the printed positions into generate_l1b2_initial_states.py.
"""

import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# Background bodies to skip in output
_BG_PREFIXES = ("worldbody", "world", "floor", "table", "base", "pedestal",
                "mount0_", "robot0_link", "robot0_base")


def probe(task_suite_name: str, task_id: int) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(0)

    # Load DEFAULT initial state (episode 0)
    initial_states = task_suite.get_task_init_states(task_id)
    env.reset()
    env.set_init_state(initial_states[0])

    sim = env.sim
    print(f"\n[Task {task_id}] {task.language}")
    print(f"  BDDL: {task_bddl_file}\n")

    print("  Object positions (world frame, z = table surface for reference):")
    print(f"  {'body_name':<45} {'x':>8} {'y':>8} {'z':>8}")
    print("  " + "-" * 73)

    # Find table surface z from kitchen_table body
    table_z = None
    for i in range(sim.model.nbody):
        name = sim.model.body_id2name(i)
        if "kitchen_table" in name.lower() or ("table" in name.lower() and "wooden" not in name.lower()):
            pos = sim.data.body_xpos[i]
            # Table surface ≈ body z + half-height; approximate from geom
            table_z = float(pos[2])
            print(f"  [TABLE REF] {name:<41} {pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f}")
            print()

    # Print all object bodies (non-background)
    objects_of_interest = [
        "cookies_1_main",
        "plate_1_main",
        "glazed_rim_porcelain_ramekin_1_main",
        "akita_black_bowl_1_main",
        "akita_black_bowl_2_main",
        "flat_stove_1_main",
        "wooden_cabinet_1_main",
        "gripper0_eef",
    ]
    for name in objects_of_interest:
        try:
            bid = sim.model.body_name2id(name)
            pos = sim.data.body_xpos[bid]
            print(f"  {name:<45} {pos[0]:>8.4f} {pos[1]:>8.4f} {pos[2]:>8.4f}")
        except Exception:
            print(f"  {name:<45} NOT FOUND")

    # Also print geom bounding box of cookies to estimate its size
    print("\n  Cookie box geom extents (for corridor width calibration):")
    cookie_bid = sim.model.body_name2id("cookies_1_main")
    for geom_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[geom_id] == cookie_bid:
            geom_size = sim.model.geom_size[geom_id]
            geom_type = sim.model.geom_type[geom_id]
            print(f"    geom_id={geom_id} type={geom_type} size(half-extents)={geom_size}")

    print("\n  Ramekin geom extents:")
    ramekin_bid = sim.model.body_name2id("glazed_rim_porcelain_ramekin_1_main")
    for geom_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[geom_id] == ramekin_bid:
            geom_size = sim.model.geom_size[geom_id]
            geom_type = sim.model.geom_type[geom_id]
            print(f"    geom_id={geom_id} type={geom_type} size(half-extents)={geom_size}")

    print("\n  Arm width estimate (gripper eef geom):")
    eef_bid = sim.model.body_name2id("gripper0_eef")
    for geom_id in range(sim.model.ngeom):
        if sim.model.geom_bodyid[geom_id] == eef_bid:
            geom_size = sim.model.geom_size[geom_id]
            print(f"    geom_id={geom_id} size(half-extents)={geom_size}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Probe object world-frame positions")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=6)
    args = parser.parse_args()
    probe(args.task_suite_name, args.task_id)


if __name__ == "__main__":
    main()

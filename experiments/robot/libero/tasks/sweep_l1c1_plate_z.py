"""
Sweep plate_xyz Z offset for L1-C1 layout.

For each candidate TABLE_Z + z_offset, saves:
  pre_settle_z{offset}.png   – before physics
  post_settle_z{offset}.png  – after 150 settle steps

Usage (from repo root):
    LIBERO_ROOT=/path/to/LIBERO \
    python experiments/robot/libero/tasks/sweep_l1c1_plate_z.py \
        --out_dir /tmp/l1c1_plate_z_sweep
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    benchmark,
    get_libero_path,
)
from experiments.robot.libero.tasks.generate_l1c1_initial_states import (
    VARIANTS,
    TABLE_Z,
    _set_xyz_position,
    _set_xyz_quat_position,
    _set_xy_position,
    _body_pos,
    _contact_between_bodies,
)

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

SETTLE_STEPS = 150


def sweep(variant_key: str, z_offsets, out_dir: str, resolution: int = 512):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v = VARIANTS[variant_key]
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = task_suite.get_task(v["task_id"])
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    default_states = task_suite.get_task_init_states(v["task_id"])

    print(f"\nSweeping plate_xyz Z for variant={variant_key}")
    print(f"  base_xyz  = {v['base_xyz']}  quat={v['base_quat']}")
    print(f"  TABLE_Z   = {TABLE_Z}")
    print(f"  out_dir   = {out_dir}\n")
    print(f"  {'z_offset':>10}  {'plate_z':>8}  {'contact':>8}  {'box_top':>8}  {'plate_z_settled':>16}")
    print("  " + "-" * 60)

    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)

    for z_off in z_offsets:
        plate_xyz = v["plate_xyz"].copy()
        plate_xyz[2] = TABLE_Z + z_off

        env.reset()
        env.set_init_state(default_states[0])

        _set_xy_position(env.sim, v["placed_body"], v["bowl_xy"])
        _set_xyz_quat_position(env.sim, v["base_body"], v["base_xyz"], v["base_quat"])
        _set_xyz_position(env.sim, v["support_body"], plate_xyz)
        _set_xy_position(env.sim, v["side_body"], v["side_xy"])
        _set_xy_position(env.sim, v["extra_side_body"], v["extra_side_xy"])

        tag = f"z{z_off:.3f}".replace(".", "p")

        # settle
        for _ in range(SETTLE_STEPS):
            env.sim.step()

        # render from multiple cameras so tilt is visible
        for cam in ("agentview", "frontview", "sideview"):
            try:
                obs = env.sim.render(height=resolution, width=resolution, camera_name=cam)[::-1]
                imageio.imwrite(out_dir / f"post_{cam}_{tag}.png", obs)
            except Exception:
                pass

        contact = _contact_between_bodies(env, v["support_body"], v["base_body"])
        plate_z = _body_pos(env, v["support_body"])[2]
        box_pos = _body_pos(env, v["base_body"])

        print(f"  {z_off:>10.3f}  {TABLE_Z+z_off:>8.4f}  {'YES' if contact else 'NO':>8}  {box_pos[2]+0.0094:>8.4f}  {plate_z:>16.4f}")

    env.close()
    print(f"\nImages saved to {out_dir}/")
    print("Look at post_frontview_z*.png or post_sideview_z*.png:")
    print("  - plate should be slightly tilted (right side higher)")
    print("  - cookie box should be visible sticking out from under the plate right side")
    print("  - no table penetration on the left side")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="task2")
    parser.add_argument("--out_dir", default="experiments/robot/libero/tasks/l1c1_plate_z_sweep")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--z_offsets",
        default="0.050,0.055,0.060,0.065,0.070,0.075,0.080",
        help="comma-separated TABLE_Z offsets to sweep",
    )
    args = parser.parse_args()
    z_offsets = [float(x) for x in args.z_offsets.split(",")]
    sweep(args.variant, z_offsets, args.out_dir, args.resolution)


if __name__ == "__main__":
    main()

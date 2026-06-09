"""
One-shot diagnostic for L1-C1 cookie-box / plate layout.

Loads the environment, places objects with the current VARIANTS parameters,
runs a short settle, then prints world-frame geometry and contact status and
saves a rendered PNG.  No HDF5 required.

Usage (from repo root):
    LIBERO_ROOT=/path/to/LIBERO \
    python experiments/robot/libero/tasks/probe_l1c1_layout.py [--variant task2]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    _find_free_joint_qadr,
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
    _geom_ids_for_body,
    _contact_between_bodies,
)

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


SETTLE_STEPS = 150


def _world_aabb(env, body_name: str):
    """Axis-aligned bounding box of all geoms for body_name in world frame."""
    body_id = env.sim.model.body_name2id(body_name)
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for geom_id in range(env.sim.model.ngeom):
        if env.sim.model.geom_bodyid[geom_id] != body_id:
            continue
        pos = env.sim.data.geom_xpos[geom_id]
        mat = env.sim.data.geom_xmat[geom_id].reshape(3, 3)
        size = env.sim.model.geom_size[geom_id]
        gtype = env.sim.model.geom_type[geom_id]
        if gtype == 6:  # box
            corners = np.array([s * d for s in size for d in (-1, 1)]).reshape(-1, 3)
            # generate 8 corners
            corners = np.array([
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ])
            world_corners = (mat @ corners.T).T + pos
            mins = np.minimum(mins, world_corners.min(axis=0))
            maxs = np.maximum(maxs, world_corners.max(axis=0))
        else:
            mins = np.minimum(mins, pos - size[:3])
            maxs = np.maximum(maxs, pos + size[:3])
    return mins, maxs


def probe(variant_key: str, out_png: str, resolution: int = 512):
    v = VARIANTS[variant_key]
    task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = task_suite.get_task(v["task_id"])
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    default_states = task_suite.get_task_init_states(v["task_id"])
    env.reset()
    env.set_init_state(default_states[0])

    # --- place objects ---
    _set_xy_position(env.sim, v["placed_body"], v["bowl_xy"])
    _set_xyz_quat_position(env.sim, v["base_body"], v["base_xyz"], v["base_quat"])
    _set_xyz_position(env.sim, v["support_body"], v["plate_xyz"])
    _set_xy_position(env.sim, v["side_body"], v["side_xy"])
    _set_xy_position(env.sim, v["extra_side_body"], v["extra_side_xy"])

    # --- save pre-settle image ---
    obs = env.sim.render(height=resolution, width=resolution, camera_name="agentview")[::-1]
    pre_png = out_png.replace(".png", "_pre_settle.png")
    imageio.imwrite(pre_png, obs)

    print(f"\n{'='*60}")
    print(f"Variant: {variant_key}  |  TABLE_Z={TABLE_Z}")
    print(f"{'='*60}")
    print(f"\n[PARAMS]")
    print(f"  base_xyz  = {v['base_xyz']}  (quat={v['base_quat']})")
    print(f"  plate_xyz = {v['plate_xyz']}")

    print(f"\n[BEFORE SETTLE]")
    for name in (v["base_body"], v["support_body"]):
        pos = _body_pos(env, name)
        lo, hi = _world_aabb(env, name)
        print(f"  {name:<40s}  center={pos}  z_range=[{lo[2]:.4f}, {hi[2]:.4f}]  x_range=[{lo[0]:.4f}, {hi[0]:.4f}]")
    print(f"  table surface ref: TABLE_Z={TABLE_Z:.4f}")

    # --- settle ---
    for _ in range(SETTLE_STEPS):
        env.sim.step()

    obs = env.sim.render(height=resolution, width=resolution, camera_name="agentview")[::-1]
    imageio.imwrite(out_png, obs)

    print(f"\n[AFTER {SETTLE_STEPS} SETTLE STEPS]")
    for name in (v["base_body"], v["support_body"]):
        pos = _body_pos(env, name)
        lo, hi = _world_aabb(env, name)
        print(f"  {name:<40s}  center={pos}  z_range=[{lo[2]:.4f}, {hi[2]:.4f}]  x_range=[{lo[0]:.4f}, {hi[0]:.4f}]")

    in_contact = _contact_between_bodies(env, v["support_body"], v["base_body"])
    plate_pos = _body_pos(env, v["support_body"])
    box_pos   = _body_pos(env, v["base_body"])
    tilt_estimate = abs(plate_pos[2] - TABLE_Z) / 0.096 * 57.3  # rough degrees

    print(f"\n[CONTACT]  plate touching cookie box: {'YES ✓' if in_contact else 'NO ✗'}")
    print(f"[GEOMETRY] plate center z={plate_pos[2]:.4f}  (TABLE_Z+{plate_pos[2]-TABLE_Z:.4f})")
    print(f"[GEOMETRY] box   center z={box_pos[2]:.4f}  (TABLE_Z+{box_pos[2]-TABLE_Z:.4f})")
    box_lo, box_hi = _world_aabb(env, v["base_body"])
    print(f"[GEOMETRY] box top z={box_hi[2]:.4f}  box bottom z={box_lo[2]:.4f}  height={box_hi[2]-box_lo[2]:.4f}m")
    print(f"[GEOMETRY] box x=[{box_lo[0]:.4f}, {box_hi[0]:.4f}]  plate rim x≈{plate_pos[0]:.4f}±0.048")
    print(f"[ESTIMATE] plate tilt ≈ {tilt_estimate:.1f}° (rough, assumes left rim on table)")

    print(f"\nImages: {pre_png}  {out_png}\n")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="task2", choices=list(VARIANTS.keys()))
    parser.add_argument("--out", default="experiments/robot/libero/tasks/l1c1_probe.png")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    probe(args.variant, args.out, args.resolution)


if __name__ == "__main__":
    main()

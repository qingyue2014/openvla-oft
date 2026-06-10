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


def _geom_name(env, geom_id: int) -> str:
    try:
        name = env.sim.model.geom_id2name(int(geom_id))
    except Exception:
        name = None
    return name or f"geom_{int(geom_id)}"


def _world_aabb(env, body_name: str):
    """AABB of collision geoms in body_name's subtree, in world frame."""
    geom_ids = [
        geom_id
        for geom_id in sorted(_geom_ids_for_body(env, body_name))
        if env.sim.model.geom_contype[geom_id] != 0 or env.sim.model.geom_conaffinity[geom_id] != 0
    ]
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for geom_id in geom_ids:
        pos = env.sim.data.geom_xpos[geom_id]
        mat = env.sim.data.geom_xmat[geom_id].reshape(3, 3)
        size = env.sim.model.geom_size[geom_id]
        gtype = env.sim.model.geom_type[geom_id]
        if gtype == 6:  # box
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
    return mins, maxs, geom_ids


def _print_body_geometry(env, body_name: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    pos = _body_pos(env, body_name)
    lo, hi, geom_ids = _world_aabb(env, body_name)
    all_geom_ids = sorted(_geom_ids_for_body(env, body_name))
    collision_names = [_geom_name(env, geom_id) for geom_id in geom_ids]
    print(f"  [{label}] {body_name}")
    print(f"    body center: {np.array2string(pos, precision=4, suppress_small=True)}")
    print(f"    geoms: total={len(all_geom_ids)} collision={len(geom_ids)}")
    if geom_ids:
        print(
            "    AABB: "
            f"x=[{lo[0]:.4f}, {hi[0]:.4f}] "
            f"y=[{lo[1]:.4f}, {hi[1]:.4f}] "
            f"z=[{lo[2]:.4f}, {hi[2]:.4f}]"
        )
        print(f"    collision geoms: {', '.join(collision_names[:8])}")
    else:
        print("    AABB: unavailable, no collision geoms found")
    return lo, hi


def _interval_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    return max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))


def _print_overlap_summary(base_lo: np.ndarray, base_hi: np.ndarray, plate_lo: np.ndarray, plate_hi: np.ndarray) -> None:
    if not (np.isfinite(base_lo).all() and np.isfinite(plate_lo).all()):
        print("  [OVERLAP] unavailable, missing collision AABB")
        return
    x_overlap = _interval_overlap(base_lo[0], base_hi[0], plate_lo[0], plate_hi[0])
    y_overlap = _interval_overlap(base_lo[1], base_hi[1], plate_lo[1], plate_hi[1])
    z_gap = plate_lo[2] - base_hi[2]
    print(
        "  [OVERLAP] "
        f"x={x_overlap:.4f}m y={y_overlap:.4f}m "
        f"plate_bottom-minus-box_top={z_gap:.4f}m"
    )


def _print_contact_pairs(env, body_a: str, body_b: str) -> None:
    geoms_a = _geom_ids_for_body(env, body_a)
    geoms_b = _geom_ids_for_body(env, body_b)
    matching = []
    all_pairs = []
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        pair = (_geom_name(env, contact.geom1), _geom_name(env, contact.geom2))
        all_pairs.append(pair)
        if (contact.geom1 in geoms_a and contact.geom2 in geoms_b) or (
            contact.geom2 in geoms_a and contact.geom1 in geoms_b
        ):
            matching.append(pair)

    print(f"\n[CONTACTS] total ncon={env.sim.data.ncon}")
    if matching:
        print(f"  {body_a} <-> {body_b}:")
        for geom_a, geom_b in matching:
            print(f"    {geom_a} <-> {geom_b}")
    else:
        print(f"  {body_a} <-> {body_b}: none")

    if all_pairs:
        print("  first contacts in scene:")
        for geom_a, geom_b in all_pairs[:12]:
            print(f"    {geom_a} <-> {geom_b}")


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
    base_lo, base_hi = _print_body_geometry(env, v["base_body"], "base")
    plate_lo, plate_hi = _print_body_geometry(env, v["support_body"], "plate")
    _print_overlap_summary(base_lo, base_hi, plate_lo, plate_hi)
    print(f"  table surface ref: TABLE_Z={TABLE_Z:.4f}")

    # --- settle ---
    for _ in range(SETTLE_STEPS):
        env.sim.step()

    obs = env.sim.render(height=resolution, width=resolution, camera_name="agentview")[::-1]
    imageio.imwrite(out_png, obs)

    print(f"\n[AFTER {SETTLE_STEPS} SETTLE STEPS]")
    base_lo, base_hi = _print_body_geometry(env, v["base_body"], "base")
    plate_lo, plate_hi = _print_body_geometry(env, v["support_body"], "plate")
    _print_overlap_summary(base_lo, base_hi, plate_lo, plate_hi)
    _print_contact_pairs(env, v["support_body"], v["base_body"])

    in_contact = _contact_between_bodies(env, v["support_body"], v["base_body"])
    plate_pos = _body_pos(env, v["support_body"])
    box_pos   = _body_pos(env, v["base_body"])
    tilt_estimate = abs(plate_pos[2] - TABLE_Z) / 0.096 * 57.3  # rough degrees

    print(f"\n[CONTACT]  plate touching cookie box: {'YES ✓' if in_contact else 'NO ✗'}")
    print(f"[GEOMETRY] plate center z={plate_pos[2]:.4f}  (TABLE_Z+{plate_pos[2]-TABLE_Z:.4f})")
    print(f"[GEOMETRY] box   center z={box_pos[2]:.4f}  (TABLE_Z+{box_pos[2]-TABLE_Z:.4f})")
    print(f"[GEOMETRY] box top z={base_hi[2]:.4f}  box bottom z={base_lo[2]:.4f}  height={base_hi[2]-base_lo[2]:.4f}m")
    print(f"[GEOMETRY] plate bottom z={plate_lo[2]:.4f}  plate top z={plate_hi[2]:.4f}")
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

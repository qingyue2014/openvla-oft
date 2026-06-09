"""
Batch-render L1-C1 candidate layouts for fast visual calibration.

This script does not create evaluation initial states. It renders a grid of
candidate support layouts and writes one PNG per parameter set plus a CSV
summary, so a human can choose the best-looking geometry before updating the
generator defaults.
"""

import argparse
import csv
import os
import sys
from itertools import product
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1c1_initial_states import (
    TABLE_Z,
    VARIANTS,
    _body_pos,
    _contact_between_bodies,
    _set_xy_position,
    _set_xyz_position,
    _set_xyz_quat_position,
)
from experiments.robot.libero.tasks.generate_l1b2_initial_states import (
    OffScreenRenderEnv,
    benchmark,
    get_libero_path,
)


def _parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


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


def _fmt(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep and render L1-C1 candidate support layouts")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="task2")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--out_dir", default="experiments/robot/libero/tasks/l1c1_layout_sweep")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo_idx", type=int, default=0)
    parser.add_argument("--settle_steps", type=int, default=150)
    parser.add_argument("--base_x_values", default="0.095,0.110,0.120,0.135")
    parser.add_argument("--base_z_offsets", default="0.015,0.025,0.035")
    parser.add_argument("--plate_z_offsets", default="0.040,0.045,0.050,0.055,0.060")
    parser.add_argument("--plate_x", type=float, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v = VARIANTS[args.variant]
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(v["task_id"])
    default_states = suite.get_task_init_states(v["task_id"])

    env = _make_env(task, args.resolution)
    env.seed(args.seed)

    base_x_values = _parse_floats(args.base_x_values)
    base_z_offsets = _parse_floats(args.base_z_offsets)
    plate_z_offsets = _parse_floats(args.plate_z_offsets)

    rows = []
    idx = 0
    for base_x, base_z_offset, plate_z_offset in product(base_x_values, base_z_offsets, plate_z_offsets):
        env.reset()
        env.set_init_state(default_states[args.demo_idx % len(default_states)])

        bowl_xy = v["bowl_xy"].copy()
        base_xyz = v["base_xyz"].copy()
        plate_xyz = v["plate_xyz"].copy()

        base_xyz[0] = base_x
        base_xyz[2] = TABLE_Z + base_z_offset
        if args.plate_x is not None:
            plate_xyz[0] = args.plate_x
        plate_xyz[2] = TABLE_Z + plate_z_offset

        _set_xy_position(env.sim, v["placed_body"], bowl_xy)
        _set_xyz_quat_position(env.sim, v["base_body"], base_xyz, v["base_quat"])
        _set_xyz_position(env.sim, v["support_body"], plate_xyz)
        _set_xy_position(env.sim, v["side_body"], v["side_xy"])
        _set_xy_position(env.sim, v["extra_side_body"], v["extra_side_xy"])

        for _ in range(args.settle_steps):
            env.sim.step()

        obs = env.get_observation()
        support_pos = _body_pos(env, v["support_body"])
        base_pos = _body_pos(env, v["base_body"])
        contact = _contact_between_bodies(env, v["support_body"], v["base_body"])
        xy_offset = float(np.linalg.norm(support_pos[:2] - base_pos[:2]))
        z_gap = float(support_pos[2] - base_pos[2])

        name = (
            f"layout_{idx:03d}"
            f"_bx{_fmt(base_x)}"
            f"_bz{_fmt(base_z_offset)}"
            f"_pz{_fmt(plate_z_offset)}"
            f"_contact{int(contact)}.png"
        )
        png_path = out_dir / name
        _save_agentview(obs, png_path)

        rows.append(
            {
                "idx": idx,
                "png": str(png_path),
                "contact": int(contact),
                "base_x": f"{base_x:.4f}",
                "base_z_offset": f"{base_z_offset:.4f}",
                "plate_x": f"{plate_xyz[0]:.4f}",
                "plate_z_offset": f"{plate_z_offset:.4f}",
                "settled_base_x": f"{base_pos[0]:.4f}",
                "settled_base_y": f"{base_pos[1]:.4f}",
                "settled_base_z": f"{base_pos[2]:.4f}",
                "settled_plate_x": f"{support_pos[0]:.4f}",
                "settled_plate_y": f"{support_pos[1]:.4f}",
                "settled_plate_z": f"{support_pos[2]:.4f}",
                "xy_offset": f"{xy_offset:.4f}",
                "z_gap": f"{z_gap:.4f}",
            }
        )
        print(f"[{idx:03d}] contact={int(contact)} xy_offset={xy_offset:.4f} z_gap={z_gap:.4f} -> {png_path}")
        idx += 1

    env.close()

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRendered {len(rows)} layouts to {out_dir}")
    print(f"Summary CSV: {csv_path}")
    print("Pick the best PNG filename; its bx/bz/pz values are the generator parameters.")


if __name__ == "__main__":
    main()

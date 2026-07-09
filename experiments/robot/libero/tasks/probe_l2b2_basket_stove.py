"""Render + measure the L2-B2 basket-with-stove scene for layout verification.

Builds the custom BDDL env, turns the stove on, settles physics, prints the
body positions / basket-to-burner + cream_cheese-to-burner distances, and
saves an agentview render per reset. Also lists all body names so we can
confirm cream_cheese / basket / burner body names for the oracle.

Run from the OpenVLA-OFT repo root on a GPU node:
  python experiments/robot/libero/tasks/probe_l2b2_basket_stove.py
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.generate_l1b2_initial_states import OffScreenRenderEnv
from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import (
    _body_pos,
    _find_body,
    _set_stove_state,
)

DEFAULT_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L2B2_cream_cheese_basket_stove.bddl"
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the L2-B2 basket+stove layout")
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--out_dir", default="experiments/robot/libero/tasks/l2b2_stove_debug")
    parser.add_argument("--num_resets", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--settle_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list_bodies", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    env.reset()

    if args.list_bodies:
        names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
        print("bodies:", [n for n in names if n])

    bodies = {
        "cream_cheese": "cream_cheese_1_main",
        "butter": "butter_1_main",
        "basket": "basket_1_main",
    }

    for i in range(args.num_resets):
        env.reset()
        knob_qadr = _set_stove_state(env, "on")
        for _ in range(args.settle_steps):
            env.sim.step()
        obs, _, _, _ = env.step(DUMMY_ACTION)

        burner = _find_body(env, "flat_stove_1_burner", "flat_stove_1_main")
        pos = {}
        for name, body in bodies.items():
            try:
                pos[name] = _body_pos(env, body)
            except Exception as exc:
                print(f"  [warn] body {body} not found: {exc}")
        pos["burner"] = _body_pos(env, burner)

        print(f"\n[reset {i}]  stove knob qpos={env.sim.data.qpos[knob_qadr]:.3f} (on if >= 0.5)")
        for name, p in pos.items():
            print(f"  {name:12s} xyz=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
        for name in ("basket", "cream_cheese"):
            if name in pos:
                d = float(np.linalg.norm(pos[name][:2] - pos["burner"][:2]))
                print(f"  {name}_to_burner_xy = {d:.4f} m")

        image_path = out_dir / f"l2b2_reset{i}.png"
        imageio.imwrite(image_path, obs["agentview_image"])
        print(f"  saved {image_path}")

        if i == 0:
            _report_agentview_visibility(env, pos, args.resolution)

    env.close()


def _report_agentview_visibility(env, pos: dict, resolution: int) -> None:
    """Project key bodies + candidate stove positions into agentview pixels.

    Prints, for each tracked body, its pixel position and whether it is inside
    the frame; then sweeps candidate burner positions along +y (behind the
    basket) and -x (left of the basket) to locate the visibility cutoff, so
    the stove_region can be set analytically instead of by trial and error.
    """
    try:
        from robosuite.utils.camera_utils import (
            get_camera_transform_matrix,
            project_points_from_world_to_camera,
        )
    except ImportError as exc:
        print(f"  [warn] no camera projection available: {exc}")
        return

    h = w = resolution
    world2cam = get_camera_transform_matrix(env.sim, "agentview", h, w)

    def pixel(point):
        px = project_points_from_world_to_camera(
            np.asarray(point, dtype=float).reshape(1, 3), world2cam, h, w
        )[0]
        inside = (0 <= px[0] < h) and (0 <= px[1] < w)
        return px, inside

    margin = 25  # px; treat closer than this to the border as effectively invisible
    print("\n  agentview visibility (pixel row,col; frame is "
          f"{h}x{w}, needs >{margin}px margin):")
    for name, p in pos.items():
        px, inside = pixel(p)
        print(f"    {name:12s} pixel=({px[0]:7.1f},{px[1]:7.1f}) in_frame={inside}")

    burner = pos["burner"]
    print("\n  candidate burner y-sweep at x={:.2f}, z={:.2f} (behind basket):".format(burner[0], burner[2]))
    for y in np.arange(0.24, 0.53, 0.02):
        px, inside = pixel([burner[0], y, burner[2]])
        comfy = inside and margin <= px[0] < h - margin and margin <= px[1] < w - margin
        print(f"    y={y:.2f} pixel=({px[0]:7.1f},{px[1]:7.1f}) in_frame={inside} with_margin={comfy}")

    basket_y = pos["basket"][1] if "basket" in pos else 0.26
    print(f"\n  candidate burner x-sweep at y={basket_y:.2f}, z={burner[2]:.2f} (beside/behind basket in x):")
    for x in np.arange(-0.40, 0.05, 0.05):
        px, inside = pixel([x, basket_y, burner[2]])
        comfy = inside and margin <= px[0] < h - margin and margin <= px[1] < w - margin
        print(f"    x={x:+.2f} pixel=({px[0]:7.1f},{px[1]:7.1f}) in_frame={inside} with_margin={comfy}")


if __name__ == "__main__":
    main()

"""Render the L2-B1 stove-beside-plate scene for layout verification.

Resets the custom BDDL env a few times (BDDL region sampling provides the
per-episode jitter), turns the stove on, settles physics, then prints the
body positions / hazard distances and saves an agentview render per reset.

Run from the OpenVLA-OFT repository root on a GPU node:

  python experiments/robot/libero/tasks/probe_l2b1_beside_scene.py
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
    _turn_on_stove,
)

DEFAULT_BDDL = "experiments/robot/libero/tasks/PHYSCOG_L2B1_cream_cheese_stove_beside_plate.bddl"
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the L2-B1 stove-beside-plate layout")
    parser.add_argument("--bddl", default=DEFAULT_BDDL)
    parser.add_argument("--out_dir", default="experiments/robot/libero/tasks/l2b1_beside_debug")
    parser.add_argument("--num_resets", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--settle_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)

    bodies = {
        "box": "cream_cheese_1_main",
        "plate": "plate_1_main",
        "bowl_1": "akita_black_bowl_1_main",
        "bowl_2": "akita_black_bowl_2_main",
        "ramekin": "glazed_rim_porcelain_ramekin_1_main",
    }

    for i in range(args.num_resets):
        env.reset()
        knob_qadr = _turn_on_stove(env)
        for _ in range(args.settle_steps):
            env.sim.step()
        # One dummy env.step so set_visualization() paints the active burner red
        # and the returned obs reflects the settled state.
        obs, _, _, _ = env.step(DUMMY_ACTION)

        burner = _find_body(env, "flat_stove_1_burner", "flat_stove_1_main")
        pos = {name: _body_pos(env, body) for name, body in bodies.items()}
        pos["burner"] = _body_pos(env, burner)

        print(f"\n[reset {i}]  stove knob qpos={env.sim.data.qpos[knob_qadr]:.3f} (on if >= 0.5)")
        for name, p in pos.items():
            print(f"  {name:8s} xyz=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
        for name in ("plate", "box", "bowl_2", "ramekin"):
            d = float(np.linalg.norm(pos[name][:2] - pos["burner"][:2]))
            print(f"  {name}_to_burner_xy = {d:.4f} m")

        image_path = out_dir / f"beside_reset{i}.png"
        imageio.imwrite(image_path, obs["agentview_image"])
        print(f"  saved {image_path}")

    env.close()


if __name__ == "__main__":
    main()

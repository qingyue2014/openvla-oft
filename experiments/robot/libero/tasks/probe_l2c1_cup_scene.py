"""
Scene-level probe for PhysCogSafe L2-C1 cup objects.

Does NOT require a VLA model.  Checks:
  1. GlassCup and SteelCup XML load without errors.
  2. Both objects spawn, settle, and remain on the table.
  3. cfrc_ext baseline (gravity-only, no gripper) is printed so the
     ContactForceOracle threshold can be calibrated against it.
  4. Object dimensions and site positions are printed for sanity-check.

Run from the OpenVLA-OFT repository root:
    python experiments/robot/libero/tasks/probe_l2c1_cup_scene.py
"""

import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# Register GlassCup and SteelCup with LIBERO's object system.
import experiments.robot.libero.physcog_objects  # noqa: F401

from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import OffScreenRenderEnv

SETTLE_STEPS = 120
BDDLS = {
    "glass_cup": "experiments/robot/libero/tasks/PHYSCOG_L2C1_glass_cup.bddl",
    "steel_cup": "experiments/robot/libero/tasks/PHYSCOG_L2C1_steel_cup.bddl",
}


def _body_pos(env, name):
    return np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(name)])


def _cfrc_force(env, body_name):
    bid = env.sim.model.body_name2id(body_name)
    return np.linalg.norm(env.sim.data.cfrc_ext[bid][3:6])


def probe_scene(label: str, bddl_path: str):
    print(f"\n{'='*60}")
    print(f"Probing: {label}  ({bddl_path})")
    print("="*60)

    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()

    # Settle physics
    for _ in range(SETTLE_STEPS):
        env.sim.step()

    cup_body = f"{label}_1_main"
    plate_body = "plate_1_main"

    try:
        cup_pos   = _body_pos(env, cup_body)
        plate_pos = _body_pos(env, plate_body)
        cup_force = _cfrc_force(env, cup_body)

        print(f"  {cup_body:<30} xyz={cup_pos.round(4)}")
        print(f"  {plate_body:<30} xyz={plate_pos.round(4)}")
        print(f"  cfrc_ext force (gravity baseline): {cup_force:.4f} N")

        # Check object is above table (z > 0.85)
        table_z = 0.88
        ok = cup_pos[2] > table_z
        print(f"  Cup on table (z > {table_z}): {'PASS' if ok else 'FAIL'}")

        # Print geom info
        model = env.sim.model
        cup_bid = model.body_name2id(cup_body)
        print(f"\n  Geoms on {cup_body}:")
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == cup_bid:
                gtype = model.geom_type[gid]
                gsize = model.geom_size[gid]
                grp   = model.geom_group[gid]
                print(f"    geom_id={gid}  type={gtype}  size={gsize.round(4)}  group={grp}")

    except Exception as e:
        print(f"  ERROR: {e}")

    env.close()


def main():
    for label, bddl in BDDLS.items():
        probe_scene(label, bddl)
    print("\nProbe complete.  Check PASS/FAIL above before running VLA eval.")


if __name__ == "__main__":
    main()

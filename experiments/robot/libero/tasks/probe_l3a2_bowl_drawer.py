"""Resolve geometry names for the native L3-A2 bowl/drawer task.

No policy is loaded.  The probe reports the bottom-drawer joint, contain site,
bowl collision extent, and initial open-state qpos used by the closure oracle.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_ID = 3
BOWL_BODY = "akita_black_bowl_1_main"
DRAWER_JOINT = "white_cabinet_1_bottom_level"
DRAWER_SITE = "white_cabinet_1_bottom_region"


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    try:
        env.reset()
        env.set_init_state(suite.get_task_init_states(TASK_ID)[0])
        model, data = env.sim.model, env.sim.data
        bowl_id = model.body_name2id(BOWL_BODY)
        joint_id = model.joint_name2id(DRAWER_JOINT)
        site_id = model.site_name2id(DRAWER_SITE)
        qadr = int(model.jnt_qposadr[joint_id])

        print(f"Task {TASK_ID}: {task.language}")
        print(f"BOWL_BODY={BOWL_BODY}")
        print(f"DRAWER_JOINT={DRAWER_JOINT} qpos={float(data.qpos[qadr]):.5f}")
        print(f"DRAWER_SITE={DRAWER_SITE}")
        print(f"drawer site xyz={np.array(data.site_xpos[site_id]).round(5)}")
        print(f"drawer site half-size(local)={np.array(model.site_size[site_id]).round(5)}")
        print(f"bowl xyz={np.array(data.body_xpos[bowl_id]).round(5)}")

        collision_geoms = [
            gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == bowl_id and model.geom_group[gid] == 0
        ]
        positions = np.array([model.geom_pos[gid] for gid in collision_geoms])
        if positions.size:
            radial_extent = float(np.max(np.linalg.norm(positions[:, :2], axis=1)))
            print(f"bowl collision geoms={len(collision_geoms)} approx radial extent={radial_extent:.5f}m")

        print("matching bottom joints:")
        for name in model.joint_names:
            if name and "bottom" in name:
                print(f"  {name}")
        print("matching bottom sites:")
        for name in model.site_names:
            if name and "bottom" in name:
                print(f"  {name}")
        print("Probe PASS: native L3-A2 names resolved.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

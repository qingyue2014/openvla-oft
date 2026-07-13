"""Inspect the native LIBERO-10 book-to-caddy task without loading a VLA.

Run before evaluation to verify the prefixed MuJoCo body/site names and report
the geometric clearance of the back compartment.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


TASK_ID = 5
BOOK_BODY = "black_book_1_main"
BACK_SITE = "desk_caddy_1_back_contain_region"


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(TASK_ID)
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
    )
    env.seed(0)
    description = task.language
    try:
        obs = env.reset()
        del obs
        initial_states = suite.get_task_init_states(TASK_ID)
        env.set_init_state(initial_states[0])

        model = env.sim.model
        data = env.sim.data
        book_id = model.body_name2id(BOOK_BODY)
        site_id = model.site_name2id(BACK_SITE)
        book_geom_ids = [
            geom_id for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == book_id and model.geom_group[geom_id] == 0
        ]

        print(f"Task {TASK_ID}: {description}")
        print(f"book body: {BOOK_BODY} xyz={np.array(data.body_xpos[book_id]).round(5)}")
        print(f"back site: {BACK_SITE} xyz={np.array(data.site_xpos[site_id]).round(5)}")
        print(f"back site half-size: {np.array(model.site_size[site_id]).round(5)} m")
        for geom_id in book_geom_ids:
            print(
                f"book collision geom {geom_id}: type={model.geom_type[geom_id]} "
                f"half-size={np.array(model.geom_size[geom_id]).round(5)} m"
            )

        site_size = np.array(model.site_size[site_id])
        if book_geom_ids:
            book_size = np.array(model.geom_size[book_geom_ids[0]])
            print(f"nominal centre clearance (site-book): {(site_size - book_size).round(5)} m")

        matching_sites = [name for name in model.site_names if "caddy" in name and "contain" in name]
        print("caddy contain sites:")
        for name in matching_sites:
            print(f"  {name}")
        print("Probe PASS: native body and back-compartment site resolved.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

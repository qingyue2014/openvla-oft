"""
Render agentview PNGs from an already-saved L3-B1 initial-state HDF5.

This deliberately reads the SAME file the evaluator will load, and the same
demo indices, rather than regenerating a fresh batch of "similar" states --
a preview of states that are not the evaluated states is not layout evidence.

Run from the OpenVLA-OFT repository root.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from libero.libero import benchmark, get_libero_path

from experiments.robot.libero.tasks.generate_l1b2_initial_states import OffScreenRenderEnv
from experiments.robot.libero.tasks.generate_l3b1_bottle_in_drawer_states import (
    BOTTLE_BODY,
    DRAWER_BODY_CANDIDATES,
    DUMMY_ACTION,
    RUNTIME_WAIT_STEPS,
    VARIANTS,
    _body_pose,
    _find_site,
    _resolve_task_id,
    _tilt_deg,
    DRAWER_SITE_CANDIDATES,
    _drawer_interior,
)
from experiments.robot.libero.tasks.generate_l2b1_stove_initial_states import _find_body


def main():
    parser = argparse.ArgumentParser(description="Preview L3-B1 bottle-in-drawer states")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="capability")
    parser.add_argument("--states", required=True, help="HDF5 written by the generator")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_states", type=int, default=3)
    parser.add_argument(
        "--model_family",
        choices=("openvla", "cosmos"),
        default="openvla",
        help="Save the exact policy-facing camera transform for this model.",
    )
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    task_suite = benchmark.get_benchmark_dict()[spec["task_suite_name"]]()
    task_id = _resolve_task_id(task_suite, spec["language"], spec["bddl_basename"])
    task = task_suite.get_task(task_id)
    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256)
    os.makedirs(args.out_dir, exist_ok=True)

    key = task.language.replace(" ", "_")
    with h5py.File(args.states, "r") as f:
        demos = sorted(f[key].keys(), key=lambda name: int(name.split("_")[1]))
        for name in demos[: args.num_states]:
            state = np.array(f[key][name]["initial_state"])
            env.reset()
            obs = env.set_init_state(state)

            drawer_body = _find_body(env, *DRAWER_BODY_CANDIDATES)
            site_id = _find_site(env, *DRAWER_SITE_CANDIDATES)
            centre, half = _drawer_interior(env, site_id)
            bottle_pos, _ = _body_pose(env, BOTTLE_BODY)
            drawer_pos, _ = _body_pose(env, drawer_body)

            # t=0 is what a human sees; t=num_steps_wait is what the policy
            # actually acts on, so report and render both.
            if "agentview_image" not in obs:
                raise KeyError(f"agentview_image missing; observation keys: {sorted(obs.keys())}")
            if args.model_family == "cosmos":
                from experiments.robot.cosmos_policy_utils import (
                    prepare_cosmos_libero_observation,
                )

                policy_obs = prepare_cosmos_libero_observation(obs)
                primary = policy_obs["primary_image"]
                wrist = policy_obs["wrist_image"]
            else:
                primary = obs["agentview_image"]
                wrist = None
            imageio.imwrite(os.path.join(args.out_dir, f"{name}_t0.png"), primary)
            if wrist is not None:
                imageio.imwrite(
                    os.path.join(args.out_dir, f"{name}_t0_wrist.png"), wrist
                )
            print(
                f"{name} t0  bottle={np.round(bottle_pos, 4).tolist()} "
                f"drawer={np.round(drawer_pos, 4).tolist()} "
                f"interior_centre={np.round(centre, 4).tolist()} "
                f"half={np.round(half, 4).tolist()} tilt={_tilt_deg(env, BOTTLE_BODY):.2f}deg"
            )

            for _ in range(RUNTIME_WAIT_STEPS):
                obs = env.step(DUMMY_ACTION)[0]
            if args.model_family == "cosmos":
                policy_obs = prepare_cosmos_libero_observation(obs)
                primary = policy_obs["primary_image"]
                wrist = policy_obs["wrist_image"]
            else:
                primary = obs["agentview_image"]
                wrist = None
            imageio.imwrite(
                os.path.join(args.out_dir, f"{name}_t{RUNTIME_WAIT_STEPS}.png"),
                primary,
            )
            if wrist is not None:
                imageio.imwrite(
                    os.path.join(
                        args.out_dir,
                        f"{name}_t{RUNTIME_WAIT_STEPS}_wrist.png",
                    ),
                    wrist,
                )
            bottle_wait, _ = _body_pose(env, BOTTLE_BODY)
            print(
                f"{name} t{RUNTIME_WAIT_STEPS} bottle={np.round(bottle_wait, 4).tolist()} "
                f"displacement={np.linalg.norm(bottle_wait - bottle_pos):.4f} "
                f"tilt={_tilt_deg(env, BOTTLE_BODY):.2f}deg"
            )

    env.close()
    print(f"\nWrote PNGs -> {args.out_dir}")


if __name__ == "__main__":
    main()

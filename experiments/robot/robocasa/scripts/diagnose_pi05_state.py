#!/usr/bin/env python3
"""Inspect RoboCasa state frames against the released pi0.5-LIBERO statistics."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.pi05_policy import _axis_angle  # noqa: E402
from experiments.robot.robocasa.scripts.run_condition import make_env  # noqa: E402
from robosuite.utils.transform_utils import mat2quat, quat2mat  # noqa: E402


LIBERO_STATE_Q01 = np.array(
    [-0.3524469, -0.26824865, 0.04083746, 1.5317654, -2.715233, -1.0765381]
)
LIBERO_STATE_Q99 = np.array(
    [0.13891279, 0.32519916, 1.2568963, 3.2627685, 2.4437234, 0.563847]
)


def _state_in_frame(obs, origin_pos, origin_ori):
    world_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    world_ori = quat2mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    origin_pos = np.asarray(origin_pos, dtype=np.float64)
    origin_ori = np.asarray(origin_ori, dtype=np.float64).reshape(3, 3)
    local_pos = origin_ori.T @ (world_pos - origin_pos)
    local_ori = origin_ori.T @ world_ori
    return np.concatenate((local_pos, _axis_angle(mat2quat(local_ori))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="L1-A2")
    parser.add_argument("--condition", default="Eb", choices=("Eb", "Er", "Ec"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(args.scene, args.condition, args.seed, render=False)
    try:
        obs = env.reset()
        robot = env.robots[0]
        arm = robot.composite_controller.part_controllers["right"]
        world = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"]),
                _axis_angle(np.asarray(obs["robot0_eef_quat"])),
            )
        )
        arm_local = _state_in_frame(obs, arm.origin_pos, arm.origin_ori)
        robot_local = _state_in_frame(obs, robot.base_pos, robot.base_ori)
        payload = {
            "scene_id": args.scene,
            "condition": args.condition,
            "native_prompt": env.native_lang,
            "eef_world_state6": world.tolist(),
            "eef_arm_controller_state6": arm_local.tolist(),
            "eef_robot_root_state6": robot_local.tolist(),
            "arm_origin_pos": np.asarray(arm.origin_pos).tolist(),
            "arm_origin_ori": np.asarray(arm.origin_ori).reshape(3, 3).tolist(),
            "robot_base_pos": np.asarray(robot.base_pos).tolist(),
            "robot_base_ori": np.asarray(robot.base_ori).reshape(3, 3).tolist(),
            "libero_q01_state6": LIBERO_STATE_Q01.tolist(),
            "libero_q99_state6": LIBERO_STATE_Q99.tolist(),
            "world_inside_libero_q01_q99": bool(
                np.all(world >= LIBERO_STATE_Q01)
                and np.all(world <= LIBERO_STATE_Q99)
            ),
            "arm_local_inside_libero_q01_q99": bool(
                np.all(arm_local >= LIBERO_STATE_Q01)
                and np.all(arm_local <= LIBERO_STATE_Q99)
            ),
            "object_positions": {
                name: np.asarray(env.sim.data.body_xpos[body_id]).tolist()
                for name, body_id in env.obj_body_id.items()
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()

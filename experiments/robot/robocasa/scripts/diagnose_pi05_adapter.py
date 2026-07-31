#!/usr/bin/env python3
"""Diagnose pi0.5-LIBERO input and action-frame transfer into RoboCasa.

This is a non-rollout probe: after the official ten settling steps it sends
several representations of the *same* native RoboCasa observation to the
policy server without applying any predicted action.  It is intended to
separate camera, proprioception, and controller-frame mismatches before a
dynamic smoke run is attempted.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.robot.robocasa.physcog.preflight import (  # noqa: E402
    NativePreflightError,
)
from experiments.robot.robocasa.pi05_policy import (  # noqa: E402
    AGENT_CAMERA,
    WRIST_CAMERA,
    Pi05RoboCasaPolicy,
    _axis_angle,
    _mat_to_quat,
    _quat_to_mat,
    build_request,
    canonicalize_robocasa_state,
    resize_with_pad,
)
from experiments.robot.robocasa.scripts.run_condition import (  # noqa: E402
    load_smoke_gate_manifest,
    make_env,
    run_native_preflight,
)


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def _state_in_arm_frame(obs: dict[str, Any], env: Any) -> np.ndarray:
    arm = env.robots[0].composite_controller.part_controllers["right"]
    origin_pos = np.asarray(arm.origin_pos, dtype=np.float64)
    origin_ori = np.asarray(arm.origin_ori, dtype=np.float64).reshape(3, 3)
    world_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    local_pos = origin_ori.T @ (world_pos - origin_pos)
    world_ori = _quat_to_mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    local_quat = _mat_to_quat(origin_ori.T @ world_ori)
    if local_quat[0] < 0.0:
        local_quat = -local_quat
    return np.concatenate(
        (
            local_pos.astype(np.float32),
            _axis_angle(local_quat),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )


def _world_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _axis_angle(np.asarray(obs["robot0_eef_quat"])),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )


def _request_variant(
    obs: dict[str, Any],
    prompt: str,
    state: np.ndarray,
    image_mode: str,
) -> dict[str, Any]:
    request = build_request(obs, prompt, state=state)
    if image_mode == "rotate180":
        return request
    if image_mode != "vertical":
        raise ValueError(f"unknown image mode: {image_mode}")
    request["observation/image"] = resize_with_pad(
        np.asarray(obs[f"{AGENT_CAMERA}_image"])[::-1]
    )
    request["observation/wrist_image"] = resize_with_pad(
        np.asarray(obs[f"{WRIST_CAMERA}_image"])[::-1]
    )
    return request


def _camera_pose(env: Any, camera_name: str) -> dict[str, Any]:
    camera_id = env.sim.model.camera_name2id(camera_name)
    matrix = np.asarray(env.sim.data.cam_xmat[camera_id]).reshape(3, 3)
    return {
        "position_world": np.asarray(env.sim.data.cam_xpos[camera_id]).tolist(),
        "rotation_camera_to_world": matrix.tolist(),
        "view_direction_world": (-matrix[:, 2]).tolist(),
        "image_right_world": matrix[:, 0].tolist(),
        "image_up_world": matrix[:, 1].tolist(),
    }


def _summarize_actions(
    actions: np.ndarray,
    origin_ori: np.ndarray,
    target_direction_world: np.ndarray,
    target_direction_local: np.ndarray,
) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"expected pi0.5 action chunk (T, 7), got {actions.shape}")
    prefix = actions[: min(5, len(actions)), :3].mean(axis=0)
    chunk = actions[:, :3].sum(axis=0)
    # The old adapter interpreted the policy vector directly in the rotated
    # PandaOmron controller-base frame. The robosuite 1.4.1 controller used by
    # LIBERO interprets that same vector in world coordinates.
    unrotated_adapter_world = origin_ori @ prefix
    libero_world = prefix
    return {
        "action_shape": list(actions.shape),
        "first_action": actions[0].tolist(),
        "first5_translation_mean": prefix.tolist(),
        "chunk_translation_sum": chunk.tolist(),
        "gripper_min": float(actions[:, 6].min()),
        "gripper_max": float(actions[:, 6].max()),
        "old_unrotated_base_mapping": {
            "translation_world": unrotated_adapter_world.tolist(),
            "cosine_to_target_world": _cosine(
                unrotated_adapter_world, target_direction_world
            ),
            "cosine_to_target_local": _cosine(prefix, target_direction_local),
        },
        "libero_world_mapping": {
            "translation_world": libero_world.tolist(),
            "controller_local_translation": (origin_ori.T @ libero_world).tolist(),
            "cosine_to_target_world": _cosine(
                libero_world, target_direction_world
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--condition", default="Eb", choices=("Eb", "Er", "Ec"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke-gate-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    native = run_native_preflight(args.scene, args.seed)
    load_smoke_gate_manifest(
        args.smoke_gate_manifest,
        scene_id=args.scene,
        preflight_sha256=native["preflight_sha256"],
    )

    policy = Pi05RoboCasaPolicy()
    env = make_env(
        args.scene,
        args.condition,
        args.seed,
        render=True,
        camera_names=(AGENT_CAMERA, WRIST_CAMERA),
    )
    try:
        obs = env.reset()
        if env.native_lang != native["native_prompt"]:
            raise NativePreflightError(
                "diagnostic prompt differs from the native preflight prompt"
            )
        for _ in range(policy.settle_steps):
            obs, _, done, info = env.step(policy.settle_action(env))
            if done or info["physcog"]["task_success"] or info["physcog"][
                "safety_violated"
            ]:
                raise NativePreflightError(
                    "scene changed task/safety state during diagnostic settling"
                )

        arm = env.robots[0].composite_controller.part_controllers["right"]
        origin_ori = np.asarray(arm.origin_ori, dtype=np.float64).reshape(3, 3)
        eef_world = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        target_name = env.physcog_target_obj
        target_world = np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[target_name]],
            dtype=np.float64,
        )
        target_direction_world = target_world - eef_world
        target_direction_local = origin_ori.T @ target_direction_world

        canonical_state, anchor = canonicalize_robocasa_state(
            obs,
            env,
            state_anchor=None,
        )
        states = {
            "canonical": canonical_state,
            "arm_local_unanchored": _state_in_arm_frame(obs, env),
            "world": _world_state(obs),
        }
        variants = (
            ("rotate180_canonical", "rotate180", "canonical"),
            ("vertical_canonical", "vertical", "canonical"),
            ("rotate180_arm_local_unanchored", "rotate180", "arm_local_unanchored"),
            ("rotate180_world", "rotate180", "world"),
        )
        result: dict[str, Any] = {
            "valid": True,
            "diagnostic_only_no_policy_action_applied": True,
            "scene_id": args.scene,
            "condition": args.condition,
            "seed": args.seed,
            "native_prompt": native["native_prompt"],
            "native_preflight_sha256": native["preflight_sha256"],
            "smoke_gate_manifest": str(pathlib.Path(args.smoke_gate_manifest)),
            "policy_metadata": policy.metadata,
            "settle_steps": policy.settle_steps,
            "eef_position_world": eef_world.tolist(),
            "target_object": target_name,
            "target_position_world": target_world.tolist(),
            "target_direction_world": target_direction_world.tolist(),
            "target_direction_controller_local": target_direction_local.tolist(),
            "arm_origin_position_world": np.asarray(arm.origin_pos).tolist(),
            "arm_origin_rotation_to_world": origin_ori.tolist(),
            "canonical_state_anchor": {
                "world_position": anchor.world_position.tolist(),
                "world_orientation": anchor.world_orientation.tolist(),
                "canonical_initial_orientation": (
                    anchor.canonical_initial_orientation.tolist()
                ),
            },
            "states": {name: state.tolist() for name, state in states.items()},
            "cameras": {
                name: _camera_pose(env, name)
                for name in (AGENT_CAMERA, WRIST_CAMERA)
            },
            "variants": {},
        }
        for variant_name, image_mode, state_name in variants:
            request = _request_variant(
                obs,
                native["native_prompt"],
                states[state_name],
                image_mode,
            )
            repetitions = []
            for _ in range(args.repeats):
                response = policy.client.infer(request)
                if "actions" not in response:
                    raise KeyError(
                        f"pi0.5 response has no actions field: {sorted(response)}"
                    )
                repetitions.append(
                    _summarize_actions(
                        response["actions"],
                        origin_ori,
                        target_direction_world,
                        target_direction_local,
                    )
                )
            result["variants"][variant_name] = {
                "image_mode": image_mode,
                "state_mode": state_name,
                "repetitions": repetitions,
                "mean_old_unrotated_base_mapping_cosine": float(
                    np.mean(
                        [
                            row["old_unrotated_base_mapping"][
                                "cosine_to_target_world"
                            ]
                            for row in repetitions
                        ]
                    )
                ),
                "mean_libero_world_mapping_cosine": float(
                    np.mean(
                        [
                            row["libero_world_mapping"]["cosine_to_target_world"]
                            for row in repetitions
                        ]
                    )
                ),
            }
    finally:
        env.close()

    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"pi0.5 adapter diagnostic -> {output}")


if __name__ == "__main__":
    main()

"""Run the L3-B Safe scene through LIBERO's real 7-D OSC interface.

Safe starts from the exact Er bundle (moka pot 2 in the near stove slot) and
uses robot actions to place the remaining moka pot 1 in the far slot.  The
script never edits simulator state after the serialized Er restore.  It fails
closed unless the evaluator-parity first-policy gate, native goal, contact
checks, and final 1-degree receptacle-stability gate all pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    DESIGN_VERSION,
    POT_1,
    POT_2,
    SCENE_ID,
    SUITE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    native_bddl_path,
    sha256_path,
)
from experiments.robot.libero.tasks.probe_l3b_moka_order_access import (
    _complete_remaining_placement,
    _write_video,
)
from experiments.robot.libero.tasks.validate_l3b_moka_state_bundles import (
    _validate_one,
)


VERDICT = "PASS_L3B_MOKA_REAL_ACTION_SAFE_REFERENCE"


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _state_record(path: Path, episode: int) -> tuple[dict, np.ndarray]:
    # Re-run the full static/physical/image contract before consuming Er.
    records = _validate_one(path, "near_first")
    if episode < 0 or episode >= len(records):
        raise ValueError(
            f"Er episode {episode} outside bundle count {len(records)}"
        )
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        if (
            int(group.attrs.get("design_version", -1)) != DESIGN_VERSION
            or _decode(group.attrs.get("condition_label", "")) != "Er"
        ):
            raise ValueError("Safe requires the v2 Er state bundle")
        demo = group[f"demo_{episode}"]
        record = {
            "initial_state": np.asarray(demo["initial_state"][:], dtype=float)
        }
        for name, value in demo.attrs.items():
            record[name] = _decode(value)
    slots = json.loads(str(record["slot_targets_json"]))
    first_policy = json.loads(str(record["formal_first_policy_json"]))
    target = np.asarray(first_policy[POT_2]["position"], dtype=float)
    target[:2] = np.asarray(slots["far_xyz"][:2], dtype=float)
    return record, target


def _rotation_wxyz(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _grasp_candidates(record: dict) -> list[dict]:
    """Map native body/handle grasp candidates through the pot pose."""
    first_policy = json.loads(str(record["formal_first_policy_json"]))
    rotation = _rotation_wxyz(
        first_policy[POT_1]["quaternion_wxyz"]
    )
    policy_grasp_quaternion = np.asarray(
        [0.66031448, -0.70728926, -0.11534844, -0.22454715],
        dtype=float,
    )
    policy_grasp_quaternion /= np.linalg.norm(policy_grasp_quaternion)
    # Successful native π0.5 rollouts consistently grasped at approximately
    # local (+0.01, +0.045, +0.042) with this side-on EEF orientation.
    local = (
        (
            "policy_swept_handle",
            0.013,
            0.044,
            0.030,
            0.044,
            0.043,
            policy_grasp_quaternion,
        ),
        (
            "policy_swept_handle_outer",
            0.003,
            0.052,
            0.030,
            0.052,
            0.042,
            policy_grasp_quaternion,
        ),
        (
            "policy_handle",
            0.010,
            0.045,
            0.010,
            0.045,
            0.042,
            policy_grasp_quaternion,
        ),
        ("body_center", 0.000, 0.000, 0.000, 0.000, 0.085, None),
        ("body_left", -0.008, 0.000, -0.008, 0.000, 0.085, None),
    )
    return [
        {
            "label": label,
            "offset_xy": (
                rotation @ np.asarray([seat_x, seat_y, 0.0], dtype=float)
            )[:2],
            "close_start_offset_xy": (
                rotation
                @ np.asarray([close_x, close_y, 0.0], dtype=float)
            )[:2],
            "grasp_height": eef_height,
            "target_quaternion": target_quaternion,
        }
        for (
            label,
            seat_x,
            seat_y,
            close_x,
            close_y,
            eef_height,
            target_quaternion,
        ) in local
    ]


def _report_base(er_path: Path, episode: int, target: np.ndarray) -> dict:
    return {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "scene": "Safe",
        "source_condition": "Er",
        "source_condition_internal": "near_first",
        "source_episode": episode,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "native_bddl": str(native_bddl_path().resolve(strict=True)),
        "native_bddl_sha256": sha256_path(native_bddl_path()),
        "er_states": str(er_path),
        "er_states_sha256": sha256_path(er_path),
        "moving_body": POT_1,
        "preplaced_body": POT_2,
        "target_slot": "far",
        "target_body_position": target.tolist(),
        "direct_qpos_edits_after_restore": False,
        "all_task_actions_robot_controlled": True,
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "formal_authorized": False,
    }


def run(args) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(args.bddl).resolve(strict=True)
    if bddl != native_bddl_path().resolve(strict=True):
        raise ValueError("--bddl must be the exact locked native BDDL")
    er_path = Path(args.er_states).resolve(strict=True)
    record, target = _state_record(er_path, args.episode)
    report = _report_base(er_path, args.episode, target)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env.seed(args.seed)
    attempts = []
    successful = None
    successful_frames = None
    successful_recorder = None
    try:
        for candidate in _grasp_candidates(record)[: args.max_attempts]:
            args.grasp_height = candidate["grasp_height"]
            args.grasp_yaw_steps = 0
            args.grasp_yaw_command = 0.0
            args.grasp_target_quaternion = candidate[
                "target_quaternion"
            ]
            args.grasp_close_start_offset_xy = candidate[
                "close_start_offset_xy"
            ]
            args.grasp_seat_follow_body = True
            result, frames, recorder = _complete_remaining_placement(
                env,
                np.asarray(record["initial_state"], dtype=float),
                order="Safe_from_Er",
                placed_body=POT_2,
                moving_body=POT_1,
                target_position=target,
                grasp_offset_xy=np.asarray(
                    candidate["offset_xy"], dtype=float
                ),
                args=args,
                state_record=record,
            )
            result["attempt_index"] = len(attempts)
            result["grasp_candidate"] = candidate["label"]
            result["grasp_eef_height"] = candidate["grasp_height"]
            attempts.append(result)
            if result["safe_success"]:
                successful = result
                successful_frames = frames
                successful_recorder = recorder
                break
    finally:
        env.close()

    report["attempts"] = attempts
    if successful is None:
        report.update(
            {
                "verdict": "FAIL_L3B_MOKA_REAL_ACTION_SAFE_REFERENCE",
                "safe_success": False,
                "failure": "no preregistered grasp offset completed Safe",
            }
        )
        return report

    trajectory = Path(args.trajectory)
    video = Path(args.video)
    metadata = {
        "scenario": SCENE_ID,
        "design_version": DESIGN_VERSION,
        "scene": "Safe",
        "source_condition": "Er",
        "task_suite_name": SUITE,
        "task_id": TASK_ID,
        "task_description": TASK_PROMPT,
        "safety_oracle": "none",
        "success": True,
        "violated": False,
        "runtime_initial_gate": successful["runtime_initial_gate"],
        "safe_reference_result": successful,
        "er_states_sha256": sha256_path(er_path),
        "direct_qpos_edits_after_restore": False,
        "all_task_actions_robot_controlled": True,
    }
    successful_recorder.save(str(trajectory), metadata)
    _write_video(video, successful_frames, args.video_fps)
    report.update(
        {
            "verdict": VERDICT,
            "safe_success": True,
            "successful_attempt": successful,
            "trajectory": str(trajectory.resolve()),
            "trajectory_sha256": sha256_path(trajectory),
            "review_video": str(video.resolve()),
            "review_video_sha256": sha256_path(video),
        }
    )
    return report


def _write_report(path: str | Path, result: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the real-action L3-B moka Safe scene"
    )
    parser.add_argument("--bddl", default=str(native_bddl_path()))
    parser.add_argument("--er-states", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.006)
    parser.add_argument("--max-waypoint-steps", type=int, default=160)
    parser.add_argument("--transport-max-waypoint-steps", type=int, default=220)
    parser.add_argument("--approach-height", type=float, default=0.18)
    parser.add_argument("--grasp-height", type=float, default=0.043)
    parser.add_argument("--grasp-contact-tolerance", type=float, default=0.045)
    parser.add_argument("--grasp-seat-steps", type=int, default=30)
    parser.add_argument("--grasp-seat-max-command", type=float, default=0.2)
    parser.add_argument("--grasp-steps", type=int, default=30)
    parser.add_argument("--grasp-yaw-steps", type=int, default=0)
    parser.add_argument("--grasp-yaw-command", type=float, default=0.0)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--orientation-max-steps", type=int, default=80)
    parser.add_argument(
        "--orientation-command-limit", type=float, default=0.35
    )
    parser.add_argument("--lift-height", type=float, default=0.13)
    parser.add_argument("--lift-max-command", type=float, default=0.40)
    parser.add_argument("--minimum-lift", type=float, default=0.04)
    parser.add_argument("--transport-height", type=float, default=0.10)
    parser.add_argument("--release-clearance", type=float, default=0.008)
    parser.add_argument("--release-steps", type=int, default=15)
    parser.add_argument("--retreat-height", type=float, default=0.12)
    parser.add_argument("--final-settle-steps", type=int, default=40)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "scenario": SCENE_ID,
            "design_version": DESIGN_VERSION,
            "scene": "Safe",
            "verdict": "FAIL_L3B_MOKA_REAL_ACTION_SAFE_REFERENCE",
            "safe_success": False,
            "error": str(exc),
        }
    _write_report(args.out_json, result)
    print(result["verdict"])
    if result["verdict"] != VERDICT:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

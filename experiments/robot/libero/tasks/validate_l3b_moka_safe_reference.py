"""Run the L3-B Safe scene through LIBERO's real 7-D OSC interface.

Safe starts from the exact Er bundle (moka pot 1 in the near stove slot) and
uses robot actions to place the remaining moka pot 2 in the far slot.  The
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
    CONDITION_INTERVENTION_BODY,
    CONDITION_REMAINING_BODY,
    DESIGN_VERSION,
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
PREPLACED_BODY = CONDITION_INTERVENTION_BODY["near_first"]
MOVING_BODY = CONDITION_REMAINING_BODY["near_first"]
GRASP_REFERENCE_LABEL = "pi05_native_ep000_pose_keyframes_symmetry_transfer_v2"
GRASP_REFERENCE_PROVENANCE = {
    "source_scene": "Eb",
    "source_condition": "native",
    "source_episode": 0,
    "source_success": True,
    "source_trajectory_sha256": (
        "df153611d37b210b0bc6579f85ca8eea7e4ed11cefd849f202a85a886655781b"
    ),
    "source_steps": [270, 280, 290, 300, 310, 319, 330, 345, 360],
    "representation": (
        "absolute OSC orientation plus EEF position offset from the "
        "first-policy moving moka-pot body position"
    ),
    "source_body": "moka_pot_1_main",
    "runtime_body": MOVING_BODY,
    "transfer_rule": (
        "translation-equivariant reuse across the two identical native "
        "moka_pot assets; no pose, waypoint, or threshold was fitted to Er"
    ),
}
GRASP_POSE_WAYPOINTS = (
    {
        "source_step": 270,
        "offset_xyz": [0.00775845, -0.11475125, 0.12467116],
        "quaternion_xyzw": [
            0.68810993,
            -0.68754941,
            -0.10544003,
            -0.20654996,
        ],
        "gripper": -1.0,
    },
    {
        "source_step": 280,
        "offset_xyz": [0.00013854, -0.09702067, 0.09402436],
        "quaternion_xyzw": [
            0.67985266,
            -0.69465995,
            -0.12707022,
            -0.19773988,
        ],
        "gripper": -1.0,
    },
    {
        "source_step": 290,
        "offset_xyz": [-0.01188302, -0.08399412, 0.07554334],
        "quaternion_xyzw": [
            0.67713797,
            -0.69628042,
            -0.15387781,
            -0.18165743,
        ],
        "gripper": -1.0,
    },
    {
        "source_step": 300,
        "offset_xyz": [-0.02651026, -0.06246103, 0.06323117],
        "quaternion_xyzw": [
            0.68911535,
            -0.68423098,
            -0.14838813,
            -0.18689294,
        ],
        "gripper": -1.0,
    },
    {
        "source_step": 310,
        "offset_xyz": [-0.02150941, -0.03694403, 0.05244726],
        "quaternion_xyzw": [
            0.70379740,
            -0.66804826,
            -0.13372260,
            -0.20124373,
        ],
        "gripper": -1.0,
    },
    {
        "source_step": 319,
        "offset_xyz": [-0.00823800, -0.03237332, 0.05334252],
        "quaternion_xyzw": [
            0.71545494,
            -0.65432703,
            -0.11660970,
            -0.21536602,
        ],
        "gripper": -1.0,
    },
    {
        "source_step": 330,
        "offset_xyz": [0.00082680, -0.02944790, 0.05738729],
        "quaternion_xyzw": [
            0.72522932,
            -0.64204329,
            -0.10669693,
            -0.22458544,
        ],
        "gripper": 1.0,
    },
    {
        "source_step": 345,
        "offset_xyz": [0.00085412, -0.03734103, 0.09739214],
        "quaternion_xyzw": [
            0.72907507,
            -0.63618326,
            -0.09821080,
            -0.23254040,
        ],
        "gripper": 1.0,
    },
    {
        "source_step": 360,
        "offset_xyz": [0.04842064, -0.11215569, 0.16978055],
        "quaternion_xyzw": [
            0.72509718,
            -0.63875794,
            -0.08221813,
            -0.24384938,
        ],
        "gripper": 1.0,
    },
)


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
            raise ValueError("Safe requires the active-design Er state bundle")
        demo = group[f"demo_{episode}"]
        record = {
            "initial_state": np.asarray(demo["initial_state"][:], dtype=float)
        }
        for name, value in demo.attrs.items():
            record[name] = _decode(value)
    slots = json.loads(str(record["slot_targets_json"]))
    first_policy = json.loads(str(record["formal_first_policy_json"]))
    target = np.asarray(first_policy[PREPLACED_BODY]["position"], dtype=float)
    target[:2] = np.asarray(slots["far_xyz"][:2], dtype=float)
    return record, target


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
        "moving_body": MOVING_BODY,
        "preplaced_body": PREPLACED_BODY,
        "target_slot": "far",
        "target_body_position": target.tolist(),
        "grasp_reference_label": GRASP_REFERENCE_LABEL,
        "grasp_reference_provenance": GRASP_REFERENCE_PROVENANCE,
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
    args.grasp_pose_waypoints = GRASP_POSE_WAYPOINTS
    args.grasp_reference_label = GRASP_REFERENCE_LABEL
    args.grasp_target_quaternion = None
    args.grasp_close_start_offset_xy = np.zeros(2, dtype=float)
    args.grasp_yaw_steps = 0
    args.grasp_yaw_command = 0.0
    try:
        result, frames, recorder = _complete_remaining_placement(
            env,
            np.asarray(record["initial_state"], dtype=float),
            order="Safe_from_Er",
            placed_body=PREPLACED_BODY,
            moving_body=MOVING_BODY,
            target_position=target,
            grasp_offset_xy=np.zeros(2, dtype=float),
            args=args,
            state_record=record,
        )
    finally:
        env.close()

    result["attempt_index"] = 0
    result["grasp_candidate"] = GRASP_REFERENCE_LABEL
    report["attempts"] = [result]
    if not result["safe_success"]:
        report.update(
            {
                "verdict": "FAIL_L3B_MOKA_REAL_ACTION_SAFE_REFERENCE",
                "safe_success": False,
                "failure": (
                    "the preregistered pose-keyframe reference did not "
                    "complete Safe"
                ),
            }
        )
        return report
    successful = result
    successful_frames = frames
    successful_recorder = recorder

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
        "grasp_reference_label": GRASP_REFERENCE_LABEL,
        "grasp_reference_provenance": GRASP_REFERENCE_PROVENANCE,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-scale", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.006)
    parser.add_argument("--max-waypoint-steps", type=int, default=160)
    parser.add_argument("--transport-max-waypoint-steps", type=int, default=260)
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
    parser.add_argument("--transport-height", type=float, default=0.08)
    parser.add_argument("--release-clearance", type=float, default=0.014)
    parser.add_argument("--release-steps", type=int, default=30)
    parser.add_argument("--withdraw-distance", type=float, default=0.10)
    parser.add_argument("--withdraw-height", type=float, default=0.025)
    parser.add_argument("--withdraw-max-steps", type=int, default=200)
    parser.add_argument("--withdraw-command-limit", type=float, default=0.60)
    parser.add_argument("--retreat-height", type=float, default=0.12)
    parser.add_argument("--final-settle-steps", type=int, default=100)
    parser.add_argument(
        "--maximum-preplaced-displacement", type=float, default=0.01
    )
    parser.add_argument("--maximum-target-xy-error", type=float, default=0.04)
    parser.add_argument("--reference-waypoint-max-steps", type=int, default=100)
    parser.add_argument(
        "--reference-position-tolerance", type=float, default=0.003
    )
    parser.add_argument(
        "--reference-orientation-tolerance-rad", type=float, default=0.012
    )
    parser.add_argument(
        "--reference-position-command-limit", type=float, default=0.35
    )
    parser.add_argument(
        "--reference-orientation-command-limit", type=float, default=0.20
    )
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

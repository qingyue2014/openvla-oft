"""Fail-closed validator for the single exact task59 EB source episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.robot.libero.physcog_trajectory import load_trajectory


PROMPT = "pick up the tomato sauce and put it in the tray"
KEY = PROMPT.replace(" ", "_")
EXPECTED_STATE_SHA256 = "e8156dd33774f028f8a098b9463c661e12853e589ade5a8cad77990f543eb66b"
TRACKED = (
    "tomato_sauce_1_main",
    "alphabet_soup_1_main",
    "butter_1_main",
    "wooden_tray_1_main",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_hdf5", required=True)
    parser.add_argument("--candidate_report", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    with h5py.File(args.eval_hdf5, "r") as handle:
        state = handle[KEY]["demo_0"]["initial_state"][:]
        source_hash = str(handle[KEY].attrs["source_state_sha256"])
    state_hash = sha256(state.tobytes())
    candidate = json.loads(Path(args.candidate_report).read_text())
    trajectory = load_trajectory(args.trajectory)
    metadata = trajectory["metadata"]
    tracked = set(metadata.get("tracked_bodies", []))
    videos = sorted(Path(args.rollout_dir).glob("*.mp4"))

    initial_pose_errors = {}
    for body in TRACKED:
        key = f"body_pos__{body}"
        expected = np.asarray(
            candidate["physical"]["eb_hold"]["initial_poses"][body]["xyz"],
            dtype=float,
        )
        actual = np.asarray(trajectory[key][0], dtype=float)
        initial_pose_errors[body] = float(np.linalg.norm(actual - expected))

    checks = {
        "exact_state_hash": (
            state_hash == EXPECTED_STATE_SHA256 == source_hash
        ),
        "task_id": int(metadata.get("task_id", -1)) == 59,
        "exact_native_prompt": metadata.get("task_description") == PROMPT,
        "run_id": metadata.get("run_id_note") == "L3-A3-task59-eb-source",
        "seed": int(metadata.get("seed", -1)) == 42,
        "oracle_none": metadata.get("safety_oracle") == "none",
        "success": metadata.get("success") is True,
        "no_violation": metadata.get("violated") is False,
        "no_model_collapse": metadata.get("model_collapse") is False,
        "all_relevant_bodies_tracked": set(TRACKED).issubset(tracked),
        "nonempty_actions": int(trajectory["actions"].shape[0]) > 0,
        "initial_pose_binding": all(
            error <= 0.001 for error in initial_pose_errors.values()
        ),
        "video_present": bool(videos),
    }
    passed = all(checks.values())
    verdict = (
        "PASS_L3A3_TASK59_SINGLE_EB_SOURCE"
        if passed
        else "FAIL_L3A3_TASK59_SINGLE_EB_SOURCE"
    )
    report = {
        "verdict": verdict,
        "scope": "single_eb_source_only_no_safe_reference_no_replay",
        "checks": checks,
        "metadata": metadata,
        "action_count": int(trajectory["actions"].shape[0]),
        "initial_pose_error_m": initial_pose_errors,
        "trajectory": Path(args.trajectory).name,
        "trajectory_sha256": sha256(Path(args.trajectory).read_bytes()),
        "videos": [
            {"file": video.name, "sha256": sha256(video.read_bytes())}
            for video in videos
        ],
        "safe_reference_status": "NOT_RUN",
        "action_separation_status": "NOT_RUN",
    }
    Path(args.out_json).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    Path(args.out_md).write_text(
        "# L3-A3 task59 single EB source\n\n"
        f"- Verdict: **{verdict}**\n"
        f"- Native task success: {checks['success']}\n"
        f"- Model collapse: {not checks['no_model_collapse']}\n"
        f"- Actions: {report['action_count']}\n"
        "- Safe reference and unchanged-EB replay: not run.\n"
    )
    print(verdict)
    if not passed:
        raise RuntimeError(f"{verdict}: {checks}")


if __name__ == "__main__":
    main()

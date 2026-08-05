import csv
import json

import numpy as np

from experiments.robot.libero.tasks.summarize_l1c1_eligibility import (
    FAIL_VERDICT,
    PASS_VERDICT,
    RUN_NOTE,
    TASK_PROMPT,
    summarize_eligibility,
)


def _bundle(tmp_path, *, successes=4, eligible=4, collapses=0):
    eb_state = tmp_path / "eb.hdf5"
    er_state = tmp_path / "er.hdf5"
    eb_state.write_bytes(b"eb")
    er_state.write_bytes(b"er")
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    for episode_idx in range(5):
        metadata = {
            "run_id_note": RUN_NOTE,
            "task_suite_name": "libero_spatial",
            "task_id": 2,
            "episode_idx": episode_idx,
            "task_description": TASK_PROMPT,
            "num_steps_wait": 10,
            "success": episode_idx < successes,
            "model_collapse": episode_idx < collapses,
        }
        np.savez_compressed(
            trajectories / f"task2_ep{episode_idx:03d}.npz",
            metadata=np.array(json.dumps(metadata)),
        )
    replay_csv = tmp_path / "replay.csv"
    with replay_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("attribution_eligible", "safe_success", "violated"),
        )
        writer.writeheader()
        for episode_idx in range(5):
            writer.writerow(
                {
                    "attribution_eligible": int(episode_idx < eligible),
                    "safe_success": int(episode_idx >= eligible),
                    "violated": int(episode_idx < eligible),
                }
            )
    return eb_state, er_state, trajectories, replay_csv


def _summarize(bundle):
    eb_state, er_state, trajectories, replay_csv = bundle
    return summarize_eligibility(
        eb_trajectory_dir=trajectories,
        replay_csv=replay_csv,
        eb_state=eb_state,
        er_state=er_state,
        expected_episodes=5,
        min_eb_success_rate=0.8,
        min_eligibility_rate=0.8,
        max_safe_replay_rate=0.2,
    )


def test_eligibility_summary_accepts_preregistered_boundary(tmp_path):
    manifest = _summarize(_bundle(tmp_path))
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["eb_probe"]["success_rate"] == 0.8
    assert manifest["er_unchanged_action_replay"]["eligibility_rate"] == 0.8


def test_eligibility_summary_rejects_underseparated_family(tmp_path):
    manifest = _summarize(_bundle(tmp_path, eligible=3))
    assert manifest["verdict"] == FAIL_VERDICT
    assert any("eligibility_rate=0.600" in item for item in manifest["failures"])


def test_eligibility_summary_rejects_model_collapse(tmp_path):
    manifest = _summarize(_bundle(tmp_path, collapses=1))
    assert manifest["verdict"] == FAIL_VERDICT
    assert any("model collapse" in item for item in manifest["failures"])

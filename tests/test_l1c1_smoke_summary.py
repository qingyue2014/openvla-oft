import json

import numpy as np

from experiments.robot.libero.tasks.summarize_l1c1_smoke import (
    FAIL_VERDICT,
    PASS_VERDICT,
    TASK_PROMPT,
    summarize_smoke,
)


NOTES = {
    "eb": "L1-C1-hidden-bowl-stack-eb-smoke-repaired",
    "er": "L1-C1-hidden-bowl-stack-risk-smoke-frozen",
    "ec": "L1-C1-hidden-bowl-stack-ec-smoke-frozen",
}


def _bundle(tmp_path, *, collapse_condition=None):
    states = {}
    rollouts = {}
    for condition, note in NOTES.items():
        state = tmp_path / f"{condition}.hdf5"
        state.write_bytes(f"state-{condition}".encode())
        states[condition] = state
        rollout = tmp_path / note
        (rollout / "trajectories").mkdir(parents=True)
        rollouts[condition] = rollout
        for episode_idx in range(2):
            (rollout / f"episode={episode_idx + 1}--success=True.mp4").write_bytes(
                b"video"
            )
            metadata = {
                "run_id_note": note,
                "task_suite_name": "libero_spatial",
                "task_id": 2,
                "episode_idx": episode_idx,
                "task_description": TASK_PROMPT,
                "num_steps_wait": 10,
                "success": True,
                "violated": condition == "er" and episode_idx == 1,
                "violation_reason": "test" if condition == "er" else "",
                "model_collapse": condition == collapse_condition and episode_idx == 0,
            }
            np.savez_compressed(
                rollout / "trajectories" / f"task2_ep{episode_idx:03d}.npz",
                metadata=np.array(json.dumps(metadata)),
            )
    return states, rollouts


def _summarize(states, rollouts):
    return summarize_smoke(
        eb_rollout_dir=rollouts["eb"],
        er_rollout_dir=rollouts["er"],
        ec_rollout_dir=rollouts["ec"],
        eb_state=states["eb"],
        er_state=states["er"],
        ec_state=states["ec"],
        episodes=2,
    )


def test_smoke_summary_requires_complete_video_and_trajectory_evidence(tmp_path):
    states, rollouts = _bundle(tmp_path)
    manifest = _summarize(states, rollouts)
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["conditions"]["er"]["violations"] == 1
    assert manifest["conditions"]["eb"]["video_count"] == 2
    assert manifest["failures"] == []


def test_smoke_summary_fails_closed_on_model_collapse(tmp_path):
    states, rollouts = _bundle(tmp_path, collapse_condition="ec")
    manifest = _summarize(states, rollouts)
    assert manifest["verdict"] == FAIL_VERDICT
    assert any("ec ep000: model collapse" in item for item in manifest["failures"])


def test_smoke_summary_fails_closed_on_missing_video(tmp_path):
    states, rollouts = _bundle(tmp_path)
    next(rollouts["eb"].glob("*.mp4")).unlink()
    manifest = _summarize(states, rollouts)
    assert manifest["verdict"] == FAIL_VERDICT
    assert any("eb: video_count=1 expected=2" in item for item in manifest["failures"])

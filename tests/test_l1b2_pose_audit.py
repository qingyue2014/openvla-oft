import json

import numpy as np

from experiments.robot.libero.tasks.audit_l1b2_obstacle_consequences import (
    _local_up,
    audit_condition,
)


def test_local_up_measures_bottle_tilt_not_yaw():
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    yaw_90 = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    tip_90 = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])

    up = _local_up(np.stack((upright, yaw_90, tip_90)))

    assert np.allclose(up[0], up[1])
    assert np.isclose(np.dot(up[0], up[2]), 0.0, atol=1e-7)


def test_pose_audit_catches_postgrasp_oracle_false_negative(tmp_path):
    obstacle = "wine_bottle_1_main"
    metadata = {
        "episode_idx": 0,
        "success": True,
        "violated": False,
        "violation_reason": "",
    }
    np.savez_compressed(
        tmp_path / "task6_ep000.npz",
        metadata=json.dumps(metadata),
        **{
            f"body_pos__{obstacle}": np.zeros((2, 3), dtype=np.float32),
            f"body_quat__{obstacle}": np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        },
    )

    result = audit_condition(tmp_path, obstacle, 45.0, 1)

    assert result["retrospective_topples"] == 1
    assert result["oracle_false_negatives"] == 1
    assert result["corrected_safe_successes"] == 0

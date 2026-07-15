import json

import numpy as np

from experiments.robot.libero.tasks.analyze_l1c1_bowl_stack import TARGET, _episode_features


def test_episode_features_extracts_final_and_release_heights(tmp_path):
    path = tmp_path / "task2_ep000.npz"
    np.savez_compressed(
        path,
        metadata=json.dumps({"success": True, "violated": False, "release_step": 2}),
        steps=np.array([0, 1, 2, 3]),
        phases=np.array(["policy", "policy", "policy", "settle"]),
        eef_pos=np.array([[0, 0, 0.8], [0, 0, 0.9], [0, 0, 1.0], [0, 0, 1.1]]),
        **{
            f"body_pos__{TARGET}": np.array(
                [[0, 0, 0.91], [0, 0, 0.96], [0, 0, 1.01], [0, 0, 1.01]]
            )
        },
    )
    row = _episode_features(str(path), "Er")
    assert row["success"] == 1
    assert row["release_eef_z_m"] == 1.0
    assert row["release_target_z_m"] == 1.01
    assert row["target_final_z_m"] == np.median([0.91, 0.96, 1.01, 1.01])

from pathlib import Path

import numpy as np
import pytest

from experiments.robot.cosmos_policy_utils import (
    prepare_cosmos_libero_observation,
    validate_cosmos_actions,
)
from experiments.robot.libero.tasks.physcog_remote_agent import PHASES


def test_cosmos_action_and_observation_contract():
    actions = validate_cosmos_actions(np.zeros((16, 7), dtype=np.float64))
    assert actions.shape == (16, 7)
    assert actions.dtype == np.float32
    with pytest.raises(ValueError):
        validate_cosmos_actions(np.zeros((16, 8)))

    agent = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    wrist = agent + 20
    prepared = prepare_cosmos_libero_observation(
        {
            "agentview_image": agent,
            "robot0_eye_in_hand_image": wrist,
            "robot0_gripper_qpos": np.array([0.1, 0.2]),
            "robot0_eef_pos": np.array([0.3, 0.4, 0.5]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
    )
    np.testing.assert_array_equal(prepared["primary_image"], np.flipud(agent))
    np.testing.assert_array_equal(prepared["wrist_image"], np.flipud(wrist))
    assert prepared["proprio"].shape == (9,)


def test_l1c1_cosmos_wrapper_pins_runtime_and_frozen_scene_gate():
    wrapper = Path("experiments/robot/libero/tasks/run_l1c1_cosmos.sh").read_text(
        encoding="utf-8"
    )
    assert "18a2accadf4e7a3531e56754102af5a24d2316da" in wrapper
    assert "Cosmos-Policy-LIBERO-Predict2-2B" in wrapper
    assert "MODEL_OPEN_LOOP_STEPS=16" in wrapper
    assert "MODEL_FAMILY=cosmos" in wrapper
    assert "SLURM_JOB_ID" in wrapper

    smoke = PHASES[("l1c1", "cosmos_smoke")]
    assert smoke.count_env == "SMOKE_TRIALS"
    assert "SAVE_VIDEO_MODE=capped" in smoke.command
    assert "experiments/robot/libero/tasks/run_l1c1_cosmos.sh" in smoke.command
    assert any(item.endswith("check_l1c1_cascade_gate.py") for item in smoke.local_gate)
    assert len(smoke.inputs) == 5

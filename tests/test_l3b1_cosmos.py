from pathlib import Path

import numpy as np
import pytest

from experiments.robot.cosmos_policy_utils import (
    prepare_cosmos_libero_observation,
    validate_cosmos_actions,
)
from experiments.robot.libero.tasks.physcog_remote_agent import PHASES


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "experiments/robot/libero/tasks/run_cosmos_l3b1.sh"
L3B1_RUNNER = (
    ROOT / "experiments/robot/libero/tasks/run_l3b1_capability_probe.sh"
)


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


def test_cosmos_l3b1_wrapper_pins_runtime_and_native_gates():
    text = WRAPPER.read_text()
    assert "Cosmos-Policy-LIBERO-Predict2-2B" in text
    assert "cb689ec0e3347c13667d70a78a3447388f5c3bb8" in text
    assert "18a2accadf4e7a3531e56754102af5a24d2316da" in text
    assert "MODEL_FAMILY=cosmos" in text
    assert "PREVIEW_MODEL_FAMILY=cosmos" in text
    assert "MODEL_OPEN_LOOP_STEPS=16" in text
    assert "PASS_L3B1_NATIVE_CAPABILITY_SMOKE" in text
    assert "MAX_VIDEOS_PER_OUTCOME=10" in text


def test_l3b1_runner_passes_model_family_to_every_policy_arm():
    text = L3B1_RUNNER.read_text()
    assert 'MODEL_FAMILY="${MODEL_FAMILY:-openvla}"' in text
    assert 'MODEL_OPEN_LOOP_STEPS="${MODEL_OPEN_LOOP_STEPS:-8}"' in text
    assert text.count('"${POLICY_MODEL_ARGS[@]}"') == 3
    assert '--model_family "${PREVIEW_MODEL_FAMILY}"' in text


def test_remote_agent_registers_gated_cosmos_l3b1_pipeline():
    expected = {
        "cosmos_prepare": "NUM_STATES",
        "cosmos_native_cap_smoke": "SMOKE_TRIALS",
        "cosmos_native_cap_formal": "NUM_TRIALS",
        "cosmos_smoke": "SMOKE_TRIALS",
        "cosmos_formal": "NUM_TRIALS",
        "cosmos_summarize": None,
    }
    for phase, count_env in expected.items():
        spec = PHASES[("l3b1", phase)]
        assert spec.count_env == count_env
        assert "experiments/robot/libero/tasks/run_cosmos_l3b1.sh" in spec.command
    for phase in (
        "cosmos_native_cap_smoke",
        "cosmos_native_cap_formal",
        "cosmos_smoke",
        "cosmos_formal",
    ):
        command = PHASES[("l3b1", phase)].command
        assert "SAVE_VIDEO_MODE=all" in command
        assert "MAX_VIOLATION_VIDEOS=10" in command
        assert "MAX_SUCCESS_VIDEOS=10" in command
        assert "MAX_FAILURE_VIDEOS=10" in command

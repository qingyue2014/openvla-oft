from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robot.cosmos_policy_utils import (
    get_cosmos_policy,
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
    assert "experiments.robot.cosmos_policy_server" in wrapper
    assert 'EVALUATOR_PYTHON="${EVALUATOR_PYTHON:-/home/' in wrapper
    assert '"libero": "0.1.0"' in wrapper
    assert '"mujoco": "3.9.0"' in wrapper
    assert '"robosuite": "1.4.1"' in wrapper
    assert 'PYTHONPATH="${LIBERO_ROOT}:${BASE_PYTHONPATH}"' in wrapper
    assert 'test -f "${LIBERO_ROOT}/libero/libero/__init__.py"' in wrapper
    assert "PASS_L1C1_COSMOS_EVALUATOR_RUNTIME" in wrapper

    smoke = PHASES[("l1c1", "cosmos_smoke")]
    assert smoke.count_env == "SMOKE_TRIALS"
    assert "SAVE_VIDEO_MODE=capped" in smoke.command
    assert "experiments/robot/libero/tasks/run_l1c1_cosmos.sh" in smoke.command
    assert any(item.endswith("check_l1c1_cascade_gate.py") for item in smoke.local_gate)
    assert len(smoke.inputs) == 5
    assert "experiments/logs/l1c1_cosmos_server.log" in smoke.artifacts


def test_cosmos_evaluator_does_not_eagerly_import_openvla_stack():
    evaluator = Path("experiments/robot/libero/run_libero_eval.py").read_text(
        encoding="utf-8"
    )
    physcog = Path(
        "experiments/robot/libero/run_physcog_libero_l1_eval.py"
    ).read_text(encoding="utf-8")
    assert "from prismatic" not in evaluator
    assert "from prismatic" not in physcog
    assert "OPENVLA_LIBERO_NUM_ACTIONS_CHUNK = 8" in evaluator


def test_cosmos_client_uses_local_inference_protocol(monkeypatch):
    class FakeConnection:
        def __init__(self):
            self.requests = []
            self.responses = [
                {
                    "ok": True,
                    "result": {"protocol": 1, "model": "Cosmos-Policy"},
                    "error": "",
                },
                {
                    "ok": True,
                    "result": {"actions": np.zeros((16, 7))},
                    "error": "",
                },
                {"ok": True, "result": {"reset": True}, "error": ""},
            ]

        def send(self, request):
            self.requests.append(request)

        def recv(self):
            return self.responses.pop(0)

        def close(self):
            pass

    connection = FakeConnection()
    monkeypatch.setattr(
        "experiments.robot.cosmos_policy_utils.Client",
        lambda address, authkey: connection,
    )
    policy = get_cosmos_policy(
        SimpleNamespace(
            cosmos_host="127.0.0.1",
            cosmos_port=8001,
            cosmos_authkey="test-key",
            cosmos_connect_timeout_s=1,
        )
    )
    actions = policy.infer({"primary_image": np.zeros((2, 2, 3))}, "task")
    policy.reset()

    assert actions.shape == (16, 7)
    assert [request["op"] for request in connection.requests] == [
        "metadata",
        "infer",
        "reset",
    ]
    assert connection.requests[1]["task_label"] == "task"


def test_cosmos_server_only_hosts_inference():
    server = Path("experiments/robot/cosmos_policy_server.py").read_text(
        encoding="utf-8"
    )
    assert "CosmosPolicy(cfg)" in server
    assert "OffScreenRenderEnv" not in server
    assert "PASS_L1C1_COSMOS_SERVER_READY" in server

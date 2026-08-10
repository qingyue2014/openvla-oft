import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments.robot.pi05_utils import (
    get_pi05_action,
    get_pi05_policy,
    normalize_model_family,
    resize_with_pad,
    wait_for_policy_server,
)


class _FakePolicy:
    def __init__(self, actions):
        self.actions = actions
        self.request = None

    def infer(self, request):
        self.request = request
        return {"actions": self.actions}


def _observation():
    return {
        "full_image": np.full((224, 224, 3), 7, dtype=np.uint8),
        "wrist_image": np.full((224, 224, 3), 9, dtype=np.uint8),
        "state": np.arange(8, dtype=np.float64),
    }


def test_pi05_request_matches_official_libero_policy_schema():
    policy = _FakePolicy(np.zeros((10, 7), dtype=np.float64))

    actions = get_pi05_action(policy, _observation(), "pick up the bowl")

    assert len(actions) == 10
    assert actions[0].shape == (7,)
    assert set(policy.request) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert policy.request["observation/image"].dtype == np.uint8
    assert policy.request["observation/state"].shape == (8,)
    assert policy.request["observation/state"].dtype == np.float32
    assert policy.request["prompt"] == "pick up the bowl"


def test_pi05_action_validation_rejects_non_libero_shape():
    policy = _FakePolicy(np.zeros((10, 8), dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \(T, 7\)"):
        get_pi05_action(policy, _observation(), "do something")


def test_pi05_resize_preserves_aspect_ratio_and_pads():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    resized = resize_with_pad(image, 224)
    assert resized.shape == (224, 224, 3)
    assert np.all(resized[0] == 0)
    assert np.all(resized[111] == 255)


@pytest.mark.parametrize("name", ["pi05", "pi0.5", "pi_0.5", "pi-0.5"])
def test_pi05_model_family_aliases(name):
    assert normalize_model_family(name) == "pi05"


def test_pi05_actions_are_already_in_libero_environment_convention():
    evaluator = Path("experiments/robot/libero/run_libero_eval.py").read_text(
        encoding="utf-8"
    )
    assert 'if model_family == "pi05":\n        action = np.asarray(action' in evaluator
    assert "Invalid pi0.5 LIBERO action" in evaluator


def test_pi05_policy_uses_official_websocket_client(monkeypatch):
    calls = {}

    class FakeWebsocketClientPolicy:
        def __init__(self, host, port, api_key):
            calls.update(host=host, port=port, api_key=api_key)

        def get_server_metadata(self):
            return {"model": "pi05_libero"}

    package = types.ModuleType("openpi_client")
    websocket_module = types.ModuleType("openpi_client.websocket_client_policy")
    websocket_module.WebsocketClientPolicy = FakeWebsocketClientPolicy
    package.websocket_client_policy = websocket_module
    monkeypatch.setitem(sys.modules, "openpi_client", package)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", websocket_module)
    cfg = types.SimpleNamespace(
        pi05_host="policy.example",
        pi05_port=9000,
        pi05_api_key="secret",
        pi05_connect_timeout_s=1,
    )
    monkeypatch.setattr(
        "experiments.robot.pi05_utils.wait_for_policy_server",
        lambda *args: None,
    )

    policy = get_pi05_policy(cfg)

    assert isinstance(policy, FakeWebsocketClientPolicy)
    assert calls == {"host": "policy.example", "port": 9000, "api_key": "secret"}


def test_wait_for_policy_server_times_out(monkeypatch):
    def refuse_connection(*args, **kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(
        "experiments.robot.pi05_utils.socket.create_connection",
        refuse_connection,
    )
    monkeypatch.setattr("experiments.robot.pi05_utils.time.sleep", lambda *_: None)

    with pytest.raises(TimeoutError, match="waiting for pi0.5 policy server"):
        wait_for_policy_server("127.0.0.1", 8000, 0.001)


def test_l1c1_pi05_wrapper_uses_job_local_dynamic_port_and_readiness_gate():
    wrapper = Path(
        "experiments/robot/libero/tasks/run_l1c1_pi05.sh"
    ).read_text(encoding="utf-8")
    assert 'sock.bind(("127.0.0.1", 0))' in wrapper
    assert 'kill -0 "${server_pid}"' in wrapper
    assert '"/dev/tcp/127.0.0.1/${PI05_PORT}"' in wrapper
    assert "PASS_L1C1_PI05_SERVER_READY" in wrapper
    assert 'PI05_PORT="${PI05_PORT:-8000}"' not in wrapper


def test_l1c1_formal_reports_are_bound_to_the_model_log_directory():
    runner = Path(
        "experiments/robot/libero/tasks/run_l1c1_task2.sh"
    ).read_text(encoding="utf-8")
    for artifact in (
        "l1c1_safe_reference.csv",
        "l1c1_safe_reference.md",
        "l1c1_bowl_stack_eb_replay.csv",
        "l1c1_bowl_stack_ec_replay.csv",
        "l1c1_attribution.csv",
        "l1c1_attribution.md",
    ):
        assert f'${{LOG_DIR}}/{artifact}' in runner
    assert (
        "grep -q 'BENCHMARK_READY_FOR_ATTRIBUTION' "
        '"${BOWL_STACK_ATTRIBUTION_REPORT}"'
    ) in runner

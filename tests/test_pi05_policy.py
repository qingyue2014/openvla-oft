import sys
import types

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


def test_pi05_request_matches_official_libero_policy_schema() -> None:
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


def test_pi05_action_validation_rejects_non_libero_shape() -> None:
    policy = _FakePolicy(np.zeros((10, 8), dtype=np.float32))

    with pytest.raises(ValueError, match=r"shape \(T, 7\)"):
        get_pi05_action(policy, _observation(), "do something")


def test_pi05_resize_preserves_aspect_ratio_and_pads() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)

    resized = resize_with_pad(image, 224)

    assert resized.shape == (224, 224, 3)
    assert np.all(resized[0] == 0)
    assert np.all(resized[111] == 255)


@pytest.mark.parametrize("name", ["pi05", "pi0.5", "pi_0.5", "pi-0.5"])
def test_pi05_model_family_aliases(name: str) -> None:
    assert normalize_model_family(name) == "pi05"


def test_pi05_policy_uses_official_websocket_client(monkeypatch) -> None:
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
    monkeypatch.setitem(
        sys.modules,
        "openpi_client.websocket_client_policy",
        websocket_module,
    )
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
    assert calls == {
        "host": "policy.example",
        "port": 9000,
        "api_key": "secret",
    }


def test_wait_for_policy_server_times_out(monkeypatch) -> None:
    def refuse_connection(*args, **kwargs):
        raise ConnectionRefusedError

    monkeypatch.setattr(
        "experiments.robot.pi05_utils.socket.create_connection",
        refuse_connection,
    )
    monkeypatch.setattr(
        "experiments.robot.pi05_utils.time.sleep",
        lambda *_: None,
    )

    with pytest.raises(TimeoutError, match="waiting for pi0.5 policy server"):
        wait_for_policy_server("127.0.0.1", 8000, 0.001)

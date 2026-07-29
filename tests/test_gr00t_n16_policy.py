import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.robot.gr00t_n16_utils import (
    GR00T_N16_ACTION_KEYS,
    GR00T_N16_CHECKPOINT_REPO_ID,
    GR00T_N16_CHECKPOINT_REVISION,
    GR00T_N16_DEFAULT_CHECKPOINT,
    GR00T_N16_SOURCE_REVISION,
    Gr00tN16PolicyClient,
    is_gr00t_n16_model_family,
    prepare_gr00t_n16_libero_observation,
    process_gr00t_n16_libero_action,
    validate_gr00t_n16_actions,
)


def _raw_observation():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 10,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.4, 0.5]),
    }


def test_gr00t_n16_aliases_and_checkpoint_identity():
    for alias in ("gr00t", "groot", "gr00t_n16", "gr00t-n1.6"):
        assert is_gr00t_n16_model_family(alias)
    assert GR00T_N16_CHECKPOINT_REPO_ID == "0xAnkitSingh/GR00T-N1.6-LIBERO"
    assert GR00T_N16_CHECKPOINT_REVISION == "d690a226ad06e81736786f56cf879d2ed1dd3f0f"
    assert GR00T_N16_SOURCE_REVISION == "9b37aa1ce69c73c6d165233fa88128283bba4508"
    assert GR00T_N16_DEFAULT_CHECKPOINT == "/project/trllmout/models/GR00T-N1.6-LIBERO"


def test_gr00t_n16_observation_matches_official_libero_wrapper():
    raw = _raw_observation()
    obs = prepare_gr00t_n16_libero_observation(raw)
    np.testing.assert_array_equal(
        obs["video.image"][0, 0], raw["agentview_image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        obs["video.wrist_image"][0, 0],
        raw["robot0_eye_in_hand_image"][::-1, ::-1],
    )
    assert obs["video.image"].shape == (1, 1, 4, 5, 3)
    assert obs["state.gripper"].shape == (1, 1, 2)
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        assert obs[f"state.{key}"].shape == (1, 1, 1)
        assert obs[f"state.{key}"].dtype == np.float32


def test_gr00t_n16_combines_split_action_chunk():
    split = {
        f"action.{key}": np.full((1, 16, 1), index, dtype=np.float32)
        for index, key in enumerate(GR00T_N16_ACTION_KEYS)
    }
    actions = validate_gr00t_n16_actions(split)
    assert actions.shape == (16, 7)
    np.testing.assert_array_equal(actions[0], np.arange(7))


@pytest.mark.parametrize(
    ("raw_gripper", "expected"),
    ((0.0, 1.0), (0.49, 1.0), (0.51, -1.0), (1.0, -1.0)),
)
def test_gr00t_n16_gripper_transform_matches_official_wrapper(
    raw_gripper, expected
):
    action = np.zeros(7, dtype=np.float32)
    action[-1] = raw_gripper
    assert process_gr00t_n16_libero_action(action)[-1] == expected


def test_gr00t_n16_rejects_malformed_actions():
    with pytest.raises(KeyError):
        validate_gr00t_n16_actions({})
    split = {
        f"action.{key}": np.zeros((1, 16, 1), dtype=np.float32)
        for key in GR00T_N16_ACTION_KEYS
    }
    split["action.yaw"] = np.zeros((16, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_gr00t_n16_actions(split)


def test_gr00t_n16_client_matches_official_zmq_msgpack_protocol():
    msgpack = pytest.importorskip("msgpack")
    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    server = context.socket(zmq.REP)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    endpoints = []

    modality_config = {
        "video": {
            "__ModalityConfig_class__": True,
            "as_json": {"modality_keys": ["image", "wrist_image"]},
        },
        "state": {
            "__ModalityConfig_class__": True,
            "as_json": {
                "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            },
        },
        "action": {
            "__ModalityConfig_class__": True,
            "as_json": {
                "modality_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            },
        },
        "language": {
            "__ModalityConfig_class__": True,
            "as_json": {
                "modality_keys": ["annotation.human.action.task_description"]
            },
        },
    }

    def encode(value):
        if isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
        raise TypeError

    def serve():
        for _ in range(4):
            request = msgpack.unpackb(server.recv(), raw=False)
            endpoint = request["endpoint"]
            endpoints.append(endpoint)
            if endpoint == "ping":
                response = {"status": "ok"}
            elif endpoint == "get_modality_config":
                response = modality_config
            elif endpoint == "get_action":
                observation = request["data"]["observation"]
                assert observation[
                    "annotation.human.action.task_description"
                ] == ["put the bowl on the plate"]
                response = [
                    {
                        f"action.{key}": np.zeros((1, 16, 1), dtype=np.float32)
                        for key in GR00T_N16_ACTION_KEYS
                    },
                    {},
                ]
            elif endpoint == "reset":
                response = {}
            server.send(msgpack.packb(response, default=encode))

    thread = threading.Thread(target=serve)
    thread.start()
    cfg = SimpleNamespace(
        gr00t_n16_host="127.0.0.1",
        gr00t_n16_port=port,
        gr00t_n16_connect_timeout_s=2,
        gr00t_n16_request_timeout_s=2,
        gr00t_n16_api_token="",
    )
    client = Gr00tN16PolicyClient(cfg)
    actions = client.infer(
        prepare_gr00t_n16_libero_observation(_raw_observation()),
        "put the bowl on the plate",
    )
    client.reset()
    client.close()
    thread.join(timeout=2)
    server.close()
    context.term()
    assert actions.shape == (16, 7)
    assert endpoints == ["ping", "get_modality_config", "get_action", "reset"]

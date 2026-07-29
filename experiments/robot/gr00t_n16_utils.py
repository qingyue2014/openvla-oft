"""Client adapter for a GR00T N1.6 LIBERO inference server.

The NVIDIA runtime is intentionally kept in its own ``uv`` environment.  This
module implements only the official ZeroMQ/msgpack wire protocol so the
existing OpenVLA/LIBERO evaluator can query it without importing GR00T's
PyTorch and Transformers dependency stack.
"""

from __future__ import annotations

import io
import math
import time
from typing import Any, Mapping

import numpy as np

GR00T_N16_ALIASES = frozenset(
    {
        "gr00t",
        "groot",
        "gr00t_n16",
        "groot_n16",
        "gr00t-n1.6",
        "groot-n1.6",
        "gr00t_n1.6",
        "groot_n1.6",
    }
)
GR00T_N16_MODEL_FAMILY = "gr00t_n16"
GR00T_N16_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
GR00T_N16_STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
GR00T_N16_VIDEO_KEYS = ("image", "wrist_image")
GR00T_N16_LANGUAGE_KEY = "annotation.human.action.task_description"
GR00T_N16_ACTION_DIM = 7
GR00T_N16_CHECKPOINT_REPO_ID = "0xAnkitSingh/GR00T-N1.6-LIBERO"
GR00T_N16_CHECKPOINT_REVISION = "d690a226ad06e81736786f56cf879d2ed1dd3f0f"
GR00T_N16_SOURCE_REVISION = "9b37aa1ce69c73c6d165233fa88128283bba4508"
GR00T_N16_DEFAULT_CHECKPOINT = "/project/trllmout/models/GR00T-N1.6-LIBERO"


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """NVIDIA LIBERO wrapper's robosuite-compatible quaternion conversion."""
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(float(quat[3])) / denominator


def is_gr00t_n16_model_family(model_family: str) -> bool:
    return str(model_family).lower() in GR00T_N16_ALIASES


def normalize_gr00t_n16_model_family(model_family: str) -> str:
    return (
        GR00T_N16_MODEL_FAMILY
        if is_gr00t_n16_model_family(model_family)
        else str(model_family).lower()
    )


def prepare_gr00t_n16_libero_observation(
    obs: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Match NVIDIA's N1.6 ``LiberoEnv._process_observation`` exactly."""
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"Missing LIBERO observation keys for GR00T N1.6: {missing}")

    primary = np.asarray(obs["agentview_image"])[::-1, ::-1].copy()
    wrist = np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1].copy()
    if primary.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise TypeError("GR00T N1.6 LIBERO camera observations must be uint8 RGB")
    if primary.ndim != 3 or primary.shape[-1] != 3:
        raise ValueError(f"Invalid GR00T N1.6 primary image shape: {primary.shape}")
    if wrist.ndim != 3 or wrist.shape[-1] != 3:
        raise ValueError(f"Invalid GR00T N1.6 wrist image shape: {wrist.shape}")

    xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    rpy = np.asarray(
        _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
        dtype=np.float32,
    )
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    if xyz.shape != (3,) or rpy.shape != (3,) or gripper.shape != (2,):
        raise ValueError(
            "GR00T N1.6 expects EEF xyz=(3,), axis-angle=(3,), and gripper=(2,); "
            f"received {xyz.shape}, {rpy.shape}, {gripper.shape}"
        )

    def scalar(value: float) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape(1, 1, 1)

    return {
        "video.image": primary[None, None, ...],
        "video.wrist_image": wrist[None, None, ...],
        "state.x": scalar(xyz[0]),
        "state.y": scalar(xyz[1]),
        "state.z": scalar(xyz[2]),
        "state.roll": scalar(rpy[0]),
        "state.pitch": scalar(rpy[1]),
        "state.yaw": scalar(rpy[2]),
        "state.gripper": gripper.reshape(1, 1, 2),
    }


def validate_gr00t_n16_actions(action: Mapping[str, Any]) -> np.ndarray:
    """Combine the official split action fields into a finite ``(T, 7)`` chunk."""
    missing = [
        f"action.{key}"
        for key in GR00T_N16_ACTION_KEYS
        if f"action.{key}" not in action
    ]
    if missing:
        raise KeyError(f"GR00T N1.6 response is missing action fields: {missing}")

    columns = []
    horizon = None
    for key in GR00T_N16_ACTION_KEYS:
        value = np.asarray(action[f"action.{key}"], dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 1:
            raise ValueError(
                f"GR00T N1.6 action.{key} must have shape (1, T, 1), got {value.shape}"
            )
        if horizon is None:
            horizon = value.shape[1]
        elif value.shape[1] != horizon:
            raise ValueError("GR00T N1.6 action fields have inconsistent horizons")
        columns.append(value[0, :, 0])

    actions = np.stack(columns, axis=-1)
    if actions.shape[0] == 0:
        raise ValueError("GR00T N1.6 returned an empty action chunk")
    if not np.isfinite(actions).all():
        raise ValueError("GR00T N1.6 returned NaN or infinite actions")
    return actions


def process_gr00t_n16_libero_action(action: Any) -> np.ndarray:
    """Apply the official LIBERO wrapper's gripper conversion.

    N1.6 predicts gripper values in ``[0, 1]``. NVIDIA's wrapper maps these to
    ``[-1, 1]``, binarizes, and then flips the sign for robosuite.
    """
    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (GR00T_N16_ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid GR00T N1.6 LIBERO action: shape={result.shape}")
    result[-1] = -np.sign(2.0 * result[-1] - 1.0)
    return result


def _encode_wire_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        output = io.BytesIO()
        np.save(output, value, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    raise TypeError(f"Unsupported GR00T wire value: {type(value)!r}")


def _decode_wire_value(value: Any) -> Any:
    if isinstance(value, dict) and "__ndarray_class__" in value:
        return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
    return value


def _modality_keys(config: Any) -> tuple[str, ...]:
    if isinstance(config, Mapping) and "__ModalityConfig_class__" in config:
        config = config.get("as_json", {})
    if not isinstance(config, Mapping):
        return ()
    return tuple(str(key) for key in config.get("modality_keys", ()))


class Gr00tN16PolicyClient:
    """Small, timeout-safe client for NVIDIA's official ``PolicyServer``."""

    def __init__(self, cfg: Any):
        try:
            import msgpack
            import zmq
        except ImportError as exc:
            raise ImportError(
                "GR00T N1.6 evaluation needs lightweight client packages "
                "`msgpack` and `pyzmq`; run the registered models:setup_gr00t_n16 phase."
            ) from exc

        self._msgpack = msgpack
        self._zmq = zmq
        self.host = str(getattr(cfg, "gr00t_n16_host", "127.0.0.1"))
        self.port = int(getattr(cfg, "gr00t_n16_port", 5555))
        self.timeout_ms = max(
            1, int(float(getattr(cfg, "gr00t_n16_request_timeout_s", 120.0)) * 1000)
        )
        self.connect_timeout_s = float(
            getattr(cfg, "gr00t_n16_connect_timeout_s", 1800.0)
        )
        self.api_token = str(getattr(cfg, "gr00t_n16_api_token", "") or "")
        self.context = zmq.Context()
        self.socket = None
        self._init_socket()
        self._wait_until_ready()
        self._validate_server_contract()
        print(
            "Connected to GR00T N1.6 policy server at "
            f"tcp://{self.host}:{self.port}"
        )

    def _init_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _pack(self, value: Any) -> bytes:
        return self._msgpack.packb(
            value, default=_encode_wire_value, use_bin_type=True
        )

    def _unpack(self, value: bytes) -> Any:
        return self._msgpack.unpackb(
            value, object_hook=_decode_wire_value, raw=False
        )

    def call_endpoint(
        self, endpoint: str, data: Mapping[str, Any] | None = None
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = dict(data)
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(self._pack(request))
            response = self._unpack(self.socket.recv())
        except self._zmq.error.ZMQError:
            self._init_socket()
            raise
        if isinstance(response, Mapping) and "error" in response:
            raise RuntimeError(f"GR00T N1.6 server error: {response['error']}")
        return response

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.connect_timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                response = self.call_endpoint("ping")
                if isinstance(response, Mapping) and response.get("status") == "ok":
                    return
            except self._zmq.error.ZMQError as exc:
                last_error = exc
            time.sleep(1.0)
        raise TimeoutError(
            f"Timed out waiting for GR00T N1.6 at tcp://{self.host}:{self.port}"
        ) from last_error

    def _validate_server_contract(self) -> None:
        config = self.call_endpoint("get_modality_config")
        expected = {
            "video": GR00T_N16_VIDEO_KEYS,
            "state": GR00T_N16_STATE_KEYS,
            "action": GR00T_N16_ACTION_KEYS,
            "language": (GR00T_N16_LANGUAGE_KEY,),
        }
        actual = {
            modality: _modality_keys(config.get(modality))
            for modality in expected
        }
        if actual != expected:
            raise ValueError(
                "The connected checkpoint is not the expected N1.6 LIBERO_PANDA "
                f"model. Expected modalities {expected}, received {actual}."
            )

    def infer(
        self, observation: Mapping[str, Any], task_label: str
    ) -> np.ndarray:
        request = dict(observation)
        request[GR00T_N16_LANGUAGE_KEY] = [str(task_label)]
        response = self.call_endpoint(
            "get_action", {"observation": request, "options": None}
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise ValueError(
                "GR00T N1.6 server must return (action, info), "
                f"received {type(response)!r}"
            )
        return validate_gr00t_n16_actions(response[0])

    def reset(self) -> None:
        self.call_endpoint("reset", {"options": None})

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def get_gr00t_n16_policy(cfg: Any) -> Gr00tN16PolicyClient:
    return Gr00tN16PolicyClient(cfg)

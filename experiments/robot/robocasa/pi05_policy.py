"""Official pi0.5-LIBERO policy adapter for RoboCasa smoke evaluation.

This is deliberately labelled as a cross-simulator, cross-embodiment smoke
adapter.  The released pi0.5-LIBERO checkpoint emits the same 6-DoF
delta-OSC plus gripper action used by the Panda arm in RoboCasa, but it was not
trained for RoboCasa or PandaOmron.  The adapter therefore keeps the mobile
base and torso fixed and must not be presented as a native RoboCasa policy.
"""

from __future__ import annotations

from collections import deque
import os
import socket
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image

AGENT_CAMERA = "robot0_agentview_center"
WRIST_CAMERA = "robot0_eye_in_hand"
PI05_ACTION_DIM = 7
ROBOCASA_ACTION_DIM = 12


def resize_with_pad(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Apply OpenPI's aspect-preserving LIBERO image transform."""

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)

    pil = Image.fromarray(image)
    width, height = pil.size
    ratio = max(width / size, height / size)
    resized_width = int(width / ratio)
    resized_height = int(height / ratio)
    resized = pil.resize((resized_width, resized_height), Image.BILINEAR)
    padded = Image.new(resized.mode, (size, size), 0)
    padded.paste(
        resized,
        ((size - resized_width) // 2, (size - resized_height) // 2),
    )
    return np.asarray(padded)


def wait_for_server(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(
        f"timed out waiting for pi0.5 policy server at ws://{host}:{port}"
    ) from last_error


def _axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert a robosuite xyzw quaternion to a rotation vector."""

    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("zero quaternion")
    quat = quat / norm
    xyz = quat[:3]
    w = float(np.clip(quat[3], -1.0, 1.0))
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (xyz / sin_half * angle).astype(np.float32)


def build_request(obs: Mapping[str, Any], lang: str) -> dict[str, Any]:
    """Build the exact request expected by the released pi05_libero server."""

    center_key = f"{AGENT_CAMERA}_image"
    wrist_key = f"{WRIST_CAMERA}_image"
    required = (
        center_key,
        wrist_key,
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"RoboCasa observation is missing pi0.5 inputs: {missing}")

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _axis_angle(np.asarray(obs["robot0_eef_quat"])),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    if state.shape != (8,):
        raise ValueError(f"expected 8-D pi0.5 state, got {state.shape}")
    return {
        # Robosuite camera observations are vertically flipped.
        "observation/image": resize_with_pad(np.asarray(obs[center_key])[::-1]),
        "observation/wrist_image": resize_with_pad(
            np.asarray(obs[wrist_key])[::-1]
        ),
        "observation/state": state,
        "prompt": str(lang),
    }


def map_libero_action_to_pandaomron(action: np.ndarray, env: Any) -> np.ndarray:
    """Freeze PandaOmron base/torso and map the released 7-D arm action."""

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (PI05_ACTION_DIM,):
        raise ValueError(f"expected pi0.5 action shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("pi0.5 returned non-finite actions")

    low, high = env.action_spec
    if np.asarray(low).shape != (ROBOCASA_ACTION_DIM,):
        raise ValueError(
            "pi0.5 RoboCasa smoke requires the 12-D PandaOmron action "
            f"interface, got {np.asarray(low).shape}"
        )
    split = dict(env.robots[0].composite_controller._action_split_indexes)
    expected = {
        "right": (0, 6),
        "right_gripper": (6, 7),
        "base": (7, 10),
        "torso": (10, 11),
    }
    if split != expected:
        raise ValueError(f"unexpected PandaOmron action split: {split}")

    mapped = np.zeros(ROBOCASA_ACTION_DIM, dtype=np.float32)
    mapped[:PI05_ACTION_DIM] = action
    mapped[11] = -1.0  # HybridMobileBase arm-control mode.
    return np.clip(mapped, np.asarray(low), np.asarray(high))


class Pi05RoboCasaPolicy:
    """Chunked websocket client for the released ``pi05_libero`` checkpoint."""

    requires_camera_obs = True
    camera_names = (AGENT_CAMERA, WRIST_CAMERA)
    model_label = "pi05_libero_cross_sim"

    def __init__(self) -> None:
        self.host = os.environ.get("PI05_HOST", "127.0.0.1")
        self.port = int(os.environ.get("PI05_PORT", "8000"))
        self.replan_steps = int(os.environ.get("PI05_REPLAN_STEPS", "5"))
        timeout_s = float(os.environ.get("PI05_CONNECT_TIMEOUT_S", "900"))
        if self.replan_steps < 1:
            raise ValueError("PI05_REPLAN_STEPS must be positive")
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise ImportError(
                "pi0.5 RoboCasa smoke requires the official openpi-client"
            ) from exc
        wait_for_server(self.host, self.port, timeout_s)
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=self.host,
            port=self.port,
            api_key=os.environ.get("PI05_API_KEY") or None,
        )
        self.metadata = self.client.get_server_metadata()
        self._queue: deque[np.ndarray] = deque()
        print(
            "Connected to pi0.5 for cross-simulator RoboCasa smoke; "
            f"server=ws://{self.host}:{self.port} metadata={self.metadata}"
        )

    def reset(self) -> None:
        self._queue.clear()

    def __call__(self, obs: Mapping[str, Any], lang: str, env: Any) -> np.ndarray:
        if not self._queue:
            response = self.client.infer(build_request(obs, lang))
            if "actions" not in response:
                raise KeyError(
                    "pi0.5 response has no actions field: "
                    f"{sorted(response)}"
                )
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2 or actions.shape[1] != PI05_ACTION_DIM:
                raise ValueError(
                    "expected pi0.5 action chunk with shape (T, 7), "
                    f"got {actions.shape}"
                )
            if not len(actions):
                raise ValueError("pi0.5 returned an empty action chunk")
            self._queue.extend(actions[: self.replan_steps])
        return map_libero_action_to_pandaomron(self._queue.popleft(), env)


def make_policy() -> Pi05RoboCasaPolicy:
    return Pi05RoboCasaPolicy()

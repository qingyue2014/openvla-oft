"""Official pi0.5-LIBERO policy adapter for RoboCasa smoke evaluation.

This is deliberately labelled as a cross-simulator, cross-embodiment smoke
adapter.  The released pi0.5-LIBERO checkpoint emits the same 6-DoF
delta-OSC plus gripper action used by the Panda arm in RoboCasa, but it was not
trained for RoboCasa or PandaOmron.  The adapter therefore keeps the mobile
base and torso fixed and must not be presented as a native RoboCasa policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
# Mean first-policy pose after the official ten-step wait, measured over the
# 20 native LIBERO task-8 trajectories in the validated pi0.5 capability run.
# This is an initial-pose anchor, not the all-timestep dataset mean.
LIBERO_INITIAL_EEF_POS = np.array(
    [-0.20640835, 0.000760356, 1.1756006],
    dtype=np.float32,
)
LIBERO_INITIAL_EEF_QUAT = np.array(
    [0.9996163, -0.000711894, -0.027665326, -0.000039881],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CanonicalStateAnchor:
    """Initial pose used to preserve world-frame proprioceptive deltas."""

    world_position: np.ndarray
    world_orientation: np.ndarray
    canonical_initial_orientation: np.ndarray


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


def preprocess_camera_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the released OpenPI LIBERO evaluation image transform exactly."""

    # OpenPI examples/libero/main.py rotates each robosuite observation by
    # 180 degrees before resize_with_pad. A vertical flip alone mirrors the
    # scene horizontally relative to pi0.5's training distribution.
    return resize_with_pad(np.asarray(image)[::-1, ::-1], size=size)


def preprocess_camera_image_for_mode(
    image: np.ndarray,
    *,
    mode: str,
    size: int = 224,
) -> np.ndarray:
    """Preprocess a RoboCasa image under an explicit camera convention."""

    if mode == "rotate180":
        return preprocess_camera_image(image, size=size)
    if mode == "vertical":
        return resize_with_pad(np.asarray(image)[::-1], size=size)
    raise ValueError(
        "PI05_IMAGE_MODE must be 'rotate180' or 'vertical', "
        f"got {mode!r}"
    )


def pi05_preprocessing_label(mode: str) -> str:
    if mode == "rotate180":
        return "pi05_libero_rotate180_resize_with_pad_224"
    if mode == "vertical":
        return "pi05_robocasa_vertical_resize_with_pad_224"
    raise ValueError(
        "PI05_IMAGE_MODE must be 'rotate180' or 'vertical', "
        f"got {mode!r}"
    )


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
    """Match OpenPI's released LIBERO xyzw quaternion conversion."""

    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("zero quaternion")
    quat = quat / norm
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(w)) / denominator).astype(np.float32)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized xyzw quaternion."""

    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quat / np.linalg.norm(quat)


def canonicalize_robocasa_state(
    obs: Mapping[str, Any],
    env: Any,
    *,
    state_anchor: CanonicalStateAnchor | None,
) -> tuple[np.ndarray, CanonicalStateAnchor]:
    """Align the initial pose to LIBERO while preserving world-frame deltas.

    LIBERO's robosuite 1.4.1 state and delta actions share world-coordinate
    axes. PandaOmron's absolute world position is far outside the LIBERO state
    distribution, so the initial pose is translated to the measured LIBERO
    first-policy pose. Subsequent position and orientation changes remain in
    world coordinates;
    rotating those changes into the PandaOmron base would make proprioception
    disagree with the world-frame action emitted by the checkpoint.
    """

    world_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    world_ori = _quat_to_mat(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    if state_anchor is None:
        state_anchor = CanonicalStateAnchor(
            world_position=world_pos.copy(),
            world_orientation=world_ori.copy(),
            canonical_initial_orientation=_quat_to_mat(
                LIBERO_INITIAL_EEF_QUAT
            ),
        )
    canonical_pos = LIBERO_INITIAL_EEF_POS + (
        world_pos - state_anchor.world_position
    )
    world_orientation_delta = world_ori @ state_anchor.world_orientation.T
    canonical_ori = (
        world_orientation_delta @ state_anchor.canonical_initial_orientation
    )
    canonical_quat = _mat_to_quat(canonical_ori)
    # LIBERO's downward-facing initial pose is represented by a positive
    # x-axis rotation near +pi. Select the equivalent quaternion branch that
    # preserves that representation instead of jumping to approximately -pi.
    if canonical_quat[0] < 0.0:
        canonical_quat = -canonical_quat

    state = np.concatenate(
        (
            canonical_pos.astype(np.float32),
            _axis_angle(canonical_quat),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    if state.shape != (8,):
        raise ValueError(f"expected 8-D canonical pi0.5 state, got {state.shape}")
    return state, state_anchor


def build_request(
    obs: Mapping[str, Any],
    lang: str,
    *,
    state: np.ndarray | None = None,
) -> dict[str, Any]:
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

    if state is None:
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _axis_angle(np.asarray(obs["robot0_eef_quat"])),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"expected 8-D pi0.5 state, got {state.shape}")
    return {
        "observation/image": preprocess_camera_image(obs[center_key]),
        "observation/wrist_image": preprocess_camera_image(obs[wrist_key]),
        "observation/state": state,
        "prompt": str(lang),
    }


def map_libero_action_to_pandaomron(action: np.ndarray, env: Any) -> np.ndarray:
    """Freeze the mobile body and map LIBERO's world-frame 7-D arm action.

    The robosuite 1.4.1 OSC used to collect and evaluate LIBERO applies its
    delta pose directly in world coordinates. Current RoboCasa's PandaOmron
    OSC instead expects the delta in ``arm.origin_ori``'s base frame. The
    PandaOmron arm base is rotated by approximately +90 degrees around world z,
    so copying the six pose components verbatim sends the hand along the wrong
    axes.
    """

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
        "torso": (6, 7),
        "base": (7, 10),
        "right_gripper": (10, 11),
    }
    if split != expected:
        raise ValueError(f"unexpected PandaOmron action split: {split}")

    arm = env.robots[0].composite_controller.part_controllers["right"]
    origin_ori = np.asarray(arm.origin_ori, dtype=np.float64).reshape(3, 3)
    world_to_controller = origin_ori.T

    mapped = np.zeros(ROBOCASA_ACTION_DIM, dtype=np.float32)
    mapped[:3] = world_to_controller @ action[:3]
    # A rotation vector transforms between coordinate frames in the same way
    # as a translation vector (R.T @ rotvec).
    mapped[3:6] = world_to_controller @ action[3:6]
    mapped[10] = action[6]
    mapped[11] = -1.0  # HybridMobileBase arm-control mode.
    return np.clip(mapped, np.asarray(low), np.asarray(high))


class Pi05RoboCasaPolicy:
    """Chunked websocket client for the released ``pi05_libero`` checkpoint."""

    requires_camera_obs = True
    camera_names = (AGENT_CAMERA, WRIST_CAMERA)
    model_label = (
        "pi05_libero_cross_sim_initial_pose_world_delta_to_panda_base"
    )
    # Match examples/libero/main.py: objects settle for ten simulator steps
    # under LIBERO_DUMMY_ACTION before the first policy request.
    settle_steps = 10

    def __init__(self) -> None:
        self.host = os.environ.get("PI05_HOST", "127.0.0.1")
        self.port = int(os.environ.get("PI05_PORT", "8000"))
        self.image_mode = os.environ.get("PI05_IMAGE_MODE", "rotate180")
        self.policy_preprocessing = pi05_preprocessing_label(self.image_mode)
        self.model_label = (
            "pi05_libero_cross_sim_initial_pose_world_delta_to_panda_base"
            f"_image_{self.image_mode}"
        )
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
        self._state_anchor: CanonicalStateAnchor | None = None
        print(
            "Connected to pi0.5 for cross-simulator RoboCasa smoke; "
            f"server=ws://{self.host}:{self.port} metadata={self.metadata}"
        )

    def reset(self) -> None:
        self._queue.clear()
        self._state_anchor = None

    @staticmethod
    def settle_action(env: Any) -> np.ndarray:
        return map_libero_action_to_pandaomron(
            np.array([0.0] * 6 + [-1.0], dtype=np.float32),
            env,
        )

    def policy_view_image(self, obs: Mapping[str, Any]) -> np.ndarray:
        """Return the exact center-camera pixels consumed by this policy."""

        return preprocess_camera_image_for_mode(
            obs[f"{AGENT_CAMERA}_image"],
            mode=self.image_mode,
        )

    def __call__(self, obs: Mapping[str, Any], lang: str, env: Any) -> np.ndarray:
        if not self._queue:
            state, self._state_anchor = canonicalize_robocasa_state(
                obs,
                env,
                state_anchor=self._state_anchor,
            )
            request = build_request(obs, lang, state=state)
            if self.image_mode != "rotate180":
                request["observation/image"] = preprocess_camera_image_for_mode(
                    obs[f"{AGENT_CAMERA}_image"],
                    mode=self.image_mode,
                )
                request["observation/wrist_image"] = (
                    preprocess_camera_image_for_mode(
                        obs[f"{WRIST_CAMERA}_image"],
                        mode=self.image_mode,
                    )
                )
            response = self.client.infer(request)
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

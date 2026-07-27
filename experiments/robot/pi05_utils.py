"""Utilities for evaluating the official OpenPI pi0.5 LIBERO policy.

The pi0.5 model runs in the OpenPI environment and is queried through the
official ``openpi-client`` websocket protocol.  Keeping inference in a
separate process avoids dependency conflicts between OpenPI and this
repository's OpenVLA-specific Transformers fork.
"""

import socket
import time
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image


PI05_ACTION_DIM = 7
PI05_IMAGE_SIZE = 224
PI05_MODEL_FAMILY_ALIASES = {"pi05", "pi0.5", "pi_0.5", "pi-0.5"}


def normalize_model_family(model_family: str) -> str:
    """Normalize common pi0.5 spellings to the stable ``pi05`` run label."""
    normalized = str(model_family).lower()
    return "pi05" if normalized in PI05_MODEL_FAMILY_ALIASES else normalized


def resize_with_pad(
    image: np.ndarray,
    size: Union[int, tuple],
    method: int = Image.BILINEAR,
) -> np.ndarray:
    """Match OpenPI's LIBERO ``resize_with_pad`` preprocessing."""
    if isinstance(size, int):
        height, width = size, size
    else:
        height, width = size

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")

    pil_image = Image.fromarray(image)
    current_width, current_height = pil_image.size
    if (current_height, current_width) == (height, width):
        return image

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=method)

    padded = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    padded.paste(resized, (pad_width, pad_height))
    return np.asarray(padded)


def wait_for_policy_server(host: str, port: int, timeout_s: float) -> None:
    """Wait until the OpenPI TCP endpoint is reachable, with a finite timeout."""
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=min(2.0, timeout_s)):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"Timed out after {timeout_s:g}s waiting for pi0.5 policy server "
        f"at ws://{host}:{port}"
    ) from last_error


def get_pi05_policy(cfg: Any) -> Any:
    """Connect to an OpenPI policy server hosting ``pi05_libero``."""
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise ImportError(
            "pi0.5 evaluation requires the official openpi-client package. "
            "Install this project with `pip install -e '.[pi05]'`, then start "
            "the OpenPI LIBERO policy server before launching evaluation."
        ) from exc

    wait_for_policy_server(
        cfg.pi05_host,
        cfg.pi05_port,
        cfg.pi05_connect_timeout_s,
    )
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=cfg.pi05_host,
        port=cfg.pi05_port,
        api_key=cfg.pi05_api_key or None,
    )
    metadata = policy.get_server_metadata()
    print(
        "Connected to pi0.5 policy server "
        f"at ws://{cfg.pi05_host}:{cfg.pi05_port}; metadata={metadata}"
    )
    return policy


def get_pi05_action(
    policy: Any,
    obs: Dict[str, Any],
    task_label: str,
) -> List[np.ndarray]:
    """Query the OpenPI server and validate its LIBERO action chunk."""
    request = {
        "observation/image": np.asarray(obs["full_image"], dtype=np.uint8),
        "observation/wrist_image": np.asarray(obs["wrist_image"], dtype=np.uint8),
        "observation/state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": str(task_label),
    }
    response = policy.infer(request)
    if "actions" not in response:
        raise KeyError(f"pi0.5 server response has no 'actions' field: {sorted(response)}")

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != PI05_ACTION_DIM:
        raise ValueError(
            "Expected pi0.5 LIBERO actions with shape (T, 7), "
            f"got {actions.shape}"
        )
    if len(actions) == 0:
        raise ValueError("pi0.5 server returned an empty action chunk")
    if not np.isfinite(actions).all():
        raise ValueError("pi0.5 server returned NaN or infinite action values")

    return [actions[index].copy() for index in range(len(actions))]

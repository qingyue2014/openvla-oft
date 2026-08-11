"""Adapter for NVIDIA Cosmos Policy's official LIBERO checkpoint.

The heavy ``cosmos_policy`` dependency is imported lazily so importing the
OpenVLA-OFT evaluation package does not require the separate Cosmos runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import pickle
import socket
import struct
import time
from typing import Any, Mapping

import numpy as np


COSMOS_MODEL_ALIASES = frozenset({"cosmos", "cosmos_policy", "cosmos-policy"})
COSMOS_LIBERO_REPO_ID = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
COSMOS_DEFAULT_CHECKPOINT = Path("/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B")
COSMOS_ACTION_DIM = 7
COSMOS_CHUNK_SIZE = 16
COSMOS_CONFIG_MODULE_PATH = "cosmos_policy/config/config.py"
COSMOS_CHECKPOINT_FILENAME = "Cosmos-Policy-LIBERO-Predict2-2B.pt"
COSMOS_TOKENIZER_REPO_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS_TOKENIZER_REVISION = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
COSMOS_DEFAULT_TOKENIZER = Path(
    "/project/trllmout/models/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"
)
_MESSAGE_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_NDARRAY_WIRE_MARKER = "__physcog_ndarray_v1__"


def is_cosmos_model_family(model_family: str) -> bool:
    return model_family.lower() in COSMOS_MODEL_ALIASES


def resolve_cosmos_package_root(cosmos_policy_module: Any) -> Path:
    """Resolve both regular and PEP 420 namespace-package installations."""
    module_file = getattr(cosmos_policy_module, "__file__", None)
    if module_file:
        candidates = (Path(module_file).resolve().parent,)
    else:
        candidates = tuple(
            Path(location).resolve()
            for location in getattr(cosmos_policy_module, "__path__", ())
        )

    for candidate in candidates:
        if (candidate / "config" / "config.py").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate Cosmos Policy's config/config.py from its installed "
        f"package paths: {[str(path) for path in candidates]}"
    )


@contextmanager
def defer_unused_cosmos_base_checkpoint_downloads(checkpoint_db: Any):
    """Avoid eager downloads for config defaults replaced by our local checkpoint.

    Cosmos Policy scans every experiment while constructing its config. The
    LIBERO experiment eagerly resolves its *training* base-checkpoint URI even
    though ``load_model_from_checkpoint`` replaces that value with
    ``ckpt_path`` before model validation and loading. Keep the URI unresolved
    during config construction so evaluation needs only the complete official
    LIBERO inference checkpoint selected by the caller.
    """
    original_get_checkpoint_by_hf = checkpoint_db.get_checkpoint_by_hf
    get_checkpoint_path = checkpoint_db.get_checkpoint_path
    cache_clear = getattr(get_checkpoint_path, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    checkpoint_db.get_checkpoint_by_hf = lambda uri: uri
    try:
        yield
    finally:
        checkpoint_db.get_checkpoint_by_hf = original_get_checkpoint_by_hf
        if cache_clear is not None:
            cache_clear()


@contextmanager
def use_local_cosmos_tokenizer(checkpoint_utils: Any, tokenizer_path: Path):
    """Resolve the pinned Wan VAE tokenizer without network access."""
    original_hf_hub_download = checkpoint_utils.hf_hub_download

    def resolve_pinned_tokenizer(*, repo_id: str, filename: str, **kwargs):
        if repo_id == COSMOS_TOKENIZER_REPO_ID and filename == "tokenizer/tokenizer.pth":
            return str(tokenizer_path)
        return original_hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)

    checkpoint_utils.hf_hub_download = resolve_pinned_tokenizer
    try:
        yield
    finally:
        checkpoint_utils.hf_hub_download = original_hf_hub_download


def validate_cosmos_actions(actions: Any) -> np.ndarray:
    """Return a finite ``(T, 7)`` action chunk or raise a useful error."""
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim == 1:
        action_array = action_array[None, :]
    if action_array.ndim != 2 or action_array.shape[1] != COSMOS_ACTION_DIM:
        raise ValueError(
            "Cosmos Policy must return a (T, 7) LIBERO delta-action chunk; " f"received shape {action_array.shape}."
        )
    if action_array.shape[0] == 0:
        raise ValueError("Cosmos Policy returned an empty action chunk.")
    if not np.isfinite(action_array).all():
        raise ValueError("Cosmos Policy returned non-finite actions.")
    return action_array


def prepare_cosmos_libero_observation(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Match the official Cosmos Policy LIBERO observation transformation."""
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_gripper_qpos",
        "robot0_eef_pos",
        "robot0_eef_quat",
    )
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"Missing LIBERO observation keys for Cosmos Policy: {missing}")
    return {
        "primary_image": np.flipud(np.asarray(obs["agentview_image"])),
        "wrist_image": np.flipud(np.asarray(obs["robot0_eye_in_hand_image"])),
        "proprio": np.concatenate(
            (
                np.asarray(obs["robot0_gripper_qpos"]),
                np.asarray(obs["robot0_eef_pos"]),
                np.asarray(obs["robot0_eef_quat"]),
            )
        ),
    }


class CosmosPolicy:
    """Thin wrapper around the official ``nvlabs/cosmos-policy`` inference API."""

    def __init__(self, cfg: Any):
        try:
            import cosmos_policy
            from cosmos_policy._src.imaginaire.utils import checkpoint_db
            from cosmos_policy.experiments.robot.cosmos_utils import (
                get_action,
                get_model,
                init_t5_text_embeddings_cache,
                load_dataset_stats,
            )
            from cosmos_policy.utils import checkpoint_utils
            from cosmos_policy.experiments.robot.libero.run_libero_eval import (
                PolicyEvalConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "Cosmos Policy evaluation requires the official "
                "https://github.com/nvlabs/cosmos-policy runtime. Install it "
                "in its documented CUDA environment before selecting "
                "--model_family cosmos."
            ) from exc

        checkpoint = Path(str(getattr(cfg, "pretrained_checkpoint", "") or COSMOS_DEFAULT_CHECKPOINT)).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Cosmos Policy checkpoint directory not found: {checkpoint}. "
                "Run the registered Superpod phase models:setup_cosmos."
            )

        required_files = (
            COSMOS_CHECKPOINT_FILENAME,
            "config.json",
            "libero_dataset_statistics.json",
            "libero_t5_embeddings.pkl",
        )
        missing = [name for name in required_files if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete Cosmos Policy checkpoint at {checkpoint}; missing: {missing}")
        tokenizer_path = Path(
            str(getattr(cfg, "cosmos_tokenizer_path", "") or COSMOS_DEFAULT_TOKENIZER)
        ).expanduser()
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Cosmos Wan VAE tokenizer not found: {tokenizer_path}. "
                "Accept access to nvidia/Cosmos-Predict2-2B-Video2World, then "
                "rerun the registered Superpod phase models:setup_cosmos."
            )

        # Resolve the namespace package to validate this exact source install,
        # but pass the module-style path expected by Cosmos' config loader.
        resolve_cosmos_package_root(cosmos_policy)
        cosmos_cfg = PolicyEvalConfig(
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=str(checkpoint / COSMOS_CHECKPOINT_FILENAME),
            config_file=COSMOS_CONFIG_MODULE_PATH,
            dataset_stats_path=str(checkpoint / "libero_dataset_statistics.json"),
            t5_text_embeddings_path=str(checkpoint / "libero_t5_embeddings.pkl"),
            use_wrist_image=True,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            chunk_size=COSMOS_CHUNK_SIZE,
            num_open_loop_steps=int(getattr(cfg, "num_open_loop_steps", COSMOS_CHUNK_SIZE)),
            trained_with_image_aug=True,
            use_jpeg_compression=True,
            # The shared evaluator supplies the official vertically flipped
            # images, so no second flip belongs inside Cosmos preprocessing.
            flip_images=False,
            num_denoising_steps_action=int(getattr(cfg, "cosmos_num_denoising_steps", 5)),
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
            seed=int(getattr(cfg, "seed", 7)),
            task_suite_name=str(getattr(cfg, "task_suite_name", "libero_spatial")),
        )

        init_t5_text_embeddings_cache(cosmos_cfg.t5_text_embeddings_path)
        dataset_stats = load_dataset_stats(cosmos_cfg.dataset_stats_path)
        with (
            defer_unused_cosmos_base_checkpoint_downloads(checkpoint_db),
            use_local_cosmos_tokenizer(checkpoint_utils, tokenizer_path),
        ):
            model, train_cfg = get_model(cosmos_cfg)
        train_chunk_size = train_cfg.dataloader_train.dataset.chunk_size
        if train_chunk_size != COSMOS_CHUNK_SIZE:
            raise ValueError(
                "Unexpected Cosmos checkpoint action chunk size: " f"{train_chunk_size} (expected {COSMOS_CHUNK_SIZE})."
            )

        self.cfg = cosmos_cfg
        self.model = model
        self.dataset_stats = dataset_stats
        self._get_action = get_action

    def infer(self, observation: Mapping[str, Any], task_label: str) -> np.ndarray:
        result = self._get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            dict(observation),
            task_label,
            seed=self.cfg.seed,
            randomize_seed=False,
            num_denoising_steps_action=self.cfg.num_denoising_steps_action,
            # Avoid VAE-decoding future images/values that PhysCog does not use.
            generate_future_state_and_value_in_parallel=False,
        )
        if not isinstance(result, Mapping) or "actions" not in result:
            raise ValueError("Official Cosmos Policy inference did not return 'actions'.")
        return validate_cosmos_actions(result["actions"])

    def reset(self) -> None:
        """Cosmos Policy is stateless across action-chunk queries."""


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _encode_wire_value(value: Any) -> Any:
    """Convert NumPy values to a version-neutral pickle representation.

    NumPy 2.x pickles refer to ``numpy._core`` modules that NumPy 1.x cannot
    import.  The Cosmos server and frozen LIBERO evaluator intentionally use
    those different runtimes, so only built-in Python containers and raw
    ndarray bytes may cross this boundary.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object-dtype arrays are not allowed on the Cosmos wire protocol")
        contiguous = np.ascontiguousarray(value)
        return {
            _NDARRAY_WIRE_MARKER: True,
            "dtype": contiguous.dtype.str,
            "shape": tuple(int(dimension) for dimension in contiguous.shape),
            "data": contiguous.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _encode_wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_encode_wire_value(item) for item in value)
    if isinstance(value, list):
        return [_encode_wire_value(item) for item in value]
    return value


def _decode_wire_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get(_NDARRAY_WIRE_MARKER) is True:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise TypeError("Object-dtype arrays are not allowed on the Cosmos wire protocol")
            shape = tuple(int(dimension) for dimension in value["shape"])
            expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            data = value["data"]
            if not isinstance(data, bytes) or len(data) != expected_size:
                raise ValueError("Malformed ndarray payload on the Cosmos wire protocol")
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {key: _decode_wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_decode_wire_value(item) for item in value)
    if isinstance(value, list):
        return [_decode_wire_value(item) for item in value]
    return value


def _recv_message(sock: socket.socket) -> Any | None:
    header = _recv_exact(sock, _MESSAGE_HEADER.size)
    if header is None:
        return None
    (size,) = _MESSAGE_HEADER.unpack(header)
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Cosmos policy message is too large: {size} bytes")
    payload = _recv_exact(sock, size)
    if payload is None:
        raise ConnectionError("Cosmos policy connection closed mid-message")
    return _decode_wire_value(pickle.loads(payload))


def _send_message(sock: socket.socket, value: Any) -> None:
    payload = pickle.dumps(_encode_wire_value(value), protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Cosmos policy message is too large: {len(payload)} bytes")
    sock.sendall(_MESSAGE_HEADER.pack(len(payload)) + payload)


class CosmosPolicyClient:
    """Localhost client keeping Cosmos dependencies out of the simulator process."""

    def __init__(self, cfg: Any):
        self.host = str(getattr(cfg, "cosmos_host", "127.0.0.1"))
        self.port = int(getattr(cfg, "cosmos_port", 0))
        self.connect_timeout_s = float(
            getattr(cfg, "cosmos_connect_timeout_s", 900.0)
        )
        if self.port <= 0:
            raise ValueError("cosmos_port must be positive for remote inference")
        if self.connect_timeout_s <= 0:
            raise ValueError("cosmos_connect_timeout_s must be positive")
        self._request({"op": "ping"})

    def _request(self, request: Mapping[str, Any]) -> Any:
        deadline = time.monotonic() + self.connect_timeout_s
        while True:
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=min(5.0, self.connect_timeout_s)
                )
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out connecting to Cosmos policy server at "
                        f"{self.host}:{self.port}"
                    )
                time.sleep(1.0)
        with sock:
            sock.settimeout(self.connect_timeout_s)
            _send_message(sock, dict(request))
            response = _recv_message(sock)
        if not isinstance(response, Mapping):
            raise RuntimeError("Cosmos policy server returned an invalid response")
        if not response.get("ok"):
            raise RuntimeError(
                f"Cosmos policy server error: {response.get('error', 'unknown error')}"
            )
        return response.get("result")

    def infer(self, observation: Mapping[str, Any], task_label: str) -> np.ndarray:
        result = self._request(
            {
                "op": "infer",
                "observation": dict(observation),
                "task_label": str(task_label),
            }
        )
        return validate_cosmos_actions(result)

    def reset(self) -> None:
        """Cosmos Policy is stateless across action-chunk queries."""


def serve_cosmos_policy(policy: CosmosPolicy, host: str, port: int) -> None:
    """Serve one official Cosmos policy on a loopback-only TCP endpoint."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(8)
        print(f"COSMOS_POLICY_SERVER_READY host={host} port={port}", flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                while True:
                    try:
                        request = _recv_message(connection)
                    except Exception as exc:
                        try:
                            _send_message(
                                connection,
                                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                            )
                        except Exception:
                            pass
                        break
                    if request is None:
                        break
                    try:
                        operation = request.get("op")
                        if operation == "ping":
                            result = "pong"
                        elif operation == "infer":
                            result = policy.infer(
                                request["observation"], request["task_label"]
                            )
                        else:
                            raise ValueError(f"Unsupported Cosmos server operation: {operation}")
                        response = {"ok": True, "result": result}
                    except Exception as exc:
                        response = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    _send_message(connection, response)


def get_cosmos_policy(cfg: Any) -> CosmosPolicy | CosmosPolicyClient:
    if int(getattr(cfg, "cosmos_port", 0)) > 0:
        return CosmosPolicyClient(cfg)
    return CosmosPolicy(cfg)

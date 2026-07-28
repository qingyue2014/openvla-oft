"""Adapter for NVIDIA Cosmos Policy's official LIBERO checkpoint.

The heavy ``cosmos_policy`` dependency is imported lazily so importing the
OpenVLA-OFT evaluation package does not require the separate Cosmos runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np


COSMOS_MODEL_ALIASES = frozenset({"cosmos", "cosmos_policy", "cosmos-policy"})
COSMOS_DEFAULT_CHECKPOINT = Path(
    "/project/trllmout/models/Cosmos-Policy-LIBERO-Predict2-2B"
)
COSMOS_ACTION_DIM = 7
COSMOS_CHUNK_SIZE = 16
COSMOS_CONFIG_MODULE_PATH = "cosmos_policy/config/config.py"
COSMOS_CHECKPOINT_FILENAME = "Cosmos-Policy-LIBERO-Predict2-2B.pt"
COSMOS_TOKENIZER_REPO_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS_DEFAULT_TOKENIZER = Path(
    "/project/trllmout/models/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"
)


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
        "Could not locate Cosmos Policy config/config.py from package paths: "
        f"{[str(path) for path in candidates]}"
    )


@contextmanager
def defer_unused_cosmos_base_checkpoint_downloads(checkpoint_db: Any):
    """Avoid eager downloads for config defaults replaced by our checkpoint."""
    original = checkpoint_db.get_checkpoint_by_hf
    get_path = checkpoint_db.get_checkpoint_path
    cache_clear = getattr(get_path, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    checkpoint_db.get_checkpoint_by_hf = lambda uri: uri
    try:
        yield
    finally:
        checkpoint_db.get_checkpoint_by_hf = original
        if cache_clear is not None:
            cache_clear()


@contextmanager
def use_local_cosmos_tokenizer(checkpoint_utils: Any, tokenizer_path: Path):
    """Resolve the pinned Wan VAE tokenizer without network access."""
    original = checkpoint_utils.hf_hub_download

    def resolve_pinned_tokenizer(*, repo_id: str, filename: str, **kwargs):
        if (
            repo_id == COSMOS_TOKENIZER_REPO_ID
            and filename == "tokenizer/tokenizer.pth"
        ):
            return str(tokenizer_path)
        return original(repo_id=repo_id, filename=filename, **kwargs)

    checkpoint_utils.hf_hub_download = resolve_pinned_tokenizer
    try:
        yield
    finally:
        checkpoint_utils.hf_hub_download = original


def validate_cosmos_actions(actions: Any) -> np.ndarray:
    """Return a finite ``(T, 7)`` action chunk or raise a useful error."""
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim == 1:
        action_array = action_array[None, :]
    if action_array.ndim != 2 or action_array.shape[1] != COSMOS_ACTION_DIM:
        raise ValueError(
            "Cosmos Policy must return a (T, 7) LIBERO action chunk; "
            f"received shape {action_array.shape}."
        )
    if action_array.shape[0] == 0:
        raise ValueError("Cosmos Policy returned an empty action chunk.")
    if not np.isfinite(action_array).all():
        raise ValueError("Cosmos Policy returned non-finite actions.")
    return action_array


def prepare_cosmos_libero_observation(
    obs: Mapping[str, Any],
) -> dict[str, np.ndarray]:
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
        raise KeyError(f"Missing LIBERO observation keys for Cosmos: {missing}")
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
    """Thin wrapper around the official ``nvlabs/cosmos-policy`` API."""

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
            from cosmos_policy.experiments.robot.libero.run_libero_eval import (
                PolicyEvalConfig,
            )
            from cosmos_policy.utils import checkpoint_utils
        except ImportError as exc:
            raise ImportError(
                "Cosmos evaluation requires the official cosmos-policy runtime."
            ) from exc

        checkpoint = Path(
            str(
                getattr(cfg, "pretrained_checkpoint", "")
                or COSMOS_DEFAULT_CHECKPOINT
            )
        ).expanduser()
        required_files = (
            COSMOS_CHECKPOINT_FILENAME,
            "config.json",
            "libero_dataset_statistics.json",
            "libero_t5_embeddings.pkl",
        )
        missing = [
            name for name in required_files if not (checkpoint / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete Cosmos checkpoint at {checkpoint}; missing: {missing}"
            )

        tokenizer_path = Path(
            str(
                getattr(cfg, "cosmos_tokenizer_path", "")
                or COSMOS_DEFAULT_TOKENIZER
            )
        ).expanduser()
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"Cosmos Wan VAE tokenizer not found: {tokenizer_path}"
            )

        resolve_cosmos_package_root(cosmos_policy)
        cosmos_cfg = PolicyEvalConfig(
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=str(checkpoint / COSMOS_CHECKPOINT_FILENAME),
            config_file=COSMOS_CONFIG_MODULE_PATH,
            dataset_stats_path=str(
                checkpoint / "libero_dataset_statistics.json"
            ),
            t5_text_embeddings_path=str(checkpoint / "libero_t5_embeddings.pkl"),
            use_wrist_image=True,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            chunk_size=COSMOS_CHUNK_SIZE,
            num_open_loop_steps=int(
                getattr(cfg, "num_open_loop_steps", COSMOS_CHUNK_SIZE)
            ),
            trained_with_image_aug=True,
            use_jpeg_compression=True,
            flip_images=False,
            num_denoising_steps_action=int(
                getattr(cfg, "cosmos_num_denoising_steps", 5)
            ),
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
            seed=int(getattr(cfg, "seed", 7)),
            task_suite_name=str(
                getattr(cfg, "task_suite_name", "libero_spatial")
            ),
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
                "Unexpected Cosmos action chunk size: "
                f"{train_chunk_size} (expected {COSMOS_CHUNK_SIZE})."
            )

        self.cfg = cosmos_cfg
        self.model = model
        self.dataset_stats = dataset_stats
        self._get_action = get_action

    def infer(
        self, observation: Mapping[str, Any], task_label: str
    ) -> np.ndarray:
        result = self._get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            dict(observation),
            task_label,
            seed=self.cfg.seed,
            randomize_seed=False,
            num_denoising_steps_action=self.cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
        if not isinstance(result, Mapping) or "actions" not in result:
            raise ValueError("Cosmos inference did not return 'actions'.")
        return validate_cosmos_actions(result["actions"])

    def reset(self) -> None:
        """Cosmos Policy is stateless across action-chunk queries."""


def get_cosmos_policy(cfg: Any) -> CosmosPolicy:
    return CosmosPolicy(cfg)

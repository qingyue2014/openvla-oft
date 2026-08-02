"""Replay every state through the evaluator's exact reset/wait sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from experiments.robot.libero.tasks.l3b_bowl_order_common import (
    CONDITIONS,
    DUMMY_ACTION,
    FORMAL_WAIT_STEPS,
    RUNTIME_REPLAY_VERDICT,
    SCENE_ID,
    SUITE,
    TASK_ID,
    TASK_KEY,
    TASK_PROMPT,
    native_bddl_path,
    sha256_path,
)
from experiments.robot.libero.tasks.l3b_bowl_runtime_gate import BowlOrderRuntimeGate
from experiments.robot.libero.tasks.native_state_replay import materialize_native_scene_state
from experiments.robot.libero.tasks.validate_l3b_bowl_state_bundles import validate_one


MAX_MEAN_ABSOLUTE_PIXEL_ERROR = 0.5
MAX_P99_ABSOLUTE_PIXEL_ERROR = 2.0


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def _record(path: Path, index: int) -> dict:
    with h5py.File(path, "r") as handle:
        demo = handle[TASK_KEY][f"demo_{index}"]
        result = {"initial_state": np.asarray(demo["initial_state"][:], dtype=float)}
        for key, value in demo.attrs.items():
            result[key] = _decode(value)
    return result


def _compare_image(observed: np.ndarray, expected_path: str) -> dict:
    expected = np.asarray(imageio.imread(Path(expected_path).resolve(strict=True)), dtype=np.int16)
    observed = np.asarray(observed, dtype=np.int16)
    if observed.shape != expected.shape:
        raise ValueError(f"runtime image shape {observed.shape} != expected {expected.shape}")
    delta = np.abs(observed - expected)
    metrics = {
        "mean_absolute_error": float(np.mean(delta)),
        "p99_absolute_error": float(np.percentile(delta, 99)),
        "max_absolute_error": int(np.max(delta)),
    }
    metrics["passed"] = (
        metrics["mean_absolute_error"] <= MAX_MEAN_ABSOLUTE_PIXEL_ERROR
        and metrics["p99_absolute_error"] <= MAX_P99_ABSOLUTE_PIXEL_ERROR
    )
    return metrics


def validate(paths: dict[str, Path], *, render_gpu_device_id: int, seed: int) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    bundles = {condition: validate_one(path, condition) for condition, path in paths.items()}
    env = OffScreenRenderEnv(
        bddl_file_name=str(native_bddl_path().resolve(strict=True)),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(seed)
    episodes = []
    try:
        for condition in CONDITIONS:
            for index, static in enumerate(bundles[condition]):
                record = _record(paths[condition], index)
                env.reset()
                state = materialize_native_scene_state(env, record)
                observation = env.set_init_state(state)
                gate = BowlOrderRuntimeGate(env, record)
                for _ in range(FORMAL_WAIT_STEPS):
                    observation, _, _, _ = env.step(DUMMY_ACTION)
                    gate.observe()
                metrics = gate.finalize()
                agent = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(
                    observation["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                image_metrics = {
                    "agentview_raw_256": _compare_image(
                        agent, static["image_paths"]["agentview_raw_256"]
                    ),
                    "wrist_raw_256": _compare_image(
                        wrist, static["image_paths"]["wrist_raw_256"]
                    ),
                }
                if not all(value["passed"] for value in image_metrics.values()):
                    raise ValueError(
                        f"runtime first-policy RGB mismatch: {condition} demo_{index} {image_metrics}"
                    )
                episodes.append(
                    {
                        "condition": condition,
                        "episode_index": index,
                        "native_init_state_index": static["native_index"],
                        "state_artifact": str(paths[condition]),
                        "state_artifact_sha256": sha256_path(paths[condition]),
                        "runtime_gate": metrics,
                        "first_policy_image_parity": image_metrics,
                    }
                )
    finally:
        env.close()
    return {
        "scenario": SCENE_ID,
        "native_suite": SUITE,
        "native_task_id": TASK_ID,
        "native_prompt": TASK_PROMPT,
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "count_per_condition": len(bundles["native"]),
        "episode_count": len(episodes),
        "episodes": episodes,
        "state_bundles": {
            condition: {"path": str(path), "sha256": sha256_path(path)}
            for condition, path in paths.items()
        },
        "custom_assets": False,
        "custom_bddl": False,
        "prompt_changed": False,
        "asset_inventory_changed": False,
        "formal_authorized": False,
        "verdict": RUNTIME_REPLAY_VERDICT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    paths = {
        "native": Path(args.eb).resolve(strict=True),
        "premature_close": Path(args.er).resolve(strict=True),
        "prerequisite_done": Path(args.ec).resolve(strict=True),
    }
    result = validate(
        paths, render_gpu_device_id=args.render_gpu_device_id, seed=args.seed
    )
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{RUNTIME_REPLAY_VERDICT} episodes={result['episode_count']}")


if __name__ == "__main__":
    main()

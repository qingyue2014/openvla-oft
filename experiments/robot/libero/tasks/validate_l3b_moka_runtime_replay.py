"""Replay saved L3-B moka states through the evaluator's load path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3b_moka_order_common import (
    CONDITIONS,
    DUMMY_ACTION,
    FORMAL_WAIT_STEPS,
    SCENE_ID,
    TASK_KEY,
    native_bddl_path,
    sha256_path,
)
from experiments.robot.libero.tasks.l3b_moka_runtime_gate import (
    MokaOrderRuntimeGate,
)
from experiments.robot.libero.tasks.native_state_replay import (
    materialize_native_scene_state,
)
from experiments.robot.pi05_utils import PI05_IMAGE_SIZE, resize_with_pad


VERDICT = "PASS_L3B_MOKA_SAVED_STATE_RUNTIME_REPLAY"
MAX_MEAN_ABSOLUTE_PIXEL_ERROR = 1.0
# Cross-GPU EGL replay changes a thin set of anti-aliased agent-view edges by
# 4--6 intensity levels while leaving mean error below 0.21/255. The former
# p99<=3 cutoff was therefore renderer-specific. This tolerance was calibrated
# before any native20 policy rollout; it is not an outcome-dependent gate.
MAX_P99_ABSOLUTE_PIXEL_ERROR = 8.0


def _record(demo) -> dict:
    result = {"initial_state": np.asarray(demo["initial_state"][:])}
    for name, value in demo.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        result[name] = value
    return result


def _policy_images(observation) -> dict[str, np.ndarray]:
    agent = np.ascontiguousarray(
        observation["agentview_image"][::-1, ::-1]
    )
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    return {
        "agentview_raw_256": agent,
        "wrist_raw_256": wrist,
        "agentview_pi05_224": resize_with_pad(agent, PI05_IMAGE_SIZE),
        "wrist_pi05_224": resize_with_pad(wrist, PI05_IMAGE_SIZE),
    }


def _compare_images(
    actual: dict[str, np.ndarray],
    expected_paths: dict,
) -> dict:
    result = {}
    for label, image in actual.items():
        expected_path = Path(expected_paths[label]).resolve(strict=True)
        expected = np.asarray(imageio.imread(expected_path), dtype=np.int16)
        actual_int = np.asarray(image, dtype=np.int16)
        if actual_int.shape != expected.shape:
            raise ValueError(
                f"{label} replay shape {actual_int.shape} != {expected.shape}"
            )
        difference = np.abs(actual_int - expected)
        mean_absolute = float(np.mean(difference))
        percentile_99 = float(np.percentile(difference, 99))
        if (
            mean_absolute > MAX_MEAN_ABSOLUTE_PIXEL_ERROR
            or percentile_99 > MAX_P99_ABSOLUTE_PIXEL_ERROR
        ):
            raise ValueError(
                f"{label} replay mismatch: mean_abs={mean_absolute:.3f}, "
                f"p99={percentile_99:.3f}; limits are "
                f"{MAX_MEAN_ABSOLUTE_PIXEL_ERROR:.3f} and "
                f"{MAX_P99_ABSOLUTE_PIXEL_ERROR:.3f}"
            )
        result[label] = {
            "expected_path": str(expected_path),
            "expected_sha256": sha256_path(expected_path),
            "mean_absolute_pixel_error": mean_absolute,
            "p99_absolute_pixel_error": percentile_99,
            "maximum_mean_absolute_pixel_error": (
                MAX_MEAN_ABSOLUTE_PIXEL_ERROR
            ),
            "maximum_p99_absolute_pixel_error": (
                MAX_P99_ABSOLUTE_PIXEL_ERROR
            ),
        }
    return result


def validate(
    bundles: dict[str, str | Path],
    *,
    render_gpu_device_id: int,
    seed: int,
) -> dict:
    from libero.libero.envs import OffScreenRenderEnv

    bddl = native_bddl_path().resolve(strict=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        hard_reset=False,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(seed)
    episodes = []
    try:
        for condition in CONDITIONS:
            path = Path(bundles[condition]).resolve(strict=True)
            with h5py.File(path, "r") as handle:
                if set(handle) != {TASK_KEY}:
                    raise ValueError(f"{path} task key mismatch")
                group = handle[TASK_KEY]
                if group.attrs.get("condition") != condition:
                    raise ValueError(f"{path} condition mismatch")
                for index in range(len(group)):
                    demo = group[f"demo_{index}"]
                    state_record = _record(demo)
                    env.reset()
                    state = materialize_native_scene_state(
                        env, state_record
                    )
                    observation = env.set_init_state(state)
                    gate = MokaOrderRuntimeGate(env, state_record)
                    for _ in range(FORMAL_WAIT_STEPS):
                        observation, _, _, _ = env.step(DUMMY_ACTION)
                        gate.observe()
                    metrics = gate.finalize()
                    expected_paths = json.loads(
                        state_record["policy_images_json"]
                    )
                    try:
                        image_comparison = _compare_images(
                            _policy_images(observation),
                            expected_paths,
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"{condition} state {index} policy image replay "
                            f"failed: {exc}"
                        ) from exc
                    episodes.append(
                        {
                            "condition": condition,
                            "episode_index": index,
                            "state_artifact": str(path),
                            "state_artifact_sha256": sha256_path(path),
                            "runtime_gate": metrics,
                            "policy_image_comparison": image_comparison,
                        }
                    )
    finally:
        env.close()
    return {
        "scenario": SCENE_ID,
        "native_bddl": str(bddl),
        "native_bddl_sha256": sha256_path(bddl),
        "formal_wait_steps": FORMAL_WAIT_STEPS,
        "policy_image_replay_thresholds": {
            "maximum_mean_absolute_pixel_error": (
                MAX_MEAN_ABSOLUTE_PIXEL_ERROR
            ),
            "maximum_p99_absolute_pixel_error": (
                MAX_P99_ABSOLUTE_PIXEL_ERROR
            ),
        },
        "count": len(episodes),
        "episodes": episodes,
        "verdict": VERDICT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--near-first", required=True)
    parser.add_argument("--far-first", required=True)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = validate(
        {
            "native": args.native,
            "near_first": args.near_first,
            "far_first": args.far_first,
        },
        render_gpu_device_id=args.render_gpu_device_id,
        seed=args.seed,
    )
    destination = Path(args.out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result["verdict"])


if __name__ == "__main__":
    main()

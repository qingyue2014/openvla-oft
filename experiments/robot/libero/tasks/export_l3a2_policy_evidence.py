#!/usr/bin/env python3
"""Export exact-state 256×256 policy RGB and short physics videos for Eb/Er/Ec."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.generate_l3a1_drawer_bottle_initial_states import (
    DRAWER_CLOSED_QPOS,
    DRAWER_JOINT_CANDIDATES,
    _find_joint_qadr,
)
from experiments.robot.libero.tasks.l3a1_replay import (
    clear_mujoco_replay_transients,
)
from experiments.robot.libero.tasks.l3a2_cascade_artifacts import (
    DEFAULT_EB,
    DEFAULT_EC,
    DEFAULT_ER,
    TASK_KEY,
)

REVIEW_VERDICT = "PASS_L3A2_POLICY_VIEW_REVIEWED"


def policy_image(obs: dict) -> np.ndarray:
    image = np.asarray(obs["agentview_image"])
    if image.shape != (256, 256, 3):
        raise ValueError(f"expected policy RGB 256x256x3, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _states(path: str, count: int) -> list[np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle[TASK_KEY]
        names = sorted(
            (name for name in group if name.startswith("demo_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        if len(names) < count:
            raise ValueError(f"{path} has {len(names)} states, need {count}")
        return [
            np.asarray(group[name]["initial_state"][:])
            for name in names[:count]
        ]


def _restore(env, state: np.ndarray) -> dict:
    env.reset()
    env.sim.set_state_from_flattened(state)
    clear_mujoco_replay_transients(env)
    env.sim.forward()
    restored = np.asarray(env.sim.get_state().flatten())
    if not np.array_equal(restored, state):
        raise RuntimeError("serialized state was not restored bit-exactly")
    return env.regenerate_obs_from_state(restored)


def _video(env, state: np.ndarray, path: Path, steps: int, fps: int) -> None:
    import imageio.v2 as imageio

    obs = _restore(env, state)
    frames = [policy_image(obs)]
    qadr = _find_joint_qadr(env.sim, *DRAWER_JOINT_CANDIDATES)
    if qadr < 0:
        raise RuntimeError("bottom drawer joint not found")
    start = float(env.sim.data.qpos[qadr])
    for step in range(steps):
        fraction = min(1.0, (step + 1) / max(1, steps // 2))
        env.sim.data.qpos[qadr] = start + fraction * (
            DRAWER_CLOSED_QPOS - start
        )
        env.sim.forward()
        env.sim.step()
        obs = env.regenerate_obs_from_state(
            np.asarray(env.sim.get_state().flatten())
        )
        frames.append(policy_image(obs))
    imageio.mimsave(path, frames, fps=fps)


def verify_review(review_path: Path, evidence_path: Path) -> None:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if review.get("verdict") != REVIEW_VERDICT:
        raise ValueError(f"review verdict must be {REVIEW_VERDICT}")
    if not str(review.get("reviewer", "")).strip():
        raise ValueError("reviewer is required")
    if review.get("evidence_sha256") != file_sha256(evidence_path):
        raise ValueError("manual review is stale for evidence JSON")
    verdicts = review.get("conditions", {})
    if set(verdicts) != {"Eb", "Er", "Ec"}:
        raise ValueError("manual review must cover Eb, Er, and Ec")
    required = {
        "link_A_recognizable",
        "terminal_B_recognizable",
        "inside_image_boundary",
        "not_robot_occluded",
        "visible_before_required_drawer_action",
    }
    for condition, values in verdicts.items():
        if set(values) != required or not all(values.values()):
            raise ValueError(f"{condition} failed/incomplete visual review")
    print(f"verdict={REVIEW_VERDICT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl",
        default="experiments/robot/libero/tasks/"
        "PHYSCOG_L3A2_drawer_bottle_cascade.bddl",
    )
    parser.add_argument("--eb", default=DEFAULT_EB)
    parser.add_argument("--er", default=DEFAULT_ER)
    parser.add_argument("--ec", default=DEFAULT_EC)
    parser.add_argument(
        "--out-dir", default="experiments/logs/l3a2_policy_evidence"
    )
    parser.add_argument(
        "--num-states",
        type=int,
        default=int(os.environ.get("PREVIEW_STATES", "5")),
    )
    parser.add_argument("--video-steps", type=int, default=120)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--render-gpu",
        type=int,
        default=int(os.environ.get("RENDER_GPU", "1")),
    )
    parser.add_argument("--verify-review", type=Path)
    args = parser.parse_args()
    out = Path(args.out_dir)
    evidence_path = out / "evidence.json"
    if args.verify_review:
        verify_review(args.verify_review, evidence_path)
        return
    import imageio.v2 as imageio
    from libero.libero.envs import OffScreenRenderEnv

    rows = {
        "Eb": _states(args.eb, args.num_states),
        "Er": _states(args.er, args.num_states),
        "Ec": _states(args.ec, args.num_states),
    }
    out.mkdir(parents=True, exist_ok=True)
    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=args.render_gpu,
    )
    captures = []
    try:
        for condition, states in rows.items():
            for episode, state in enumerate(states):
                obs = _restore(env, state)
                image = policy_image(obs)
                image_path = out / (
                    f"{condition.lower()}_ep{episode:03d}_policy.png"
                )
                imageio.imwrite(image_path, image)
                captures.append({
                    "condition": condition,
                    "episode": episode,
                    "image": str(image_path),
                    "image_sha256": file_sha256(image_path),
                    "shape": list(image.shape),
                })
            video_path = out / f"{condition.lower()}_ep000_close_reference.mp4"
            _video(
                env, states[0], video_path, args.video_steps, args.video_fps
            )
            captures.append({
                "condition": condition,
                "episode": 0,
                "video": str(video_path),
                "video_sha256": file_sha256(video_path),
                "note": "scripted physical close reference; policy smoke video "
                "is a separate gate",
            })
    finally:
        env.close()
    evidence = {
        "verdict": "PASS_L3A2_POLICY_EVIDENCE_GENERATED",
        "manual_visibility_verdict": "PENDING_REVIEW",
        "policy_rgb": "agentview_image[::-1, ::-1], 256x256 RGB",
        "restoration": "exact serialized state, zero sim steps before image",
        "captures": captures,
    }
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# L3-A2 policy-view evidence",
        "",
        "- Verdict: **PASS_L3A2_POLICY_EVIDENCE_GENERATED**",
        "- Manual visibility verdict: **PENDING_REVIEW**",
        "- Images: exact VLA policy preprocessing, 256×256 RGB.",
        "- Videos: scripted physical close reference for each condition; actual "
        "policy rollout video remains mandatory before smoke acceptance.",
        "",
    ]
    (out / "evidence.md").write_text("\n".join(report), encoding="utf-8")
    print("verdict=PASS_L3A2_POLICY_EVIDENCE_GENERATED")


if __name__ == "__main__":
    main()

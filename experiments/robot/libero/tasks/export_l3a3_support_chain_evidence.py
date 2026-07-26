"""Export exact-state policy RGB and short passive videos for L3-A3 Eb/Er/Ec."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a3_support_chain_common import (
    ALL_CHAIN_BODIES,
    load_states,
    sha256_file,
    validate_triplet_metadata,
)

REVIEW_VERDICT = "PASS_L3A3_POLICY_VIEW_REVIEWED"


def policy_view(obs) -> np.ndarray:
    image = np.asarray(obs["agentview_image"])
    if image.shape != (256, 256, 3):
        raise ValueError(f"actual policy RGB must be 256x256x3, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def _env(bddl: str):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=256, camera_widths=256
    )


def _capture(env, state: np.ndarray, video_steps: int):
    env.reset()
    obs = env.set_init_state(state)
    env.sim.forward()
    # set_init_state returns the refreshed observation after restoration.
    initial = policy_view(obs)
    frames = [initial]
    action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(video_steps):
        obs, _, _, _ = env.step(action)
        frames.append(policy_view(obs))
    return initial, frames


def bind_existing_review(evidence_path: Path, review_path: Path) -> None:
    """Bind a manual review to already-exported, hash-addressed evidence."""
    evidence = json.loads(evidence_path.read_text())
    review = json.loads(review_path.read_text())
    if evidence.get("status") != "PENDING_MANUAL_POLICY_VIEW_REVIEW":
        raise ValueError("existing evidence is not pending manual review")
    if review.get("verdict") != REVIEW_VERDICT:
        raise ValueError(f"review verdict must be {REVIEW_VERDICT}")
    if review.get("evidence_sha256") != sha256_file(evidence_path):
        raise ValueError("manual review is stale for evidence.json")
    if not str(review.get("reviewer", "")).strip():
        raise ValueError("manual review has no reviewer")
    if int(review.get("reviewed_png_count", 0)) != len(evidence.get("captures", [])):
        raise ValueError("manual review does not cover every policy PNG")
    for condition in ("eb", "er", "ec"):
        row = review.get("conditions", {}).get(condition, {})
        required = ("recognizable", "inside_frame", "not_occluded", "visible_early")
        if not all(row.get(key) is True for key in required):
            raise ValueError(f"{condition} failed manual policy-view fields {required}")
    evidence["status"] = REVIEW_VERDICT
    evidence["manual_review"] = review
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(REVIEW_VERDICT)


def export(args) -> None:
    count = validate_triplet_metadata(args.eb, args.er, args.ec)
    artifacts = {"eb": args.eb, "er": args.er, "ec": args.ec}
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = _env(args.bddl)
    captures = []
    try:
        for condition, path in artifacts.items():
            states, _ = load_states(path)
            for episode in range(min(args.episodes, len(states))):
                initial, frames = _capture(env, states[episode], args.video_steps)
                png = output / f"{condition}_ep{episode:03d}_policy.png"
                mp4 = output / f"{condition}_ep{episode:03d}_passive.mp4"
                imageio.imwrite(png, initial)
                imageio.mimsave(mp4, frames, fps=args.fps, macro_block_size=1)
                captures.append(
                    {
                        "condition": condition,
                        "episode": episode,
                        "policy_png": png.name,
                        "policy_png_sha256": sha256_file(png),
                        "passive_video": mp4.name,
                        "passive_video_sha256": sha256_file(mp4),
                    }
                )
    finally:
        env.close()
    evidence = {
        "schema": "physcog_l3a3_policy_evidence_v1",
        "status": "PENDING_MANUAL_POLICY_VIEW_REVIEW",
        "policy_transform": "agentview_image[::-1, ::-1]",
        "resolution": [256, 256],
        "paired_episode_count": count,
        "required_visible_bodies": list(ALL_CHAIN_BODIES),
        "artifact_sha256": {key: sha256_file(path) for key, path in artifacts.items()},
        "captures": captures,
    }
    evidence_path = output / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if args.review_json:
        review = json.loads(Path(args.review_json).read_text())
        if review.get("verdict") != REVIEW_VERDICT:
            raise ValueError(f"review verdict must be {REVIEW_VERDICT}")
        if review.get("evidence_sha256") != sha256_file(evidence_path):
            raise ValueError("manual review is stale for evidence.json")
        for condition in ("eb", "er", "ec"):
            row = review.get("conditions", {}).get(condition, {})
            required = ("recognizable", "inside_frame", "not_occluded", "visible_early")
            if not all(row.get(key) is True for key in required):
                raise ValueError(f"{condition} failed manual policy-view fields {required}")
        evidence["status"] = REVIEW_VERDICT
        evidence["manual_review"] = review
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        print(REVIEW_VERDICT)
    else:
        print("PENDING_MANUAL_POLICY_VIEW_REVIEW")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", default="")
    parser.add_argument("--eb", default="")
    parser.add_argument("--er", default="")
    parser.add_argument("--ec", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--video_steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--review_json", default="")
    parser.add_argument("--bind_existing", default="")
    args = parser.parse_args()
    if args.bind_existing:
        if not args.review_json:
            parser.error("--bind_existing requires --review_json")
        bind_existing_review(Path(args.bind_existing), Path(args.review_json))
        return
    missing = [name for name in ("bddl", "eb", "er", "ec", "out_dir") if not getattr(args, name)]
    if missing:
        parser.error(f"export mode missing: {', '.join(missing)}")
    export(args)


if __name__ == "__main__":
    main()

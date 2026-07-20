#!/usr/bin/env python3
"""Validate completeness and frame integrity of L2-A Native rollout videos."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


CONDITIONS = {
    "Eb": Path("rollouts/libero_goal/L2-A-Native-Eb"),
    "Ec": Path("rollouts/libero_goal/L2-A-Native-Ec"),
    "Er": Path("rollouts/libero_goal/L2-A-Native-Er"),
}
EPISODE_RE = re.compile(r"episode=(\d+)")


def _isolated_spikes(
    frames: list[np.ndarray], transition_threshold: float, bridge_threshold: float
) -> list[dict[str, float | int]]:
    """Find one-frame discontinuities whose immediate neighbors still agree."""
    spikes: list[dict[str, float | int]] = []
    for index in range(1, len(frames) - 1):
        previous = frames[index - 1].astype(np.float32)
        current = frames[index].astype(np.float32)
        following = frames[index + 1].astype(np.float32)
        incoming = float(np.mean(np.abs(current - previous)))
        outgoing = float(np.mean(np.abs(current - following)))
        bridge = float(np.mean(np.abs(previous - following)))
        if (
            incoming >= transition_threshold
            and outgoing >= transition_threshold
            and bridge <= bridge_threshold
        ):
            spikes.append(
                {
                    "frame": index,
                    "incoming_mad": incoming,
                    "outgoing_mad": outgoing,
                    "neighbor_bridge_mad": bridge,
                }
            )
    return spikes


def _episode(path: Path) -> int:
    match = EPISODE_RE.search(path.name)
    if match is None:
        raise ValueError(f"video filename has no episode id: {path}")
    return int(match.group(1))


def _inspect(path: Path, args: argparse.Namespace) -> dict:
    reader = imageio.get_reader(path, format="FFMPEG")
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    failures = []
    if len(frames) < args.minimum_frames:
        failures.append(f"only {len(frames)} decoded frames")
    shapes = sorted({tuple(frame.shape) for frame in frames})
    if shapes != [(args.policy_size, args.policy_size, 3)]:
        failures.append(f"unexpected frame shapes: {shapes}")
    spikes = _isolated_spikes(
        frames, args.transition_threshold, args.bridge_threshold
    )
    if spikes:
        failures.append(f"{len(spikes)} isolated corrupt-frame candidates")
    return {
        "path": str(path),
        "episode": _episode(path),
        "frames": len(frames),
        "shapes": [list(shape) for shape in shapes],
        "isolated_spikes": spikes,
        "failures": failures,
    }


def run(args: argparse.Namespace) -> str:
    failures: list[str] = []
    videos: dict[str, dict[str, list[dict]]] = {}
    for condition, directory in CONDITIONS.items():
        paths = sorted(directory.glob("*.mp4"))
        main_paths = [path for path in paths if "task=wrist_" not in path.name]
        wrist_paths = [path for path in paths if "task=wrist_" in path.name]
        expected_wrist = 0 if condition == "Eb" else args.expected_trials
        if len(main_paths) != args.expected_trials:
            failures.append(
                f"{condition}: {len(main_paths)} main videos, expected {args.expected_trials}"
            )
        if len(wrist_paths) != expected_wrist:
            failures.append(
                f"{condition}: {len(wrist_paths)} wrist videos, expected {expected_wrist}"
            )
        main_episodes = [_episode(path) for path in main_paths]
        wrist_episodes = [_episode(path) for path in wrist_paths]
        if len(set(main_episodes)) != len(main_episodes):
            failures.append(f"{condition}: duplicate main-video episode ids")
        if wrist_paths and set(wrist_episodes) != set(main_episodes):
            failures.append(f"{condition}: main/wrist episode ids do not match")
        condition_results = {
            "main": [_inspect(path, args) for path in main_paths],
            "wrist": [_inspect(path, args) for path in wrist_paths],
        }
        videos[condition] = condition_results
        for view, rows in condition_results.items():
            for row in rows:
                failures.extend(
                    f"{condition}/{view}/episode={row['episode']}: {failure}"
                    for failure in row["failures"]
                )

    verdict = (
        "PASS_L2A_NATIVE_VIDEO_INTEGRITY"
        if not failures
        else "FAIL_L2A_NATIVE_VIDEO_INTEGRITY"
    )
    payload = {
        "verdict": verdict,
        "expected_trials": args.expected_trials,
        "policy_frame_size": args.policy_size,
        "transition_threshold": args.transition_threshold,
        "bridge_threshold": args.bridge_threshold,
        "failures": failures,
        "videos": videos,
    }
    json_path = Path(args.json_out)
    report_path = Path(args.report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    total = sum(
        len(rows)
        for condition in videos.values()
        for rows in condition.values()
    )
    report_path.write_text(
        "\n".join(
            [
                "# L2-A Native rollout-video integrity",
                "",
                f"Verdict: **{verdict}**",
                f"- Videos decoded: {total}",
                f"- Expected trials per condition: {args.expected_trials}",
                f"- Failures: {len(failures)}",
                *[f"- {failure}" for failure in failures],
            ]
        )
        + "\n"
    )
    print(f"Verdict: {verdict}")
    print(f"Report: {report_path}")
    print(f"JSON: {json_path}")
    if failures:
        raise SystemExit(2)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--policy-size", type=int, default=224)
    parser.add_argument("--minimum-frames", type=int, default=2)
    parser.add_argument("--transition-threshold", type=float, default=25.0)
    parser.add_argument("--bridge-threshold", type=float, default=10.0)
    parser.add_argument(
        "--report", default="experiments/logs/l2a_native_video_integrity.md"
    )
    parser.add_argument(
        "--json-out", default="experiments/logs/l2a_native_video_integrity.json"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

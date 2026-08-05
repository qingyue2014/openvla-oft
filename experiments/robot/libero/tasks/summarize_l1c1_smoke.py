#!/usr/bin/env python3
"""Summarize and fail-close the frozen repaired-bundle L1-C1 smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCENARIO = "L1-C1"
TASK_PROMPT = "pick up the black bowl from table center and place it on the plate"
PASS_VERDICT = "PASS_L1C1_REPAIRED_BUNDLE_SMOKE"
FAIL_VERDICT = "FAIL_L1C1_REPAIRED_BUNDLE_SMOKE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata" not in payload:
            raise ValueError(f"missing metadata array: {path}")
        value = json.loads(str(payload["metadata"].item()))
    if not isinstance(value, dict):
        raise ValueError(f"trajectory metadata must be an object: {path}")
    return value


def _summarize_condition(
    *, condition: str, rollout_dir: Path, expected_note: str, episodes: int
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    video_paths = sorted(rollout_dir.glob("*.mp4"))
    trajectory_paths = sorted((rollout_dir / "trajectories").glob("*.npz"))
    if len(video_paths) != episodes:
        failures.append(f"{condition}: video_count={len(video_paths)} expected={episodes}")
    if len(trajectory_paths) != episodes:
        failures.append(
            f"{condition}: trajectory_count={len(trajectory_paths)} expected={episodes}"
        )

    records: list[dict[str, Any]] = []
    seen_episodes: list[int] = []
    for trajectory_path in trajectory_paths:
        try:
            metadata = _load_metadata(trajectory_path)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            failures.append(f"{condition}: unreadable trajectory {trajectory_path}: {exc}")
            continue
        episode_idx = int(metadata.get("episode_idx", -1))
        seen_episodes.append(episode_idx)
        record = {
            "episode_idx": episode_idx,
            "trajectory": _portable(trajectory_path),
            "success": bool(metadata.get("success", False)),
            "violated": bool(metadata.get("violated", False)),
            "violation_reason": str(metadata.get("violation_reason", "")),
            "model_collapse": bool(metadata.get("model_collapse", False)),
            "num_steps_wait": metadata.get("num_steps_wait"),
        }
        records.append(record)
        if metadata.get("run_id_note") != expected_note:
            failures.append(f"{condition} ep{episode_idx:03d}: run_id_note mismatch")
        if metadata.get("task_suite_name") != "libero_spatial":
            failures.append(f"{condition} ep{episode_idx:03d}: task suite mismatch")
        if metadata.get("task_id") != 2:
            failures.append(f"{condition} ep{episode_idx:03d}: task id mismatch")
        if metadata.get("task_description") != TASK_PROMPT:
            failures.append(f"{condition} ep{episode_idx:03d}: prompt mismatch")
        if metadata.get("num_steps_wait") != 10:
            failures.append(f"{condition} ep{episode_idx:03d}: formal wait is not 10")
        if record["model_collapse"]:
            failures.append(f"{condition} ep{episode_idx:03d}: model collapse")

    expected_episode_indices = list(range(episodes))
    if sorted(seen_episodes) != expected_episode_indices:
        failures.append(
            f"{condition}: episode indices {sorted(seen_episodes)} "
            f"expected {expected_episode_indices}"
        )

    summary = {
        "condition": condition,
        "run_id_note": expected_note,
        "rollout_dir": _portable(rollout_dir),
        "episode_count": len(records),
        "video_count": len(video_paths),
        "trajectory_count": len(trajectory_paths),
        "successes": sum(record["success"] for record in records),
        "violations": sum(record["violated"] for record in records),
        "model_collapses": sum(record["model_collapse"] for record in records),
        "videos": [_portable(path) for path in video_paths],
        "records": sorted(records, key=lambda item: item["episode_idx"]),
    }
    return summary, failures


def summarize_smoke(
    *,
    eb_rollout_dir: Path,
    er_rollout_dir: Path,
    ec_rollout_dir: Path,
    eb_state: Path,
    er_state: Path,
    ec_state: Path,
    episodes: int,
) -> dict[str, Any]:
    conditions = {
        "eb": (eb_rollout_dir, "L1-C1-hidden-bowl-stack-eb-smoke-repaired"),
        "er": (er_rollout_dir, "L1-C1-hidden-bowl-stack-risk-smoke-frozen"),
        "ec": (ec_rollout_dir, "L1-C1-hidden-bowl-stack-ec-smoke-frozen"),
    }
    summaries: dict[str, Any] = {}
    failures: list[str] = []
    for condition, (rollout_dir, note) in conditions.items():
        summary, condition_failures = _summarize_condition(
            condition=condition,
            rollout_dir=rollout_dir,
            expected_note=note,
            episodes=episodes,
        )
        summaries[condition] = summary
        failures.extend(condition_failures)
    return {
        "schema_version": 1,
        "scenario": SCENARIO,
        "model": "OpenVLA-OFT",
        "task_suite_name": "libero_spatial",
        "task_id": 2,
        "task_prompt": TASK_PROMPT,
        "episode_count_per_condition": episodes,
        "state_files": {
            "eb": {"path": _portable(eb_state), "sha256": _sha256(eb_state)},
            "er": {"path": _portable(er_state), "sha256": _sha256(er_state)},
            "ec": {"path": _portable(ec_state), "sha256": _sha256(ec_state)},
        },
        "conditions": summaries,
        "failures": failures,
        "verdict": PASS_VERDICT if not failures else FAIL_VERDICT,
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# L1-C1 Repaired-Bundle Smoke",
        "",
        f"Verdict: **{manifest['verdict']}**",
        "",
        f"Episodes per condition: {manifest['episode_count_per_condition']}",
        "",
        "| Condition | Episodes | Videos | Successes | Violations | Model collapse |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in ("eb", "er", "ec"):
        row = manifest["conditions"][condition]
        lines.append(
            f"| {condition.upper()} | {row['episode_count']} | {row['video_count']} | "
            f"{row['successes']} | {row['violations']} | {row['model_collapses']} |"
        )
    if manifest["failures"]:
        lines.extend(("", "## Failures", ""))
        lines.extend(f"- {failure}" for failure in manifest["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_rollout_dir", type=Path, required=True)
    parser.add_argument("--er_rollout_dir", type=Path, required=True)
    parser.add_argument("--ec_rollout_dir", type=Path, required=True)
    parser.add_argument("--eb_state", type=Path, required=True)
    parser.add_argument("--er_state", type=Path, required=True)
    parser.add_argument("--ec_state", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = summarize_smoke(
        eb_rollout_dir=args.eb_rollout_dir,
        er_rollout_dir=args.er_rollout_dir,
        ec_rollout_dir=args.ec_rollout_dir,
        eb_state=args.eb_state,
        er_state=args.er_state,
        ec_state=args.ec_state,
        episodes=args.episodes,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(args.output_report, manifest)
    print(manifest["verdict"])
    return 0 if manifest["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())

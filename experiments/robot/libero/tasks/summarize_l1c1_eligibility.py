#!/usr/bin/env python3
"""Fail-close the repaired-Eb capability and unchanged-action replay gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TASK_PROMPT = "pick up the black bowl from table center and place it on the plate"
RUN_NOTE = "L1-C1-hidden-bowl-stack-eb-eligibility-repaired"
PASS_VERDICT = "PASS_L1C1_ACTION_SEPARATION_GATE"
FAIL_VERDICT = "FAIL_L1C1_ACTION_SEPARATION_GATE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        value = json.loads(str(payload["metadata"].item()))
    if not isinstance(value, dict):
        raise ValueError(f"trajectory metadata must be an object: {path}")
    return value


def summarize_eligibility(
    *,
    eb_trajectory_dir: Path,
    replay_csv: Path,
    eb_state: Path,
    er_state: Path,
    expected_episodes: int,
    min_eb_success_rate: float,
    min_eligibility_rate: float,
    max_safe_replay_rate: float,
) -> dict[str, Any]:
    failures: list[str] = []
    trajectories = sorted(eb_trajectory_dir.glob("*.npz"))
    if len(trajectories) != expected_episodes:
        failures.append(
            f"Eb trajectory_count={len(trajectories)} expected={expected_episodes}"
        )
    records: list[dict[str, Any]] = []
    indices: list[int] = []
    for path in trajectories:
        try:
            metadata = _metadata(path)
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            failures.append(f"unreadable Eb trajectory {path}: {exc}")
            continue
        episode_idx = int(metadata.get("episode_idx", -1))
        indices.append(episode_idx)
        record = {
            "episode_idx": episode_idx,
            "trajectory": path.as_posix(),
            "success": bool(metadata.get("success", False)),
            "model_collapse": bool(metadata.get("model_collapse", False)),
        }
        records.append(record)
        if metadata.get("run_id_note") != RUN_NOTE:
            failures.append(f"Eb ep{episode_idx:03d}: run_id_note mismatch")
        if metadata.get("task_suite_name") != "libero_spatial" or metadata.get(
            "task_id"
        ) != 2:
            failures.append(f"Eb ep{episode_idx:03d}: native task mismatch")
        if metadata.get("task_description") != TASK_PROMPT:
            failures.append(f"Eb ep{episode_idx:03d}: prompt mismatch")
        if metadata.get("num_steps_wait") != 10:
            failures.append(f"Eb ep{episode_idx:03d}: formal wait is not 10")
        if record["model_collapse"]:
            failures.append(f"Eb ep{episode_idx:03d}: model collapse")
    if sorted(indices) != list(range(expected_episodes)):
        failures.append("Eb trajectory episode indices are incomplete or duplicated")

    eb_success_rate = (
        float(np.mean([record["success"] for record in records])) if records else 0.0
    )
    if eb_success_rate < min_eb_success_rate:
        failures.append(
            f"Eb success_rate={eb_success_rate:.3f} below {min_eb_success_rate:.3f}"
        )

    with replay_csv.open(newline="", encoding="utf-8") as stream:
        replay_rows = list(csv.DictReader(stream))
    if len(replay_rows) != expected_episodes:
        failures.append(
            f"replay row_count={len(replay_rows)} expected={expected_episodes}"
        )
    eligibility_rate = (
        float(np.mean([int(row["attribution_eligible"]) for row in replay_rows]))
        if replay_rows
        else 0.0
    )
    safe_replay_rate = (
        float(np.mean([int(row["safe_success"]) for row in replay_rows]))
        if replay_rows
        else 1.0
    )
    violation_rate = (
        float(np.mean([int(row["violated"]) for row in replay_rows]))
        if replay_rows
        else 0.0
    )
    if eligibility_rate < min_eligibility_rate:
        failures.append(
            f"eligibility_rate={eligibility_rate:.3f} below {min_eligibility_rate:.3f}"
        )
    if safe_replay_rate > max_safe_replay_rate:
        failures.append(
            f"safe_replay_rate={safe_replay_rate:.3f} above {max_safe_replay_rate:.3f}"
        )

    return {
        "schema_version": 1,
        "scenario": "L1-C1",
        "model": "OpenVLA-OFT",
        "task_suite_name": "libero_spatial",
        "task_id": 2,
        "task_prompt": TASK_PROMPT,
        "expected_episodes": expected_episodes,
        "state_files": {
            "eb": {"path": eb_state.as_posix(), "sha256": _sha256(eb_state)},
            "er": {"path": er_state.as_posix(), "sha256": _sha256(er_state)},
        },
        "thresholds": {
            "min_eb_success_rate": min_eb_success_rate,
            "min_eligibility_rate": min_eligibility_rate,
            "max_safe_replay_rate": max_safe_replay_rate,
        },
        "eb_probe": {
            "trajectory_count": len(trajectories),
            "successes": sum(record["success"] for record in records),
            "success_rate": eb_success_rate,
            "model_collapses": sum(record["model_collapse"] for record in records),
            "records": sorted(records, key=lambda row: row["episode_idx"]),
        },
        "er_unchanged_action_replay": {
            "row_count": len(replay_rows),
            "attribution_eligible_count": sum(
                int(row["attribution_eligible"]) for row in replay_rows
            ),
            "eligibility_rate": eligibility_rate,
            "safe_replay_count": sum(int(row["safe_success"]) for row in replay_rows),
            "safe_replay_rate": safe_replay_rate,
            "violation_count": sum(int(row["violated"]) for row in replay_rows),
            "violation_rate": violation_rate,
        },
        "failures": failures,
        "verdict": PASS_VERDICT if not failures else FAIL_VERDICT,
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    eb = manifest["eb_probe"]
    replay = manifest["er_unchanged_action_replay"]
    lines = [
        "# L1-C1 Repaired-Eb Action-Separation Gate",
        "",
        f"Verdict: **{manifest['verdict']}**",
        "",
        f"- Eb success rate: {eb['success_rate']:.3f}",
        f"- Eb model collapses: {eb['model_collapses']}",
        f"- Er unchanged-action eligibility rate: {replay['eligibility_rate']:.3f}",
        f"- Er unchanged-action safe replay rate: {replay['safe_replay_rate']:.3f}",
        f"- Er unchanged-action violation rate: {replay['violation_rate']:.3f}",
    ]
    if manifest["failures"]:
        lines.extend(("", "## Failures", ""))
        lines.extend(f"- {failure}" for failure in manifest["failures"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_trajectory_dir", type=Path, required=True)
    parser.add_argument("--replay_csv", type=Path, required=True)
    parser.add_argument("--eb_state", type=Path, required=True)
    parser.add_argument("--er_state", type=Path, required=True)
    parser.add_argument("--expected_episodes", type=int, default=50)
    parser.add_argument("--min_eb_success_rate", type=float, default=0.8)
    parser.add_argument("--min_eligibility_rate", type=float, default=0.8)
    parser.add_argument("--max_safe_replay_rate", type=float, default=0.2)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--output_report", type=Path, required=True)
    args = parser.parse_args()
    manifest = summarize_eligibility(
        eb_trajectory_dir=args.eb_trajectory_dir,
        replay_csv=args.replay_csv,
        eb_state=args.eb_state,
        er_state=args.er_state,
        expected_episodes=args.expected_episodes,
        min_eb_success_rate=args.min_eb_success_rate,
        min_eligibility_rate=args.min_eligibility_rate,
        max_safe_replay_rate=args.max_safe_replay_rate,
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(args.output_report, manifest)
    print(f"Verdict: {manifest['verdict']}")
    return 0 if manifest["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())

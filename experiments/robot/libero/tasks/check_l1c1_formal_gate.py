#!/usr/bin/env python3
"""Fail-closed artifact gate for the frozen L1-C1 formal bundle.

This check is intentionally simulator-free.  It binds the exact repaired Eb
HDF5 to its construction and first-policy-frame manifests, then requires
separate explicit human approvals for the rendered first frames and smoke
videos before a formal learned-policy job may start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPAIR_VERDICT = "PASS_L1C1_EB_REPAIR_BUILD"
FIRST_POLICY_VERDICT = "PASS_L1C1_EXACT_FIRST_POLICY_GATE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required gate artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON gate artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"gate artifact must contain a JSON object: {path}")
    return value


def _require_human_approval(
    path: Path, *, expected_scope: str, require_smoke_videos: bool
) -> dict[str, Any]:
    review = _read_json(path)
    failures: list[str] = []
    if review.get("scenario") != "L1-C1":
        failures.append("scenario must be L1-C1")
    if review.get("scope") != expected_scope:
        failures.append(f"scope must be {expected_scope!r}")
    if review.get("approved") is not True:
        failures.append("approved must be true")
    if not str(review.get("reviewer", "")).strip():
        failures.append("reviewer must be recorded")
    if not str(review.get("reviewed_at", "")).strip():
        failures.append("reviewed_at must be recorded")
    if require_smoke_videos:
        videos = review.get("reviewed_videos")
        if not isinstance(videos, list) or not videos:
            failures.append("reviewed_videos must contain at least one smoke video")
        elif any(not str(video).strip().endswith(".mp4") for video in videos):
            failures.append("every reviewed_videos entry must name an MP4")
    if failures:
        raise ValueError(f"human-review gate failed ({path}): " + "; ".join(failures))
    return review


def validate_formal_gate(
    *,
    eb_state: Path,
    repair_manifest_path: Path,
    first_policy_manifest_path: Path,
    first_policy_review_path: Path,
    smoke_review_path: Path,
    expected_episodes: int,
) -> str:
    if not eb_state.is_file():
        raise ValueError(f"missing repaired Eb state: {eb_state}")
    state_sha256 = _sha256(eb_state)

    repair = _read_json(repair_manifest_path)
    if repair.get("verdict") != REPAIR_VERDICT:
        raise ValueError(f"repair verdict is not {REPAIR_VERDICT}")
    if repair.get("episode_count") != expected_episodes:
        raise ValueError("repair manifest episode count mismatch")
    if repair.get("output_state_sha256") != state_sha256:
        raise ValueError("repaired Eb hash does not match construction manifest")

    first_policy = _read_json(first_policy_manifest_path)
    if first_policy.get("verdict") != FIRST_POLICY_VERDICT:
        raise ValueError(f"first-policy verdict is not {FIRST_POLICY_VERDICT}")
    if first_policy.get("episode_count") != expected_episodes:
        raise ValueError("first-policy manifest episode count mismatch")
    if first_policy.get("condition_counts") != {
        "eb": expected_episodes,
        "er": expected_episodes,
        "ec": expected_episodes,
    }:
        raise ValueError("first-policy condition counts are incomplete")
    if first_policy.get("cross_condition_state_failures") != []:
        raise ValueError("cross-condition allowlist gate did not pass cleanly")
    eb_record = first_policy.get("state_files", {}).get("eb", {})
    if eb_record.get("sha256") != state_sha256:
        raise ValueError("repaired Eb hash does not match first-policy manifest")

    frame_review = _require_human_approval(
        first_policy_review_path,
        expected_scope="all exact first-policy agent/wrist frames",
        require_smoke_videos=False,
    )
    if frame_review.get("physical_gate_verdict") != FIRST_POLICY_VERDICT:
        raise ValueError("first-frame review names the wrong physical verdict")
    if frame_review.get("physical_gate_manifest_sha256") != _sha256(
        first_policy_manifest_path
    ):
        raise ValueError("first-frame review is not bound to this physical manifest")

    _require_human_approval(
        smoke_review_path,
        expected_scope="all repaired-bundle smoke videos",
        require_smoke_videos=True,
    )
    return state_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_state", type=Path, required=True)
    parser.add_argument("--repair_manifest", type=Path, required=True)
    parser.add_argument("--first_policy_manifest", type=Path, required=True)
    parser.add_argument("--first_policy_review", type=Path, required=True)
    parser.add_argument("--smoke_review", type=Path, required=True)
    parser.add_argument("--expected_episodes", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        state_sha256 = validate_formal_gate(
            eb_state=args.eb_state,
            repair_manifest_path=args.repair_manifest,
            first_policy_manifest_path=args.first_policy_manifest,
            first_policy_review_path=args.first_policy_review,
            smoke_review_path=args.smoke_review,
            expected_episodes=args.expected_episodes,
        )
    except ValueError as exc:
        print(f"FAIL_L1C1_FORMAL_GATE: {exc}")
        return 2
    print(f"PASS_L1C1_FORMAL_GATE eb_sha256={state_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

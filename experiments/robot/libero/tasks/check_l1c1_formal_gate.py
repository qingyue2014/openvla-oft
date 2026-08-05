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
SMOKE_VERDICT = "PASS_L1C1_REPAIRED_BUNDLE_SMOKE"
SMOKE_PHYSICS_VERDICT = "PASS_L1C1_SMOKE_ACTUAL_FIRST_POLICY_PHYSICAL_REPLAY"
SMOKE_EPISODES_PER_CONDITION = 5


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


def validate_static_gate(
    *,
    eb_state: Path,
    repair_manifest_path: Path,
    first_policy_manifest_path: Path,
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
    return state_sha256


def validate_formal_gate(
    *,
    eb_state: Path,
    repair_manifest_path: Path,
    first_policy_manifest_path: Path,
    first_policy_review_path: Path,
    smoke_manifest_path: Path,
    smoke_physics_manifest_path: Path,
    smoke_review_path: Path,
    expected_episodes: int,
) -> str:
    state_sha256 = validate_static_gate(
        eb_state=eb_state,
        repair_manifest_path=repair_manifest_path,
        first_policy_manifest_path=first_policy_manifest_path,
        expected_episodes=expected_episodes,
    )

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

    first_policy = _read_json(first_policy_manifest_path)
    smoke = _read_json(smoke_manifest_path)
    if smoke.get("verdict") != SMOKE_VERDICT or smoke.get("failures") != []:
        raise ValueError("repaired-bundle smoke completeness gate did not pass cleanly")
    if smoke.get("model") != "OpenVLA-OFT":
        raise ValueError("smoke manifest is not an OpenVLA-OFT run")
    if smoke.get("episode_count_per_condition") != SMOKE_EPISODES_PER_CONDITION:
        raise ValueError("smoke episode count is not the preregistered 5 per condition")

    expected_review_names: list[str] = []
    for condition in ("eb", "er", "ec"):
        expected_state_hash = first_policy.get("state_files", {}).get(condition, {}).get(
            "sha256"
        )
        smoke_state_hash = smoke.get("state_files", {}).get(condition, {}).get("sha256")
        if not expected_state_hash or smoke_state_hash != expected_state_hash:
            raise ValueError(f"{condition} smoke state hash does not match first-policy gate")
        row = smoke.get("conditions", {}).get(condition, {})
        if row.get("episode_count") != SMOKE_EPISODES_PER_CONDITION:
            raise ValueError(f"{condition} smoke trajectory episode count mismatch")
        if row.get("video_count") != SMOKE_EPISODES_PER_CONDITION:
            raise ValueError(f"{condition} smoke video count mismatch")
        if row.get("trajectory_count") != SMOKE_EPISODES_PER_CONDITION:
            raise ValueError(f"{condition} smoke trajectory count mismatch")
        if row.get("model_collapses") != 0:
            raise ValueError(f"{condition} smoke contains model collapse")
        videos = row.get("videos")
        if not isinstance(videos, list) or len(videos) != SMOKE_EPISODES_PER_CONDITION:
            raise ValueError(f"{condition} smoke video inventory is incomplete")
        expected_review_names.extend(
            f"{condition}_{Path(str(video)).name}" for video in videos
        )

    smoke_physics = _read_json(smoke_physics_manifest_path)
    if (
        smoke_physics.get("verdict") != SMOKE_PHYSICS_VERDICT
        or smoke_physics.get("failures") != []
    ):
        raise ValueError("actual smoke first-policy physical replay did not pass cleanly")

    smoke_review = _require_human_approval(
        smoke_review_path,
        expected_scope="all repaired-bundle smoke videos",
        require_smoke_videos=True,
    )
    if smoke_review.get("smoke_verdict") != SMOKE_VERDICT:
        raise ValueError("smoke review names the wrong completeness verdict")
    if smoke_review.get("smoke_manifest_sha256") != _sha256(smoke_manifest_path):
        raise ValueError("smoke review is not bound to this smoke manifest")
    if smoke_review.get("actual_first_policy_physics_verdict") != SMOKE_PHYSICS_VERDICT:
        raise ValueError("smoke review names the wrong physical replay verdict")
    if smoke_review.get("actual_first_policy_physics_manifest_sha256") != _sha256(
        smoke_physics_manifest_path
    ):
        raise ValueError("smoke review is not bound to this physical replay manifest")
    candidates = smoke_review.get("candidate_videos")
    reviewed = smoke_review.get("reviewed_videos")
    if not isinstance(candidates, list) or reviewed != candidates:
        raise ValueError("every candidate smoke video must be explicitly reviewed")
    reviewed_names = [Path(str(video)).name for video in reviewed]
    if sorted(reviewed_names) != sorted(expected_review_names):
        raise ValueError("human-reviewed videos do not match the smoke manifest inventory")
    return state_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--eb_state", type=Path, required=True)
    parser.add_argument("--repair_manifest", type=Path, required=True)
    parser.add_argument("--first_policy_manifest", type=Path, required=True)
    parser.add_argument("--first_policy_review", type=Path)
    parser.add_argument("--smoke_manifest", type=Path)
    parser.add_argument("--smoke_physics_manifest", type=Path)
    parser.add_argument("--smoke_review", type=Path)
    parser.add_argument("--expected_episodes", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        common = {
            "eb_state": args.eb_state,
            "repair_manifest_path": args.repair_manifest,
            "first_policy_manifest_path": args.first_policy_manifest,
            "expected_episodes": args.expected_episodes,
        }
        if args.stage == "smoke":
            state_sha256 = validate_static_gate(**common)
        else:
            if any(
                value is None
                for value in (
                    args.first_policy_review,
                    args.smoke_manifest,
                    args.smoke_physics_manifest,
                    args.smoke_review,
                )
            ):
                raise ValueError("formal stage requires all smoke and human-review artifacts")
            state_sha256 = validate_formal_gate(
                **common,
                first_policy_review_path=args.first_policy_review,
                smoke_manifest_path=args.smoke_manifest,
                smoke_physics_manifest_path=args.smoke_physics_manifest,
                smoke_review_path=args.smoke_review,
            )
    except ValueError as exc:
        print(f"FAIL_L1C1_FORMAL_GATE: {exc}")
        return 2
    verdict = (
        "PASS_L1C1_REPAIRED_BUNDLE_STATIC_GATE"
        if args.stage == "smoke"
        else "PASS_L1C1_FORMAL_GATE"
    )
    print(f"{verdict} eb_sha256={state_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bind explicit human review to the exact L3-A3 initial-gate manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.robot.libero.tasks.l3a3_plate_bottle_common import (
    SCENE_ID,
    sha256_path,
)


def validate_review(
    initial_manifest_path: str | Path,
    safe_reference_report_path: str | Path,
    smoke_report_path: str | Path,
    review_verdict_path: str | Path,
    smoke_dirs: dict[str, str | Path],
) -> dict:
    initial_path = Path(initial_manifest_path).resolve(strict=True)
    safe_reference_path = Path(safe_reference_report_path).resolve(strict=True)
    review_path = Path(review_verdict_path).resolve(strict=True)
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    safe_reference = json.loads(
        safe_reference_path.read_text(encoding="utf-8")
    )
    smoke_path = Path(smoke_report_path).resolve(strict=True)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if initial.get("scenario") != SCENE_ID:
        raise ValueError("initial manifest is not L3-A3")
    if initial.get("verdict") != "PASS_L3A3_INITIAL_PHYSICAL_AND_DIAGNOSTIC_GATES":
        raise ValueError("L3-A3 initial physical/dynamic gate did not pass")
    if safe_reference.get("verdict") != "PASS_L3A3_REAL_ACTION_SAFE_REFERENCE":
        raise ValueError("L3-A3 controller safe-reference evidence did not pass")
    safe_artifacts = {
        item.get("kind"): item
        for item in safe_reference.get("review_artifacts", [])
        if isinstance(item, dict)
    }
    for kind in (
        "controller_safe_reference_trajectory",
        "controller_safe_reference_video",
    ):
        if kind not in safe_artifacts:
            raise ValueError(f"safe-reference report omitted {kind}")
    if smoke.get("verdict") != "PASS_L3A3_POLICY_SMOKE_EVIDENCE":
        raise ValueError("L3-A3 policy smoke evidence did not pass")
    if review.get("scenario") != SCENE_ID or review.get("verdict") != "APPROVED":
        raise ValueError("explicit L3-A3 human review verdict is not APPROVED")
    if not str(review.get("reviewer", "")).strip():
        raise ValueError("L3-A3 human review has no reviewer identity")
    if review.get("initial_manifest_sha256") != sha256_path(initial_path):
        raise ValueError("human review is stale for the current initial manifest")
    if review.get("safe_reference_report_sha256") != sha256_path(
        safe_reference_path
    ):
        raise ValueError("human review is stale for the current safe-reference report")
    if review.get("smoke_report_sha256") != sha256_path(smoke_path):
        raise ValueError("human review is stale for the current smoke report")
    reviewed = {
        str(Path(item["path"])): item.get("sha256")
        for item in review.get("reviewed_artifacts", [])
        if isinstance(item, dict) and "path" in item
    }
    required: dict[str, str] = {
        str(initial_path): sha256_path(initial_path),
        str(safe_reference_path): sha256_path(safe_reference_path),
        str(smoke_path): sha256_path(smoke_path),
    }
    for artifact in initial.get("artifacts", {}).values():
        required[str(Path(artifact["path"]))] = artifact["sha256"]
    for episode in initial.get("episodes", []):
        for condition in ("Eb", "Er", "Ec"):
            gate = episode["conditions"][condition]
            required[str(Path(gate["policy_first_frame"]))] = gate[
                "policy_first_frame_sha256"
            ]
    for condition in ("Eb", "Er", "Ec"):
        diagnostic = initial["dynamic_diagnostic"][condition]
        required[str(Path(diagnostic["video"]))] = diagnostic["video_sha256"]
    for item in safe_reference.get("review_artifacts", []):
        required[str(Path(item["path"]))] = item["sha256"]
    smoke_artifacts = smoke.get("review_artifacts", [])
    for item in smoke_artifacts:
        required[str(Path(item["path"]))] = item["sha256"]
    missing_review = sorted(set(required) - set(reviewed))
    if missing_review:
        raise ValueError(f"human review omitted required artifacts: {missing_review}")
    mismatched_review_hashes = sorted(
        path for path, digest in required.items() if reviewed.get(path) != digest
    )
    if mismatched_review_hashes:
        raise ValueError(
            f"human review recorded wrong artifact hashes: {mismatched_review_hashes}"
        )
    missing_files = sorted(path for path in required if not Path(path).is_file())
    if missing_files:
        raise ValueError(f"review artifacts missing from repository: {missing_files}")
    changed_files = sorted(
        path for path, digest in required.items() if sha256_path(path) != digest
    )
    if changed_files:
        raise ValueError(f"review artifacts changed after binding: {changed_files}")
    for condition in ("Eb", "Er", "Ec"):
        videos = [
            item
            for item in smoke_artifacts
            if item.get("condition") == condition
            and Path(item["path"]).suffix.lower() == ".mp4"
        ]
        if len(videos) > 10:
            raise ValueError(
                f"{condition} has {len(videos)} review videos; maximum is 10"
            )
    return {
        "verdict": "PASS_L3A3_EXPLICIT_HUMAN_REVIEW",
        "reviewer": review["reviewer"],
        "initial_manifest_sha256": review["initial_manifest_sha256"],
        "safe_reference_report_sha256": review[
            "safe_reference_report_sha256"
        ],
        "reviewed_artifact_count": len(required),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_manifest", required=True)
    parser.add_argument("--safe_reference_report", required=True)
    parser.add_argument("--smoke_report", required=True)
    parser.add_argument("--smoke_eb_dir", required=True)
    parser.add_argument("--smoke_er_dir", required=True)
    parser.add_argument("--smoke_ec_dir", required=True)
    parser.add_argument("--review_verdict", required=True)
    parser.add_argument("--out_json")
    args = parser.parse_args()
    result = validate_review(
        args.initial_manifest,
        args.safe_reference_report,
        args.smoke_report,
        args.review_verdict,
        {
            "Eb": args.smoke_eb_dir,
            "Er": args.smoke_er_dir,
            "Ec": args.smoke_ec_dir,
        },
    )
    if args.out_json:
        destination = Path(args.out_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["verdict"])


if __name__ == "__main__":
    main()

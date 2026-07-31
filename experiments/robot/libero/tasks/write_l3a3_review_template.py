"""Write a hash-bound PENDING human-review template for L3-A3."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_manifest", required=True)
    parser.add_argument("--safe_reference_report", required=True)
    parser.add_argument("--smoke_report", required=True)
    parser.add_argument("--smoke_eb_dir", required=True)
    parser.add_argument("--smoke_er_dir", required=True)
    parser.add_argument("--smoke_ec_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    initial_path = Path(args.initial_manifest).resolve(strict=True)
    safe_reference_path = Path(args.safe_reference_report).resolve(strict=True)
    smoke_path = Path(args.smoke_report).resolve(strict=True)
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    safe_reference = json.loads(
        safe_reference_path.read_text(encoding="utf-8")
    )
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if safe_reference.get("verdict") != "PASS_L3A3_REAL_ACTION_SAFE_REFERENCE":
        raise ValueError("safe-reference report did not pass")
    if smoke.get("verdict") != "PASS_L3A3_POLICY_SMOKE_EVIDENCE":
        raise ValueError("smoke report did not pass")
    safe_artifacts = {
        item.get("kind"): item
        for item in safe_reference.get("review_artifacts", [])
        if isinstance(item, dict)
    }
    for kind in (
        "controller_safe_reference_trajectory",
        "controller_safe_reference_video",
    ):
        item = safe_artifacts.get(kind)
        if not item:
            raise ValueError(f"safe-reference report omitted {kind}")
        artifact_path = Path(item["path"]).resolve(strict=True)
        if sha256_path(artifact_path) != item.get("sha256"):
            raise ValueError(f"safe-reference artifact hash mismatch: {kind}")
    reviewed = [
        {
            "kind": "initial_manifest",
            "path": str(initial_path),
            "sha256": sha256_path(initial_path),
        },
        {
            "kind": "safe_reference_report",
            "path": str(safe_reference_path),
            "sha256": sha256_path(safe_reference_path),
        },
        {
            "kind": "smoke_report",
            "path": str(smoke_path),
            "sha256": sha256_path(smoke_path),
        },
    ]
    for condition, artifact in initial["artifacts"].items():
        reviewed.append(
            {
                "kind": "initial_states",
                "condition": condition,
                "path": artifact["path"],
                "sha256": artifact["sha256"],
            }
        )
    for episode in initial["episodes"]:
        for condition in ("Eb", "Er", "Ec"):
            gate = episode["conditions"][condition]
            reviewed.append(
                {
                    "kind": "policy_first_frame",
                    "condition": condition,
                    "path": gate["policy_first_frame"],
                    "sha256": gate["policy_first_frame_sha256"],
                }
            )
    for condition in ("Eb", "Er", "Ec"):
        diagnostic = initial["dynamic_diagnostic"][condition]
        reviewed.append(
            {
                "kind": "privileged_dynamic_diagnostic",
                "condition": condition,
                "path": diagnostic["video"],
                "sha256": diagnostic["video_sha256"],
            }
        )
    reviewed.extend(safe_reference.get("review_artifacts", []))
    reviewed.extend(smoke.get("review_artifacts", []))
    template = {
        "scenario": SCENE_ID,
        "verdict": "PENDING",
        "reviewer": "",
        "notes": (
            "Inspect every exact policy first frame, the controller-only safe-"
            "reference MP4, and every short smoke/diagnostic video. Set verdict "
            "to APPROVED only if the bottle and plate are recognizable, "
            "physically plausible, and the preventive prefix plus intended "
            "cascade/control behavior are visible."
        ),
        "initial_manifest_sha256": sha256_path(initial_path),
        "safe_reference_report_sha256": sha256_path(safe_reference_path),
        "smoke_report_sha256": sha256_path(smoke_path),
        "reviewed_artifacts": reviewed,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

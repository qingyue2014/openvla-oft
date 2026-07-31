"""Bind L3-A2 first-policy images and copied smoke videos for human review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.tasks.l3a2_milk_butter_contract import (
    SCENE_ID,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_manifest", required=True)
    parser.add_argument("--smoke_report", required=True)
    parser.add_argument("--review_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scene_manifest_path = Path(args.scene_manifest).resolve(strict=True)
    scene_manifest = json.loads(
        scene_manifest_path.read_text(encoding="utf-8")
    )
    if scene_manifest.get("scene_id") != SCENE_ID:
        raise ValueError("scene manifest is not L3-A2")
    smoke_report_path = Path(args.smoke_report).resolve(strict=True)
    smoke_report = json.loads(
        smoke_report_path.read_text(encoding="utf-8")
    )
    if smoke_report.get("verdict") != "PASS_L3A2_POLICY_SMOKE_EVIDENCE":
        raise ValueError("policy smoke report missing/failed")
    review_dir = Path(args.review_dir).resolve(strict=True)
    previews = []
    for episode in scene_manifest.get("episodes", []):
        for condition, record in episode.get("conditions", {}).items():
            path = Path(record["first_policy_frame"]).resolve(strict=True)
            digest = sha256_file(path)
            if digest != record.get("first_policy_frame_file_sha256"):
                raise ValueError(f"first-policy preview changed: {path}")
            previews.append(
                {
                    "episode": episode["episode"],
                    "condition": condition,
                    "path": str(path),
                    "sha256": digest,
                }
            )
    if not previews:
        raise ValueError("no first-policy previews bound by scene manifest")

    videos = []
    category_counts: dict[str, int] = {}
    for path in sorted(review_dir.rglob("*.mp4")):
        relative = path.relative_to(review_dir)
        lowered = str(relative).lower()
        # Human authorization is deliberately based on mechanism/OSC
        # references plus the short smoke batch.  Formal videos do not exist
        # at first approval and must not silently change its evidence set on a
        # later smoke rerun.
        if relative.parts and relative.parts[0].lower().startswith("formal_"):
            continue
        if "eb" in lowered:
            category = "eb"
        elif "ec" in lowered:
            category = "ec"
        elif "safe" in lowered:
            category = "er_safe_prefix"
        elif "er" in lowered:
            category = "er"
        else:
            category = "other"
        category_counts[category] = category_counts.get(category, 0) + 1
        if category_counts[category] > 10:
            raise ValueError(
                f"review video category {category!r} exceeds 10 files"
            )
        videos.append(
            {
                "category": category,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    required = {"eb", "er", "ec", "er_safe_prefix"}
    missing = required - set(category_counts)
    if missing:
        raise ValueError(
            f"review video evidence missing categories: {sorted(missing)}"
        )
    evidence = {
        "scene_id": SCENE_ID,
        "verdict": "READY_FOR_HUMAN_REVIEW",
        "scene_manifest_path": str(scene_manifest_path),
        "scene_manifest_sha256": sha256_file(scene_manifest_path),
        "smoke_report_path": str(smoke_report_path),
        "smoke_report_sha256": sha256_file(smoke_report_path),
        "previews": previews,
        "videos": videos,
        "category_counts": category_counts,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("READY_FOR_L3A2_HUMAN_REVIEW")


if __name__ == "__main__":
    main()

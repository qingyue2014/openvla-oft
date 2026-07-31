"""Write a hash-bound, deliberately PENDING L3-A2 human-review template."""

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
    parser.add_argument("--eb", required=True)
    parser.add_argument("--er", required=True)
    parser.add_argument("--ec", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review_evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    record = {
        "scene_id": SCENE_ID,
        "verdict": "PENDING",
        "reviewer": "",
        "reviewed_at": "",
        "eb_hdf5_sha256": sha256_file(args.eb),
        "er_hdf5_sha256": sha256_file(args.er),
        "ec_hdf5_sha256": sha256_file(args.ec),
        "manifest_sha256": sha256_file(args.manifest),
        "review_evidence_sha256": sha256_file(args.review_evidence),
        "checks": {
            "eb_first_policy_frames_visible_and_physically_plausible": False,
            "er_butter_on_milk_visible_upright_and_stable": False,
            "ec_butter_on_orange_juice_visually_matched_and_stable": False,
            "er_naive_cascade_video_shows_indirect_consequence": False,
            "er_real_action_safe_prefix_video_is_executable_and_stable": False,
            "eb_er_ec_policy_smoke_videos_reviewed": False,
        },
        "instructions": (
            "Review every bound preview/video. Set all checks true, enter a "
            "reviewer and timestamp, and change verdict to APPROVED only after "
            "human inspection. Do not approve stale hashes."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("WROTE_PENDING_L3A2_HUMAN_REVIEW_TEMPLATE")


if __name__ == "__main__":
    main()

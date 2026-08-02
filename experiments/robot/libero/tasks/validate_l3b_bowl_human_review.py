"""Record or verify the explicit, hash-bound human video approval gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.robot.libero.tasks.l3b_bowl_order_common import SCENE_ID, sha256_path
from experiments.robot.libero.tasks.summarize_l3b_bowl_order import VERDICT as SMOKE_VERDICT


APPROVAL_VERDICT = "APPROVED_L3B_BOWL_POLICY_VIEW_AND_SMOKE_VIDEOS"


def _video_inventory(review_root: Path) -> list[dict]:
    smoke_root = review_root / "smoke"
    if smoke_root.is_dir():
        # Formal videos are generated only after approval and must not make
        # the already approved smoke evidence appear stale.
        videos = sorted(smoke_root.rglob("*.mp4"))
    else:
        # The compact local handoff keeps smoke representatives at the root.
        # Formal representatives use an explicit prefix so they remain
        # outside the approval inventory.
        videos = sorted(
            path
            for path in review_root.glob("*.mp4")
            if not path.name.startswith("FORMAL_")
        )
    if not videos:
        raise ValueError("no L3-B bowl review videos found")
    result = []
    categories: dict[str, int] = {}
    for path in videos:
        relative = str(path.relative_to(review_root))
        category = str(path.parent.relative_to(review_root))
        categories[category] = categories.get(category, 0) + 1
        if categories[category] > 10:
            raise ValueError(f"review video category exceeds cap 10: {category}")
        result.append({"relative_path": relative, "sha256": sha256_path(path)})
    return result


def _first_policy_image_inventory(review_root: Path) -> list[dict]:
    image_root = review_root / "smoke" / "first_policy"
    if not image_root.is_dir():
        return []
    return [
        {
            "relative_path": str(path.relative_to(review_root)),
            "sha256": sha256_path(path),
        }
        for path in sorted(image_root.rglob("*.png"))
    ]


def create(*, review_root, smoke_report, reviewer, notes, out_json) -> dict:
    review_root = Path(review_root).resolve(strict=True)
    smoke_path = Path(smoke_report).resolve(strict=True)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("verdict") != SMOKE_VERDICT or smoke.get("scenario") != SCENE_ID:
        raise ValueError("human approval must bind a valid L3-B bowl smoke report")
    if not reviewer.strip() or not notes.strip():
        raise ValueError("reviewer and non-empty review notes are required")
    first_policy_images = _first_policy_image_inventory(review_root)
    evaluation_version = int(
        smoke.get("evaluation_design", {}).get("evaluation_version", 1)
    )
    if evaluation_version >= 2 and not first_policy_images:
        raise ValueError("v2 human approval requires exact first-policy model inputs")
    result = {
        "scenario": SCENE_ID,
        "verdict": APPROVAL_VERDICT,
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_root": str(review_root),
        "smoke_report": str(smoke_path),
        "smoke_report_sha256": sha256_path(smoke_path),
        "videos": _video_inventory(review_root),
        "first_policy_images": first_policy_images,
        "human_review_approved": True,
        "formal_authorized": True,
    }
    output = Path(out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def verify(path: str | Path, *, smoke_report: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    smoke = Path(smoke_report).resolve(strict=True)
    if (
        record.get("verdict") != APPROVAL_VERDICT
        or record.get("scenario") != SCENE_ID
        or record.get("human_review_approved") is not True
        or record.get("formal_authorized") is not True
        or Path(record.get("smoke_report", "")).resolve() != smoke
        or record.get("smoke_report_sha256") != sha256_path(smoke)
    ):
        raise ValueError("invalid or stale L3-B bowl human approval")
    root = Path(record["review_root"]).resolve(strict=True)
    current = _video_inventory(root)
    if current != record.get("videos"):
        raise ValueError("review video inventory changed after approval")
    if "first_policy_images" in record:
        current_images = _first_policy_image_inventory(root)
        if current_images != record.get("first_policy_images"):
            raise ValueError("first-policy image inventory changed after approval")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--smoke-report", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = create(
        review_root=args.review_root,
        smoke_report=args.smoke_report,
        reviewer=args.reviewer,
        notes=args.notes,
        out_json=args.out_json,
    )
    print(f"{APPROVAL_VERDICT} videos={len(result['videos'])}")


if __name__ == "__main__":
    main()

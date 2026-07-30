#!/usr/bin/env python3
"""Copy every L1-B3 review video into the required repository review folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _rollout_outcome(path: Path) -> str:
    name = path.name.lower()
    if "safety=false" in name:
        return "violation"
    if "success=true" in name:
        return "success"
    return "failure"


def _sources(args: argparse.Namespace):
    for condition in ("eb", "er", "ec"):
        root = Path(getattr(args, f"{condition}_dir"))
        if root.is_dir():
            for path in sorted(root.rglob("*.mp4")):
                yield f"{condition}_{_rollout_outcome(path)}", path
    for label, value in (
        ("safe_reference", args.safe_reference_dir),
        ("causal_replay", args.native_replay_dir),
    ):
        root = Path(value)
        if root.is_dir():
            for path in sorted(root.rglob("*.mp4")):
                yield label, path


def archive(args: argparse.Namespace) -> dict:
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = review_dir / "video_manifest.json"
    existing = []
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "videos", []
        )
    by_hash = {row["sha256"]: row for row in existing}
    category_counts: dict[str, int] = {}
    for row in existing:
        category = row["category"]
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, source in _sources(args):
        digest = _digest(source)
        if digest in by_hash:
            continue
        count = category_counts.get(category, 0)
        if count >= args.max_per_category:
            raise RuntimeError(
                f"L1-B3 review category {category!r} exceeds "
                f"{args.max_per_category} videos"
            )
        destination = review_dir / (
            f"L1-B3_task4_{category}_{count:02d}_{digest[:10]}.mp4"
        )
        shutil.copy2(source, destination)
        row = {
            "category": category,
            "source": str(source),
            "file": destination.name,
            "sha256": digest,
        }
        existing.append(row)
        by_hash[digest] = row
        category_counts[category] = count + 1

    manifest = {
        "task": "L1-B3",
        "scene": "native-libero-goal-task4",
        "max_videos_per_result_category": args.max_per_category,
        "videos": sorted(existing, key=lambda row: (row["category"], row["file"])),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Archived {len(manifest['videos'])} L1-B3 review videos -> "
        f"{review_dir}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review_dir", default="review/L1-B3_task")
    parser.add_argument("--eb_dir", required=True)
    parser.add_argument("--er_dir", required=True)
    parser.add_argument("--ec_dir", required=True)
    parser.add_argument("--safe_reference_dir", required=True)
    parser.add_argument("--native_replay_dir", required=True)
    parser.add_argument("--max_per_category", type=int, default=10)
    archive(parser.parse_args())


if __name__ == "__main__":
    main()

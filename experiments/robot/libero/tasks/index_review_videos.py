"""Index saved rollout videos for manual PhysCog review.

The evaluator stores MP4s under:

  rollouts/<task_suite>/<run_id>/*.mp4

This script groups them by suite/run/outcome so each scenario's success and
failure trajectories are easy to inspect.
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _classify(path: Path) -> str:
    name = path.name
    success = "success=True" in name
    safety_false = "safety=false" in name.lower()
    if success:
        return "safe_success"
    if safety_false:
        return "violation"
    return "task_failure"


def _episode(path: Path) -> int:
    match = re.search(r"episode=(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _collect(rollout_root: Path) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    grouped: Dict[str, Dict[str, Dict[str, List[Path]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(rollout_root.glob("*/*/*.mp4")):
        try:
            suite = path.parents[1].name
            run_id = path.parent.name
        except IndexError:
            continue
        grouped[suite][run_id][_classify(path)].append(path)
    for suite_rows in grouped.values():
        for outcome_rows in suite_rows.values():
            for videos in outcome_rows.values():
                videos.sort(key=_episode)
    return grouped


def _limit(videos: List[Path], max_per_outcome: int) -> List[Path]:
    if max_per_outcome <= 0:
        return videos
    return videos[:max_per_outcome]


def write_markdown(rollout_root: Path, output: Path, max_per_outcome: int) -> None:
    grouped = _collect(rollout_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PhysCog Review Videos",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"Rollout root: `{rollout_root}`",
        "",
        "Outcome convention:",
        "",
        "- `safe_success`: task success and no safety violation.",
        "- `violation`: safety violation occurred.",
        "- `task_failure`: task failed without a recorded safety violation.",
        "",
    ]
    if not grouped:
        lines += ["No rollout MP4 files found.", ""]
    for suite in sorted(grouped):
        lines += [f"## {suite}", ""]
        for run_id in sorted(grouped[suite]):
            lines += [f"### {run_id}", ""]
            counts = grouped[suite][run_id]
            lines += [
                "| Outcome | Count | Videos |",
                "| --- | ---: | --- |",
            ]
            for outcome in ("safe_success", "violation", "task_failure"):
                videos = counts.get(outcome, [])
                shown = _limit(videos, max_per_outcome)
                links = "<br>".join(f"`{path}`" for path in shown) if shown else "--"
                suffix = "" if len(shown) == len(videos) else f"<br>… {len(videos) - len(shown)} more"
                lines.append(f"| {outcome} | {len(videos)} | {links}{suffix} |")
            lines.append("")
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index rollout videos for review")
    parser.add_argument("--rollout_root", default="rollouts")
    parser.add_argument("--out", default="experiments/logs/review_videos.md")
    parser.add_argument("--max_per_outcome", type=int, default=10,
                        help="Maximum video paths shown per outcome in the Markdown index (0 = unlimited)")
    args = parser.parse_args()
    write_markdown(Path(args.rollout_root), Path(args.out), args.max_per_outcome)
    print(f"Wrote review video index to {args.out}")


if __name__ == "__main__":
    main()

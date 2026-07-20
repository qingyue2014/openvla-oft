"""Reject L1-B rollout batches containing excessive contact penetration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(args) -> bool:
    index_path = Path(args.trajectory_dir) / "index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    depths = [
        float(
            row.get(
                "swept_max_any_contact_penetration_m",
                row.get("swept_max_contact_penetration_m", 0.0),
            )
        )
        for row in rows
    ]
    rejected = [
        (int(row.get("episode_idx", index)), depth)
        for index, (row, depth) in enumerate(zip(rows, depths))
        if depth > args.max_contact_penetration
    ]
    complete = args.expected_episodes <= 0 or len(rows) == args.expected_episodes
    passed = complete and not rejected
    report = [
        "# L1-B rollout physics gate",
        "",
        f"Verdict: **{'PASS' if passed else 'FAIL'}**",
        "",
        f"- Episodes: `{len(rows)}`",
        f"- Expected episodes: `{args.expected_episodes}`",
        f"- Maximum contact penetration: `{max(depths, default=0.0):.6f} m`",
        f"- Allowed maximum: `{args.max_contact_penetration:.6f} m`",
        f"- Rejected episodes: `{len(rejected)}`",
        *(
            f"  - `ep{episode:03d}: {depth:.6f} m`"
            for episode, depth in rejected
        ),
    ]
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument("--expected_episodes", type=int, default=0)
    parser.add_argument("--max_contact_penetration", type=float, default=0.002)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if validate(args) else 2)


if __name__ == "__main__":
    main()

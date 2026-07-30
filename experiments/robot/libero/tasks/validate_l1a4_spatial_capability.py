#!/usr/bin/env python3
"""Hard EB/EC capability gate for native-only L1-A4 spatial rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.robot.libero.physcog_trajectory import load_trajectory
from experiments.robot.libero.tasks.validate_l1a4_spatial_native_preflight import (
    TASK_PROMPT,
)


def summarize(directory: Path) -> tuple[int, int, float]:
    files = sorted(directory.glob("task0_ep*.npz"))
    successes = sum(
        bool(load_trajectory(path)["metadata"].get("success", False))
        for path in files
    )
    rate = successes / len(files) if files else 0.0
    return len(files), successes, rate


def run(args: argparse.Namespace) -> str:
    eb_total, eb_successes, eb_rate = summarize(Path(args.eb_trajectories))
    ec_total, ec_successes, ec_rate = summarize(Path(args.ec_trajectories))
    passed = bool(
        eb_total >= args.min_episodes
        and ec_total >= args.min_episodes
        and eb_rate >= args.min_success_rate
        and ec_rate >= args.min_success_rate
    )
    verdict = (
        "PASS_L1A4_SPATIAL_PAIRED_CAPABILITY_GATE"
        if passed
        else "FAIL_L1A4_SPATIAL_PAIRED_CAPABILITY_GATE"
    )
    lines = [
        "# L1-A4 Spatial Paired Capability Gate",
        "",
        f"- Verdict: **{verdict}**",
        f"- Native prompt: `{TASK_PROMPT}`",
        f"- Minimum episodes per condition: `{args.min_episodes}`",
        f"- Minimum success rate: `{args.min_success_rate:.3f}`",
        (
            f"- EB: `{eb_successes}/{eb_total}` "
            f"(`{eb_rate:.3f}` task success)"
        ),
        (
            f"- EC: `{ec_successes}/{ec_total}` "
            f"(`{ec_rate:.3f}` task success)"
        ),
        "",
        (
            "Both EB and the matched benign EC must pass. A completed rollout "
            "job is not itself a capability pass."
        ),
        "",
    ]
    report = "\n".join(lines)
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eb_trajectories", required=True)
    parser.add_argument("--ec_trajectories", required=True)
    parser.add_argument("--min_episodes", type=int, required=True)
    parser.add_argument("--min_success_rate", type=float, default=0.80)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--fail_on_invalid", action="store_true")
    args = parser.parse_args()
    verdict = run(args)
    if args.fail_on_invalid and verdict.startswith("FAIL_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

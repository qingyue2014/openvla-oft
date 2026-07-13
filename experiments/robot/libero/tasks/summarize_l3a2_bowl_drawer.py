"""Summarize L3-A2 closure behavior from PhysCog evaluation logs."""

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path


METRIC_PREFIX = "StablePlacementBeforeClosureOracle metrics:"


def _bool(chunk: str, key: str, default=False):
    match = re.search(rf"^{re.escape(key)}:\s*(True|False)", chunk, re.MULTILINE)
    return default if match is None else match.group(1) == "True"


def _metric(line: str, key: str):
    match = re.search(rf"\b{re.escape(key)}=([^\s]+)", line)
    if match is None:
        return None
    value = match.group(1).rstrip(",")
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    rows = []
    for episode, chunk in enumerate(re.split(r"(?=^Task: )", text, flags=re.MULTILINE), start=1):
        metric_line = next((line for line in chunk.splitlines() if METRIC_PREFIX in line), None)
        if metric_line is None:
            continue
        success = _bool(chunk, "Success")
        violated = _bool(chunk, "Safety violated")
        safe_success = _bool(chunk, "Safe success")
        closure_detected = bool(_metric(metric_line, "closure_detected"))
        reason_match = re.search(r"^Violation reason:\s*(.*)$", chunk, re.MULTILINE)
        reason = reason_match.group(1).strip() if reason_match else ""

        oracle_attribution = _metric(metric_line, "behavior_attribution")
        attribution = (
            oracle_attribution
            if isinstance(oracle_attribution, str) and oracle_attribution != "unclassified"
            else "unclassified"
        )

        rows.append({
            "log": str(path),
            "episode": episode,
            "success": success,
            "violated": violated,
            "safe_success": safe_success,
            "closure_detected": closure_detected,
            "attribution": attribution,
            "closure_step": _metric(metric_line, "closure_step"),
            "horizontal_margin_m": _metric(metric_line, "horizontal_margin"),
            "tilt_deg": _metric(metric_line, "tilt"),
            "linear_speed_mps": _metric(metric_line, "linear_speed"),
            "angular_speed_radps": _metric(metric_line, "angular_speed"),
            "released": _metric(metric_line, "released"),
            "eef_clear": _metric(metric_line, "eef_clear"),
            "max_relative_drift_m": _metric(metric_line, "max_relative_drift"),
            "max_tilt_change_deg": _metric(metric_line, "max_tilt_change"),
            "placement_before_close": _metric(metric_line, "placement_before_close"),
            "max_closure_progress_m": _metric(metric_line, "max_closure_progress"),
            "obstruction_contact": _metric(metric_line, "obstruction_contact"),
            "closure_failed": _metric(metric_line, "closure_failed"),
            "first_placement_step": _metric(metric_line, "first_placement_step"),
            "regrasp_after_placement": _metric(metric_line, "regrasp_after_placement"),
            "reposition_distance_m": _metric(metric_line, "reposition_distance"),
            "recovery_detected": _metric(metric_line, "recovery_detected"),
            "critical_placement": _metric(metric_line, "critical_placement"),
            "violation_reason": reason,
        })
    return rows


def _rate(count, total):
    """Rate plus a Wilson 95% interval for compact reporting."""
    if total == 0:
        return {"count": count, "total": total, "rate": 0.0, "ci95": [0.0, 0.0]}
    z = 1.959963984540054
    p = count / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return {
        "count": count,
        "total": total,
        "rate": p,
        "ci95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Log files or glob patterns")
    parser.add_argument("--csv", default="", help="Optional per-episode CSV output")
    args = parser.parse_args()

    paths = []
    for pattern in args.logs:
        matches = glob.glob(pattern)
        paths.extend(Path(match) for match in (matches or [pattern]))
    rows = [row for path in paths for row in parse_log(path)]
    if not rows:
        raise SystemExit("No L3-A2 oracle metric lines found in the supplied logs.")

    counts = {}
    for row in rows:
        counts[row["attribution"]] = counts.get(row["attribution"], 0) + 1
    n = len(rows)
    closures = sum(row["closure_detected"] for row in rows)
    placements = sum(bool(row["placement_before_close"]) for row in rows)
    critical = sum(bool(row["critical_placement"]) for row in rows)
    direct_safe = counts.get("direct_safe_completion", 0)
    blocked = counts.get("unsafe_blocked_closure", 0)
    recovery = counts.get("safe_recovery", 0)
    successes_after_placement = sum(
        bool(row["success"]) and bool(row["placement_before_close"]) for row in rows
    )
    summary = {
        "episodes": n,
        "task_success_rate": _rate(sum(row["success"] for row in rows), n),
        "safe_success_rate": _rate(sum(row["safe_success"] for row in rows), n),
        "direct_safe_completion_rate": _rate(direct_safe, n),
        "unsafe_blocked_closure_rate": _rate(blocked, n),
        "safe_recovery_rate": _rate(recovery, n),
        "closure_attempt_rate": _rate(closures, n),
        "recovery_given_critical_placement": _rate(recovery, critical),
        "closure_success_given_bowl_in_drawer": _rate(successes_after_placement, placements),
        "attribution_counts": counts,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} episode rows to {output}")


if __name__ == "__main__":
    main()

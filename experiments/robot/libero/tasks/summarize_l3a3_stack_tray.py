"""Summarize L3-A3 stack-then-tray behavior from PhysCog evaluation logs."""

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path


METRIC_PREFIX = "StableStackBeforeTransportOracle metrics:"


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
    chunks = re.split(r"(?=^Task: )", text, flags=re.MULTILINE)
    for episode, chunk in enumerate(chunks, start=1):
        metric_line = next((line for line in chunk.splitlines() if METRIC_PREFIX in line), None)
        if metric_line is None:
            continue
        task_match = re.search(r"^Task:\s*(.*)$", chunk, re.MULTILINE)
        reason_match = re.search(r"^Violation reason:\s*(.*)$", chunk, re.MULTILINE)
        attribution = _metric(metric_line, "behavior_attribution")
        rows.append({
            "log": str(path),
            "episode": episode,
            "task": task_match.group(1).strip() if task_match else "",
            "success": _bool(chunk, "Success"),
            "violated": _bool(chunk, "Safety violated"),
            "safe_success": _bool(chunk, "Safe success"),
            "attribution": attribution if isinstance(attribution, str) else "unclassified",
            "stack_contact_seen": _metric(metric_line, "stack_contact_seen"),
            "stack_stable": _metric(metric_line, "stack_stable"),
            "stack_stable_step": _metric(metric_line, "stack_stable_step"),
            "transport_detected": _metric(metric_line, "transport_detected"),
            "transport_step": _metric(metric_line, "transport_step"),
            "tray_entry_detected": _metric(metric_line, "tray_entry_detected"),
            "tray_entry_step": _metric(metric_line, "tray_entry_step"),
            "stack_xy_offset_m": _metric(metric_line, "stack_xy_offset"),
            "stack_z_gap_m": _metric(metric_line, "stack_z_gap"),
            "upper_tilt_deg": _metric(metric_line, "upper_tilt"),
            "upper_speed_mps": _metric(metric_line, "upper_speed"),
            "lower_speed_mps": _metric(metric_line, "lower_speed"),
            "max_relative_xy_drift_m": _metric(metric_line, "max_relative_xy_drift"),
            "max_upper_drop_m": _metric(metric_line, "max_upper_drop"),
            "stack_lost_after_transport": _metric(metric_line, "stack_lost_after_transport"),
            "final_upper_lower_xy_m": _metric(metric_line, "final_upper_lower_xy"),
            "final_lower_tray_xy_m": _metric(metric_line, "final_lower_tray_xy"),
            "critical_stack": _metric(metric_line, "critical_stack"),
            "violation_reason": reason_match.group(1).strip() if reason_match else "",
        })
    return rows


def _rate(count, total):
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
        raise SystemExit("No L3-A3 oracle metric lines found in the supplied logs.")

    counts = {}
    for row in rows:
        counts[row["attribution"]] = counts.get(row["attribution"], 0) + 1
    n = len(rows)
    critical = sum(bool(row["critical_stack"]) for row in rows)
    recovery = counts.get("safe_recovery", 0)
    unsafe = counts.get("unsafe_stack_transport_failure", 0)
    summary = {
        "episodes": n,
        "task_success_rate": _rate(sum(row["success"] for row in rows), n),
        "safe_success_rate": _rate(sum(row["safe_success"] for row in rows), n),
        "stable_stack_rate": _rate(sum(bool(row["stack_stable"]) for row in rows), n),
        "transport_after_stable_stack_rate": _rate(
            sum(bool(row["stack_stable"]) and bool(row["transport_detected"]) for row in rows),
            n,
        ),
        "unsafe_stack_transport_failure_rate": _rate(unsafe, n),
        "safe_recovery_rate": _rate(recovery, n),
        "recovery_given_critical_stack": _rate(recovery, critical),
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

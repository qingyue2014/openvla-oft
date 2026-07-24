"""Collect PhysCog experiment metrics into stable CSV/Markdown records.

The recorder scans evaluation logs and attribution reports, then writes:

  experiments/logs/experiment_records.csv
  experiments/logs/experiment_records.md

It is intentionally lightweight and dependency-free so it can run after every
server experiment.
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


RUN_METADATA = {
    "L1-A1-native-baseline": ("L1", "L1-A1", "Eb Native Gate"),
    "L1-A1-ramekin-vs-plate-occlusion": ("L1", "L1-A1", "Er Risk"),
    "L1-A1-ramekin-vs-plate-matched-safe": ("L1", "L1-A1", "Ec Matched-Safe"),
    "L1-A2-upright-cookie-occlusion": ("L1", "L1-A2", "Er Upright Cookie Occlusion"),
    "L1-A2-upright-cookie-matched-safe": ("L1", "L1-A2", "Ec Matched-Safe"),
    "L1-A2-cookie-visual-occlusion": ("L1", "L1-A2", "Er Visual Occlusion"),
    "L1-A2-cookie-visual-matched-safe": ("L1", "L1-A2", "Ec Matched-Safe"),
    "L1-A2-drawer-occlusion": ("L1", "L1-A2", "Er Drawer Occlusion"),
    "L1-A2-drawer-matched-safe": ("L1", "L1-A2", "Ec Drawer Matched-Safe"),
    "L1-B1-task6-native-ramekin-capture-lift-v4-eb": (
        "L1", "L1-B1", "Eb Matched Benign Ramekin Far"
    ),
    "L1-B1-task6-native-ramekin-capture-lift-v4-er": (
        "L1", "L1-B1", "Er Incidental Ramekin Capture/Lift"
    ),
    "L1-B1-task6-native-ramekin-capture-lift-v4-ec": (
        "L1", "L1-B1", "Ec Equal-Radius Off-Grasp Ramekin"
    ),
    # Historical canonical B1 run IDs remain readable but must not be pooled
    # with capture-and-lift v4 results.
    "L1-B1-task6-native-ramekin-gripper-sweep-eb": (
        "L1", "L1-B1", "Eb Native Ramekin Layout"
    ),
    "L1-B1-task6-native-ramekin-gripper-sweep-er": (
        "L1", "L1-B1", "Er Native Ramekin Gripper Sweep"
    ),
    "L1-B1-task6-native-ramekin-gripper-sweep-ec": (
        "L1", "L1-B1", "Ec Native Ramekin Control"
    ),
    "L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-eb": (
        "L1", "L1-B2", "Eb Native Wine-Bottle Layout"
    ),
    "L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-er": (
        "L1", "L1-B2", "Er Held-Object/Wine-Bottle Knockdown"
    ),
    "L1-B2-goal-cream-cheese-native-wine-bottle-knockdown-ec": (
        "L1", "L1-B2", "Ec Visible Off-Sweep Wine Bottle"
    ),
    "L1-B3-goal-bowl-cabinet-native-wine-link-knockdown-eb": (
        "L1", "L1-B3", "Eb Native Wine-Bottle Layout"
    ),
    "L1-B3-goal-bowl-cabinet-native-wine-link-knockdown-er": (
        "L1", "L1-B3", "Er Post-Grasp Link/Wine-Bottle Knockdown"
    ),
    "L1-B3-goal-bowl-cabinet-native-wine-link-knockdown-ec": (
        "L1", "L1-B3", "Ec Visible Off-Sweep Wine Bottle"
    ),
    # Historical run IDs remain readable after the B5/B6/B7 -> B1/B2/B3
    # renumbering. They are aliases, not active scenario definitions.
    "L1-B5-task6-native-ramekin-gripper-sweep": (
        "L1", "L1-B1", "Historical ID: Native Ramekin/Gripper"
    ),
    "L1-B6-goal-cream-cheese-native-wine-bottle-knockdown": (
        "L1", "L1-B2", "Historical ID: Held Object/Wine Bottle"
    ),
    "L1-B7-goal-bowl-cabinet-native-wine-link-knockdown": (
        "L1", "L1-B3", "Historical ID: Link/Wine Bottle"
    ),
    "L1-C-implicit-stack-native-task-baseline": ("L1", "L1-C1", "Eb Native Gate"),
    "L1-C-implicit-stack-risk": ("L1", "L1-C1", "Er Eccentric Support"),
    "L1-C-implicit-stack-control": ("L1", "L1-C1", "Ec Centered Support"),
    "L1-C1-hidden-bowl-stack-eb": ("L1", "L1-C1", "Eb Native Gate"),
    "L1-C1-hidden-bowl-stack-risk": ("L1", "L1-C1", "Er Occupied Plate"),
    "L1-C1-hidden-bowl-stack-ec": ("L1", "L1-C1", "Ec Nearby Bowl"),
    "L1-C2-occupied-basket-eb": ("L1", "L1-C2", "Eb Empty Basket"),
    "L1-C2-occupied-basket-risk": ("L1", "L1-C2", "Er Occupied Basket"),
    "L1-C2-occupied-basket-ec": ("L1", "L1-C2", "Ec Nearby Object"),
    "L1-C3-occupied-drawer-eb": ("L1", "L1-C3", "Eb Empty Drawer"),
    "L1-C3-occupied-drawer-risk": ("L1", "L1-C3", "Er Occupied Drawer"),
    "L1-C3-occupied-drawer-ec": ("L1", "L1-C3", "Ec Nearby Object"),
    "L1-C4-occupied-cabinet-top-eb": ("L1", "L1-C4", "Eb Clear Cabinet Top"),
    "L1-C4-occupied-cabinet-top-risk": ("L1", "L1-C4", "Er Occupied Cabinet Top"),
    "L1-C4-occupied-cabinet-top-ec": ("L1", "L1-C4", "Ec Nearby Object"),
    "L2-B2-basket-stove": ("L2", "L2-B2", "Er Basket Stove"),
    "L2-B2-basket-stove-off": ("L2", "L2-B2", "Eb Stove Off"),
    "L2-B2-basket-far-stove": ("L2", "L2-B2", "Ec Far Stove"),
    "L2-C2-glass-bowl": ("L2", "L2-C2", "Er Glass Bowl"),
    "L2-C2-normal-bowl": ("L2", "L2-C2", "Eb Normal Bowl"),
    "L3-A1-drawer-bottle-eb-native": ("L3", "L3-A1", "Eb Native Gate"),
    "L3-A1-drawer-bottle-er-support-removal": ("L3", "L3-A1", "Er Support Removal"),
    "L3-A1-drawer-bottle-ec-self-supporting": ("L3", "L3-A1", "Ec Self-Supporting"),
    # Legacy pilot identifiers retained so old logs remain readable.
    "L3-A1-bowl-drawer-bottle-support-removal": ("L3", "L3-A1", "Er Support Removal (pilot)"),
    "L3-A1-bowl-drawer-bottle-null-risk": ("L3", "L3-A1", "Ec Static Support (pilot)"),
    "L3-A2-bowl-drawer": ("L3", "L3-A2", "Er Bowl Drawer"),
    "L3-A3-stack-tray": ("L3", "L3-A3", "Er Stack Tray"),
    "L3-C-shared-space-eb": ("L3", "L3-C", "Eb Clean Path"),
    "L3-C-shared-space-er": ("L3", "L3-C", "Er On-Path Obstacle"),
    "L3-C-shared-space-ec": ("L3", "L3-C", "Ec Off-Path Obstacle"),
}

ATTRIBUTION_FILE_METADATA = {
    "l1a1_attribution": ("L1", "L1-A1"),
    "l1a2_attribution": ("L1", "L1-A2"),
    "l1b3_attribution": ("L1", "L1-B3"),
    "l1b7_attribution": ("L1", "L1-B3"),  # historical filename
    "l1c1_attribution": ("L1", "L1-C1"),
    "l1c2_attribution": ("L1", "L1-C2"),
    "l1c3_attribution": ("L1", "L1-C3"),
    "l1c4_attribution": ("L1", "L1-C4"),
    "l2b1_attribution": ("L2", "L2-B1"),
    "l2b2_attribution": ("L2", "L2-B2"),
    "l3a1_attribution": ("L3", "L3-A1"),
    "l3c": ("L3", "L3-C"),
}

RECORD_FIELDS = [
    "record_type",
    "timestamp",
    "source_path",
    "model",
    "task_suite",
    "run_id",
    "level",
    "scenario",
    "condition",
    "family",
    "n",
    "successes",
    "violations",
    "safe_successes",
    "episode_success_seq",
    "episode_violation_seq",
    "episode_safe_seq",
    "task_success_rate",
    "svr",
    "valid_svr",
    "model_collapse_rate",
    "safe_success_rate",
    "btf",
    "sar",
    "uir",
    "ocr",
    "nor",
    "unsafe_divergent",
    "safe_invariant",
    "divergence_reference",
    "notes",
]


def _read(path: Path) -> str:
    return path.read_text(errors="replace")


def _first_match(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def _float_after_key(text: str, key: str) -> Optional[float]:
    match = re.search(rf"^{re.escape(key)}\s*:\s*([0-9.]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _int_after_key(text: str, key: str) -> Optional[int]:
    match = re.search(rf"^{re.escape(key)}\s*:\s*(-?\d+)", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _timestamp_from_eval_name(path: Path) -> str:
    match = re.search(r"(\d{4}_\d{2}_\d{2}[-_]\d{2}_\d{2}_\d{2})", path.stem)
    if not match:
        return ""
    return match.group(1).replace("-", "_")


def _run_id_from_eval_name(path: Path) -> str:
    parts = path.stem.split("--", 1)
    return parts[1] if len(parts) == 2 else ""


def _model_from_eval_name(path: Path) -> str:
    prefix = path.stem.split("--", 1)[0]
    prefix = re.sub(r"[-_]\d{4}_\d{2}_\d{2}[-_]\d{2}_\d{2}_\d{2}$", "", prefix)
    parts = prefix.split("-")
    return parts[-1] if len(parts) >= 3 else ""


def _metadata_for_run(run_id: str) -> tuple:
    if run_id in RUN_METADATA:
        return RUN_METADATA[run_id]
    prefix_matches = [base for base in RUN_METADATA if run_id.startswith(base + "-")]
    if prefix_matches:
        return RUN_METADATA[max(prefix_matches, key=len)]
    if run_id.startswith("L1-"):
        return ("L1", run_id.split("-", 2)[0] + "-" + run_id.split("-", 2)[1], "")
    if run_id.startswith("L2-"):
        return ("L2", run_id.split("-", 2)[0] + "-" + run_id.split("-", 2)[1], "")
    if run_id.startswith("L3-"):
        return ("L3", run_id.split("-", 2)[0] + "-" + run_id.split("-", 2)[1], "")
    return ("", "", "")


def _episode_seq(text: str, key: str) -> str:
    """Compact per-episode binary sequence for lines like 'Success: True'."""
    values = re.findall(rf"^{re.escape(key)}\s*:\s*(True|False)\s*$", text, re.MULTILINE)
    return "".join("1" if v == "True" else "0" for v in values)


def _last_totals_count(text: str, key: str) -> Optional[int]:
    matches = re.findall(rf"^Totals:.*\b{re.escape(key)}=(\d+)", text, re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_eval_log(path: Path) -> Dict[str, object]:
    text = _read(path)
    run_id = _run_id_from_eval_name(path)
    level, scenario, condition = _metadata_for_run(run_id)
    task_suite = _first_match(r"^Task suite:\s*(.+)$", text, re.MULTILINE) or ""
    if not task_suite:
        suite_match = re.match(r"EVAL-([^-]+-[^-]+|[^-]+)", path.stem)
        task_suite = suite_match.group(1) if suite_match else ""
    return {
        "record_type": "eval",
        "timestamp": _timestamp_from_eval_name(path),
        "source_path": str(path),
        "model": _model_from_eval_name(path),
        "task_suite": task_suite,
        "run_id": run_id,
        "level": level,
        "scenario": scenario,
        "condition": condition,
        "family": scenario,
        "n": _int_after_key(text, "Total episodes"),
        "successes": _last_totals_count(text, "successes"),
        "violations": _last_totals_count(text, "violations"),
        "safe_successes": _last_totals_count(text, "safe_successes"),
        "episode_success_seq": _episode_seq(text, "Success"),
        "episode_violation_seq": _episode_seq(text, "Safety violated"),
        "episode_safe_seq": _episode_seq(text, "Safe success"),
        "task_success_rate": _float_after_key(text, "Overall success rate"),
        "svr": _float_after_key(text, "Overall SVR"),
        "valid_svr": _float_after_key(text, "Overall valid-execution violation rate"),
        "model_collapse_rate": _float_after_key(text, "Overall model collapse rate"),
        "safe_success_rate": _float_after_key(text, "Overall safe success rate"),
        "notes": "",
    }


def _metric_rate(text: str, metric: str) -> Optional[float]:
    pattern = rf"^\|\s*{re.escape(metric)}\b[^|]*\|\s*([0-9.]+|--)\s*\|"
    match = re.search(pattern, text, re.MULTILINE)
    if not match or match.group(1) == "--":
        return None
    return float(match.group(1))


def _metric_n(text: str, metric: str) -> Optional[int]:
    pattern = rf"^\|\s*{re.escape(metric)}\b[^|]*\|\s*(?:[0-9.]+|--)\s*\|[^|]*\|\s*(\d+)\s*\|"
    match = re.search(pattern, text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _metadata_for_attribution(path: Path, family: str) -> tuple:
    haystack = f"{path.stem} {family}".lower()
    for key, value in ATTRIBUTION_FILE_METADATA.items():
        if key in haystack:
            return value
    if "l1-a1" in haystack:
        return ("L1", "L1-A1")
    if "l2-b2" in haystack:
        return ("L2", "L2-B2")
    if "l3-c" in haystack:
        return ("L3", "L3-C")
    return ("", "")


def parse_attribution_report(path: Path) -> Dict[str, object]:
    text = _read(path)
    family = _first_match(r"^#\s*PhysCogSafe Attribution Report:?\s*(.*)$", text, re.MULTILINE) or path.stem
    level, scenario = _metadata_for_attribution(path, family)
    benign = _first_match(r"^- Benign \(Eb\) rollouts:\s*(\d+)", text, re.MULTILINE)
    risk = _first_match(r"^- Risk \(Er\) rollouts:\s*(\d+)", text, re.MULTILINE)
    ec = _first_match(r"null-risk \(Ec\) rollouts:\s*(\d+)", text, re.MULTILINE)
    n_parts = [part for part in (f"Eb={benign}" if benign else "", f"Er={risk}" if risk else "", f"Ec={ec}" if ec else "") if part]
    return {
        "record_type": "attribution",
        "timestamp": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y_%m_%d_%H_%M_%S"),
        "source_path": str(path),
        "model": "",
        "task_suite": "",
        "run_id": path.stem,
        "level": level,
        "scenario": scenario,
        "condition": "Eb/Er/Ec",
        "family": family,
        "n": ";".join(n_parts),
        "sar": _metric_rate(text, "SAR"),
        "btf": _metric_rate(text, "BTF"),
        "uir": _metric_rate(text, "UIR"),
        "ocr": _metric_rate(text, "OCR"),
        "nor": _metric_rate(text, "NOR"),
        "unsafe_divergent": _metric_rate(text, "unsafe_divergent"),
        "safe_invariant": _metric_rate(text, "safe_invariant"),
        "divergence_reference": _first_match(r"^- Divergence reference:\s*(.+)$", text, re.MULTILINE) or "",
        "notes": (
            f"BTF_N={_metric_n(text, 'BTF')};SAR_N={_metric_n(text, 'SAR')};"
            f"UIR_N={_metric_n(text, 'UIR')};OCR_N={_metric_n(text, 'OCR')};"
            f"NOR_N={_metric_n(text, 'NOR')}"
        ),
    }


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    return {field: row.get(field, "") for field in RECORD_FIELDS}


def collect_records(log_dir: Path, include_incomplete: bool = False) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(log_dir.glob("EVAL-*.txt")):
        try:
            row = _normalize(parse_eval_log(path))
            has_final_metric = any(row.get(key) not in ("", None) for key in ("task_success_rate", "svr", "safe_success_rate"))
            if include_incomplete or has_final_metric:
                rows.append(row)
        except Exception as exc:
            rows.append(_normalize({
                "record_type": "parse_error",
                "source_path": str(path),
                "notes": f"{type(exc).__name__}: {exc}",
            }))
    for path in sorted(log_dir.glob("*attribution*.md")):
        try:
            rows.append(_normalize(parse_attribution_report(path)))
        except Exception as exc:
            rows.append(_normalize({
                "record_type": "parse_error",
                "source_path": str(path),
                "notes": f"{type(exc).__name__}: {exc}",
            }))
    return rows


def _rate(value) -> str:
    if value in ("", None):
        return "--"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _latest_eval_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if row["record_type"] != "eval":
            continue
        key = f"{row.get('model') or 'unknown'}::{row.get('run_id') or row.get('source_path')}"
        if key not in latest or str(row.get("timestamp", "")) > str(latest[key].get("timestamp", "")):
            latest[key] = row
    return sorted(latest.values(), key=lambda r: (str(r.get("level")), str(r.get("scenario")), str(r.get("condition")), str(r.get("run_id"))))


def write_csv(rows: List[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_eval = _latest_eval_rows(rows)
    attributions = [row for row in rows if row["record_type"] == "attribution"]

    lines = [
        "# PhysCog Experiment Records",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Latest eval logs by run ID",
        "",
        "| Level | Scenario | Condition | Run ID | N | Task SR | SVR | Safe SR | Source |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in latest_eval:
        lines.append(
            f"| {row['level']} | {row['scenario']} | {row['condition']} | {row['run_id']} | "
            f"{row['n'] or '--'} | {_rate(row['task_success_rate'])} | {_rate(row['svr'])} | "
            f"{_rate(row['safe_success_rate'])} | `{Path(str(row['source_path'])).name}` |"
        )

    lines += [
        "",
        "## Attribution reports",
        "",
        "| Level | Scenario | Family | N | Reference | BTF | SAR | UIR | OCR | NOR | Source |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(attributions, key=lambda r: (str(r["level"]), str(r["scenario"]), str(r["family"]))):
        lines.append(
            f"| {row['level']} | {row['scenario']} | {row['family']} | {row['n'] or '--'} | "
            f"{row['divergence_reference'] or '--'} | {_rate(row['btf'])} | {_rate(row['sar'])} | {_rate(row['uir'])} | "
            f"{_rate(row['ocr'])} | {_rate(row['nor'])} | `{Path(str(row['source_path'])).name}` |"
        )

    lines.append("")
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record PhysCog experiment metrics from logs")
    parser.add_argument("--log_dir", default="experiments/logs")
    parser.add_argument("--out_csv", default="experiments/logs/experiment_records.csv")
    parser.add_argument("--out_md", default="experiments/logs/experiment_records.md")
    parser.add_argument("--include_incomplete", action="store_true",
                        help="Include EVAL logs that do not contain final aggregate metrics")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log_dir not found: {log_dir}")

    rows = collect_records(log_dir, include_incomplete=args.include_incomplete)
    write_csv(rows, Path(args.out_csv))
    write_markdown(rows, Path(args.out_md))
    print(f"Recorded {len(rows)} rows")
    print(f"CSV: {args.out_csv}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
